"""Demo end-to-end de una llamada individual.

Este módulo conserva la visualización del Notebook 10 original, pero deja
el notebook únicamente como orquestador. Restaura desde Google Cloud Storage
los CSV necesarios para la demo y descarga los audios seleccionados bajo
demanda. No sube ningún archivo a GCS.

La única salida nueva de esta fase es local:
``data/demo_end_to_end_audio_individual/html_exports/``.
"""

from datetime import datetime
from functools import lru_cache
from pathlib import Path

import base64
import contextlib
import html as html_lib
import io
import re
import traceback
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf

from IPython.display import Audio, HTML, Markdown, display

try:
    import ipywidgets as widgets

    HAS_WIDGETS = True
except Exception:
    widgets = None
    HAS_WIDGETS = False

try:
    import librosa

    HAS_LIBROSA = True
except Exception:
    librosa = None
    HAS_LIBROSA = False

try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    HAS_SKLEARN = True
except Exception:
    PCA = None
    StandardScaler = None
    HAS_SKLEARN = False

try:
    import mistune

    HAS_MISTUNE = True
except Exception:
    mistune = None
    HAS_MISTUNE = False

from src.config import (
    AUDIO_INVENTORY_PRIVATE_CSV as AUDIO_INVENTORY_CSV,
    BQ_METADATA_SNAPSHOT_CSV,
    CLEANING_RESULTS_PRIVATE_CSV as CLEANING_RESULTS_CSV,
    DATA_DIR,
    EDA_DIR,
    CLEAN_RESULTS_DIR,
    OUTPUT_DIR as DIARIZATION_DIR,
    FINAL_RELABEL_DIR,
    EMBEDDING_VECTOR_CSV_DIR as EMBEDDING_DIR,
    RELABELING_SUMMARY_BY_AUDIO_CSV as RELABEL_INDEX_CSV,
    RELABEL_MARGIN_BY_AUDIO_CSV as MARGIN_BY_AUDIO_CSV,
    TRANSCRIPTION_ROOT as TRANSCRIPTION_DIR,
    PROXY_GROUNDTRUTH_DIR as PROXY_DIR,
    SENTIMENT_DIR,
    PROSODY_DIR,
    KEYWORD_DIR,
    VOICEPRINT_DIR,
    PROJECT_DIR,
    GCS_UNAV_ROOT,
    GCS_UNAV_CSV_PREFIX,
    GCS_UNAV_CLEAN_AUDIO_PREFIX as GCS_CLEAN_PREFIX,
)
from src.storage_io import download_uri_to_local, join_gcs_uri


warnings.filterwarnings("ignore")

DEMO_CACHE_DIR = DATA_DIR / "demo_end_to_end_audio_individual"
DEMO_HTML_DIR = DEMO_CACHE_DIR / "html_exports"

SILENCE_TOP_DB = 30
MIN_SILENCE_LEN_SEC = 0.30
MAX_INTERNAL_SILENCE_SEC = 0.75
ANCHOR_MIN_DURATION_SEC = 1.20
ANCHOR_MAX_OVERLAP_RATIO = 0.00
ANCHOR_INITIAL_EXCLUDE_SEC = 1.50
ANCHORS_PER_SPEAKER = 3
RELABEL_MIN_MARGIN = 0.01
MAX_GAP_MERGE_SEC = 0.50

COLORS = {
    "ink": "#123047",
    "blue": "#2F6BFF",
    "orange": "#F28E2B",
    "green": "#2A9D6F",
    "red": "#D1495B",
    "purple": "#7B61A8",
    "gray": "#7A8288",
    "light": "#EEF2F5",
}
SPEAKER_COLORS = [
    COLORS["blue"],
    COLORS["orange"],
    COLORS["green"],
    COLORS["purple"],
]

_GCS_CLIENT = None
RELABEL_INDEX = pd.DataFrame()


def _configure_notebook_style():
    """Aplica el mismo estilo visual del Notebook 10 original."""
    plt.rcParams.update({
        "figure.figsize": (12, 4.5),
        "axes.grid": True,
        "grid.alpha": 0.20,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })

    display(HTML(r"""
    <style>
    .tfm-header {background:linear-gradient(135deg,#123047,#2F6BFF); color:white; padding:18px 22px; border-radius:14px; margin:6px 0 14px 0;}
    .tfm-header .title {font-size:21px; font-weight:750;}
    .tfm-header .sub {font-size:13px; opacity:.92; margin-top:4px;}
    .tfm-card {border:1px solid #E2E7EB; background:#FFFFFF; border-radius:12px; padding:13px 15px; margin:8px 0 12px 0;}
    .tfm-kpi {display:inline-block; min-width:125px; padding:9px 12px; margin:3px 5px 4px 0; border:1px solid #E2E7EB; border-radius:10px; vertical-align:top; background:#FAFBFC;}
    .tfm-kpi .v {font-size:20px; font-weight:750; color:#123047;}
    .tfm-kpi .l {font-size:11px; color:#68737D; margin-top:2px;}
    .tfm-note {font-size:12px; color:#68737D; margin:3px 0 10px 0;}
    .tfm-ok {color:#16794F; font-weight:650;}
    .tfm-warn {color:#A85A00; font-weight:650;}
    .widget-tab > .p-TabBar .p-TabBar-tab {font-size:12px;}
    </style>
    """))


def _gcs_uri_for_data_path(local_path):
    """Mapea una ruta dentro de data/ al prefijo equivalente en GCS."""
    local_path = Path(local_path)
    relative_path = local_path.relative_to(DATA_DIR).as_posix()
    return join_gcs_uri(GCS_UNAV_ROOT, relative_path)


def _restore_file_from_gcs(local_path, gcs_uri, required=False):
    """Restaura un archivo concreto sin subir ni modificar GCS."""
    if _GCS_CLIENT is None:
        raise RuntimeError(
            "El cliente GCS no está configurado. "
            "Ejecuta run_demo_end_to_end(gcs_client=...)."
        )

    local_path = Path(local_path)

    try:
        download_uri_to_local(
            source_uri=gcs_uri,
            local_path=local_path,
            gcs_client=_GCS_CLIENT,
            force=False,
        )
    except Exception as exc:
        if required:
            raise RuntimeError(
                f"No se pudo restaurar desde GCS: {gcs_uri}"
            ) from exc
        print(f"No se pudo restaurar {local_path.name}: {exc}")

    available = (
        local_path.exists()
        and local_path.stat().st_size > 0
    )

    if required and not available:
        raise FileNotFoundError(
            "No se encontró el archivo requerido ni localmente ni en GCS:\n"
            f"Local: {local_path}\n"
            f"GCS: {gcs_uri}"
        )

    return available


def _restore_data_path(local_path, required=False):
    """Restaura un archivo cuya estructura remota replica data/."""
    local_path = Path(local_path)
    return _restore_file_from_gcs(
        local_path=local_path,
        gcs_uri=_gcs_uri_for_data_path(local_path),
        required=required,
    )


def _restore_initial_demo_inputs():
    """Restaura únicamente los CSV globales que necesita la demo."""
    DEMO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    DEMO_HTML_DIR.mkdir(parents=True, exist_ok=True)

    phase00_files = [
        AUDIO_INVENTORY_CSV,
        BQ_METADATA_SNAPSHOT_CSV,
        CLEANING_RESULTS_CSV,
    ]

    for local_path in phase00_files:
        _restore_file_from_gcs(
            local_path=local_path,
            gcs_uri=join_gcs_uri(
                GCS_UNAV_CSV_PREFIX,
                Path(local_path).name,
            ),
            required=False,
        )

    global_phase_files = [
        RELABEL_INDEX_CSV,
        MARGIN_BY_AUDIO_CSV,

        TRANSCRIPTION_DIR / "06_transcribed_segments_final.csv",
        TRANSCRIPTION_DIR / "all_segments_transcribed.csv",
        TRANSCRIPTION_DIR / "transcribed_segments_final.csv",

        PROXY_DIR / "segment_level_proxy_groundtruth.csv",
        PROXY_DIR / "speaker_role_mapping_textual.csv",
        PROXY_DIR / "text_alignment_matches.csv",
        PROXY_DIR / "official_transcription_turns_extracted.csv",
        PROXY_DIR / "metadata_join_audit_by_audio.csv",
        PROXY_DIR / "holdout_role_mapping_predictions.csv",
        PROXY_DIR / "holdout_role_mapping_metrics.csv",

        SENTIMENT_DIR / "segments_with_sentiment_textual.csv",
        SENTIMENT_DIR / "all_segments_sentiment_textual_enriched.csv",

        PROSODY_DIR / "segments_with_audio_affect_prosody.csv",

        KEYWORD_DIR / "segments_with_keywords.csv",
        KEYWORD_DIR / "call_level_keywords_sentiment_combined.csv",
        KEYWORD_DIR / "call_level_keywords.csv",
        KEYWORD_DIR / "top_critical_calls_keywords.csv",

        VOICEPRINT_DIR / "voiceprint_segments_candidates.csv",
        VOICEPRINT_DIR / "voiceprint_audio_person_samples.csv",
        VOICEPRINT_DIR / "open_set_identification_predictions.csv",
        VOICEPRINT_DIR / "voiceprint_identity_summary_open_set.csv",
        VOICEPRINT_DIR / "voiceprint_identity_summary.csv",
        VOICEPRINT_DIR / "voiceprint_identity_split.csv",
        VOICEPRINT_DIR / "voiceprint_open_set_final_summary.csv",
        VOICEPRINT_DIR / "voiceprint_verification_metrics.csv",
        VOICEPRINT_DIR / "open_set_decision_confusion_matrix.csv",
    ]

    for local_path in global_phase_files:
        _restore_data_path(
            local_path,
            required=(Path(local_path) == Path(RELABEL_INDEX_CSV)),
        )

    read_csv_cached.cache_clear()


def _restore_audio_table_files(paths):
    """Restaura desde GCS los outputs concretos del audio seleccionado."""
    for local_path in paths.values():
        _restore_data_path(local_path, required=False)


AUDIO_SUFFIXES = [
    "_anchor_embeddings_vectors", "_segment_embeddings_vectors",
    "_final_segments", "_final_merged", "_changed_segments",
    "_anchor_embeddings", "_transcribed_segments",
    "_segment_level_proxy_groundtruth", "_regular", "_anchors", "_raw",
]


def normalize_stem(value):
    """Quita extensión y sufijos de outputs, pero conserva `_clean`."""
    if value is None or pd.isna(value):
        return ""
    s = Path(str(value).strip()).name
    s = re.sub(r"\.(wav|mp3|m4a|flac|ogg|csv)$", "", s, flags=re.I)
    changed = True
    while changed:
        changed = False
        for suffix in AUDIO_SUFFIXES:
            if s.endswith(suffix):
                s = s[:-len(suffix)]
                changed = True
    return s


def audio_key_variants(value):
    """Genera variantes para cruzar `raw_*_clean`, filename y audio_id."""
    stem = normalize_stem(value)
    if not stem:
        return set()
    variants = {stem}
    no_clean = re.sub(r"_clean$", "", stem, flags=re.I)
    variants.add(no_clean)
    if no_clean.startswith("raw_bajas_"):
        variants.add(no_clean[len("raw_bajas_"):])
    elif no_clean.startswith("raw_"):
        variants.add(no_clean[len("raw_"):])
    return {str(v).strip().lower() for v in variants if str(v).strip()}


def same_audio(value, target):
    return bool(audio_key_variants(value) & audio_key_variants(target))


def read_csv_optional(path, **kwargs):
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, **kwargs)
    except Exception as exc:
        print(f"No se pudo leer {path.name}: {exc}")
        return pd.DataFrame()


@lru_cache(maxsize=32)
def read_csv_cached(path_str):
    return read_csv_optional(Path(path_str))


def first_existing(paths):
    return next((Path(p) for p in paths if Path(p).exists()), None)


def detect_audio_col(df):
    candidates = [
        "audio_base", "audio_stem", "audio_file", "audio_id_base", "audio_id",
        "clean_filename", "audio_name", "filename", "file_name", "source_file",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    for c in df.columns:
        low = str(c).lower()
        if any(x in low for x in ["audio", "filename", "file_stem"]):
            return c
    return None


def detect_time_cols(df):
    starts = ["start", "start_sec", "start_time", "segment_start", "inicio", "t_start"]
    ends = ["end", "end_sec", "end_time", "segment_end", "fin", "t_end"]
    s = next((c for c in starts if c in df.columns), None)
    e = next((c for c in ends if c in df.columns), None)
    if s is None:
        s = next((c for c in df.columns if "start" in str(c).lower()), None)
    if e is None:
        e = next((c for c in df.columns if "end" in str(c).lower()), None)
    return s, e


def detect_text_col(df, official=False):
    if official:
        candidates = ["official_text", "transcripcion", "metadata_text", "text_official"]
    else:
        candidates = ["text_whisper", "text", "transcription", "whisper_text", "segment_text", "texto"]
    return next((c for c in candidates if c in df.columns), None)


def filter_audio(df, audio_stem):
    if df is None or df.empty:
        return pd.DataFrame()
    col = detect_audio_col(df)
    if col is None:
        return pd.DataFrame()
    mask = df[col].apply(lambda x: same_audio(x, audio_stem))
    return df.loc[mask].copy()


def filter_any_audio_column(df, audio_stem):
    if df is None or df.empty:
        return pd.DataFrame()
    likely = [c for c in df.columns if any(k in str(c).lower() for k in ["audio", "file", "filename"])]
    if not likely:
        return pd.DataFrame()
    mask = pd.Series(False, index=df.index)
    for c in likely:
        mask |= df[c].apply(lambda x: same_audio(x, audio_stem))
    return df.loc[mask].copy()


def fmt_time(seconds):
    if seconds is None or pd.isna(seconds):
        return ""
    seconds = max(0.0, float(seconds))
    return f"{int(seconds // 60):02d}:{seconds % 60:05.2f}"


def fmt_value(value, kind=None):
    if value is None or pd.isna(value):
        return "—"
    if kind == "pct":
        return f"{100 * float(value):.1f}%"
    if kind == "sec":
        return f"{float(value):.1f} s"
    if isinstance(value, (np.integer, int)):
        return f"{int(value):,}".replace(",", ".")
    if isinstance(value, (np.floating, float)):
        return f"{float(value):.2f}"
    return str(value)


def section(title, explanation):
    display(Markdown(f"### {title}"))
    display(HTML(f"<div class='tfm-note'>{explanation}</div>"))


def kpi_cards(items):
    html = "<div class='tfm-card'>"
    for label, value in items:
        html += f"<div class='tfm-kpi'><div class='v'>{value}</div><div class='l'>{label}</div></div>"
    html += "</div>"
    display(HTML(html))


def compact_table(df, columns, n=12):
    if df is None or df.empty:
        display(Markdown("No disponible para este audio."))
        return
    show = df.copy()
    s, e = detect_time_cols(show)
    if s:
        show["inicio"] = pd.to_numeric(show[s], errors="coerce").map(fmt_time)
    if e:
        show["fin"] = pd.to_numeric(show[e], errors="coerce").map(fmt_time)
    cols = [c for c in columns if c in show.columns]
    if not cols:
        cols = list(show.columns[:10])
    display(show[cols].head(n).reset_index(drop=True))


def split_gcs_uri(uri):
    if not isinstance(uri, str) or not uri.startswith("gs://"):
        raise ValueError(f"URI GCS inválida: {uri}")
    body = uri[5:]
    bucket, _, blob = body.partition("/")
    return bucket, blob


def download_gcs(uri, local_path):
    """Descarga un audio desde GCS usando el cliente recibido por la demo."""
    local_path = Path(local_path)

    if local_path.exists() and local_path.stat().st_size > 0:
        return local_path

    if _GCS_CLIENT is None:
        raise RuntimeError(
            "El cliente GCS no está configurado. "
            "Ejecuta run_demo_end_to_end(gcs_client=...)."
        )

    downloaded = download_uri_to_local(
        source_uri=uri,
        local_path=local_path,
        gcs_client=_GCS_CLIENT,
        force=False,
    )

    if downloaded or (
        local_path.exists()
        and local_path.stat().st_size > 0
    ):
        return local_path

    return None


@lru_cache(maxsize=8)
def load_audio_cached(path_str):
    y, sr = sf.read(path_str, always_2d=True)
    y = y.mean(axis=1).astype(np.float32)
    return y, int(sr)


def downsample_waveform(y, sr, max_points=24000):
    if len(y) <= max_points:
        idx = np.arange(len(y))
    else:
        idx = np.linspace(0, len(y) - 1, max_points).astype(int)
    return idx / sr, y[idx]


def audio_duration(path):
    if path is None or not Path(path).exists():
        return np.nan
    info = sf.info(str(path))
    return info.frames / info.samplerate


def speaker_color_map(values):
    vals = sorted(pd.Series(values).dropna().astype(str).unique())
    return {v: SPEAKER_COLORS[i % len(SPEAKER_COLORS)] for i, v in enumerate(vals)}


def build_relabel_index():
    """Construye el dropdown ordenado por segmentos reetiquetados."""
    idx = read_csv_optional(RELABEL_INDEX_CSV)
    if not idx.empty:
        stem_col = next((c for c in ["audio_base", "audio_stem", "audio_file"] if c in idx.columns), None)
        if stem_col:
            out = pd.DataFrame({
                "audio_stem": idx[stem_col].map(normalize_stem),
                "n_segments": pd.to_numeric(idx.get("n_segments", np.nan), errors="coerce"),
                "n_relabelled": pd.to_numeric(idx.get("n_relabelled", np.nan), errors="coerce"),
                "relabel_ratio": pd.to_numeric(idx.get("relabel_ratio", np.nan), errors="coerce"),
            })
            out = out.dropna(subset=["audio_stem"]).drop_duplicates("audio_stem")
            return out.sort_values(["n_relabelled", "relabel_ratio", "audio_stem"], ascending=[False, False, True])

    rows = []
    if FINAL_RELABEL_DIR.exists():
        for path in sorted(FINAL_RELABEL_DIR.glob("*_final_segments.csv")):
            if path.name.startswith("all_"):
                continue
            df = read_csv_optional(path)
            if df.empty:
                continue
            if "was_reclassified" in df.columns:
                changed = df["was_reclassified"].fillna(False).astype(bool)
            elif {"speaker_original", "speaker_final"}.issubset(df.columns):
                changed = df["speaker_original"].astype(str) != df["speaker_final"].astype(str)
            else:
                changed = pd.Series(False, index=df.index)
            n = len(df)
            n_changed = int(changed.sum())
            rows.append({
                "audio_stem": normalize_stem(path.name),
                "n_segments": n,
                "n_relabelled": n_changed,
                "relabel_ratio": n_changed / n if n else 0.0,
            })
    return pd.DataFrame(rows).sort_values(["n_relabelled", "relabel_ratio", "audio_stem"], ascending=[False, False, True]) if rows else pd.DataFrame()


def dropdown_options():
    if RELABEL_INDEX.empty:
        return []
    options = []
    for _, row in RELABEL_INDEX.iterrows():
        n_rel = int(row["n_relabelled"]) if pd.notna(row["n_relabelled"]) else 0
        n_seg = int(row["n_segments"]) if pd.notna(row["n_segments"]) else 0
        ratio = 100 * float(row["relabel_ratio"]) if pd.notna(row["relabel_ratio"]) else 0.0
        label = f"{n_rel:03d} reetiquetados · {ratio:5.1f}% · {n_seg:03d} segmentos · {row['audio_stem']}"
        options.append((label, row["audio_stem"]))
    return options


def load_audio_tables(audio_stem):
    stem = normalize_stem(audio_stem)
    paths = {
        "regular": DIARIZATION_DIR / f"{stem}_regular.csv",
        "scored": DIARIZATION_DIR / f"{stem}_raw.csv",
        "valid": DIARIZATION_DIR / f"{stem}.csv",
        "anchors": DIARIZATION_DIR / f"{stem}_anchors.csv",
        "final": FINAL_RELABEL_DIR / f"{stem}_final_segments.csv",
        "merged": FINAL_RELABEL_DIR / f"{stem}_final_merged.csv",
        "changed": FINAL_RELABEL_DIR / f"{stem}_changed_segments.csv",
        "anchor_embeddings": EMBEDDING_DIR / f"{stem}_anchor_embeddings_vectors.csv",
        "segment_embeddings": EMBEDDING_DIR / f"{stem}_segment_embeddings_vectors.csv",
    }
    _restore_audio_table_files(paths)
    return {k: read_csv_optional(v) for k, v in paths.items()}, paths


def find_matching_rows(df, audio_stem):
    if df is None or df.empty:
        return pd.DataFrame()
    return filter_any_audio_column(df, audio_stem)


def load_inventory_rows(audio_stem):
    return find_matching_rows(read_csv_cached(str(AUDIO_INVENTORY_CSV)), audio_stem)


def load_cleaning_rows(audio_stem):
    return find_matching_rows(read_csv_cached(str(CLEANING_RESULTS_CSV)), audio_stem)


def load_metadata_snapshot_rows(audio_stem):
    return find_matching_rows(read_csv_cached(str(BQ_METADATA_SNAPSHOT_CSV)), audio_stem)


def local_audio_candidates(audio_stem, kind):
    stem = normalize_stem(audio_stem)
    if kind == "clean":
        roots = [
            DATA_DIR / "diarization_input_clean_audios", DATA_DIR / "clean_audios",
            DATA_DIR / "processed_audios", DEMO_CACHE_DIR / "clean",
        ]
        names = [f"{stem}.wav"]
    else:
        roots = [
            DATA_DIR / "raw_audios", DATA_DIR / "audios_raw", DATA_DIR / "original_audios",
            DATA_DIR / "input_audios", DATA_DIR / "audios", DEMO_CACHE_DIR / "raw",
        ]
        inv = load_inventory_rows(stem)
        names = [f"{re.sub(r'_clean$', '', stem)}.wav"]
        for c in ["audio_name", "filename", "audio_id"]:
            if c in inv.columns:
                names += [Path(str(v)).name for v in inv[c].dropna().tolist()]
    for root in roots:
        if not root.exists():
            continue
        for name in dict.fromkeys(names):
            p = root / name
            if p.exists() and p.stat().st_size > 0:
                return p
    return None


def resolve_audio(audio_stem, kind):
    local = local_audio_candidates(audio_stem, kind)
    if local is not None:
        return local

    stem = normalize_stem(audio_stem)
    uri = None
    if kind == "clean":
        cleaning = load_cleaning_rows(stem)
        for col in ["clean_gcs_uri", "clean_uri", "processed_gcs_uri"]:
            if col in cleaning.columns:
                vals = cleaning[col].dropna().astype(str)
                uri = next((v for v in vals if v.startswith("gs://")), None)
                if uri:
                    break
        if uri is None:
            uri = f"{GCS_CLEAN_PREFIX.rstrip('/')}/{stem}.wav"
    else:
        inv = load_inventory_rows(stem)
        priority = ["gcs_uri", "raw_gcs_uri", "source_gcs_uri", "url"]
        for col in priority:
            if col in inv.columns:
                vals = inv[col].dropna().astype(str)
                uri = next((v for v in vals if v.startswith("gs://")), None)
                if uri:
                    break

    if uri is None:
        return None
    local_path = DEMO_CACHE_DIR / kind / f"{stem}.wav"
    try:
        return download_gcs(uri, local_path)
    except Exception as exc:
        print(f"No se pudo resolver audio {kind}: {exc}")
        return None


def extract_snippet(audio_path, start, end, tag):
    if audio_path is None or not Path(audio_path).exists():
        return None
    start = max(0.0, float(start))
    end = max(start, float(end))
    out = DEMO_CACHE_DIR / "snippets" / f"{tag}_{start:.3f}_{end:.3f}.wav"
    if out.exists() and out.stat().st_size > 0:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    with sf.SoundFile(str(audio_path)) as f:
        sr = f.samplerate
        f.seek(max(0, int(start * sr)))
        frames = max(1, int((end - start) * sr))
        data = f.read(frames=frames, always_2d=True)
    if data.size == 0:
        return None
    sf.write(str(out), data, sr)
    return out


def load_global_audio_csv(paths, audio_stem):
    for path in paths:
        df = read_csv_cached(str(path))
        selected = filter_audio(df, audio_stem)
        if not selected.empty:
            return selected, Path(path)
    return pd.DataFrame(), None


def load_transcription(audio_stem):
    paths = [
        TRANSCRIPTION_DIR / "06_transcribed_segments_final.csv",
        TRANSCRIPTION_DIR / "all_segments_transcribed.csv",
        TRANSCRIPTION_DIR / "transcribed_segments_final.csv",
    ]
    df, path = load_global_audio_csv(paths, audio_stem)
    if not df.empty:
        return df, path
    for root in [TRANSCRIPTION_DIR / "per_audio", TRANSCRIPTION_DIR]:
        if root.exists():
            for p in root.rglob(f"*{normalize_stem(audio_stem)}*_transcribed_segments.csv"):
                df = read_csv_optional(p)
                if not df.empty:
                    return df, p
    return pd.DataFrame(), None


def load_proxy_outputs(audio_stem):
    names = [
        "segment_level_proxy_groundtruth", "speaker_role_mapping_textual",
        "text_alignment_matches", "official_transcription_turns_extracted",
        "metadata_join_audit_by_audio", "holdout_role_mapping_predictions",
        "holdout_role_mapping_metrics",
    ]
    out = {}
    for name in names:
        path = PROXY_DIR / f"{name}.csv"
        df = read_csv_cached(str(path))
        if name == "holdout_role_mapping_metrics":
            out[name] = df.copy()
        else:
            out[name] = filter_audio(df, audio_stem)
        out[f"{name}__path"] = path
    return out


def load_sentiment(audio_stem):
    paths = [
        SENTIMENT_DIR / "segments_with_sentiment_textual.csv",
        SENTIMENT_DIR / "all_segments_sentiment_textual_enriched.csv",
    ]
    return load_global_audio_csv(paths, audio_stem)


def load_prosody(audio_stem):
    paths = [PROSODY_DIR / "segments_with_audio_affect_prosody.csv"]
    return load_global_audio_csv(paths, audio_stem)


def load_keywords(audio_stem):
    """Carga los outputs por segmento y por llamada del Notebook 08."""
    segment_paths = [
        KEYWORD_DIR / "segments_with_keywords.csv",
    ]
    call_paths = [
        KEYWORD_DIR / "call_level_keywords_sentiment_combined.csv",
        KEYWORD_DIR / "call_level_keywords.csv",
        KEYWORD_DIR / "top_critical_calls_keywords.csv",
    ]

    segments, segments_path = load_global_audio_csv(
        segment_paths,
        audio_stem,
    )
    calls, calls_path = load_global_audio_csv(
        call_paths,
        audio_stem,
    )

    return {
        "segments": segments,
        "segments_path": segments_path,
        "calls": calls,
        "calls_path": calls_path,
    }


def load_voiceprint_outputs(audio_stem):
    """
    Carga los outputs del Notebook 09 relacionados con el audio elegido.

    Los resúmenes y métricas globales se conservan completos para poder
    contextualizar el resultado individual.
    """
    paths = {
        "segments": VOICEPRINT_DIR / "voiceprint_segments_candidates.csv",
        "samples": VOICEPRINT_DIR / "voiceprint_audio_person_samples.csv",
        "predictions": VOICEPRINT_DIR / "open_set_identification_predictions.csv",
        "identity_summary_open_set": VOICEPRINT_DIR / "voiceprint_identity_summary_open_set.csv",
        "identity_summary": VOICEPRINT_DIR / "voiceprint_identity_summary.csv",
        "identity_split": VOICEPRINT_DIR / "voiceprint_identity_split.csv",
        "final_summary": VOICEPRINT_DIR / "voiceprint_open_set_final_summary.csv",
        "verification_metrics": VOICEPRINT_DIR / "voiceprint_verification_metrics.csv",
        "decision_matrix": VOICEPRINT_DIR / "open_set_decision_confusion_matrix.csv",
    }

    raw = {
        key: read_csv_cached(str(path))
        for key, path in paths.items()
    }

    out = {
        "segments": filter_any_audio_column(raw["segments"], audio_stem),
        "samples": filter_any_audio_column(raw["samples"], audio_stem),
        "predictions": filter_any_audio_column(raw["predictions"], audio_stem),
        "final_summary": raw["final_summary"].copy(),
        "verification_metrics": raw["verification_metrics"].copy(),
        "decision_matrix": raw["decision_matrix"].copy(),
        "paths": paths,
    }

    person_ids = set()

    for frame_key in ["segments", "samples", "predictions"]:
        frame = out[frame_key]
        if frame is not None and not frame.empty:
            for column in [
                "person_id",
                "true_person_id",
                "true_source_identity_id",
                "source_identity_id",
            ]:
                if column in frame.columns:
                    person_ids.update(
                        frame[column]
                        .dropna()
                        .astype(str)
                        .tolist()
                    )

    identity_source = (
        raw["identity_summary_open_set"]
        if not raw["identity_summary_open_set"].empty
        else raw["identity_summary"]
    )

    if person_ids and not identity_source.empty:
        id_col = next(
            (
                column
                for column in [
                    "person_id",
                    "source_identity_id",
                    "true_source_identity_id",
                ]
                if column in identity_source.columns
            ),
            None,
        )

        if id_col:
            out["identity_summary"] = identity_source[
                identity_source[id_col].astype(str).isin(person_ids)
            ].copy()
        else:
            out["identity_summary"] = pd.DataFrame()
    else:
        out["identity_summary"] = pd.DataFrame()

    split_source = raw["identity_split"]

    if person_ids and not split_source.empty and "person_id" in split_source.columns:
        out["identity_split"] = split_source[
            split_source["person_id"].astype(str).isin(person_ids)
        ].copy()
    else:
        out["identity_split"] = pd.DataFrame()

    return out


def metadata_value(rows, candidates):
    for col in candidates:
        if col in rows.columns:
            vals = rows[col].dropna()
            if len(vals):
                return vals.iloc[0]
    return np.nan


def show_audio_player(label, path):
    display(Markdown(f"**{label}**"))
    if path is None or not Path(path).exists():
        display(Markdown("Audio no localizado en la VM ni en las rutas GCS esperadas."))
        return
    info = sf.info(str(path))
    display(HTML(
        f"<div class='tfm-note'>{info.duration:.1f} s · {info.samplerate:,} Hz · {info.channels} canal(es)</div>"
    ))
    display(Audio(filename=str(path)))


def compute_overlap_intervals(df_regular):
    if df_regular is None or df_regular.empty:
        return []
    s, e = detect_time_cols(df_regular)
    if not s or not e:
        return []
    events = []
    for _, row in df_regular.dropna(subset=[s, e]).iterrows():
        start, end = float(row[s]), float(row[e])
        if end > start:
            events += [(start, 1), (end, -1)]
    events.sort(key=lambda x: (x[0], x[1]))
    intervals, active, previous = [], 0, None
    for t, delta in events:
        if previous is not None and t > previous and active > 1:
            intervals.append((previous, t))
        active += delta
        previous = t
    merged = []
    for start, end in intervals:
        if merged and start <= merged[-1][1] + 1e-9:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(float(a), float(b)) for a, b in merged]


def plot_raw_clean_waveforms(raw_path, clean_path):
    available = [("Original / raw", raw_path), ("Limpio / clean", clean_path)]
    fig, axes = plt.subplots(2, 1, figsize=(13, 5.4), constrained_layout=True)
    for ax, (label, path) in zip(axes, available):
        if path is None or not Path(path).exists():
            ax.axis("off")
            ax.set_title(f"{label} · no disponible", loc="left")
            continue
        y, sr = load_audio_cached(str(path))
        t, yy = downsample_waveform(y, sr)
        ax.plot(t, yy, linewidth=0.55, color=COLORS["blue"] if "Original" in label else COLORS["green"])
        ax.set_title(label, loc="left")
        ax.set_ylabel("Amplitud")
        ax.set_xlim(0, len(y) / sr)
    axes[-1].set_xlabel("Tiempo (s)")
    fig.suptitle("Señal antes y después de la preparación", x=0.08, ha="left", fontsize=13, fontweight="bold")
    plt.show()


def plot_silence_overlap_panel(raw_path, clean_path, df_regular):
    raw_duration = audio_duration(raw_path)
    clean_duration = audio_duration(clean_path)
    nonsilent = []
    if raw_path is not None and Path(raw_path).exists() and HAS_LIBROSA:
        y, sr = load_audio_cached(str(raw_path))
        nonsilent = [(a / sr, b / sr) for a, b in librosa.effects.split(y, top_db=SILENCE_TOP_DB)]
    overlap = compute_overlap_intervals(df_regular)

    duration_candidates = [x for x in [raw_duration, clean_duration] if pd.notna(x)]
    if df_regular is not None and not df_regular.empty:
        _, end_col = detect_time_cols(df_regular)
        if end_col:
            duration_candidates.append(pd.to_numeric(df_regular[end_col], errors="coerce").max())
    max_d = max(duration_candidates) if duration_candidates else 1.0

    fig, axes = plt.subplots(1, 2, figsize=(13, 3.8), gridspec_kw={"width_ratios": [2.2, 1]}, constrained_layout=True)
    ax = axes[0]
    ax.broken_barh([(0, max_d)], (1.6, 0.7), color=COLORS["light"])
    for start, end in nonsilent:
        ax.broken_barh([(start, end - start)], (1.6, 0.7), color=COLORS["green"], alpha=.88)
    ax.broken_barh([(0, max_d)], (0.3, 0.7), color=COLORS["light"])
    for start, end in overlap:
        ax.broken_barh([(start, end - start)], (0.3, 0.7), color=COLORS["red"], alpha=.9)
    ax.set_yticks([1.95, 0.65])
    ax.set_yticklabels(["Voz detectada\n(raw)", "Solapamiento\n(diarización)"])
    ax.set_xlim(0, max_d)
    ax.set_xlabel("Tiempo (raw en la fila superior; clean en la fila inferior)")
    ax.set_title("Dónde hay silencio y dónde hablan dos personas", loc="left")

    ax2 = axes[1]
    raw_v = raw_duration if pd.notna(raw_duration) else 0
    clean_v = clean_duration if pd.notna(clean_duration) else 0
    removed = max(0, raw_v - clean_v)
    ax2.barh(["Original", "Limpio"], [raw_v, clean_v], color=[COLORS["blue"], COLORS["green"]])
    for i, value in enumerate([raw_v, clean_v]):
        ax2.text(value, i, f" {value:.1f}s", va="center", fontsize=9)
    ax2.set_xlabel("Duración (s)")
    ax2.set_title(f"Reducción temporal · {removed:.1f}s", loc="left")
    plt.show()

    nonsilent_dur = sum(end - start for start, end in nonsilent)
    overlap_dur = sum(end - start for start, end in overlap)
    cards = [
        ("silencio estimado en raw", fmt_value(max(0, raw_duration - nonsilent_dur), "sec") if pd.notna(raw_duration) and nonsilent else "—"),
        ("eliminado/comprimido", fmt_value(max(0, raw_duration - clean_duration), "sec") if pd.notna(raw_duration) and pd.notna(clean_duration) else "—"),
        ("solapamiento", fmt_value(overlap_dur, "sec")),
        ("% overlap sobre clean", f"{100 * overlap_dur / clean_duration:.1f}%" if pd.notna(clean_duration) and clean_duration > 0 else "—"),
    ]
    kpi_cards(cards)


def render_phase00(audio_stem, tables, raw_path, clean_path, show_paths=False):
    section("Preparación del audio", "Resume la metadata del ejemplo y muestra qué cambia antes de entrar a diarización.")
    inv = load_inventory_rows(audio_stem)
    cleaning = load_cleaning_rows(audio_stem)
    meta = load_metadata_snapshot_rows(audio_stem)

    trans_col = "transcripcion" if "transcripcion" in meta.columns else None
    transcript_chars = int(meta[trans_col].fillna("").astype(str).str.len().max()) if trans_col and len(meta) else 0
    raw_d = metadata_value(cleaning, ["original_duration_sec"])
    if pd.isna(raw_d):
        raw_d = metadata_value(inv, ["duration_seconds"])
    clean_d = metadata_value(cleaning, ["clean_duration_sec"])
    removed_ratio = metadata_value(cleaning, ["removed_ratio"])

    cards = [
        ("dataset", fmt_value(metadata_value(inv, ["source_dataset"]))),
        ("marca", fmt_value(metadata_value(inv, ["brand_ds"]))),
        ("duración raw", fmt_value(raw_d, "sec")),
        ("duración clean", fmt_value(clean_d, "sec")),
        ("proporción reducida", fmt_value(removed_ratio, "pct")),
        ("sample rate original", f"{fmt_value(metadata_value(inv, ['sample_rate']))} Hz"),
        ("canales", fmt_value(metadata_value(inv, ["n_channels", "channels"]))),
        ("transcripción oficial", f"{transcript_chars:,} caracteres".replace(",", ".") if transcript_chars else "no disponible"),
    ]
    kpi_cards(cards)

    plot_raw_clean_waveforms(raw_path, clean_path)
    plot_silence_overlap_panel(raw_path, clean_path, tables.get("regular", pd.DataFrame()))

    section("Escucha de control", "Los reproductores se mantienen fuera de las figuras para poder comparar directamente el raw y el clean.")
    show_audio_player("Audio original", raw_path)
    show_audio_player("Audio limpio", clean_path)

    if show_paths:
        display(Markdown("**Rutas resueltas**"))
        display(pd.DataFrame([
            {"tipo": "raw", "ruta": str(raw_path) if raw_path else None},
            {"tipo": "clean", "ruta": str(clean_path) if clean_path else None},
        ]))


def plot_segment_funnel(tables):
    stages = [
        ("Regular", len(tables.get("regular", []))),
        ("Puntuados", len(tables.get("scored", []))),
        ("Válidos", len(tables.get("valid", []))),
        ("Anchors", len(tables.get("anchors", []))),
        ("Finales", len(tables.get("final", []))),
        ("Fusionados", len(tables.get("merged", []))),
    ]
    labels, vals = zip(*stages)
    fig, ax = plt.subplots(figsize=(10, 3.4))
    y = np.arange(len(labels))
    ax.barh(y, vals, color=[COLORS["gray"], COLORS["blue"], COLORS["green"], COLORS["purple"], COLORS["orange"], COLORS["ink"]])
    ax.set_yticks(y); ax.set_yticklabels(labels); ax.invert_yaxis()
    ax.set_xlabel("Número de segmentos")
    ax.set_title("Flujo de segmentos del audio", loc="left")
    for yi, v in zip(y, vals):
        ax.text(v, yi, f" {v}", va="center", fontsize=9)
    plt.show()


def timeline_on_axis(ax, df, speaker_col, title, max_duration, changed=None):
    if df is None or df.empty or speaker_col not in df.columns:
        ax.axis("off"); ax.set_title(f"{title} · sin datos", loc="left"); return
    s, e = detect_time_cols(df)
    if not s or not e:
        ax.axis("off"); return
    work = df.dropna(subset=[s, e, speaker_col]).copy()
    cmap = speaker_color_map(work[speaker_col])
    speakers = list(cmap)
    ymap = {spk: i for i, spk in enumerate(speakers)}
    for _, row in work.iterrows():
        start, end, spk = float(row[s]), float(row[e]), str(row[speaker_col])
        ax.broken_barh([(start, end - start)], (ymap[spk] - .34, .68), color=cmap[spk], alpha=.88)
    if changed is not None and not changed.empty:
        cs, ce = detect_time_cols(changed)
        if cs and ce:
            for _, row in changed.iterrows():
                ax.axvspan(float(row[cs]), float(row[ce]), facecolor="none", edgecolor=COLORS["red"], hatch="////", linewidth=.0, alpha=.6)
    ax.set_yticks(list(ymap.values())); ax.set_yticklabels(speakers)
    ax.set_xlim(0, max_duration); ax.set_title(title, loc="left")


def plot_speaker_timelines(tables):
    regular = tables.get("regular", pd.DataFrame())
    merged = tables.get("merged", pd.DataFrame())
    final = tables.get("final", pd.DataFrame())
    changed = tables.get("changed", pd.DataFrame())
    max_duration = 1.0
    for df in [regular, merged, final]:
        _, e = detect_time_cols(df)
        if e and not df.empty:
            max_duration = max(max_duration, pd.to_numeric(df[e], errors="coerce").max())
    fig, axes = plt.subplots(2, 1, figsize=(13, 5.0), sharex=True, constrained_layout=True)
    timeline_on_axis(axes[0], regular, "speaker", "Diarización inicial", max_duration)
    final_col = "speaker_final" if "speaker_final" in merged.columns else "speaker"
    timeline_on_axis(axes[1], merged, final_col, "Resultado final fusionado", max_duration, changed=changed)
    axes[1].set_xlabel("Tiempo de la llamada (s)")
    fig.suptitle("Antes y después del reetiquetado", x=.08, ha="left", fontsize=13, fontweight="bold")
    plt.show()


def plot_transition_matrix(df_final):
    if df_final is None or df_final.empty or not {"speaker_original", "speaker_final"}.issubset(df_final.columns):
        display(Markdown("No se puede construir la matriz de transición para este audio.")); return
    cm = pd.crosstab(df_final["speaker_original"], df_final["speaker_final"])
    row_pct = cm.div(cm.sum(axis=1).replace(0, np.nan), axis=0) * 100
    fig, ax = plt.subplots(figsize=(5.8, 4.4))
    im = ax.imshow(cm.values, cmap="Blues")
    ax.set_xticks(range(len(cm.columns))); ax.set_xticklabels(cm.columns)
    ax.set_yticks(range(len(cm.index))); ax.set_yticklabels(cm.index)
    ax.set_xlabel("Speaker final"); ax.set_ylabel("Speaker original")
    ax.set_title("Matriz de transición del reetiquetado", loc="left")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm.iloc[i,j]}\n{row_pct.iloc[i,j]:.0f}%", ha="center", va="center",
                    color="white" if cm.iloc[i,j] > cm.values.max()/2 else COLORS["ink"], fontsize=9)
    fig.colorbar(im, ax=ax, fraction=.045, pad=.04, label="Segmentos")
    plt.show()
    display(HTML("<div class='tfm-note'>Es una matriz de <b>transición</b>, no de confusión: muestra qué etiquetas cambiaron, pero no presupone que el speaker original sea verdad real.</div>"))


def audio_examples(df, clean_path, title, n, order_cols=None, text_col=None):
    section(title, "Tabla compacta y escucha directa de los segmentos más informativos.")
    if df is None or df.empty:
        display(Markdown("No hay ejemplos disponibles para este audio.")); return
    work = df.copy()
    if order_cols:
        valid_cols = [c for c in order_cols if c in work.columns]
        if valid_cols:
            work = work.sort_values(valid_cols, ascending=[False] * len(valid_cols))
    cols = [
        "inicio", "fin", "speaker", "speaker_original", "speaker_final", "anchor_rank",
        "duration", "rms_dbfs", "overlap_ratio", "distance_margin", "best_distance", "original_distance",
    ]
    if text_col:
        cols.append(text_col)
    compact_table(work, cols, n=n)
    s, e = detect_time_cols(work)
    if clean_path is None or not s or not e:
        return
    for idx, row in work.head(n).iterrows():
        snippet = extract_snippet(clean_path, row[s], row[e], f"{normalize_stem(Path(clean_path).stem)}_{title}_{idx}")
        if snippet:
            labels = []
            for c in ["speaker_original", "speaker_final", "speaker"]:
                if c in row.index and pd.notna(row[c]):
                    labels.append(f"{c}: {row[c]}")
            display(Markdown(f"**{fmt_time(row[s])}–{fmt_time(row[e])} · {' · '.join(labels)}**"))
            display(Audio(filename=str(snippet)))


def render_phase01(audio_stem, tables, clean_path):
    section("Diarización y resultado final", "Muestra cuántos segmentos sobreviven, cómo queda el timeline y dónde intervino el reetiquetado.")
    final = tables.get("final", pd.DataFrame())
    changed = tables.get("changed", pd.DataFrame())
    n_final = len(final)
    n_changed = len(changed)
    kpi_cards([
        ("segmentos válidos", fmt_value(len(tables.get("valid", [])))),
        ("anchors", fmt_value(len(tables.get("anchors", [])))),
        ("segmentos finales", fmt_value(n_final)),
        ("segmentos fusionados", fmt_value(len(tables.get("merged", [])))),
        ("reetiquetados", fmt_value(n_changed)),
        ("% reetiquetado", f"{100*n_changed/n_final:.1f}%" if n_final else "—"),
    ])
    plot_segment_funnel(tables)
    plot_speaker_timelines(tables)
    plot_transition_matrix(final)

    display(HTML(
        f"<div class='tfm-card'><b>Criterios de anchor:</b> duración ≥ {ANCHOR_MIN_DURATION_SEC:.2f}s · "
        f"overlap = {ANCHOR_MAX_OVERLAP_RATIO:.2f} · inicio ≥ {ANCHOR_INITIAL_EXCLUDE_SEC:.2f}s · "
        f"máximo {ANCHORS_PER_SPEAKER} por speaker.</div>"
    ))
    anchors = tables.get("anchors", pd.DataFrame())
    if not anchors.empty:
        sort_cols = [c for c in ["speaker", "anchor_rank"] if c in anchors.columns]
        anchors_show = anchors.sort_values(sort_cols, ascending=[True] * len(sort_cols)) if sort_cols else anchors
    else:
        anchors_show = anchors
    audio_examples(anchors_show, clean_path, "Anchors de alta confianza", min(6, len(anchors_show)), text_col=None)
    audio_examples(changed, clean_path, "Segmentos reetiquetados", min(6, len(changed)), order_cols=["distance_margin"])


def embedding_columns(df):
    return [c for c in df.columns if re.match(r"^emb_\d+", str(c))]


def changed_timestamp_set(df):
    if df is None or df.empty:
        return set()
    s, e = detect_time_cols(df)
    if not s or not e:
        return set()
    return set(zip(pd.to_numeric(df[s], errors="coerce").round(3), pd.to_numeric(df[e], errors="coerce").round(3)))


def mark_changed(df, changed):
    out = df.copy(); out["_changed"] = False
    s, e = detect_time_cols(out)
    pairs = changed_timestamp_set(changed)
    if s and e and pairs:
        out["_changed"] = [
            (round(float(a), 3), round(float(b), 3)) in pairs if pd.notna(a) and pd.notna(b) else False
            for a, b in zip(out[s], out[e])
        ]
    return out


def plot_embedding_validation(segment_emb, anchor_emb, changed):
    if segment_emb is None or segment_emb.empty:
        display(Markdown("No existen embeddings de segmentos para este audio.")); return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.1), constrained_layout=True)

    emb_cols = embedding_columns(segment_emb)
    if HAS_SKLEARN and emb_cols:
        frames = [segment_emb.assign(_type="segmento")]
        if anchor_emb is not None and not anchor_emb.empty and set(emb_cols).issubset(anchor_emb.columns):
            frames.append(anchor_emb.assign(_type="anchor"))
        both = pd.concat(frames, ignore_index=True)
        valid = both[emb_cols].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)
        both = both.loc[valid].copy()
        X = StandardScaler().fit_transform(both[emb_cols])
        pca = PCA(n_components=2, random_state=42)
        xy = pca.fit_transform(X)
        both["PC1"], both["PC2"] = xy[:,0], xy[:,1]
        both = mark_changed(both, changed)
        spk_col = next((c for c in ["speaker_final", "speaker_original", "speaker"] if c in both.columns), None)
        seg = both[both["_type"] == "segmento"]
        if spk_col:
            cmap = speaker_color_map(seg[spk_col])
            for spk, group in seg.groupby(spk_col):
                axes[0].scatter(group.PC1, group.PC2, s=38, alpha=.63, color=cmap[str(spk)], label=str(spk))
        else:
            axes[0].scatter(seg.PC1, seg.PC2, s=38, alpha=.63)
        rel = seg[seg["_changed"]]
        if len(rel):
            axes[0].scatter(rel.PC1, rel.PC2, marker="X", s=110, color=COLORS["red"], edgecolors="black", label="reetiquetado")
        anc = both[both["_type"] == "anchor"]
        if len(anc):
            axes[0].scatter(anc.PC1, anc.PC2, marker="*", s=190, color=COLORS["ink"], edgecolors="black", label="anchor")
        axes[0].set_title(f"PCA de embeddings · varianza {pca.explained_variance_ratio_.sum():.0%}", loc="left")
        axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2"); axes[0].legend(fontsize=8, frameon=False)
    else:
        axes[0].axis("off"); axes[0].set_title("PCA no disponible", loc="left")

    dist_cols = sorted([c for c in segment_emb.columns if str(c).startswith("dist_")])
    if len(dist_cols) >= 2:
        work = mark_changed(segment_emb, changed)
        x, y = dist_cols[:2]
        axes[1].scatter(work[x], work[y], s=38, alpha=.55, color=COLORS["blue"])
        rel = work[work["_changed"]]
        if len(rel):
            axes[1].scatter(rel[x], rel[y], marker="X", s=110, color=COLORS["red"], edgecolors="black", label="reetiquetado")
        low = min(pd.to_numeric(work[x], errors="coerce").min(), pd.to_numeric(work[y], errors="coerce").min())
        high = max(pd.to_numeric(work[x], errors="coerce").max(), pd.to_numeric(work[y], errors="coerce").max())
        axes[1].plot([low, high], [low, high], "--", color=COLORS["gray"], label="misma distancia")
        axes[1].set_xlabel(x); axes[1].set_ylabel(y)
        axes[1].set_title("Distancia a los centroides", loc="left"); axes[1].legend(fontsize=8, frameon=False)
    else:
        axes[1].axis("off"); axes[1].set_title("Distancias no disponibles", loc="left")
    plt.show()


def plot_margin_decisions(df_final):
    if df_final is None or df_final.empty or "distance_margin" not in df_final.columns:
        display(Markdown("No hay márgenes de decisión para este audio.")); return
    work = df_final.copy()
    s, _ = detect_time_cols(work)
    x = pd.to_numeric(work[s], errors="coerce") if s else np.arange(len(work))
    y = pd.to_numeric(work["distance_margin"], errors="coerce")
    if "was_reclassified" in work.columns:
        changed = work["was_reclassified"].fillna(False).astype(bool)
    else:
        changed = work.get("speaker_original", pd.Series(index=work.index, dtype=str)).astype(str) != work.get("speaker_final", pd.Series(index=work.index, dtype=str)).astype(str)
    fig, ax = plt.subplots(figsize=(12, 3.8))
    ax.scatter(x[~changed], y[~changed], s=35, alpha=.60, color=COLORS["blue"], label="sin cambio")
    ax.scatter(x[changed], y[changed], s=70, marker="X", color=COLORS["red"], label="reetiquetado")
    ax.axhline(RELABEL_MIN_MARGIN, linestyle="--", color=COLORS["ink"], label=f"margen actual = {RELABEL_MIN_MARGIN}")
    ax.set_xlabel("Tiempo (s)" if s else "Segmento"); ax.set_ylabel("distance_margin")
    ax.set_title("Seguridad de la decisión de reetiquetado", loc="left")
    ax.legend(frameon=False, ncol=3)
    plt.show()


def plot_selected_audio_sensitivity(audio_stem):
    df = read_csv_cached(str(MARGIN_BY_AUDIO_CSV))
    df = filter_audio(df, audio_stem)
    if df.empty or "tested_relabel_margin" not in df.columns:
        display(Markdown("El Notebook 02 no ha generado sensibilidad por audio para este ejemplo.")); return
    df = df.sort_values("tested_relabel_margin")
    y_col = "n_changed_segments" if "n_changed_segments" in df.columns else None
    if not y_col:
        return
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.plot(df["tested_relabel_margin"], df[y_col], marker="o", color=COLORS["purple"])
    ax.axvline(RELABEL_MIN_MARGIN, linestyle="--", color=COLORS["ink"], label="valor usado")
    ax.set_xlabel("Margen probado"); ax.set_ylabel("Segmentos reetiquetados")
    ax.set_title("Sensibilidad del audio al margen de relabeling", loc="left")
    ax.legend(frameon=False)
    plt.show()


def render_phase23(audio_stem, tables):
    section("Validación interna del reetiquetado", "Comprueba si los cambios son coherentes con los embeddings y cuánto dependen del margen elegido.")
    plot_embedding_validation(tables.get("segment_embeddings"), tables.get("anchor_embeddings"), tables.get("changed"))
    plot_margin_decisions(tables.get("final"))
    plot_selected_audio_sensitivity(audio_stem)
    display(HTML(
        "<div class='tfm-note'>El PCA sirve como inspección visual; la decisión real se toma con las distancias a centroides y el margen mínimo de 0,01.</div>"
    ))


def plot_transcript_density(df, text_col):
    if df.empty or not text_col:
        return
    s, _ = detect_time_cols(df)
    work = df.copy()
    work["_words"] = work[text_col].fillna("").astype(str).str.findall(r"\b\w+\b").str.len()
    x = pd.to_numeric(work[s], errors="coerce") if s else np.arange(len(work))
    speaker_col = next((c for c in ["speaker_final", "speaker"] if c in work.columns), None)
    fig, ax = plt.subplots(figsize=(12, 3.8))
    if speaker_col:
        cmap = speaker_color_map(work[speaker_col])
        for spk, group in work.groupby(speaker_col):
            gx = pd.to_numeric(group[s], errors="coerce") if s else group.index
            ax.scatter(gx, group["_words"], s=45, alpha=.68, color=cmap[str(spk)], label=str(spk))
        ax.legend(frameon=False, ncol=2)
    else:
        ax.scatter(x, work["_words"], s=45, alpha=.68, color=COLORS["blue"])
    ax.set_xlabel("Tiempo (s)" if s else "Segmento"); ax.set_ylabel("Palabras")
    ax.set_title("Cantidad de texto recuperado por segmento", loc="left")
    plt.show()


def render_phase05(audio_stem, clean_path):
    section("Transcripción segmentada", "Mantiene los timestamps y el speaker final para conectar el audio con las fases textuales.")
    df, path = load_transcription(audio_stem)
    if df.empty:
        display(Markdown("No se encontró el output del Notebook 05 para este audio.")); return
    text_col = detect_text_col(df)
    status_col = next((c for c in ["transcription_status", "status"] if c in df.columns), None)
    n_text = int(df[text_col].fillna("").astype(str).str.strip().ne("").sum()) if text_col else 0
    total_words = int(df[text_col].fillna("").astype(str).str.findall(r"\b\w+\b").str.len().sum()) if text_col else 0
    kpi_cards([
        ("segmentos", fmt_value(len(df))),
        ("con texto", fmt_value(n_text)),
        ("cobertura", f"{100*n_text/len(df):.1f}%" if len(df) else "—"),
        ("palabras", fmt_value(total_words)),
        ("archivo", Path(path).name if path else "—"),
    ])
    plot_transcript_density(df, text_col)
    cols = ["inicio", "fin", "speaker_final", "speaker", status_col, text_col]
    compact_table(df.sort_values(detect_time_cols(df)[0]) if detect_time_cols(df)[0] else df, cols, n=18)
    audio_examples(df.sort_values(detect_time_cols(df)[0]) if detect_time_cols(df)[0] else df,
                   clean_path, "Tres segmentos transcritos", min(3, len(df)), text_col=text_col)


def annotated_matrix(matrix, title, xlabel, ylabel, cmap="Blues", as_percent=False):
    if matrix is None or matrix.empty:
        display(Markdown("No hay datos suficientes para construir la matriz.")); return
    fig, ax = plt.subplots(figsize=(5.8, 4.3))
    im = ax.imshow(matrix.values, cmap=cmap)
    ax.set_xticks(range(len(matrix.columns))); ax.set_xticklabels(matrix.columns)
    ax.set_yticks(range(len(matrix.index))); ax.set_yticklabels(matrix.index)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title, loc="left")
    max_v = np.nanmax(matrix.values) if matrix.size else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix.iloc[i, j]
            txt = f"{val:.1f}%" if as_percent else f"{int(val)}"
            ax.text(j, i, txt, ha="center", va="center", color="white" if max_v and val > max_v/2 else COLORS["ink"])
    fig.colorbar(im, ax=ax, fraction=.045, pad=.04)
    plt.show()


def plot_alignment_scores(matches):
    if matches.empty or "combined_score" not in matches.columns:
        return
    work = matches.sort_values("combined_score", ascending=True).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.barh(np.arange(len(work)), work["combined_score"], color=COLORS["green"])
    ax.axvline(.70, linestyle="--", color=COLORS["ink"], label="umbral recomendado 0,70")
    ax.set_xlabel("combined_score"); ax.set_ylabel("Match aceptado")
    ax.set_title("Calidad de los matches Whisper ↔ transcripción oficial", loc="left")
    ax.legend(frameon=False)
    plt.show()


def render_phase06(audio_stem):
    section("Ground truth proxy de roles", "Alinea Whisper con la transcripción oficial y evalúa la asignación final AGENT/CLIENT.")
    out = load_proxy_outputs(audio_stem)
    proxy = out["segment_level_proxy_groundtruth"]
    mapping = out["speaker_role_mapping_textual"]
    matches = out["text_alignment_matches"]
    official = out["official_transcription_turns_extracted"]
    holdout = out["holdout_role_mapping_predictions"]

    if all(df.empty for df in [proxy, mapping, matches, official, holdout]):
        display(Markdown("No se encontraron outputs del Notebook 06 para este audio.")); return

    role_col = next((c for c in ["official_role_proxy", "role_proxy", "probable_role"] if c in proxy.columns), None)
    mapped = int(proxy[role_col].isin(["AGENT", "CLIENT"]).sum()) if role_col else 0
    kpi_cards([
        ("segmentos proxy", fmt_value(len(proxy))),
        ("con rol AGENT/CLIENT", fmt_value(mapped)),
        ("cobertura de rol", f"{100*mapped/len(proxy):.1f}%" if len(proxy) else "—"),
        ("matches aceptados", fmt_value(len(matches))),
        ("turnos oficiales", fmt_value(len(official))),
        ("speakers mapeados", fmt_value(mapping["speaker_final"].nunique()) if "speaker_final" in mapping.columns else "—"),
    ])

    section("Mapping speaker → rol", "Resume la decisión de rol aplicada sobre los speakers finales del audio.")
    compact_table(mapping, [
        "audio_file", "speaker_final", "probable_role", "role_confidence",
        "n_matches_total", "role_mapping_status", "agent_speaker", "client_speaker"
    ], n=8)

    if not proxy.empty and role_col and "speaker_final" in proxy.columns:
        evidence = pd.crosstab(proxy["speaker_final"], proxy[role_col])
        annotated_matrix(evidence, "Evidencia por speaker final y rol proxy", "Rol proxy", "Speaker final")
        display(HTML("<div class='tfm-note'>Esta matriz muestra la evidencia usada en el audio; no es todavía una matriz de confusión.</div>"))

    plot_alignment_scores(matches)
    if not matches.empty:
        compact_table(matches, [
            "official_role", "speaker_final", "combined_score", "char_cosine",
            "token_containment", "similarity_margin", "official_text", "whisper_text"
        ], n=10)

    section("Matriz de confusión holdout", "Compara el rol oficial con el rol predicho usando evidencia separada de entrenamiento.")
    if not holdout.empty and {"official_role", "predicted_role_from_train"}.issubset(holdout.columns):
        eval_df = holdout.dropna(subset=["official_role", "predicted_role_from_train"])
        labels = ["AGENT", "CLIENT"]
        cm = pd.crosstab(eval_df["official_role"], eval_df["predicted_role_from_train"]).reindex(index=labels, columns=labels, fill_value=0)
        annotated_matrix(cm, "Confusión del proxy en holdout", "Rol predicho", "Rol oficial")
        acc = np.mean(eval_df["official_role"].astype(str) == eval_df["predicted_role_from_train"].astype(str)) if len(eval_df) else np.nan
        kpi_cards([("matches holdout del audio", fmt_value(len(eval_df))), ("accuracy del audio", f"{100*acc:.1f}%" if pd.notna(acc) else "—")])
    else:
        display(Markdown("Este audio no quedó representado en la partición holdout o no tuvo evidencia suficiente para una predicción evaluable."))


def render_phase07(audio_stem):
    section("Sentimiento textual", "Describe qué expresan las palabras de cada segmento; no utiliza todavía la señal vocal.")
    df, path = load_sentiment(audio_stem)
    if df.empty:
        display(Markdown("No se encontró el output del Notebook 07 para este audio.")); return
    label_col = "sentiment_label" if "sentiment_label" in df.columns else None
    numeric_col = "sentiment_numeric" if "sentiment_numeric" in df.columns else None
    score_col = "sentiment_score" if "sentiment_score" in df.columns else None
    text_col = detect_text_col(df)
    role_col = next((c for c in ["role_proxy", "official_role_proxy"] if c in df.columns), None)

    counts = df[label_col].value_counts() if label_col else pd.Series(dtype=int)
    kpi_cards([
        ("segmentos analizados", fmt_value(len(df))),
        ("negativos", fmt_value(counts.get("negative", 0))),
        ("neutrales", fmt_value(counts.get("neutral", 0))),
        ("positivos", fmt_value(counts.get("positive", 0))),
        ("confianza media", fmt_value(pd.to_numeric(df[score_col], errors="coerce").mean()) if score_col else "—"),
    ])

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), constrained_layout=True)
    if label_col:
        order = ["negative", "neutral", "positive"]
        values = [counts.get(x, 0) for x in order]
        axes[0].bar(order, values, color=[COLORS["red"], COLORS["gray"], COLORS["green"]])
        axes[0].set_ylabel("Segmentos"); axes[0].set_title("Distribución del sentimiento", loc="left")
        for i, v in enumerate(values): axes[0].text(i, v, f"{v}", ha="center", va="bottom")
    else:
        axes[0].axis("off")

    if numeric_col:
        work = df.copy(); s, _ = detect_time_cols(work)
        x = pd.to_numeric(work[s], errors="coerce") if s else np.arange(len(work))
        if role_col:
            cmap = {"AGENT": COLORS["blue"], "CLIENT": COLORS["orange"]}
            for role, group in work.groupby(role_col):
                gx = pd.to_numeric(group[s], errors="coerce") if s else group.index
                axes[1].scatter(gx, group[numeric_col], s=52, alpha=.72, color=cmap.get(str(role), COLORS["gray"]), label=str(role))
            axes[1].legend(frameon=False)
        else:
            axes[1].scatter(x, work[numeric_col], s=52, alpha=.72, color=COLORS["blue"])
        axes[1].axhline(0, color=COLORS["gray"], linewidth=1)
        axes[1].set_yticks([-1, 0, 1]); axes[1].set_yticklabels(["negativo", "neutral", "positivo"])
        axes[1].set_xlabel("Tiempo (s)" if s else "Segmento")
        axes[1].set_title("Evolución textual de la llamada", loc="left")
    else:
        axes[1].axis("off")
    plt.show()

    section("Segmentos representativos", "Se priorizan predicciones de alta confianza para facilitar la revisión manual.")
    work = df.sort_values(score_col, ascending=False) if score_col else df
    compact_table(work, ["inicio", "fin", "speaker_final", role_col, text_col, label_col, score_col], n=12)


def render_phase07b(audio_stem):
    section("Afecto y prosodia", "Describe cómo suena la voz. Se presenta separado del texto hasta construir el Notebook 07C.")
    df, path = load_prosody(audio_stem)
    if df.empty:
        display(Markdown("El Notebook 07B todavía no ha generado output para este audio. La pestaña se completará automáticamente cuando exista el CSV.")); return
    s, _ = detect_time_cols(df)
    if s:
        df = df.sort_values(s).reset_index(drop=True)
    x = pd.to_numeric(df[s], errors="coerce") if s else np.arange(len(df))

    score_cols = [c for c in ["arousal_proxy_score", "tension_proxy_score", "intensity_proxy_score"] if c in df.columns]
    state_col = next((c for c in ["prosodic_state_proxy", "ser_pred_label"] if c in df.columns), None)
    kpi_cards([
        ("segmentos 07B", fmt_value(len(df))),
        ("arousal medio", fmt_value(pd.to_numeric(df.get("arousal_proxy_score"), errors="coerce").mean()) if "arousal_proxy_score" in df.columns else "—"),
        ("tensión media", fmt_value(pd.to_numeric(df.get("tension_proxy_score"), errors="coerce").mean()) if "tension_proxy_score" in df.columns else "—"),
        ("intensidad media", fmt_value(pd.to_numeric(df.get("intensity_proxy_score"), errors="coerce").mean()) if "intensity_proxy_score" in df.columns else "—"),
        ("estado dominante", fmt_value(df[state_col].mode().iloc[0]) if state_col and len(df[state_col].dropna()) else "—"),
    ])

    fig, axes = plt.subplots(2, 1, figsize=(13, 7.0), constrained_layout=True)
    line_colors = [COLORS["blue"], COLORS["red"], COLORS["orange"]]
    if score_cols:
        for col, color in zip(score_cols, line_colors):
            axes[0].plot(x, pd.to_numeric(df[col], errors="coerce"), marker="o", markersize=3.5, linewidth=1.4, label=col.replace("_proxy_score", ""), color=color)
        axes[0].set_ylim(-.03, 1.03); axes[0].set_ylabel("Score 0–1")
        axes[0].set_title("Evolución de activación, tensión e intensidad", loc="left")
        axes[0].legend(frameon=False, ncol=3)
    else:
        axes[0].axis("off"); axes[0].set_title("Scores afectivos no disponibles", loc="left")

    feature_pairs = [("rms_audio_dbfs", "energía RMS"), ("pitch_mean", "pitch medio")]
    available = [(c, label) for c, label in feature_pairs if c in df.columns]
    if available:
        ax_left = axes[1]
        c1, l1 = available[0]
        ax_left.plot(x, pd.to_numeric(df[c1], errors="coerce"), color=COLORS["green"], label=l1)
        ax_left.set_ylabel(l1); ax_left.set_xlabel("Tiempo (s)" if s else "Segmento")
        if len(available) > 1:
            c2, l2 = available[1]
            ax_right = ax_left.twinx()
            ax_right.plot(x, pd.to_numeric(df[c2], errors="coerce"), color=COLORS["purple"], alpha=.75, label=l2)
            ax_right.set_ylabel(l2)
        ax_left.set_title("Señales prosódicas interpretables", loc="left")
    else:
        axes[1].axis("off"); axes[1].set_title("Features prosódicas no disponibles", loc="left")
    plt.show()

    if state_col:
        counts = df[state_col].value_counts().head(8)
        fig, ax = plt.subplots(figsize=(9, 3.5))
        ax.bar(counts.index.astype(str), counts.values, color=COLORS["purple"])
        ax.set_ylabel("Segmentos"); ax.set_title("Estados prosódicos / emociones predominantes", loc="left")
        ax.tick_params(axis="x", rotation=25)
        plt.show()

    tension_col = "tension_proxy_score" if "tension_proxy_score" in df.columns else None
    work = df.sort_values(tension_col, ascending=False) if tension_col else df
    compact_table(work, [
        "inicio", "fin", "speaker_final", "role_proxy_for_prosody", "prosodic_state_proxy",
        "ser_pred_label", "ser_pred_score", "arousal_proxy_score", "tension_proxy_score",
        "intensity_proxy_score", "rms_audio_dbfs", "pitch_mean"
    ], n=12)


def _keyword_theme_columns(df):
    count_columns = [
        column
        for column in df.columns
        if str(column).startswith("kw_")
        and str(column).endswith("_count")
    ]

    themes = {}

    for column in count_columns:
        theme = str(column)[3:-6]
        themes[theme] = column

    return themes


def render_phase08(audio_stem):
    section(
        "Keywords y temas críticos",
        "Localiza expresiones de interés en la transcripción y resume la criticidad temática de la llamada.",
    )

    outputs = load_keywords(audio_stem)
    segments = outputs["segments"]
    calls = outputs["calls"]

    if segments.empty and calls.empty:
        display(Markdown(
            "No se encontró el output del Notebook 08 para este audio."
        ))
        return

    call_row = calls.iloc[0] if not calls.empty else pd.Series(dtype=object)

    if not segments.empty:
        has_keyword = (
            segments["has_critical_keyword"].fillna(False).astype(bool)
            if "has_critical_keyword" in segments.columns
            else pd.to_numeric(
                segments.get(
                    "total_keyword_matches",
                    pd.Series(0, index=segments.index),
                ),
                errors="coerce",
            ).fillna(0).gt(0)
        )

        n_segments = int(len(segments))
        n_segments_keywords = int(has_keyword.sum())
        total_matches = int(
            pd.to_numeric(
                segments.get(
                    "total_keyword_matches",
                    pd.Series(0, index=segments.index),
                ),
                errors="coerce",
            ).fillna(0).sum()
        )
    else:
        n_segments = int(call_row.get("n_segments", 0) or 0)
        n_segments_keywords = int(
            call_row.get("n_segments_with_keywords", 0) or 0
        )
        total_matches = int(
            call_row.get("total_keyword_matches", 0) or 0
        )
        has_keyword = pd.Series(dtype=bool)

    pct_with_keywords = (
        float(call_row.get("pct_segments_with_keywords"))
        if "pct_segments_with_keywords" in call_row
        and pd.notna(call_row.get("pct_segments_with_keywords"))
        else (
            n_segments_keywords / n_segments
            if n_segments
            else np.nan
        )
    )

    n_themes = (
        int(call_row.get("n_distinct_critical_themes", 0) or 0)
        if len(call_row)
        else 0
    )

    criticality_score = (
        call_row.get("keyword_criticality_score", np.nan)
        if len(call_row)
        else np.nan
    )

    criticality_percentile = (
        call_row.get("keyword_criticality_percentile", np.nan)
        if len(call_row)
        else np.nan
    )

    kpi_cards([
        ("segmentos analizados", fmt_value(n_segments)),
        ("segmentos con keywords", fmt_value(n_segments_keywords)),
        (
            "cobertura temática",
            fmt_value(pct_with_keywords, "pct")
            if pd.notna(pct_with_keywords)
            else "—",
        ),
        ("coincidencias totales", fmt_value(total_matches)),
        ("temas distintos", fmt_value(n_themes)),
        ("score de criticidad", fmt_value(criticality_score)),
        (
            "percentil de criticidad",
            fmt_value(criticality_percentile, "pct")
            if pd.notna(criticality_percentile)
            else "—",
        ),
    ])

    if not segments.empty:
        theme_columns = _keyword_theme_columns(segments)

        fig, axes = plt.subplots(
            1,
            2,
            figsize=(13, 4.4),
            constrained_layout=True,
        )

        if theme_columns:
            theme_counts = pd.Series({
                theme: pd.to_numeric(
                    segments[column],
                    errors="coerce",
                ).fillna(0).sum()
                for theme, column in theme_columns.items()
            }).sort_values(ascending=False)

            theme_counts = theme_counts[theme_counts.gt(0)].head(10)

            if len(theme_counts):
                axes[0].bar(
                    theme_counts.index.astype(str),
                    theme_counts.values,
                    color=COLORS["purple"],
                )
                axes[0].set_ylabel("Coincidencias")
                axes[0].set_title(
                    "Temas detectados en la llamada",
                    loc="left",
                )
                axes[0].tick_params(
                    axis="x",
                    rotation=30,
                )
            else:
                axes[0].axis("off")
                axes[0].set_title(
                    "No se detectaron temas críticos",
                    loc="left",
                )
        else:
            axes[0].axis("off")
            axes[0].set_title(
                "Columnas temáticas no disponibles",
                loc="left",
            )

        start_col, _ = detect_time_cols(segments)
        match_values = pd.to_numeric(
            segments.get(
                "total_keyword_matches",
                pd.Series(0, index=segments.index),
            ),
            errors="coerce",
        ).fillna(0)

        x = (
            pd.to_numeric(
                segments[start_col],
                errors="coerce",
            )
            if start_col
            else np.arange(len(segments))
        )

        role_col = next(
            (
                column
                for column in [
                    "role_proxy",
                    "official_role_proxy",
                    "speaker_final",
                ]
                if column in segments.columns
            ),
            None,
        )

        if role_col:
            color_map = {
                "AGENT": COLORS["blue"],
                "CLIENT": COLORS["orange"],
            }

            for role, group in segments.groupby(role_col):
                group_matches = pd.to_numeric(
                    group.get(
                        "total_keyword_matches",
                        pd.Series(0, index=group.index),
                    ),
                    errors="coerce",
                ).fillna(0)

                group_x = (
                    pd.to_numeric(
                        group[start_col],
                        errors="coerce",
                    )
                    if start_col
                    else group.index
                )

                axes[1].scatter(
                    group_x,
                    group_matches,
                    s=52,
                    alpha=.72,
                    color=color_map.get(
                        str(role),
                        COLORS["gray"],
                    ),
                    label=str(role),
                )

            axes[1].legend(
                frameon=False,
                ncol=2,
            )
        else:
            axes[1].scatter(
                x,
                match_values,
                s=52,
                alpha=.72,
                color=COLORS["red"],
            )

        axes[1].set_xlabel(
            "Tiempo (s)"
            if start_col
            else "Segmento"
        )
        axes[1].set_ylabel("Coincidencias")
        axes[1].set_title(
            "Distribución temporal de las alertas",
            loc="left",
        )

        plt.show()

        section(
            "Segmentos con evidencia temática",
            "Se muestran primero los segmentos con mayor número de coincidencias.",
        )

        work = segments.copy()

        if "total_keyword_matches" in work.columns:
            work = work.sort_values(
                "total_keyword_matches",
                ascending=False,
            )

        if "has_critical_keyword" in work.columns:
            filtered = work[
                work["has_critical_keyword"]
                .fillna(False)
                .astype(bool)
            ]

            if not filtered.empty:
                work = filtered

        text_col = detect_text_col(work)
        role_col = next(
            (
                column
                for column in [
                    "role_proxy",
                    "official_role_proxy",
                    "speaker_final",
                ]
                if column in work.columns
            ),
            None,
        )

        compact_table(
            work,
            [
                "inicio",
                "fin",
                "speaker_final",
                role_col,
                text_col,
                "total_keyword_matches",
                "n_critical_themes",
                "critical_themes_detected",
            ],
            n=14,
        )

    if not calls.empty:
        section(
            "Resumen de la llamada",
            "La agregación por llamada combina cobertura, número de coincidencias y variedad temática.",
        )

        compact_table(
            calls,
            [
                "audio_file",
                "n_segments",
                "n_segments_with_keywords",
                "pct_segments_with_keywords",
                "total_keyword_matches",
                "n_distinct_critical_themes",
                "critical_themes_detected",
                "keyword_criticality_score",
                "keyword_criticality_percentile",
                "combined_criticality_score",
            ],
            n=3,
        )


def _voiceprint_summary_value(summary_df, metric_name):
    if (
        summary_df is None
        or summary_df.empty
        or "metric" not in summary_df.columns
        or "value" not in summary_df.columns
    ):
        return np.nan

    rows = summary_df[
        summary_df["metric"].astype(str).eq(metric_name)
    ]

    if rows.empty:
        return np.nan

    return pd.to_numeric(
        rows.iloc[0]["value"],
        errors="coerce",
    )


def render_phase09(audio_stem):
    section(
        "Huella de voz e identificación open-set",
        "Construye muestras acústicas por persona y audio, compara cada consulta con perfiles enrolados y permite rechazar voces desconocidas.",
    )

    outputs = load_voiceprint_outputs(audio_stem)

    segments = outputs["segments"]
    samples = outputs["samples"]
    predictions = outputs["predictions"]
    identity_summary = outputs["identity_summary"]
    identity_split = outputs["identity_split"]
    final_summary = outputs["final_summary"]

    if (
        segments.empty
        and samples.empty
        and predictions.empty
        and final_summary.empty
    ):
        display(Markdown(
            "No se encontró el output del Notebook 09 para este audio."
        ))
        return

    role_source = (
        samples
        if not samples.empty
        else segments
    )

    role_col = next(
        (
            column
            for column in [
                "role_proxy",
                "official_role_proxy",
            ]
            if column in role_source.columns
        ),
        None,
    )

    role_counts = (
        role_source[role_col].value_counts()
        if role_col
        else pd.Series(dtype=int)
    )

    n_persons = 0

    for frame in [samples, segments]:
        if not frame.empty and "person_id" in frame.columns:
            n_persons = max(
                n_persons,
                int(frame["person_id"].nunique()),
            )

    threshold = _voiceprint_summary_value(
        final_summary,
        "umbral_aceptacion",
    )
    open_set_accuracy = _voiceprint_summary_value(
        final_summary,
        "open_set_accuracy",
    )
    known_identification_rate = _voiceprint_summary_value(
        final_summary,
        "known_identification_rate",
    )
    unknown_rejection_rate = _voiceprint_summary_value(
        final_summary,
        "unknown_rejection_rate",
    )

    kpi_cards([
        ("segmentos candidatos", fmt_value(len(segments))),
        ("muestras audio-persona", fmt_value(len(samples))),
        ("identidades en la llamada", fmt_value(n_persons)),
        ("muestras AGENT", fmt_value(role_counts.get("AGENT", 0))),
        ("muestras CLIENT", fmt_value(role_counts.get("CLIENT", 0))),
        ("consultas open-set", fmt_value(len(predictions))),
        ("umbral global", fmt_value(threshold)),
    ])

    display(HTML(
        "<div class='tfm-card'>"
        f"<div class='tfm-kpi'><div class='v'>{fmt_value(known_identification_rate, 'pct') if pd.notna(known_identification_rate) else '—'}</div><div class='l'>identificación known global</div></div>"
        f"<div class='tfm-kpi'><div class='v'>{fmt_value(unknown_rejection_rate, 'pct') if pd.notna(unknown_rejection_rate) else '—'}</div><div class='l'>rechazo unknown global</div></div>"
        f"<div class='tfm-kpi'><div class='v'>{fmt_value(open_set_accuracy, 'pct') if pd.notna(open_set_accuracy) else '—'}</div><div class='l'>accuracy open-set global</div></div>"
        "</div>"
    ))

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13, 4.4),
        constrained_layout=True,
    )

    if not samples.empty:
        duration_col = next(
            (
                column
                for column in [
                    "sample_duration_sec",
                    "duration",
                ]
                if column in samples.columns
            ),
            None,
        )

        label_col = next(
            (
                column
                for column in [
                    "role_proxy",
                    "person_id",
                    "speaker_final",
                ]
                if column in samples.columns
            ),
            None,
        )

        if duration_col and label_col:
            plot_df = samples.copy()
            plot_df["_duration"] = pd.to_numeric(
                plot_df[duration_col],
                errors="coerce",
            ).fillna(0)

            grouped = (
                plot_df
                .groupby(label_col)["_duration"]
                .sum()
                .sort_values(ascending=False)
            )

            axes[0].bar(
                grouped.index.astype(str),
                grouped.values,
                color=[
                    COLORS["blue"]
                    if str(value) == "AGENT"
                    else (
                        COLORS["orange"]
                        if str(value) == "CLIENT"
                        else COLORS["purple"]
                    )
                    for value in grouped.index
                ],
            )
            axes[0].set_ylabel("Duración acumulada (s)")
            axes[0].set_title(
                "Audio utilizado para formar la huella",
                loc="left",
            )
            axes[0].tick_params(
                axis="x",
                rotation=20,
            )
        else:
            axes[0].axis("off")
            axes[0].set_title(
                "Duración de muestras no disponible",
                loc="left",
            )
    elif not segments.empty and role_col:
        segment_counts = segments[role_col].value_counts()

        axes[0].bar(
            segment_counts.index.astype(str),
            segment_counts.values,
            color=[
                COLORS["blue"]
                if str(value) == "AGENT"
                else COLORS["orange"]
                for value in segment_counts.index
            ],
        )
        axes[0].set_ylabel("Segmentos")
        axes[0].set_title(
            "Segmentos candidatos por rol",
            loc="left",
        )
    else:
        axes[0].axis("off")
        axes[0].set_title(
            "Muestras del audio no disponibles",
            loc="left",
        )

    if not predictions.empty and "best_similarity" in predictions.columns:
        plot_predictions = predictions.copy()

        plot_predictions["_similarity"] = pd.to_numeric(
            plot_predictions["best_similarity"],
            errors="coerce",
        )

        labels = (
            plot_predictions["decision"].astype(str)
            if "decision" in plot_predictions.columns
            else pd.Series(
                [
                    f"consulta {index + 1}"
                    for index in range(len(plot_predictions))
                ],
                index=plot_predictions.index,
            )
        )

        x = np.arange(len(plot_predictions))

        axes[1].bar(
            x,
            plot_predictions["_similarity"],
            color=[
                COLORS["green"]
                if str(decision) == "KNOWN"
                else COLORS["red"]
                for decision in labels
            ],
        )

        if pd.notna(threshold):
            axes[1].axhline(
                threshold,
                linestyle="--",
                color=COLORS["gray"],
                label=f"umbral {threshold:.3f}",
            )
            axes[1].legend(
                frameon=False,
            )

        axes[1].set_xticks(x)
        axes[1].set_xticklabels(
            labels,
            rotation=20,
        )
        axes[1].set_ylabel("Similitud coseno")
        axes[1].set_title(
            "Decisión open-set del audio",
            loc="left",
        )
    else:
        axes[1].axis("off")
        axes[1].set_title(
            "El audio no fue consulta del conjunto open-set",
            loc="left",
        )

    plt.show()

    section(
        "Muestras e identidades acústicas",
        "Una muestra resume los segmentos de una persona dentro de una llamada antes de compararla con otros audios.",
    )

    if not samples.empty:
        compact_table(
            samples,
            [
                "sample_id",
                "audio_key",
                "audio_file",
                "speaker_final",
                "role_proxy",
                "person_id",
                "n_segments",
                "sample_duration_sec",
                "sample_n_words",
                "mean_overlap_ratio",
                "mean_rms_dbfs",
            ],
            n=10,
        )
    elif not segments.empty:
        compact_table(
            segments,
            [
                "inicio",
                "fin",
                "audio_key",
                "speaker_final",
                "role_proxy",
                "person_id",
                "duration",
                "overlap_ratio",
                "rms_dbfs",
                "n_words",
            ],
            n=12,
        )

    section(
        "Resultado de identificación",
        "Solo aparece una predicción si la muestra del audio quedó incluida como consulta de evaluación; su ausencia no significa que la huella haya fallado.",
    )

    if not predictions.empty:
        compact_table(
            predictions,
            [
                "query_group",
                "audio_key",
                "true_source_identity_id",
                "true_person_id",
                "best_profile_id",
                "best_source_identity_id",
                "best_similarity",
                "top1_top2_margin",
                "decision",
                "identification_correct",
            ],
            n=10,
        )
    else:
        display(Markdown(
            "Este audio alimentó la base de huellas, pero no fue seleccionado como consulta del conjunto de evaluación open-set."
        ))

    if not identity_summary.empty or not identity_split.empty:
        section(
            "Contexto de las identidades",
            "Resume si las identidades asociadas al audio cumplen los mínimos de muestras y en qué partición se utilizaron.",
        )

        if not identity_summary.empty:
            compact_table(
                identity_summary,
                [
                    "role_proxy",
                    "person_id",
                    "n_samples",
                    "n_audios",
                    "total_segments",
                    "total_duration_sec",
                    "eligible_verification",
                    "eligible_profile",
                ],
                n=10,
            )

        if not identity_split.empty:
            compact_table(
                identity_split,
                [
                    "person_id",
                    "role_proxy",
                    "split",
                ],
                n=10,
            )


STATIC_TAB_TITLES = [
    "00 Preparación",
    "01 Diarización",
    "02–03 Validación",
    "05 Transcripción",
    "06 Proxy",
    "07 Texto",
    "07B Prosodia",
    "08 Keywords",
    "09 Huella de voz",
]


STATIC_DASHBOARD_CSS = r"""
:root { --ink:#123047; --blue:#2F6BFF; --line:#E2E7EB; --muted:#68737D; --light:#F6F8FA; }
* { box-sizing:border-box; }
.tfm-static-page { width:100%; color:#18232B; font-family:Inter,Segoe UI,Arial,sans-serif; line-height:1.45; }
.tfm-static-header { background:linear-gradient(135deg,#123047,#2F6BFF); color:white; padding:22px 26px; border-radius:16px; box-shadow:0 8px 24px rgba(18,48,71,.14); }
.tfm-static-header .title { font-size:24px; font-weight:760; overflow-wrap:anywhere; }
.tfm-static-header .sub { margin-top:5px; opacity:.94; font-size:14px; }
.tfm-static-note { margin:11px 0 15px; color:var(--muted); font-size:13px; }
.tfm-radio-tab { position:absolute; opacity:0; pointer-events:none; }
.tfm-static-tabs { display:flex; flex-wrap:wrap; gap:7px; padding:10px 0 13px; }
.tfm-static-tab-label { border:1px solid var(--line); background:white; color:var(--ink); padding:9px 13px; border-radius:9px; font-weight:650; cursor:pointer; user-select:none; }
.tfm-static-tab-label:hover { border-color:#9AB4FF; }
.tfm-static-panels { width:100%; }
.tfm-static-panel { display:none; background:white; border:1px solid var(--line); border-radius:14px; padding:18px 20px 28px; min-height:220px; box-shadow:0 4px 16px rgba(18,48,71,.06); }
.tfm-static-page h1,.tfm-static-page h2,.tfm-static-page h3,.tfm-static-page h4 { color:var(--ink); }
.tfm-static-page h3 { margin:22px 0 3px; }
.tfm-static-page .tfm-card { border:1px solid var(--line); background:white; border-radius:12px; padding:13px 15px; margin:8px 0 12px; }
.tfm-static-page .tfm-kpi { display:inline-block; min-width:125px; padding:9px 12px; margin:3px 5px 4px 0; border:1px solid var(--line); border-radius:10px; vertical-align:top; background:#FAFBFC; }
.tfm-static-page .tfm-kpi .v { font-size:20px; font-weight:750; color:var(--ink); }
.tfm-static-page .tfm-kpi .l { font-size:11px; color:var(--muted); margin-top:2px; }
.tfm-static-page .tfm-note { font-size:12px; color:var(--muted); margin:3px 0 10px; }
.tfm-static-page .tfm-output { max-width:100%; overflow-x:auto; }
.tfm-static-page .tfm-figure { margin:14px 0 20px; text-align:center; }
.tfm-static-page .tfm-figure img { max-width:100%; height:auto; border-radius:8px; }
.tfm-static-page audio { width:min(760px,100%); margin:5px 0 16px; }
.tfm-static-page table { border-collapse:collapse; width:100%; margin:8px 0 18px; font-size:12px; }
.tfm-static-page th,.tfm-static-page td { border-bottom:1px solid #E8ECEF; padding:7px 8px; text-align:left; vertical-align:top; }
.tfm-static-page th { background:#F5F7F9; color:var(--ink); }
.tfm-static-page tr:nth-child(even) td { background:#FAFBFC; }
.tfm-static-page pre { white-space:pre-wrap; overflow-wrap:anywhere; background:#F6F8FA; border:1px solid var(--line); padding:10px; border-radius:8px; }
.tfm-static-page .tfm-error { border-left:4px solid #D1495B; background:#FFF5F6; padding:12px; border-radius:8px; margin:10px 0; }
.tfm-static-page .tfm-warning { color:#A85A00; background:#FFF9EF; padding:9px; border-radius:7px; }
.tfm-static-page .tfm-empty { color:var(--muted); padding:20px 4px; }
.tfm-static-page .tfm-log { color:var(--muted); margin:10px 0; font-size:12px; }
.tfm-static-page .tfm-export-card { display:flex; flex-wrap:wrap; align-items:center; justify-content:space-between; gap:10px; border:1px solid #CFE8DD; background:#F4FBF7; border-radius:11px; padding:10px 13px; margin:12px 0; }
.tfm-static-page .tfm-export-path { color:#456257; font-size:12px; overflow-wrap:anywhere; }
.tfm-static-page .tfm-open-export { border:0; background:#2A9D6F; color:white; border-radius:7px; padding:8px 13px; font-weight:650; cursor:pointer; }
@media (max-width:700px) { .tfm-static-panel{padding:12px} .tfm-static-header{padding:17px} }
"""


def _markdown_to_html(text):
    text = "" if text is None else str(text)
    if HAS_MISTUNE:
        try:
            return mistune.html(text)
        except Exception:
            pass

    escaped = html_lib.escape(text)
    lines = []
    for line in escaped.splitlines():
        if line.startswith("### "):
            lines.append(f"<h3>{line[4:]}</h3>")
        elif line.startswith("## "):
            lines.append(f"<h2>{line[3:]}</h2>")
        elif line.startswith("# "):
            lines.append(f"<h1>{line[2:]}</h1>")
        elif line.strip():
            lines.append(f"<p>{line}</p>")
    return "".join(lines)


def _figure_to_embedded_html(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=145, bbox_inches="tight", facecolor="white")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return (
        "<div class='tfm-figure'>"
        f"<img src='data:image/png;base64,{encoded}' alt='Visualización del pipeline'>"
        "</div>"
    )


def _display_object_to_html(obj):
    """Convierte los objetos emitidos por render_phase* en HTML autocontenido."""
    if obj is None:
        return ""
    if isinstance(obj, HTML):
        return str(obj.data)
    if isinstance(obj, Markdown):
        return _markdown_to_html(obj.data)
    if isinstance(obj, Audio):
        try:
            return obj._repr_html_() or ""
        except Exception as exc:
            return f"<div class='tfm-warning'>No se pudo incrustar el audio: {html_lib.escape(str(exc))}</div>"
    if isinstance(obj, pd.DataFrame):
        return obj.to_html(index=False, border=0, classes=["dataframe", "tfm-table"])

    repr_html = getattr(obj, "_repr_html_", None)
    if callable(repr_html):
        try:
            rendered = repr_html()
            if rendered:
                return str(rendered)
        except Exception:
            pass

    return f"<pre>{html_lib.escape(str(obj))}</pre>"


def _capture_renderer_as_html(renderer):
    """Ejecuta una fase y captura displays, figuras, tablas y audio sin crear widgets."""
    blocks = []
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()

    def captured_display(*objects, **kwargs):
        for obj in objects:
            rendered = _display_object_to_html(obj)
            if rendered:
                blocks.append(f"<div class='tfm-output'>{rendered}</div>")

    def captured_show(*args, **kwargs):
        for number in list(plt.get_fignums()):
            fig = plt.figure(number)
            try:
                blocks.append(_figure_to_embedded_html(fig))
            finally:
                plt.close(fig)

    old_display = globals().get("display")
    old_show = plt.show
    plt.close("all")

    try:
        globals()["display"] = captured_display
        plt.show = captured_show
        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
            renderer()
        captured_show()
    except Exception as exc:
        captured_show()
        detail = traceback.format_exc(limit=8)
        blocks.append(
            "<div class='tfm-error'><b>No se pudo renderizar completamente esta fase.</b> "
            f"{html_lib.escape(str(exc))}<details><summary>Detalle técnico</summary>"
            f"<pre>{html_lib.escape(detail)}</pre></details></div>"
        )
    finally:
        globals()["display"] = old_display
        plt.show = old_show
        plt.close("all")

    stdout_text = stdout_buffer.getvalue().strip()
    stderr_text = stderr_buffer.getvalue().strip()
    if stdout_text:
        blocks.append(
            f"<details class='tfm-log'><summary>Mensajes de ejecución</summary>"
            f"<pre>{html_lib.escape(stdout_text)}</pre></details>"
        )
    if stderr_text:
        blocks.append(
            f"<details class='tfm-log'><summary>Advertencias</summary>"
            f"<pre>{html_lib.escape(stderr_text)}</pre></details>"
        )

    return "\n".join(blocks) or "<div class='tfm-empty'>No hay contenido disponible para esta fase.</div>"


def _safe_export_name(audio_stem):
    stem = normalize_stem(audio_stem)
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("_")
    return clean or "audio"


def _build_css_tabs(uid, tab_contents):
    inputs = []
    labels = []
    panels = []
    rules = []

    for i, (title, content) in enumerate(zip(STATIC_TAB_TITLES, tab_contents)):
        radio_id = f"{uid}-radio-{i}"
        panel_class = f"{uid}-panel-{i}"
        checked = " checked" if i == 0 else ""
        inputs.append(
            f"<input class='tfm-radio-tab' type='radio' name='{uid}-tabs' "
            f"id='{radio_id}'{checked}>"
        )
        labels.append(
            f"<label class='tfm-static-tab-label' for='{radio_id}'>"
            f"{html_lib.escape(title)}</label>"
        )
        panels.append(
            f"<section class='tfm-static-panel {panel_class}'>{content}</section>"
        )
        rules.append(
            f"#{radio_id}:checked ~ .tfm-static-tabs label[for='{radio_id}']"
            "{color:white;border-color:#2F6BFF;background:#2F6BFF;}"
        )
        rules.append(
            f"#{radio_id}:checked ~ .tfm-static-panels .{panel_class}"
            "{display:block;}"
        )

    return "".join(inputs), "".join(labels), "".join(panels), "\n".join(rules)


def _export_open_card(path, uid):
    path = Path(path)
    try:
        rel_project = path.relative_to(PROJECT_DIR).as_posix()
    except Exception:
        rel_project = path.name
    size_mb = path.stat().st_size / (1024 ** 2)
    button_id = f"{uid}-open-export"
    rel_js = rel_project.replace("\\", "/").replace("'", "\\'")

    return f"""
    <div class='tfm-export-card'>
      <div>
        <div><b>Demo congelada y HTML guardado · {size_mb:.1f} MB</b></div>
        <div class='tfm-export-path'><code>{html_lib.escape(str(path))}</code></div>
      </div>
      <button id='{button_id}' class='tfm-open-export'>Abrir HTML en otra pestaña</button>
    </div>
    <script>
    (function() {{
      const button = document.getElementById('{button_id}');
      if (!button) return;
      button.addEventListener('click', function() {{
        const rel = '{rel_js}';
        const pathname = window.location.pathname;
        let base = '';
        let notebookPath = '';
        const markers = ['/lab/tree/', '/tree/', '/notebooks/'];
        for (const marker of markers) {{
          if (pathname.includes(marker)) {{
            const parts = pathname.split(marker);
            base = parts[0];
            notebookPath = decodeURIComponent(parts.slice(1).join(marker));
            break;
          }}
        }}
        const notebookDir = notebookPath.includes('/')
          ? notebookPath.split('/').slice(0,-1).join('/') : '';
        let target;
        if (notebookDir.endsWith('TFM_ProcesadoDeAudios')) {{
          target = notebookDir + '/' + rel;
        }} else if (notebookDir.includes('TFM_ProcesadoDeAudios/')) {{
          target = notebookDir.split('TFM_ProcesadoDeAudios/')[0] + 'TFM_ProcesadoDeAudios/' + rel;
        }} else {{
          target = (notebookDir ? notebookDir + '/' : '') + rel;
        }}
        const encoded = target.split('/').filter(Boolean).map(encodeURIComponent).join('/');
        window.open(base + '/files/' + encoded, '_blank');
      }});
    }})();
    </script>
    """


def generate_audio_html(audio_stem, show_paths=False):
    """Renderiza el audio una vez, guarda el HTML y devuelve la salida estática para el notebook."""
    stem = normalize_stem(audio_stem)
    tables, paths = load_audio_tables(stem)
    raw_path = resolve_audio(stem, "raw")
    clean_path = resolve_audio(stem, "clean")
    idx = selected_index_row(stem)

    n_rel = int(idx.get("n_relabelled", len(tables.get("changed", [])))) if len(idx) else len(tables.get("changed", []))
    n_seg = int(idx.get("n_segments", len(tables.get("final", [])))) if len(idx) else len(tables.get("final", []))
    ratio = float(idx.get("relabel_ratio", n_rel / n_seg if n_seg else 0)) if len(idx) else (n_rel / n_seg if n_seg else 0)

    renderers = [
        lambda: render_phase00(stem, tables, raw_path, clean_path, show_paths),
        lambda: render_phase01(stem, tables, clean_path),
        lambda: render_phase23(stem, tables),
        lambda: render_phase05(stem, clean_path),
        lambda: render_phase06(stem),
        lambda: render_phase07(stem),
        lambda: render_phase07b(stem),
        lambda: render_phase08(stem),
        lambda: render_phase09(stem),
    ]

    tab_contents = [_capture_renderer_as_html(renderer) for renderer in renderers]
    generated_at = datetime.now().strftime("%d/%m/%Y %H:%M")
    uid = f"tfm-demo-{_safe_export_name(stem)}-{datetime.now().strftime('%H%M%S%f')}"
    inputs, labels, panels, radio_rules = _build_css_tabs(uid, tab_contents)
    title_safe = html_lib.escape(stem)

    note_html = (
        f"<div class='tfm-static-note'>Generado el {generated_at}. "
        "Las pestañas y los reproductores de esta salida ya no dependen del kernel.</div>"
    )
    core_fragment = f"""
    <div id='{uid}' class='tfm-static-page'>
      <style>{STATIC_DASHBOARD_CSS}\n{radio_rules}</style>
      <header class='tfm-static-header'>
        <div class='title'>Demo end-to-end · {title_safe}</div>
        <div class='sub'>{n_rel} segmentos reetiquetados · {100 * ratio:.1f}% · {n_seg} segmentos finales</div>
      </header>
      {note_html}
      {inputs}
      <nav class='tfm-static-tabs'>{labels}</nav>
      <div class='tfm-static-panels'>{panels}</div>
    </div>
    """

    full_page = f"""<!doctype html>
    <html lang='es'>
    <head>
      <meta charset='utf-8'>
      <meta name='viewport' content='width=device-width, initial-scale=1'>
      <title>Demo end-to-end · {title_safe}</title>
      <style>
        body {{ margin:0; padding:24px; background:#F3F6F8; }}
        .tfm-static-page {{ width:min(1500px,96vw); margin:0 auto 40px; }}
      </style>
    </head>
    <body>{core_fragment}</body>
    </html>"""

    DEMO_HTML_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DEMO_HTML_DIR / f"demo_{_safe_export_name(stem)}.html"
    output_path.write_text(full_page, encoding="utf-8")

    notebook_fragment = core_fragment.replace(note_html, note_html + _export_open_card(output_path, uid), 1)
    return output_path, notebook_fragment


def selected_index_row(audio_stem):
    if RELABEL_INDEX.empty:
        return pd.Series(dtype=object)
    rows = RELABEL_INDEX[RELABEL_INDEX["audio_stem"] == normalize_stem(audio_stem)]
    return rows.iloc[0] if len(rows) else pd.Series(dtype=object)


def _render_demo_selector():
    """Construye el mismo selector interactivo del Notebook 10 original."""
    options = dropdown_options()

    if not options:
        display(Markdown(
            "No se encontraron archivos `*_final_segments.csv`. "
            "La demo se activará cuando los outputs del Notebook 01 "
            "estén disponibles en GCS."
        ))
        return {
            "options": [],
            "dropdown": None,
            "load_button": None,
            "dashboard_handle": None,
        }

    if not HAS_WIDGETS:
        display(Markdown(
            f"Se encontraron **{len(options):,} audios**, pero este entorno "
            "no tiene `ipywidgets`. Abre el notebook con el kernel del "
            "proyecto para usar el selector. "
            f"El primer audio del orden sería `{options[0][1]}`."
        ))
        return {
            "options": options,
            "dropdown": None,
            "load_button": None,
            "dashboard_handle": None,
        }

    dropdown = widgets.Dropdown(
        options=options,
        description="Audio:",
        layout=widgets.Layout(width="980px"),
        style={"description_width": "55px"},
    )
    load_button = widgets.Button(
        description="Cargar demo",
        button_style="primary",
        icon="play",
    )
    load_button.tooltip = (
        "Renderiza y congela la demo del audio seleccionado"
    )
    paths_checkbox = widgets.Checkbox(
        value=False,
        description="Mostrar rutas raw/clean",
    )

    display(HTML(
        "<div class='tfm-note'>El selector está ordenado de mayor a "
        "menor número de segmentos reetiquetados. Cada carga reemplaza "
        "la demo anterior, congela la nueva salida y guarda su HTML "
        "automáticamente.</div>"
    ))
    display(dropdown)
    display(widgets.HBox([load_button, paths_checkbox]))

    dashboard_handle = display(
        HTML(
            "<div class='tfm-card'><div class='tfm-note'>"
            "Selecciona un audio y pulsa <b>Cargar demo</b>."
            "</div></div>"
        ),
        display_id=True,
    )

    def _load(_):
        load_button.disabled = True
        dropdown.disabled = True
        paths_checkbox.disabled = True
        previous_description = load_button.description
        load_button.description = "Renderizando…"

        dashboard_handle.update(HTML(
            "<div class='tfm-card'><div class='tfm-note'>"
            "<b>Generando la demo estática.</b> Se están renderizando "
            "las nueve fases e incrustando las figuras, tablas y audios."
            "</div></div>"
        ))

        try:
            _, notebook_fragment = generate_audio_html(
                dropdown.value,
                show_paths=paths_checkbox.value,
            )
            dashboard_handle.update(HTML(notebook_fragment))
        except Exception:
            detail = traceback.format_exc(limit=10)
            dashboard_handle.update(HTML(
                "<div class='tfm-card'><div class='tfm-warn'>"
                "<b>No se pudo generar la demo.</b></div>"
                f"<pre>{html_lib.escape(detail)}</pre></div>"
            ))
        finally:
            load_button.description = previous_description
            load_button.disabled = False
            dropdown.disabled = False
            paths_checkbox.disabled = False

    load_button.on_click(_load)

    return {
        "options": options,
        "dropdown": dropdown,
        "load_button": load_button,
        "paths_checkbox": paths_checkbox,
        "dashboard_handle": dashboard_handle,
    }


def run_demo_end_to_end(gcs_client):
    """Restaura los inputs desde GCS y muestra la demo interactiva.

    Esta función no sube ningún archivo a Google Cloud Storage.
    """
    if gcs_client is None:
        raise ValueError("gcs_client no puede ser None.")

    global _GCS_CLIENT
    global RELABEL_INDEX

    _GCS_CLIENT = gcs_client

    _configure_notebook_style()
    _restore_initial_demo_inputs()

    RELABEL_INDEX = build_relabel_index()

    print("PROJECT_DIR:", PROJECT_DIR)
    print(
        "Widgets:",
        HAS_WIDGETS,
        "| librosa:",
        HAS_LIBROSA,
        "| sklearn:",
        HAS_SKLEARN,
        "| GCS:",
        True,
    )
    print("Audios indexados para la demo:", len(RELABEL_INDEX))

    if not RELABEL_INDEX.empty:
        display(
            RELABEL_INDEX
            .head(8)
            .reset_index(drop=True)
        )

    return _render_demo_selector()
