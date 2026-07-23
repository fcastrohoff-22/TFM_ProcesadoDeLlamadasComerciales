"""Dashboard global de resultados del TFM — versión ligera.

Este módulo reemplaza al Notebook 11: el notebook queda como orquestador y
llama una sola función pública, ``run_dashboard_resultados_globales``.

A diferencia de la versión anterior (que incrustaba decenas de miles de
registros a nivel de segmento y generaba un HTML de decenas de MB), esta
versión calcula en Python los **resultados agregados por fase** y solo incrusta
esos resúmenes. El resultado es un HTML autónomo y ligero, con:

- pestañas por fase (00 a 09), cada una con sus métricas finales y un gráfico
  representativo;
- un único filtro simple de corpus (Bajas vs Comerciales vs Todos);
- sin tablas a nivel de segmento.

La fase solo LEE outputs desde Google Cloud Storage y los restaura localmente
cuando hace falta. No sube, elimina ni modifica ningún objeto en GCS.
"""

from __future__ import annotations

import html as html_lib
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from IPython.display import HTML
from pandas.errors import EmptyDataError

from src import config as cfg
from src.storage_io import download_uri_to_local, join_gcs_uri

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURACIÓN LOCAL
# ============================================================

DATA_DIR = Path(cfg.DATA_DIR)
PROJECT_DIR = Path(getattr(cfg, "PROJECT_DIR", DATA_DIR.parent))
DASHBOARD_DIR = DATA_DIR / "global_results_dashboard"
HTML_DIR = DASHBOARD_DIR / "html_exports"
HTML_PATH = HTML_DIR / "dashboard_resultados_globales.html"

GCS_UNAV_ROOT = str(cfg.GCS_UNAV_ROOT)
GCS_UNAV_CSV_PREFIX = str(cfg.GCS_UNAV_CSV_PREFIX)

COLORS = {
    "navy": "#123047", "blue": "#2F6BFF", "green": "#2A9D6F",
    "orange": "#F28E2B", "red": "#D1495B", "purple": "#7B61A8",
    "teal": "#2A9D8F", "gray": "#7A8288", "line": "#DDE5EA", "bg": "#F4FBF7",
}

_GCS_CLIENT: Any = None


def _cfg_path(name: str, fallback: Path) -> Path:
    return Path(getattr(cfg, name, fallback))


EDA_DIR = _cfg_path("EDA_DIR", DATA_DIR / "eda")
CLEAN_RESULTS_DIR = _cfg_path("CLEAN_RESULTS_DIR", DATA_DIR / "clean_results")
DIARIZATION_DIR = _cfg_path("OUTPUT_DIR", DATA_DIR / "diarization_outputs")
FINAL_RELABEL_DIR = _cfg_path("FINAL_RELABEL_DIR", DIARIZATION_DIR / "final_relabel")
CONSOLIDATED_DIR = _cfg_path("CONSOLIDATED_DIR", DIARIZATION_DIR / "consolidated")
TRANSCRIPTION_DIR = _cfg_path("TRANSCRIPTION_ROOT", DATA_DIR / "transcription_outputs")
PROXY_DIR = _cfg_path("PROXY_GROUNDTRUTH_DIR", DATA_DIR / "proxy_groundtruth_outputs")
SENTIMENT_DIR = _cfg_path("SENTIMENT_DIR", DATA_DIR / "sentiment_outputs")
PROSODY_DIR = _cfg_path("PROSODY_DIR", DATA_DIR / "prosody_outputs")
FUSION_DIR = _cfg_path("SENTIMENT_FUSION_DIR", DATA_DIR / "sentiment_fusion_outputs")
KEYWORD_DIR = _cfg_path("KEYWORD_DIR", DATA_DIR / "keyword_outputs")
VOICEPRINT_DIR = _cfg_path("VOICEPRINT_DIR", DATA_DIR / "voiceprint_outputs")


def _first_path(*paths: Path) -> Path:
    for path in paths:
        if Path(path).exists():
            return Path(path)
    return Path(paths[0])


# Datasets necesarios para los agregados por fase (subconjunto del original).
DATASET_PATHS: dict[str, Path] = {
    # Fase 00
    "cleaning_results": _cfg_path(
        "CLEANING_RESULTS_PRIVATE_CSV",
        CLEAN_RESULTS_DIR / "audio_cleaning_results_private.csv",
    ),
    "silence_threshold_summary": _cfg_path(
        "SILENCE_THRESHOLD_SUMMARY_CSV", EDA_DIR / "silence_threshold_summary.csv"
    ),
    # Fases 01-04
    "diarization_summary": _cfg_path(
        "DIARIZATION_SUMMARY_CSV", DIARIZATION_DIR / "diarization_summary.csv"
    ),
    "scored_segments": _cfg_path(
        "DIARIZATION_ALL_SCORED_SEGMENTS_CSV",
        DIARIZATION_DIR / "diarization_all_scored_segments.csv",
    ),
    "anchor_segments": _cfg_path(
        "DIARIZATION_ALL_ANCHOR_SEGMENTS_CSV",
        DIARIZATION_DIR / "diarization_all_anchor_segments.csv",
    ),
    "relabel_summary": _cfg_path(
        "RELABEL_SUMMARY_CSV", FINAL_RELABEL_DIR / "relabel_summary.csv"
    ),
    "relabel_summary_by_audio": FINAL_RELABEL_DIR / "relabeling_summary_by_audio.csv",
    "final_merged_segments": _first_path(
        _cfg_path(
            "CONSOLIDATED_ALL_FINAL_MERGED_SEGMENTS_CSV",
            CONSOLIDATED_DIR / "all_final_merged_segments.csv",
        ),
        _cfg_path(
            "ALL_FINAL_MERGED_SEGMENTS_CSV",
            FINAL_RELABEL_DIR / "all_final_merged_segments.csv",
        ),
    ),
    # Fase 05
    "transcription_summary": _first_path(
        _cfg_path(
            "TRANSCRIPTION_FINAL_SUMMARY_CSV",
            TRANSCRIPTION_DIR / "06_transcription_summary_final.csv",
        ),
        _cfg_path(
            "TRANSCRIPTION_SUMMARY_CSV", TRANSCRIPTION_DIR / "transcription_summary.csv"
        ),
    ),
    # Fase 06
    "proxy_metrics": _cfg_path(
        "PROXY_TEXTUAL_METRICS_CSV", PROXY_DIR / "textual_proxy_metrics_summary.csv"
    ),
    "holdout_metrics": PROXY_DIR / "holdout_role_mapping_metrics.csv",
    "segment_proxy": _cfg_path(
        "PROXY_SEGMENT_LEVEL_CSV", PROXY_DIR / "segment_level_proxy_groundtruth.csv"
    ),
    # Fase 07A
    "sentiment_summary": _cfg_path(
        "SENTIMENT_SUMMARY_CSV", SENTIMENT_DIR / "sentiment_textual_summary_for_memory.csv"
    ),
    "sentiment_segments": _cfg_path(
        "SEGMENTS_WITH_SENTIMENT_CSV", SENTIMENT_DIR / "segments_with_sentiment_textual.csv"
    ),
    "role_sentiment": _cfg_path(
        "ROLE_SENTIMENT_CSV", SENTIMENT_DIR / "role_level_sentiment_textual.csv"
    ),
    # Fase 07B
    "prosody_summary": _cfg_path(
        "PROSODY_SUMMARY_CSV", PROSODY_DIR / "prosody_audio_affect_summary_for_memory.csv"
    ),
    "prosody_segments": _cfg_path(
        "SEGMENTS_PROSODY_CSV", PROSODY_DIR / "segments_with_audio_affect_prosody.csv"
    ),
    "role_prosody": _cfg_path(
        "ROLE_PROSODY_CSV", PROSODY_DIR / "role_level_audio_affect_prosody.csv"
    ),
    "ser_predictions": _cfg_path(
        "SER_PREDICTIONS_CSV", PROSODY_DIR / "ser_model_predictions.csv"
    ),
    # Fase 07C
    "fusion_summary": _cfg_path(
        "FUSION_SUMMARY_CSV", FUSION_DIR / "fusion_summary_for_memory.csv"
    ),
    "fusion_segments": _cfg_path(
        "FUSION_SEGMENTS_CSV", FUSION_DIR / "segments_audio_text_fusion.csv"
    ),
    "fusion_role_level": _cfg_path(
        "FUSION_ROLE_LEVEL_CSV", FUSION_DIR / "role_level_audio_text_fusion.csv"
    ),
    "fusion_correlations": _cfg_path(
        "FUSION_CORRELATIONS_CSV", FUSION_DIR / "correlations_audio_text.csv"
    ),
    "fusion_disagreement": _cfg_path(
        "FUSION_DISAGREEMENT_CSV",
        FUSION_DIR / "disagreement_masked_frustration_segments.csv",
    ),
    # Fase 08
    "keyword_segments": _cfg_path(
        "SEGMENTS_WITH_KEYWORDS_CSV", KEYWORD_DIR / "segments_with_keywords.csv"
    ),
    "keyword_calls": _cfg_path(
        "CALL_LEVEL_KEYWORDS_CSV", KEYWORD_DIR / "call_level_keywords.csv"
    ),
    # Fase 09
    "voiceprint_verification_metrics": _cfg_path(
        "METRICS_CSV", VOICEPRINT_DIR / "voiceprint_metrics_summary.csv"
    ),
    "voiceprint_final_summary": _cfg_path(
        "VOICEPRINT_FINAL_SUMMARY_CSV",
        VOICEPRINT_DIR / "voiceprint_final_summary_for_memory.csv",
    ),
}

PHASE00_DATASETS = {"cleaning_results", "silence_threshold_summary"}


# ============================================================
# RESTAURACIÓN DESDE GCS (solo lectura)
# ============================================================

def _gcs_uri_for_dataset(name: str, local_path: Path) -> str:
    if name in PHASE00_DATASETS:
        return join_gcs_uri(GCS_UNAV_CSV_PREFIX, Path(local_path).name)
    relative_path = Path(local_path).relative_to(DATA_DIR).as_posix()
    return join_gcs_uri(GCS_UNAV_ROOT, relative_path)


def _restore_dataset(name: str, local_path: Path, force: bool = False) -> dict[str, Any]:
    local_path = Path(local_path)
    result = {"dataset": name, "available": False, "downloaded": False, "error": ""}
    if _GCS_CLIENT is None:
        result["available"] = local_path.exists() and local_path.stat().st_size > 0
        return result
    try:
        result["downloaded"] = bool(
            download_uri_to_local(
                source_uri=_gcs_uri_for_dataset(name, local_path),
                local_path=local_path, gcs_client=_GCS_CLIENT, force=force,
            )
        )
    except Exception as exc:
        result["error"] = str(exc)
    result["available"] = local_path.exists() and local_path.stat().st_size > 0
    return result


def _restore_inputs(force: bool = False) -> pd.DataFrame:
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
    HTML_DIR.mkdir(parents=True, exist_ok=True)
    rows = [_restore_dataset(n, p, force=force) for n, p in DATASET_PATHS.items()]
    return pd.DataFrame(rows)


def _read_csv_optional(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except (EmptyDataError, Exception):
        return pd.DataFrame()


def _load_datasets() -> dict[str, pd.DataFrame]:
    return {name: _read_csv_optional(path) for name, path in DATASET_PATHS.items()}


# ============================================================
# UTILIDADES DE CORPUS Y COLUMNAS
# ============================================================

def _norm_corpus(value: Any) -> str:
    """Clasifica un audio en Bajas / Comerciales / No identificado."""
    text = str(value).strip().lower()
    if "baja" in text:
        return "Bajas"
    if "comercial" in text or "venta" in text or text.startswith("raw_") and "baja" not in text:
        return "Comerciales"
    if text in {"raw", "general", "comerciales"}:
        return "Comerciales"
    return "No identificado"


def _audio_col(df: pd.DataFrame):
    for c in [
        "audio_file", "audio_file_norm", "audio_stem_norm", "audio",
        "audio_id", "audio_stem", "filename", "file",
    ]:
        if c in df.columns:
            return c
    return None


def _corpus_series(df: pd.DataFrame) -> pd.Series:
    """Serie de corpus por fila (para el filtro simple)."""
    if df.empty:
        return pd.Series(dtype="object")
    for c in ["corpus", "source_corpus", "source_dataset", "dataset", "source"]:
        if c in df.columns:
            return df[c].apply(_norm_corpus)
    ac = _audio_col(df)
    if ac:
        return df[ac].apply(_norm_corpus)
    return pd.Series(["No identificado"] * len(df), index=df.index)


def _filter_corpus(df: pd.DataFrame, corpus: str) -> pd.DataFrame:
    """Filtra un DataFrame por corpus (Todos = sin filtrar)."""
    if df.empty or corpus == "Todos":
        return df
    return df[_corpus_series(df).eq(corpus)]


def _num(value, default=np.nan):
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _metric(df, name, metric_cols=("metric", "Metric", "metrica", "métrica"),
            value_cols=("value", "Value", "valor")):
    """Valor de una métrica en una tabla larga metric/value (o metrica/valor)."""
    if df.empty:
        return np.nan
    mc = next((c for c in metric_cols if c in df.columns), None)
    vc = next((c for c in value_cols if c in df.columns), None)
    if mc is None or vc is None:
        return np.nan
    hit = df[df[mc].astype(str).str.lower() == name.lower()]
    return _num(hit.iloc[0][vc]) if len(hit) else np.nan


def _parse_theme_cell(value) -> list:
    """
    Extrae los temas de una celda que puede venir como:
    - repr de lista de Python: "['incidencia_tecnica', 'baja_cancelacion']"
    - lista real ya parseada (si pandas la leyó como objeto)
    - texto separado por | o ,
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(v).strip() for v in value if str(v).strip()]

    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []

    if text.startswith("[") and text.endswith("]"):
        import ast
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple, set)):
                return [str(v).strip() for v in parsed if str(v).strip()]
        except (ValueError, SyntaxError):
            # repr malformado: extraer tokens entre comillas como respaldo
            tokens = re.findall(r"['\"]([^'\"]+)['\"]", text)
            if tokens:
                return [t.strip() for t in tokens if t.strip()]

    return [t.strip() for t in re.split(r"[|,]", text) if t.strip()]


def _top_themes_from_column(series: pd.Series, top_n: int = 6) -> list:
    """Cuenta los temas más frecuentes de una columna con listas de temas por fila."""
    counts: dict[str, int] = {}
    for value in series.dropna():
        for theme in _parse_theme_cell(value):
            counts[theme] = counts.get(theme, 0) + 1
    top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    return [[k, v] for k, v in top]


# ============================================================
# CÁLCULO DE AGREGADOS POR FASE (respetando el filtro de corpus)
# ============================================================

def compute_phase_aggregates(datasets: dict[str, pd.DataFrame], corpus: str) -> dict[str, Any]:
    """
    Calcula los resultados finales agregados de cada fase para un corpus dado.

    Devuelve un dict por fase con: kpis (lista de [label, valor]) y chart
    (dict con tipo y datos ya agregados, listos para incrustar ligero).
    """
    agg: dict[str, Any] = {}

    def ff(df):  # filtro corpus abreviado
        return _filter_corpus(df, corpus)

    # ---- Fase 00: limpieza ----
    clean = ff(datasets.get("cleaning_results", pd.DataFrame()))
    dur_orig = dur_clean = np.nan
    if not clean.empty:
        oc = next((c for c in ["duration_original_sec", "original_duration_sec", "duration_before"] if c in clean.columns), None)
        cc = next((c for c in ["duration_clean_sec", "clean_duration_sec", "duration_after"] if c in clean.columns), None)
        if oc:
            dur_orig = _num(pd.to_numeric(clean[oc], errors="coerce").mean())
        if cc:
            dur_clean = _num(pd.to_numeric(clean[cc], errors="coerce").mean())
    reduction_pct = (
        100 * (dur_orig - dur_clean) / dur_orig
        if not np.isnan(_num(dur_orig)) and not np.isnan(_num(dur_clean)) and dur_orig else np.nan
    )
    agg["00"] = {
        "kpis": [
            ["Audios en el corpus", len(clean) if not clean.empty else np.nan, "int"],
            ["Duración media original (s)", dur_orig, "float1"],
            ["Duración media limpia (s)", dur_clean, "float1"],
        ],
        "charts": [
            {"type": "bars", "title": "Duración media (s)",
             "items": [["Original", dur_orig], ["Limpio", dur_clean]], "unit": "s"},
            {"type": "donut", "title": "Reducción media de duración", "pct": reduction_pct,
             "label": "reducción", "color": COLORS["blue"]},
        ],
    }

    # ---- Fases 01-04: diarización ----
    scored = ff(datasets.get("scored_segments", pd.DataFrame()))
    anchors = ff(datasets.get("anchor_segments", pd.DataFrame()))
    merged = ff(datasets.get("final_merged_segments", pd.DataFrame()))
    relabel = datasets.get("relabel_summary", pd.DataFrame())
    relabel_by_audio = ff(datasets.get("relabel_summary_by_audio", pd.DataFrame()))

    n_final = len(merged) if not merged.empty else np.nan
    n_audios_final = merged[_audio_col(merged)].nunique() if not merged.empty and _audio_col(merged) else np.nan

    # Fuente principal: relabeling_summary_by_audio.csv (columnas reales
    # confirmadas: audio_base/audio_file, n_segments, n_relabelled).
    n_relabeled = n_relabel_base = np.nan
    if not relabel_by_audio.empty:
        seg_col = next((c for c in ["n_segments", "n_final_segments"] if c in relabel_by_audio.columns), None)
        rel_col = next((c for c in ["n_relabelled", "n_relabeled", "n_changed_segments"] if c in relabel_by_audio.columns), None)
        if seg_col and rel_col:
            n_relabel_base = _num(pd.to_numeric(relabel_by_audio[seg_col], errors="coerce").sum())
            n_relabeled = _num(pd.to_numeric(relabel_by_audio[rel_col], errors="coerce").sum())

    if np.isnan(n_relabeled) and not merged.empty:  # fallback: columna booleana por segmento
        ch = next((c for c in ["changed", "relabeled", "was_relabeled"] if c in merged.columns), None)
        if ch:
            n_relabeled = int(merged[ch].astype(str).str.lower().isin(["true", "1", "yes"]).sum())
            n_relabel_base = n_final

    if np.isnan(n_relabeled):  # último fallback: métrica agregada en relabel_summary
        pct_relabel = _metric(relabel, "pct_relabeled")
        pct_relabel = pct_relabel * 100 if not np.isnan(pct_relabel) and pct_relabel <= 1 else pct_relabel
    else:
        pct_relabel = (
            100 * n_relabeled / n_relabel_base
            if n_relabel_base and not np.isnan(n_relabel_base) else np.nan
        )
    agg["01"] = {
        "kpis": [
            ["Segmentos puntuados", len(scored) if not scored.empty else np.nan, "int"],
            ["Anchors", len(anchors) if not anchors.empty else np.nan, "int"],
            ["Segmentos finales", n_final, "int"],
            ["Audios consolidados", n_audios_final, "int"],
            ["Reetiquetados", pct_relabel, "pct1"],
        ],
        "charts": [
            {"type": "funnel", "title": "Embudo de segmentos",
             "items": [["Puntuados", len(scored) if not scored.empty else 0],
                       ["Anchors", len(anchors) if not anchors.empty else 0],
                       ["Finales", n_final if not np.isnan(_num(n_final)) else 0]]},
            {"type": "donut", "title": "Segmentos reetiquetados", "pct": pct_relabel,
             "label": "reetiquetados", "color": COLORS["orange"]},
        ],
    }

    # ---- Fase 05: transcripción ----
    trans = ff(datasets.get("transcription_summary", pd.DataFrame()))
    cov = np.nan
    with_text = without_text = np.nan
    if not trans.empty and "n_segments_with_text" in trans.columns and "n_diarized_segments" in trans.columns:
        with_text = int(trans["n_segments_with_text"].sum())
        total = int(trans["n_diarized_segments"].sum())
        without_text = total - with_text
        cov = 100 * with_text / max(1, total)
    agg["05"] = {
        "kpis": [
            ["Cobertura textual", cov, "pct2"],
            ["Segmentos con texto", with_text, "int"],
            ["Segmentos sin texto", without_text, "int"],
        ],
        "charts": [
            {"type": "donut", "title": "Cobertura textual", "pct": cov,
             "label": "con texto", "color": COLORS["blue"]},
            {"type": "bars", "title": "Segmentos con vs. sin texto",
             "items": [["Con texto", with_text], ["Sin texto", without_text]], "unit": ""},
        ],
    }

    # ---- Fase 06: proxy de roles ----
    holdout = datasets.get("holdout_metrics", pd.DataFrame())
    seg_proxy = ff(datasets.get("segment_proxy", pd.DataFrame()))
    acc = _metric(holdout, "holdout_accuracy")
    bacc = _metric(holdout, "holdout_balanced_accuracy")
    acc = acc * 100 if not np.isnan(acc) and acc <= 1 else acc
    bacc = bacc * 100 if not np.isnan(bacc) and bacc <= 1 else bacc
    n_agent = n_client = np.nan
    if not seg_proxy.empty and "official_role_proxy" in seg_proxy.columns:
        rp = seg_proxy["official_role_proxy"].astype(str).str.upper()
        n_agent = int((rp == "AGENT").sum())
        n_client = int((rp == "CLIENT").sum())
    agg["06"] = {
        "kpis": [
            ["Accuracy holdout", acc, "pct2"],
            ["Balanced accuracy", bacc, "pct2"],
            ["Segmentos AGENT", n_agent, "int"],
            ["Segmentos CLIENT", n_client, "int"],
        ],
        "charts": [
            {"type": "bars", "title": "Segmentos por rol proxy",
             "items": [["AGENT", n_agent], ["CLIENT", n_client]], "unit": ""},
            {"type": "donut", "title": "Balanced accuracy holdout", "pct": bacc,
             "label": "balanced acc.", "color": COLORS["green"]},
        ],
    }

    # ---- Fase 07A: sentimiento textual ----
    # Nombres reales confirmados en sentiment_textual_summary_for_memory.csv:
    # pct_negative_segments / pct_neutral_segments / pct_positive_segments,
    # AGENT_avg_sentiment / CLIENT_avg_sentiment, segments_analyzed_sentiment.
    sent_seg = ff(datasets.get("sentiment_segments", pd.DataFrame()))
    sent = datasets.get("sentiment_summary", pd.DataFrame())
    role_sent = datasets.get("role_sentiment", pd.DataFrame())

    pct_neg = pct_neu = pct_pos = np.nan
    agent_s = client_s = np.nan
    n_sent_seg = np.nan

    if not sent_seg.empty and "sentiment_label" in sent_seg.columns:
        n_sent_seg = len(sent_seg)
        lab = sent_seg["sentiment_label"].astype(str).str.lower()
        pct_neg = 100 * (lab == "negative").mean()
        pct_neu = 100 * (lab == "neutral").mean()
        pct_pos = 100 * (lab == "positive").mean()
        rc = next((c for c in ["role_proxy", "official_role_proxy", "role"] if c in sent_seg.columns), None)
        nc = "sentiment_numeric" if "sentiment_numeric" in sent_seg.columns else None
        if rc and nc:
            grp = sent_seg.groupby(sent_seg[rc].astype(str).str.upper())[nc].mean()
            agent_s = _num(grp.get("AGENT"))
            client_s = _num(grp.get("CLIENT"))

    if np.isnan(pct_neg):  # Fallback: summary con nombres reales del pipeline
        pct_neg = _metric(sent, "pct_negative_segments")
        pct_neu = _metric(sent, "pct_neutral_segments")
        pct_pos = _metric(sent, "pct_positive_segments")
        if np.isnan(pct_neg):  # Fallback adicional: nombre genérico corto
            pct_neg = _metric(sent, "pct_negative")
            pct_neu = _metric(sent, "pct_neutral")
            pct_pos = _metric(sent, "pct_positive")
        pct_neg = pct_neg * 100 if not np.isnan(pct_neg) and pct_neg <= 1 else pct_neg
        pct_neu = pct_neu * 100 if not np.isnan(pct_neu) and pct_neu <= 1 else pct_neu
        pct_pos = pct_pos * 100 if not np.isnan(pct_pos) and pct_pos <= 1 else pct_pos
        n_sent_seg = _metric(sent, "segments_analyzed_sentiment")

    if np.isnan(_num(agent_s)):
        agent_s = _metric(sent, "AGENT_avg_sentiment")
        client_s = _metric(sent, "CLIENT_avg_sentiment")
        if np.isnan(_num(agent_s)) and not role_sent.empty:
            rc = next((c for c in ["role_proxy", "official_role_proxy", "role"] if c in role_sent.columns), None)
            mc = next((c for c in ["avg_sentiment", "mean_sentiment"] if c in role_sent.columns), None)
            if rc and mc:
                for _, r in role_sent.iterrows():
                    if str(r[rc]).upper() == "AGENT":
                        agent_s = _num(r[mc])
                    elif str(r[rc]).upper() == "CLIENT":
                        client_s = _num(r[mc])

    agg["07A"] = {
        "kpis": [
            ["Segmentos analizados", n_sent_seg, "int"],
            ["Neutrales", pct_neu, "pct1"],
            ["Negativos", pct_neg, "pct1"],
            ["Sent. medio AGENT", agent_s, "signed3"],
            ["Sent. medio CLIENT", client_s, "signed3"],
        ],
        "charts": [
            {"type": "stacked", "title": "Distribución de sentimiento",
             "items": [["Negativo", pct_neg, COLORS["red"]],
                       ["Neutral", pct_neu, COLORS["gray"]],
                       ["Positivo", pct_pos, COLORS["green"]]]},
            {"type": "diverging", "title": "Sentimiento medio por rol",
             "items": [["Agente", agent_s], ["Cliente", client_s]]},
        ],
    }

    # ---- Fase 07B: afecto de audio ----
    # Nombres reales confirmados en segments_with_audio_affect_prosody.csv:
    # audio_stem_norm/audio_file_norm, role_proxy_for_prosody, arousal_proxy_score,
    # tension_proxy_score, intensity_proxy_score, prosodic_state_proxy, ser_pred_label.
    pros_seg = ff(datasets.get("prosody_segments", pd.DataFrame()))
    ser = ff(datasets.get("ser_predictions", pd.DataFrame()))
    role_pros = datasets.get("role_prosody", pd.DataFrame())

    ser_items, state_items = [], []
    arousal_m = tension_m = intensity_m = np.nan
    n_pros = np.nan
    role_col_pros = None

    if not pros_seg.empty:
        n_pros = len(pros_seg)
        if "arousal_proxy_score" in pros_seg.columns:
            arousal_m = _num(pd.to_numeric(pros_seg["arousal_proxy_score"], errors="coerce").mean())
        if "tension_proxy_score" in pros_seg.columns:
            tension_m = _num(pd.to_numeric(pros_seg["tension_proxy_score"], errors="coerce").mean())
        if "intensity_proxy_score" in pros_seg.columns:
            intensity_m = _num(pd.to_numeric(pros_seg["intensity_proxy_score"], errors="coerce").mean())
        state_col = next((c for c in ["prosodic_state_proxy", "ser_pred_label"] if c in pros_seg.columns), None)
        if state_col:
            top = pros_seg[state_col].astype(str).value_counts().head(6)
            state_items = [[k, int(v)] for k, v in top.items() if k.lower() != "nan"]
        role_col_pros = next(
            (c for c in ["role_proxy_for_prosody", "role_proxy", "role"] if c in pros_seg.columns), None
        )

    ser_src = ser if not ser.empty else pros_seg
    if not ser_src.empty:
        lc = next(
            (c for c in ["ser_pred_label", "ser_label", "emotion", "label", "predicted_emotion"]
             if c in ser_src.columns), None,
        )
        if lc:
            top = ser_src[lc].astype(str).value_counts().head(6)
            ser_items = [[k, int(v)] for k, v in top.items() if k.lower() != "nan"]

    # Segundo gráfico: activación/tensión por rol, desde tabla agregada por rol
    # (role_level_audio_affect_prosody.csv) si existe; si no, desde los segmentos.
    role_arousal_items = []
    if not role_pros.empty:
        rc = next((c for c in ["role_proxy_for_prosody", "role_proxy", "role"] if c in role_pros.columns), None)
        ac = next((c for c in ["arousal_proxy_score_mean", "arousal_proxy_score"] if c in role_pros.columns), None)
        if rc and ac:
            for _, r in role_pros.iterrows():
                role_arousal_items.append([str(r[rc]).upper(), _num(r[ac])])
    elif not pros_seg.empty and role_col_pros and "arousal_proxy_score" in pros_seg.columns:
        grp = pros_seg.groupby(pros_seg[role_col_pros].astype(str).str.upper())["arousal_proxy_score"].mean()
        role_arousal_items = [[k, _num(v)] for k, v in grp.items()]

    agg["07B"] = {
        "kpis": [
            ["Segmentos con afecto", n_pros, "int"],
            ["Activación media", arousal_m, "float3"],
            ["Tensión media", tension_m, "float3"],
            ["Intensidad media", intensity_m, "float3"],
        ],
        "charts": [
            {"type": "bars", "title": "Etiquetas SER más frecuentes",
             "items": ser_items or [["Sin datos", 0]], "unit": ""},
            {"type": "bars", "title": "Activación media por rol",
             "items": role_arousal_items or [["Sin datos", 0]], "unit": "", "color": COLORS["orange"]},
        ],
    }

    # ---- Fase 07C: fusión audio-texto ----
    # Nombres reales confirmados en fusion_summary_for_memory.csv: columnas en
    # ESPAÑOL "metrica"/"valor" (no "metric"/"value"). Métricas: segmentos_totales,
    # segmentos_comparables, pearson_<indicador>, pct_frustracion_enmascarada.
    # correlations_audio_text.csv usa columnas score_audio/pearson_r/spearman_r.
    fusion = datasets.get("fusion_summary", pd.DataFrame())
    fus_seg = ff(datasets.get("fusion_segments", pd.DataFrame()))
    fus_role = datasets.get("fusion_role_level", pd.DataFrame())
    fus_disagree = ff(datasets.get("fusion_disagreement", pd.DataFrame()))
    corr = datasets.get("fusion_correlations", pd.DataFrame())

    total_seg = _metric(fusion, "segmentos_totales")
    comparable_seg = _metric(fusion, "segmentos_comparables")
    if np.isnan(total_seg) and not fus_seg.empty:
        total_seg = len(fus_seg)
        if {"has_audio", "has_text"}.issubset(fus_seg.columns):
            comparable_seg = int((fus_seg["has_audio"] & fus_seg["has_text"]).sum())
    cov_fus = (100 * comparable_seg / total_seg) if (not np.isnan(total_seg) and total_seg) else np.nan

    n_disc = len(fus_disagree) if not fus_disagree.empty else np.nan
    if np.isnan(n_disc):
        pct_disc = _metric(fusion, "pct_frustracion_enmascarada")
        if not np.isnan(pct_disc) and not np.isnan(comparable_seg):
            n_disc = round(pct_disc / 100 * comparable_seg)

    corr_items = []
    corr_act = np.nan
    if not corr.empty:
        score_col = next((c for c in ["score_audio", "indicator", "variable"] if c in corr.columns), None)
        r_col = next((c for c in ["pearson_r", "correlation", "corr", "value"] if c in corr.columns), None)
        if score_col and r_col:
            for _, r in corr.iterrows():
                corr_items.append([str(r[score_col]).replace("_proxy_score", "").replace("ser_neg_prob", "ser_neg"), _num(r[r_col])])
            hit = corr[corr[score_col].astype(str).str.contains("arousal", case=False, na=False)]
            if len(hit):
                corr_act = _num(hit.iloc[0][r_col])
    if not corr_items:
        corr_act = _metric(fusion, "pearson_arousal_proxy_score")

    agg["07C"] = {
        "kpis": [
            ["Cobertura audio–texto", cov_fus, "pct1"],
            ["Discordancia voz≠texto", n_disc, "int"],
            ["Corr. texto·activación", corr_act, "signed3"],
        ],
        "charts": [
            {"type": "donut", "title": "Cobertura comparable", "pct": cov_fus,
             "label": "audio–texto", "color": COLORS["teal"]},
            {"type": "diverging", "title": "Correlación sentimiento vs. indicadores de audio",
             "items": corr_items or [["Sin datos", 0]]},
        ],
    }

    # ---- Fase 08: keyword spotting ----
    kw_seg = ff(datasets.get("keyword_segments", pd.DataFrame()))
    pct_kw = np.nan
    theme_items = []
    if not kw_seg.empty:
        if "has_critical_keyword" in kw_seg.columns:
            pct_kw = 100 * _num(kw_seg["has_critical_keyword"].astype(str).str.lower().isin(["true", "1", "yes"]).mean())
        tc = next((c for c in [
            "critical_themes_detected", "critical_theme", "theme", "critical_themes",
        ] if c in kw_seg.columns), None)
        if tc:
            theme_items = _top_themes_from_column(kw_seg[tc], top_n=6)
    agg["08"] = {
        "kpis": [
            ["Segmentos con keyword", pct_kw, "pct1"],
            ["Temas críticos detectados", len(theme_items) if theme_items else np.nan, "int"],
        ],
        "charts": [
            {"type": "bars", "title": "Temas críticos más frecuentes",
             "items": theme_items or [["Sin datos", 0]], "unit": ""},
            {"type": "donut", "title": "Segmentos con keyword crítica", "pct": pct_kw,
             "label": "con keyword", "color": COLORS["purple"]},
        ],
    }

    # ---- Fase 09: huella de voz ----
    # Columna real confirmada: "dataset" (no "set"), valores como
    # calibration_agents / test_agents_unseen / test_clients_repeated.
    vp = datasets.get("voiceprint_verification_metrics", pd.DataFrame())
    vp_items = []
    acc_items = []
    auc_main = eer_main = np.nan
    if not vp.empty:
        sc = next((c for c in ["dataset", "set", "conjunto", "subset"] if c in vp.columns), None)
        ac = next((c for c in ["auc", "AUC"] if c in vp.columns), None)
        accc = next((c for c in ["accuracy", "acc", "Accuracy"] if c in vp.columns), None)
        eerc = next((c for c in ["eer", "EER"] if c in vp.columns), None)
        if sc and ac:
            for _, r in vp.iterrows():
                vp_items.append([str(r[sc]), _num(r[ac])])
            hit = vp[vp[sc].astype(str).str.contains("no vistos|unseen|held", case=False, na=False)]
            auc_main = _num(hit.iloc[0][ac]) if len(hit) else _num(vp[ac].max())
            if eerc and len(hit):
                eer_main = _num(hit.iloc[0][eerc])
        if sc and accc:
            for _, r in vp.iterrows():
                acc_items.append([str(r[sc]), 100 * _num(r[accc])])
    agg["09"] = {
        "kpis": [
            ["AUC agentes no vistos", auc_main, "float3"],
            ["EER agentes no vistos", eer_main, "float3"],
        ],
        "charts": [
            {"type": "bars", "title": "AUC por conjunto",
             "items": vp_items or [["Sin datos", 0]], "unit": "", "vmax": 1.0},
            {"type": "bars", "title": "Accuracy por conjunto (%)",
             "items": acc_items or [["Sin datos", 0]], "unit": "%", "vmax": 100, "color": COLORS["green"]},
        ],
    }

    return agg


# ============================================================
# FORMATO Y GRÁFICOS SVG (ligeros)
# ============================================================

def _fmt(value, kind="int"):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    if kind == "int":
        return f"{int(round(value)):,}".replace(",", ".")
    if kind == "pct1":
        return f"{value:.1f}%".replace(".", ",")
    if kind == "pct2":
        return f"{value:.2f}%".replace(".", ",")
    if kind == "float1":
        return f"{value:.1f}".replace(".", ",")
    if kind == "float3":
        return f"{value:.3f}".replace(".", ",")
    if kind == "signed3":
        return f"{value:+.3f}".replace(".", ",")
    return str(value)


def _svg_bars(items, unit="", vmax=None, color=None, height=None):
    color = color or COLORS["teal"]
    items = [(str(l), _num(v, 0.0)) for l, v in items]
    if not items or all(v == 0 for _, v in items):
        return "<div style='color:#7A8288;font-size:13px;'>Sin datos.</div>"
    vmax = vmax or max(v for _, v in items) or 1
    row_h = 34
    height = height or (len(items) * row_h + 16)
    width, label_w, pad = 460, 150, 55
    bar_w = width - label_w - pad
    rows = []
    for i, (label, value) in enumerate(items):
        y = 8 + i * row_h
        w = max(2, bar_w * (value / vmax))
        vtxt = f"{value:,.0f}".replace(",", ".") if value >= 100 else f"{value:.2f}".replace(".", ",")
        rows.append(
            f"<text x='0' y='{y + 18:.0f}' font-size='12' fill='#123047'>{html_lib.escape(label[:22])}</text>"
            f"<rect x='{label_w}' y='{y:.0f}' width='{w:.0f}' height='20' rx='4' fill='{color}'/>"
            f"<text x='{label_w + w + 6:.0f}' y='{y + 15:.0f}' font-size='11' fill='#456257'>{vtxt}{unit}</text>"
        )
    return f"<svg viewBox='0 0 {width} {height}' width='100%' style='max-width:480px'>" + "".join(rows) + "</svg>"


def _svg_donut(pct, label, color):
    if pct is None or (isinstance(pct, float) and np.isnan(pct)):
        return "<div style='color:#7A8288;font-size:13px;'>Sin datos.</div>"
    import math
    r, cx, cy = 58, 120, 72
    circ = 2 * math.pi * r
    frac = max(0.0, min(1.0, pct / 100))
    disp = f"{pct:.1f}%".replace(".", ",")
    return (
        f"<svg viewBox='0 0 240 148' width='100%' style='max-width:260px'>"
        f"<circle cx='{cx}' cy='{cy}' r='{r}' fill='none' stroke='#E4F0EA' stroke-width='17'/>"
        f"<circle cx='{cx}' cy='{cy}' r='{r}' fill='none' stroke='{color}' stroke-width='17' "
        f"stroke-dasharray='{circ*frac:.1f} {circ:.1f}' stroke-linecap='round' transform='rotate(-90 {cx} {cy})'/>"
        f"<text x='{cx}' y='{cy+2}' font-size='26' font-weight='750' fill='#123047' text-anchor='middle'>{disp}</text>"
        f"<text x='{cx}' y='{cy+22}' font-size='11' fill='#456257' text-anchor='middle'>{html_lib.escape(label)}</text>"
        f"</svg>"
    )


def _svg_stacked(items):
    """Barra apilada horizontal 100% (para distribución de sentimiento)."""
    items = [(str(l), _num(v, 0.0), c) for l, v, c in items]
    total = sum(v for _, v, _ in items) or 1
    width, h = 460, 46
    x = 0
    parts, legend = [], []
    for label, value, color in items:
        w = width * (value / total)
        parts.append(f"<rect x='{x:.0f}' y='0' width='{w:.0f}' height='{h}' fill='{color}'/>")
        if w > 40:
            parts.append(f"<text x='{x + w/2:.0f}' y='{h/2 + 4:.0f}' font-size='12' fill='white' text-anchor='middle'>{value:.0f}%</text>")
        legend.append(f"<span style='font-size:12px;color:#456257;margin-right:14px;'><span style='display:inline-block;width:10px;height:10px;background:{color};border-radius:2px;margin-right:4px;'></span>{html_lib.escape(label)}</span>")
        x += w
    return (
        f"<svg viewBox='0 0 {width} {h}' width='100%' style='max-width:480px'>" + "".join(parts) + "</svg>"
        f"<div style='margin-top:8px;'>{''.join(legend)}</div>"
    )


def _svg_funnel(items):
    """Embudo simple (segmentos puntuados → anchors → finales)."""
    items = [(str(l), _num(v, 0.0)) for l, v in items]
    if not items or all(v == 0 for _, v in items):
        return "<div style='color:#7A8288;font-size:13px;'>Sin datos.</div>"
    vmax = max(v for _, v in items) or 1
    width, row_h = 460, 44
    rows = []
    for i, (label, value) in enumerate(items):
        w = max(20, (width - 120) * (value / vmax))
        x = (width - 120 - w) / 2
        y = 6 + i * row_h
        rows.append(
            f"<rect x='{x:.0f}' y='{y:.0f}' width='{w:.0f}' height='30' rx='4' fill='{COLORS['teal']}' opacity='{1 - i*0.18:.2f}'/>"
            f"<text x='{width-110}' y='{y + 20:.0f}' font-size='12' fill='#123047'>{html_lib.escape(label)}: {value:,.0f}</text>".replace(",", ".")
        )
    return f"<svg viewBox='0 0 {width} {len(items)*row_h + 12}' width='100%' style='max-width:480px'>" + "".join(rows) + "</svg>"


def _svg_diverging(items):
    """Barras divergentes centradas en 0 (para correlaciones y medias con signo)."""
    items = [(str(l), _num(v, 0.0)) for l, v in items]
    if not items or all(v == 0 for _, v in items):
        return "<div style='color:#7A8288;font-size:13px;'>Sin datos.</div>"
    width, mid, row_h = 460, 230, 40
    span = max(0.05, max(abs(v) for _, v in items) * 1.25)
    height = 12 + len(items) * row_h
    rows = [f"<line x1='{mid}' y1='4' x2='{mid}' y2='{height-4}' stroke='#CFE8DD' stroke-width='1'/>"]
    for i, (label, value) in enumerate(items):
        y = 8 + i * row_h
        w = (mid - 65) * (abs(value) / span)
        color = COLORS["red"] if value < 0 else COLORS["green"]
        x = mid - w if value < 0 else mid
        vtxt = f"{value:+.3f}".replace(".", ",")
        rows.append(
            f"<text x='0' y='{y + 15:.0f}' font-size='12' fill='#123047'>{html_lib.escape(label[:20])}</text>"
            f"<rect x='{x:.0f}' y='{y:.0f}' width='{w:.0f}' height='20' rx='4' fill='{color}'/>"
            f"<text x='{(x - 5) if value < 0 else (x + w + 5):.0f}' y='{y + 15:.0f}' font-size='11' "
            f"fill='#456257' text-anchor='{'end' if value < 0 else 'start'}'>{vtxt}</text>"
        )
    return f"<svg viewBox='0 0 {width} {height}' width='100%' style='max-width:480px'>" + "".join(rows) + "</svg>"


def _render_chart(chart: dict) -> str:
    t = chart.get("type")
    if t == "bars":
        return _svg_bars(chart["items"], chart.get("unit", ""), chart.get("vmax"), chart.get("color", COLORS["teal"]))
    if t == "donut":
        return _svg_donut(chart.get("pct"), chart.get("label", ""), chart.get("color", COLORS["blue"]))
    if t == "stacked":
        return _svg_stacked(chart["items"])
    if t == "funnel":
        return _svg_funnel(chart["items"])
    if t == "diverging":
        return _svg_diverging(chart["items"])
    return ""


# ============================================================
# RENDER DEL DASHBOARD (pestañas + filtro de corpus)
# ============================================================

PHASE_TABS = [
    ("00", "00 · Limpieza"),
    ("01", "01–04 · Diarización"),
    ("05", "05 · Transcripción"),
    ("06", "06 · Rol proxy"),
    ("07A", "07A · Sentimiento"),
    ("07B", "07B · Afecto audio"),
    ("07C", "07C · Fusión"),
    ("08", "08 · Keywords"),
    ("09", "09 · Huella de voz"),
]


def _render_kpis(kpis) -> str:
    cards = []
    for label, value, kind in kpis:
        cards.append(
            "<div style='flex:1;min-width:140px;background:#F4FBF7;border:1px solid #CFE8DD;"
            "border-radius:12px;padding:16px 18px;'>"
            f"<div style='font-size:30px;font-weight:800;color:#123047;line-height:1;'>{_fmt(value, kind)}</div>"
            f"<div style='font-size:12px;color:#456257;margin-top:6px;'>{html_lib.escape(label)}</div>"
            "</div>"
        )
    return f"<div style='display:flex;flex-wrap:wrap;gap:12px;margin-bottom:16px;'>{''.join(cards)}</div>"


def _render_phase_panel(phase_id: str, agg: dict) -> str:
    data = agg.get(phase_id, {})
    kpis = _render_kpis(data.get("kpis", []))
    charts = data.get("charts")
    if charts is None:  # compatibilidad con fases que aún usen "chart" singular
        single = data.get("chart")
        charts = [single] if single else []

    chart_cards = []
    for chart in charts:
        chart_svg = _render_chart(chart) if chart else ""
        if not chart_svg:
            continue
        chart_cards.append(
            "<div style='flex:1;min-width:260px;background:#FCFEFD;border:1px solid #E4F0EA;"
            "border-radius:12px;padding:16px 18px;'>"
            f"<div style='font-size:11.5px;font-weight:650;color:#2A9D8F;text-transform:uppercase;"
            f"letter-spacing:.04em;margin-bottom:10px;'>{html_lib.escape(chart.get('title', ''))}</div>"
            f"{chart_svg}</div>"
        )
    chart_block = (
        f"<div style='display:flex;flex-wrap:wrap;gap:14px;'>{''.join(chart_cards)}</div>"
        if chart_cards else ""
    )
    return kpis + chart_block


def _render_corpus_version(corpus_key: str, agg: dict) -> str:
    """Renderiza todas las pestañas para un corpus dado (oculto salvo el activo)."""
    panels = []
    for pid, _label in PHASE_TABS:
        panels.append(
            f"<div class='tfm-panel' data-phase='{pid}'>{_render_phase_panel(pid, agg)}</div>"
        )
    return (
        f"<div class='tfm-corpus' data-corpus='{corpus_key}' style='display:none;'>"
        + "".join(panels) + "</div>"
    )


def build_html(aggregates_by_corpus: dict[str, dict], availability: pd.DataFrame) -> str:
    """Ensambla el HTML autónomo y ligero con pestañas y filtro de corpus."""
    n_avail = int(availability["available"].sum()) if not availability.empty else 0
    n_total = len(availability) if not availability.empty else 0
    generated = datetime.now().strftime("%d/%m/%Y %H:%M")

    tabs = "".join(
        f"<button class='tfm-tab' data-phase='{pid}'>{html_lib.escape(label)}</button>"
        for pid, label in PHASE_TABS
    )
    corpus_versions = "".join(
        _render_corpus_version(ckey, agg) for ckey, agg in aggregates_by_corpus.items()
    )
    corpus_options = "".join(
        f"<option value='{c}'>{c}</option>" for c in ["Todos", "Bajas", "Comerciales"]
    )

    script = """
<script>
(function(){
  const root = document.getElementById('tfm-dash');
  let corpus = 'Todos', phase = '00';
  function apply(){
    root.querySelectorAll('.tfm-corpus').forEach(el=>{
      el.style.display = (el.dataset.corpus===corpus) ? 'block' : 'none';
      el.querySelectorAll('.tfm-panel').forEach(p=>{
        p.style.display = (p.dataset.phase===phase) ? 'block' : 'none';
      });
    });
    root.querySelectorAll('.tfm-tab').forEach(t=>{
      t.style.background = (t.dataset.phase===phase) ? '#2A9D6F' : '#EEF3F1';
      t.style.color = (t.dataset.phase===phase) ? 'white' : '#33484F';
    });
  }
  root.querySelectorAll('.tfm-tab').forEach(t=>t.addEventListener('click',()=>{phase=t.dataset.phase;apply();}));
  const sel = document.getElementById('tfm-corpus-filter');
  if(sel) sel.addEventListener('change',()=>{corpus=sel.value;apply();});
  apply();
})();
</script>
"""

    return (
        "<div id='tfm-dash' style='font-family:-apple-system,Segoe UI,Roboto,sans-serif;color:#123047;"
        "background:#FFFFFF;border:1px solid #DDE5EA;border-radius:16px;padding:26px 30px;max-width:920px;'>"
        "<div style='display:flex;justify-content:space-between;align-items:baseline;flex-wrap:wrap;gap:8px;'>"
        "<div style='font-size:12px;font-weight:700;letter-spacing:.06em;text-transform:uppercase;color:#2A9D8F;'>Dashboard global de resultados · TFM</div>"
        f"<div style='font-size:11px;color:#7A8288;'>{n_avail}/{n_total} outputs · {generated}</div>"
        "</div>"
        "<h1 style='margin:8px 0 14px;font-size:28px;font-weight:800;'>Resultados por fase del pipeline</h1>"
        "<div style='display:flex;align-items:center;gap:10px;margin-bottom:14px;'>"
        "<label style='font-size:13px;color:#456257;font-weight:600;'>Corpus:</label>"
        f"<select id='tfm-corpus-filter' style='padding:6px 10px;border:1px solid #CFE8DD;border-radius:8px;font-size:13px;color:#123047;background:#F4FBF7;'>{corpus_options}</select>"
        "</div>"
        f"<div style='display:flex;flex-wrap:wrap;gap:6px;margin-bottom:18px;'>{tabs}</div>"
        f"{corpus_versions}"
        "</div>"
        f"{script}"
    ).replace(
        "class='tfm-tab'",
        "class='tfm-tab' style='border:0;border-radius:8px;padding:8px 12px;font-size:12.5px;font-weight:600;cursor:pointer;background:#EEF3F1;color:#33484F;'"
    )


# ============================================================
# FUNCIÓN PÚBLICA
# ============================================================

def run_dashboard_resultados_globales(
    gcs_client=None, force_restore: bool = False, save_html_snapshot: bool = False,
):
    """
    Genera el dashboard global de resultados (versión ligera) y lo devuelve
    como HTML para mostrar en el notebook.

    Pestañas por fase (00–09), cada una con sus métricas finales agregadas y un
    gráfico representativo. Filtro simple de corpus (Todos / Bajas / Comerciales).
    Solo lee outputs; nunca sube ni modifica nada en GCS.
    """
    global _GCS_CLIENT
    _GCS_CLIENT = gcs_client

    availability = _restore_inputs(force=force_restore)
    datasets = _load_datasets()

    aggregates_by_corpus = {
        corpus: compute_phase_aggregates(datasets, corpus)
        for corpus in ["Todos", "Bajas", "Comerciales"]
    }
    html = build_html(aggregates_by_corpus, availability)

    n_avail = int(availability["available"].sum()) if not availability.empty else 0
    print(f"Outputs disponibles: {n_avail}/{len(availability)}")
    print("Subidas a GCS: deshabilitadas (esta fase solo lee).")

    if save_html_snapshot:
        DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
        HTML_DIR.mkdir(parents=True, exist_ok=True)
        full = (
            "<!DOCTYPE html><html lang='es'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width, initial-scale=1'>"
            "<title>Dashboard global de resultados · TFM</title></head>"
            f"<body style='margin:0;padding:20px;background:#EEF3F1;'>{html}</body></html>"
        )
        HTML_PATH.write_text(full, encoding="utf-8")
        size_kb = HTML_PATH.stat().st_size / 1024
        print(f"HTML guardado: {HTML_PATH} ({size_kb:.0f} KB)")

    return HTML(html)
