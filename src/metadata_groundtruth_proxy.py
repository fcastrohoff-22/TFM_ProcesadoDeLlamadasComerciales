"""Fase 06: metadata oficial y ground truth proxy de roles (AGENT/CLIENT).

Módulo descompuesto en funciones pequeñas a nivel de módulo, siguiendo el
mismo patrón que ``diarizacion.py`` y ``transcripcion_contextual.py``. La
lógica de negocio vive aquí; el notebook 06 orquesta los pasos en celdas
visibles y define todos los parámetros ajustables (umbrales de alineación,
reglas de asignación de rol) en su celda de CONFIGURACIÓN.

El único paso pesado (que recorre audios) es la alineación textual entre la
transcripción de Whisper y la transcripción oficial de BigQuery; ese paso
mantiene checkpoints. El resto de pasos son transformaciones sobre tablas ya
cargadas y se suben a GCS al final.
"""

import re
import unicodedata
from pathlib import Path

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from src.config import ANONYMIZATION_SALT, hash_value
from src.storage_io import upload_file, upload_directory, download_directory


# ============================================================
# NORMALIZACIÓN DE TEXTO Y TOKENS (para el match textual)
# ============================================================

# Stopwords muy básicas para que no dominen palabras vacías en el match.
_MATCH_STOPWORDS = {
    "de", "la", "el", "y", "que", "en", "a", "un", "una", "los", "las",
    "por", "para", "con", "no", "si", "se", "lo", "le", "me", "es", "su",
}


def text_hash(value):
    """Hash de 16 caracteres del texto (sin sal), o NaN si está vacío."""
    import hashlib
    if pd.isna(value):
        return np.nan
    value = str(value)
    if value.strip() == "":
        return np.nan
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def count_words(text):
    """Cuenta palabras (tokens \\w+) de un texto."""
    if pd.isna(text):
        return 0
    return len(re.findall(r"\w+", str(text), flags=re.UNICODE))


def normalize_role(role):
    """Normaliza un rol libre a AGENT / CLIENT / NaN."""
    if pd.isna(role):
        return np.nan
    s = str(role).strip().upper()
    s = re.sub(r"[^A-ZÁÉÍÓÚÑ_ ]", "", s)
    if any(x in s for x in ["AGENTE", "AGENT", "ASESOR", "OPERADOR", "COMERCIAL"]):
        return "AGENT"
    if any(x in s for x in ["CLIENTE", "CUSTOMER", "USUARIO"]):
        return "CLIENT"
    return np.nan


def strip_accents(s):
    """Elimina acentos de una cadena."""
    return "".join(
        c for c in unicodedata.normalize("NFKD", str(s))
        if not unicodedata.combining(c)
    )


def normalize_text_for_match(text):
    """Minúsculas, sin acentos, solo alfanumérico y espacios simples."""
    if pd.isna(text):
        return ""
    s = strip_accents(str(text).lower())
    s = re.sub(r"[^a-z0-9ñáéíóúü\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def tokens_for_match(text):
    """Tokens significativos (sin stopwords) para comparar textos."""
    s = normalize_text_for_match(text)
    toks = re.findall(r"[a-z0-9ñ]+", s)
    return [t for t in toks if t not in _MATCH_STOPWORDS]


def token_containment(a, b):
    """Proporción de tokens compartidos respecto al conjunto más pequeño."""
    ta, tb = set(tokens_for_match(a)), set(tokens_for_match(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(1, min(len(ta), len(tb)))


def token_jaccard(a, b):
    """Índice de Jaccard entre los tokens de dos textos."""
    ta, tb = set(tokens_for_match(a)), set(tokens_for_match(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(1, len(ta | tb))


# ============================================================
# IDENTIDAD DE AUDIO (variantes de nombre para cruzar tablas)
# ============================================================

def normalize_audio_stem(name):
    """Nombre de audio sin extensión .wav ni sufijo _clean."""
    if pd.isna(name):
        return np.nan
    s = Path(str(name).strip()).name
    s = re.sub(r"\.wav$", "", s, flags=re.I)
    s = re.sub(r"_clean$", "", s, flags=re.I)
    return s


def infer_source_dataset_from_name(name):
    """Deduce el dataset de origen (raw / raw_bajas) a partir del nombre."""
    s = normalize_audio_stem(name)
    if pd.isna(s):
        return np.nan
    if str(s).startswith("raw_bajas_"):
        return "raw_bajas"
    if str(s).startswith("raw_"):
        return "raw"
    return np.nan


def strip_source_prefix(stem):
    """Quita el prefijo raw_ / raw_bajas_ de un stem."""
    if pd.isna(stem):
        return np.nan
    s = str(stem).strip()
    if s.startswith("raw_bajas_"):
        return s[len("raw_bajas_"):]
    if s.startswith("raw_"):
        return s[len("raw_"):]
    return s


def make_audio_key_variants(name, source_dataset=None):
    """Variantes de clave de audio para cruzar Whisper, diarización y metadata."""
    stem = normalize_audio_stem(name)
    if pd.isna(stem):
        return []
    base = strip_source_prefix(stem)
    variants = {stem, base}
    if source_dataset == "raw_bajas":
        variants.add(f"raw_bajas_{base}")
    if source_dataset == "raw":
        variants.add(f"raw_{base}")
    return [v for v in variants if isinstance(v, str) and v.strip()]


# ============================================================
# CARGA DE SEGMENTOS DIARIZADOS + TRANSCRIPCIÓN WHISPER
# ============================================================

def load_segments_and_whisper(segments_csv, whisper_candidates):
    """
    Carga el consolidado de segmentos (Notebook 04) y la transcripción de
    Whisper (Notebook 05), añadiendo columnas normalizadas para el match.

    Devuelve (df_segments, df_whisper, whisper_path).
    """
    segments_csv = Path(segments_csv)
    if not segments_csv.exists():
        raise FileNotFoundError(f"No existe SEGMENTS_CSV: {segments_csv}")

    df_segments = pd.read_csv(segments_csv)
    required = ["audio_file", "start", "end", "duration", "speaker_final"]
    missing = [c for c in required if c not in df_segments.columns]
    if missing:
        raise ValueError(f"Faltan columnas necesarias en segmentos: {missing}")

    whisper_path = next((Path(p) for p in whisper_candidates if Path(p).exists()), None)
    if whisper_path is None:
        raise FileNotFoundError(f"No se encontró Whisper segmentado en: {whisper_candidates}")

    df_whisper = pd.read_csv(whisper_path)

    text_col = (
        "text" if "text" in df_whisper.columns
        else "transcription" if "transcription" in df_whisper.columns
        else None
    )
    if text_col is None:
        raise ValueError("No se encontró columna de texto en Whisper ('text' o 'transcription').")

    for c in ["audio_file", "start", "end", "speaker_final"]:
        if c not in df_whisper.columns:
            raise ValueError(f"Falta columna en Whisper: {c}")

    df_whisper["whisper_text"] = df_whisper[text_col].fillna("").astype(str).str.strip()
    df_whisper["whisper_norm"] = df_whisper["whisper_text"].apply(normalize_text_for_match)
    df_whisper["whisper_word_count"] = df_whisper["whisper_text"].apply(count_words)

    return df_segments, df_whisper, whisper_path


# ============================================================
# CARGA DE METADATA OFICIAL DESDE BIGQUERY
# ============================================================

def _normalize_str_col(s):
    return (
        s.astype("string").fillna("").str.strip().str.replace(r"\.0$", "", regex=True)
    )


def load_official_metadata_from_bigquery(
    bq_client, bq_project_id, bq_dataset, bq_metadata_sources, output_dir,
):
    """
    Carga la metadata oficial desde BigQuery conservando la columna
    'transcripcion' (misma fuente que el Notebook 00). Añade audio_id_base,
    hashes anonimizados y flag de transcripción oficial disponible.

    Guarda un snapshot local (privado, no se sube) para trazabilidad.
    """
    frames = []
    for source_dataset, table_name in bq_metadata_sources.items():
        print(f"Cargando BigQuery: {source_dataset} | {table_name}")
        bq_sql = f"SELECT * FROM `{bq_project_id}.{bq_dataset}.{table_name}`"
        df_tmp = bq_client.query(bq_sql).to_dataframe()
        df_tmp["source_dataset"] = source_dataset
        df_tmp["bq_table"] = table_name
        frames.append(df_tmp)

    df_metadata_original = pd.concat(frames, ignore_index=True)

    if "transcripcion" not in df_metadata_original.columns:
        raise ValueError(
            "BigQuery no devolvió columna 'transcripcion'. Revisa el esquema."
        )

    df_meta = df_metadata_original.copy()

    for c in [
        "filename", "customer_id", "agent_id", "brand_ds", "duration_min",
        "transcripcion", "url", "tipo_llamada", "mono_stereo",
        "baja_total_30_dias", "source_dataset", "bq_table",
    ]:
        if c not in df_meta.columns:
            df_meta[c] = pd.NA

    for c in ["source_dataset", "bq_table", "filename", "customer_id", "agent_id", "brand_ds"]:
        df_meta[c] = _normalize_str_col(df_meta[c])

    df_meta["audio_id"] = df_meta["filename"].apply(normalize_audio_stem).astype("string").str.strip()
    df_meta["audio_id_base"] = df_meta["audio_id"].apply(strip_source_prefix).astype("string").str.strip()

    df_meta["audio_hash"] = df_meta["filename"].apply(hash_value)
    df_meta["customer_hash"] = df_meta["customer_id"].apply(hash_value)
    df_meta["agent_hash"] = df_meta["agent_id"].apply(hash_value)

    df_meta["has_official_transcription"] = (
        df_meta["transcripcion"].fillna("").astype(str).str.strip().ne("")
    )

    # Snapshot local privado (contiene transcripción oficial e IDs) — no se sube.
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = output_dir / "bq_metadata_notebook00_exact_with_transcripcion.csv"
    snapshot_cols = [
        "source_dataset", "bq_table", "filename", "audio_id", "audio_id_base",
        "customer_id", "agent_id", "brand_ds", "duration_min", "transcripcion",
        "url", "tipo_llamada", "mono_stereo", "baja_total_30_dias",
        "audio_hash", "customer_hash", "agent_hash", "has_official_transcription",
    ]
    snapshot_cols = [c for c in snapshot_cols if c in df_meta.columns]
    df_meta[snapshot_cols].to_csv(snapshot_path, index=False)

    return df_meta


# ============================================================
# MATCH AUDIO DIARIZADO ↔ METADATA BIGQUERY
# ============================================================

def match_audio_to_metadata(df_segments, df_meta):
    """
    Cruza cada audio diarizado con su metadata oficial de BigQuery mediante
    tres estrategias en cascada: (1) source_dataset + audio_id, (2) audio_id
    único, (3) audio_id duplicado eligiendo la duración más cercana.

    Devuelve (seg_match_audit, coverage) con una fila por audio diarizado.
    """
    seg_audio = df_segments[["audio_file"]].drop_duplicates().copy()

    if "source_dataset" in df_segments.columns:
        seg_source = (
            df_segments[["audio_file", "source_dataset"]]
            .drop_duplicates("audio_file")
            .rename(columns={"source_dataset": "segment_source_dataset"})
        )
        seg_audio = seg_audio.merge(seg_source, on="audio_file", how="left")
    else:
        seg_audio["segment_source_dataset"] = seg_audio["audio_file"].apply(
            infer_source_dataset_from_name
        )

    seg_audio["segment_stem"] = seg_audio["audio_file"].apply(normalize_audio_stem).astype("string").str.strip()
    seg_audio["segment_audio_id_base"] = seg_audio["segment_stem"].apply(strip_source_prefix).astype("string").str.strip()

    seg_counts = (
        df_segments.groupby("audio_file")
        .agg(
            n_diarized_segments=("audio_file", "size"),
            diarized_duration_sec=("duration", "sum"),
            n_speakers=("speaker_final", "nunique"),
            first_start=("start", "min"),
            last_end=("end", "max"),
        )
        .reset_index()
    )
    seg_audio = seg_audio.merge(seg_counts, on="audio_file", how="left")
    seg_audio["diarized_duration_min"] = seg_audio["diarized_duration_sec"] / 60

    meta_lookup = df_meta.copy()
    meta_lookup["source_dataset"] = _normalize_str_col(meta_lookup["source_dataset"])
    meta_lookup["audio_id_base"] = _normalize_str_col(meta_lookup["audio_id_base"])

    meta_cols = [
        "source_dataset", "bq_table", "filename", "audio_id", "audio_id_base",
        "audio_hash", "customer_hash", "agent_hash", "customer_id", "agent_id",
        "brand_ds", "duration_min", "transcripcion", "url",
        "tipo_llamada", "mono_stereo", "baja_total_30_dias",
        "has_official_transcription",
    ]
    meta_cols = [c for c in meta_cols if c in meta_lookup.columns]

    # 1) Match estricto por source_dataset + audio_id.
    seg_with_source = seg_audio[
        seg_audio["segment_source_dataset"].notna()
        & seg_audio["segment_source_dataset"].astype(str).str.strip().ne("")
        & ~seg_audio["segment_source_dataset"].astype(str).str.lower().isin(["nan", "none", "unknown"])
    ].copy()

    match_source = seg_with_source.merge(
        meta_lookup[meta_cols],
        left_on=["segment_source_dataset", "segment_audio_id_base"],
        right_on=["source_dataset", "audio_id_base"], how="left",
    )
    match_source = match_source[match_source["filename"].notna()].copy()
    match_source["match_method"] = "source_dataset_plus_audio_id"
    matched_audio_files = set(match_source["audio_file"].dropna().unique())

    # 2) Fallback por audio_id único en BigQuery.
    remaining = seg_audio[~seg_audio["audio_file"].isin(matched_audio_files)].copy()
    meta_unique = (
        meta_lookup.groupby("audio_id_base")
        .filter(lambda x: x[["source_dataset", "filename"]].drop_duplicates().shape[0] == 1)
        .copy()
    )
    match_unique = remaining.merge(
        meta_unique[meta_cols], left_on="segment_audio_id_base",
        right_on="audio_id_base", how="left",
    )
    match_unique = match_unique[match_unique["filename"].notna()].copy()
    match_unique["match_method"] = "audio_id_unique_fallback"
    matched_audio_files.update(match_unique["audio_file"].dropna().unique())

    # 3) Fallback por audio_id duplicado, eligiendo duración más cercana.
    remaining = seg_audio[~seg_audio["audio_file"].isin(matched_audio_files)].copy()
    meta_dupes = (
        meta_lookup.groupby("audio_id_base")
        .filter(lambda x: x[["source_dataset", "filename"]].drop_duplicates().shape[0] > 1)
        .copy()
    )
    match_duration = remaining.merge(
        meta_dupes[meta_cols], left_on="segment_audio_id_base",
        right_on="audio_id_base", how="left",
    )
    match_duration = match_duration[match_duration["filename"].notna()].copy()

    if len(match_duration):
        match_duration["duration_min_num"] = pd.to_numeric(match_duration.get("duration_min"), errors="coerce")
        match_duration["duration_diff_abs"] = (
            match_duration["diarized_duration_min"] - match_duration["duration_min_num"]
        ).abs()
        match_duration = (
            match_duration.sort_values(["audio_file", "duration_diff_abs"], na_position="last")
            .drop_duplicates("audio_file", keep="first").copy()
        )
        match_duration["match_method"] = "audio_id_duplicate_duration_fallback"

    matches = pd.concat([match_source, match_unique, match_duration], ignore_index=True, sort=False)
    method_rank = {
        "source_dataset_plus_audio_id": 1,
        "audio_id_unique_fallback": 2,
        "audio_id_duplicate_duration_fallback": 3,
    }
    if len(matches):
        matches["method_rank"] = matches["match_method"].map(method_rank).fillna(99)
        matches = (
            matches.sort_values(["audio_file", "method_rank"])
            .drop_duplicates("audio_file", keep="first").copy()
        )

    match_cols = ["audio_file", "match_method"] + meta_cols
    match_cols = [c for c in match_cols if c in matches.columns]

    seg_match_audit = seg_audio.merge(
        matches[match_cols] if len(matches) else pd.DataFrame(columns=match_cols),
        on="audio_file", how="left",
    )
    seg_match_audit["metadata_matched"] = seg_match_audit["match_method"].notna()
    seg_match_audit["has_official_transcription"] = (
        seg_match_audit.get("has_official_transcription", pd.Series(False, index=seg_match_audit.index))
        .fillna(False).astype(bool)
    )

    coverage = seg_match_audit["metadata_matched"].mean()
    return seg_match_audit, coverage


# ============================================================
# EXTRACCIÓN DE TURNOS OFICIALES CON ROLES EXPLÍCITOS
# ============================================================

# Solo se aceptan roles que aparezcan explícitamente como prefijo en el texto
# oficial (AGENT:, CLIENTE:, etc.). Un turno puede ocupar varias líneas hasta
# el siguiente prefijo.
ROLE_PREFIX_RE = re.compile(
    r"^\s*(?P<role>AGENT|AGENTE|ASESOR|ASESORA|OPERADOR|OPERADORA|COMERCIAL|"
    r"CLIENT|CLIENTE|CUSTOMER|USUARIO|USUARIA)\s*[#\d_\-\. ]*\s*[:\-]\s*(?P<text>.*)$",
    flags=re.IGNORECASE,
)


def normalize_role_strict(role):
    """Normaliza estrictamente un rol de prefijo textual a AGENT / CLIENT / NaN."""
    if role is None or pd.isna(role):
        return np.nan
    r = strip_accents(str(role).strip().upper())
    r = re.sub(r"[^A-ZÑ]", "", r)
    if r in {"AGENT", "AGENTE", "ASESOR", "ASESORA", "OPERADOR", "OPERADORA", "COMERCIAL"}:
        return "AGENT"
    if r in {"CLIENT", "CLIENTE", "CUSTOMER", "USUARIO", "USUARIA"}:
        return "CLIENT"
    return np.nan


def parse_official_turns_role_prefixed(text, min_words_official, min_text_chars):
    """
    Extrae turnos oficiales solo cuando hay prefijo textual de rol explícito.
    Las líneas sin prefijo se concatenan al turno anterior. Filtra turnos por
    número mínimo de palabras y caracteres.
    """
    if pd.isna(text) or str(text).strip() == "":
        return []

    raw = str(text).replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.strip() for ln in raw.split("\n") if ln.strip()]

    turns = []
    current = None

    for line in lines:
        m = ROLE_PREFIX_RE.match(line)
        if m:
            if current is not None and str(current.get("official_text", "")).strip():
                turns.append(current)
            current = {
                "official_role_raw": m.group("role"),
                "official_role": normalize_role_strict(m.group("role")),
                "official_role_source": "explicit_text_prefix",
                "official_text": m.group("text").strip(),
                "official_start": None,
                "official_end": None,
            }
        else:
            if current is not None:
                current["official_text"] = (
                    str(current.get("official_text", "")).strip() + " " + line
                ).strip()

    if current is not None and str(current.get("official_text", "")).strip():
        turns.append(current)

    cleaned = []
    for t in turns:
        role = t.get("official_role")
        txt = str(t.get("official_text", "")).strip()
        if role not in {"AGENT", "CLIENT"}:
            continue
        norm = normalize_text_for_match(txt)
        wc = count_words(txt)
        if wc < min_words_official or len(norm) < min_text_chars:
            continue
        cleaned.append({
            "official_turn_idx": len(cleaned),
            "official_role_raw": t.get("official_role_raw"),
            "official_role": role,
            "official_role_source": t.get("official_role_source"),
            "official_start": t.get("official_start"),
            "official_end": t.get("official_end"),
            "official_text": txt,
            "official_norm": norm,
            "official_word_count": wc,
            "official_text_hash": text_hash(txt),
        })
    return cleaned


def extract_official_turns(
    seg_match_audit, min_words_official, min_text_chars, require_both_roles_per_audio,
):
    """
    Extrae los turnos oficiales AGENT/CLIENT de todos los audios con metadata.

    Devuelve (df_official_turns, df_official_turns_all, role_counts_all,
    eligible_audio_for_role_proxy).
    """
    if "transcripcion" not in seg_match_audit.columns:
        raise ValueError("seg_match_audit no tiene columna 'transcripcion'.")

    turn_rows = []
    for _, r in seg_match_audit[seg_match_audit["metadata_matched"]].iterrows():
        turns = parse_official_turns_role_prefixed(
            r.get("transcripcion", np.nan), min_words_official, min_text_chars,
        )
        for t in turns:
            turn_rows.append({
                "audio_file": r["audio_file"],
                "source_dataset": r.get("source_dataset", pd.NA),
                "audio_id": r.get("audio_id", pd.NA),
                "filename": r.get("filename", pd.NA),
                "customer_hash": r.get("customer_hash", pd.NA),
                "agent_hash": r.get("agent_hash", pd.NA),
                **t,
            })

    df_official_turns_all = pd.DataFrame(turn_rows)
    if len(df_official_turns_all) == 0:
        raise RuntimeError(
            "No se extrajeron turnos oficiales AGENT/CLIENT con prefijo textual explícito."
        )

    role_counts_all = (
        df_official_turns_all.groupby(["audio_file", "official_role"])
        .size().unstack(fill_value=0).reset_index()
    )
    for c in ["AGENT", "CLIENT"]:
        if c not in role_counts_all.columns:
            role_counts_all[c] = 0
    role_counts_all["has_both_roles"] = role_counts_all["AGENT"].gt(0) & role_counts_all["CLIENT"].gt(0)

    eligible = set(role_counts_all.loc[role_counts_all["has_both_roles"], "audio_file"])

    if require_both_roles_per_audio:
        df_official_turns = df_official_turns_all[
            df_official_turns_all["audio_file"].isin(eligible)
        ].copy()
    else:
        df_official_turns = df_official_turns_all.copy()

    return df_official_turns, df_official_turns_all, role_counts_all, eligible


# ============================================================
# ALINEACIÓN TEXTUAL WHISPER ↔ TRANSCRIPCIÓN OFICIAL
# ============================================================

def build_whisper_windows_for_audio(
    df_audio, max_window_size, max_gap_between_segments_sec, min_words_whisper_window,
):
    """
    Construye ventanas de segmentos consecutivos del mismo speaker (hasta
    ``max_window_size``), para comparar bloques de Whisper con cada turno
    oficial. Devuelve un DataFrame de ventanas.
    """
    df_audio = df_audio.sort_values(["start", "end"]).reset_index(drop=True).copy()
    df_audio["segment_order_in_audio"] = np.arange(len(df_audio))
    windows = []

    for i in range(len(df_audio)):
        base = df_audio.iloc[i]
        speaker = base["speaker_final"]
        text_parts = []
        segment_indices = []
        start = float(base["start"])
        last_end = None

        for w in range(1, max_window_size + 1):
            j = i + w - 1
            if j >= len(df_audio):
                break
            row = df_audio.iloc[j]
            if row["speaker_final"] != speaker:
                break
            if last_end is not None and float(row["start"]) - last_end > max_gap_between_segments_sec:
                break
            txt = str(row.get("whisper_text", "") or "").strip()
            text_parts.append(txt)
            segment_indices.append(int(row["segment_order_in_audio"]))
            last_end = float(row["end"])

            combined_text = " ".join([t for t in text_parts if t]).strip()
            if count_words(combined_text) < min_words_whisper_window:
                continue

            windows.append({
                "audio_file": base["audio_file"],
                "speaker_final": speaker,
                "window_start": start,
                "window_end": last_end,
                "window_size": w,
                "window_first_segment_order": segment_indices[0],
                "window_last_segment_order": segment_indices[-1],
                "window_segment_orders": "|".join(map(str, segment_indices)),
                "whisper_text": combined_text,
                "whisper_norm": normalize_text_for_match(combined_text),
                "whisper_word_count": count_words(combined_text),
            })
    return pd.DataFrame(windows)


def align_one_audio(audio_file, df_turns_audio, df_whisper_audio, params):
    """
    Alinea los turnos oficiales de un audio con ventanas de Whisper usando
    similitud coseno de n-gramas de caracteres (TF-IDF) más contención de
    tokens. ``params`` es un dict con todos los umbrales (celda CONFIGURACIÓN).

    Devuelve un DataFrame de candidatos con la columna 'accepted_default'.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
    from sklearn.metrics.pairwise import linear_kernel  # type: ignore

    windows = build_whisper_windows_for_audio(
        df_whisper_audio,
        params["MAX_WINDOW_SIZE"],
        params["MAX_GAP_BETWEEN_SEGMENTS_SEC"],
        params["MIN_WORDS_WHISPER_WINDOW"],
    )
    if len(df_turns_audio) == 0 or len(windows) == 0:
        return pd.DataFrame()

    official_texts = df_turns_audio["official_norm"].fillna("").astype(str).tolist()
    window_texts = windows["whisper_norm"].fillna("").astype(str).tolist()
    all_texts = official_texts + window_texts

    if len(set([t for t in all_texts if t.strip()])) < 2:
        return pd.DataFrame()

    try:
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
        X = vectorizer.fit_transform(all_texts)
        X_off = X[:len(official_texts)]
        X_win = X[len(official_texts):]
        sim = linear_kernel(X_off, X_win)
    except Exception as e:
        print("Error vectorizando", audio_file, e)
        return pd.DataFrame()

    candidates = []
    for oi, (_, turn) in enumerate(df_turns_audio.reset_index(drop=True).iterrows()):
        scores = sim[oi]
        if scores.size == 0:
            continue
        order = np.argsort(scores)[::-1]
        best_idx = int(order[0])
        second_score = float(scores[order[1]]) if len(order) > 1 else 0.0
        best_score = float(scores[best_idx])
        win = windows.iloc[best_idx]

        containment = token_containment(turn["official_text"], win["whisper_text"])
        jaccard = token_jaccard(turn["official_text"], win["whisper_text"])
        combined = 0.65 * best_score + 0.35 * containment
        margin = best_score - second_score

        candidates.append({
            "audio_file": audio_file,
            "official_turn_idx": turn["official_turn_idx"],
            "official_role": turn["official_role"],
            "official_text_hash": turn["official_text_hash"],
            "official_text": turn["official_text"],
            "official_word_count": turn["official_word_count"],
            "speaker_final": win["speaker_final"],
            "window_start": win["window_start"],
            "window_end": win["window_end"],
            "window_size": win["window_size"],
            "window_first_segment_order": win["window_first_segment_order"],
            "window_last_segment_order": win["window_last_segment_order"],
            "window_segment_orders": win["window_segment_orders"],
            "whisper_text": win["whisper_text"],
            "whisper_word_count": win["whisper_word_count"],
            "char_cosine": best_score,
            "second_char_cosine": second_score,
            "similarity_margin": margin,
            "token_containment": containment,
            "token_jaccard": jaccard,
            "combined_score": combined,
        })

    cand = pd.DataFrame(candidates)
    if len(cand) == 0:
        return cand

    cand["accepted_default"] = (
        cand["combined_score"].ge(params["ACCEPT_COMBINED_SCORE"])
        & cand["char_cosine"].ge(params["ACCEPT_CHAR_COSINE"])
        & cand["token_containment"].ge(params["ACCEPT_TOKEN_CONTAINMENT"])
        & cand["similarity_margin"].ge(params["ACCEPT_MARGIN"])
    )
    return cand


CANDIDATE_COLUMNS = [
    "audio_file", "official_turn_idx", "official_role", "official_text_hash",
    "official_text", "official_word_count", "speaker_final", "window_start",
    "window_end", "window_size", "window_first_segment_order",
    "window_last_segment_order", "window_segment_orders", "whisper_text",
    "whisper_word_count", "char_cosine", "second_char_cosine",
    "similarity_margin", "token_containment", "token_jaccard",
    "combined_score", "accepted_default",
]


def run_text_alignment(
    df_official_turns, df_whisper, params, checkpoint_dir,
    save_checkpoint_every_n=50, max_audios_to_process=None, progress_callback=None,
):
    """
    Ejecuta la alineación textual audio por audio (paso pesado con checkpoint).

    Devuelve (text_alignment_candidates, alignment_processing_summary).
    """
    candidate_audios = sorted(
        set(df_official_turns["audio_file"]).intersection(set(df_whisper["audio_file"]))
    )
    if max_audios_to_process is not None:
        candidate_audios = candidate_audios[:max_audios_to_process]

    all_candidates = []
    processed_rows = []
    total = len(candidate_audios)

    for i, audio_file in enumerate(candidate_audios, start=1):
        if progress_callback is not None:
            progress_callback(i, total, audio_file)

        df_turns_audio = df_official_turns[df_official_turns["audio_file"] == audio_file].copy()
        df_whisper_audio = df_whisper[df_whisper["audio_file"] == audio_file].copy()

        cand = align_one_audio(audio_file, df_turns_audio, df_whisper_audio, params)
        if len(cand) > 0:
            all_candidates.append(cand)

        processed_rows.append({
            "audio_file": audio_file,
            "n_official_turns": len(df_turns_audio),
            "n_whisper_segments": len(df_whisper_audio),
            "n_candidates": len(cand),
            "n_accepted_default": (
                int(cand["accepted_default"].sum())
                if len(cand) and "accepted_default" in cand.columns else 0
            ),
        })

        if save_checkpoint_every_n and i % save_checkpoint_every_n == 0:
            ckpt = pd.concat(all_candidates, ignore_index=True) if all_candidates else pd.DataFrame()
            ckpt_path = Path(checkpoint_dir) / f"text_alignment_candidates_checkpoint_{i:04d}.csv"
            ckpt.to_csv(ckpt_path, index=False)

    text_alignment_candidates = (
        pd.concat(all_candidates, ignore_index=True)
        if all_candidates else pd.DataFrame(columns=CANDIDATE_COLUMNS)
    )
    alignment_processing_summary = pd.DataFrame(
        processed_rows,
        columns=["audio_file", "n_official_turns", "n_whisper_segments",
                 "n_candidates", "n_accepted_default"],
    )
    return text_alignment_candidates, alignment_processing_summary


# ============================================================
# MAPEO SPEAKER -> ROL (estricto, para diagnóstico/holdout)
# ============================================================

def infer_one_to_one_speaker_role_mapping_strict(matches_df, params):
    """
    Mapeo estricto speaker->rol basado en matches aceptados aislados.
    Se conserva para diagnóstico y evaluación holdout; no es el mapping
    principal. ``params`` trae los umbrales de rol.
    """
    if matches_df is None or len(matches_df) == 0:
        return pd.DataFrame()

    rows = []
    for audio_file, df_a in matches_df.groupby("audio_file"):
        roles_present = set(df_a["official_role"].dropna().unique())
        if params["REQUIRE_BOTH_ROLES_PER_AUDIO"] and not {"AGENT", "CLIENT"}.issubset(roles_present):
            continue

        counts = (
            df_a.groupby(["speaker_final", "official_role"])
            .agg(
                n_matches_role=("official_role", "size"),
                mean_combined_score=("combined_score", "mean"),
                mean_char_cosine=("char_cosine", "mean"),
            ).reset_index()
        )
        totals = counts.groupby("speaker_final")["n_matches_role"].sum().reset_index(name="n_matches_total")
        counts = counts.merge(totals, on="speaker_final", how="left")
        counts["role_share"] = counts["n_matches_role"] / counts["n_matches_total"].replace(0, np.nan)

        def best_for_role(role):
            sub = counts[counts["official_role"] == role].copy()
            sub = sub[
                sub["n_matches_role"].ge(params["MIN_MATCHES_PER_ROLE_IN_AUDIO"])
                & sub["n_matches_total"].ge(params["MIN_MATCHES_FOR_SPEAKER_ROLE"])
                & sub["role_share"].ge(params["MIN_ROLE_PURITY"])
            ]
            if len(sub) == 0:
                return None
            sub = sub.sort_values(
                ["role_share", "n_matches_role", "mean_combined_score"],
                ascending=[False, False, False],
            )
            return sub.iloc[0].to_dict()

        best_agent = best_for_role("AGENT")
        best_client = best_for_role("CLIENT")
        if best_agent is None or best_client is None:
            continue
        if params["REQUIRE_ONE_TO_ONE_AGENT_CLIENT_MAPPING"] and best_agent["speaker_final"] == best_client["speaker_final"]:
            continue

        for role, best in [("AGENT", best_agent), ("CLIENT", best_client)]:
            rows.append({
                "audio_file": audio_file,
                "speaker_final": best["speaker_final"],
                "probable_role": role,
                "n_matches_role": int(best["n_matches_role"]),
                "n_matches_total": int(best["n_matches_total"]),
                "role_confidence": float(best["role_share"]),
                "mean_combined_score": float(best["mean_combined_score"]),
                "mean_char_cosine": float(best["mean_char_cosine"]),
                "role_mapping_status": "accepted_speaker_role_strict_matches",
                "role_assignment_method": "strict_accepted_matches_one_to_one",
            })

    return pd.DataFrame(rows)


# ============================================================
# MAPEO SPEAKER -> ROL POR EVIDENCIA TEXTUAL AGREGADA (principal)
# ============================================================

def score_text_pair(text_a, text_b):
    """Score textual entre dos textos (TF-IDF char coseno + overlap léxico)."""
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
    from sklearn.metrics.pairwise import linear_kernel  # type: ignore

    norm_a = normalize_text_for_match(str(text_a or "").strip())
    norm_b = normalize_text_for_match(str(text_b or "").strip())

    if count_words(norm_a) == 0 or count_words(norm_b) == 0:
        return {"char_cosine": 0.0, "token_containment": 0.0, "token_jaccard": 0.0, "combined_score": 0.0}

    try:
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
        X = vectorizer.fit_transform([norm_a, norm_b])
        char_cosine = float(linear_kernel(X[0:1], X[1:2])[0, 0])
    except Exception:
        char_cosine = 0.0

    containment = float(token_containment(norm_a, norm_b))
    jaccard = float(token_jaccard(norm_a, norm_b))
    combined = 0.55 * char_cosine + 0.45 * containment
    return {"char_cosine": char_cosine, "token_containment": containment,
            "token_jaccard": jaccard, "combined_score": float(combined)}


def select_role_evidence_segments(df_audio_speaker, params):
    """Selecciona segmentos de evidencia textual para un speaker (bajo overlap, texto útil)."""
    if df_audio_speaker is None or len(df_audio_speaker) == 0:
        return pd.DataFrame()

    df = df_audio_speaker.copy()
    df["whisper_text"] = df.get("whisper_text", "").fillna("").astype(str).str.strip()
    if "whisper_word_count" not in df.columns:
        df["whisper_word_count"] = df["whisper_text"].apply(count_words)

    df = df[
        df["whisper_text"].ne("")
        & df["whisper_word_count"].ge(params["ROLE_EVIDENCE_MIN_WORDS_PER_SEGMENT"])
        & pd.to_numeric(df["duration"], errors="coerce").fillna(0).gt(0)
    ].copy()
    if len(df) == 0:
        return df

    if "transcription_status" in df.columns:
        ok_mask = df["transcription_status"].fillna("").astype(str).str.lower().eq("ok")
        if ok_mask.any():
            df = df[ok_mask].copy()

    if "overlap_ratio" not in df.columns:
        df["overlap_ratio"] = 0.0
    if "rms_dbfs" not in df.columns:
        df["rms_dbfs"] = np.nan

    df["overlap_ratio_num"] = pd.to_numeric(df["overlap_ratio"], errors="coerce").fillna(0.0)
    df["duration_num"] = pd.to_numeric(df["duration"], errors="coerce").fillna(0.0)
    df["rms_dbfs_num"] = pd.to_numeric(df["rms_dbfs"], errors="coerce")
    df["low_overlap_preferred"] = df["overlap_ratio_num"].le(params["ROLE_EVIDENCE_MAX_OVERLAP_RATIO_STRICT"])

    strict = df[df["low_overlap_preferred"]].copy()
    if (
        len(strict) >= params["ROLE_EVIDENCE_MIN_SEGMENTS_PER_SPEAKER"]
        and strict["duration_num"].sum() >= params["ROLE_EVIDENCE_MIN_SECONDS_PER_SPEAKER"]
        and strict["whisper_word_count"].sum() >= params["ROLE_EVIDENCE_MIN_WORDS_PER_SPEAKER"]
    ):
        pool = strict
    else:
        pool = df

    pool = pool.sort_values(
        ["overlap_ratio_num", "whisper_word_count", "duration_num", "start"],
        ascending=[True, False, False, True],
    ).copy()

    selected_rows = []
    total_seconds = 0.0
    for _, row in pool.iterrows():
        if len(selected_rows) >= params["ROLE_EVIDENCE_MAX_SEGMENTS_PER_SPEAKER"]:
            break
        selected_rows.append(row)
        total_seconds += float(row.get("duration_num", 0.0) or 0.0)
        if total_seconds >= params["ROLE_EVIDENCE_TARGET_SECONDS_PER_SPEAKER"]:
            break

    return pd.DataFrame(selected_rows)


def build_aggregated_role_evidence(df_whisper_input, params):
    """Construye una fila de evidencia textual por audio-speaker."""
    rows = []
    for (audio_file, speaker_final), df_sp in df_whisper_input.groupby(
        ["audio_file", "speaker_final"], dropna=False
    ):
        if pd.isna(speaker_final):
            continue
        selected = select_role_evidence_segments(df_sp, params)
        if len(selected) == 0:
            continue

        evidence_text = " ".join(selected["whisper_text"].fillna("").astype(str).str.strip()).strip()
        evidence_words = count_words(evidence_text)
        evidence_seconds = float(pd.to_numeric(selected.get("duration", 0), errors="coerce").fillna(0).sum())
        max_overlap = float(pd.to_numeric(selected.get("overlap_ratio", 0), errors="coerce").fillna(0).max()) if "overlap_ratio" in selected.columns else 0.0
        mean_overlap = float(pd.to_numeric(selected.get("overlap_ratio", 0), errors="coerce").fillna(0).mean()) if "overlap_ratio" in selected.columns else 0.0

        status = "sufficient_evidence"
        if len(selected) < params["ROLE_EVIDENCE_MIN_SEGMENTS_PER_SPEAKER"]:
            status = "low_segment_count"
        elif evidence_seconds < params["ROLE_EVIDENCE_MIN_SECONDS_PER_SPEAKER"]:
            status = "low_duration"
        elif evidence_words < params["ROLE_EVIDENCE_MIN_WORDS_PER_SPEAKER"]:
            status = "low_word_count"

        rows.append({
            "audio_file": audio_file,
            "speaker_final": speaker_final,
            "n_evidence_segments": int(len(selected)),
            "speaker_evidence_seconds": evidence_seconds,
            "speaker_evidence_words": int(evidence_words),
            "mean_evidence_overlap_ratio": mean_overlap,
            "max_evidence_overlap_ratio": max_overlap,
            "first_evidence_start": float(pd.to_numeric(selected["start"], errors="coerce").min()) if "start" in selected.columns else np.nan,
            "last_evidence_end": float(pd.to_numeric(selected["end"], errors="coerce").max()) if "end" in selected.columns else np.nan,
            "evidence_status": status,
            "evidence_text": evidence_text,
            "evidence_text_preview": evidence_text[:300],
            "evidence_segment_orders": "|".join(selected.get("segment_order_in_audio", selected.index).astype(str).tolist()),
        })
    return pd.DataFrame(rows)


def infer_aggregated_speaker_role_mapping(df_whisper_input, df_official_turns_input, params):
    """
    Asigna SPEAKER_XX a AGENT/CLIENT por evidencia textual agregada (mapping
    principal). Devuelve (mapping, evidencia, matriz de scores, auditoría).
    """
    evidence = build_aggregated_role_evidence(df_whisper_input, params)
    mapping_rows, score_rows, audit_rows = [], [], []

    if len(evidence) == 0 or df_official_turns_input is None or len(df_official_turns_input) == 0:
        return pd.DataFrame(), evidence, pd.DataFrame(), pd.DataFrame()

    for audio_file, df_ev_audio in evidence.groupby("audio_file"):
        df_turns_audio = df_official_turns_input[df_official_turns_input["audio_file"] == audio_file].copy()
        roles_present = set(df_turns_audio["official_role"].dropna().unique())

        if params["REQUIRE_BOTH_ROLES_PER_AUDIO"] and not {"AGENT", "CLIENT"}.issubset(roles_present):
            audit_rows.append({"audio_file": audio_file, "aggregate_mapping_status": "missing_both_official_roles",
                               "n_speakers_with_evidence": df_ev_audio["speaker_final"].nunique(),
                               "roles_present": "|".join(sorted(roles_present))})
            continue

        official_by_role = {}
        for role in ["AGENT", "CLIENT"]:
            official_by_role[role] = " ".join(
                df_turns_audio.loc[df_turns_audio["official_role"] == role, "official_text"]
                .fillna("").astype(str).str.strip().tolist()
            ).strip()

        if count_words(official_by_role["AGENT"]) == 0 or count_words(official_by_role["CLIENT"]) == 0:
            audit_rows.append({"audio_file": audio_file, "aggregate_mapping_status": "empty_official_role_text",
                               "n_speakers_with_evidence": df_ev_audio["speaker_final"].nunique(),
                               "roles_present": "|".join(sorted(roles_present))})
            continue

        df_ev_audio = df_ev_audio.copy()
        df_ev_audio["speaker_evidence_seconds"] = pd.to_numeric(df_ev_audio["speaker_evidence_seconds"], errors="coerce").fillna(0)
        df_ev_audio["speaker_evidence_words"] = pd.to_numeric(df_ev_audio["speaker_evidence_words"], errors="coerce").fillna(0)
        df_ev_audio = df_ev_audio.sort_values(
            ["speaker_evidence_words", "speaker_evidence_seconds", "n_evidence_segments"],
            ascending=[False, False, False],
        )

        if df_ev_audio["speaker_final"].nunique() < 2:
            audit_rows.append({"audio_file": audio_file, "aggregate_mapping_status": "less_than_two_speakers_with_evidence",
                               "n_speakers_with_evidence": df_ev_audio["speaker_final"].nunique(),
                               "roles_present": "|".join(sorted(roles_present))})
            continue

        selected_speakers = df_ev_audio.drop_duplicates("speaker_final").head(2).copy()
        speakers = selected_speakers["speaker_final"].tolist()

        speaker_scores = {}
        for _, ev in selected_speakers.iterrows():
            sp = ev["speaker_final"]
            speaker_scores[sp] = {}
            for role in ["AGENT", "CLIENT"]:
                scores = score_text_pair(ev["evidence_text"], official_by_role[role])
                speaker_scores[sp][role] = scores
                score_rows.append({
                    "audio_file": audio_file, "speaker_final": sp, "official_role": role,
                    "char_cosine": scores["char_cosine"], "token_containment": scores["token_containment"],
                    "token_jaccard": scores["token_jaccard"], "combined_score": scores["combined_score"],
                    "speaker_evidence_seconds": float(ev["speaker_evidence_seconds"]),
                    "speaker_evidence_words": int(ev["speaker_evidence_words"]),
                    "n_evidence_segments": int(ev["n_evidence_segments"]),
                    "evidence_status": ev["evidence_status"],
                })

        sp_a, sp_b = speakers[0], speakers[1]
        combo_1 = speaker_scores[sp_a]["AGENT"]["combined_score"] + speaker_scores[sp_b]["CLIENT"]["combined_score"]
        combo_2 = speaker_scores[sp_a]["CLIENT"]["combined_score"] + speaker_scores[sp_b]["AGENT"]["combined_score"]

        if combo_1 >= combo_2:
            assignment = {sp_a: "AGENT", sp_b: "CLIENT"}
            best_pair_score, second_pair_score = combo_1, combo_2
        else:
            assignment = {sp_a: "CLIENT", sp_b: "AGENT"}
            best_pair_score, second_pair_score = combo_2, combo_1

        pair_margin = float(best_pair_score - second_pair_score)

        if best_pair_score < params["MIN_AGGREGATED_PAIR_SCORE"]:
            status = "rejected_low_aggregate_score"
        elif pair_margin < params["MIN_AGGREGATED_PAIR_MARGIN"]:
            status = "assigned_low_margin_aggregate_text"
        else:
            status = "accepted_speaker_role"

        audit_rows.append({
            "audio_file": audio_file, "aggregate_mapping_status": status,
            "n_speakers_with_evidence": df_ev_audio["speaker_final"].nunique(),
            "roles_present": "|".join(sorted(roles_present)),
            "best_pair_score": float(best_pair_score), "second_pair_score": float(second_pair_score),
            "pair_score_margin": pair_margin, "speaker_a": sp_a, "speaker_b": sp_b,
        })

        if status == "rejected_low_aggregate_score":
            continue

        for sp, role in assignment.items():
            ev = selected_speakers[selected_speakers["speaker_final"] == sp].iloc[0]
            score_agent = speaker_scores[sp]["AGENT"]["combined_score"]
            score_client = speaker_scores[sp]["CLIENT"]["combined_score"]
            assigned_score = speaker_scores[sp][role]["combined_score"]
            other_role = "CLIENT" if role == "AGENT" else "AGENT"
            other_score = speaker_scores[sp][other_role]["combined_score"]

            mapping_rows.append({
                "audio_file": audio_file, "speaker_final": sp, "probable_role": role,
                "n_matches_role": np.nan, "n_matches_total": int(ev["n_evidence_segments"]),
                "role_confidence": float(max(0.0, assigned_score - other_score)),
                "mean_combined_score": float(assigned_score),
                "mean_char_cosine": float(speaker_scores[sp][role]["char_cosine"]),
                "score_vs_agent": float(score_agent), "score_vs_client": float(score_client),
                "assigned_role_score": float(assigned_score), "other_role_score": float(other_score),
                "speaker_role_score_margin": float(assigned_score - other_score),
                "best_pair_score": float(best_pair_score), "second_pair_score": float(second_pair_score),
                "pair_score_margin": pair_margin, "n_evidence_segments": int(ev["n_evidence_segments"]),
                "speaker_evidence_seconds": float(ev["speaker_evidence_seconds"]),
                "speaker_evidence_words": int(ev["speaker_evidence_words"]),
                "mean_evidence_overlap_ratio": float(ev["mean_evidence_overlap_ratio"]),
                "max_evidence_overlap_ratio": float(ev["max_evidence_overlap_ratio"]),
                "first_evidence_start": ev["first_evidence_start"], "last_evidence_end": ev["last_evidence_end"],
                "evidence_status": ev["evidence_status"], "evidence_text_preview": ev["evidence_text_preview"],
                "role_mapping_status": status, "role_assignment_method": "aggregated_speaker_text_one_to_one",
            })

    return pd.DataFrame(mapping_rows), evidence, pd.DataFrame(score_rows), pd.DataFrame(audit_rows)


# ============================================================
# PROPAGACIÓN DEL ROL AL DATASET DE SEGMENTOS
# ============================================================

def enrich_mapping_with_metadata(speaker_role_mapping, seg_match_audit):
    """Añade metadata a nivel audio al mapping de roles."""
    if len(speaker_role_mapping) == 0:
        return speaker_role_mapping
    meta_cols = [
        c for c in ["audio_file", "agent_hash", "customer_hash", "brand_ds", "duration_min"]
        if c in seg_match_audit.columns
    ]
    return speaker_role_mapping.merge(
        seg_match_audit[meta_cols].drop_duplicates("audio_file"),
        on="audio_file", how="left",
    )


def propagate_role_to_segments(df_segments, speaker_role_mapping):
    """
    Propaga el rol proxy (AGENT/CLIENT) a cada segmento diarizado según el
    mapping speaker->rol aceptado. Devuelve segment_level_proxy_textual.
    """
    segment_level = df_segments.copy()
    segment_level["segment_order_in_audio"] = segment_level.groupby("audio_file").cumcount()

    if len(speaker_role_mapping):
        accepted_mapping = speaker_role_mapping[
            speaker_role_mapping["role_mapping_status"].isin([
                "accepted_speaker_role", "assigned_low_margin_aggregate_text",
            ])
        ].copy()

        map_cols = [
            "audio_file", "speaker_final", "probable_role", "role_confidence",
            "n_matches_total", "role_mapping_status", "role_assignment_method",
            "score_vs_agent", "score_vs_client", "assigned_role_score", "other_role_score",
            "speaker_role_score_margin", "best_pair_score", "second_pair_score", "pair_score_margin",
            "n_evidence_segments", "speaker_evidence_seconds", "speaker_evidence_words",
            "mean_evidence_overlap_ratio", "max_evidence_overlap_ratio", "evidence_status",
        ]
        map_cols = [c for c in map_cols if c in accepted_mapping.columns]
        segment_level = segment_level.merge(
            accepted_mapping[map_cols], on=["audio_file", "speaker_final"], how="left",
        )
    else:
        for col in ["probable_role", "role_mapping_status", "role_assignment_method"]:
            segment_level[col] = pd.NA
        for col in ["role_confidence", "n_matches_total"]:
            segment_level[col] = np.nan

    segment_level["official_role_proxy"] = segment_level["probable_role"]
    segment_level["proxy_method"] = np.where(
        segment_level["official_role_proxy"].notna(),
        "aggregated_speaker_text_one_to_one", "no_textual_proxy",
    )
    segment_level["proxy_confidence"] = segment_level["role_confidence"]
    return segment_level


# ============================================================
# EVALUACIÓN HOLDOUT DEL MAPPING SPEAKER -> ROLE
# ============================================================

def evaluate_mapping_holdout(accepted_matches, params):
    """
    Evalúa el mapping estricto con un split determinístico 70/30 sobre los
    matches aceptados. Devuelve (holdout_metrics, holdout_predictions, cm_df).
    """
    from sklearn.metrics import (  # type: ignore
        accuracy_score, balanced_accuracy_score,
        precision_recall_fscore_support, confusion_matrix,
    )

    holdout_metrics_rows = []
    holdout_predictions = pd.DataFrame()
    cm_df = pd.DataFrame()

    if accepted_matches is None or len(accepted_matches) == 0:
        return pd.DataFrame(), holdout_predictions, cm_df

    tmp = accepted_matches.copy()
    tmp["split_key"] = tmp.apply(
        lambda r: hash(str(r["audio_file"]) + "_" + str(r["official_turn_idx"])) % 10, axis=1
    )
    train = tmp[tmp["split_key"] < 7].copy()
    test = tmp[tmp["split_key"] >= 7].copy()

    if len(train) and len(test):
        train_map = infer_one_to_one_speaker_role_mapping_strict(train, params)
        if len(train_map):
            train_map = train_map.rename(columns={
                "probable_role": "predicted_role_from_train",
                "role_confidence": "train_role_confidence",
            })
            holdout_predictions = test.merge(
                train_map[["audio_file", "speaker_final", "predicted_role_from_train",
                           "train_role_confidence", "n_matches_total"]],
                on=["audio_file", "speaker_final"], how="left",
            )
            eval_df = holdout_predictions[holdout_predictions["predicted_role_from_train"].notna()].copy()

            if len(eval_df):
                y_true = eval_df["official_role"].astype(str)
                y_pred = eval_df["predicted_role_from_train"].astype(str)
                labels = ["AGENT", "CLIENT"]
                precision, recall, f1, support = precision_recall_fscore_support(
                    y_true, y_pred, labels=labels, zero_division=0
                )
                holdout_metrics_rows.extend([
                    {"metric": "holdout_accuracy", "value": accuracy_score(y_true, y_pred)},
                    {"metric": "holdout_balanced_accuracy", "value": balanced_accuracy_score(y_true, y_pred)},
                    {"metric": "holdout_n_eval_matches", "value": len(eval_df)},
                    {"metric": "holdout_n_audios_eval", "value": eval_df["audio_file"].nunique()},
                ])
                for role, p, r, f, s in zip(labels, precision, recall, f1, support):
                    holdout_metrics_rows.extend([
                        {"metric": f"precision_{role}", "value": p},
                        {"metric": f"recall_{role}", "value": r},
                        {"metric": f"f1_{role}", "value": f},
                        {"metric": f"support_{role}", "value": int(s)},
                    ])
                cm = confusion_matrix(y_true, y_pred, labels=labels)
                cm_df = pd.DataFrame(
                    cm, index=[f"true_{x}" for x in labels], columns=[f"pred_{x}" for x in labels]
                )

    return pd.DataFrame(holdout_metrics_rows), holdout_predictions, cm_df


# ============================================================
# RESUMEN DE MÉTRICAS TEXTUALES
# ============================================================

def build_textual_proxy_summary(
    df_segments, df_whisper, df_official_turns, text_alignment_candidates,
    accepted_matches, speaker_role_mapping_strict_matches, speaker_role_mapping,
    segment_level_proxy_textual,
):
    """Construye la tabla resumen de métricas de la fase 06."""
    summary_rows = [
        {"metric": "audios_diarized", "value": int(df_segments["audio_file"].nunique())},
        {"metric": "segments_diarized", "value": int(len(df_segments))},
        {"metric": "segments_with_whisper_text",
         "value": int(df_whisper["whisper_text"].fillna("").astype(str).str.strip().ne("").sum())},
        {"metric": "audios_with_official_turns_usable_text", "value": int(df_official_turns["audio_file"].nunique())},
        {"metric": "official_turns_usable_text", "value": int(len(df_official_turns))},
        {"metric": "alignment_candidates", "value": int(len(text_alignment_candidates))},
        {"metric": "accepted_textual_matches_strict", "value": int(len(accepted_matches))},
        {"metric": "audios_with_accepted_textual_matches_strict",
         "value": int(accepted_matches["audio_file"].nunique()) if len(accepted_matches) else 0},
        {"metric": "speakers_with_accepted_textual_matches_strict",
         "value": int(accepted_matches[["audio_file", "speaker_final"]].drop_duplicates().shape[0]) if len(accepted_matches) else 0},
        {"metric": "strict_match_based_speaker_role_mappings",
         "value": int(len(speaker_role_mapping_strict_matches)) if len(speaker_role_mapping_strict_matches) else 0},
        {"metric": "aggregated_speaker_role_mappings",
         "value": int(len(speaker_role_mapping)) if len(speaker_role_mapping) else 0},
        {"metric": "audios_labeled_by_aggregated_proxy",
         "value": int(segment_level_proxy_textual.loc[segment_level_proxy_textual["official_role_proxy"].notna(), "audio_file"].nunique())},
        {"metric": "segments_labeled_by_aggregated_proxy",
         "value": int(segment_level_proxy_textual["official_role_proxy"].notna().sum())},
    ]
    return pd.DataFrame(summary_rows)


# ============================================================
# MATCHES ACEPTADOS Y SENSIBILIDAD DE UMBRALES
# ============================================================

def select_accepted_matches(text_alignment_candidates):
    """Filtra los candidatos aceptados por defecto y los ordena."""
    if len(text_alignment_candidates) == 0:
        return pd.DataFrame()
    accepted = text_alignment_candidates[text_alignment_candidates["accepted_default"]].copy()
    return accepted.sort_values(
        ["audio_file", "official_turn_idx", "combined_score"],
        ascending=[True, True, False],
    )


def compute_threshold_sensitivity(text_alignment_candidates):
    """Tabla de sensibilidad de cobertura de matches por umbral y métrica."""
    threshold_rows = []
    if len(text_alignment_candidates) > 0:
        for metric in ["combined_score", "char_cosine", "token_containment"]:
            for thr in [0.50, 0.60, 0.70, 0.80, 0.90]:
                df_thr = text_alignment_candidates[text_alignment_candidates[metric].ge(thr)]
                threshold_rows.append({
                    "metric_used": metric, "threshold": thr,
                    "accepted_matches": len(df_thr),
                    "audios_covered": df_thr["audio_file"].nunique(),
                    "speakers_covered": df_thr[["audio_file", "speaker_final"]].drop_duplicates().shape[0],
                    "agent_matches": int((df_thr["official_role"] == "AGENT").sum()),
                    "client_matches": int((df_thr["official_role"] == "CLIENT").sum()),
                    "mean_combined_score": round(df_thr["combined_score"].mean(), 4) if len(df_thr) else np.nan,
                    "mean_char_cosine": round(df_thr["char_cosine"].mean(), 4) if len(df_thr) else np.nan,
                    "mean_token_containment": round(df_thr["token_containment"].mean(), 4) if len(df_thr) else np.nan,
                })
    return pd.DataFrame(threshold_rows)


# ============================================================
# GUARDADO LOCAL + SUBIDA FINAL A GCS
# ============================================================

def save_and_upload_outputs(
    result_frames, output_dir, gcs_client, gcs_prefix, upload_to_gcs=True,
):
    """
    Guarda los CSV de la fase 06 en local y (si upload_to_gcs) los sube a GCS.

    ``result_frames`` es un dict {nombre_archivo_sin_ext: DataFrame}. Los
    DataFrames vacíos o None se omiten. Devuelve el dict de rutas guardadas.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = {}

    for name, df in result_frames.items():
        if df is None or len(df) == 0:
            continue
        path = output_dir / f"{name}.csv"
        # 'transcripcion' es privada: nunca se exporta en las auditorías.
        df.drop(columns=["transcripcion"], errors="ignore").to_csv(path, index=False)
        saved[name] = path

    if upload_to_gcs and gcs_client is not None:
        for path in saved.values():
            upload_file(
                path, gcs_client, gcs_prefix=gcs_prefix,
                base_dir=output_dir, skip_unchanged=True,
            )

    return saved


# ============================================================
# RESTAURACIÓN / SALTO DE FASE COMPLETA
# ============================================================

def restore_phase_outputs_from_gcs(gcs_client, output_dir, gcs_prefix, data_dir):
    """Restaura desde GCS los outputs de la fase 06 (para poder saltar la fase)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    download_directory(
        local_dir=output_dir, gcs_prefix=gcs_prefix,
        gcs_client=gcs_client, base_dir=Path(data_dir),
    )


def phase_outputs_complete(segment_level_csv):
    """Indica si el output principal de la fase 06 ya existe localmente."""
    return Path(segment_level_csv).exists()


def ensure_phase_outputs_in_gcs(gcs_client, output_dir, gcs_prefix):
    """
    Sube a GCS todo lo que haya en la carpeta de outputs de la fase 06,
    sin depender de si la fase se ejecutó o se saltó en esta corrida.

    Cubre el caso de un compañero que corrió la fase en local pero nunca
    subió el resultado. ``skip_unchanged=True`` lo hace barato si ya estaba.
    """
    output_dir = Path(output_dir)
    if gcs_client is None:
        print("Publicación GCS omitida (sin cliente).")
        return
    if not output_dir.exists():
        print("No hay carpeta de outputs local que sincronizar.")
        return
    upload_directory(
        local_dir=output_dir, gcs_prefix=gcs_prefix,
        gcs_client=gcs_client, skip_unchanged=True,
    )
    print("Sincronización con GCS asegurada (incondicional).")
