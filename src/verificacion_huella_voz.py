"""Fase 09: verificación por pares e identificación open-set de huella de voz.

El módulo contiene la lógica reusable de preparación, evaluación pairwise,
construcción de perfiles e identificación open-set. La orquestación, los
controles y las visualizaciones quedan visibles en el Notebook 09.
"""

from __future__ import annotations

from datetime import datetime, timezone
from itertools import combinations
import hashlib
import json
from pathlib import Path
import random
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from src.config import (
    TRANSCRIPTION_ALL_SEGMENTS_CSV,
    TRANSCRIPTION_FINAL_SEGMENTS_CSV,
    TRANSCRIPTION_ROOT,
    VOICEPRINT_CHECKPOINT_DIR,
    VOICEPRINT_DIR,
    VOICEPRINT_FIGURES_DIR,
    VOICEPRINT_FINAL_SUMMARY_CSV,
    VOICEPRINT_OPEN_SET_SUMMARY_CSV,
    VOICEPRINT_ROLE_MAPPING_CSV,
    VOICEPRINT_SEGMENT_EMBEDDINGS_CSV,
    VOICEPRINT_SEGMENT_PROXY_CSV,
    VOICEPRINT_SUCCESS_JSON,
    ensure_phase09_directories,
)
from src.io_utils import (
    csv_is_usable,
    read_csv_robust,
    write_csv_atomic,
    write_json_atomic,
)


# ============================================================
# CONFIGURACIÓN CIENTÍFICA ORIGINAL
# ============================================================

RANDOM_SEED = 42

MIN_SEGMENT_DURATION_SEC = 1.50
MAX_SEGMENT_DURATION_SEC = 20.00
MAX_OVERLAP_RATIO = 0.05
MIN_RMS_DBFS = -40.0
MIN_WORDS_PER_SEGMENT = 0

MIN_SEGMENTS_PER_AUDIO_PERSON = 1
MIN_SECONDS_PER_AUDIO_PERSON = 1.50
MIN_SAMPLES_PER_IDENTITY = 2
MIN_TOTAL_SECONDS_PER_IDENTITY = 10.0

AGENT_TEST_SIZE = 0.30
USE_CLIENTS_IN_CALIBRATION = False

MAX_POSITIVE_PAIRS_PER_IDENTITY = 500
NEGATIVE_MULTIPLIER = 3
MAX_NEGATIVE_PAIRS = 200_000
THRESHOLD_STRATEGY = "youden"

OPEN_SET_MIN_SAMPLES_PER_IDENTITY = 3
OPEN_SET_MIN_TOTAL_SECONDS_PER_IDENTITY = 10.0
MIN_ENROLLMENT_SAMPLES = 2
MIN_QUERY_SAMPLES = 1
QUERY_FRACTION = 0.30
UNKNOWN_IDENTITY_FRACTION = 0.20
TEST_KNOWN_IDENTITY_FRACTION = 0.30
MIN_IDENTITIES_FOR_FORMAL_OPEN_SET = 6
OPEN_SET_THRESHOLD_STRATEGY = "eer"

BUILD_CLIENT_PROFILES = True
EMBEDDING_MODEL_LABEL = "embedding_precalculado_notebook_01"
NOTEBOOK_VERSION = "09_voiceprint_pairwise_open_set_v3"

TRANSCRIPTION_CANDIDATES = [
    TRANSCRIPTION_FINAL_SEGMENTS_CSV,
    TRANSCRIPTION_ALL_SEGMENTS_CSV,
    TRANSCRIPTION_ROOT / "transcribed_segments_final.csv",
]


# ============================================================
# OUTPUTS ORIGINALES DE LA FASE 09
# ============================================================

VOICEPRINT_SEGMENTS_CSV = VOICEPRINT_DIR / "voiceprint_segments_candidates.csv"
VOICEPRINT_SAMPLES_CSV = VOICEPRINT_DIR / "voiceprint_audio_person_samples.csv"
VOICEPRINT_IDENTITY_SUMMARY_CSV = VOICEPRINT_DIR / "voiceprint_identity_summary.csv"
VOICEPRINT_IDENTITY_SPLIT_CSV = VOICEPRINT_DIR / "voiceprint_identity_split.csv"

VOICEPRINT_PAIRS_CALIBRATION_CSV = VOICEPRINT_DIR / "voiceprint_pairs_calibration.csv"
VOICEPRINT_PAIRS_AGENT_TEST_CSV = VOICEPRINT_DIR / "voiceprint_pairs_test_agents.csv"
VOICEPRINT_PAIRS_CLIENT_TEST_CSV = VOICEPRINT_DIR / "voiceprint_pairs_test_clients.csv"
VOICEPRINT_PAIRWISE_THRESHOLD_CSV = VOICEPRINT_DIR / "voiceprint_threshold_summary.csv"
VOICEPRINT_PAIRWISE_METRICS_CSV = VOICEPRINT_DIR / "voiceprint_metrics_summary.csv"
VOICEPRINT_PAIRWISE_CONFUSION_CSV = VOICEPRINT_DIR / "voiceprint_confusion_matrices.csv"

VOICEPRINT_OPEN_SET_IDENTITY_SUMMARY_CSV = (
    VOICEPRINT_DIR / "voiceprint_identity_summary_open_set.csv"
)
VOICEPRINT_OPEN_SET_SPLIT_CSV = (
    VOICEPRINT_CHECKPOINT_DIR / "04_voiceprint_open_set_split.csv"
)
VOICEPRINT_IDENTITY_PARTITION_JSON = (
    VOICEPRINT_CHECKPOINT_DIR / "04_voiceprint_identity_partition.json"
)
VOICEPRINT_CALIBRATION_PROFILES_CSV = (
    VOICEPRINT_CHECKPOINT_DIR / "05_agent_profiles_calibration.csv"
)
VOICEPRINT_TEST_PROFILES_CSV = (
    VOICEPRINT_CHECKPOINT_DIR / "05_agent_profiles_test.csv"
)
VOICEPRINT_CALIBRATION_SCORES_CSV = (
    VOICEPRINT_CHECKPOINT_DIR / "06_calibration_query_profile_scores.csv"
)
VOICEPRINT_CALIBRATION_PREDICTIONS_CSV = (
    VOICEPRINT_CHECKPOINT_DIR / "06_calibration_top1_predictions.csv"
)
VOICEPRINT_TEST_KNOWN_SCORES_CSV = (
    VOICEPRINT_CHECKPOINT_DIR / "07_test_known_query_profile_scores.csv"
)
VOICEPRINT_TEST_KNOWN_PREDICTIONS_CSV = (
    VOICEPRINT_CHECKPOINT_DIR / "07_test_known_top1_predictions.csv"
)
VOICEPRINT_TEST_UNKNOWN_SCORES_CSV = (
    VOICEPRINT_CHECKPOINT_DIR / "07_test_unknown_query_profile_scores.csv"
)
VOICEPRINT_TEST_UNKNOWN_PREDICTIONS_CSV = (
    VOICEPRINT_CHECKPOINT_DIR / "07_test_unknown_top1_predictions.csv"
)

VOICEPRINT_THRESHOLDS_JSON = VOICEPRINT_DIR / "voiceprint_thresholds.json"
VOICEPRINT_OPEN_SET_PREDICTIONS_CSV = (
    VOICEPRINT_DIR / "open_set_identification_predictions.csv"
)
VOICEPRINT_IDENTIFICATION_METRICS_JSON = (
    VOICEPRINT_DIR / "open_set_identification_metrics.json"
)
VOICEPRINT_VERIFICATION_METRICS_CSV = (
    VOICEPRINT_DIR / "voiceprint_verification_metrics.csv"
)
VOICEPRINT_OPEN_SET_CONFUSION_CSV = (
    VOICEPRINT_DIR / "open_set_decision_confusion_matrix.csv"
)
VOICEPRINT_AGENT_PROFILES_CSV = VOICEPRINT_DIR / "agent_profiles_operational.csv"
VOICEPRINT_CLIENT_PROFILES_CSV = VOICEPRINT_DIR / "client_profiles_operational.csv"
VOICEPRINT_MODEL_METADATA_JSON = VOICEPRINT_DIR / "voiceprint_model_metadata.json"

FIG_SIMILARITY_CALIBRATION = (
    VOICEPRINT_FIGURES_DIR / "similarity_distribution_calibration_agents.png"
)
FIG_SIMILARITY_AGENT_TEST = (
    VOICEPRINT_FIGURES_DIR / "similarity_distribution_test_agents.png"
)
FIG_SIMILARITY_CLIENT_TEST = (
    VOICEPRINT_FIGURES_DIR / "similarity_distribution_test_clients.png"
)
FIG_ROC_CALIBRATION = VOICEPRINT_FIGURES_DIR / "roc_calibration_agents.png"
FIG_ROC_AGENT_TEST = VOICEPRINT_FIGURES_DIR / "roc_test_agents.png"
FIG_ROC_CLIENT_TEST = VOICEPRINT_FIGURES_DIR / "roc_test_clients.png"
FIG_IDENTITY_REPETITION = VOICEPRINT_FIGURES_DIR / "identity_repetition_by_role.png"
FIG_OPEN_SET_CALIBRATION = (
    VOICEPRINT_FIGURES_DIR / "voiceprint_calibration_similarity_distribution.png"
)
FIG_OPEN_SET_KNOWN_UNKNOWN = (
    VOICEPRINT_FIGURES_DIR / "open_set_best_similarity_known_vs_unknown.png"
)
FIG_OPEN_SET_ROC = VOICEPRINT_FIGURES_DIR / "voiceprint_calibration_roc.png"

PAIRWISE_OUTPUTS = [
    VOICEPRINT_SEGMENTS_CSV,
    VOICEPRINT_SAMPLES_CSV,
    VOICEPRINT_IDENTITY_SUMMARY_CSV,
    VOICEPRINT_IDENTITY_SPLIT_CSV,
    VOICEPRINT_PAIRS_CALIBRATION_CSV,
    VOICEPRINT_PAIRS_AGENT_TEST_CSV,
    VOICEPRINT_PAIRS_CLIENT_TEST_CSV,
    VOICEPRINT_PAIRWISE_THRESHOLD_CSV,
    VOICEPRINT_PAIRWISE_METRICS_CSV,
    VOICEPRINT_PAIRWISE_CONFUSION_CSV,
    VOICEPRINT_FINAL_SUMMARY_CSV,
]

OPEN_SET_REQUIRED_OUTPUTS = [
    VOICEPRINT_AGENT_PROFILES_CSV,
    VOICEPRINT_THRESHOLDS_JSON,
    VOICEPRINT_MODEL_METADATA_JSON,
    VOICEPRINT_OPEN_SET_PREDICTIONS_CSV,
    VOICEPRINT_IDENTIFICATION_METRICS_JSON,
    VOICEPRINT_VERIFICATION_METRICS_CSV,
    VOICEPRINT_OPEN_SET_CONFUSION_CSV,
    VOICEPRINT_OPEN_SET_SUMMARY_CSV,
    VOICEPRINT_SUCCESS_JSON,
]


# ============================================================
# UTILIDADES GENERALES
# ============================================================


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_hash_dict(payload: dict) -> str:
    serialized = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def seed_everything(seed: int = RANDOM_SEED) -> None:
    np.random.seed(seed)
    random.seed(seed)


def save_dataframe_checkpoint(df: pd.DataFrame, path: Path) -> Path:
    path = Path(path)
    write_csv_atomic(df, path)
    return path


def save_json_checkpoint(payload: dict, path: Path) -> Path:
    path = Path(path)
    write_json_atomic(payload, path)
    return path


def load_dataframe_checkpoint(
    path: Path,
    force: bool = False,
    required_columns: Sequence[str] | None = None,
) -> pd.DataFrame | None:
    if force:
        return None
    required_columns = list(required_columns or [])
    if not csv_is_usable(Path(path), required_columns=required_columns):
        return None
    return pd.read_csv(path)


def load_json_checkpoint(path: Path, force: bool = False) -> dict | None:
    path = Path(path)
    if force or not path.exists() or path.stat().st_size == 0:
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def read_csv_required(path: Path, name: str) -> pd.DataFrame:
    path = Path(path)
    if not csv_is_usable(path):
        raise FileNotFoundError(f"No existe o no es legible {name}: {path}")
    return pd.read_csv(path)


def read_first_existing(
    candidates: Iterable[Path],
    required: bool = False,
) -> tuple[pd.DataFrame, Path | None]:
    for candidate in candidates:
        path = Path(candidate)
        if csv_is_usable(path):
            return pd.read_csv(path), path
    if required:
        raise FileNotFoundError(
            "No se encontró ninguno de los archivos candidatos: "
            + ", ".join(str(Path(p)) for p in candidates)
        )
    return pd.DataFrame(), None


def get_embedding_columns(df: pd.DataFrame) -> list[str]:
    return sorted(column for column in df.columns if column.startswith("emb_"))


def normalize_audio_key(value):
    if pd.isna(value):
        return pd.NA
    text = str(value).strip()
    if not text:
        return pd.NA
    return Path(text).stem


def normalize_identifier(value):
    if pd.isna(value):
        return pd.NA
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "<na>"}:
        return pd.NA
    return text


def normalize_identifier_series(series: pd.Series) -> pd.Series:
    return series.map(normalize_identifier).astype("string")


def choose_existing_col(
    df: pd.DataFrame,
    candidates: Sequence[str],
    required: bool = False,
    label: str = "columna",
) -> str | None:
    for column in candidates:
        if column in df.columns:
            return column
    if required:
        raise ValueError(
            f"No se encontró {label}. Candidatas: {list(candidates)}. "
            f"Columnas disponibles: {list(df.columns)[:60]}"
        )
    return None


def add_time_key(
    df: pd.DataFrame,
    start_col: str = "start",
    end_col: str = "end",
    decimals: int = 3,
) -> pd.DataFrame:
    output = df.copy()
    if start_col in output.columns and end_col in output.columns:
        start = pd.to_numeric(output[start_col], errors="coerce").round(decimals)
        end = pd.to_numeric(output[end_col], errors="coerce").round(decimals)
        output["time_key"] = start.astype("string") + "_" + end.astype("string")
    return output


def l2_normalize_matrix(matrix) -> np.ndarray:
    array = np.asarray(matrix, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms <= 1e-12] = 1.0
    return array / norms


def cosine_from_normalized(vector_a, vector_b) -> float:
    return float(np.dot(vector_a, vector_b))


def cosine_similarity_matrix(matrix_a, matrix_b) -> np.ndarray:
    """Similitud coseno matricial usada por la evaluación open-set."""
    normalized_a = l2_normalize_matrix(matrix_a)
    normalized_b = l2_normalize_matrix(matrix_b)
    return normalized_a @ normalized_b.T


def voiceprint_outputs_complete() -> bool:
    manifest = load_json_checkpoint(VOICEPRINT_SUCCESS_JSON)
    if manifest is None or manifest.get("status") != "completed":
        return False
    reusable_outputs = PAIRWISE_OUTPUTS + [
        VOICEPRINT_OPEN_SET_IDENTITY_SUMMARY_CSV,
        VOICEPRINT_OPEN_SET_SPLIT_CSV,
        VOICEPRINT_IDENTITY_PARTITION_JSON,
        VOICEPRINT_CALIBRATION_PROFILES_CSV,
        VOICEPRINT_TEST_PROFILES_CSV,
        VOICEPRINT_CALIBRATION_SCORES_CSV,
        VOICEPRINT_CALIBRATION_PREDICTIONS_CSV,
        VOICEPRINT_TEST_KNOWN_SCORES_CSV,
        VOICEPRINT_TEST_KNOWN_PREDICTIONS_CSV,
        VOICEPRINT_TEST_UNKNOWN_SCORES_CSV,
        VOICEPRINT_TEST_UNKNOWN_PREDICTIONS_CSV,
    ] + OPEN_SET_REQUIRED_OUTPUTS
    return all(
        Path(path).exists() and Path(path).stat().st_size > 0
        for path in reusable_outputs
    )


def load_voiceprint_outputs() -> dict[str, object]:
    return {
        "segments": read_csv_robust(VOICEPRINT_SEGMENTS_CSV),
        "samples": read_csv_robust(VOICEPRINT_SAMPLES_CSV),
        "identity_summary": read_csv_robust(VOICEPRINT_IDENTITY_SUMMARY_CSV),
        "identity_split": read_csv_robust(VOICEPRINT_IDENTITY_SPLIT_CSV),
        "pairwise_metrics": read_csv_robust(VOICEPRINT_PAIRWISE_METRICS_CSV),
        "pairwise_confusion": read_csv_robust(VOICEPRINT_PAIRWISE_CONFUSION_CSV),
        "final_summary": read_csv_robust(VOICEPRINT_FINAL_SUMMARY_CSV),
        "open_set_summary": read_csv_robust(VOICEPRINT_OPEN_SET_SUMMARY_CSV),
        "open_set_predictions": read_csv_robust(VOICEPRINT_OPEN_SET_PREDICTIONS_CSV),
        "verification_metrics": read_csv_robust(VOICEPRINT_VERIFICATION_METRICS_CSV),
        "open_set_confusion": read_csv_robust(VOICEPRINT_OPEN_SET_CONFUSION_CSV),
        "manifest": load_json_checkpoint(VOICEPRINT_SUCCESS_JSON) or {},
    }


# ============================================================
# PREPARACIÓN DE SEGMENTOS Y MUESTRAS
# ============================================================


def load_voiceprint_inputs() -> dict[str, object]:
    embeddings = read_csv_required(
        VOICEPRINT_SEGMENT_EMBEDDINGS_CSV,
        "embeddings de segmentos",
    )
    role_mapping = read_csv_required(
        VOICEPRINT_ROLE_MAPPING_CSV,
        "mapping speaker-rol proxy",
    )
    segment_proxy = (
        pd.read_csv(VOICEPRINT_SEGMENT_PROXY_CSV)
        if csv_is_usable(VOICEPRINT_SEGMENT_PROXY_CSV)
        else pd.DataFrame()
    )
    transcriptions, transcription_path = read_first_existing(
        TRANSCRIPTION_CANDIDATES,
        required=False,
    )
    embedding_columns = get_embedding_columns(embeddings)
    if not embedding_columns:
        raise ValueError(
            "No se detectaron columnas emb_0000, emb_0001, etc. "
            "Revisa el output del Notebook 01."
        )
    return {
        "embeddings": embeddings,
        "role_mapping": role_mapping,
        "segment_proxy": segment_proxy,
        "transcriptions": transcriptions,
        "transcription_path": transcription_path,
        "embedding_columns": embedding_columns,
    }


def normalize_voiceprint_inputs(
    embeddings: pd.DataFrame,
    role_mapping: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, str]]:
    df_embeddings = embeddings.copy()
    if "audio_file" not in df_embeddings.columns and "audio_stem" in df_embeddings.columns:
        df_embeddings["audio_file"] = df_embeddings["audio_stem"].astype("string") + ".wav"
    if "audio_stem" not in df_embeddings.columns and "audio_file" in df_embeddings.columns:
        df_embeddings["audio_stem"] = df_embeddings["audio_file"].map(normalize_audio_key)
    if "audio_file" not in df_embeddings.columns:
        raise ValueError("Los embeddings no contienen audio_file ni audio_stem.")
    df_embeddings["audio_key"] = df_embeddings["audio_file"].map(normalize_audio_key)

    speaker_col_embeddings = choose_existing_col(
        df_embeddings,
        ["speaker_final", "speaker_relabel", "speaker", "label"],
        required=True,
        label="speaker en embeddings",
    )
    if speaker_col_embeddings != "speaker_final":
        df_embeddings["speaker_final"] = df_embeddings[speaker_col_embeddings]
    df_embeddings["speaker_final"] = normalize_identifier_series(df_embeddings["speaker_final"])

    df_role_mapping = role_mapping.copy()
    if "audio_file" not in df_role_mapping.columns and "audio_stem" in df_role_mapping.columns:
        df_role_mapping["audio_file"] = df_role_mapping["audio_stem"].astype("string") + ".wav"
    if "audio_stem" not in df_role_mapping.columns and "audio_file" in df_role_mapping.columns:
        df_role_mapping["audio_stem"] = df_role_mapping["audio_file"].map(normalize_audio_key)
    if "audio_file" not in df_role_mapping.columns:
        raise ValueError("El mapping de roles no contiene audio_file ni audio_stem.")
    df_role_mapping["audio_key"] = df_role_mapping["audio_file"].map(normalize_audio_key)

    role_col = choose_existing_col(
        df_role_mapping,
        ["probable_role", "assigned_role", "official_role_proxy", "role_proxy", "role"],
        required=True,
        label="rol proxy en mapping",
    )
    if role_col != "role_proxy":
        df_role_mapping["role_proxy"] = df_role_mapping[role_col]
    df_role_mapping["role_proxy"] = (
        df_role_mapping["role_proxy"].astype("string").str.upper().str.strip()
    )

    speaker_col_mapping = choose_existing_col(
        df_role_mapping,
        ["speaker_final", "speaker", "speaker_label"],
        required=True,
        label="speaker en mapping",
    )
    if speaker_col_mapping != "speaker_final":
        df_role_mapping["speaker_final"] = df_role_mapping[speaker_col_mapping]
    df_role_mapping["speaker_final"] = normalize_identifier_series(
        df_role_mapping["speaker_final"]
    )

    for id_column in ["agent_hash", "customer_hash"]:
        if id_column in df_role_mapping.columns:
            df_role_mapping[id_column] = normalize_identifier_series(
                df_role_mapping[id_column]
            )

    mapping_columns = [
        "audio_key",
        "audio_file",
        "speaker_final",
        "role_proxy",
        "role_confidence",
        "proxy_confidence",
        "role_mapping_status",
        "agent_hash",
        "customer_hash",
        "brand_ds",
        "duration_min",
    ]
    mapping_columns = [column for column in mapping_columns if column in df_role_mapping.columns]
    mapping_small = (
        df_role_mapping[mapping_columns]
        .drop_duplicates(subset=["audio_key", "speaker_final"], keep="first")
        .copy()
    )
    selected_columns = {
        "speaker_embeddings": str(speaker_col_embeddings),
        "speaker_mapping": str(speaker_col_mapping),
        "role_mapping": str(role_col),
    }
    return df_embeddings, df_role_mapping, mapping_small, selected_columns


def merge_embeddings_with_roles(
    embeddings: pd.DataFrame,
    mapping_small: pd.DataFrame,
    transcriptions: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    voice_segments = embeddings.merge(
        mapping_small,
        on=["audio_key", "speaker_final"],
        how="left",
        suffixes=("", "_mapping"),
    )
    voice_segments["role_proxy"] = (
        voice_segments["role_proxy"].astype("string").str.upper().str.strip()
    )

    agent_ids = (
        voice_segments["agent_hash"]
        if "agent_hash" in voice_segments.columns
        else pd.Series(pd.NA, index=voice_segments.index, dtype="string")
    )
    customer_ids = (
        voice_segments["customer_hash"]
        if "customer_hash" in voice_segments.columns
        else pd.Series(pd.NA, index=voice_segments.index, dtype="string")
    )
    person_id = pd.Series(pd.NA, index=voice_segments.index, dtype="string")
    person_id.loc[voice_segments["role_proxy"].eq("AGENT")] = agent_ids
    person_id.loc[voice_segments["role_proxy"].eq("CLIENT")] = customer_ids
    voice_segments["person_id"] = normalize_identifier_series(person_id)

    if transcriptions is not None and not transcriptions.empty:
        transcription = transcriptions.copy()
        if "audio_file" not in transcription.columns and "audio_stem" in transcription.columns:
            transcription["audio_file"] = transcription["audio_stem"].astype("string") + ".wav"
        if "audio_file" in transcription.columns:
            transcription["audio_key"] = transcription["audio_file"].map(normalize_audio_key)
            voice_segments = add_time_key(voice_segments)
            transcription = add_time_key(transcription)
            transcription_columns = [
                "audio_key",
                "time_key",
                "speaker_final",
                "n_words",
                "n_chars",
                "transcription_status",
            ]
            transcription_columns = [
                column for column in transcription_columns if column in transcription.columns
            ]
            if {"audio_key", "time_key"}.issubset(transcription_columns):
                merge_keys = ["audio_key", "time_key"]
                if (
                    "speaker_final" in transcription_columns
                    and "speaker_final" in voice_segments.columns
                ):
                    transcription["speaker_final"] = normalize_identifier_series(
                        transcription["speaker_final"]
                    )
                    merge_keys.append("speaker_final")
                voice_segments = voice_segments.merge(
                    transcription[transcription_columns].drop_duplicates(merge_keys),
                    on=merge_keys,
                    how="left",
                    suffixes=("", "_transcription"),
                )

    coverage = (
        voice_segments
        .assign(has_person=voice_segments["person_id"].notna())
        .groupby("role_proxy", dropna=False)
        .agg(
            n_segments=("audio_key", "size"),
            n_with_person=("has_person", "sum"),
            n_audios=("audio_key", "nunique"),
            n_persons=("person_id", "nunique"),
        )
        .reset_index()
    )
    return voice_segments, coverage


def filter_voiceprint_segments(
    voice_segments: pd.DataFrame,
    embedding_columns: Sequence[str],
) -> pd.DataFrame:
    output = voice_segments.copy()
    for column in ["duration", "overlap_ratio", "rms_dbfs", "n_words"]:
        if column in output.columns:
            output[column] = pd.to_numeric(output[column], errors="coerce")

    mask = output["role_proxy"].isin(["AGENT", "CLIENT"])
    mask &= output["person_id"].notna()
    if "duration" in output.columns:
        mask &= output["duration"].between(
            MIN_SEGMENT_DURATION_SEC,
            MAX_SEGMENT_DURATION_SEC,
            inclusive="both",
        )
    if "overlap_ratio" in output.columns:
        mask &= output["overlap_ratio"].fillna(0) <= MAX_OVERLAP_RATIO
    if "rms_dbfs" in output.columns:
        mask &= output["rms_dbfs"].fillna(-999) >= MIN_RMS_DBFS
    if "valid_export" in output.columns:
        valid_export = output["valid_export"]
        if valid_export.dtype == object or isinstance(valid_export.dtype, pd.StringDtype):
            valid_export = valid_export.astype("string").str.lower().isin(
                ["true", "1", "yes", "ok"]
            )
        mask &= valid_export.fillna(False).astype(bool)
    if MIN_WORDS_PER_SEGMENT > 0 and "n_words" in output.columns:
        mask &= output["n_words"].fillna(0) >= MIN_WORDS_PER_SEGMENT

    filtered = output.loc[mask].copy().reset_index(drop=True)
    if filtered.empty:
        return filtered
    filtered[list(embedding_columns)] = l2_normalize_matrix(
        filtered[list(embedding_columns)].to_numpy(dtype=np.float32)
    )
    return filtered


def build_audio_person_samples(
    voiceprint_segments: pd.DataFrame,
    embedding_columns: Sequence[str],
) -> pd.DataFrame:
    group_columns = ["person_id", "role_proxy", "audio_key"]
    for column in ["agent_hash", "customer_hash", "brand_ds"]:
        if column in voiceprint_segments.columns:
            group_columns.append(column)

    aggregation = {"speaker_final": "first"}
    if "audio_file" in voiceprint_segments.columns:
        aggregation["audio_file"] = "first"
    if "duration" in voiceprint_segments.columns:
        aggregation["duration"] = ["sum", "mean", "count"]
    else:
        aggregation["person_id"] = "size"
    if "overlap_ratio" in voiceprint_segments.columns:
        aggregation["overlap_ratio"] = "mean"
    if "rms_dbfs" in voiceprint_segments.columns:
        aggregation["rms_dbfs"] = "mean"
    if "n_words" in voiceprint_segments.columns:
        aggregation["n_words"] = "sum"

    metadata = voiceprint_segments.groupby(group_columns, dropna=False).agg(aggregation)
    metadata.columns = [
        "_".join(str(item) for item in column if str(item)).strip("_")
        for column in metadata.columns
    ]
    metadata = metadata.reset_index().rename(
        columns={
            "duration_sum": "sample_duration_sec",
            "duration_mean": "mean_segment_duration_sec",
            "duration_count": "n_segments",
            "person_id_size": "n_segments",
            "overlap_ratio_mean": "mean_overlap_ratio",
            "rms_dbfs_mean": "mean_rms_dbfs",
            "n_words_sum": "sample_n_words",
            "speaker_final_first": "speaker_final",
            "audio_file_first": "audio_file",
        }
    )

    embedding_matrix = (
        voiceprint_segments.groupby(group_columns, dropna=False)[list(embedding_columns)]
        .mean()
        .reset_index()
    )
    samples = metadata.merge(embedding_matrix, on=group_columns, how="left")
    samples[list(embedding_columns)] = l2_normalize_matrix(
        samples[list(embedding_columns)].to_numpy(dtype=np.float32)
    )

    sample_mask = pd.Series(True, index=samples.index)
    if "n_segments" in samples.columns:
        sample_mask &= samples["n_segments"].fillna(0) >= MIN_SEGMENTS_PER_AUDIO_PERSON
    if "sample_duration_sec" in samples.columns:
        sample_mask &= (
            samples["sample_duration_sec"].fillna(0) >= MIN_SECONDS_PER_AUDIO_PERSON
        )
    samples = samples.loc[sample_mask].copy().reset_index(drop=True)
    samples["sample_id"] = [f"S{index:06d}" for index in range(len(samples))]
    return samples


def summarize_identities(samples: pd.DataFrame) -> pd.DataFrame:
    aggregation = {"sample_id": "count", "audio_key": "nunique"}
    if "n_segments" in samples.columns:
        aggregation["n_segments"] = "sum"
    if "sample_duration_sec" in samples.columns:
        aggregation["sample_duration_sec"] = "sum"
    summary = (
        samples.groupby(["role_proxy", "person_id"], dropna=False)
        .agg(aggregation)
        .reset_index()
        .rename(
            columns={
                "sample_id": "n_samples",
                "audio_key": "n_audios",
                "n_segments": "total_segments",
                "sample_duration_sec": "total_duration_sec",
            }
        )
    )
    if "total_duration_sec" not in summary.columns:
        summary["total_duration_sec"] = np.nan
    if "total_segments" not in summary.columns:
        summary["total_segments"] = np.nan
    summary["eligible_verification"] = (
        summary["n_samples"].ge(MIN_SAMPLES_PER_IDENTITY)
        & summary["total_duration_sec"]
        .fillna(MIN_TOTAL_SECONDS_PER_IDENTITY)
        .ge(MIN_TOTAL_SECONDS_PER_IDENTITY)
    )
    return summary


# ============================================================
# VERIFICACIÓN PAIRWISE
# ============================================================


def split_pairwise_identities(
    identity_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    eligible = identity_summary[identity_summary["eligible_verification"].astype(bool)].copy()
    eligible_agents = (
        eligible.loc[eligible["role_proxy"].eq("AGENT"), "person_id"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    eligible_clients = (
        eligible.loc[eligible["role_proxy"].eq("CLIENT"), "person_id"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    if len(eligible_agents) >= 4:
        agent_calibration_ids, agent_test_ids = train_test_split(
            eligible_agents,
            test_size=AGENT_TEST_SIZE,
            random_state=RANDOM_SEED,
        )
    elif len(eligible_agents) >= 2:
        agent_calibration_ids = eligible_agents[:-1]
        agent_test_ids = eligible_agents[-1:]
    else:
        agent_calibration_ids = eligible_agents
        agent_test_ids = []

    if USE_CLIENTS_IN_CALIBRATION and len(eligible_clients) >= 4:
        client_calibration_ids, client_test_ids = train_test_split(
            eligible_clients,
            test_size=0.50,
            random_state=RANDOM_SEED,
        )
    else:
        client_calibration_ids = []
        client_test_ids = eligible_clients

    rows = []
    for person_id in agent_calibration_ids:
        rows.append({"person_id": person_id, "role_proxy": "AGENT", "split": "calibration"})
    for person_id in agent_test_ids:
        rows.append({"person_id": person_id, "role_proxy": "AGENT", "split": "test_agent"})
    for person_id in client_calibration_ids:
        rows.append({"person_id": person_id, "role_proxy": "CLIENT", "split": "calibration"})
    for person_id in client_test_ids:
        rows.append({"person_id": person_id, "role_proxy": "CLIENT", "split": "test_client"})
    split = pd.DataFrame(rows, columns=["person_id", "role_proxy", "split"])

    calibration_set = set(split.loc[split["split"].eq("calibration"), "person_id"].astype(str))
    test_set = set(split.loc[split["split"].ne("calibration"), "person_id"].astype(str))
    if calibration_set.intersection(test_set):
        raise ValueError("Hay fuga de identidad entre calibración y test.")

    partition = {
        "eligible_agents": eligible_agents,
        "eligible_clients": eligible_clients,
        "agent_calibration_ids": list(agent_calibration_ids),
        "agent_test_ids": list(agent_test_ids),
        "client_calibration_ids": list(client_calibration_ids),
        "client_test_ids": list(client_test_ids),
    }
    return split, partition


def partition_from_pairwise_split(
    identity_summary: pd.DataFrame,
    identity_split: pd.DataFrame,
) -> dict[str, list[str]]:
    eligible = identity_summary[
        identity_summary["eligible_verification"].astype(bool)
    ].copy()
    return {
        "eligible_agents": (
            eligible.loc[eligible["role_proxy"].eq("AGENT"), "person_id"]
            .dropna().astype(str).unique().tolist()
        ),
        "eligible_clients": (
            eligible.loc[eligible["role_proxy"].eq("CLIENT"), "person_id"]
            .dropna().astype(str).unique().tolist()
        ),
        "agent_calibration_ids": (
            identity_split.loc[
                identity_split["role_proxy"].eq("AGENT")
                & identity_split["split"].eq("calibration"),
                "person_id",
            ].dropna().astype(str).tolist()
        ),
        "agent_test_ids": (
            identity_split.loc[
                identity_split["role_proxy"].eq("AGENT")
                & identity_split["split"].eq("test_agent"),
                "person_id",
            ].dropna().astype(str).tolist()
        ),
        "client_calibration_ids": (
            identity_split.loc[
                identity_split["role_proxy"].eq("CLIENT")
                & identity_split["split"].eq("calibration"),
                "person_id",
            ].dropna().astype(str).tolist()
        ),
        "client_test_ids": (
            identity_split.loc[
                identity_split["role_proxy"].eq("CLIENT")
                & identity_split["split"].eq("test_client"),
                "person_id",
            ].dropna().astype(str).tolist()
        ),
    }


def subset_samples_by_ids(
    samples: pd.DataFrame,
    identity_ids: Sequence[str],
    role: str | None = None,
) -> pd.DataFrame:
    identity_set = {str(value) for value in identity_ids}
    subset = samples[samples["person_id"].astype(str).isin(identity_set)].copy()
    if role is not None:
        subset = subset[subset["role_proxy"].eq(role)].copy()
    return subset.reset_index(drop=True)


def build_pairwise_sample_sets(
    samples: pd.DataFrame,
    partition: dict[str, list[str]],
) -> dict[str, pd.DataFrame]:
    agent_calibration = subset_samples_by_ids(
        samples,
        partition["agent_calibration_ids"],
        role="AGENT",
    )
    agent_test = subset_samples_by_ids(
        samples,
        partition["agent_test_ids"],
        role="AGENT",
    )
    client_test = subset_samples_by_ids(
        samples,
        partition["client_test_ids"],
        role="CLIENT",
    )
    if USE_CLIENTS_IN_CALIBRATION:
        client_calibration = subset_samples_by_ids(
            samples,
            partition["client_calibration_ids"],
            role="CLIENT",
        )
        calibration = pd.concat(
            [agent_calibration, client_calibration],
            ignore_index=True,
        )
    else:
        calibration = agent_calibration.copy()
    return {
        "calibration": calibration,
        "agent_test": agent_test,
        "client_test": client_test,
    }


def build_verification_pairs(
    samples: pd.DataFrame,
    embedding_columns: Sequence[str],
    max_positive_pairs_per_identity: int = MAX_POSITIVE_PAIRS_PER_IDENTITY,
    negative_multiplier: int = NEGATIVE_MULTIPLIER,
    max_negative_pairs: int = MAX_NEGATIVE_PAIRS,
    random_state: int = RANDOM_SEED,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    dataframe = samples.reset_index(drop=True).copy()
    if dataframe.empty or dataframe["person_id"].nunique() < 2:
        return pd.DataFrame()

    matrix = l2_normalize_matrix(dataframe[list(embedding_columns)].to_numpy())
    person_values = dataframe["person_id"].astype(str).to_numpy()
    role_values = (
        dataframe["role_proxy"].astype(str).to_numpy()
        if "role_proxy" in dataframe.columns
        else np.array(["UNKNOWN"] * len(dataframe))
    )
    sample_ids = dataframe["sample_id"].astype(str).to_numpy()
    audio_values = (
        dataframe["audio_key"].astype(str).to_numpy()
        if "audio_key" in dataframe.columns
        else np.array([""] * len(dataframe))
    )

    rows: list[dict] = []
    for _, indices in dataframe.groupby("person_id").indices.items():
        indices = list(indices)
        if len(indices) < 2:
            continue
        positive_pairs = [
            (index_a, index_b)
            for index_a, index_b in combinations(indices, 2)
            if audio_values[index_a] != audio_values[index_b]
        ]
        if not positive_pairs:
            positive_pairs = list(combinations(indices, 2))
        if len(positive_pairs) > max_positive_pairs_per_identity:
            selected = rng.choice(
                len(positive_pairs),
                size=max_positive_pairs_per_identity,
                replace=False,
            )
            positive_pairs = [positive_pairs[index] for index in selected]
        for index_a, index_b in positive_pairs:
            rows.append(
                {
                    "sample_id_a": sample_ids[index_a],
                    "sample_id_b": sample_ids[index_b],
                    "person_id_a": person_values[index_a],
                    "person_id_b": person_values[index_b],
                    "role_a": role_values[index_a],
                    "role_b": role_values[index_b],
                    "audio_a": audio_values[index_a],
                    "audio_b": audio_values[index_b],
                    "same_identity": 1,
                    "similarity": cosine_from_normalized(
                        matrix[index_a],
                        matrix[index_b],
                    ),
                }
            )

    positive_count = len(rows)
    if positive_count == 0:
        return pd.DataFrame(rows)
    target_negative = min(max_negative_pairs, positive_count * negative_multiplier)
    negative_rows = []
    attempts = 0
    max_attempts = max(target_negative * 20, 10_000)
    while len(negative_rows) < target_negative and attempts < max_attempts:
        attempts += 1
        index_a, index_b = rng.choice(len(dataframe), size=2, replace=False)
        if person_values[index_a] == person_values[index_b]:
            continue
        negative_rows.append(
            {
                "sample_id_a": sample_ids[index_a],
                "sample_id_b": sample_ids[index_b],
                "person_id_a": person_values[index_a],
                "person_id_b": person_values[index_b],
                "role_a": role_values[index_a],
                "role_b": role_values[index_b],
                "audio_a": audio_values[index_a],
                "audio_b": audio_values[index_b],
                "same_identity": 0,
                "similarity": cosine_from_normalized(
                    matrix[index_a],
                    matrix[index_b],
                ),
            }
        )
    rows.extend(negative_rows)
    return pd.DataFrame(rows)


def compute_eer_threshold(y_true, scores) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_true, scores)
    false_negative_rate = 1 - true_positive_rate
    index = int(np.nanargmin(np.abs(false_positive_rate - false_negative_rate)))
    eer = float((false_positive_rate[index] + false_negative_rate[index]) / 2)
    return eer, float(thresholds[index]), false_positive_rate, true_positive_rate, thresholds


def choose_threshold_from_pairs(
    pairs: pd.DataFrame,
    strategy: str = THRESHOLD_STRATEGY,
) -> dict[str, object]:
    y_true = pairs["same_identity"].astype(int).to_numpy()
    scores = pairs["similarity"].astype(float).to_numpy()
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_true, scores)
    false_negative_rate = 1 - true_positive_rate
    if strategy == "eer":
        selected_index = int(
            np.nanargmin(np.abs(false_positive_rate - false_negative_rate))
        )
    else:
        selected_index = int(np.nanargmax(true_positive_rate - false_positive_rate))
    eer_index = int(np.nanargmin(np.abs(false_positive_rate - false_negative_rate)))
    return {
        "threshold": float(thresholds[selected_index]),
        "strategy": strategy,
        "eer": float(
            (false_positive_rate[eer_index] + false_negative_rate[eer_index]) / 2
        ),
        "eer_threshold": float(thresholds[eer_index]),
        "fpr": false_positive_rate,
        "tpr": true_positive_rate,
        "thresholds": thresholds,
    }


def evaluate_pairs(
    pairs: pd.DataFrame,
    threshold: float | None = None,
    label: str = "dataset",
) -> dict[str, object]:
    if pairs is None or pairs.empty:
        return {
            "dataset": label,
            "n_pairs": 0,
            "n_positive": 0,
            "n_negative": 0,
            "auc": np.nan,
            "eer": np.nan,
            "threshold": threshold,
            "accuracy": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "f1": np.nan,
        }
    y_true = pairs["same_identity"].astype(int).to_numpy()
    scores = pairs["similarity"].astype(float).to_numpy()
    if len(np.unique(y_true)) < 2:
        auc = np.nan
        eer = np.nan
    else:
        auc = float(roc_auc_score(y_true, scores))
        eer, _, _, _, _ = compute_eer_threshold(y_true, scores)
    if threshold is None:
        threshold = (
            float(choose_threshold_from_pairs(pairs)["threshold"])
            if len(np.unique(y_true)) == 2
            else np.nan
        )
    if pd.isna(threshold):
        accuracy = precision = recall = f1 = np.nan
    else:
        predictions = (scores >= float(threshold)).astype(int)
        accuracy = float(accuracy_score(y_true, predictions))
        precision = float(precision_score(y_true, predictions, zero_division=0))
        recall = float(recall_score(y_true, predictions, zero_division=0))
        f1 = float(f1_score(y_true, predictions, zero_division=0))
    return {
        "dataset": label,
        "n_pairs": int(len(pairs)),
        "n_positive": int((y_true == 1).sum()),
        "n_negative": int((y_true == 0).sum()),
        "auc": auc,
        "eer": eer,
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def confusion_table(
    pairs: pd.DataFrame,
    threshold: float,
    label: str,
) -> pd.DataFrame:
    if pairs is None or pairs.empty or pd.isna(threshold):
        return pd.DataFrame(
            columns=["true_label", "pred_genuine", "pred_impostor", "dataset"]
        )
    y_true = pairs["same_identity"].astype(int).to_numpy()
    y_pred = (pairs["similarity"].astype(float).to_numpy() >= threshold).astype(int)
    matrix = confusion_matrix(y_true, y_pred, labels=[1, 0])
    return (
        pd.DataFrame(
            matrix,
            index=["true_genuine", "true_impostor"],
            columns=["pred_genuine", "pred_impostor"],
        )
        .assign(dataset=label)
        .reset_index()
        .rename(columns={"index": "true_label"})
    )


def build_pairwise_summary(
    embeddings_count: int,
    candidates: pd.DataFrame,
    samples: pd.DataFrame,
    identity_summary: pd.DataFrame,
    partition: dict[str, list[str]],
    threshold: float,
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    rows = [
        {"métrica": "Segmentos con embeddings", "valor": embeddings_count},
        {"métrica": "Segmentos candidatos para huella", "valor": len(candidates)},
        {"métrica": "Muestras audio-persona", "valor": len(samples)},
        {"métrica": "Identidades totales", "valor": identity_summary["person_id"].nunique()},
        {
            "métrica": "Identidades elegibles",
            "valor": int(identity_summary["eligible_verification"].sum()),
        },
        {"métrica": "Agentes elegibles", "valor": len(partition["eligible_agents"])},
        {
            "métrica": "Clientes repetidos elegibles",
            "valor": len(partition["eligible_clients"]),
        },
        {"métrica": "Umbral calibrado", "valor": round(float(threshold), 4)},
    ]
    for _, metric in metrics.iterrows():
        for metric_name, column in [("AUC", "auc"), ("EER", "eer"), ("F1", "f1")]:
            value = metric.get(column)
            rows.append(
                {
                    "métrica": f"{metric_name} - {metric['dataset']}",
                    "valor": None if pd.isna(value) else round(float(value), 4),
                }
            )
    return pd.DataFrame(rows)


# ============================================================
# IDENTIFICACIÓN OPEN-SET
# ============================================================


def prepare_open_set_identity_summary(samples: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    open_set_samples = samples.copy()
    open_set_samples["source_identity_id"] = normalize_identifier_series(
        open_set_samples["person_id"]
    )
    summary = (
        open_set_samples
        .groupby(["role_proxy", "person_id", "source_identity_id"], dropna=False)
        .agg(
            n_samples=("sample_id", "size"),
            n_calls=("audio_key", "nunique"),
            total_segments=("n_segments", "sum"),
            total_duration_sec=("sample_duration_sec", "sum"),
            mean_sample_duration_sec=("sample_duration_sec", "mean"),
        )
        .reset_index()
    )
    summary["eligible_profile"] = (
        summary["n_samples"].ge(OPEN_SET_MIN_SAMPLES_PER_IDENTITY)
        & summary["total_duration_sec"].ge(OPEN_SET_MIN_TOTAL_SECONDS_PER_IDENTITY)
    )
    return open_set_samples, summary


def deterministic_identity_partition(
    identity_ids: Sequence[str],
    random_state: int = RANDOM_SEED,
) -> dict[str, list[str]]:
    identities = sorted({str(value) for value in identity_ids})
    rng = np.random.default_rng(random_state)
    identities = list(rng.permutation(identities))
    identity_count = len(identities)
    if identity_count < MIN_IDENTITIES_FOR_FORMAL_OPEN_SET:
        raise ValueError(
            f"Solo hay {identity_count} agentes elegibles. Se necesitan al menos "
            f"{MIN_IDENTITIES_FOR_FORMAL_OPEN_SET} para separar calibración, "
            "test conocido y test unknown."
        )
    unknown_count = max(1, int(round(identity_count * UNKNOWN_IDENTITY_FRACTION)))
    unknown_count = min(unknown_count, identity_count - 4)
    unknown_ids = identities[:unknown_count]
    known_pool = identities[unknown_count:]
    test_known_count = max(
        2,
        int(round(len(known_pool) * TEST_KNOWN_IDENTITY_FRACTION)),
    )
    test_known_count = min(test_known_count, len(known_pool) - 2)
    return {
        "calibration_known": sorted(known_pool[test_known_count:]),
        "test_known": sorted(known_pool[:test_known_count]),
        "test_unknown": sorted(unknown_ids),
    }


def split_identity_calls(
    samples: pd.DataFrame,
    identity_ids: Sequence[str],
    group_label: str,
    random_state: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    rows = []
    identity_set = {str(value) for value in identity_ids}
    subset = samples[samples["person_id"].astype(str).isin(identity_set)].copy()
    for person_id, group in subset.groupby("person_id", sort=True):
        group = group.sort_values(["audio_key", "sample_id"]).copy()
        indices = list(rng.permutation(group.index.to_list()))
        sample_count = len(indices)
        query_count = max(
            MIN_QUERY_SAMPLES,
            int(round(sample_count * QUERY_FRACTION)),
        )
        query_count = min(query_count, sample_count - MIN_ENROLLMENT_SAMPLES)
        if query_count < MIN_QUERY_SAMPLES:
            raise ValueError(
                f"{person_id} no permite separar enrollment/query: "
                f"{sample_count} muestras."
            )
        query_indices = set(indices[:query_count])
        for index in indices:
            row = group.loc[index]
            rows.append(
                {
                    "sample_id": row["sample_id"],
                    "person_id": str(person_id),
                    "source_identity_id": row["source_identity_id"],
                    "role_proxy": row["role_proxy"],
                    "audio_key": row["audio_key"],
                    "identity_group": group_label,
                    "sample_split": "query" if index in query_indices else "enrollment",
                }
            )
    return pd.DataFrame(rows)


def build_open_set_split(
    samples: pd.DataFrame,
    agent_identity_ids: Sequence[str],
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    partition = deterministic_identity_partition(agent_identity_ids)
    calibration = split_identity_calls(
        samples,
        partition["calibration_known"],
        "calibration_known",
        RANDOM_SEED + 11,
    )
    test_known = split_identity_calls(
        samples,
        partition["test_known"],
        "test_known",
        RANDOM_SEED + 22,
    )
    unknown = samples[
        samples["person_id"].astype(str).isin(set(partition["test_unknown"]))
    ][
        ["sample_id", "person_id", "source_identity_id", "role_proxy", "audio_key"]
    ].copy()
    unknown["identity_group"] = "test_unknown"
    unknown["sample_split"] = "query"
    split = pd.concat([calibration, test_known, unknown], ignore_index=True)
    impossible = split[
        split["identity_group"].eq("test_unknown")
        & split["sample_split"].eq("enrollment")
    ]
    if not impossible.empty:
        raise AssertionError("Una identidad unknown apareció en enrollment.")
    return split, partition


def get_samples_for_split(
    samples: pd.DataFrame,
    split: pd.DataFrame,
    identity_group: str,
    sample_split: str,
) -> pd.DataFrame:
    sample_ids = set(
        split.loc[
            split["identity_group"].eq(identity_group)
            & split["sample_split"].eq(sample_split),
            "sample_id",
        ]
    )
    return samples[samples["sample_id"].isin(sample_ids)].copy().reset_index(drop=True)


def calculate_within_profile_similarity(
    sample_embeddings,
    centroid,
) -> dict[str, float]:
    similarities = cosine_similarity_matrix(
        sample_embeddings,
        np.asarray(centroid).reshape(1, -1),
    ).reshape(-1)
    return {
        "within_similarity_mean": float(np.mean(similarities)),
        "within_similarity_std": float(np.std(similarities)),
        "within_similarity_min": float(np.min(similarities)),
        "within_similarity_max": float(np.max(similarities)),
    }


def build_speaker_profiles(
    enrollment_samples: pd.DataFrame,
    embedding_columns: Sequence[str],
    profile_set_name: str,
) -> pd.DataFrame:
    rows = []
    for person_id, group in enrollment_samples.groupby("person_id", sort=True):
        matrix = l2_normalize_matrix(
            group[list(embedding_columns)].to_numpy(dtype=np.float32)
        )
        centroid = l2_normalize_matrix(matrix.mean(axis=0).reshape(1, -1)).reshape(-1)
        consistency = calculate_within_profile_similarity(matrix, centroid)
        row = {
            "profile_id": str(person_id),
            "source_identity_id": normalize_identifier(
                group["source_identity_id"].iloc[0]
            ),
            "role_proxy": group["role_proxy"].iloc[0],
            "profile_set": profile_set_name,
            "n_enrollment_samples": int(len(group)),
            "n_enrollment_calls": int(group["audio_key"].nunique()),
            "n_enrollment_segments": int(group["n_segments"].fillna(0).sum()),
            "total_enrollment_duration_sec": float(
                group["sample_duration_sec"].fillna(0).sum()
            ),
            "mean_sample_duration_sec": float(
                group["sample_duration_sec"].fillna(0).mean()
            ),
            "mean_overlap_ratio": (
                float(group["mean_overlap_ratio"].mean())
                if "mean_overlap_ratio" in group.columns
                else np.nan
            ),
            "mean_rms_dbfs": (
                float(group["mean_rms_dbfs"].mean())
                if "mean_rms_dbfs" in group.columns
                else np.nan
            ),
            "embedding_dim": len(embedding_columns),
            "embedding_model": EMBEDDING_MODEL_LABEL,
            "notebook_version": NOTEBOOK_VERSION,
            **consistency,
        }
        row.update(
            {
                column: float(value)
                for column, value in zip(embedding_columns, centroid)
            }
        )
        rows.append(row)
    profiles = pd.DataFrame(rows)
    if not profiles.empty:
        profiles[list(embedding_columns)] = l2_normalize_matrix(
            profiles[list(embedding_columns)].to_numpy(dtype=np.float32)
        )
    return profiles


def score_queries_against_profiles(
    query_samples: pd.DataFrame,
    profiles: pd.DataFrame,
    embedding_columns: Sequence[str],
    query_group: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if query_samples.empty:
        return pd.DataFrame(), pd.DataFrame()
    if profiles.empty:
        raise ValueError(f"No hay perfiles para consultar {query_group}.")
    query_matrix = query_samples[list(embedding_columns)].to_numpy(dtype=np.float32)
    profile_matrix = profiles[list(embedding_columns)].to_numpy(dtype=np.float32)
    similarities = cosine_similarity_matrix(query_matrix, profile_matrix)
    profile_ids = profiles["profile_id"].map(normalize_identifier).astype("string").to_numpy()
    profile_source_ids = (
        profiles["source_identity_id"].map(normalize_identifier).astype("string").to_numpy()
    )
    profile_id_set = {str(value) for value in profile_ids if not pd.isna(value)}

    score_rows = []
    prediction_rows = []
    for query_position, (_, query) in enumerate(
        query_samples.reset_index(drop=True).iterrows()
    ):
        query_scores = similarities[query_position]
        order = np.argsort(-query_scores)
        best_index = int(order[0])
        second_index = int(order[1]) if len(order) > 1 else None
        best_profile_id = str(profile_ids[best_index])
        best_source_identity_id = normalize_identifier(profile_source_ids[best_index])
        best_score = float(query_scores[best_index])
        if second_index is None:
            second_profile_id = pd.NA
            second_source_identity_id = pd.NA
            second_score = np.nan
            margin = np.nan
        else:
            second_profile_id = str(profile_ids[second_index])
            second_source_identity_id = normalize_identifier(
                profile_source_ids[second_index]
            )
            second_score = float(query_scores[second_index])
            margin = best_score - second_score

        true_identifier = normalize_identifier(query.get("person_id", pd.NA))
        true_person_id = "UNLABELED" if pd.isna(true_identifier) else str(true_identifier)
        true_is_enrolled = true_person_id in profile_id_set
        prediction_rows.append(
            {
                "query_group": query_group,
                "sample_id": query["sample_id"],
                "audio_key": query["audio_key"],
                "true_person_id": true_person_id,
                "true_source_identity_id": normalize_identifier(
                    query.get("source_identity_id", pd.NA)
                ),
                "true_is_enrolled": bool(true_is_enrolled),
                "best_profile_id": best_profile_id,
                "best_source_identity_id": best_source_identity_id,
                "best_similarity": best_score,
                "second_profile_id": second_profile_id,
                "second_source_identity_id": second_source_identity_id,
                "second_similarity": second_score,
                "top1_top2_margin": margin,
                "top1_correct_before_threshold": bool(
                    true_is_enrolled and best_profile_id == true_person_id
                ),
            }
        )
        for profile_position, profile_id in enumerate(profile_ids):
            candidate_profile_id = str(profile_id)
            score_rows.append(
                {
                    "query_group": query_group,
                    "sample_id": query["sample_id"],
                    "audio_key": query["audio_key"],
                    "true_person_id": true_person_id,
                    "true_is_enrolled": bool(true_is_enrolled),
                    "candidate_profile_id": candidate_profile_id,
                    "same_identity": int(
                        true_is_enrolled and candidate_profile_id == true_person_id
                    ),
                    "similarity": float(query_scores[profile_position]),
                }
            )
    return pd.DataFrame(score_rows), pd.DataFrame(prediction_rows)


def choose_verification_threshold(
    score_table: pd.DataFrame,
    strategy: str = OPEN_SET_THRESHOLD_STRATEGY,
) -> dict[str, object]:
    if score_table.empty:
        raise ValueError("No hay scores para calibrar el umbral.")
    y_true = score_table["same_identity"].astype(int).to_numpy()
    scores = score_table["similarity"].astype(float).to_numpy()
    if len(np.unique(y_true)) < 2:
        raise ValueError("La calibración necesita scores genuine e impostor.")
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_true, scores)
    false_negative_rate = 1 - true_positive_rate
    eer_index = int(np.nanargmin(np.abs(false_positive_rate - false_negative_rate)))
    youden_index = int(np.nanargmax(true_positive_rate - false_positive_rate))
    selected_index = eer_index if strategy.lower() == "eer" else youden_index
    return {
        "strategy": strategy.lower(),
        "acceptance_threshold": float(thresholds[selected_index]),
        "eer": float(
            (false_positive_rate[eer_index] + false_negative_rate[eer_index]) / 2
        ),
        "eer_threshold": float(thresholds[eer_index]),
        "youden_threshold": float(thresholds[youden_index]),
        "calibration_auc": float(roc_auc_score(y_true, scores)),
        "n_scores": int(len(score_table)),
        "n_genuine_scores": int((y_true == 1).sum()),
        "n_impostor_scores": int((y_true == 0).sum()),
    }


def apply_open_set_decision(
    predictions: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    output = predictions.copy()
    if output.empty:
        return output
    output["decision"] = np.where(
        output["best_similarity"].ge(threshold),
        "KNOWN",
        "UNKNOWN",
    )
    output["predicted_person_id"] = pd.Series(pd.NA, index=output.index, dtype="string")
    output.loc[output["decision"].eq("KNOWN"), "predicted_person_id"] = (
        output.loc[output["decision"].eq("KNOWN"), "best_profile_id"].astype("string")
    )
    output["predicted_source_identity_id"] = pd.Series(
        pd.NA,
        index=output.index,
        dtype="string",
    )
    output.loc[
        output["decision"].eq("KNOWN"),
        "predicted_source_identity_id",
    ] = output.loc[
        output["decision"].eq("KNOWN"),
        "best_source_identity_id",
    ].astype("string")
    output["provisional_unknown_id"] = pd.Series(
        pd.NA,
        index=output.index,
        dtype="string",
    )
    output.loc[output["decision"].eq("UNKNOWN"), "provisional_unknown_id"] = (
        "UNKNOWN::"
        + output.loc[output["decision"].eq("UNKNOWN"), "sample_id"].astype(str)
    )

    known_truth = output["true_is_enrolled"].fillna(False).astype(bool)
    output["identification_correct"] = False
    known_comparison = (
        output.loc[known_truth, "predicted_person_id"]
        .eq(output.loc[known_truth, "true_person_id"].astype("string"))
        .fillna(False)
    )
    output.loc[known_truth, "identification_correct"] = (
        output.loc[known_truth, "decision"].eq("KNOWN") & known_comparison
    )
    output.loc[~known_truth, "identification_correct"] = output.loc[
        ~known_truth,
        "decision",
    ].eq("UNKNOWN")
    output["identification_correct"] = (
        output["identification_correct"].fillna(False).astype(bool)
    )
    return output


def safe_ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else np.nan


def evaluate_score_table(
    score_table: pd.DataFrame,
    label: str,
    threshold: float,
) -> dict[str, object]:
    if score_table.empty:
        return {"dataset": label, "n_scores": 0}
    y_true = score_table["same_identity"].astype(int).to_numpy()
    scores = score_table["similarity"].astype(float).to_numpy()
    y_pred = scores >= threshold
    if len(np.unique(y_true)) == 2:
        auc = float(roc_auc_score(y_true, scores))
        false_positive_rate, true_positive_rate, _ = roc_curve(y_true, scores)
        false_negative_rate = 1 - true_positive_rate
        eer_index = int(
            np.nanargmin(np.abs(false_positive_rate - false_negative_rate))
        )
        eer = float(
            (false_positive_rate[eer_index] + false_negative_rate[eer_index]) / 2
        )
    else:
        auc = np.nan
        eer = np.nan
    return {
        "dataset": label,
        "n_scores": int(len(score_table)),
        "n_genuine": int((y_true == 1).sum()),
        "n_impostor": int((y_true == 0).sum()),
        "threshold": float(threshold),
        "auc": auc,
        "eer": eer,
        "verification_accuracy": float(accuracy_score(y_true, y_pred)),
        "verification_precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "verification_recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "verification_f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def evaluate_open_set_predictions(
    predictions: pd.DataFrame,
    calibration_scores: pd.DataFrame,
    test_known_scores: pd.DataFrame,
    threshold: float,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    known_rows = predictions[
        predictions["true_is_enrolled"].fillna(False).astype(bool)
    ].copy()
    unknown_rows = predictions[
        ~predictions["true_is_enrolled"].fillna(False).astype(bool)
    ].copy()
    identification_metrics = {
        "created_at_utc": utc_now_iso(),
        "threshold": threshold,
        "n_known_queries": int(len(known_rows)),
        "n_unknown_queries": int(len(unknown_rows)),
        "known_top1_accuracy_before_rejection": safe_ratio(
            int(known_rows["top1_correct_before_threshold"].sum()),
            len(known_rows),
        ),
        "known_identification_rate": safe_ratio(
            int(known_rows["identification_correct"].sum()),
            len(known_rows),
        ),
        "known_false_rejection_rate": safe_ratio(
            int(known_rows["decision"].eq("UNKNOWN").sum()),
            len(known_rows),
        ),
        "unknown_rejection_rate": safe_ratio(
            int(unknown_rows["decision"].eq("UNKNOWN").sum()),
            len(unknown_rows),
        ),
        "unknown_false_acceptance_rate": safe_ratio(
            int(unknown_rows["decision"].eq("KNOWN").sum()),
            len(unknown_rows),
        ),
        "overall_open_set_accuracy": safe_ratio(
            int(predictions["identification_correct"].sum()),
            len(predictions),
        ),
    }
    verification_metrics = pd.DataFrame(
        [
            evaluate_score_table(
                calibration_scores,
                "calibration_known",
                threshold,
            ),
            evaluate_score_table(
                test_known_scores,
                "test_known",
                threshold,
            ),
        ]
    )
    return identification_metrics, verification_metrics, pd.concat(
        [known_rows.assign(true_class="KNOWN"), unknown_rows.assign(true_class="UNKNOWN")],
        ignore_index=True,
    )


def build_open_set_confusion(predictions: pd.DataFrame) -> pd.DataFrame:
    decision_true = np.where(
        predictions["true_is_enrolled"].fillna(False).astype(bool),
        "KNOWN",
        "UNKNOWN",
    )
    decision_pred = predictions["decision"].astype(str)
    matrix = confusion_matrix(
        decision_true,
        decision_pred,
        labels=["KNOWN", "UNKNOWN"],
    )
    return pd.DataFrame(
        matrix,
        index=["true_known", "true_unknown"],
        columns=["pred_known", "pred_unknown"],
    ).reset_index(names="true_class")


def build_operational_profiles(
    samples: pd.DataFrame,
    identity_summary_open_set: pd.DataFrame,
    embedding_columns: Sequence[str],
    role: str,
) -> pd.DataFrame:
    eligible_ids = set(
        identity_summary_open_set.loc[
            identity_summary_open_set["role_proxy"].eq(role)
            & identity_summary_open_set["eligible_profile"].astype(bool),
            "person_id",
        ].astype(str)
    )
    operational_samples = samples[
        samples["person_id"].astype(str).isin(eligible_ids)
    ].copy()
    return build_speaker_profiles(
        operational_samples,
        embedding_columns,
        "operational_all_available_calls",
    )


def build_model_metadata(
    embedding_columns: Sequence[str],
    threshold_info: dict,
    agent_profiles: pd.DataFrame,
    client_profiles: pd.DataFrame,
) -> dict[str, object]:
    config_snapshot = {
        "notebook_version": NOTEBOOK_VERSION,
        "embedding_model": EMBEDDING_MODEL_LABEL,
        "embedding_dim": len(embedding_columns),
        "segment_filters": {
            "min_duration_sec": MIN_SEGMENT_DURATION_SEC,
            "max_duration_sec": MAX_SEGMENT_DURATION_SEC,
            "max_overlap_ratio": MAX_OVERLAP_RATIO,
            "min_rms_dbfs": MIN_RMS_DBFS,
            "min_words": MIN_WORDS_PER_SEGMENT,
        },
        "sample_filters": {
            "min_segments_per_audio_person": MIN_SEGMENTS_PER_AUDIO_PERSON,
            "min_seconds_per_audio_person": MIN_SECONDS_PER_AUDIO_PERSON,
        },
        "profile_filters": {
            "min_samples_per_identity": OPEN_SET_MIN_SAMPLES_PER_IDENTITY,
            "min_total_seconds_per_identity": OPEN_SET_MIN_TOTAL_SECONDS_PER_IDENTITY,
        },
        "threshold": threshold_info,
    }
    return {
        "created_at_utc": utc_now_iso(),
        "notebook_version": NOTEBOOK_VERSION,
        "config_hash": stable_hash_dict(config_snapshot),
        "embedding_source_path": str(VOICEPRINT_SEGMENT_EMBEDDINGS_CSV),
        "role_mapping_source_path": str(VOICEPRINT_ROLE_MAPPING_CSV),
        "embedding_model": EMBEDDING_MODEL_LABEL,
        "embedding_dim": len(embedding_columns),
        "embedding_columns": list(embedding_columns),
        "distance_metric": "cosine_similarity",
        "acceptance_threshold": float(threshold_info["acceptance_threshold"]),
        "threshold_strategy": threshold_info["strategy"],
        "agent_profiles_path": str(VOICEPRINT_AGENT_PROFILES_CSV),
        "client_profiles_path": (
            str(VOICEPRINT_CLIENT_PROFILES_CSV) if BUILD_CLIENT_PROFILES else None
        ),
        "n_agent_profiles": int(len(agent_profiles)),
        "n_client_profiles": int(len(client_profiles)),
        "identification_output": {
            "known": "source_identity_id (agent_hash o customer_hash)",
            "unknown": None,
            "provisional_unknown_id": "UNKNOWN::<sample_id>",
        },
        "important_note": (
            "UNKNOWN no debe incorporarse automáticamente a la base oficial. "
            "Requiere confirmación externa o varias observaciones consistentes."
        ),
    }


def identify_query_samples(
    query_samples: pd.DataFrame,
    profiles_path: Path = VOICEPRINT_AGENT_PROFILES_CSV,
    threshold_path: Path = VOICEPRINT_THRESHOLDS_JSON,
) -> pd.DataFrame:
    profiles = read_csv_required(profiles_path, "perfiles operacionales")
    threshold_payload = load_json_checkpoint(threshold_path)
    if threshold_payload is None:
        raise FileNotFoundError(f"No existe el umbral open-set: {threshold_path}")
    threshold = float(threshold_payload["acceptance_threshold"])
    query_embedding_columns = get_embedding_columns(query_samples)
    profile_embedding_columns = get_embedding_columns(profiles)
    if query_embedding_columns != profile_embedding_columns:
        raise ValueError(
            "Las dimensiones o nombres de embedding de la query no coinciden "
            "con los perfiles guardados."
        )
    inference_queries = query_samples.copy()
    if "person_id" not in inference_queries.columns:
        inference_queries["person_id"] = "UNLABELED"
    if "source_identity_id" not in inference_queries.columns:
        inference_queries["source_identity_id"] = pd.NA
    _, predictions = score_queries_against_profiles(
        inference_queries,
        profiles,
        profile_embedding_columns,
        "external_inference",
    )
    return apply_open_set_decision(predictions, threshold)


def build_open_set_final_summary(
    voiceprint_segments: pd.DataFrame,
    samples: pd.DataFrame,
    identity_summary_open_set: pd.DataFrame,
    threshold_info: dict,
    identification_metrics: dict,
    agent_profiles: pd.DataFrame,
    client_profiles: pd.DataFrame,
) -> pd.DataFrame:
    rows = [
        {"metric": "segmentos_candidatos", "value": len(voiceprint_segments)},
        {"metric": "muestras_audio_persona", "value": len(samples)},
        {
            "metric": "agentes_elegibles",
            "value": int(
                identity_summary_open_set.loc[
                    identity_summary_open_set["role_proxy"].eq("AGENT")
                    & identity_summary_open_set["eligible_profile"].astype(bool),
                    "person_id",
                ].nunique()
            ),
        },
        {
            "metric": "clientes_elegibles",
            "value": int(
                identity_summary_open_set.loc[
                    identity_summary_open_set["role_proxy"].eq("CLIENT")
                    & identity_summary_open_set["eligible_profile"].astype(bool),
                    "person_id",
                ].nunique()
            ),
        },
        {
            "metric": "umbral_aceptacion",
            "value": float(threshold_info["acceptance_threshold"]),
        },
        {
            "metric": "auc_calibracion",
            "value": threshold_info["calibration_auc"],
        },
        {"metric": "eer_calibracion", "value": threshold_info["eer"]},
        {
            "metric": "known_identification_rate",
            "value": identification_metrics["known_identification_rate"],
        },
        {
            "metric": "unknown_rejection_rate",
            "value": identification_metrics["unknown_rejection_rate"],
        },
        {
            "metric": "open_set_accuracy",
            "value": identification_metrics["overall_open_set_accuracy"],
        },
        {
            "metric": "perfiles_operacionales_agentes",
            "value": len(agent_profiles),
        },
        {
            "metric": "perfiles_operacionales_clientes",
            "value": len(client_profiles),
        },
    ]
    return pd.DataFrame(rows)


def build_run_manifest(
    model_metadata: dict,
    final_summary: pd.DataFrame,
) -> dict[str, object]:
    required_outputs = [
        VOICEPRINT_AGENT_PROFILES_CSV,
        VOICEPRINT_THRESHOLDS_JSON,
        VOICEPRINT_MODEL_METADATA_JSON,
        VOICEPRINT_OPEN_SET_PREDICTIONS_CSV,
        VOICEPRINT_IDENTIFICATION_METRICS_JSON,
        VOICEPRINT_VERIFICATION_METRICS_CSV,
    ]
    missing_outputs = [
        str(path)
        for path in required_outputs
        if not Path(path).exists() or Path(path).stat().st_size == 0
    ]
    return {
        "status": "completed" if not missing_outputs else "incomplete",
        "completed_at_utc": utc_now_iso(),
        "notebook_version": NOTEBOOK_VERSION,
        "config_hash": model_metadata["config_hash"],
        "required_outputs": [str(path) for path in required_outputs],
        "missing_outputs": missing_outputs,
        "summary": dict(zip(final_summary["metric"], final_summary["value"])),
    }


# ============================================================
# FIGURAS
# ============================================================


def plot_similarity_distribution(
    pairs: pd.DataFrame,
    title: str,
    threshold: float | None = None,
):
    if pairs is None or pairs.empty:
        return None
    figure, axis = plt.subplots(figsize=(8, 5))
    impostor = pairs.loc[pairs["same_identity"].eq(0), "similarity"]
    genuine = pairs.loc[pairs["same_identity"].eq(1), "similarity"]
    if not impostor.empty:
        axis.hist(impostor, bins=40, alpha=0.6, label="Impostor")
    if not genuine.empty:
        axis.hist(genuine, bins=40, alpha=0.6, label="Genuine")
    if threshold is not None and not pd.isna(threshold):
        axis.axvline(threshold, linestyle="--", linewidth=2, label=f"Umbral = {threshold:.3f}")
    axis.set_title(title)
    axis.set_xlabel("Similitud coseno")
    axis.set_ylabel("Número de pares")
    axis.legend()
    axis.grid(True, alpha=0.3)
    return figure


def plot_roc_curve(pairs: pd.DataFrame, title: str):
    if pairs is None or pairs.empty or pairs["same_identity"].nunique() < 2:
        return None
    y_true = pairs["same_identity"].astype(int).to_numpy()
    scores = pairs["similarity"].astype(float).to_numpy()
    false_positive_rate, true_positive_rate, _ = roc_curve(y_true, scores)
    auc = roc_auc_score(y_true, scores)
    figure, axis = plt.subplots(figsize=(6, 5))
    axis.plot(false_positive_rate, true_positive_rate, label=f"AUC = {auc:.3f}")
    axis.plot([0, 1], [0, 1], linestyle="--", label="Azar")
    axis.set_title(title)
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.legend()
    axis.grid(True, alpha=0.3)
    return figure


def plot_identity_repetition(identity_summary: pd.DataFrame):
    if identity_summary is None or identity_summary.empty:
        return None
    figure, axis = plt.subplots(figsize=(8, 5))
    plot_data = identity_summary.copy()
    plot_data["n_samples_capped"] = plot_data["n_samples"].clip(upper=20)
    for role, group in plot_data.groupby("role_proxy"):
        axis.hist(group["n_samples_capped"], bins=20, alpha=0.6, label=role)
    axis.set_title("Repetición de identidades por rol")
    axis.set_xlabel("Número de muestras por identidad, truncado en 20")
    axis.set_ylabel("Número de identidades")
    axis.legend()
    axis.grid(True, alpha=0.3)
    return figure


def plot_open_set_known_unknown(
    predictions: pd.DataFrame,
    threshold: float,
):
    if predictions is None or predictions.empty:
        return None
    known = predictions[
        predictions["true_is_enrolled"].fillna(False).astype(bool)
    ]
    unknown = predictions[
        ~predictions["true_is_enrolled"].fillna(False).astype(bool)
    ]
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.hist(known["best_similarity"].dropna(), bins=30, alpha=0.6, label="Queries conocidas")
    axis.hist(unknown["best_similarity"].dropna(), bins=30, alpha=0.6, label="Queries unknown")
    axis.axvline(threshold, linestyle="--", linewidth=2, label=f"Umbral = {threshold:.3f}")
    axis.set_title("Decisión open-set sobre el mejor perfil")
    axis.set_xlabel("Mejor similitud encontrada")
    axis.set_ylabel("Número de queries")
    axis.legend()
    axis.grid(True, alpha=0.3)
    return figure


def save_figure(figure, path: Path) -> Path | None:
    if figure is None:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160, bbox_inches="tight")
    return path


ensure_phase09_directories()
seed_everything()
