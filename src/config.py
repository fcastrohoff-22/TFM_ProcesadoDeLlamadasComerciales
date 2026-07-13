"""Configuración centralizada de rutas y fuentes del proyecto."""

import os
from pathlib import Path
from dotenv import load_dotenv # type: ignore

# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"

load_dotenv(PROJECT_DIR / ".env")
HF_TOKEN = os.environ.get("HF_TOKEN")

GCS_AUDIO_SOURCES = {
    "raw": "gs://catedras_audio_detection/pipelineA/raw/",
    "raw_bajas": "gs://catedras_audio_detection/pipelineA/raw_bajas/",
}

GCS_UNAV_ROOT = "gs://catedras_audio_detection/pipelineA/procesados_UNAV/"
GCS_UNAV_RAW_UNIFIED_PREFIX = GCS_UNAV_ROOT + "raw_unified/"
GCS_UNAV_CLEAN_AUDIO_PREFIX = GCS_UNAV_ROOT + "clean_audios/"
GCS_UNAV_CSV_PREFIX = GCS_UNAV_ROOT + "csv_outputs/"

BQ_PROJECT_ID = "mm-bi-catedras-upm"
BQ_DATASET = "AUDIO_DETECTION"

BQ_METADATA_SOURCES = {
    "raw": "tablon_audios",
    "raw_bajas": "tablon_audios_bajas",
}

def split_gcs_uri(gcs_uri: str) -> tuple[str, str]:
    """Separa una ruta GCS en nombre del bucket y prefijo."""
    if not gcs_uri.startswith("gs://"):
        raise ValueError("La ruta debe empezar por 'gs://'")

    path = gcs_uri[5:]
    bucket_name, _, prefix = path.partition("/")

    if not bucket_name:
        raise ValueError("La ruta GCS no contiene un nombre de bucket.")

    return bucket_name, prefix

# ============================================================
# ANONIMIZACIÓN
# ============================================================

ANONYMIZATION_SALT = os.environ.get(
    "TFM_ANONYMIZATION_SALT",
    "tfm_local_salt_not_for_final_release",
)

# ============================================================
# CONFIGURACIÓN FASE 00
# ============================================================

EDA_DIR = DATA_DIR / "eda"
CLEAN_RESULTS_DIR = DATA_DIR / "clean_results"

AUDIO_INVENTORY_PRIVATE_CSV = EDA_DIR / "audio_inventory_private.csv"
AUDIO_INVENTORY_PUBLIC_CSV = EDA_DIR / "audio_inventory_public_anonymized.csv"
BQ_METADATA_SNAPSHOT_CSV = EDA_DIR / "bq_metadata_snapshot.csv"
SILENCE_THRESHOLD_SUMMARY_CSV = EDA_DIR / "silence_threshold_summary.csv"

CLEANING_RESULTS_PRIVATE_CSV = CLEAN_RESULTS_DIR / "audio_cleaning_results_private.csv"
CLEANING_RESULTS_PUBLIC_CSV = CLEAN_RESULTS_DIR / "audio_cleaning_results_public_anonymized.csv"
VALID_AUDIO_PRIVATE_CSV = CLEAN_RESULTS_DIR / "audio_valid_for_diarization_private.csv"
VALID_AUDIO_PUBLIC_CSV = CLEAN_RESULTS_DIR / "audio_valid_for_diarization_public_anonymized.csv"


def ensure_phase00_directories() -> None:
    """Crea las carpetas locales necesarias para la fase 00."""
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    CLEAN_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# CONFIGURACIÓN FASE 01
# ============================================================

GCS_CLEAN_AUDIO_PREFIX = GCS_UNAV_CLEAN_AUDIO_PREFIX
GCS_DIARIZATION_OUTPUT_PREFIX = GCS_UNAV_ROOT + "diarization_outputs/"

INPUT_DIR = DATA_DIR / "diarization_input_clean_audios"
OUTPUT_DIR = DATA_DIR / "diarization_outputs"

# Salidas consolidadas de diarización
DIARIZATION_SUMMARY_CSV = OUTPUT_DIR / "diarization_summary.csv"
DIARIZATION_ALL_REGULAR_SEGMENTS_CSV = OUTPUT_DIR / "diarization_all_regular_segments.csv"
DIARIZATION_ALL_SCORED_SEGMENTS_CSV = OUTPUT_DIR / "diarization_all_scored_segments.csv"
DIARIZATION_ALL_VALID_SEGMENTS_CSV = OUTPUT_DIR / "diarization_all_valid_segments.csv"
DIARIZATION_ALL_ANCHOR_SEGMENTS_CSV = OUTPUT_DIR / "diarization_all_anchor_segments.csv"
DIARIZATION_ERRORS_CSV = OUTPUT_DIR / "diarization_errors.csv"


def ensure_phase01_directories() -> None:
    """Crea las carpetas locales necesarias para la fase 01."""
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# CONFIGURACIÓN DE REETIQUETADO Y EMBEDDINGS

FINAL_RELABEL_DIR = OUTPUT_DIR / "final_relabel"

EMBEDDING_VECTOR_CSV_DIR = FINAL_RELABEL_DIR / "embedding_vectors_csv"
RELABEL_SUMMARY_CSV = FINAL_RELABEL_DIR / "relabel_summary.csv"
ALL_FINAL_SEGMENTS_CSV = FINAL_RELABEL_DIR / "all_final_segments.csv"
ALL_FINAL_MERGED_SEGMENTS_CSV = FINAL_RELABEL_DIR / "all_final_merged_segments.csv"
ALL_ANCHOR_EMBEDDINGS_CSV = FINAL_RELABEL_DIR / "all_anchor_embeddings.csv"
ALL_CHANGED_SEGMENTS_CSV = FINAL_RELABEL_DIR / "all_changed_segments.csv"
RELABELING_SUMMARY_BY_AUDIO_CSV = FINAL_RELABEL_DIR / "relabeling_summary_by_audio.csv"
ALL_ANCHOR_EMBEDDING_VECTORS_CSV = EMBEDDING_VECTOR_CSV_DIR / "all_anchor_embeddings_vectors.csv"
ALL_SEGMENT_EMBEDDING_VECTORS_CSV = EMBEDDING_VECTOR_CSV_DIR / "all_segment_embeddings_vectors.csv"

SAVE_EMBEDDING_VECTOR_CSVS = True
EMBEDDING_MODEL_ID = "pyannote/wespeaker-voxceleb-resnet34-LM"
EMBEDDING_SAMPLE_RATE = 16000
RELABEL_MIN_MARGIN = 0.01

def ensure_relabel_directories() -> None:
    """Crea las carpetas necesarias para reetiquetado y embeddings."""
    FINAL_RELABEL_DIR.mkdir(parents=True, exist_ok=True)
    EMBEDDING_VECTOR_CSV_DIR.mkdir(parents=True, exist_ok=True)