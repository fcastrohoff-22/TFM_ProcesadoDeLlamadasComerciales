"""Configuración centralizada de rutas y fuentes del proyecto."""

import os
from pathlib import Path


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"

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

