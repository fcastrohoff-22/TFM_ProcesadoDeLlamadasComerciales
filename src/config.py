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

import hashlib

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

ANONYMIZATION_SALT = os.environ.get(
    "TFM_ANONYMIZATION_SALT",
    "tfm_local_salt_not_for_final_release",
)


def hash_value(value, salt: str = ANONYMIZATION_SALT):
    """Genera un hash reproducible de 16 caracteres."""
    if pd.isna(value):
        return np.nan

    value = str(value).strip()

    return hashlib.sha256(
        f"{salt}_{value}".encode("utf-8")
    ).hexdigest()[:16]

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
GCS_FINAL_RELABEL_PREFIX = GCS_DIARIZATION_OUTPUT_PREFIX + "final_relabel/"
GCS_EMBEDDING_VECTOR_PREFIX = GCS_FINAL_RELABEL_PREFIX + "embedding_vectors_csv/"

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
# ============================================================
# CONFIGURACIÓN FASE 02
# Validación interna y sensibilidad de diarización
# ============================================================

# Archivos consolidados de diarización que consume la fase 02
# (generados por el Notebook 01)
SCORED_SEGMENTS_CSV = DIARIZATION_ALL_SCORED_SEGMENTS_CSV
ANCHOR_SEGMENTS_CSV = DIARIZATION_ALL_ANCHOR_SEGMENTS_CSV

FINAL_SEGMENTS_CSV = ALL_FINAL_SEGMENTS_CSV
CHANGED_SEGMENTS_CSV = ALL_CHANGED_SEGMENTS_CSV

# Análisis de overlap / anchors
OVERLAP_ANALYSIS_DIR = OUTPUT_DIR / "overlap_analysis"

THRESHOLD_SUMMARY_CSV = (
    OVERLAP_ANALYSIS_DIR / "overlap_threshold_summary.csv"
)
AUDIO_OVERLAP_SUMMARY_CSV = (
    OVERLAP_ANALYSIS_DIR / "audio_overlap_summary.csv"
)
SEGMENT_OVERLAP_DISTRIBUTION_CSV = (
    OVERLAP_ANALYSIS_DIR / "segment_overlap_distribution.csv"
)
OVERLAP_RECOMMENDATION_TXT = (
    OVERLAP_ANALYSIS_DIR / "overlap_threshold_recommendation.txt"
)

OVERLAP_THRESHOLDS = [0.00, 0.01, 0.03, 0.05, 0.10, 0.15, 0.20, 0.30]

# Análisis de sensibilidad del margen de reetiquetado
MARGIN_ANALYSIS_DIR = OUTPUT_DIR / "relabel_margin_sensitivity"

RELABEL_MARGIN_DETAIL_CSV = (
    MARGIN_ANALYSIS_DIR / "relabel_margin_sensitivity_detail.csv"
)
RELABEL_MARGIN_BY_AUDIO_CSV = (
    MARGIN_ANALYSIS_DIR / "relabel_margin_sensitivity_by_audio.csv"
)
RELABEL_MARGIN_SUMMARY_CSV = (
    MARGIN_ANALYSIS_DIR / "relabel_margin_sensitivity_summary.csv"
)
RELABEL_MARGIN_SKIPPED_CSV = (
    MARGIN_ANALYSIS_DIR / "relabel_margin_sensitivity_skipped_files.csv"
)

RELABEL_MARGIN_OPTIONS = [0.00, 0.005, 0.01, 0.02, 0.03, 0.05, 0.10]

# Margen usado actualmente en el Notebook 01 (para comparar deltas)
CURRENT_RELABEL_MIN_MARGIN = 0.01

# Parámetros del pipeline original necesarios para la simulación
MIN_SEGMENT_DURATION_SEC = 0.70
MIN_RMS_DBFS = -40.0
MIN_ANCHOR_DURATION_SEC = 1.20
INITIAL_EXCLUDE_SEC_FOR_ANCHORS = 1.50
NUM_SPEAKERS = 2
ANCHORS_PER_SPEAKER = 3
MAX_GAP_MERGE_SEC = 0.50


def ensure_phase02_directories() -> None:
    """Crea las carpetas locales necesarias para la fase 02."""
    OVERLAP_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    MARGIN_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# CONFIGURACIÓN FASE 03
# Evaluación interna de diarización, relabeling y embeddings
# ============================================================

# Entradas generadas por las fases 00 y 01
EVALUATION_SUMMARY_CSV = DIARIZATION_SUMMARY_CSV
EVALUATION_SCORED_SEGMENTS_CSV = DIARIZATION_ALL_SCORED_SEGMENTS_CSV
EVALUATION_VALID_SEGMENTS_CSV = DIARIZATION_ALL_VALID_SEGMENTS_CSV
EVALUATION_ANCHOR_SEGMENTS_CSV = DIARIZATION_ALL_ANCHOR_SEGMENTS_CSV

EVALUATION_FINAL_SEGMENTS_CSV = ALL_FINAL_SEGMENTS_CSV
EVALUATION_FINAL_MERGED_SEGMENTS_CSV = ALL_FINAL_MERGED_SEGMENTS_CSV
EVALUATION_CHANGED_SEGMENTS_CSV = ALL_CHANGED_SEGMENTS_CSV
EVALUATION_RELABEL_SUMMARY_CSV = RELABEL_SUMMARY_CSV
EVALUATION_ANCHOR_EMBEDDINGS_CSV = ALL_ANCHOR_EMBEDDINGS_CSV
EVALUATION_ANCHOR_EMBEDDING_VECTORS_CSV = (
    ALL_ANCHOR_EMBEDDING_VECTORS_CSV
)

# Compatibilidad con una versión anterior del Notebook 01
LEGACY_ANCHOR_EMBEDDING_VECTORS_CSV = (
    FINAL_RELABEL_DIR / "all_anchor_embeddings_vectors.csv"
)

# Parámetros del análisis
EVALUATION_RANDOM_SEED = 42
MAX_PAIRS_PER_IDENTITY = 20
MAX_NEGATIVE_PAIRS = 20000

MAX_OVERLAP_RATIO_FOR_VOICEPRINT = 0.05
MIN_DURATION_FOR_VOICEPRINT = 1.20
MIN_RMS_DBFS_FOR_VOICEPRINT = -40.0


def ensure_phase03_directories() -> None:
    """Crea las carpetas locales necesarias para la fase 03."""
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_RELABEL_DIR.mkdir(parents=True, exist_ok=True)
    EMBEDDING_VECTOR_CSV_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# CONFIGURACIÓN FASE 04
# Consolidación de segmentos finales de diarización
# ============================================================

CONSOLIDATED_DIR = OUTPUT_DIR / "consolidated"
GCS_CONSOLIDATED_PREFIX = (
    GCS_DIARIZATION_OUTPUT_PREFIX + "consolidated/"
)

CONSOLIDATED_ALL_FINAL_MERGED_SEGMENTS_CSV = (
    CONSOLIDATED_DIR / "all_final_merged_segments.csv"
)
CONSOLIDATED_ALL_FINAL_MERGED_SEGMENTS_DEDUP_CSV = (
    CONSOLIDATED_DIR / "all_final_merged_segments_dedup.csv"
)
CONSOLIDATED_AUDIT_FINAL_MERGED_FILES_CSV = (
    CONSOLIDATED_DIR / "audit_final_merged_files.csv"
)
CONSOLIDATED_DROPPED_DUPLICATE_FILES_CSV = (
    CONSOLIDATED_DIR
    / "dropped_duplicate_final_merged_files.csv"
)
CONSOLIDATED_PREVIOUS_BACKUP_CSV = (
    CONSOLIDATED_DIR
    / "all_final_merged_segments_previous_backup.csv"
)

EXPECTED_AUDIOS_CONSOLIDATION = 1200


def ensure_phase04_directories() -> None:
    """Crea la carpeta local necesaria para la fase 04."""
    FINAL_RELABEL_DIR.mkdir(parents=True, exist_ok=True)
    CONSOLIDATED_DIR.mkdir(parents=True, exist_ok=True)



# ============================================================
# CONFIGURACIÓN FASE 05
# Transcripción contextual con Whisper
# ============================================================

TRANSCRIPTION_ROOT = DATA_DIR / "transcription_outputs"
GCS_TRANSCRIPTION_PREFIX = GCS_UNAV_ROOT + "transcription_outputs/"

TRANSCRIPTION_RUNS_DIR = TRANSCRIPTION_ROOT / "runs"
TRANSCRIPTION_PER_AUDIO_DIR = TRANSCRIPTION_ROOT / "per_audio"
TRANSCRIPTION_WORDS_DIR = TRANSCRIPTION_ROOT / "per_audio_words"
TRANSCRIPTION_ASR_DIR = TRANSCRIPTION_ROOT / "per_audio_asr_segments"
TRANSCRIPTION_TEXT_DIR = TRANSCRIPTION_ROOT / "per_audio_full_text"

TRANSCRIPTION_ALL_SEGMENTS_CSV = (
    TRANSCRIPTION_ROOT / "all_segments_transcribed.csv"
)
TRANSCRIPTION_SUMMARY_CSV = (
    TRANSCRIPTION_ROOT / "transcription_summary.csv"
)
TRANSCRIPTION_FINAL_SEGMENTS_CSV = (
    TRANSCRIPTION_ROOT / "06_transcribed_segments_final.csv"
)
TRANSCRIPTION_FINAL_SUMMARY_CSV = (
    TRANSCRIPTION_ROOT / "06_transcription_summary_final.csv"
)
TRANSCRIPTION_ACTIVE_RUN_JSON = TRANSCRIPTION_ROOT / "active_run.json"

TRANSCRIPTION_MODEL_SIZE = "large-v3-turbo"
TRANSCRIPTION_LANGUAGE = "es"
TRANSCRIPTION_TARGET_SR = 16000
TRANSCRIPTION_METHOD = "full_audio_word_alignment_v3_turbo"
TRANSCRIPTION_EXPECTED_AUDIOS = 1181
TRANSCRIPTION_SAVE_EVERY_N_AUDIOS = 25
TRANSCRIPTION_NEAREST_WORD_TOLERANCE_SEC = 0.40
TRANSCRIPTION_LOW_WORD_PROBABILITY = 0.50


def ensure_phase05_directories() -> None:
    """Crea las carpetas locales necesarias para la fase 05."""
    TRANSCRIPTION_ROOT.mkdir(parents=True, exist_ok=True)
    TRANSCRIPTION_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    TRANSCRIPTION_PER_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    TRANSCRIPTION_WORDS_DIR.mkdir(parents=True, exist_ok=True)
    TRANSCRIPTION_ASR_DIR.mkdir(parents=True, exist_ok=True)
    TRANSCRIPTION_TEXT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# CONFIGURACIÓN FASE 06
# Metadata oficial y ground truth proxy textual
# ============================================================

PROXY_GROUNDTRUTH_DIR = DATA_DIR / "proxy_groundtruth_outputs"
PROXY_FIGURES_DIR = PROXY_GROUNDTRUTH_DIR / "figures"
PROXY_CHECKPOINT_DIR = PROXY_GROUNDTRUTH_DIR / "checkpoints"
GCS_PROXY_GROUNDTRUTH_PREFIX = (
    GCS_UNAV_ROOT + "proxy_groundtruth_outputs/"
)

PROXY_SEGMENT_LEVEL_CSV = (
    PROXY_GROUNDTRUTH_DIR / "segment_level_proxy_groundtruth.csv"
)
PROXY_SPEAKER_ROLE_MAPPING_CSV = (
    PROXY_GROUNDTRUTH_DIR / "speaker_role_mapping_textual.csv"
)
PROXY_TEXTUAL_METRICS_CSV = (
    PROXY_GROUNDTRUTH_DIR / "textual_proxy_metrics_summary.csv"
)
PROXY_ALIGNMENT_SUMMARY_CSV = (
    PROXY_GROUNDTRUTH_DIR / "alignment_processing_summary.csv"
)


def ensure_phase06_directories() -> None:
    """Crea las carpetas locales necesarias para la fase 06."""
    PROXY_GROUNDTRUTH_DIR.mkdir(parents=True, exist_ok=True)
    PROXY_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    PROXY_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# CONFIGURACIÓN FASE 07A
# Sentimiento textual
# ============================================================

SENTIMENT_DIR = DATA_DIR / "sentiment_outputs"
SENTIMENT_FIGURES_DIR = SENTIMENT_DIR / "figures"
GCS_SENTIMENT_PREFIX = GCS_UNAV_ROOT + "sentiment_outputs/"

SEGMENTS_WITH_SENTIMENT_CSV = (
    SENTIMENT_DIR / "segments_with_sentiment_textual.csv"
)
ALL_SEGMENTS_SENTIMENT_ENRICHED_CSV = (
    SENTIMENT_DIR / "all_segments_sentiment_textual_enriched.csv"
)
CALL_SENTIMENT_CSV = (
    SENTIMENT_DIR / "call_level_sentiment_textual.csv"
)
CALL_ROLE_SENTIMENT_CSV = (
    SENTIMENT_DIR / "call_role_level_sentiment_textual.csv"
)
ROLE_SENTIMENT_CSV = (
    SENTIMENT_DIR / "role_level_sentiment_textual.csv"
)
SPEAKER_SENTIMENT_CSV = (
    SENTIMENT_DIR / "speaker_level_sentiment_textual.csv"
)
TEMPORAL_SENTIMENT_CSV = (
    SENTIMENT_DIR / "temporal_sentiment_textual.csv"
)
SENTIMENT_SUMMARY_CSV = (
    SENTIMENT_DIR / "sentiment_textual_summary_for_memory.csv"
)
SENTIMENT_CHECKPOINT_CSV = (
    SENTIMENT_DIR / "sentiment_segments_checkpoint.csv"
)
SENTIMENT_MODEL_NAME = "pysentimiento/robertuito-sentiment-analysis"


def ensure_phase07a_directories() -> None:
    """Crea las carpetas locales necesarias para la fase 07A."""
    SENTIMENT_DIR.mkdir(parents=True, exist_ok=True)
    SENTIMENT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# CONFIGURACIÓN FASE 07B
# Análisis afectivo desde audio
# ============================================================

PROSODY_DIR = DATA_DIR / "prosody_outputs"
PROSODY_FIGURES_DIR = PROSODY_DIR / "figures"
GCS_PROSODY_PREFIX = GCS_UNAV_ROOT + "prosody_outputs/"

SEGMENTS_PROSODY_CSV = (
    PROSODY_DIR / "segments_with_audio_affect_prosody.csv"
)
CALL_PROSODY_CSV = (
    PROSODY_DIR / "call_level_audio_affect_prosody.csv"
)
CALL_ROLE_PROSODY_CSV = (
    PROSODY_DIR / "call_role_level_audio_affect_prosody.csv"
)
CALL_SPEAKER_PROSODY_CSV = (
    PROSODY_DIR / "call_speaker_level_audio_affect_prosody.csv"
)
ROLE_PROSODY_CSV = (
    PROSODY_DIR / "role_level_audio_affect_prosody.csv"
)
PROSODY_SUMMARY_CSV = (
    PROSODY_DIR / "prosody_audio_affect_summary_for_memory.csv"
)
SER_PREDICTIONS_CSV = PROSODY_DIR / "ser_model_predictions.csv"
AUDIO_TEXT_COMPARISON_CSV = (
    PROSODY_DIR / "audio_vs_textual_sentiment_comparison.csv"
)
SER_CHECKPOINT_CSV = (
    PROSODY_DIR / "ser_model_predictions_checkpoint.csv"
)
PROSODY_FEATURES_CHECKPOINT_CSV = (
    PROSODY_DIR / "prosody_features_checkpoint.csv"
)
SER_MODEL_ID = "UMUTeam/w2v-bert-emotion-es"


def ensure_phase07b_directories() -> None:
    """Crea las carpetas locales necesarias para la fase 07B."""
    PROSODY_DIR.mkdir(parents=True, exist_ok=True)
    PROSODY_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# CONFIGURACIÓN FASE 07C
# Fusión audio-texto
# ============================================================

SENTIMENT_FUSION_DIR = DATA_DIR / "sentiment_fusion_outputs"
SENTIMENT_FUSION_FIGURES_DIR = SENTIMENT_FUSION_DIR / "figures"
GCS_SENTIMENT_FUSION_PREFIX = (
    GCS_UNAV_ROOT + "sentiment_fusion_outputs/"
)

FUSION_SEGMENTS_CSV = (
    SENTIMENT_FUSION_DIR / "segments_audio_text_fusion.csv"
)
FUSION_CORRELATIONS_CSV = (
    SENTIMENT_FUSION_DIR / "correlations_audio_text.csv"
)
FUSION_CONFUSION_CSV = (
    SENTIMENT_FUSION_DIR / "confusion_prosodic_state_vs_sentiment.csv"
)
FUSION_DISAGREEMENT_CSV = (
    SENTIMENT_FUSION_DIR
    / "disagreement_masked_frustration_segments.csv"
)
FUSION_ROLE_LEVEL_CSV = (
    SENTIMENT_FUSION_DIR / "role_level_audio_text_fusion.csv"
)
FUSION_CALL_LEVEL_CSV = (
    SENTIMENT_FUSION_DIR / "call_level_audio_text_fusion.csv"
)
FUSION_SUMMARY_CSV = (
    SENTIMENT_FUSION_DIR / "fusion_summary_for_memory.csv"
)


def ensure_phase07c_directories() -> None:
    """Crea las carpetas locales necesarias para la fase 07C."""
    SENTIMENT_FUSION_DIR.mkdir(parents=True, exist_ok=True)
    SENTIMENT_FUSION_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# CONFIGURACIÓN FASE 08
# Keyword spotting
# ============================================================

KEYWORD_DIR = DATA_DIR / "keyword_outputs"
GCS_KEYWORD_PREFIX = GCS_UNAV_ROOT + "keyword_outputs/"

SEGMENTS_WITH_KEYWORDS_CSV = (
    KEYWORD_DIR / "segments_with_keywords.csv"
)
CALL_LEVEL_KEYWORDS_CSV = (
    KEYWORD_DIR / "call_level_keywords.csv"
)
TOP_CRITICAL_CALLS_CSV = (
    KEYWORD_DIR / "top_critical_calls_keywords.csv"
)
CALL_KEYWORDS_SENTIMENT_CSV = (
    KEYWORD_DIR / "call_level_keywords_sentiment_combined.csv"
)


def ensure_phase08_directories() -> None:
    """Crea la carpeta local necesaria para la fase 08."""
    KEYWORD_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# CONFIGURACIÓN FASE 09
# Verificación e identificación open-set por huella de voz
# ============================================================

VOICEPRINT_DIR = DATA_DIR / "voiceprint_outputs"
VOICEPRINT_FIGURES_DIR = VOICEPRINT_DIR / "figures"
VOICEPRINT_CHECKPOINT_DIR = VOICEPRINT_DIR / "checkpoints"
GCS_VOICEPRINT_PREFIX = GCS_UNAV_ROOT + "voiceprint_outputs/"

VOICEPRINT_SEGMENT_EMBEDDINGS_CSV = (
    EMBEDDING_VECTOR_CSV_DIR / "all_segment_embeddings_vectors.csv"
)
VOICEPRINT_ANCHOR_EMBEDDINGS_CSV = (
    EMBEDDING_VECTOR_CSV_DIR / "all_anchor_embeddings_vectors.csv"
)
VOICEPRINT_ROLE_MAPPING_CSV = PROXY_SPEAKER_ROLE_MAPPING_CSV
VOICEPRINT_SEGMENT_PROXY_CSV = PROXY_SEGMENT_LEVEL_CSV

VOICEPRINT_FINAL_SUMMARY_CSV = (
    VOICEPRINT_DIR / "voiceprint_final_summary_for_memory.csv"
)
VOICEPRINT_OPEN_SET_SUMMARY_CSV = (
    VOICEPRINT_DIR / "voiceprint_open_set_final_summary.csv"
)
VOICEPRINT_SUCCESS_JSON = (
    VOICEPRINT_DIR / "_SUCCESS_voiceprint_open_set.json"
)


def ensure_phase09_directories() -> None:
    """Crea las carpetas locales necesarias para la fase 09."""
    VOICEPRINT_DIR.mkdir(parents=True, exist_ok=True)
    VOICEPRINT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    VOICEPRINT_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
