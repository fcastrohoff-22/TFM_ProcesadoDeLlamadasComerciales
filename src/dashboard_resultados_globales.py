"""Dashboard global e interactivo de resultados del TFM.

Este módulo reemplaza el Notebook 11 original. El notebook queda como
orquestador y llama una sola función pública: ``run_dashboard_resultados_globales``.

La fase solo lee outputs existentes desde Google Cloud Storage y los restaura
localmente cuando hace falta. No sube, elimina ni modifica ningún objeto en GCS.
El dashboard se genera como HTML autónomo con filtros y pestañas en JavaScript.
No depende del kernel después de cargarse en JupyterLab.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import html as html_lib
import json
import math
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from IPython.display import HTML
from plotly.offline import get_plotlyjs

from pandas.errors import EmptyDataError

from src import config as cfg
from src.storage_io import download_uri_to_local, join_gcs_uri

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURACIÓN LOCAL DEL DASHBOARD
# ============================================================

DATA_DIR = Path(cfg.DATA_DIR)
PROJECT_DIR = Path(cfg.PROJECT_DIR)
DASHBOARD_DIR = DATA_DIR / "global_results_dashboard"
HTML_DIR = DASHBOARD_DIR / "html_exports"
HTML_PATH = HTML_DIR / "dashboard_resultados_globales.html"

COLORS = {
    "blue": "#2F6BFF",
    "navy": "#123047",
    "orange": "#F28E2B",
    "green": "#2A9D6F",
    "red": "#D1495B",
    "purple": "#7B61A8",
    "teal": "#2A9D8F",
    "gray": "#7A8288",
    "light": "#F4F7FA",
}

GCS_UNAV_ROOT = str(cfg.GCS_UNAV_ROOT)
GCS_UNAV_CSV_PREFIX = str(cfg.GCS_UNAV_CSV_PREFIX)

_GCS_CLIENT = None


# ============================================================
# RUTAS DE ENTRADA
# ============================================================


def _cfg_path(name: str, fallback: Path) -> Path:
    """Obtiene una ruta desde ``src.config`` con fallback compatible."""
    return Path(getattr(cfg, name, fallback))


EDA_DIR = _cfg_path("EDA_DIR", DATA_DIR / "eda")
CLEAN_RESULTS_DIR = _cfg_path("CLEAN_RESULTS_DIR", DATA_DIR / "clean_results")
DIARIZATION_DIR = _cfg_path("OUTPUT_DIR", DATA_DIR / "diarization_outputs")
FINAL_RELABEL_DIR = _cfg_path(
    "FINAL_RELABEL_DIR",
    DIARIZATION_DIR / "final_relabel",
)
CONSOLIDATED_DIR = _cfg_path(
    "CONSOLIDATED_DIR",
    DIARIZATION_DIR / "consolidated",
)
TRANSCRIPTION_DIR = _cfg_path(
    "TRANSCRIPTION_ROOT",
    DATA_DIR / "transcription_outputs",
)
PROXY_DIR = _cfg_path(
    "PROXY_GROUNDTRUTH_DIR",
    DATA_DIR / "proxy_groundtruth_outputs",
)
SENTIMENT_DIR = _cfg_path(
    "SENTIMENT_DIR",
    DATA_DIR / "sentiment_outputs",
)
PROSODY_DIR = _cfg_path(
    "PROSODY_DIR",
    DATA_DIR / "prosody_outputs",
)
FUSION_DIR = _cfg_path(
    "SENTIMENT_FUSION_DIR",
    DATA_DIR / "sentiment_fusion_outputs",
)
KEYWORD_DIR = _cfg_path(
    "KEYWORD_DIR",
    DATA_DIR / "keyword_outputs",
)
VOICEPRINT_DIR = _cfg_path(
    "VOICEPRINT_DIR",
    DATA_DIR / "voiceprint_outputs",
)


def _first_path(*paths: Path) -> Path:
    for path in paths:
        if Path(path).exists():
            return Path(path)
    return Path(paths[0])


DATASET_PATHS: dict[str, Path] = {
    # Fase 00
    "audio_inventory": _cfg_path(
        "AUDIO_INVENTORY_PRIVATE_CSV",
        EDA_DIR / "audio_inventory_private.csv",
    ),
    "metadata_snapshot": _cfg_path(
        "BQ_METADATA_SNAPSHOT_CSV",
        EDA_DIR / "bq_metadata_snapshot.csv",
    ),
    "cleaning_results": _cfg_path(
        "CLEANING_RESULTS_PRIVATE_CSV",
        CLEAN_RESULTS_DIR / "audio_cleaning_results_private.csv",
    ),
    "silence_threshold_summary": _cfg_path(
        "SILENCE_THRESHOLD_SUMMARY_CSV",
        EDA_DIR / "silence_threshold_summary.csv",
    ),

    # Fases 01–04
    "diarization_summary": _cfg_path(
        "DIARIZATION_SUMMARY_CSV",
        DIARIZATION_DIR / "diarization_summary.csv",
    ),
    "regular_segments": _cfg_path(
        "DIARIZATION_ALL_REGULAR_SEGMENTS_CSV",
        DIARIZATION_DIR / "diarization_all_regular_segments.csv",
    ),
    "scored_segments": _cfg_path(
        "DIARIZATION_ALL_SCORED_SEGMENTS_CSV",
        DIARIZATION_DIR / "diarization_all_scored_segments.csv",
    ),
    "valid_segments": _cfg_path(
        "DIARIZATION_ALL_VALID_SEGMENTS_CSV",
        DIARIZATION_DIR / "diarization_all_valid_segments.csv",
    ),
    "anchor_segments": _cfg_path(
        "DIARIZATION_ALL_ANCHOR_SEGMENTS_CSV",
        DIARIZATION_DIR / "diarization_all_anchor_segments.csv",
    ),
    "relabel_summary": _cfg_path(
        "RELABEL_SUMMARY_CSV",
        FINAL_RELABEL_DIR / "relabel_summary.csv",
    ),
    "final_segments": _cfg_path(
        "ALL_FINAL_SEGMENTS_CSV",
        FINAL_RELABEL_DIR / "all_final_segments.csv",
    ),
    "changed_segments": _cfg_path(
        "ALL_CHANGED_SEGMENTS_CSV",
        FINAL_RELABEL_DIR / "all_changed_segments.csv",
    ),
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
    "relabel_margin_summary": _cfg_path(
        "RELABEL_MARGIN_SUMMARY_CSV",
        DIARIZATION_DIR
        / "relabel_margin_sensitivity"
        / "relabel_margin_sensitivity_summary.csv",
    ),
    "relabel_margin_by_audio": _cfg_path(
        "RELABEL_MARGIN_BY_AUDIO_CSV",
        DIARIZATION_DIR
        / "relabel_margin_sensitivity"
        / "relabel_margin_sensitivity_by_audio.csv",
    ),
    "overlap_threshold_summary": _cfg_path(
        "THRESHOLD_SUMMARY_CSV",
        DIARIZATION_DIR / "overlap_analysis" / "overlap_threshold_summary.csv",
    ),
    "audio_overlap_summary": _cfg_path(
        "AUDIO_OVERLAP_SUMMARY_CSV",
        DIARIZATION_DIR / "overlap_analysis" / "audio_overlap_summary.csv",
    ),

    # Fase 05
    "transcribed_segments": _first_path(
        _cfg_path(
            "TRANSCRIPTION_FINAL_SEGMENTS_CSV",
            TRANSCRIPTION_DIR / "06_transcribed_segments_final.csv",
        ),
        _cfg_path(
            "TRANSCRIPTION_ALL_SEGMENTS_CSV",
            TRANSCRIPTION_DIR / "all_segments_transcribed.csv",
        ),
    ),
    "transcription_summary": _cfg_path(
        "TRANSCRIPTION_SUMMARY_CSV",
        TRANSCRIPTION_DIR / "transcription_summary.csv",
    ),
    "transcription_final_summary": _cfg_path(
        "TRANSCRIPTION_FINAL_SUMMARY_CSV",
        TRANSCRIPTION_DIR / "06_transcription_summary_final.csv",
    ),

    # Fase 06
    "metadata_join_audit": PROXY_DIR / "metadata_join_audit_by_audio.csv",
    "official_role_presence": PROXY_DIR / "official_role_presence_by_audio.csv",
    "metadata_role_summary": PROXY_DIR / "metadata_role_summary.csv",
    "official_turns": PROXY_DIR / "official_transcription_turns_extracted.csv",
    "alignment_candidates": PROXY_DIR / "text_alignment_candidates.csv",
    "alignment_matches": PROXY_DIR / "text_alignment_matches.csv",
    "alignment_threshold_sensitivity": PROXY_DIR
    / "text_alignment_threshold_sensitivity.csv",
    "speaker_role_mapping": _cfg_path(
        "PROXY_SPEAKER_ROLE_MAPPING_CSV",
        PROXY_DIR / "speaker_role_mapping_textual.csv",
    ),
    "segment_proxy": _cfg_path(
        "PROXY_SEGMENT_LEVEL_CSV",
        PROXY_DIR / "segment_level_proxy_groundtruth.csv",
    ),
    "proxy_metrics": _cfg_path(
        "PROXY_TEXTUAL_METRICS_CSV",
        PROXY_DIR / "textual_proxy_metrics_summary.csv",
    ),
    "alignment_processing_summary": _cfg_path(
        "PROXY_ALIGNMENT_SUMMARY_CSV",
        PROXY_DIR / "alignment_processing_summary.csv",
    ),
    "holdout_predictions": PROXY_DIR / "holdout_role_mapping_predictions.csv",
    "holdout_metrics": PROXY_DIR / "holdout_role_mapping_metrics.csv",

    # Fase 07A
    "text_sentiment_segments": _first_path(
        _cfg_path(
            "SEGMENTS_WITH_SENTIMENT_CSV",
            SENTIMENT_DIR / "segments_with_sentiment_textual.csv",
        ),
        _cfg_path(
            "ALL_SEGMENTS_SENTIMENT_ENRICHED_CSV",
            SENTIMENT_DIR / "all_segments_sentiment_textual_enriched.csv",
        ),
    ),
    "call_sentiment": _cfg_path(
        "CALL_SENTIMENT_CSV",
        SENTIMENT_DIR / "call_level_sentiment_textual.csv",
    ),
    "call_role_sentiment": _cfg_path(
        "CALL_ROLE_SENTIMENT_CSV",
        SENTIMENT_DIR / "call_role_level_sentiment_textual.csv",
    ),
    "role_sentiment": _cfg_path(
        "ROLE_SENTIMENT_CSV",
        SENTIMENT_DIR / "role_level_sentiment_textual.csv",
    ),
    "sentiment_summary": _cfg_path(
        "SENTIMENT_SUMMARY_CSV",
        SENTIMENT_DIR / "sentiment_textual_summary_for_memory.csv",
    ),

    # Fase 07B
    "prosody_segments": _cfg_path(
        "SEGMENTS_PROSODY_CSV",
        PROSODY_DIR / "segments_with_audio_affect_prosody.csv",
    ),
    "call_prosody": _cfg_path(
        "CALL_PROSODY_CSV",
        PROSODY_DIR / "call_level_audio_affect_prosody.csv",
    ),
    "role_prosody": _cfg_path(
        "ROLE_PROSODY_CSV",
        PROSODY_DIR / "role_level_audio_affect_prosody.csv",
    ),
    "prosody_summary": _cfg_path(
        "PROSODY_SUMMARY_CSV",
        PROSODY_DIR / "prosody_audio_affect_summary_for_memory.csv",
    ),
    "ser_predictions": _cfg_path(
        "SER_PREDICTIONS_CSV",
        PROSODY_DIR / "ser_model_predictions.csv",
    ),
    "audio_text_comparison": _cfg_path(
        "AUDIO_TEXT_COMPARISON_CSV",
        PROSODY_DIR / "audio_vs_textual_sentiment_comparison.csv",
    ),

    # Fase 07C
    "fusion_segments": _cfg_path(
        "FUSION_SEGMENTS_CSV",
        FUSION_DIR / "segments_audio_text_fusion.csv",
    ),
    "fusion_correlations": _cfg_path(
        "FUSION_CORRELATIONS_CSV",
        FUSION_DIR / "correlations_audio_text.csv",
    ),
    "fusion_confusion": _cfg_path(
        "FUSION_CONFUSION_CSV",
        FUSION_DIR / "confusion_prosodic_state_vs_sentiment.csv",
    ),
    "fusion_disagreement": _cfg_path(
        "FUSION_DISAGREEMENT_CSV",
        FUSION_DIR / "disagreement_masked_frustration_segments.csv",
    ),
    "fusion_role_level": _cfg_path(
        "FUSION_ROLE_LEVEL_CSV",
        FUSION_DIR / "role_level_audio_text_fusion.csv",
    ),
    "fusion_call_level": _cfg_path(
        "FUSION_CALL_LEVEL_CSV",
        FUSION_DIR / "call_level_audio_text_fusion.csv",
    ),
    "fusion_summary": _cfg_path(
        "FUSION_SUMMARY_CSV",
        FUSION_DIR / "fusion_summary_for_memory.csv",
    ),

    # Fase 08
    "keyword_segments": _cfg_path(
        "SEGMENTS_WITH_KEYWORDS_CSV",
        KEYWORD_DIR / "segments_with_keywords.csv",
    ),
    "keyword_calls": _cfg_path(
        "CALL_LEVEL_KEYWORDS_CSV",
        KEYWORD_DIR / "call_level_keywords.csv",
    ),
    "critical_calls": _cfg_path(
        "TOP_CRITICAL_CALLS_CSV",
        KEYWORD_DIR / "top_critical_calls_keywords.csv",
    ),
    "keyword_sentiment_calls": _cfg_path(
        "CALL_KEYWORDS_SENTIMENT_CSV",
        KEYWORD_DIR / "call_level_keywords_sentiment_combined.csv",
    ),

    # Fase 09
    "voiceprint_candidates": VOICEPRINT_DIR / "voiceprint_segments_candidates.csv",
    "voiceprint_samples": VOICEPRINT_DIR / "voiceprint_audio_person_samples.csv",
    "voiceprint_predictions": VOICEPRINT_DIR
    / "open_set_identification_predictions.csv",
    "voiceprint_identity_summary_open_set": VOICEPRINT_DIR
    / "voiceprint_identity_summary_open_set.csv",
    "voiceprint_identity_summary": VOICEPRINT_DIR
    / "voiceprint_identity_summary.csv",
    "voiceprint_identity_split": VOICEPRINT_DIR / "voiceprint_identity_split.csv",
    "voiceprint_open_set_summary": _cfg_path(
        "VOICEPRINT_OPEN_SET_SUMMARY_CSV",
        VOICEPRINT_DIR / "voiceprint_open_set_final_summary.csv",
    ),
    "voiceprint_final_summary": _cfg_path(
        "VOICEPRINT_FINAL_SUMMARY_CSV",
        VOICEPRINT_DIR / "voiceprint_final_summary_for_memory.csv",
    ),
    "voiceprint_verification_metrics": VOICEPRINT_DIR
    / "voiceprint_verification_metrics.csv",
    "voiceprint_confusion": VOICEPRINT_DIR
    / "open_set_decision_confusion_matrix.csv",
    "voiceprint_pairwise_scores": VOICEPRINT_DIR
    / "voiceprint_pairwise_scores.csv",
}

PHASE00_DATASETS = {
    "audio_inventory",
    "metadata_snapshot",
    "cleaning_results",
    "silence_threshold_summary",
}

PHASE_LABELS = {
    "audio_inventory": "00 · Inventario",
    "metadata_snapshot": "00 · Metadata",
    "cleaning_results": "00 · Limpieza",
    "silence_threshold_summary": "00 · Limpieza",
    "diarization_summary": "01 · Diarización",
    "regular_segments": "01 · Diarización",
    "scored_segments": "01 · Diarización",
    "valid_segments": "01 · Diarización",
    "anchor_segments": "01 · Anchors",
    "relabel_summary": "01 · Reetiquetado",
    "final_segments": "01 · Reetiquetado",
    "changed_segments": "01 · Reetiquetado",
    "final_merged_segments": "04 · Consolidación",
    "relabel_margin_summary": "02 · Sensibilidad",
    "relabel_margin_by_audio": "02 · Sensibilidad",
    "overlap_threshold_summary": "02 · Sensibilidad",
    "audio_overlap_summary": "02 · Sensibilidad",
    "transcribed_segments": "05 · Transcripción",
    "transcription_summary": "05 · Transcripción",
    "transcription_final_summary": "05 · Transcripción",
    "metadata_join_audit": "06 · Proxy textual",
    "official_role_presence": "06 · Proxy textual",
    "metadata_role_summary": "06 · Proxy textual",
    "official_turns": "06 · Proxy textual",
    "alignment_candidates": "06 · Proxy textual",
    "alignment_matches": "06 · Proxy textual",
    "alignment_threshold_sensitivity": "06 · Proxy textual",
    "speaker_role_mapping": "06 · Proxy textual",
    "segment_proxy": "06 · Proxy textual",
    "proxy_metrics": "06 · Proxy textual",
    "alignment_processing_summary": "06 · Proxy textual",
    "holdout_predictions": "06 · Proxy textual",
    "holdout_metrics": "06 · Proxy textual",
    "text_sentiment_segments": "07A · Sentimiento textual",
    "call_sentiment": "07A · Sentimiento textual",
    "call_role_sentiment": "07A · Sentimiento textual",
    "role_sentiment": "07A · Sentimiento textual",
    "sentiment_summary": "07A · Sentimiento textual",
    "prosody_segments": "07B · Afecto acústico",
    "call_prosody": "07B · Afecto acústico",
    "role_prosody": "07B · Afecto acústico",
    "prosody_summary": "07B · Afecto acústico",
    "ser_predictions": "07B · Afecto acústico",
    "audio_text_comparison": "07B · Afecto acústico",
    "fusion_segments": "07C · Fusión audio-texto",
    "fusion_correlations": "07C · Fusión audio-texto",
    "fusion_confusion": "07C · Fusión audio-texto",
    "fusion_disagreement": "07C · Fusión audio-texto",
    "fusion_role_level": "07C · Fusión audio-texto",
    "fusion_call_level": "07C · Fusión audio-texto",
    "fusion_summary": "07C · Fusión audio-texto",
    "keyword_segments": "08 · Keywords",
    "keyword_calls": "08 · Keywords",
    "critical_calls": "08 · Keywords",
    "keyword_sentiment_calls": "08 · Keywords",
    "voiceprint_candidates": "09 · Huella de voz",
    "voiceprint_samples": "09 · Huella de voz",
    "voiceprint_predictions": "09 · Huella de voz",
    "voiceprint_identity_summary_open_set": "09 · Huella de voz",
    "voiceprint_identity_summary": "09 · Huella de voz",
    "voiceprint_identity_split": "09 · Huella de voz",
    "voiceprint_open_set_summary": "09 · Huella de voz",
    "voiceprint_final_summary": "09 · Huella de voz",
    "voiceprint_verification_metrics": "09 · Huella de voz",
    "voiceprint_confusion": "09 · Huella de voz",
    "voiceprint_pairwise_scores": "09 · Huella de voz",
}


# ============================================================
# GCS: SOLO DESCARGA
# ============================================================


def _gcs_uri_for_data_path(local_path: Path) -> str:
    """Mapea una ruta local dentro de data/ a su equivalente en GCS."""
    relative_path = Path(local_path).relative_to(DATA_DIR).as_posix()
    return join_gcs_uri(GCS_UNAV_ROOT, relative_path)


def _gcs_uri_for_dataset(name: str, local_path: Path) -> str:
    """Construye la URI remota respetando la estructura real del proyecto."""
    if name in PHASE00_DATASETS:
        return join_gcs_uri(GCS_UNAV_CSV_PREFIX, Path(local_path).name)
    return _gcs_uri_for_data_path(local_path)


def _restore_dataset(name: str, local_path: Path, force: bool = False) -> dict[str, Any]:
    """Restaura un CSV desde GCS sin escribir nada en el bucket."""
    if _GCS_CLIENT is None:
        raise RuntimeError(
            "El cliente GCS no está configurado. Ejecuta "
            "run_dashboard_resultados_globales(gcs_client=...)."
        )

    local_path = Path(local_path)
    gcs_uri = _gcs_uri_for_dataset(name, local_path)
    downloaded = False
    error = ""

    try:
        downloaded = bool(
            download_uri_to_local(
                source_uri=gcs_uri,
                local_path=local_path,
                gcs_client=_GCS_CLIENT,
                force=force,
            )
        )
    except Exception as exc:
        error = str(exc)

    available = local_path.exists() and local_path.stat().st_size > 0

    return {
        "dataset": name,
        "phase": PHASE_LABELS.get(name, ""),
        "local_path": str(local_path),
        "gcs_uri": gcs_uri,
        "available": available,
        "downloaded": downloaded,
        "error": error,
    }


def _restore_dashboard_inputs(force: bool = False) -> pd.DataFrame:
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
    HTML_DIR.mkdir(parents=True, exist_ok=True)

    rows = [
        _restore_dataset(name, path, force=force)
        for name, path in DATASET_PATHS.items()
    ]
    return pd.DataFrame(rows)


# ============================================================
# CARGA Y NORMALIZACIÓN
# ============================================================


def _read_csv_optional(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except EmptyDataError:
        return pd.DataFrame()
    except Exception as exc:
        print(f"No se pudo leer {path.name}: {exc}")
        return pd.DataFrame()


def _load_datasets() -> dict[str, pd.DataFrame]:
    return {
        name: _read_csv_optional(path)
        for name, path in DATASET_PATHS.items()
    }


def _first_col(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    if df.empty:
        return None
    exact = {str(column).lower(): str(column) for column in df.columns}
    for candidate in candidates:
        if candidate.lower() in exact:
            return exact[candidate.lower()]
    return None


def _audio_col(df: pd.DataFrame) -> str | None:
    return _first_col(
        df,
        [
            "audio_id_base",
            "audio_file",
            "audio_base",
            "audio_stem",
            "file_stem",
            "audio_id",
            "filename",
            "file",
            "source_file",
            "clean_audio_file",
            "call_id",
            "recording_id",
        ],
    )


def _text_col(df: pd.DataFrame) -> str | None:
    return _first_col(
        df,
        [
            "text",
            "transcription",
            "transcribed_text",
            "whisper_text",
            "segment_text",
            "texto",
            "transcripcion_whisper",
            "official_text",
        ],
    )


def _role_col(df: pd.DataFrame) -> str | None:
    return _first_col(
        df,
        [
            "official_role_proxy",
            "role_proxy",
            "proxy_role",
            "probable_role",
            "role",
            "official_role",
            "speaker_role",
        ],
    )


def _margin_col(df: pd.DataFrame) -> str | None:
    return _first_col(
        df,
        [
            "mean_label_margin",
            "distance_margin",
            "label_margin",
            "margin",
            "relabel_margin",
            "tested_relabel_margin",
        ],
    )


def _anchor_col(df: pd.DataFrame) -> str | None:
    return _first_col(
        df,
        [
            "is_anchor",
            "anchor",
            "anchor_flag",
            "selected_as_anchor",
        ],
    )


def _speaker_col(df: pd.DataFrame) -> str | None:
    return _first_col(
        df,
        [
            "speaker_final",
            "speaker",
            "speaker_original",
            "label",
        ],
    )


def _normalize_role(value: Any) -> str:
    text = str(value).strip().lower()
    if text in {"", "nan", "none", "null", "<na>"}:
        return "Sin rol"
    if "client" in text or "cliente" in text or text in {"customer", "user"}:
        return "Cliente"
    if "agent" in text or "agente" in text or "operator" in text:
        return "Agente"
    return str(value).strip().title()


def _normalize_corpus_value(value: Any) -> str:
    text = str(value).strip().lower()
    if "baja" in text:
        return "Bajas"
    if "comercial" in text or "venta" in text:
        return "Comerciales"
    if text in {"raw", "general", "comerciales"}:
        return "Comerciales"
    return "No identificado"


def _add_filter_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    output = df.copy()

    corpus_col = _first_col(
        output,
        [
            "corpus",
            "source_corpus",
            "dataset",
            "source_type",
            "audio_source",
            "source",
        ],
    )
    audio_col = _audio_col(output)

    if corpus_col:
        output["_dashboard_corpus"] = output[corpus_col].apply(
            _normalize_corpus_value
        )
    elif audio_col:
        output["_dashboard_corpus"] = output[audio_col].apply(
            _normalize_corpus_value
        )
    else:
        output["_dashboard_corpus"] = "No identificado"

    role_col = _role_col(output)
    if role_col:
        output["_dashboard_role"] = output[role_col].apply(_normalize_role)
    else:
        output["_dashboard_role"] = "Sin rol"

    margin_col = _margin_col(output)
    if margin_col:
        output["_dashboard_margin"] = pd.to_numeric(
            output[margin_col],
            errors="coerce",
        )

    anchor_col = _anchor_col(output)
    if anchor_col:
        output["_dashboard_anchor"] = (
            output[anchor_col]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin({"true", "1", "yes", "si", "sí", "anchor"})
        )

    return output


def _prepare_datasets(
    datasets: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    return {
        name: _add_filter_columns(df)
        for name, df in datasets.items()
    }


def _apply_filters(
    df: pd.DataFrame,
    corpus: str,
    role: str,
    margin_range: tuple[float, float],
    anchor_mode: str,
) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    output = df.copy()

    if corpus != "Todos" and "_dashboard_corpus" in output.columns:
        output = output[output["_dashboard_corpus"].eq(corpus)]

    if role != "Todos" and "_dashboard_role" in output.columns:
        output = output[output["_dashboard_role"].eq(role)]

    if "_dashboard_margin" in output.columns:
        minimum, maximum = margin_range
        margin_values = pd.to_numeric(
            output["_dashboard_margin"],
            errors="coerce",
        )
        output = output[
            margin_values.isna()
            | margin_values.between(minimum, maximum, inclusive="both")
        ]

    if anchor_mode != "Todos" and "_dashboard_anchor" in output.columns:
        keep_anchor = anchor_mode == "Solo anchors"
        output = output[output["_dashboard_anchor"].eq(keep_anchor)]

    return output




# ============================================================
# DASHBOARD HTML AUTÓNOMO
# ============================================================


def _global_margin_range(
    datasets: dict[str, pd.DataFrame],
) -> tuple[float, float]:
    values: list[pd.Series] = []
    for name in ["final_segments", "changed_segments", "relabel_margin_by_audio"]:
        df = datasets.get(name, pd.DataFrame())
        margin_col = _margin_col(df)
        if not df.empty and margin_col:
            series = pd.to_numeric(df[margin_col], errors="coerce").dropna()
            if not series.empty:
                values.append(series)

    if not values:
        return 0.0, 1.0

    combined = pd.concat(values, ignore_index=True)
    minimum = float(combined.min())
    maximum = float(combined.max())

    if math.isclose(minimum, maximum):
        maximum = minimum + 0.01

    return minimum, maximum


def _global_roles(datasets: dict[str, pd.DataFrame]) -> list[str]:
    roles: set[str] = set()
    for df in datasets.values():
        if "_dashboard_role" in df.columns:
            roles.update(
                role
                for role in df["_dashboard_role"].dropna().astype(str).unique()
                if role not in {"", "Sin rol", "Nan", "None", "<NA>"}
            )

    preferred = [role for role in ["Cliente", "Agente"] if role in roles]
    remaining = sorted(roles - set(preferred))
    return ["Todos", *preferred, *remaining]


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if pd.isna(value):
        return None
    return str(value)


def _json_dumps(payload: Any) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        default=_json_default,
        separators=(",", ":"),
    )


def _series_from_candidates(
    df: pd.DataFrame,
    candidates: Iterable[str],
    default: Any = None,
) -> pd.Series:
    column = _first_col(df, candidates)
    if column:
        return df[column]
    return pd.Series([default] * len(df), index=df.index)


def _numeric_series(
    df: pd.DataFrame,
    candidates: Iterable[str],
) -> pd.Series:
    return pd.to_numeric(
        _series_from_candidates(df, candidates, np.nan),
        errors="coerce",
    )


def _boolean_series(
    df: pd.DataFrame,
    candidates: Iterable[str],
) -> pd.Series:
    column = _first_col(df, candidates)
    if not column:
        return pd.Series(pd.array([pd.NA] * len(df), dtype="boolean"), index=df.index)

    values = df[column]
    normalized = values.astype(str).str.strip().str.lower()
    result = normalized.isin(
        {"true", "1", "yes", "si", "sí", "same", "positive", "match", "known"}
    ).astype("boolean")
    result[values.isna()] = pd.NA
    return result


def _base_dashboard_frame(df: pd.DataFrame) -> pd.DataFrame:
    output = pd.DataFrame(index=df.index)
    audio_col = _audio_col(df)
    output["audio"] = (
        df[audio_col].astype("string")
        if audio_col
        else pd.Series([pd.NA] * len(df), index=df.index, dtype="string")
    )
    output["corpus"] = df.get(
        "_dashboard_corpus",
        pd.Series(["No identificado"] * len(df), index=df.index),
    ).astype("string")
    output["role"] = df.get(
        "_dashboard_role",
        pd.Series(["Sin rol"] * len(df), index=df.index),
    ).astype("string")
    output["margin"] = pd.to_numeric(
        df.get(
            "_dashboard_margin",
            pd.Series([np.nan] * len(df), index=df.index),
        ),
        errors="coerce",
    )
    anchor_values = df.get(
        "_dashboard_anchor",
        pd.Series([pd.NA] * len(df), index=df.index),
    )
    output["anchor"] = anchor_values.astype("boolean")
    return output


def _clean_records(df: pd.DataFrame, max_rows: int | None = None) -> list[dict[str, Any]]:
    if df.empty:
        return []

    output = df.copy()
    if max_rows is not None and len(output) > max_rows:
        output = output.sample(n=max_rows, random_state=42)

    output = output.replace({np.inf: np.nan, -np.inf: np.nan})
    output = output.astype(object).where(pd.notna(output), None)
    return output.to_dict(orient="records")


def _build_final_segment_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty:
        return []
    output = _base_dashboard_frame(df)
    start = _numeric_series(df, ["start", "segment_start", "start_sec"])
    end = _numeric_series(df, ["end", "segment_end", "end_sec"])
    duration = _numeric_series(df, ["duration", "duration_sec", "segment_duration"])
    duration = duration.where(duration.notna(), end - start)
    output["start"] = start
    output["end"] = end
    output["duration"] = duration
    output["overlap"] = _numeric_series(df, ["overlap_ratio", "overlap"])
    output["speaker"] = _series_from_candidates(
        df,
        ["speaker_final", "speaker", "speaker_original", "label"],
        "Sin speaker",
    ).astype("string")
    output["speaker_original"] = _series_from_candidates(
        df,
        ["speaker_original", "speaker", "label"],
        "",
    ).astype("string")
    output["changed"] = _boolean_series(
        df,
        ["was_reclassified", "was_relabelled", "changed"],
    )
    return _clean_records(output)


def _build_scored_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty:
        return []
    output = _base_dashboard_frame(df)
    output["overlap"] = _numeric_series(df, ["overlap_ratio", "overlap"])
    output["speaker"] = _series_from_candidates(
        df,
        ["speaker_final", "speaker", "speaker_original", "label"],
        "Sin speaker",
    ).astype("string")
    return _clean_records(output)


def _build_anchor_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty:
        return []
    output = _base_dashboard_frame(df)
    output["anchor"] = True
    output["speaker"] = _series_from_candidates(
        df,
        ["speaker_final", "speaker", "speaker_original", "label"],
        "Sin speaker",
    ).astype("string")
    return _clean_records(output)


def _build_audio_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty:
        return []
    output = _base_dashboard_frame(df)
    output["before_sec"] = _numeric_series(
        df,
        [
            "duration_before_sec",
            "original_duration_sec",
            "duration_original_sec",
            "raw_duration_sec",
            "duration_before",
        ],
    )
    output["after_sec"] = _numeric_series(
        df,
        [
            "duration_after_sec",
            "clean_duration_sec",
            "duration_clean_sec",
            "processed_duration_sec",
            "duration_after",
        ],
    )
    return _clean_records(output)


def _build_transcription_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty:
        return []
    output = _base_dashboard_frame(df)
    text_col = _text_col(df)
    text = (
        df[text_col].fillna("").astype(str)
        if text_col
        else pd.Series([""] * len(df), index=df.index)
    )
    output["has_text"] = text.str.strip().ne("")
    output["word_count"] = text.str.split().str.len().fillna(0).astype(int)
    output["text"] = text.str.slice(0, 180)
    output["start"] = _numeric_series(df, ["start", "segment_start", "start_sec"])
    output["end"] = _numeric_series(df, ["end", "segment_end", "end_sec"])
    output["speaker"] = _series_from_candidates(
        df,
        ["speaker_final", "speaker", "speaker_original", "label"],
        "",
    ).astype("string")
    output["status"] = _series_from_candidates(
        df,
        ["transcription_status", "status", "alignment_status"],
        "",
    ).astype("string")
    return _clean_records(output, max_rows=60000)


def _build_mapping_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty:
        return []
    output = _base_dashboard_frame(df)
    output["status"] = _series_from_candidates(
        df,
        ["role_mapping_status", "mapping_status", "status"],
        "Sin estado",
    ).fillna("Sin estado").astype("string")
    return _clean_records(output)


def _build_sentiment_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty:
        return []
    output = _base_dashboard_frame(df)
    output["sentiment"] = _series_from_candidates(
        df,
        [
            "sentiment_label",
            "sentiment",
            "label_sentiment",
            "sentiment_category",
            "sentiment_textual",
        ],
        "Sin etiqueta",
    ).fillna("Sin etiqueta").astype("string")
    text_col = _text_col(df)
    if text_col:
        output["text"] = df[text_col].fillna("").astype(str).str.slice(0, 180)
    else:
        output["text"] = ""
    return _clean_records(output, max_rows=60000)


def _build_prosody_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty:
        return []
    output = _base_dashboard_frame(df)
    output["prosody"] = _series_from_candidates(
        df,
        [
            "prosodic_state",
            "audio_affect_label",
            "audio_emotion",
            "ser_label",
            "emotion_label",
            "predicted_emotion",
            "affective_state",
        ],
        "Sin etiqueta",
    ).fillna("Sin etiqueta").astype("string")
    output["arousal"] = _numeric_series(
        df,
        ["arousal", "arousal_score", "mean_arousal"],
    )
    output["tension"] = _numeric_series(
        df,
        ["tension", "tension_score", "mean_tension"],
    )
    return _clean_records(output, max_rows=60000)


def _build_fusion_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty:
        return []
    output = _base_dashboard_frame(df)
    output["sentiment"] = _series_from_candidates(
        df,
        [
            "sentiment_label",
            "sentiment",
            "text_sentiment_label",
            "sentiment_textual",
            "textual_sentiment",
        ],
        "Sin etiqueta",
    ).fillna("Sin etiqueta").astype("string")
    output["prosody"] = _series_from_candidates(
        df,
        [
            "prosodic_state",
            "audio_affect_label",
            "audio_emotion",
            "ser_label",
            "emotion_label",
            "predicted_emotion",
        ],
        "Sin etiqueta",
    ).fillna("Sin etiqueta").astype("string")
    output["disagreement"] = _boolean_series(
        df,
        [
            "is_disagreement",
            "disagreement_flag",
            "masked_frustration_flag",
            "audio_text_disagreement",
        ],
    )
    text_col = _text_col(df)
    output["text"] = (
        df[text_col].fillna("").astype(str).str.slice(0, 180)
        if text_col
        else ""
    )
    return _clean_records(output, max_rows=60000)


def _keyword_count_columns(df: pd.DataFrame) -> list[str]:
    return [
        str(column)
        for column in df.columns
        if str(column).lower().endswith("_count")
        and str(column).lower()
        not in {"keyword_total_count", "total_keyword_count"}
    ]


def _build_keyword_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty:
        return []
    output = _base_dashboard_frame(df)
    output["criticality"] = _numeric_series(
        df,
        ["criticality_score", "critical_score", "risk_score"],
    )
    output["critical"] = _boolean_series(
        df,
        ["has_critical_keyword", "has_keyword", "critical_keyword_flag"],
    )
    output["keyword_total"] = _numeric_series(
        df,
        ["keyword_total_count", "total_keyword_count", "n_keywords"],
    )
    output["category"] = _series_from_candidates(
        df,
        ["keyword", "keyword_category", "topic", "theme", "tema"],
        "",
    ).astype("string")
    text_col = _text_col(df)
    output["text"] = (
        df[text_col].fillna("").astype(str).str.slice(0, 180)
        if text_col
        else ""
    )

    count_columns = _keyword_count_columns(df)
    for column in count_columns[:30]:
        output[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)

    return _clean_records(output, max_rows=60000)


def _build_voiceprint_pair_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty:
        return []
    output = _base_dashboard_frame(df)
    output["score"] = _numeric_series(
        df,
        [
            "similarity_score",
            "cosine_similarity",
            "verification_score",
            "best_similarity",
            "top1_score",
            "score",
        ],
    )
    output["same"] = _boolean_series(
        df,
        ["is_same_identity", "same_identity", "true_match", "target", "label"],
    )
    return _clean_records(output, max_rows=100000)


def _build_voiceprint_prediction_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty:
        return []
    output = _base_dashboard_frame(df)
    output["decision"] = _series_from_candidates(
        df,
        ["decision", "prediction", "open_set_decision", "predicted_status"],
        "Sin decisión",
    ).fillna("Sin decisión").astype("string")
    output["correct"] = _boolean_series(
        df,
        ["identification_correct", "correct", "is_correct"],
    )
    output["true_enrolled"] = _boolean_series(
        df,
        ["true_is_enrolled", "is_enrolled", "known_identity"],
    )
    output["score"] = _numeric_series(
        df,
        ["best_similarity", "top1_score", "similarity_score", "score"],
    )
    output["predicted_id"] = _series_from_candidates(
        df,
        ["predicted_person_id", "predicted_identity", "predicted_id"],
        "",
    ).astype("string")
    output["true_id"] = _series_from_candidates(
        df,
        ["true_person_id", "true_identity", "person_id"],
        "",
    ).astype("string")
    return _clean_records(output, max_rows=60000)


def _small_table_payload(df: pd.DataFrame, max_rows: int = 250) -> list[dict[str, Any]]:
    if df.empty:
        return []
    selected = df.head(max_rows).copy()
    selected = selected.replace({np.inf: np.nan, -np.inf: np.nan})
    selected = selected.astype(object).where(pd.notna(selected), None)
    return selected.to_dict(orient="records")


def _build_dashboard_payload(
    datasets: dict[str, pd.DataFrame],
    availability: pd.DataFrame,
) -> dict[str, Any]:
    keyword_source = datasets.get("keyword_sentiment_calls", pd.DataFrame())
    if keyword_source.empty:
        keyword_source = datasets.get("keyword_calls", pd.DataFrame())

    payload = {
        "generated_at": datetime.now().strftime("%d/%m/%Y %H:%M"),
        "margin_range": list(_global_margin_range(datasets)),
        "roles": _global_roles(datasets),
        "availability": _small_table_payload(
            availability[
                ["phase", "dataset", "available", "downloaded", "error"]
            ].copy(),
            max_rows=500,
        ),
        "audio_inventory": _build_audio_records(
            datasets.get("audio_inventory", pd.DataFrame())
        ),
        "cleaning_results": _build_audio_records(
            datasets.get("cleaning_results", pd.DataFrame())
        ),
        "diarization_summary": _build_audio_records(
            datasets.get("diarization_summary", pd.DataFrame())
        ),
        "final_segments": _build_final_segment_records(
            datasets.get("final_segments", pd.DataFrame())
        ),
        "scored_segments": _build_scored_records(
            datasets.get("scored_segments", pd.DataFrame())
        ),
        "anchor_segments": _build_anchor_records(
            datasets.get("anchor_segments", pd.DataFrame())
        ),
        "transcribed_segments": _build_transcription_records(
            datasets.get("transcribed_segments", pd.DataFrame())
        ),
        "segment_proxy": _build_transcription_records(
            datasets.get("segment_proxy", pd.DataFrame())
        ),
        "speaker_role_mapping": _build_mapping_records(
            datasets.get("speaker_role_mapping", pd.DataFrame())
        ),
        "text_sentiment": _build_sentiment_records(
            datasets.get("text_sentiment_segments", pd.DataFrame())
        ),
        "prosody": _build_prosody_records(
            datasets.get("prosody_segments", pd.DataFrame())
        ),
        "fusion": _build_fusion_records(
            datasets.get("fusion_segments", pd.DataFrame())
        ),
        "keyword_segments": _build_keyword_records(
            datasets.get("keyword_segments", pd.DataFrame())
        ),
        "keyword_calls": _build_keyword_records(keyword_source),
        "critical_calls": _build_keyword_records(
            datasets.get("critical_calls", pd.DataFrame())
        ),
        "voiceprint_pairs": _build_voiceprint_pair_records(
            datasets.get("voiceprint_pairwise_scores", pd.DataFrame())
        ),
        "voiceprint_predictions": _build_voiceprint_prediction_records(
            datasets.get("voiceprint_predictions", pd.DataFrame())
        ),
        "voiceprint_confusion": _small_table_payload(
            datasets.get("voiceprint_confusion", pd.DataFrame()),
            max_rows=100,
        ),
        "voiceprint_metrics": _small_table_payload(
            datasets.get("voiceprint_verification_metrics", pd.DataFrame()),
            max_rows=200,
        ),
        "voiceprint_summary": _small_table_payload(
            datasets.get("voiceprint_open_set_summary", pd.DataFrame()),
            max_rows=200,
        ),
    }
    return payload


DASHBOARD_CSS = r"""
:root {
  --navy:#123047; --blue:#2F6BFF; --green:#2A9D6F; --orange:#F28E2B;
  --red:#D1495B; --purple:#7B61A8; --gray:#6F7C85; --line:#DDE5EA;
  --light:#F4F7FA; --white:#FFFFFF;
}
* { box-sizing:border-box; }
body { margin:0; background:#EEF3F6; color:#263843; font-family:Inter,Arial,sans-serif; }
.tfm-page { width:min(1500px,97vw); margin:18px auto 40px; }
.tfm-header { background:linear-gradient(135deg,var(--navy),var(--blue)); color:white;
  border-radius:16px; padding:22px 26px; box-shadow:0 8px 24px rgba(18,48,71,.16); }
.tfm-title { font-size:27px; font-weight:760; }
.tfm-subtitle { opacity:.92; font-size:13px; margin-top:5px; }
.tfm-toolbar { display:grid; grid-template-columns:repeat(4,minmax(170px,1fr)); gap:10px;
  background:white; border:1px solid var(--line); border-radius:14px; padding:13px;
  margin:12px 0; position:sticky; top:5px; z-index:20; box-shadow:0 3px 12px rgba(18,48,71,.08); }
.tfm-control label { display:block; font-size:11px; font-weight:700; color:#5F6D77; margin-bottom:4px; }
.tfm-control select,.tfm-control input { width:100%; border:1px solid #CBD6DD; border-radius:8px;
  padding:8px 9px; background:white; color:#243845; }
.tfm-range-values { display:flex; gap:8px; font-size:11px; color:#5F6D77; margin-top:4px; }
.tfm-actions { grid-column:1/-1; display:flex; gap:9px; flex-wrap:wrap; align-items:center; }
.tfm-btn { border:0; border-radius:8px; padding:9px 14px; font-weight:700; cursor:pointer; }
.tfm-btn.primary { background:var(--blue); color:white; }
.tfm-btn.secondary { background:#E9EFF3; color:var(--navy); }
.tfm-btn.export { background:var(--green); color:white; margin-left:auto; }
.tfm-note { background:#F4F7FA; border-left:4px solid var(--blue); border-radius:8px;
  padding:10px 12px; margin:10px 0; font-size:12px; color:#4B5A64; }
.tfm-tabs { display:flex; flex-wrap:wrap; gap:7px; margin:12px 0 8px; }
.tfm-tab { border:1px solid var(--line); background:white; color:var(--navy); border-radius:9px;
  padding:9px 13px; font-weight:700; cursor:pointer; }
.tfm-tab.active { background:var(--navy); color:white; border-color:var(--navy); }
.tfm-panel { display:none; background:white; border:1px solid var(--line); border-radius:14px;
  padding:17px; min-height:340px; box-shadow:0 4px 16px rgba(18,48,71,.06); }
.tfm-panel.active { display:block; }
.tfm-kpis { display:grid; grid-template-columns:repeat(4,minmax(150px,1fr)); gap:10px; margin-bottom:12px; }
.tfm-kpi { border:1px solid var(--line); background:#FBFCFD; border-radius:11px; padding:12px; }
.tfm-kpi .value { color:var(--navy); font-size:22px; font-weight:760; }
.tfm-kpi .label { color:#66747D; font-size:11px; margin-top:4px; }
.tfm-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:12px; }
.tfm-card { border:1px solid var(--line); border-radius:12px; padding:9px; background:white; min-height:320px; }
.tfm-card.full { grid-column:1/-1; }
.tfm-chart { width:100%; height:360px; }
.tfm-section-title { font-size:17px; font-weight:750; color:var(--navy); margin:4px 0 9px; }
.tfm-table-wrap { max-height:390px; overflow:auto; border:1px solid var(--line); border-radius:10px; }
table { width:100%; border-collapse:collapse; font-size:11px; }
th { position:sticky; top:0; background:#EDF3F7; color:var(--navy); text-align:left; z-index:1; }
th,td { padding:7px 8px; border-bottom:1px solid #E8EDF0; vertical-align:top; }
tr:nth-child(even) td { background:#FBFCFD; }
.tfm-empty { min-height:260px; display:flex; align-items:center; justify-content:center;
  color:#71808A; background:#F8FAFB; border-radius:9px; text-align:center; padding:20px; }
.tfm-filter-summary { font-size:12px; color:#52616B; margin-left:auto; }
@media(max-width:900px){ .tfm-toolbar{grid-template-columns:1fr 1fr}.tfm-grid{grid-template-columns:1fr}.tfm-kpis{grid-template-columns:1fr 1fr}.tfm-card.full{grid-column:auto} }
"""


DASHBOARD_JS = r"""
(function(){
const DATA = window.__TFM_DASHBOARD_DATA__;
const COLORS = {blue:'#2F6BFF',navy:'#123047',orange:'#F28E2B',green:'#2A9D6F',red:'#D1495B',purple:'#7B61A8',gray:'#7A8288'};
const cfg = {responsive:true,displaylogo:false,modeBarButtonsToRemove:['lasso2d','select2d']};
const $ = (id) => document.getElementById(id);
const esc = (value) => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[c]));
const num = (value) => { const x=Number(value); return Number.isFinite(x)?x:null; };
const fmtInt = (value) => Number(value||0).toLocaleString('es-ES');
const fmtPct = (value) => Number.isFinite(value) ? (100*value).toFixed(1).replace('.',',')+' %' : 'N/D';
const uniqueAudio = (rows) => new Set(rows.map(r=>r.audio).filter(Boolean)).size;
const countBy = (rows,key) => { const out={}; rows.forEach(r=>{ const k=String(r[key]??'Sin dato'); out[k]=(out[k]||0)+1; }); return out; };
const sum = (rows,key) => rows.reduce((a,r)=>a+(num(r[key])||0),0);
const truthy = (v) => v===true || ['true','1','yes','si','sí'].includes(String(v).toLowerCase());

function currentFilters(){
  return {
    corpus:$('filter-corpus').value,
    role:$('filter-role').value,
    marginMin:Math.min(Number($('filter-margin-min').value),Number($('filter-margin-max').value)),
    marginMax:Math.max(Number($('filter-margin-min').value),Number($('filter-margin-max').value)),
    anchor:$('filter-anchor').value
  };
}
function filtered(rows){
  const f=currentFilters();
  return (rows||[]).filter(r=>{
    if(f.corpus!=='Todos' && String(r.corpus)!==f.corpus) return false;
    if(f.role!=='Todos' && String(r.role)!==f.role) return false;
    const m=num(r.margin); if(m!==null && (m<f.marginMin || m>f.marginMax)) return false;
    if(f.anchor!=='Todos' && r.anchor!==null && r.anchor!==undefined){
      const wanted=f.anchor==='Solo anchors'; if(truthy(r.anchor)!==wanted) return false;
    }
    return true;
  });
}
function kpis(containerId,items){
  $(containerId).innerHTML=items.map(x=>`<div class="tfm-kpi"><div class="value">${esc(x[1])}</div><div class="label">${esc(x[0])}</div></div>`).join('');
}
function emptyPlot(id,message){ $(id).innerHTML=`<div class="tfm-empty">${esc(message)}</div>`; }
function bar(id,counts,title,color=COLORS.blue,horizontal=false){
  const entries=Object.entries(counts).sort((a,b)=>b[1]-a[1]);
  if(!entries.length){emptyPlot(id,'No hay datos disponibles para esta vista.');return;}
  const labels=entries.map(x=>x[0]), values=entries.map(x=>x[1]);
  const trace=horizontal?{type:'bar',x:values,y:labels,orientation:'h',marker:{color},text:values,textposition:'auto'}:{type:'bar',x:labels,y:values,marker:{color},text:values,textposition:'auto'};
  Plotly.react(id,[trace],{title:{text:title,font:{size:16}},margin:{l:horizontal?150:55,r:20,t:50,b:70},paper_bgcolor:'white',plot_bgcolor:'white'},cfg);
}
function histogram(id,values,title,color=COLORS.blue,groups=null){
  const clean=values.map(num).filter(v=>v!==null);
  if(!clean.length){emptyPlot(id,'No hay valores numéricos disponibles.');return;}
  let traces;
  if(groups){
    const grouped={}; values.forEach((v,i)=>{const n=num(v);if(n===null)return;const g=String(groups[i]??'Sin grupo');(grouped[g]??=[]).push(n);});
    traces=Object.entries(grouped).map(([name,x],i)=>({type:'histogram',x,name,opacity:.68,nbinsx:40}));
  } else traces=[{type:'histogram',x:clean,marker:{color},nbinsx:40}];
  Plotly.react(id,traces,{barmode:'overlay',title:{text:title,font:{size:16}},margin:{l:55,r:20,t:50,b:55},paper_bgcolor:'white',plot_bgcolor:'white'},cfg);
}
function table(containerId,rows,columns,maxRows=100){
  const root=$(containerId); if(!rows.length){root.innerHTML='<div class="tfm-empty">No hay filas para los filtros actuales.</div>';return;}
  const cols=columns.filter(c=>rows.some(r=>r[c]!==undefined && r[c]!==null && String(r[c])!==''));
  const body=rows.slice(0,maxRows).map(r=>'<tr>'+cols.map(c=>`<td>${esc(r[c])}</td>`).join('')+'</tr>').join('');
  root.innerHTML=`<div class="tfm-table-wrap"><table><thead><tr>${cols.map(c=>`<th>${esc(c)}</th>`).join('')}</tr></thead><tbody>${body}</tbody></table></div>`;
}
function heatmap(id,rows,xKey,yKey,title){
  const xVals=[...new Set(rows.map(r=>String(r[xKey]??'Sin dato')))];
  const yVals=[...new Set(rows.map(r=>String(r[yKey]??'Sin dato')))];
  if(!xVals.length || !yVals.length){emptyPlot(id,'No hay datos suficientes para la matriz.');return;}
  const z=yVals.map(y=>xVals.map(x=>rows.filter(r=>String(r[xKey]??'Sin dato')===x && String(r[yKey]??'Sin dato')===y).length));
  Plotly.react(id,[{type:'heatmap',x:xVals,y:yVals,z,colorscale:'Blues',text:z,texttemplate:'%{text}'}],{title:{text:title,font:{size:16}},margin:{l:110,r:20,t:50,b:80}},cfg);
}
function renderSummary(){
  const inv=filtered(DATA.audio_inventory), clean=filtered(DATA.cleaning_results), final=filtered(DATA.final_segments), trans=filtered(DATA.transcribed_segments), proxy=filtered(DATA.segment_proxy), kw=filtered(DATA.keyword_calls), vp=filtered(DATA.voiceprint_predictions);
  const withText=trans.filter(r=>truthy(r.has_text)).length;
  const vpEvaluable=vp.filter(r=>r.correct!==null && r.correct!==undefined);
  const vpCorrect=vpEvaluable.length?vpEvaluable.filter(r=>truthy(r.correct)).length/vpEvaluable.length:NaN;
  kpis('summary-kpis',[
    ['Audios inventariados',fmtInt(uniqueAudio(inv))],
    ['Horas antes / después',(sum(clean,'before_sec')/3600).toFixed(1).replace('.',',')+' / '+(sum(clean,'after_sec')/3600).toFixed(1).replace('.',',')],
    ['Segmentos finales',fmtInt(final.length)],
    ['Cobertura de transcripción',fmtPct(trans.length?withText/trans.length:NaN)],
    ['Segmentos con rol proxy',fmtInt(proxy.filter(r=>r.role && r.role!=='Sin rol').length)],
    ['Llamadas con keywords',fmtInt(uniqueAudio(kw))],
    ['Consultas huella de voz',fmtInt(vp.length)],
    ['Exactitud open-set',fmtPct(vpCorrect)]
  ]);
  const stages=['Inventario','Diarización final','Transcripción','Roles proxy','Keywords'];
  const vals=[uniqueAudio(inv),uniqueAudio(final),uniqueAudio(trans),uniqueAudio(proxy),uniqueAudio(kw)];
  Plotly.react('summary-funnel',[{type:'funnel',y:stages,x:vals,textinfo:'value+percent initial',marker:{color:[COLORS.navy,COLORS.blue,COLORS.green,COLORS.orange,COLORS.purple]}}],{title:{text:'Cobertura acumulada del pipeline',font:{size:16}},margin:{l:120,r:20,t:50,b:40}},cfg);
  const available=DATA.availability.filter(r=>truthy(r.available)).length;
  const total=DATA.availability.length;
  bar('summary-availability',{'Disponibles':available,'No disponibles':Math.max(total-available,0)},'Disponibilidad de outputs',COLORS.green);
  table('summary-table',DATA.availability,['phase','dataset','available','downloaded','error'],200);
}
function renderDiarization(){
  const final=filtered(DATA.final_segments), scored=filtered(DATA.scored_segments), anchors=filtered(DATA.anchor_segments);
  kpis('diarization-kpis',[
    ['Segmentos puntuados',fmtInt(scored.length)],['Anchors',fmtInt(anchors.length)],['Segmentos finales',fmtInt(final.length)],['Reetiquetados',fmtInt(final.filter(r=>truthy(r.changed)).length)]
  ]);
  histogram('margin-hist',final.map(r=>r.margin),'Distribución del margen de reetiquetado',COLORS.blue,final.map(r=>truthy(r.changed)?'Reetiquetado':'Sin cambio'));
  histogram('overlap-hist',scored.map(r=>r.overlap),'Distribución del solapamiento',COLORS.orange);
  bar('anchor-bar',countBy(anchors,'speaker'),'Anchors por speaker',COLORS.green);
  histogram('duration-hist',final.map(r=>r.duration),'Duración de segmentos finales',COLORS.purple);
  table('diarization-table',final,['audio','start','end','duration','speaker_original','speaker','changed','margin','anchor'],120);
}
function renderTranscription(){
  const trans=filtered(DATA.transcribed_segments), proxy=filtered(DATA.segment_proxy), mappings=filtered(DATA.speaker_role_mapping);
  const withText=trans.filter(r=>truthy(r.has_text));
  kpis('transcription-kpis',[
    ['Segmentos transcritos',fmtInt(trans.length)],['Con texto',fmtInt(withText.length)],['Con rol proxy',fmtInt(proxy.filter(r=>r.role && r.role!=='Sin rol').length)],['Mappings speaker→rol',fmtInt(mappings.length)]
  ]);
  bar('coverage-bar',{'Con texto':withText.length,'Sin texto':trans.length-withText.length},'Cobertura de transcripción',COLORS.green);
  histogram('words-hist',withText.map(r=>r.word_count),'Longitud textual por segmento',COLORS.blue);
  bar('roles-bar',countBy(proxy,'role'),'Distribución de roles',COLORS.orange);
  bar('mapping-bar',countBy(mappings,'status'),'Estado del mapping speaker → rol',COLORS.purple,true);
  table('transcription-table',proxy.length?proxy:trans,['audio','start','end','speaker','role','status','text'],120);
}
function renderSentiment(){
  const text=filtered(DATA.text_sentiment), prosody=filtered(DATA.prosody), fusion=filtered(DATA.fusion);
  kpis('sentiment-kpis',[
    ['Segmentos con sentimiento textual',fmtInt(text.length)],['Segmentos con afecto acústico',fmtInt(prosody.length)],['Segmentos fusionados',fmtInt(fusion.length)],['Desacuerdos detectados',fmtInt(fusion.filter(r=>truthy(r.disagreement)).length)]
  ]);
  bar('text-sentiment-bar',countBy(text,'sentiment'),'Sentimiento textual',COLORS.blue);
  bar('prosody-bar',countBy(prosody,'prosody'),'Estado afectivo acústico',COLORS.orange);
  heatmap('fusion-heatmap',fusion,'sentiment','prosody','Relación audio–texto');
  const disagreements=fusion.filter(r=>truthy(r.disagreement));
  table('sentiment-table',disagreements.length?disagreements:fusion,['audio','role','sentiment','prosody','disagreement','text'],120);
}
function renderKeywords(){
  const seg=filtered(DATA.keyword_segments), calls=filtered(DATA.keyword_calls), critical=filtered(DATA.critical_calls);
  const segCriticalKnown=seg.filter(r=>r.critical!==null && r.critical!==undefined);
  const totals={}; calls.forEach(r=>Object.keys(r).filter(k=>k.endsWith('_count') && !['keyword_total_count','total_keyword_count'].includes(k)).forEach(k=>{totals[k]=(totals[k]||0)+(num(r[k])||0);}));
  const labels={}; Object.entries(totals).forEach(([k,v])=>{labels[k.replace(/^kw_/,'').replace(/_count$/,'').replaceAll('_',' ')]=v;});
  kpis('keyword-kpis',[
    ['Segmentos analizados',fmtInt(seg.length)],['Llamadas analizadas',fmtInt(calls.length)],['Llamadas priorizadas',fmtInt(critical.length)],['Categorías detectadas',fmtInt(Object.keys(labels).length)]
  ]);
  bar('keywords-bar',labels,'Keywords o temas más frecuentes',COLORS.orange,true);
  histogram('criticality-hist',calls.map(r=>r.criticality),'Criticidad por llamada',COLORS.red);
  bar('critical-coverage',{'Con keyword crítica':segCriticalKnown.filter(r=>truthy(r.critical)).length,'Sin keyword crítica':segCriticalKnown.filter(r=>!truthy(r.critical)).length},'Cobertura de keywords críticas',COLORS.red);
  const tableRows=(critical.length?critical:calls).slice().sort((a,b)=>(num(b.criticality)||0)-(num(a.criticality)||0));
  table('keyword-table',tableRows,['audio','role','criticality','keyword_total','critical','category','text'],120);
}
function renderVoiceprint(){
  const pairs=filtered(DATA.voiceprint_pairs), pred=filtered(DATA.voiceprint_predictions);
  const labelledPairs=pairs.filter(r=>r.same!==null && r.same!==undefined);
  const same=labelledPairs.filter(r=>truthy(r.same)), diff=labelledPairs.filter(r=>!truthy(r.same));
  kpis('voiceprint-kpis',[
    ['Pares evaluados',fmtInt(pairs.length)],['Misma identidad',fmtInt(same.length)],['Consultas open-set',fmtInt(pred.length)],['Predicciones correctas',fmtInt(pred.filter(r=>truthy(r.correct)).length)]
  ]);
  if(pairs.some(r=>num(r.score)!==null)){
    Plotly.react('voice-score-hist',[
      {type:'histogram',x:same.map(r=>num(r.score)).filter(v=>v!==null),name:'Misma identidad',opacity:.68,nbinsx:45},
      {type:'histogram',x:diff.map(r=>num(r.score)).filter(v=>v!==null),name:'Identidad distinta',opacity:.68,nbinsx:45}
    ],{barmode:'overlay',title:{text:'Distribución de scores de similitud vocal',font:{size:16}},margin:{l:55,r:20,t:50,b:55}},cfg);
  } else emptyPlot('voice-score-hist','No hay scores de similitud disponibles.');
  bar('voice-decision-bar',countBy(pred,'decision'),'Decisiones open-set',COLORS.purple);
  const conf=DATA.voiceprint_confusion;
  if(conf.length){
    const labelKey=Object.keys(conf[0]).find(k=>typeof conf[0][k]==='string') || Object.keys(conf[0])[0];
    const cols=Object.keys(conf[0]).filter(k=>k!==labelKey);
    const z=conf.map(r=>cols.map(c=>num(r[c])||0));
    Plotly.react('voice-confusion',[{type:'heatmap',x:cols,y:conf.map(r=>String(r[labelKey])),z,colorscale:'Blues',text:z,texttemplate:'%{text}'}],{title:{text:'Matriz de decisiones open-set',font:{size:16}},margin:{l:90,r:20,t:50,b:70}},cfg);
  } else emptyPlot('voice-confusion','No se encontró la matriz open-set.');
  table('voice-metrics',[...DATA.voiceprint_metrics,...DATA.voiceprint_summary],Object.keys((DATA.voiceprint_metrics[0]||DATA.voiceprint_summary[0]||{})),200);
  table('voice-table',pred,['audio','decision','correct','true_enrolled','score','predicted_id','true_id'],120);
}
function renderAll(){
  const f=currentFilters();
  $('margin-min-value').textContent=f.marginMin.toFixed(4);
  $('margin-max-value').textContent=f.marginMax.toFixed(4);
  $('filter-summary').textContent=`Corpus: ${f.corpus} · Rol: ${f.role} · Margen: ${f.marginMin.toFixed(4)}–${f.marginMax.toFixed(4)} · Anchors: ${f.anchor}`;
  renderSummary();renderDiarization();renderTranscription();renderSentiment();renderKeywords();renderVoiceprint();
  setTimeout(()=>document.querySelectorAll('.tfm-panel.active .tfm-chart').forEach(el=>{try{Plotly.Plots.resize(el)}catch(e){}}),80);
}
function resetFilters(){
  $('filter-corpus').value='Todos'; $('filter-role').value='Todos';
  $('filter-margin-min').value=DATA.margin_range[0]; $('filter-margin-max').value=DATA.margin_range[1];
  $('filter-anchor').value='Todos'; renderAll();
}
function downloadHtml(){
  const blob=new Blob(['<!doctype html>\n'+document.documentElement.outerHTML],{type:'text/html;charset=utf-8'});
  const url=URL.createObjectURL(blob); const a=document.createElement('a'); a.href=url; a.download='dashboard_resultados_globales.html';
  document.body.appendChild(a); a.click(); a.remove(); setTimeout(()=>URL.revokeObjectURL(url),1000);
}
document.querySelectorAll('.tfm-tab').forEach(btn=>btn.addEventListener('click',()=>{
  document.querySelectorAll('.tfm-tab').forEach(x=>x.classList.remove('active'));
  document.querySelectorAll('.tfm-panel').forEach(x=>x.classList.remove('active'));
  btn.classList.add('active'); $(btn.dataset.target).classList.add('active');
  setTimeout(()=>document.querySelectorAll('#'+btn.dataset.target+' .tfm-chart').forEach(el=>{try{Plotly.Plots.resize(el)}catch(e){}}),80);
}));
$('apply-filters').addEventListener('click',renderAll); $('reset-filters').addEventListener('click',resetFilters); $('download-html').addEventListener('click',downloadHtml);
$('filter-margin-min').addEventListener('input',()=>{$('margin-min-value').textContent=Number($('filter-margin-min').value).toFixed(4)});
$('filter-margin-max').addEventListener('input',()=>{$('margin-max-value').textContent=Number($('filter-margin-max').value).toFixed(4)});
renderAll();
})();
"""


def _dashboard_body(payload: dict[str, Any]) -> str:
    minimum, maximum = payload["margin_range"]
    roles_options = "".join(
        f"<option value='{html_lib.escape(str(role))}'>{html_lib.escape(str(role))}</option>"
        for role in payload["roles"]
    )
    data_json = _json_dumps(payload).replace("</", "<\\/")

    return f"""
<div class='tfm-page'>
  <header class='tfm-header'>
    <div class='tfm-title'>Dashboard global de resultados del TFM</div>
    <div class='tfm-subtitle'>Resultados agregados de limpieza, diarización, transcripción, roles, sentimiento, keywords y huella de voz · Generado {html_lib.escape(payload['generated_at'])}</div>
  </header>
  <div class='tfm-toolbar'>
    <div class='tfm-control'><label>Corpus</label><select id='filter-corpus'><option>Todos</option><option>Bajas</option><option>Comerciales</option></select></div>
    <div class='tfm-control'><label>Rol</label><select id='filter-role'>{roles_options}</select></div>
    <div class='tfm-control'><label>Anchors</label><select id='filter-anchor'><option>Todos</option><option>Solo anchors</option><option>Excluir anchors</option></select></div>
    <div class='tfm-control'><label>Margen mínimo</label><input id='filter-margin-min' type='range' min='{minimum}' max='{maximum}' step='{max((maximum-minimum)/100,0.001)}' value='{minimum}'><div class='tfm-range-values'><span id='margin-min-value'>{minimum:.4f}</span></div></div>
    <div class='tfm-control'><label>Margen máximo</label><input id='filter-margin-max' type='range' min='{minimum}' max='{maximum}' step='{max((maximum-minimum)/100,0.001)}' value='{maximum}'><div class='tfm-range-values'><span id='margin-max-value'>{maximum:.4f}</span></div></div>
    <div class='tfm-actions'>
      <button id='apply-filters' class='tfm-btn primary'>Aplicar filtros</button>
      <button id='reset-filters' class='tfm-btn secondary'>Restablecer filtros</button>
      <span id='filter-summary' class='tfm-filter-summary'></span>
      <button id='download-html' class='tfm-btn export'>Descargar HTML</button>
    </div>
  </div>
  <div class='tfm-note'>Los filtros y las pestañas funcionan dentro del HTML y no dependen del kernel de Jupyter. El dashboard solo lee outputs restaurados desde GCS.</div>
  <nav class='tfm-tabs'>
    <button class='tfm-tab active' data-target='panel-summary'>Resumen ejecutivo</button>
    <button class='tfm-tab' data-target='panel-diarization'>Diarización</button>
    <button class='tfm-tab' data-target='panel-transcription'>Transcripción y roles</button>
    <button class='tfm-tab' data-target='panel-sentiment'>Sentimiento audio-texto</button>
    <button class='tfm-tab' data-target='panel-keywords'>Keywords</button>
    <button class='tfm-tab' data-target='panel-voiceprint'>Huella de voz</button>
  </nav>
  <section id='panel-summary' class='tfm-panel active'><div id='summary-kpis' class='tfm-kpis'></div><div class='tfm-grid'><div class='tfm-card'><div id='summary-funnel' class='tfm-chart'></div></div><div class='tfm-card'><div id='summary-availability' class='tfm-chart'></div></div><div class='tfm-card full'><div class='tfm-section-title'>Cobertura de outputs</div><div id='summary-table'></div></div></div></section>
  <section id='panel-diarization' class='tfm-panel'><div id='diarization-kpis' class='tfm-kpis'></div><div class='tfm-grid'><div class='tfm-card'><div id='margin-hist' class='tfm-chart'></div></div><div class='tfm-card'><div id='overlap-hist' class='tfm-chart'></div></div><div class='tfm-card'><div id='anchor-bar' class='tfm-chart'></div></div><div class='tfm-card'><div id='duration-hist' class='tfm-chart'></div></div><div class='tfm-card full'><div class='tfm-section-title'>Segmentos filtrados</div><div id='diarization-table'></div></div></div></section>
  <section id='panel-transcription' class='tfm-panel'><div id='transcription-kpis' class='tfm-kpis'></div><div class='tfm-grid'><div class='tfm-card'><div id='coverage-bar' class='tfm-chart'></div></div><div class='tfm-card'><div id='words-hist' class='tfm-chart'></div></div><div class='tfm-card'><div id='roles-bar' class='tfm-chart'></div></div><div class='tfm-card'><div id='mapping-bar' class='tfm-chart'></div></div><div class='tfm-card full'><div class='tfm-section-title'>Muestra filtrada</div><div id='transcription-table'></div></div></div></section>
  <section id='panel-sentiment' class='tfm-panel'><div id='sentiment-kpis' class='tfm-kpis'></div><div class='tfm-grid'><div class='tfm-card'><div id='text-sentiment-bar' class='tfm-chart'></div></div><div class='tfm-card'><div id='prosody-bar' class='tfm-chart'></div></div><div class='tfm-card full'><div id='fusion-heatmap' class='tfm-chart'></div></div><div class='tfm-card full'><div class='tfm-section-title'>Casos de desacuerdo o muestra fusionada</div><div id='sentiment-table'></div></div></div></section>
  <section id='panel-keywords' class='tfm-panel'><div id='keyword-kpis' class='tfm-kpis'></div><div class='tfm-grid'><div class='tfm-card'><div id='keywords-bar' class='tfm-chart'></div></div><div class='tfm-card'><div id='criticality-hist' class='tfm-chart'></div></div><div class='tfm-card'><div id='critical-coverage' class='tfm-chart'></div></div><div class='tfm-card full'><div class='tfm-section-title'>Llamadas o segmentos priorizados</div><div id='keyword-table'></div></div></div></section>
  <section id='panel-voiceprint' class='tfm-panel'><div id='voiceprint-kpis' class='tfm-kpis'></div><div class='tfm-grid'><div class='tfm-card'><div id='voice-score-hist' class='tfm-chart'></div></div><div class='tfm-card'><div id='voice-decision-bar' class='tfm-chart'></div></div><div class='tfm-card'><div id='voice-confusion' class='tfm-chart'></div></div><div class='tfm-card'><div class='tfm-section-title'>Métricas</div><div id='voice-metrics'></div></div><div class='tfm-card full'><div class='tfm-section-title'>Predicciones open-set</div><div id='voice-table'></div></div></div></section>
</div>
<script>window.__TFM_DASHBOARD_DATA__={data_json};</script>
<script>{DASHBOARD_JS}</script>
"""


def _build_full_html(payload: dict[str, Any]) -> str:
    plotly_js = get_plotlyjs()
    body = _dashboard_body(payload)
    return f"""<!doctype html>
<html lang='es'>
<head>
  <meta charset='utf-8'>
  <meta name='viewport' content='width=device-width,initial-scale=1'>
  <title>Dashboard global de resultados del TFM</title>
  <style>{DASHBOARD_CSS}</style>
  <script>{plotly_js}</script>
</head>
<body>{body}</body>
</html>"""


def _jupyter_iframe_fragment(path: Path) -> str:
    path = Path(path)
    try:
        relative = path.relative_to(PROJECT_DIR).as_posix()
    except Exception:
        relative = path.name
    relative_js = relative.replace("\\", "/").replace("'", "\\'")
    size_mb = path.stat().st_size / (1024 ** 2)
    uid = f"tfm-global-{datetime.now().strftime('%H%M%S%f')}"
    iframe_id = f"{uid}-frame"
    open_id = f"{uid}-open"
    cache = datetime.now().strftime("%Y%m%d%H%M%S%f")

    return f"""
<div id='{uid}' style='width:100%;'>
  <div style='display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;gap:10px;border:1px solid #CFE8DD;background:#F4FBF7;border-radius:11px;padding:10px 13px;margin:10px 0 12px;'>
    <div><div><b>Dashboard interactivo y HTML guardado · {size_mb:.1f} MB</b></div><div style='color:#456257;font-size:12px;overflow-wrap:anywhere;'><code>{html_lib.escape(str(path))}</code></div></div>
    <button id='{open_id}' style='border:0;background:#2A9D6F;color:white;border-radius:7px;padding:8px 13px;font-weight:650;cursor:pointer;'>Abrir HTML en otra pestaña</button>
  </div>
  <iframe id='{iframe_id}' style='width:100%;height:1020px;border:1px solid #DDE5EA;border-radius:12px;background:white;' loading='eager'></iframe>
</div>
<script>
(function(){{
  const rel='{relative_js}'; const pathname=window.location.pathname; let base=''; let notebookPath='';
  const markers=['/lab/tree/','/tree/','/notebooks/'];
  for(const marker of markers){{if(pathname.includes(marker)){{const parts=pathname.split(marker);base=parts[0];notebookPath=decodeURIComponent(parts.slice(1).join(marker));break;}}}}
  const notebookDir=notebookPath.includes('/')?notebookPath.split('/').slice(0,-1).join('/'):'';
  let target;
  if(notebookDir.endsWith('TFM_ProcesadoDeAudios')) target=notebookDir+'/'+rel;
  else if(notebookDir.includes('TFM_ProcesadoDeAudios/')) target=notebookDir.split('TFM_ProcesadoDeAudios/')[0]+'TFM_ProcesadoDeAudios/'+rel;
  else target=(notebookDir?notebookDir+'/':'')+rel;
  const encoded=target.split('/').filter(Boolean).map(encodeURIComponent).join('/');
  const url=base+'/files/'+encoded+'?v={cache}';
  const frame=document.getElementById('{iframe_id}'); const button=document.getElementById('{open_id}');
  if(frame) frame.src=url; if(button) button.addEventListener('click',()=>window.open(url,'_blank'));
}})();
</script>
"""


# ============================================================
# FUNCIÓN PÚBLICA
# ============================================================


def run_dashboard_resultados_globales(
    gcs_client: Any,
    force_restore: bool = False,
    save_html_snapshot: bool = False,
) -> HTML:
    """Restaura outputs desde GCS y genera un dashboard HTML autónomo.

    Las pestañas, los filtros y la descarga del HTML funcionan con JavaScript
    dentro del archivo generado. Después de la carga inicial no requieren que el
    kernel permanezca activo. El argumento ``save_html_snapshot`` se conserva por
    compatibilidad; el HTML se guarda siempre porque forma parte de la demo.

    Esta función no sube, elimina ni modifica objetos en Google Cloud Storage.
    """
    if gcs_client is None:
        raise ValueError("gcs_client no puede ser None.")

    global _GCS_CLIENT
    _GCS_CLIENT = gcs_client

    availability = _restore_dashboard_inputs(force=force_restore)
    datasets = _prepare_datasets(_load_datasets())
    payload = _build_dashboard_payload(datasets, availability)

    HTML_DIR.mkdir(parents=True, exist_ok=True)
    HTML_PATH.write_text(_build_full_html(payload), encoding="utf-8")

    available_count = int(availability["available"].sum())
    total_count = len(availability)
    loaded_rows = sum(len(df) for df in datasets.values())

    print("PROJECT_DIR:", PROJECT_DIR)
    print("GCS configurado:", True)
    print("Datasets disponibles:", f"{available_count}/{total_count}")
    print("Filas cargadas en total:", f"{loaded_rows:,}".replace(",", "."))
    print("Subidas a GCS:", "deshabilitadas")
    print("HTML local:", HTML_PATH)
    print("Interacción:", "pestañas y filtros independientes del kernel")

    return HTML(_jupyter_iframe_fragment(HTML_PATH))
