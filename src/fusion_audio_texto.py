"""Fase 07C: fusión y comparación entre afecto de audio y sentimiento textual.

El notebook mantiene visibles la cobertura, concordancia, discordancia,
agregaciones y visualizaciones. Este módulo concentra únicamente la lógica de
normalización, unión, cálculo y persistencia de los outputs originales.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from src.io_utils import write_csv_atomic


# ============================================================
# UTILIDADES Y NORMALIZACIÓN
# ============================================================


def read_csv_safe(path: Path | None) -> pd.DataFrame:
    """Lee un CSV no vacío; devuelve DataFrame vacío cuando no está disponible."""
    if path is None:
        return pd.DataFrame()
    candidate = Path(path)
    if not candidate.exists() or candidate.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(candidate)
    except Exception as error:
        print(f"No se pudo leer {candidate.name}: {error}")
        return pd.DataFrame()


def first_existing(paths: Iterable[Path]) -> Path | None:
    """Devuelve el primer archivo existente y no vacío."""
    for path in paths:
        candidate = Path(path)
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    return None


def detect_col(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    """Detecta la primera columna existente entre varias candidatas."""
    for column in candidates:
        if column in df.columns:
            return column
    return None


def normalize_audio_stem(value: object) -> str:
    """Normaliza extensiones y sufijos técnicos del pipeline."""
    if pd.isna(value):
        return ""
    stem = Path(str(value)).name
    stem = re.sub(
        r"\.(wav|mp3|m4a|flac|ogg)$", "", stem, flags=re.IGNORECASE
    )
    for suffix in [
        "_final_segments",
        "_final_merged",
        "_transcribed_segments",
        "_raw",
    ]:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem


def normalize_sentiment_label(value: object) -> str | float:
    """Normaliza la etiqueta textual al español, como el Notebook 07C original."""
    if pd.isna(value):
        return np.nan
    label = str(value).strip().lower()
    if label in {
        "pos",
        "positive",
        "positivo",
        "label_2",
        "5 stars",
        "4 stars",
    }:
        return "positivo"
    if label in {
        "neg",
        "negative",
        "negativo",
        "label_0",
        "1 star",
        "2 stars",
    }:
        return "negativo"
    if label in {"neu", "neutral", "label_1", "3 stars"}:
        return "neutral"
    return label


def sentiment_label_to_numeric(value: object) -> float:
    """Convierte etiquetas normalizadas a -1, 0 y 1."""
    label = normalize_sentiment_label(value)
    return {"negativo": -1.0, "neutral": 0.0, "positivo": 1.0}.get(
        label, np.nan
    )


def find_ser_negative_prob_col(df: pd.DataFrame) -> str | None:
    """Localiza una probabilidad SER negativa sin asumir un idioma único."""
    negative_terms = [
        "enfado",
        "enojo",
        "ira",
        "anger",
        "angry",
        "rabia",
        "tristeza",
        "sad",
        "disgust",
        "asco",
        "miedo",
        "fear",
        "frustrac",
        "negative",
        "negativo",
    ]
    probability_columns = [
        column for column in df.columns if str(column).startswith("ser_prob_")
    ]
    for column in probability_columns:
        if any(
            term in column.lower()
            for term in ["enfado", "anger", "angry", "ira", "rabia"]
        ):
            return column
    for column in probability_columns:
        if any(term in column.lower() for term in negative_terms):
            return column
    return None


def outputs_complete(paths: Iterable[Path]) -> bool:
    """Comprueba existencia y contenido de outputs requeridos."""
    return all(
        Path(path).exists() and Path(path).stat().st_size > 0
        for path in paths
    )


# ============================================================
# CONSTRUCCIÓN Y VALIDACIÓN DE LA FUSIÓN
# ============================================================


def _prepare_join_keys(
    df: pd.DataFrame,
    audio_candidates: Iterable[str],
    start_candidates: Iterable[str],
    end_candidates: Iterable[str],
) -> pd.DataFrame:
    out = df.copy()
    audio_col = detect_col(out, audio_candidates)
    start_col = detect_col(out, start_candidates)
    end_col = detect_col(out, end_candidates)
    if audio_col is None or start_col is None or end_col is None:
        raise ValueError(
            "No se pudieron construir llaves audio/start/end para la fusión."
        )
    out["audio_stem_norm"] = out[audio_col].apply(normalize_audio_stem)
    out["start_round"] = pd.to_numeric(out[start_col], errors="coerce").round(3)
    out["end_round"] = pd.to_numeric(out[end_col], errors="coerce").round(3)
    return out


def reconstruct_comparison(
    df_audio: pd.DataFrame,
    df_text: pd.DataFrame,
) -> pd.DataFrame:
    """Reconstruye audio↔texto por audio y timestamps cuando el CSV 07B es obsoleto."""
    if df_audio.empty or df_text.empty:
        return pd.DataFrame()

    audio = _prepare_join_keys(
        df_audio,
        ["audio_stem_norm", "audio_stem", "audio_file_norm", "audio_file"],
        ["start", "start_sec", "start_time"],
        ["end", "end_sec", "end_time"],
    )
    text = _prepare_join_keys(
        df_text,
        ["audio_stem_norm", "audio_stem", "audio_file", "filename"],
        ["start", "start_sec", "start_time"],
        ["end", "end_sec", "end_time"],
    )

    join_columns = ["audio_stem_norm", "start_round", "end_round"]
    text_columns = list(join_columns)
    for column in [
        "sentiment_uid",
        "sentiment_label_raw",
        "sentiment_label",
        "sentiment_label_norm",
        "sentiment_numeric",
        "sentiment_score",
        "sentiment_textual_label",
        "sentiment_textual_score",
        "text_whisper",
        "text",
        "transcription",
        "whisper_text",
        "role_proxy",
        "speaker_final",
        "interval_mmss",
    ]:
        if column in text.columns and column not in text_columns:
            text_columns.append(column)

    text_unique = text[text_columns].drop_duplicates(join_columns)
    return audio.merge(
        text_unique,
        on=join_columns,
        how="left",
        suffixes=("", "_text"),
    )


def normalize_fusion_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza columnas sin eliminar las originales y calcula disponibilidad."""
    if df.empty:
        return pd.DataFrame()
    out = df.copy()

    if "audio_stem_norm" not in out.columns:
        audio_col = detect_col(
            out, ["audio_stem", "audio_file_norm", "audio_file"]
        )
        out["audio_stem_norm"] = (
            out[audio_col].apply(normalize_audio_stem)
            if audio_col is not None
            else np.nan
        )

    for time_column in ["start", "end"]:
        if time_column not in out.columns:
            alternative = detect_col(
                out,
                [
                    f"{time_column}_round",
                    f"{time_column}_sec",
                    f"{time_column}_time",
                ],
            )
            if alternative is not None:
                out[time_column] = pd.to_numeric(
                    out[alternative], errors="coerce"
                )

    role_col = detect_col(
        out,
        [
            "role_proxy_for_prosody",
            "role_proxy",
            "official_role_proxy",
            "probable_role",
        ],
    )
    out["role"] = (
        out[role_col].astype("string").fillna("desconocido").str.lower()
        if role_col is not None
        else "desconocido"
    )
    speaker_col = detect_col(
        out, ["speaker_for_prosody", "speaker_final", "speaker"]
    )
    out["speaker"] = out[speaker_col] if speaker_col is not None else np.nan

    for column in [
        "arousal_proxy_score",
        "tension_proxy_score",
        "intensity_proxy_score",
        "calm_proxy_score",
    ]:
        if column not in out.columns:
            out[column] = np.nan
        out[column] = pd.to_numeric(out[column], errors="coerce")

    negative_probability_col = find_ser_negative_prob_col(out)
    out["ser_neg_prob"] = (
        pd.to_numeric(out[negative_probability_col], errors="coerce")
        if negative_probability_col is not None
        else np.nan
    )
    if "ser_pred_label" in out.columns:
        out["ser_pred_label_norm"] = out["ser_pred_label"].apply(
            normalize_sentiment_label
        )

    text_label_col = detect_col(
        out,
        [
            "sentiment_label",
            "sentiment_textual_label",
            "text_sentiment_label",
            "sentiment_label_norm",
        ],
    )
    out["sentiment_label_norm"] = (
        out[text_label_col].apply(normalize_sentiment_label)
        if text_label_col is not None
        else np.nan
    )

    if "sentiment_numeric" not in out.columns:
        out["sentiment_numeric"] = np.nan
    out["sentiment_numeric"] = pd.to_numeric(
        out["sentiment_numeric"], errors="coerce"
    )
    derived_sentiment_numeric = out["sentiment_label_norm"].apply(
        sentiment_label_to_numeric
    )
    out["sentiment_numeric"] = out["sentiment_numeric"].fillna(
        derived_sentiment_numeric
    )

    prosodic_state_col = detect_col(
        out,
        [
            "prosodic_state_proxy",
            "prosodic_state",
            "audio_affect_label",
            "affective_state",
        ],
    )
    if "prosodic_state_proxy" not in out.columns:
        out["prosodic_state_proxy"] = (
            out[prosodic_state_col]
            if prosodic_state_col is not None
            else np.nan
        )

    out["has_audio"] = (
        out["arousal_proxy_score"].notna()
        | out["tension_proxy_score"].notna()
        | out["prosodic_state_proxy"].notna()
    )
    out["has_text"] = (
        out["sentiment_numeric"].notna()
        | out["sentiment_label_norm"].notna()
    )
    return add_demo_compatibility_columns(out)


def add_demo_compatibility_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Añade aliases esperados por la demo sin borrar columnas científicas."""
    out = df.copy()

    if "sentiment_label" not in out.columns:
        out["sentiment_label"] = out.get("sentiment_label_norm", np.nan)
    else:
        missing = out["sentiment_label"].isna()
        if "sentiment_label_norm" in out.columns:
            out.loc[missing, "sentiment_label"] = out.loc[
                missing, "sentiment_label_norm"
            ]

    if "prosodic_state" not in out.columns:
        out["prosodic_state"] = out.get("prosodic_state_proxy", np.nan)
    else:
        missing = out["prosodic_state"].isna()
        if "prosodic_state_proxy" in out.columns:
            out.loc[missing, "prosodic_state"] = out.loc[
                missing, "prosodic_state_proxy"
            ]

    if "arousal_score" not in out.columns:
        out["arousal_score"] = out.get("arousal_proxy_score", np.nan)
    if "tension_score" not in out.columns:
        out["tension_score"] = out.get("tension_proxy_score", np.nan)
    return out


def comparison_is_usable(df: pd.DataFrame) -> bool:
    """Valida que un CSV restaurado contenga señales reales de ambas modalidades."""
    normalized = normalize_fusion_schema(df)
    if normalized.empty:
        return False
    comparable = normalized["has_audio"] & normalized["has_text"]
    return bool(comparable.any())


def build_fusion_dataframe(
    df_audio: pd.DataFrame,
    df_text: pd.DataFrame,
    df_existing_comparison: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, str]:
    """Usa la comparación 07B solo si es válida; de lo contrario la reconstruye."""
    existing = (
        df_existing_comparison.copy()
        if isinstance(df_existing_comparison, pd.DataFrame)
        else pd.DataFrame()
    )
    if comparison_is_usable(existing):
        normalized = normalize_fusion_schema(existing)
        return normalized, "comparison_07B_validada"

    rebuilt = reconstruct_comparison(df_audio, df_text)
    normalized = normalize_fusion_schema(rebuilt)
    source = (
        "reconstruido_07A+07B_comparison_obsoleta"
        if not existing.empty
        else "reconstruido_07A+07B"
    )
    return normalized, source


def fusion_output_is_compatible(path: Path) -> bool:
    """Determina si un output previo puede reutilizarse también en la demo."""
    df = read_csv_safe(path)
    if df.empty:
        return False
    normalized = normalize_fusion_schema(df)
    required = {
        "sentiment_label",
        "prosodic_state",
        "sentiment_label_norm",
        "prosodic_state_proxy",
        "has_audio",
        "has_text",
    }
    if not required.issubset(normalized.columns):
        return False
    return bool((normalized["has_audio"] & normalized["has_text"]).any())


# ============================================================
# ANÁLISIS DE COBERTURA, CONCORDANCIA Y DISCORDANCIA
# ============================================================


def compute_coverage(df_fusion: pd.DataFrame) -> dict[str, int | float]:
    """Calcula cobertura global de ambas modalidades."""
    if df_fusion.empty:
        return {
            "segments_total": 0,
            "both": 0,
            "only_audio": 0,
            "only_text": 0,
            "pct_comparable": 0.0,
        }
    both = int((df_fusion["has_audio"] & df_fusion["has_text"]).sum())
    only_audio = int(
        (df_fusion["has_audio"] & ~df_fusion["has_text"]).sum()
    )
    only_text = int(
        (~df_fusion["has_audio"] & df_fusion["has_text"]).sum()
    )
    return {
        "segments_total": len(df_fusion),
        "both": both,
        "only_audio": only_audio,
        "only_text": only_text,
        "pct_comparable": round(100 * both / max(1, len(df_fusion)), 1),
    }


def comparable_segments(df_fusion: pd.DataFrame) -> pd.DataFrame:
    """Selecciona segmentos con audio y texto disponibles."""
    if df_fusion.empty:
        return pd.DataFrame()
    return df_fusion[
        df_fusion["has_audio"] & df_fusion["has_text"]
    ].copy()


def compute_correlations(df_comparable: pd.DataFrame) -> pd.DataFrame:
    """Calcula Pearson y Spearman como el Notebook 07C original."""
    rows: list[dict[str, object]] = []
    if df_comparable.empty:
        return pd.DataFrame()

    from scipy.stats import pearsonr, spearmanr

    audio_scores = [
        column
        for column in [
            "arousal_proxy_score",
            "tension_proxy_score",
            "intensity_proxy_score",
            "ser_neg_prob",
        ]
        if column in df_comparable.columns
    ]
    for column in audio_scores:
        subset = df_comparable[["sentiment_numeric", column]].dropna()
        if len(subset) < 5:
            continue
        pearson_r, pearson_p = pearsonr(
            subset["sentiment_numeric"], subset[column]
        )
        spearman_r, spearman_p = spearmanr(
            subset["sentiment_numeric"], subset[column]
        )
        rows.append(
            {
                "score_audio": column,
                "n": len(subset),
                "pearson_r": round(pearson_r, 3),
                "pearson_p": round(pearson_p, 4),
                "spearman_r": round(spearman_r, 3),
                "spearman_p": round(spearman_p, 4),
            }
        )
    return pd.DataFrame(rows)


def compute_confusion_matrix(df_comparable: pd.DataFrame) -> pd.DataFrame:
    """Cruza estado prosódico con sentimiento textual normalizado."""
    if df_comparable.empty:
        return pd.DataFrame()
    required = {"prosodic_state_proxy", "sentiment_label_norm"}
    if not required.issubset(df_comparable.columns):
        return pd.DataFrame()

    valid = df_comparable[
        df_comparable["prosodic_state_proxy"].notna()
        & df_comparable["sentiment_label_norm"].notna()
    ].copy()
    if valid.empty:
        return pd.DataFrame()
    return pd.crosstab(
        valid["prosodic_state_proxy"], valid["sentiment_label_norm"]
    )


def compute_disagreement(
    df_comparable: pd.DataFrame,
    audio_high_percentile: float = 75,
    text_nonnegative_threshold: float = 0.0,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Calcula frustración enmascarada y negatividad contenida."""
    if df_comparable.empty:
        return pd.DataFrame(), {}

    out = df_comparable.copy()
    thresholds: dict[str, float] = {}
    for column in [
        "tension_proxy_score",
        "arousal_proxy_score",
        "ser_neg_prob",
    ]:
        if column in out.columns and out[column].notna().any():
            thresholds[column] = float(
                np.nanpercentile(out[column], audio_high_percentile)
            )

    audio_negative = pd.Series(False, index=out.index)
    for column, threshold in thresholds.items():
        audio_negative |= out[column] >= threshold

    text_nonnegative = (
        out["sentiment_numeric"] >= text_nonnegative_threshold
    )
    text_negative = out["sentiment_numeric"] < text_nonnegative_threshold
    out["masked_frustration"] = text_nonnegative & audio_negative
    out["contained_negativity"] = text_negative & (~audio_negative)

    # Aliases consumidos por la demo actual.
    out["is_disagreement"] = (
        out["masked_frustration"] | out["contained_negativity"]
    )
    out["disagreement_flag"] = out["is_disagreement"]
    return add_demo_compatibility_columns(out), thresholds


def build_fusion_aggregates(
    df_disagreement: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Agrega scores y discordancias por rol y por llamada+rol."""
    if df_disagreement.empty:
        return pd.DataFrame(), pd.DataFrame()

    aggregation: dict[str, str] = {}
    for column in [
        "sentiment_numeric",
        "arousal_proxy_score",
        "tension_proxy_score",
        "intensity_proxy_score",
        "ser_neg_prob",
        "masked_frustration",
        "contained_negativity",
    ]:
        if column in df_disagreement.columns:
            aggregation[column] = "mean"
    if not aggregation:
        return pd.DataFrame(), pd.DataFrame()

    role_level = df_disagreement.groupby("role").agg(aggregation)
    role_level["n_segmentos"] = df_disagreement.groupby("role").size()
    role_level = role_level.reset_index()

    call_level = pd.DataFrame()
    if "audio_stem_norm" in df_disagreement.columns:
        call_level = df_disagreement.groupby(
            ["audio_stem_norm", "role"]
        ).agg(aggregation)
        call_level["n_segmentos"] = df_disagreement.groupby(
            ["audio_stem_norm", "role"]
        ).size()
        call_level = call_level.reset_index()
    return role_level, call_level


def build_fusion_summary(
    df_fusion: pd.DataFrame,
    correlations: pd.DataFrame,
    df_disagreement: pd.DataFrame,
) -> pd.DataFrame:
    """Construye el resumen compacto con métricas originales."""
    rows: list[dict[str, object]] = []
    if not df_fusion.empty:
        both = int((df_fusion["has_audio"] & df_fusion["has_text"]).sum())
        rows.extend(
            [
                {"metrica": "segmentos_totales", "valor": len(df_fusion)},
                {"metrica": "segmentos_comparables", "valor": both},
            ]
        )
    if not correlations.empty:
        for _, row in correlations.iterrows():
            rows.append(
                {
                    "metrica": f"pearson_{row['score_audio']}",
                    "valor": row["pearson_r"],
                }
            )
    if (
        not df_disagreement.empty
        and "masked_frustration" in df_disagreement.columns
    ):
        rows.append(
            {
                "metrica": "pct_frustracion_enmascarada",
                "valor": round(
                    100 * df_disagreement["masked_frustration"].mean(), 2
                ),
            }
        )
    return pd.DataFrame(rows)


# ============================================================
# GUARDADO DE OUTPUTS ORIGINALES
# ============================================================


def save_fusion_outputs(
    df_segments: pd.DataFrame,
    correlations: pd.DataFrame,
    confusion: pd.DataFrame,
    role_level: pd.DataFrame,
    call_level: pd.DataFrame,
    summary: pd.DataFrame,
    paths: Mapping[str, Path],
) -> list[Path]:
    """Guarda los siete outputs existentes de 07C sin crear archivos nuevos."""
    saved: list[Path] = []

    segment_path = Path(paths["segments"])
    write_csv_atomic(add_demo_compatibility_columns(df_segments), segment_path)
    saved.append(segment_path)

    if not correlations.empty:
        correlation_path = Path(paths["correlations"])
        write_csv_atomic(correlations, correlation_path)
        saved.append(correlation_path)

    if not confusion.empty:
        confusion_path = Path(paths["confusion"])
        confusion_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = confusion_path.with_suffix(confusion_path.suffix + ".tmp")
        confusion.to_csv(temporary)
        temporary.replace(confusion_path)
        saved.append(confusion_path)

    if "masked_frustration" in df_segments.columns:
        disagreement = df_segments[df_segments["masked_frustration"]].copy()
        if not disagreement.empty:
            disagreement_path = Path(paths["disagreement"])
            write_csv_atomic(disagreement, disagreement_path)
            saved.append(disagreement_path)

    if not role_level.empty:
        role_path = Path(paths["role"])
        write_csv_atomic(role_level, role_path)
        saved.append(role_path)

    if not call_level.empty:
        call_path = Path(paths["call"])
        write_csv_atomic(call_level, call_path)
        saved.append(call_path)

    summary_path = Path(paths["summary"])
    write_csv_atomic(summary, summary_path)
    saved.append(summary_path)
    return saved


def load_fusion_outputs(paths: Mapping[str, Path]) -> dict[str, pd.DataFrame]:
    """Carga outputs disponibles; la matriz conserva su índice como primera columna."""
    loaded: dict[str, pd.DataFrame] = {}
    for name, path in paths.items():
        candidate = Path(path)
        if candidate.exists() and candidate.stat().st_size > 0:
            loaded[name] = pd.read_csv(candidate)
    return loaded
