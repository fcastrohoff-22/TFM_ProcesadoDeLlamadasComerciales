"""Fase 07A: análisis de sentimiento textual por segmento.

El notebook orquesta la restauración, la decisión de reutilizar resultados,
la carga del modelo y las visualizaciones. Este módulo conserva la lógica
reutilizable de preparación, inferencia con checkpoint, agregación y guardado.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from src.io_utils import write_csv_atomic


# ============================================================
# UTILIDADES DE COLUMNAS Y NORMALIZACIÓN
# ============================================================


def first_existing_path(paths: Iterable[Path]) -> Path | None:
    """Devuelve el primer archivo existente y no vacío."""
    for path in paths:
        candidate = Path(path)
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    return None


def detect_text_col(df: pd.DataFrame) -> str:
    """Detecta la columna de texto generada por Whisper."""
    candidates = [
        "text",
        "text_clean",
        "whisper_text",
        "text_whisper",
        "transcription_text",
        "transcription",
        "transcript",
        "transcribed_text",
        "segment_text",
    ]
    for column in candidates:
        if column in df.columns:
            return column
    raise ValueError(
        "No se encontró columna de texto Whisper. "
        f"Columnas disponibles: {list(df.columns)}"
    )


def normalize_audio_name(value: object) -> str:
    """Normaliza un nombre de audio conservando el identificador original."""
    if pd.isna(value):
        return ""
    name = Path(str(value)).name
    return name[:-4] if name.lower().endswith(".wav") else name


def add_join_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Crea las mismas cinco llaves de unión utilizadas por el Notebook 07 original."""
    out = df.copy()

    if "audio_stem" in out.columns:
        out["_join_audio"] = out["audio_stem"].map(normalize_audio_name)
    elif "audio_file" in out.columns:
        out["_join_audio"] = out["audio_file"].map(normalize_audio_name)
    else:
        raise ValueError("No hay audio_stem ni audio_file para construir la unión.")

    if "segment_id_raw" in out.columns:
        out["_join_segment_id"] = out["segment_id_raw"].astype("string").fillna("")
    else:
        out["_join_segment_id"] = ""

    for column in ["start", "end"]:
        if column in out.columns:
            out[f"_join_{column}"] = pd.to_numeric(
                out[column], errors="coerce"
            ).round(3)
        else:
            out[f"_join_{column}"] = np.nan

    if "speaker_final" in out.columns:
        out["_join_speaker_final"] = (
            out["speaker_final"].astype("string").fillna("")
        )
    elif "speaker" in out.columns:
        out["_join_speaker_final"] = out["speaker"].astype("string").fillna("")
    else:
        out["_join_speaker_final"] = ""

    return out


def choose_role_col(df: pd.DataFrame) -> str | None:
    """Detecta la etiqueta de rol proxy generada por la fase 06."""
    for column in [
        "official_role_proxy",
        "role_proxy",
        "assigned_role",
        "probable_role",
        "role",
    ]:
        if column in df.columns:
            return column
    return None


def normalize_role(value: object) -> str:
    """Normaliza el rol a AGENT, CLIENT o UNKNOWN."""
    if pd.isna(value):
        return "UNKNOWN"
    normalized = str(value).strip().upper()
    return normalized if normalized in {"AGENT", "CLIENT"} else "UNKNOWN"


def normalize_sentiment_label(label: object) -> str | float:
    """Normaliza las etiquetas del modelo a negative / neutral / positive."""
    if pd.isna(label):
        return np.nan
    value = str(label).strip().lower()
    mapping = {
        "neg": "negative",
        "negative": "negative",
        "negativo": "negative",
        "label_0": "negative",
        "neu": "neutral",
        "neutral": "neutral",
        "label_1": "neutral",
        "pos": "positive",
        "positive": "positive",
        "positivo": "positive",
        "label_2": "positive",
    }
    return mapping.get(value, value)


def sentiment_to_numeric(label: object) -> float:
    """Convierte negative=-1, neutral=0 y positive=1."""
    normalized = normalize_sentiment_label(label)
    return {
        "negative": -1.0,
        "neutral": 0.0,
        "positive": 1.0,
    }.get(normalized, np.nan)


def seconds_to_mmss(value: object) -> str:
    """Convierte segundos a MM:SS para auditoría visual."""
    if pd.isna(value):
        return ""
    seconds_total = float(value)
    minutes = int(seconds_total // 60)
    seconds = int(round(seconds_total % 60))
    return f"{minutes:02d}:{seconds:02d}"


def add_time_interval(df: pd.DataFrame) -> pd.DataFrame:
    """Añade el intervalo legible de cada segmento."""
    out = df.copy()
    if {"start", "end"}.issubset(out.columns):
        out["interval_mmss"] = out.apply(
            lambda row: (
                f"{seconds_to_mmss(row['start'])} - "
                f"{seconds_to_mmss(row['end'])}"
            ),
            axis=1,
        )
    return out


def outputs_complete(paths: Iterable[Path]) -> bool:
    """Comprueba que todos los outputs requeridos existan y no estén vacíos."""
    return all(
        Path(path).exists() and Path(path).stat().st_size > 0
        for path in paths
    )


# ============================================================
# PREPARACIÓN DE LA BASE TEXTUAL
# ============================================================


def load_text_base(
    transcription_candidates: Iterable[Path],
    proxy_segment_csv: Path,
    min_words_for_sentiment: int,
) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    """Carga Whisper y roles proxy, y prepara la base y el subconjunto del modelo."""
    transcription_path = first_existing_path(transcription_candidates)
    if transcription_path is None:
        raise FileNotFoundError(
            "No se encontró ningún CSV de transcripción Whisper entre: "
            f"{[str(path) for path in transcription_candidates]}"
        )

    df_transcribed = pd.read_csv(transcription_path)
    text_col = detect_text_col(df_transcribed)
    df_transcribed = df_transcribed.copy()
    df_transcribed["text_whisper"] = (
        df_transcribed[text_col].fillna("").astype(str).str.strip()
    )

    proxy_path = Path(proxy_segment_csv)
    if proxy_path.exists() and proxy_path.stat().st_size > 0:
        df_proxy = pd.read_csv(proxy_path)
        role_col = choose_role_col(df_proxy)
        df_proxy = df_proxy.copy()
        df_proxy["role_proxy"] = (
            df_proxy[role_col].map(normalize_role)
            if role_col is not None
            else "UNKNOWN"
        )

        proxy_keep_cols = [
            column
            for column in [
                "audio_file",
                "audio_stem",
                "segment_id_raw",
                "start",
                "end",
                "speaker_final",
                "role_proxy",
                "probable_role",
                "official_role_proxy",
                "role_confidence",
                "proxy_confidence",
                "proxy_method",
                "role_mapping_status",
                "n_matches_total",
            ]
            if column in df_proxy.columns
        ]

        df_t_join = add_join_keys(df_transcribed)
        df_p_join = add_join_keys(df_proxy[proxy_keep_cols])
        join_cols = [
            "_join_audio",
            "_join_segment_id",
            "_join_start",
            "_join_end",
            "_join_speaker_final",
        ]
        df_p_join = df_p_join.drop_duplicates(subset=join_cols)

        proxy_columns_to_add = [
            column
            for column in proxy_keep_cols
            if (
                column not in df_transcribed.columns
                or column
                in {
                    "role_proxy",
                    "probable_role",
                    "official_role_proxy",
                    "role_confidence",
                    "proxy_confidence",
                    "proxy_method",
                    "role_mapping_status",
                    "n_matches_total",
                }
            )
        ]

        df_base = df_t_join.merge(
            df_p_join[[*join_cols, *proxy_columns_to_add]],
            on=join_cols,
            how="left",
            suffixes=("", "_proxy"),
        )
        if "role_proxy" not in df_base.columns:
            df_base["role_proxy"] = "UNKNOWN"
        df_base["role_proxy"] = df_base["role_proxy"].map(normalize_role)
    else:
        df_base = df_transcribed.copy()
        df_base["role_proxy"] = "UNKNOWN"

    df_base["text_whisper"] = (
        df_base["text_whisper"].fillna("").astype(str).str.strip()
    )
    df_base["n_chars"] = df_base["text_whisper"].str.len()
    df_base["n_words"] = df_base["text_whisper"].apply(
        lambda text: len(text.split()) if text else 0
    )

    if "audio_stem" not in df_base.columns and "audio_file" in df_base.columns:
        df_base["audio_stem"] = df_base["audio_file"].map(normalize_audio_name)
    if "audio_file" not in df_base.columns and "audio_stem" in df_base.columns:
        df_base["audio_file"] = df_base["audio_stem"].astype(str) + ".wav"
    if "speaker_final" not in df_base.columns and "speaker" in df_base.columns:
        df_base["speaker_final"] = df_base["speaker"]
    if "speaker_final" not in df_base.columns:
        df_base["speaker_final"] = ""

    df_base = add_time_interval(df_base)
    segment_component = (
        df_base["segment_id_raw"].astype("string").fillna("")
        if "segment_id_raw" in df_base.columns
        else pd.Series(range(len(df_base)), index=df_base.index).astype(str)
    )
    start_component = pd.to_numeric(
        df_base["start"] if "start" in df_base.columns else np.nan,
        errors="coerce",
    ).round(3).astype(str)
    end_component = pd.to_numeric(
        df_base["end"] if "end" in df_base.columns else np.nan,
        errors="coerce",
    ).round(3).astype(str)

    # Se conserva el formato exacto del UID original para reutilizar checkpoints.
    df_base["sentiment_uid"] = (
        df_base["audio_stem"].astype(str)
        + "__"
        + segment_component.astype(str)
        + "__"
        + start_component
        + "__"
        + end_component
        + "__"
        + df_base["speaker_final"].astype(str)
    )

    df_sent_input = df_base[
        df_base["text_whisper"].notna()
        & df_base["text_whisper"].astype(str).str.strip().ne("")
        & df_base["n_words"].ge(min_words_for_sentiment)
    ].copy().reset_index(drop=True)

    return df_base, df_sent_input, transcription_path


def build_quality_summary(df_base: pd.DataFrame) -> pd.DataFrame:
    """Genera el control de calidad previo al modelo."""
    return pd.DataFrame(
        [
            {"metric": "audios_total", "value": df_base["audio_file"].nunique()},
            {"metric": "segments_total", "value": len(df_base)},
            {
                "metric": "segments_with_text",
                "value": int((df_base["n_chars"] > 0).sum()),
            },
            {
                "metric": "segments_without_text",
                "value": int((df_base["n_chars"] == 0).sum()),
            },
            {
                "metric": "segments_with_role_proxy",
                "value": int(
                    df_base["role_proxy"].isin(["AGENT", "CLIENT"]).sum()
                ),
            },
            {
                "metric": "avg_words_per_segment",
                "value": round(df_base["n_words"].mean(), 2),
            },
            {
                "metric": "median_words_per_segment",
                "value": round(df_base["n_words"].median(), 2),
            },
        ]
    )


# ============================================================
# MODELO Y CHECKPOINT
# ============================================================


def load_sentiment_pipeline(
    model_name: str,
    batch_size_gpu: int = 32,
    batch_size_cpu: int = 8,
):
    """Carga el modelo únicamente cuando el notebook confirma que hace falta."""
    import torch
    from transformers import pipeline

    device = 0 if torch.cuda.is_available() else -1
    batch_size = batch_size_gpu if device == 0 else batch_size_cpu
    sentiment_pipeline = pipeline(
        task="sentiment-analysis",
        model=model_name,
        tokenizer=model_name,
        device=device,
    )
    return sentiment_pipeline, batch_size, ("cuda" if device == 0 else "cpu")


def apply_sentiment_with_checkpoint(
    df_sent_input: pd.DataFrame,
    sentiment_pipeline,
    batch_size: int,
    checkpoint_csv: Path,
    max_length: int = 128,
) -> pd.DataFrame:
    """Aplica sentimiento por lotes y conserva un checkpoint reanudable."""
    checkpoint_path = Path(checkpoint_csv)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    if checkpoint_path.exists() and checkpoint_path.stat().st_size > 0:
        df_checkpoint = pd.read_csv(checkpoint_path)
        if "sentiment_uid" not in df_checkpoint.columns:
            print("Checkpoint previo incompatible; se ignora.")
            df_checkpoint = pd.DataFrame()
    else:
        df_checkpoint = pd.DataFrame()

    processed_uids = (
        set(df_checkpoint["sentiment_uid"].astype(str))
        if not df_checkpoint.empty
        else set()
    )
    pending = df_sent_input[
        ~df_sent_input["sentiment_uid"].astype(str).isin(processed_uids)
    ].copy()

    print("Segmentos ya procesados:", len(processed_uids))
    print("Segmentos pendientes:", len(pending))

    new_rows: list[dict[str, object]] = []
    start_time = time.time()
    texts = pending["text_whisper"].fillna("").astype(str).tolist()
    uids = pending["sentiment_uid"].astype(str).tolist()

    for offset in range(0, len(texts), batch_size):
        batch_texts = texts[offset : offset + batch_size]
        batch_uids = uids[offset : offset + batch_size]

        try:
            predictions = sentiment_pipeline(
                batch_texts,
                truncation=True,
                max_length=max_length,
            )
        except Exception as error:  # conserva el comportamiento tolerante original
            print(f"Error en batch {offset}: {error}")
            predictions = [
                {"label": np.nan, "score": np.nan}
                for _ in batch_texts
            ]

        for uid, prediction in zip(batch_uids, predictions):
            raw_label = prediction.get("label", np.nan)
            label = normalize_sentiment_label(raw_label)
            new_rows.append(
                {
                    "sentiment_uid": uid,
                    "sentiment_label_raw": raw_label,
                    "sentiment_label": label,
                    "sentiment_score": prediction.get("score", np.nan),
                    "sentiment_numeric": sentiment_to_numeric(label),
                }
            )

        if new_rows:
            df_new = pd.DataFrame(new_rows)
            df_out = (
                pd.concat([df_checkpoint, df_new], ignore_index=True)
                if not df_checkpoint.empty
                else df_new.copy()
            )
            df_out = df_out.drop_duplicates(
                subset=["sentiment_uid"], keep="last"
            )
            write_csv_atomic(df_out, checkpoint_path)

        done = min(offset + batch_size, len(texts))
        total_done = len(processed_uids) + done
        elapsed_minutes = (time.time() - start_time) / 60
        print(
            f"Procesados {total_done}/{len(df_sent_input)} segmentos | "
            f"elapsed={elapsed_minutes:.1f} min",
            flush=True,
        )

    if not checkpoint_path.exists():
        # Caso extremo: input vacío. Se crea un checkpoint con esquema estable.
        empty = pd.DataFrame(
            columns=[
                "sentiment_uid",
                "sentiment_label_raw",
                "sentiment_label",
                "sentiment_score",
                "sentiment_numeric",
            ]
        )
        write_csv_atomic(empty, checkpoint_path)

    return pd.read_csv(checkpoint_path)


def merge_sentiment_results(
    df_base: pd.DataFrame,
    df_sent_input: pd.DataFrame,
    df_checkpoint: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Integra el checkpoint en la base analizada y en la base completa."""
    if "sentiment_uid" not in df_checkpoint.columns:
        raise ValueError("El checkpoint no contiene sentiment_uid.")
    checkpoint = df_checkpoint.drop_duplicates(
        subset=["sentiment_uid"], keep="last"
    )
    df_sent = df_sent_input.merge(checkpoint, on="sentiment_uid", how="left")
    df_all_enriched = df_base.merge(
        checkpoint, on="sentiment_uid", how="left"
    )
    return df_sent, df_all_enriched


# ============================================================
# AGREGACIONES ORIGINALES
# ============================================================


def classify_avg_sentiment(value: object) -> str:
    """Clasifica el promedio con los umbrales originales ±0.25."""
    if pd.isna(value):
        return "unknown"
    numeric = float(value)
    if numeric <= -0.25:
        return "negative"
    if numeric >= 0.25:
        return "positive"
    return "neutral"


def build_sentiment_distribution(df_sent: pd.DataFrame) -> pd.DataFrame:
    """Distribución general de etiquetas por segmento."""
    distribution = (
        df_sent["sentiment_label"]
        .value_counts(dropna=False)
        .rename_axis("sentiment_label")
        .reset_index(name="n_segments")
    )
    denominator = max(1, len(df_sent))
    distribution["percentage"] = (
        distribution["n_segments"] / denominator * 100
    ).round(2)
    return distribution


def _dominant_role(series: pd.Series) -> str:
    counts = series.value_counts()
    return str(counts.index[0]) if len(counts) else "UNKNOWN"


def assign_call_phase(position: object) -> str:
    """Asigna inicio, mitad o final según la posición relativa original."""
    if pd.isna(position):
        return "unknown"
    numeric = float(position)
    if numeric < 1 / 3:
        return "inicio"
    if numeric < 2 / 3:
        return "mitad"
    return "final"


def build_sentiment_aggregates(df_sent: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Construye todos los agregados originales de la fase 07A."""
    duration_agg = (
        ("duration", "sum")
        if "duration" in df_sent.columns
        else ("n_words", "count")
    )

    call_sentiment = (
        df_sent.groupby(["audio_file", "audio_stem"], dropna=False)
        .agg(
            n_segments=("text_whisper", "count"),
            total_words=("n_words", "sum"),
            total_duration=duration_agg,
            avg_sentiment=("sentiment_numeric", "mean"),
            avg_confidence=("sentiment_score", "mean"),
            pct_negative=(
                "sentiment_label",
                lambda values: (values == "negative").mean(),
            ),
            pct_neutral=(
                "sentiment_label",
                lambda values: (values == "neutral").mean(),
            ),
            pct_positive=(
                "sentiment_label",
                lambda values: (values == "positive").mean(),
            ),
            n_client_segments=(
                "role_proxy",
                lambda values: (values == "CLIENT").sum(),
            ),
            n_agent_segments=(
                "role_proxy",
                lambda values: (values == "AGENT").sum(),
            ),
        )
        .reset_index()
    )
    call_sentiment["call_sentiment_label"] = call_sentiment[
        "avg_sentiment"
    ].apply(classify_avg_sentiment)

    df_role = df_sent[df_sent["role_proxy"].isin(["AGENT", "CLIENT"])].copy()
    role_duration_agg = (
        ("duration", "sum")
        if "duration" in df_role.columns
        else ("n_words", "count")
    )

    call_role_sentiment = (
        df_role.groupby(
            ["audio_file", "audio_stem", "role_proxy"], dropna=False
        )
        .agg(
            n_segments=("text_whisper", "count"),
            total_words=("n_words", "sum"),
            total_duration=role_duration_agg,
            avg_sentiment=("sentiment_numeric", "mean"),
            avg_confidence=("sentiment_score", "mean"),
            pct_negative=(
                "sentiment_label",
                lambda values: (values == "negative").mean(),
            ),
            pct_neutral=(
                "sentiment_label",
                lambda values: (values == "neutral").mean(),
            ),
            pct_positive=(
                "sentiment_label",
                lambda values: (values == "positive").mean(),
            ),
        )
        .reset_index()
    )
    call_role_sentiment["role_sentiment_label"] = call_role_sentiment[
        "avg_sentiment"
    ].apply(classify_avg_sentiment)

    role_global_sentiment = (
        df_role.groupby("role_proxy", dropna=False)
        .agg(
            n_segments=("text_whisper", "count"),
            n_audios=("audio_file", "nunique"),
            total_words=("n_words", "sum"),
            avg_sentiment=("sentiment_numeric", "mean"),
            avg_confidence=("sentiment_score", "mean"),
            pct_negative=(
                "sentiment_label",
                lambda values: (values == "negative").mean(),
            ),
            pct_neutral=(
                "sentiment_label",
                lambda values: (values == "neutral").mean(),
            ),
            pct_positive=(
                "sentiment_label",
                lambda values: (values == "positive").mean(),
            ),
        )
        .reset_index()
    )

    speaker_sentiment = (
        df_sent.groupby(
            ["audio_file", "audio_stem", "speaker_final"], dropna=False
        )
        .agg(
            n_segments=("text_whisper", "count"),
            total_words=("n_words", "sum"),
            avg_sentiment=("sentiment_numeric", "mean"),
            avg_confidence=("sentiment_score", "mean"),
            pct_negative=(
                "sentiment_label",
                lambda values: (values == "negative").mean(),
            ),
            pct_neutral=(
                "sentiment_label",
                lambda values: (values == "neutral").mean(),
            ),
            pct_positive=(
                "sentiment_label",
                lambda values: (values == "positive").mean(),
            ),
            dominant_role=("role_proxy", _dominant_role),
        )
        .reset_index()
    )

    df_temporal = df_sent.copy()
    if {"start", "end"}.issubset(df_temporal.columns):
        call_max_end = (
            df_temporal.groupby("audio_file")["end"]
            .max()
            .rename("call_duration")
            .reset_index()
        )
        df_temporal = df_temporal.merge(
            call_max_end, on="audio_file", how="left"
        )
        df_temporal["relative_position"] = (
            pd.to_numeric(df_temporal["start"], errors="coerce")
            / df_temporal["call_duration"]
        )
        df_temporal["call_phase"] = df_temporal[
            "relative_position"
        ].apply(assign_call_phase)
    else:
        df_temporal["call_phase"] = "unknown"

    temporal_sentiment = (
        df_temporal.groupby(["call_phase", "role_proxy"], dropna=False)
        .agg(
            n_segments=("text_whisper", "count"),
            n_audios=("audio_file", "nunique"),
            avg_sentiment=("sentiment_numeric", "mean"),
            pct_negative=(
                "sentiment_label",
                lambda values: (values == "negative").mean(),
            ),
            pct_positive=(
                "sentiment_label",
                lambda values: (values == "positive").mean(),
            ),
        )
        .reset_index()
    )
    phase_order = pd.CategoricalDtype(
        categories=["inicio", "mitad", "final", "unknown"],
        ordered=True,
    )
    temporal_sentiment["call_phase"] = temporal_sentiment[
        "call_phase"
    ].astype(phase_order)
    temporal_sentiment = temporal_sentiment.sort_values(
        ["call_phase", "role_proxy"]
    ).reset_index(drop=True)

    global_temporal = (
        df_temporal.groupby("call_phase", dropna=False)
        .agg(
            avg_sentiment=("sentiment_numeric", "mean"),
            n_segments=("text_whisper", "count"),
        )
        .reset_index()
    )
    global_temporal["call_phase"] = global_temporal["call_phase"].astype(
        phase_order
    )
    global_temporal = global_temporal.sort_values("call_phase")

    client_negative_calls = (
        call_role_sentiment[
            call_role_sentiment["role_proxy"] == "CLIENT"
        ]
        .sort_values(
            ["avg_sentiment", "pct_negative"], ascending=[True, False]
        )
        .head(20)
    )

    return {
        "call_sentiment": call_sentiment,
        "call_role_sentiment": call_role_sentiment,
        "role_global_sentiment": role_global_sentiment,
        "speaker_sentiment": speaker_sentiment,
        "temporal_sentiment": temporal_sentiment,
        "global_temporal": global_temporal,
        "client_negative_calls": client_negative_calls,
    }


def build_summary_for_memory(
    df_base: pd.DataFrame,
    df_sent: pd.DataFrame,
    call_sentiment: pd.DataFrame,
    role_global_sentiment: pd.DataFrame,
) -> pd.DataFrame:
    """Construye el resumen final con los nombres de métricas originales."""
    rows: list[dict[str, object]] = [
        {"metric": "audios_base", "value": df_base["audio_file"].nunique()},
        {"metric": "segments_base", "value": len(df_base)},
        {
            "metric": "segments_with_text",
            "value": int((df_base["n_chars"] > 0).sum()),
        },
        {"metric": "segments_analyzed_sentiment", "value": len(df_sent)},
        {
            "metric": "audios_analyzed_sentiment",
            "value": df_sent["audio_file"].nunique(),
        },
        {
            "metric": "segments_with_role_proxy",
            "value": int(
                df_sent["role_proxy"].isin(["AGENT", "CLIENT"]).sum()
            ),
        },
        {
            "metric": "pct_negative_segments",
            "value": round(
                (df_sent["sentiment_label"] == "negative").mean() * 100,
                2,
            ),
        },
        {
            "metric": "pct_neutral_segments",
            "value": round(
                (df_sent["sentiment_label"] == "neutral").mean() * 100,
                2,
            ),
        },
        {
            "metric": "pct_positive_segments",
            "value": round(
                (df_sent["sentiment_label"] == "positive").mean() * 100,
                2,
            ),
        },
        {
            "metric": "avg_sentiment_global",
            "value": round(df_sent["sentiment_numeric"].mean(), 4),
        },
        {
            "metric": "negative_calls",
            "value": int(
                (call_sentiment["call_sentiment_label"] == "negative").sum()
            ),
        },
        {
            "metric": "neutral_calls",
            "value": int(
                (call_sentiment["call_sentiment_label"] == "neutral").sum()
            ),
        },
        {
            "metric": "positive_calls",
            "value": int(
                (call_sentiment["call_sentiment_label"] == "positive").sum()
            ),
        },
    ]

    for _, role_row in role_global_sentiment.iterrows():
        role = role_row["role_proxy"]
        rows.extend(
            [
                {
                    "metric": f"{role}_avg_sentiment",
                    "value": round(role_row["avg_sentiment"], 4),
                },
                {
                    "metric": f"{role}_pct_negative",
                    "value": round(role_row["pct_negative"] * 100, 2),
                },
                {
                    "metric": f"{role}_n_segments",
                    "value": int(role_row["n_segments"]),
                },
            ]
        )

    return pd.DataFrame(rows)


# ============================================================
# CARGA Y GUARDADO DE OUTPUTS
# ============================================================


def load_sentiment_outputs(paths: Mapping[str, Path]) -> dict[str, pd.DataFrame]:
    """Carga los outputs existentes con nombres lógicos estables."""
    loaded: dict[str, pd.DataFrame] = {}
    for name, path in paths.items():
        candidate = Path(path)
        if candidate.exists() and candidate.stat().st_size > 0:
            loaded[name] = pd.read_csv(candidate)
    return loaded


def save_sentiment_outputs(
    df_sent: pd.DataFrame,
    df_all_enriched: pd.DataFrame,
    aggregates: Mapping[str, pd.DataFrame],
    summary_for_memory: pd.DataFrame,
    paths: Mapping[str, Path],
) -> list[Path]:
    """Guarda únicamente los CSV originales de la fase 07A."""
    dataframes = {
        "segments": df_sent,
        "all_segments": df_all_enriched,
        "call": aggregates["call_sentiment"],
        "call_role": aggregates["call_role_sentiment"],
        "role": aggregates["role_global_sentiment"],
        "speaker": aggregates["speaker_sentiment"],
        "temporal": aggregates["temporal_sentiment"],
        "summary": summary_for_memory,
    }

    saved: list[Path] = []
    for name, dataframe in dataframes.items():
        path = Path(paths[name])
        write_csv_atomic(dataframe, path)
        saved.append(path)
    return saved
