"""Funciones para el inventariado de audios y la carga de metadata."""

import hashlib
import io
import librosa  # type: ignore
from pathlib import Path
from typing import Callable, Optional

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import soundfile as sf  # type: ignore


# INVENTARIO TÉCNICO DE AUDIOS EN GCS

def build_audio_inventory(
    gcs_client,
    bucket_name: str,
    audio_prefixes: dict[str, str],
    progress_callback: Optional[Callable] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Construye el inventario técnico de los audios WAV almacenados en GCS.

    Descarga cada audio en memoria para leer sus propiedades técnicas,
    pero no modifica ni guarda archivos en Google Cloud Storage.
    """
    audio_rows = []
    errors = []
    all_wav_blobs = []

    # Localizar todos los archivos WAV de las fuentes configuradas
    for source_dataset, audio_prefix in audio_prefixes.items():
        blobs = list(
            gcs_client.list_blobs(
                bucket_name,
                prefix=audio_prefix,
            )
        )

        wav_blobs = [
            blob
            for blob in blobs
            if blob.name.lower().endswith(".wav")
        ]

        for blob in wav_blobs:
            all_wav_blobs.append({
                "source_dataset": source_dataset,
                "blob": blob,
            })

    total_audios = len(all_wav_blobs)

    # Descargar cada audio y leer sus propiedades técnicas
    for index, item in enumerate(all_wav_blobs, start=1):
        source_dataset = item["source_dataset"]
        blob = item["blob"]
        audio_name = str(Path(blob.name).name).strip()

        if progress_callback is not None:
            progress_callback(
                index,
                total_audios,
                source_dataset,
                audio_name,
            )

        try:
            gcs_uri = f"gs://{bucket_name}/{blob.name}"
            audio_id = str(
                audio_name.replace(".wav", "")
            ).strip()

            size_bytes = blob.size
            audio_bytes = blob.download_as_bytes()

            with sf.SoundFile(io.BytesIO(audio_bytes)) as audio_file:
                sample_rate = audio_file.samplerate
                n_channels = audio_file.channels
                n_frames = audio_file.frames

                duration_seconds = (
                    n_frames / sample_rate
                    if sample_rate
                    else None
                )

            audio_rows.append({
                "source_dataset": source_dataset,
                "audio_id": audio_id,
                "audio_name": audio_name,
                "gcs_uri": gcs_uri,
                "size_bytes": size_bytes,
                "duration_seconds": duration_seconds,
                "sample_rate": sample_rate,
                "n_channels": n_channels,
            })

        except Exception as error:
            errors.append({
                "source_dataset": source_dataset,
                "audio_name": audio_name,
                "gcs_uri": f"gs://{bucket_name}/{blob.name}",
                "error": str(error),
            })

    df_audio_inventory = pd.DataFrame(audio_rows)
    df_errors = pd.DataFrame(errors)

    return df_audio_inventory, df_errors


# CARGA DE METADATA DESDE BIGQUERY

def load_metadata_from_bigquery(
    bq_client,
    project_id: str,
    dataset: str,
    metadata_sources: dict[str, str],
) -> pd.DataFrame:
    """
    Consulta y concatena la metadata de las tablas configuradas en BigQuery.

    Solo realiza operaciones de lectura.
    """
    metadata_frames = []

    for source_dataset, table_name in metadata_sources.items():
        query = f"""
        SELECT *
        FROM `{project_id}.{dataset}.{table_name}`
        """

        df_source_metadata = (
            bq_client
            .query(query)
            .to_dataframe()
        )

        df_source_metadata["source_dataset"] = source_dataset
        df_source_metadata["bq_table"] = table_name

        metadata_frames.append(df_source_metadata)

    return pd.concat(
        metadata_frames,
        ignore_index=True,
    )


# SELECCIÓN, NORMALIZACIÓN Y ANONIMIZACIÓN DE METADATA

def hash_value(
    value,
    salt: str,
):
    """Genera un hash reproducible de 16 caracteres."""
    if pd.isna(value):
        return np.nan

    value = str(value).strip()

    return hashlib.sha256(
        f"{salt}_{value}".encode("utf-8")
    ).hexdigest()[:16]


def prepare_metadata(
    df_metadata: pd.DataFrame,
    anonymization_salt: str,
) -> pd.DataFrame:
    """
    Selecciona, normaliza y anonimiza la metadata necesaria.

    Mantiene el mismo comportamiento de la celda original.
    """
    df_metadata_selected = df_metadata[
        [
            "source_dataset",
            "bq_table",
            "filename",
            "customer_id",
            "agent_id",
            "brand_ds",
            "duration_min",
            "url",
        ]
    ].copy()

    text_columns = [
        "source_dataset",
        "bq_table",
        "filename",
        "customer_id",
        "agent_id",
        "brand_ds",
    ]

    for column in text_columns:
        df_metadata_selected[column] = (
            df_metadata_selected[column]
            .astype(str)
            .str.strip()
        )

    df_metadata_selected["audio_hash"] = (
        df_metadata_selected["filename"]
        .apply(
            lambda value: hash_value(
                value,
                anonymization_salt,
            )
        )
    )

    df_metadata_selected["customer_hash"] = (
        df_metadata_selected["customer_id"]
        .apply(
            lambda value: hash_value(
                value,
                anonymization_salt,
            )
        )
    )

    df_metadata_selected["agent_hash"] = (
        df_metadata_selected["agent_id"]
        .apply(
            lambda value: hash_value(
                value,
                anonymization_salt,
            )
        )
    )

    return df_metadata_selected

# ANÁLISIS DE THRESHOLDS DE SILENCIO

def calculate_silence_thresholds(
    df_audio_inventory: pd.DataFrame,
    gcs_client,
    thresholds: list[int],
    split_gcs_uri,
    progress_callback: Optional[Callable] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calcula el porcentaje de silencio de cada audio para varios valores de top_db.

    No guarda ni modifica archivos. Devuelve el detalle por audio y la tabla resumen.
    """
    threshold_rows = []
    total_audios = len(df_audio_inventory)

    # Calcular el silencio de cada audio para todos los thresholds
    for i, row in df_audio_inventory.iterrows():
        source_dataset = row["source_dataset"]
        audio_id = row["audio_id"]
        audio_name = row["audio_name"]
        gcs_uri = row["gcs_uri"]

        if progress_callback is not None:
            progress_callback(
                i + 1,
                total_audios,
                source_dataset,
                audio_name,
            )

        try:
            bucket_name, blob_path = split_gcs_uri(gcs_uri)
            blob = gcs_client.bucket(bucket_name).blob(blob_path)
            audio_bytes = blob.download_as_bytes()

            y, sr = sf.read(io.BytesIO(audio_bytes))

            if y.ndim > 1:
                y = np.mean(y, axis=1)

            total_duration_seconds = len(y) / sr if sr else 0.0

            row_result = {
                "source_dataset": source_dataset,
                "audio_id": audio_id,
                "audio_name": audio_name,
                "duration_seconds": total_duration_seconds,
            }

            for top_db in thresholds:
                nonsilent_intervals = librosa.effects.split(
                    y,
                    top_db=top_db,
                )

                nonsilent_samples = sum(
                    end - start
                    for start, end in nonsilent_intervals
                )

                nonsilent_duration_seconds = (
                    nonsilent_samples / sr
                    if sr else 0.0
                )

                silence_duration_seconds = (
                    total_duration_seconds
                    - nonsilent_duration_seconds
                )

                silence_ratio = (
                    silence_duration_seconds / total_duration_seconds
                    if total_duration_seconds > 0
                    else np.nan
                )

                row_result[
                    f"silence_ratio_top_db_{top_db}"
                ] = silence_ratio

            threshold_rows.append(row_result)

        except Exception as error:
            threshold_rows.append({
                "source_dataset": source_dataset,
                "audio_id": audio_id,
                "audio_name": audio_name,
                "duration_seconds": np.nan,
                "error": str(error),
            })

    df_threshold_detail = pd.DataFrame(threshold_rows)

    # Crear la tabla resumen para cada threshold
    summary_rows = []

    for top_db in thresholds:
        column = f"silence_ratio_top_db_{top_db}"

        summary_rows.append({
            "top_db": top_db,
            "mean_silence_ratio": df_threshold_detail[column].mean(),
            "median_silence_ratio": df_threshold_detail[column].median(),
            "std_silence_ratio": df_threshold_detail[column].std(),
            "min_silence_ratio": df_threshold_detail[column].min(),
            "max_silence_ratio": df_threshold_detail[column].max(),
            "audios_silence_gt_60": int(
                (df_threshold_detail[column] > 0.60).sum()
            ),
        })

    df_silence_threshold_summary = (
        pd.DataFrame(summary_rows)
        .set_index("top_db")
    )

    return df_threshold_detail, df_silence_threshold_summary

# CÁLCULO DE PROPORCIÓN DE SILENCIO

def calculate_silence_proportion(
    df_audio_inventory: pd.DataFrame,
    gcs_client,
    silence_top_db: int,
    split_gcs_uri,
    progress_callback: Optional[Callable] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calcula la proporción de silencio y no silencio de cada audio.

    No guarda ni modifica archivos. Devuelve los resultados y los errores.
    """
    silence_rows = []
    errores_silencio = []

    total_audios = len(df_audio_inventory)

    for i, row in df_audio_inventory.iterrows():
        source_dataset = row["source_dataset"]
        audio_id = row["audio_id"]
        audio_name = row["audio_name"]
        gcs_uri = row["gcs_uri"]

        if progress_callback is not None:
            progress_callback(
                i + 1,
                total_audios,
                source_dataset,
                audio_name,
            )

        try:
            bucket_name, blob_path = split_gcs_uri(gcs_uri)
            blob = gcs_client.bucket(bucket_name).blob(blob_path)
            audio_bytes = blob.download_as_bytes()

            y, sr = sf.read(io.BytesIO(audio_bytes))

            if y.ndim > 1:
                y = np.mean(y, axis=1)

            total_duration_seconds = len(y) / sr if sr else 0.0

            nonsilent_intervals = librosa.effects.split(
                y,
                top_db=silence_top_db,
            )

            nonsilent_samples = sum(
                end - start
                for start, end in nonsilent_intervals
            )

            nonsilent_duration_seconds = (
                nonsilent_samples / sr
                if sr else 0.0
            )

            silence_duration_seconds = (
                total_duration_seconds
                - nonsilent_duration_seconds
            )

            silence_ratio = (
                silence_duration_seconds / total_duration_seconds
                if total_duration_seconds > 0
                else np.nan
            )

            nonsilent_ratio = (
                nonsilent_duration_seconds / total_duration_seconds
                if total_duration_seconds > 0
                else np.nan
            )

            silence_rows.append({
                "source_dataset": source_dataset,
                "audio_id": audio_id,
                "silence_duration_seconds": silence_duration_seconds,
                "silence_ratio": silence_ratio,
                "nonsilent_duration_seconds": nonsilent_duration_seconds,
                "nonsilent_ratio": nonsilent_ratio,
            })

        except Exception as error:
            errores_silencio.append({
                "source_dataset": source_dataset,
                "audio_id": audio_id,
                "audio_name": audio_name,
                "gcs_uri": gcs_uri,
                "error": str(error),
            })

    df_silence = pd.DataFrame(silence_rows)
    df_errores_silencio = pd.DataFrame(errores_silencio)

    return df_silence, df_errores_silencio