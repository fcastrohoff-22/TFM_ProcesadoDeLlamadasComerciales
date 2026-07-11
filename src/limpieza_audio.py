"""Funciones para la carga, limpieza y exportación de audios."""

import io

import librosa  # type: ignore
import numpy as np  # type: ignore
import soundfile as sf  # type: ignore

from src.config import split_gcs_uri


# CARGA Y ESTANDARIZACIÓN DEL AUDIO

def load_audio_standard_from_gcs(
    gcs_uri,
    target_sr=16000,
    gcs_client=None,
):
    """
    Descarga un audio desde GCS, lo convierte a mono
    y lo remuestrea a la frecuencia indicada.
    """
    if gcs_client is None:
        raise ValueError("Debe proporcionarse gcs_client.")

    bucket_name, blob_path = split_gcs_uri(gcs_uri)
    blob = gcs_client.bucket(bucket_name).blob(blob_path)
    audio_bytes = blob.download_as_bytes()

    y, sr = sf.read(
        io.BytesIO(audio_bytes),
        always_2d=False,
    )

    if isinstance(y, np.ndarray) and y.ndim > 1:
        y = np.mean(y, axis=1)

    y = y.astype(np.float32)

    if sr != target_sr:
        y = librosa.resample(
            y,
            orig_sr=sr,
            target_sr=target_sr,
        )
        sr = target_sr

    return y, sr, audio_bytes


def basic_audio_stats(y, sr):
    """Calcula duración, amplitud máxima y energía RMS."""
    duration_sec = len(y) / sr if sr > 0 else 0.0
    max_amp = float(np.max(np.abs(y))) if len(y) > 0 else 0.0
    rms = float(np.sqrt(np.mean(y ** 2))) if len(y) > 0 else 0.0

    return {
        "duration_sec": duration_sec,
        "max_amplitude": max_amp,
        "rms_energy": rms,
    }


# LIMPIEZA HÍBRIDA DEL AUDIO

def clean_audio_hybrid(
    y,
    sr,
    top_db=30,
    min_silence_len_sec=0.30,
    max_internal_silence_sec=0.75,
):
    """
    Elimina el silencio inicial y final y comprime los silencios
    internos que superen el máximo configurado.
    """
    original_stats = basic_audio_stats(y, sr)

    nonsilent_intervals = librosa.effects.split(
        y,
        top_db=top_db,
    )

    if len(nonsilent_intervals) == 0:
        cleaning_info = {
            "status": "empty_after_split",
            "top_db": top_db,
            "original_duration_sec": original_stats["duration_sec"],
            "clean_duration_sec": 0.0,
            "removed_duration_sec": original_stats["duration_sec"],
            "removed_ratio": (
                1.0
                if original_stats["duration_sec"] > 0
                else np.nan
            ),
            "n_nonsilent_intervals": 0,
            "trim_applied": False,
            "internal_silence_compression_applied": False,
            "max_internal_silence_sec": max_internal_silence_sec,
        }

        return np.array([], dtype=np.float32), cleaning_info

    chunks = []
    internal_silence_compression_applied = False

    for idx, (start, end) in enumerate(nonsilent_intervals):
        chunks.append(y[start:end])

        if idx < len(nonsilent_intervals) - 1:
            next_start = nonsilent_intervals[idx + 1][0]
            gap_samples = next_start - end
            gap_sec = gap_samples / sr

            if gap_sec <= min_silence_len_sec:
                gap_to_keep_samples = gap_samples
            else:
                gap_to_keep_samples = int(
                    min(
                        gap_sec,
                        max_internal_silence_sec,
                    )
                    * sr
                )

                if gap_sec > max_internal_silence_sec:
                    internal_silence_compression_applied = True

            if gap_to_keep_samples > 0:
                chunks.append(
                    np.zeros(
                        gap_to_keep_samples,
                        dtype=y.dtype,
                    )
                )

    y_clean = (
        np.concatenate(chunks)
        if len(chunks) > 0
        else np.array([], dtype=np.float32)
    )

    clean_stats = basic_audio_stats(y_clean, sr)

    removed_duration_sec = (
        original_stats["duration_sec"]
        - clean_stats["duration_sec"]
    )

    removed_ratio = (
        removed_duration_sec / original_stats["duration_sec"]
        if original_stats["duration_sec"] > 0
        else np.nan
    )

    cleaning_info = {
        "status": "ok",
        "top_db": top_db,
        "original_duration_sec": original_stats["duration_sec"],
        "clean_duration_sec": clean_stats["duration_sec"],
        "removed_duration_sec": removed_duration_sec,
        "removed_ratio": removed_ratio,
        "n_nonsilent_intervals": len(nonsilent_intervals),
        "trim_applied": True,
        "internal_silence_compression_applied": (
            internal_silence_compression_applied
        ),
        "max_internal_silence_sec": max_internal_silence_sec,
        "original_max_amplitude": original_stats["max_amplitude"],
        "clean_max_amplitude": clean_stats["max_amplitude"],
        "original_rms_energy": original_stats["rms_energy"],
        "clean_rms_energy": clean_stats["rms_energy"],
    }

    return y_clean, cleaning_info


# PROCESAMIENTO Y EXPORTACIÓN DE UN AUDIO

def process_one_audio(
    row,
    clean_gcs_prefix,
    top_db=30,
    target_sr=16000,
    min_silence_len_sec=0.30,
    max_internal_silence_sec=0.75,
    min_valid_duration_sec=10.0,
    max_removed_ratio=0.90,
    gcs_client=None,
):
    """
    Procesa un audio del inventario, aplica la limpieza y sube
    el resultado válido a la ruta configurada de GCS.
    """
    if gcs_client is None:
        raise ValueError("Debe proporcionarse gcs_client.")

    source_dataset = row["source_dataset"]
    audio_id = row["audio_id"]
    audio_name = row["audio_name"]
    gcs_uri = row["gcs_uri"]

    y, sr, _ = load_audio_standard_from_gcs(
        gcs_uri=gcs_uri,
        gcs_client=gcs_client,
        target_sr=target_sr,
    )

    y_clean, cleaning_info = clean_audio_hybrid(
        y=y,
        sr=sr,
        top_db=top_db,
        min_silence_len_sec=min_silence_len_sec,
        max_internal_silence_sec=max_internal_silence_sec,
    )

    clean_filename = f"{source_dataset}_{audio_id}_clean.wav"

    clean_bucket_name, clean_prefix = split_gcs_uri(
        clean_gcs_prefix
    )

    clean_blob_path = f"{clean_prefix}{clean_filename}"
    clean_gcs_uri = (
        f"gs://{clean_bucket_name}/{clean_blob_path}"
    )

    valid_audio = (
        cleaning_info["status"] == "ok"
        and len(y_clean) > 0
        and cleaning_info["clean_duration_sec"]
        >= min_valid_duration_sec
        and cleaning_info["removed_ratio"]
        <= max_removed_ratio
    )

    if valid_audio:
        audio_buffer = io.BytesIO()

        sf.write(
            audio_buffer,
            y_clean,
            sr,
            format="WAV",
        )

        audio_buffer.seek(0)

        clean_blob = (
            gcs_client
            .bucket(clean_bucket_name)
            .blob(clean_blob_path)
        )

        clean_blob.upload_from_file(
            audio_buffer,
            content_type="audio/wav",
        )

    result = {
        "source_dataset": source_dataset,
        "audio_id": audio_id,
        "audio_name": audio_name,
        "gcs_uri": gcs_uri,
        "clean_filename": clean_filename if valid_audio else None,
        "clean_gcs_uri": clean_gcs_uri if valid_audio else None,
        "valid_audio": valid_audio,
        "target_sr": target_sr,
        "min_valid_duration_sec": min_valid_duration_sec,
        "max_removed_ratio": max_removed_ratio,
        **cleaning_info,
    }

    return result