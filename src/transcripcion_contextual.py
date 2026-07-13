"""Fase 05: transcripción contextual y alineación con segmentos diarizados.

Este módulo está descompuesto en funciones pequeñas a nivel de módulo, igual
que ``diarizacion.py``. El estado derivado de una ejecución (identificador de
run, rutas por audio, opciones de Whisper) se agrupa en ``TranscriptionContext``
y se pasa explícitamente a cada función, en lugar de compartirse por closure.

El notebook 05 orquesta estas funciones en celdas visibles (construir contexto →
restaurar desde GCS → cargar segmentos → resolver audios → cargar modelo →
procesar en batch con checkpoints → validar → publicar), de modo que el bucle
de checkpoints y la subida incremental a GCS sean visibles y auditables.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import gc
import hashlib
import json
import re
import time
import warnings

import librosa  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import soundfile as sf  # type: ignore

from src.config import (
    DATA_DIR,
    GCS_CLEAN_AUDIO_PREFIX,
    GCS_TRANSCRIPTION_PREFIX,
    GCS_DIARIZATION_OUTPUT_PREFIX,
    GCS_UNAV_ROOT,
    INPUT_DIR,
    OUTPUT_DIR,
    CONSOLIDATED_ALL_FINAL_MERGED_SEGMENTS_CSV,
    TRANSCRIPTION_ROOT,
    TRANSCRIPTION_MODEL_SIZE,
    TRANSCRIPTION_LANGUAGE,
    TRANSCRIPTION_TARGET_SR,
    TRANSCRIPTION_METHOD,
    TRANSCRIPTION_EXPECTED_AUDIOS,
    TRANSCRIPTION_SAVE_EVERY_N_AUDIOS,
    TRANSCRIPTION_NEAREST_WORD_TOLERANCE_SEC,
    TRANSCRIPTION_LOW_WORD_PROBABILITY,
    TRANSCRIPTION_FINAL_SEGMENTS_CSV,
    TRANSCRIPTION_FINAL_SUMMARY_CSV,
    TRANSCRIPTION_ACTIVE_RUN_JSON,
    ensure_phase05_directories,
)
from src.config import split_gcs_uri
from src.storage_io import (
    upload_file,
    upload_directory,
    download_directory,
    download_prefix_to_local,
    ensure_local_file,
    join_gcs_uri,
    delete_gcs_uri,
)
from src.io_utils import write_csv_atomic, write_text_atomic, write_json_atomic
from src.identidad_audio import normalize_audio_id as _shared_normalize_audio_id

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.max_colwidth", 180)


# ============================================================
# OPCIONES DE TRANSCRIPCIÓN POR DEFECTO
# ============================================================
# Estos son solo los valores de RESPALDO. Los valores efectivos se definen
# en la celda de CONFIGURACIÓN del notebook 05 y se pasan por parámetro.

INITIAL_PROMPT = (
    "Conversación telefónica en español entre un agente y un cliente. "
    "Transcripción literal con números, fechas, nombres y términos comerciales."
)


def default_transcribe_options(language=None, initial_prompt=None, use_vad=False):
    """Opciones por defecto de faster-whisper (respaldo si el notebook no las pasa)."""
    return {
        "language": language or TRANSCRIPTION_LANGUAGE,
        "task": "transcribe",
        "beam_size": 1,
        "temperature": 0.0,
        "word_timestamps": True,
        "without_timestamps": False,
        "vad_filter": use_vad,
        "condition_on_previous_text": True,
        "prompt_reset_on_temperature": 0.5,
        "initial_prompt": initial_prompt or INITIAL_PROMPT,
        "hotwords": None,
        "repetition_penalty": 1.05,
        "no_repeat_ngram_size": 3,
        "log_progress": False,
    }


MAX_PROJECTED_BATCH_HOURS = 48.0


# ============================================================
# CONTEXTO DE EJECUCIÓN
# ============================================================

@dataclass
class TranscriptionContext:
    """Agrupa todo el estado derivado de una ejecución de la fase 05."""

    gcs_client: object
    run_id: str
    run_signature: str
    run_config: dict

    # Directorios del run activo
    run_dir: Path
    per_audio_dir: Path
    per_audio_words_dir: Path
    per_audio_asr_dir: Path
    per_audio_text_dir: Path
    per_audio_meta_dir: Path
    checkpoint_dir: Path
    quarantine_dir: Path

    # CSV/JSON del run
    run_transcribed_segments_csv: Path
    run_transcription_summary_csv: Path
    run_manifest_csv: Path
    run_config_json: Path

    # Rutas canónicas (publicación final)
    canonical_per_audio_dir: Path
    canonical_words_dir: Path
    canonical_asr_dir: Path
    canonical_text_dir: Path
    canonical_all_segments_csv: Path
    canonical_summary_csv: Path
    canonical_final_segments_csv: Path
    canonical_final_summary_csv: Path
    canonical_active_run_json: Path

    # Prefijos GCS
    gcs_root_prefix: str
    gcs_run_prefix: str

    # Parámetros de proceso
    model_size: str
    language: str
    target_sr: int
    save_checkpoint_every_n: int
    nearest_word_tolerance_sec: float
    low_word_probability: float
    expected_final_audio_ids: int

    audio_dirs: list = field(default_factory=list)
    model: object = None
    device: str = "cpu"
    compute_type: str = "int8"

    # Opciones de transcripción configurables desde el notebook
    transcribe_options: dict = field(default_factory=dict)
    retry_empty_with_audio_gain: bool = True
    max_retry_gain: float = 10.0
    target_retry_rms: float = 0.08


def build_run_signature(run_config: dict, model_size=None) -> tuple[str, str]:
    """Devuelve (run_signature, run_id) reproducibles para una configuración."""
    model_size = model_size or TRANSCRIPTION_MODEL_SIZE
    signature = hashlib.sha256(
        json.dumps(run_config, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:12]
    run_id = (
        f"{TRANSCRIPTION_METHOD}_{model_size}_{signature}".replace("-", "_")
    )
    return signature, run_id


def build_transcription_context(
    gcs_client,
    model_size=None,
    language=None,
    target_sr=None,
    transcribe_options=None,
    retry_empty_with_audio_gain=True,
    max_retry_gain=10.0,
    target_retry_rms=0.08,
    save_checkpoint_every_n=None,
    nearest_word_tolerance_sec=None,
    low_word_probability=None,
    expected_final_audio_ids=None,
) -> TranscriptionContext:
    """
    Construye el contexto de ejecución de la fase 05.

    Todos los parámetros ajustables se reciben desde el notebook (celda de
    CONFIGURACIÓN). Si no se pasan, se usan los valores de ``config.py`` como
    respaldo. Crea las carpetas del run activo y calcula la firma reproducible.
    """
    model_size = model_size or TRANSCRIPTION_MODEL_SIZE
    language = language or TRANSCRIPTION_LANGUAGE
    target_sr = target_sr or TRANSCRIPTION_TARGET_SR
    transcribe_options = transcribe_options or default_transcribe_options(language)
    save_checkpoint_every_n = save_checkpoint_every_n or TRANSCRIPTION_SAVE_EVERY_N_AUDIOS
    nearest_word_tolerance_sec = (
        nearest_word_tolerance_sec
        if nearest_word_tolerance_sec is not None
        else TRANSCRIPTION_NEAREST_WORD_TOLERANCE_SEC
    )
    low_word_probability = (
        low_word_probability
        if low_word_probability is not None
        else TRANSCRIPTION_LOW_WORD_PROBABILITY
    )
    expected_final_audio_ids = expected_final_audio_ids or TRANSCRIPTION_EXPECTED_AUDIOS

    run_config = {
        "model_size": model_size,
        "language": language,
        "target_sr": target_sr,
        "transcription_method": TRANSCRIPTION_METHOD,
        "transcribe_options": transcribe_options,
        "retry_empty_with_audio_gain": retry_empty_with_audio_gain,
        "max_retry_gain": max_retry_gain,
        "target_retry_rms": target_retry_rms,
    }
    run_signature, run_id = build_run_signature(run_config, model_size)

    run_dir = TRANSCRIPTION_ROOT / "runs" / run_id
    ctx = TranscriptionContext(
        gcs_client=gcs_client,
        run_id=run_id,
        run_signature=run_signature,
        run_config=run_config,
        run_dir=run_dir,
        per_audio_dir=run_dir / "per_audio",
        per_audio_words_dir=run_dir / "per_audio_words",
        per_audio_asr_dir=run_dir / "per_audio_asr_segments",
        per_audio_text_dir=run_dir / "per_audio_full_text",
        per_audio_meta_dir=run_dir / "per_audio_meta",
        checkpoint_dir=run_dir / "checkpoints",
        quarantine_dir=run_dir / "quarantine",
        run_transcribed_segments_csv=run_dir / "all_segments_transcribed.csv",
        run_transcription_summary_csv=run_dir / "transcription_summary.csv",
        run_manifest_csv=run_dir / "processing_manifest.csv",
        run_config_json=run_dir / "run_config.json",
        canonical_per_audio_dir=TRANSCRIPTION_ROOT / "per_audio",
        canonical_words_dir=TRANSCRIPTION_ROOT / "per_audio_words",
        canonical_asr_dir=TRANSCRIPTION_ROOT / "per_audio_asr_segments",
        canonical_text_dir=TRANSCRIPTION_ROOT / "per_audio_full_text",
        canonical_all_segments_csv=TRANSCRIPTION_ROOT / "all_segments_transcribed.csv",
        canonical_summary_csv=TRANSCRIPTION_ROOT / "transcription_summary.csv",
        canonical_final_segments_csv=TRANSCRIPTION_FINAL_SEGMENTS_CSV,
        canonical_final_summary_csv=TRANSCRIPTION_FINAL_SUMMARY_CSV,
        canonical_active_run_json=TRANSCRIPTION_ACTIVE_RUN_JSON,
        gcs_root_prefix=GCS_TRANSCRIPTION_PREFIX,
        gcs_run_prefix=GCS_TRANSCRIPTION_PREFIX.rstrip("/") + f"/runs/{run_id}/",
        model_size=model_size,
        language=language,
        target_sr=target_sr,
        save_checkpoint_every_n=save_checkpoint_every_n,
        nearest_word_tolerance_sec=nearest_word_tolerance_sec,
        low_word_probability=low_word_probability,
        expected_final_audio_ids=expected_final_audio_ids,
        audio_dirs=[DATA_DIR / "clean_audios", INPUT_DIR],
        transcribe_options=transcribe_options,
        retry_empty_with_audio_gain=retry_empty_with_audio_gain,
        max_retry_gain=max_retry_gain,
        target_retry_rms=target_retry_rms,
    )

    for directory in [
        TRANSCRIPTION_ROOT, ctx.run_dir, ctx.per_audio_dir,
        ctx.per_audio_words_dir, ctx.per_audio_asr_dir, ctx.per_audio_text_dir,
        ctx.per_audio_meta_dir, ctx.checkpoint_dir, ctx.quarantine_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    write_json_atomic(
        {
            **run_config,
            "run_id": run_id,
            "run_signature": run_signature,
            "created_or_resumed_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        ctx.run_config_json,
    )

    return ctx


# ============================================================
# IDENTIFICADORES Y RUTAS POR AUDIO
# ============================================================

def normalize_audio_id(value):
    """Normaliza un nombre de archivo a su identificador base de audio."""
    value = Path(str(value)).name
    value = re.sub(r"\.(csv|wav|mp3|flac)$", "", value, flags=re.I)

    for suffix in [
        "_final_merged", "_transcribed_segments", "_whisper_words",
        "_whisper_asr_segments", "_clean",
    ]:
        value = re.sub(f"{re.escape(suffix)}$", "", value, flags=re.I)

    changed = True
    while changed:
        changed = False
        for prefix in ["raw_bajas_", "raw_comercial_", "raw_", "bajas_", "comercial_"]:
            if value.startswith(prefix):
                value = value[len(prefix):]
                changed = True

    return value


def make_segment_uid(audio_id_base, start, end, speaker_final):
    """UID reproducible de un segmento a partir de sus límites y speaker."""
    raw = f"{audio_id_base}|{float(start):.6f}|{float(end):.6f}|{str(speaker_final)}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def fingerprint_segment_uids(segment_uids):
    """Huella del conjunto de UIDs de segmentos de un audio."""
    joined = "|".join(sorted(map(str, segment_uids)))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def per_audio_output_path(ctx, audio_id_base):
    return ctx.per_audio_dir / f"{audio_id_base}_transcribed_segments.csv"


def per_audio_words_path(ctx, audio_id_base):
    return ctx.per_audio_words_dir / f"{audio_id_base}_whisper_words.csv"


def per_audio_asr_path(ctx, audio_id_base):
    return ctx.per_audio_asr_dir / f"{audio_id_base}_whisper_asr_segments.csv"


def per_audio_text_path(ctx, audio_id_base):
    return ctx.per_audio_text_dir / f"{audio_id_base}_full_transcript.txt"


def per_audio_meta_path(ctx, audio_id_base):
    return ctx.per_audio_meta_dir / f"{audio_id_base}_summary.json"


# ============================================================
# CARGA Y VALIDACIÓN DE SEGMENTOS DIARIZADOS
# ============================================================

def load_and_validate_segments(segments_csv=None):
    """
    Carga el consolidado de segmentos finales del Notebook 04, valida columnas,
    calcula UID por segmento y agrupa por audio.

    Devuelve (df_segments, target_audio_ids, audio_fingerprints,
    expected_segment_counts).
    """
    segments_csv = segments_csv or CONSOLIDATED_ALL_FINAL_MERGED_SEGMENTS_CSV
    if not Path(segments_csv).exists():
        raise FileNotFoundError(f"No existe el consolidado de segmentos: {segments_csv}")

    df_segments = pd.read_csv(segments_csv)

    required_columns = ["audio_file", "start", "end", "duration", "speaker_final"]
    missing = [c for c in required_columns if c not in df_segments.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")

    df_segments["audio_id_base"] = df_segments["audio_file"].apply(normalize_audio_id)
    df_segments["start"] = pd.to_numeric(df_segments["start"], errors="raise")
    df_segments["end"] = pd.to_numeric(df_segments["end"], errors="raise")

    df_segments["segment_uid"] = df_segments.apply(
        lambda row: make_segment_uid(
            row["audio_id_base"], row["start"], row["end"], row["speaker_final"],
        ),
        axis=1,
    )

    if df_segments["segment_uid"].duplicated().any():
        raise ValueError("Hay segmentos duplicados en el consolidado de diarización.")

    audio_fingerprints = (
        df_segments.groupby("audio_id_base")["segment_uid"]
        .apply(fingerprint_segment_uids).to_dict()
    )
    expected_segment_counts = (
        df_segments.groupby("audio_id_base").size().astype(int).to_dict()
    )
    target_audio_ids = sorted(audio_fingerprints)

    return df_segments, target_audio_ids, audio_fingerprints, expected_segment_counts


# ============================================================
# RESOLUCIÓN ROBUSTA DE AUDIOS LIMPIOS
# ============================================================

def get_audio_name_candidates(audio_file, audio_id_base):
    """Nombres candidatos para localizar el WAV limpio de un audio."""
    candidates = [
        str(audio_file),
        f"{audio_id_base}_clean.wav",
        f"raw_{audio_id_base}_clean.wav",
        f"raw_bajas_{audio_id_base}_clean.wav",
        f"raw_comercial_{audio_id_base}_clean.wav",
        f"{audio_id_base}.wav",
        f"raw_{audio_id_base}.wav",
        f"raw_bajas_{audio_id_base}.wav",
        f"raw_comercial_{audio_id_base}.wav",
    ]
    return list(dict.fromkeys(candidates))


def resolve_audio_path(ctx, audio_file, audio_id_base):
    """Localiza el WAV limpio de un audio entre los directorios configurados."""
    for audio_dir in ctx.audio_dirs:
        for candidate in get_audio_name_candidates(audio_file, audio_id_base):
            candidate_path = audio_dir / candidate
            if candidate_path.exists():
                return candidate_path
    return None


def build_audio_catalog(ctx, df_segments, audio_fingerprints):
    """
    Construye el catálogo de audios (ruta local, existencia, nº de segmentos).

    Devuelve (df_audio_catalog, audio_paths, df_missing_audio).
    """
    rows = []
    for audio_id_base, group in df_segments.groupby("audio_id_base", sort=True):
        audio_file = str(group["audio_file"].iloc[0])
        audio_path = resolve_audio_path(ctx, audio_file, audio_id_base)
        rows.append({
            "audio_id_base": audio_id_base,
            "audio_file": audio_file,
            "audio_path": str(audio_path) if audio_path else "",
            "audio_exists": audio_path is not None,
            "n_segments": len(group),
            "diarization_fingerprint": audio_fingerprints[audio_id_base],
        })

    df_audio_catalog = pd.DataFrame(rows)
    audio_paths = {
        row.audio_id_base: Path(row.audio_path)
        for row in df_audio_catalog.itertuples()
        if row.audio_exists
    }
    df_missing = df_audio_catalog.loc[~df_audio_catalog["audio_exists"]]
    return df_audio_catalog, audio_paths, df_missing


def restore_run_from_gcs(ctx):
    """Restaura desde GCS los archivos del run activo que falten localmente."""
    if ctx.gcs_client is None:
        return 0
    bucket_name, prefix = split_gcs_uri(ctx.gcs_run_prefix)
    prefix = prefix.rstrip("/") + "/"
    ctx.run_dir.mkdir(parents=True, exist_ok=True)
    restored = 0
    for blob in ctx.gcs_client.list_blobs(bucket_name, prefix=prefix):
        relative_name = blob.name[len(prefix):]
        if not relative_name:
            continue
        destination = ctx.run_dir / relative_name
        if destination.exists():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(destination))
        restored += 1
    return restored


# ============================================================
# CARGA DEL MODELO WHISPER
# ============================================================

def load_whisper_model(ctx=None, model_size=None):
    """
    Carga el modelo faster-whisper en GPU si está disponible, si no en CPU.

    Si se pasa ``ctx``, guarda el modelo, device y compute_type en él y lo
    devuelve; si no, devuelve solo el modelo.
    Import diferido de faster_whisper para no exigirlo al importar el módulo.
    """
    import os
    from faster_whisper import WhisperModel  # type: ignore
    from src.config import HF_TOKEN

    model_size = model_size or (ctx.model_size if ctx else TRANSCRIPTION_MODEL_SIZE)
    device = "cuda" if os.system("nvidia-smi > /dev/null 2>&1") == 0 else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    cpu_threads = max(1, int(os.cpu_count() or 4))

    print("Device:", device)
    print("Compute type:", compute_type)
    print("CPU threads disponibles:", cpu_threads)
    print("Modelo:", model_size)

    model = WhisperModel(
        model_size,
        device=device,
        compute_type=compute_type,
        cpu_threads=cpu_threads,
        num_workers=1,
        use_auth_token=HF_TOKEN if HF_TOKEN else None,
    )
    print("Modelo cargado correctamente.")

    if ctx is not None:
        ctx.model = model
        ctx.device = device
        ctx.compute_type = compute_type
        return ctx

    return model


# ============================================================
# UTILIDADES DE AUDIO Y SEÑAL
# ============================================================

def load_audio_mono(audio_path, target_sr=None):
    """Carga un audio como mono float32 al sample rate objetivo."""
    target_sr = target_sr or TRANSCRIPTION_TARGET_SR
    audio, sr = sf.read(audio_path, always_2d=False)

    if isinstance(audio, np.ndarray) and audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    audio = np.asarray(audio, dtype=np.float32)

    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr).astype(np.float32)
        sr = target_sr

    return audio, sr


def safe_float(value, default=np.nan):
    """Convierte a float de forma tolerante."""
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def audio_signal_metrics(audio, sr=None):
    """Calcula duración, pico, RMS, RMS en dBFS y ratio de muestras no nulas."""
    sr = sr or TRANSCRIPTION_TARGET_SR
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        return {
            "duration_sec": 0.0, "peak": 0.0, "rms": 0.0,
            "rms_dbfs": -np.inf, "nonzero_ratio": 0.0,
        }

    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(np.square(audio), dtype=np.float64)))
    rms_dbfs = float(20 * np.log10(max(rms, 1e-12)))

    return {
        "duration_sec": float(len(audio) / sr),
        "peak": peak,
        "rms": rms,
        "rms_dbfs": rms_dbfs,
        "nonzero_ratio": float(np.mean(np.abs(audio) > 1e-6)),
    }


def apply_controlled_gain(audio, target_retry_rms=0.08, max_retry_gain=10.0):
    """Aplica una ganancia acotada para reintentar audios con RMS muy bajo."""
    audio = np.asarray(audio, dtype=np.float32)
    rms = float(np.sqrt(np.mean(np.square(audio), dtype=np.float64)))

    if audio.size == 0 or rms <= 1e-12:
        return audio.copy(), 1.0

    gain = min(float(target_retry_rms / rms), float(max_retry_gain))
    boosted = np.clip(audio * gain, -1.0, 1.0).astype(np.float32)
    return boosted, gain


def clean_joined_whisper_words(words):
    """Une palabras de Whisper y normaliza espacios y puntuación."""
    if not words:
        return ""
    text = "".join(str(word) for word in words)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text


# ============================================================
# TRANSCRIPCIÓN COMPLETA DE UN AUDIO
# ============================================================

def transcribe_full_audio(model, audio, base_options, options_override=None):
    """
    Transcribe un audio completo con Whisper devolviendo segmentos ASR,
    palabras con timestamps y el texto completo.

    ``base_options`` son las opciones de faster-whisper definidas en el
    notebook (celda de CONFIGURACIÓN).
    """
    options = dict(base_options)
    if options_override:
        options.update(options_override)

    segments_generator, info = model.transcribe(audio, **options)

    asr_rows = []
    word_rows = []
    full_text_parts = []

    # faster-whisper es lazy: iterar el generador ejecuta la transcripción.
    for asr_segment_index, segment in enumerate(segments_generator):
        segment_text = (segment.text or "").strip()

        asr_rows.append({
            "asr_segment_index": asr_segment_index,
            "start": float(segment.start),
            "end": float(segment.end),
            "text": segment_text,
            "avg_logprob": safe_float(getattr(segment, "avg_logprob", np.nan)),
            "no_speech_prob": safe_float(getattr(segment, "no_speech_prob", np.nan)),
            "compression_ratio": safe_float(getattr(segment, "compression_ratio", np.nan)),
            "temperature": safe_float(getattr(segment, "temperature", np.nan)),
        })

        if segment_text:
            full_text_parts.append(segment_text)

        for word_index, word in enumerate(segment.words or []):
            word_text = str(word.word)
            word_rows.append({
                "asr_segment_index": asr_segment_index,
                "word_index": word_index,
                "word": word_text,
                "word_clean": word_text.strip(),
                "start": float(word.start),
                "end": float(word.end),
                "probability": safe_float(getattr(word, "probability", np.nan)),
            })

    df_asr_segments = pd.DataFrame(asr_rows)
    df_words = pd.DataFrame(word_rows)
    full_text = re.sub(r"\s+", " ", " ".join(full_text_parts)).strip()

    info_dict = {
        "detected_language": str(getattr(info, "language", TRANSCRIPTION_LANGUAGE)),
        "language_probability": safe_float(getattr(info, "language_probability", np.nan)),
        "audio_duration_sec": safe_float(getattr(info, "duration", np.nan)),
        "duration_after_vad_sec": safe_float(getattr(info, "duration_after_vad", np.nan)),
    }

    return df_asr_segments, df_words, full_text, info_dict


# ============================================================
# ASIGNACIÓN DE PALABRAS A SEGMENTOS DIARIZADOS
# ============================================================

def temporal_overlap(start_a, end_a, start_b, end_b):
    """Solapamiento temporal (segundos) entre dos intervalos."""
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


def distance_point_to_interval(point, start, end):
    """Distancia de un punto a un intervalo (0 si está dentro)."""
    if start <= point <= end:
        return 0.0
    return min(abs(point - start), abs(point - end))


def assign_words_to_diarized_segments(
    df_words,
    df_audio_segments,
    nearest_tolerance_sec=None,
):
    """
    Asigna cada palabra de Whisper al segmento diarizado con más solapamiento
    (o al más cercano dentro de la tolerancia). Devuelve (df_segmentos_con_texto,
    df_words_con_asignacion).
    """
    if nearest_tolerance_sec is None:
        nearest_tolerance_sec = TRANSCRIPTION_NEAREST_WORD_TOLERANCE_SEC

    diar = df_audio_segments.sort_values(["start", "end"]).reset_index(drop=True).copy()
    diar["diar_segment_index"] = np.arange(len(diar), dtype=int)

    words = df_words.copy()
    if words.empty:
        words = pd.DataFrame(columns=[
            "asr_segment_index", "word_index", "word", "word_clean",
            "start", "end", "probability",
        ])

    assignment_rows = []

    for word in words.itertuples(index=False):
        word_start = float(word.start)
        word_end = float(word.end)
        word_mid = (word_start + word_end) / 2

        overlap_values = np.maximum(
            0.0,
            np.minimum(diar["end"].to_numpy(float), word_end)
            - np.maximum(diar["start"].to_numpy(float), word_start),
        )

        if len(overlap_values) and float(overlap_values.max()) > 0:
            best_pos = int(np.argmax(overlap_values))
            best_overlap = float(overlap_values[best_pos])
            assignment_status = "max_overlap"
            assignment_distance = distance_point_to_interval(
                word_mid,
                float(diar.iloc[best_pos]["start"]),
                float(diar.iloc[best_pos]["end"]),
            )
        else:
            distances = np.array([
                distance_point_to_interval(word_mid, float(row.start), float(row.end))
                for row in diar.itertuples(index=False)
            ])
            best_pos = int(np.argmin(distances)) if len(distances) else -1
            assignment_distance = float(distances[best_pos]) if best_pos >= 0 else np.inf
            best_overlap = 0.0

            if best_pos >= 0 and assignment_distance <= nearest_tolerance_sec:
                assignment_status = "nearest_within_tolerance"
            else:
                best_pos = -1
                assignment_status = "unassigned"

        if best_pos >= 0:
            diar_row = diar.iloc[best_pos]
            assigned_index = int(diar_row["diar_segment_index"])
            assigned_uid = str(diar_row["segment_uid"])
            assigned_speaker = str(diar_row["speaker_final"])
        else:
            assigned_index = np.nan
            assigned_uid = ""
            assigned_speaker = ""

        assignment_rows.append({
            "diar_segment_index": assigned_index,
            "assigned_segment_uid": assigned_uid,
            "assigned_speaker_final": assigned_speaker,
            "assignment_overlap_sec": best_overlap,
            "assignment_distance_sec": assignment_distance,
            "assignment_status": assignment_status,
        })

    if len(words):
        words = pd.concat(
            [words.reset_index(drop=True), pd.DataFrame(assignment_rows)],
            axis=1,
        )
    else:
        for column in [
            "diar_segment_index", "assigned_segment_uid", "assigned_speaker_final",
            "assignment_overlap_sec", "assignment_distance_sec", "assignment_status",
        ]:
            words[column] = pd.Series(dtype="object")

    assigned_words = words.loc[words["diar_segment_index"].notna()].copy()
    if len(assigned_words):
        assigned_words["diar_segment_index"] = assigned_words["diar_segment_index"].astype(int)

    output_rows = []

    for diar_row in diar.itertuples(index=False):
        segment_words = assigned_words.loc[
            assigned_words["diar_segment_index"] == int(diar_row.diar_segment_index)
        ].sort_values(["start", "end", "word_index"])

        text = clean_joined_whisper_words(
            segment_words["word"].fillna("").astype(str).tolist()
        )

        original = diar.loc[
            diar["diar_segment_index"] == int(diar_row.diar_segment_index)
        ].iloc[0].drop(labels=["diar_segment_index"]).to_dict()

        original.update({
            "segment_index_in_audio": int(diar_row.diar_segment_index),
            "text": text,
            "transcription_status": "ok" if text else "empty_transcription",
            "n_words_assigned": int(len(segment_words)),
            "avg_word_probability": (
                float(segment_words["probability"].mean()) if len(segment_words) else np.nan
            ),
            "min_word_probability": (
                float(segment_words["probability"].min()) if len(segment_words) else np.nan
            ),
            "first_word_start": (
                float(segment_words["start"].min()) if len(segment_words) else np.nan
            ),
            "last_word_end": (
                float(segment_words["end"].max()) if len(segment_words) else np.nan
            ),
        })
        output_rows.append(original)

    return pd.DataFrame(output_rows), words


def assign_asr_segments_fallback(df_asr_segments, df_audio_segments):
    """Fallback para modelos que devolvieran texto por segmento sin palabras."""
    result = df_audio_segments.sort_values(["start", "end"]).reset_index(drop=True).copy()
    result["segment_index_in_audio"] = np.arange(len(result), dtype=int)
    result["text"] = ""
    result["transcription_status"] = "empty_transcription"
    result["n_words_assigned"] = 0
    result["avg_word_probability"] = np.nan
    result["min_word_probability"] = np.nan
    result["first_word_start"] = np.nan
    result["last_word_end"] = np.nan

    assigned_texts = {idx: [] for idx in result.index}

    for asr_row in df_asr_segments.itertuples(index=False):
        if not str(asr_row.text).strip():
            continue

        overlaps = np.maximum(
            0.0,
            np.minimum(result["end"].to_numpy(float), float(asr_row.end))
            - np.maximum(result["start"].to_numpy(float), float(asr_row.start)),
        )

        if len(overlaps) and float(overlaps.max()) > 0:
            best_pos = int(np.argmax(overlaps))
        else:
            midpoint = (float(asr_row.start) + float(asr_row.end)) / 2
            distances = np.array([
                distance_point_to_interval(midpoint, float(r.start), float(r.end))
                for r in result.itertuples(index=False)
            ])
            best_pos = int(np.argmin(distances))

        assigned_texts[best_pos].append(str(asr_row.text).strip())

    for idx, texts in assigned_texts.items():
        text = re.sub(r"\s+", " ", " ".join(texts)).strip()
        result.at[idx, "text"] = text
        result.at[idx, "transcription_status"] = (
            "ok_fallback_asr_segment" if text else "empty_transcription"
        )

    return result


# ============================================================
# VALIDACIÓN Y LIMPIEZA DE OUTPUTS POR AUDIO
# ============================================================

def validate_per_audio_output(
    ctx, audio_id_base, df_segments, expected_counts, fingerprints,
    return_reason=False,
):
    """
    Valida que el CSV por audio ya existente corresponde exactamente a esta
    configuración de run y a los segmentos esperados. Base de la reanudación:
    un audio válido no se reprocesa.
    """
    path = per_audio_output_path(ctx, audio_id_base)
    reason = "ok"

    if not path.exists():
        reason = "missing"
    else:
        try:
            df = pd.read_csv(path)
            expected_count = expected_counts[audio_id_base]
            expected_fingerprint = fingerprints[audio_id_base]
            segment_uids_expected = set(
                df_segments.loc[
                    df_segments["audio_id_base"] == audio_id_base, "segment_uid"
                ].astype(str)
            )

            checks = {
                "row_count": len(df) == expected_count,
                "segment_uid": (
                    "segment_uid" in df.columns
                    and set(df["segment_uid"].astype(str)) == segment_uids_expected
                ),
                "run_id": (
                    "transcription_run_id" in df.columns
                    and set(df["transcription_run_id"].dropna().astype(str)) == {ctx.run_id}
                ),
                "run_signature": (
                    "transcription_run_signature" in df.columns
                    and set(df["transcription_run_signature"].dropna().astype(str))
                    == {ctx.run_signature}
                ),
                "model": (
                    "asr_model" in df.columns
                    and set(df["asr_model"].dropna().astype(str)) == {ctx.model_size}
                ),
                "method": (
                    "transcription_method" in df.columns
                    and set(df["transcription_method"].dropna().astype(str))
                    == {TRANSCRIPTION_METHOD}
                ),
                "fingerprint": (
                    "diarization_fingerprint" in df.columns
                    and set(df["diarization_fingerprint"].dropna().astype(str))
                    == {expected_fingerprint}
                ),
                "no_duplicate_segments": (
                    "segment_uid" in df.columns and not df["segment_uid"].duplicated().any()
                ),
                "has_text_column": "text" in df.columns,
                "has_transcribed_text": (
                    "text" in df.columns
                    and df["text"].fillna("").astype(str).str.strip().ne("").any()
                ),
                "audio_status_ok": (
                    "audio_transcription_status" in df.columns
                    and set(df["audio_transcription_status"].dropna().astype(str)) == {"ok"}
                ),
                "auxiliary_files": all(
                    artifact.exists()
                    for artifact in [
                        per_audio_words_path(ctx, audio_id_base),
                        per_audio_asr_path(ctx, audio_id_base),
                        per_audio_text_path(ctx, audio_id_base),
                        per_audio_meta_path(ctx, audio_id_base),
                    ]
                ),
            }

            failed = [name for name, passed in checks.items() if not passed]
            if failed:
                reason = "failed:" + ",".join(failed)

        except Exception as exc:
            reason = f"read_error:{exc}"

    valid = reason == "ok"
    return (valid, reason) if return_reason else valid


def remove_current_run_artifacts(ctx, audio_id_base):
    """Elimina los artefactos por audio de esta corrida (para rehacerlo limpio)."""
    for path in [
        per_audio_output_path(ctx, audio_id_base),
        per_audio_words_path(ctx, audio_id_base),
        per_audio_asr_path(ctx, audio_id_base),
        per_audio_text_path(ctx, audio_id_base),
        per_audio_meta_path(ctx, audio_id_base),
    ]:
        path.unlink(missing_ok=True)


def audit_and_clean_run_outputs(
    ctx, target_audio_ids, df_segments, expected_counts, fingerprints,
):
    """
    Audita los CSV por audio existentes: elimina los incompatibles con esta
    firma y devuelve (df_audit, valid_ids, pending_ids).
    """
    target_set = set(target_audio_ids)
    audit_rows = []

    for path in sorted(ctx.per_audio_dir.glob("*_transcribed_segments.csv")):
        audio_id_base = normalize_audio_id(path.name)

        if audio_id_base not in target_set:
            valid, reason = False, "not_in_target"
        else:
            valid, reason = validate_per_audio_output(
                ctx, audio_id_base, df_segments, expected_counts, fingerprints,
                return_reason=True,
            )

        audit_rows.append({
            "audio_id_base": audio_id_base, "path": str(path),
            "valid": valid, "reason": reason,
        })

        if not valid:
            remove_current_run_artifacts(ctx, audio_id_base)

    valid_ids = {
        audio_id for audio_id in target_audio_ids
        if validate_per_audio_output(
            ctx, audio_id, df_segments, expected_counts, fingerprints
        )
    }
    pending_ids = [a for a in target_audio_ids if a not in valid_ids]

    return pd.DataFrame(audit_rows), valid_ids, pending_ids


def upload_audio_artifacts_to_gcs(ctx, audio_id_base):
    """Sube a GCS los artefactos de un audio recién procesado (checkpoint por audio)."""
    if ctx.gcs_client is None:
        return []
    uploaded = []
    for path in [
        per_audio_output_path(ctx, audio_id_base),
        per_audio_words_path(ctx, audio_id_base),
        per_audio_asr_path(ctx, audio_id_base),
        per_audio_text_path(ctx, audio_id_base),
        per_audio_meta_path(ctx, audio_id_base),
    ]:
        if path.exists():
            was_uploaded = upload_file(
                path, ctx.gcs_client, gcs_prefix=ctx.gcs_run_prefix,
                base_dir=ctx.run_dir, skip_unchanged=True,
            )
            if was_uploaded or path.exists():
                uploaded.append(join_gcs_uri(
                    ctx.gcs_run_prefix, path.relative_to(ctx.run_dir).as_posix()
                ))
    return uploaded


# ============================================================
# PROCESAMIENTO DE UN AUDIO (con checkpoint por audio)
# ============================================================

def process_one_audio(
    ctx, audio_id_base, df_segments, audio_paths, fingerprints,
    expected_counts=None, save_outputs=True, upload_each_audio=True,
):
    """
    Transcribe un audio, alinea palabras con sus segmentos diarizados, valida
    el resultado y (si save_outputs) escribe los CSV por audio y los sube a GCS.

    Es la unidad de checkpoint: cada audio válido queda persistido y subido,
    de modo que una interrupción no obliga a reprocesarlo.
    """
    started_at = time.time()
    audio_path = audio_paths[audio_id_base]
    df_audio_segments = (
        df_segments.loc[df_segments["audio_id_base"] == audio_id_base]
        .sort_values(["start", "end"]).reset_index(drop=True).copy()
    )

    audio, sr = load_audio_mono(audio_path, ctx.target_sr)
    signal_metrics = audio_signal_metrics(audio, sr)

    if signal_metrics["duration_sec"] <= 0 or signal_metrics["peak"] <= 1e-8:
        raise RuntimeError(f"Audio vacío o casi silencioso: {signal_metrics}")

    retry_strategy = "none"
    df_asr_segments, df_words, full_text, info = transcribe_full_audio(
        ctx.model, audio, ctx.transcribe_options,
    )

    if not full_text and bool(ctx.transcribe_options.get("vad_filter", False)):
        df_asr_segments, df_words, full_text, info = transcribe_full_audio(
            ctx.model, audio, ctx.transcribe_options, options_override={"vad_filter": False},
        )
        retry_strategy = "vad_disabled"

    retry_gain = 1.0
    if not full_text and ctx.retry_empty_with_audio_gain:
        boosted_audio, retry_gain = apply_controlled_gain(
            audio, ctx.target_retry_rms, ctx.max_retry_gain,
        )
        df_asr_segments, df_words, full_text, info = transcribe_full_audio(
            ctx.model, boosted_audio, ctx.transcribe_options,
            options_override={"vad_filter": False},
        )
        retry_strategy = f"controlled_gain_x{retry_gain:.3f}"

    if not full_text:
        raise RuntimeError(
            f"Whisper no produjo texto ni en el reintento. Métricas: {signal_metrics}"
        )

    if len(df_words):
        df_output, df_words_assigned = assign_words_to_diarized_segments(
            df_words, df_audio_segments, ctx.nearest_word_tolerance_sec,
        )
    else:
        df_output = assign_asr_segments_fallback(df_asr_segments, df_audio_segments)
        df_words_assigned = df_words.copy()

    n_words_total = int(len(df_words_assigned))
    if n_words_total and "assignment_status" in df_words_assigned.columns:
        n_words_assigned = int(df_words_assigned["assignment_status"].ne("unassigned").sum())
    else:
        n_words_assigned = 0

    assignment_coverage = n_words_assigned / n_words_total if n_words_total else np.nan
    audio_status = "ok"

    common_fields = {
        "audio_id_base": audio_id_base,
        "resolved_audio_path": str(audio_path),
        "asr_model": ctx.model_size,
        "asr_device": ctx.device,
        "asr_compute_type": ctx.compute_type,
        "transcription_method": TRANSCRIPTION_METHOD,
        "transcription_run_id": ctx.run_id,
        "transcription_run_signature": ctx.run_signature,
        "diarization_fingerprint": fingerprints[audio_id_base],
        "audio_transcription_status": audio_status,
        "detected_language": info["detected_language"],
        "language_probability": info["language_probability"],
        "audio_duration_sec": info["audio_duration_sec"],
        "duration_after_vad_sec": info["duration_after_vad_sec"],
        "full_audio_n_words": n_words_total,
        "assigned_words": n_words_assigned,
        "word_assignment_coverage": assignment_coverage,
        "audio_peak": signal_metrics["peak"],
        "audio_rms_dbfs": signal_metrics["rms_dbfs"],
        "asr_retry_strategy": retry_strategy,
        "asr_retry_gain": retry_gain,
    }
    for column, value in common_fields.items():
        df_output[column] = value

    if len(df_words_assigned):
        df_words_assigned.insert(0, "audio_id_base", audio_id_base)
        df_words_assigned["asr_model"] = ctx.model_size
        df_words_assigned["transcription_run_id"] = ctx.run_id
        df_words_assigned["transcription_run_signature"] = ctx.run_signature

    if len(df_asr_segments):
        df_asr_segments.insert(0, "audio_id_base", audio_id_base)
        df_asr_segments["asr_model"] = ctx.model_size
        df_asr_segments["transcription_run_id"] = ctx.run_id
        df_asr_segments["transcription_run_signature"] = ctx.run_signature

    n_segments_with_text = int(
        df_output["text"].fillna("").astype(str).str.strip().ne("").sum()
    )
    low_confidence_words = (
        int((df_words_assigned["probability"] < ctx.low_word_probability).sum())
        if len(df_words_assigned) and "probability" in df_words_assigned.columns else 0
    )

    summary = {
        "audio_id_base": audio_id_base,
        "audio_file": str(df_audio_segments["audio_file"].iloc[0]),
        "resolved_audio_path": str(audio_path),
        "status": audio_status,
        "n_diarized_segments": len(df_output),
        "n_segments_with_text": n_segments_with_text,
        "segment_text_coverage": (
            n_segments_with_text / len(df_output) if len(df_output) else 0.0
        ),
        "n_asr_segments": len(df_asr_segments),
        "n_words_total": n_words_total,
        "n_words_assigned": n_words_assigned,
        "word_assignment_coverage": assignment_coverage,
        "n_low_confidence_words": low_confidence_words,
        "detected_language": info["detected_language"],
        "language_probability": info["language_probability"],
        "audio_duration_sec": info["audio_duration_sec"],
        "duration_after_vad_sec": info["duration_after_vad_sec"],
        "audio_peak": signal_metrics["peak"],
        "audio_rms_dbfs": signal_metrics["rms_dbfs"],
        "asr_retry_strategy": retry_strategy,
        "asr_retry_gain": retry_gain,
        "full_text_chars": len(full_text),
        "asr_model": ctx.model_size,
        "transcription_method": TRANSCRIPTION_METHOD,
        "transcription_run_id": ctx.run_id,
        "transcription_run_signature": ctx.run_signature,
        "diarization_fingerprint": fingerprints[audio_id_base],
        "elapsed_sec": round(time.time() - started_at, 2),
        "processed_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    if save_outputs:
        write_csv_atomic(df_output, per_audio_output_path(ctx, audio_id_base))
        write_csv_atomic(df_words_assigned, per_audio_words_path(ctx, audio_id_base))
        write_csv_atomic(df_asr_segments, per_audio_asr_path(ctx, audio_id_base))
        write_text_atomic(full_text, per_audio_text_path(ctx, audio_id_base))
        write_json_atomic(summary, per_audio_meta_path(ctx, audio_id_base))

        valid, reason = validate_per_audio_output(
            ctx, audio_id_base, df_segments, expected_counts, fingerprints,
            return_reason=True,
        )
        if not valid:
            remove_current_run_artifacts(ctx, audio_id_base)
            raise RuntimeError(f"El output recién creado no superó la validación: {reason}")

        if upload_each_audio:
            upload_audio_artifacts_to_gcs(ctx, audio_id_base)

    del audio
    gc.collect()

    return df_output, df_words_assigned, df_asr_segments, full_text, summary


# ============================================================
# MANIFIESTO Y CHECKPOINTS DE PROGRESO
# ============================================================

def build_manifest_from_disk(
    ctx, target_audio_ids, df_segments, expected_counts, fingerprints, extra_rows=None,
):
    """Reconstruye el manifiesto de progreso leyendo los metadatos en disco."""
    rows = []
    for audio_id_base in target_audio_ids:
        valid, reason = validate_per_audio_output(
            ctx, audio_id_base, df_segments, expected_counts, fingerprints,
            return_reason=True,
        )
        if valid and per_audio_meta_path(ctx, audio_id_base).exists():
            meta = json.loads(per_audio_meta_path(ctx, audio_id_base).read_text(encoding="utf-8"))
            row = {**meta, "output_valid": True, "validation_reason": reason}
        else:
            row = {
                "audio_id_base": audio_id_base, "status": "pending",
                "output_valid": False, "validation_reason": reason,
            }
        rows.append(row)

    manifest = pd.DataFrame(rows)

    if extra_rows:
        updates = pd.DataFrame(extra_rows)
        if len(updates):
            manifest = manifest.set_index("audio_id_base")
            updates = updates.set_index("audio_id_base")
            for idx, update_row in updates.iterrows():
                for column, value in update_row.items():
                    manifest.loc[idx, column] = value
            manifest = manifest.reset_index()

    return manifest.sort_values("audio_id_base").reset_index(drop=True)


def save_processing_checkpoint(
    ctx, target_audio_ids, df_segments, expected_counts, fingerprints,
    processed_counter, extra_rows=None, final=False,
):
    """
    Guarda el manifiesto de progreso local y lo sube a GCS.

    Este es el checkpoint que permite reanudar el batch sin reprocesar.
    """
    manifest = build_manifest_from_disk(
        ctx, target_audio_ids, df_segments, expected_counts, fingerprints,
        extra_rows=extra_rows,
    )
    write_csv_atomic(manifest, ctx.run_manifest_csv)

    checkpoint_name = (
        "checkpoint_final.csv" if final else f"checkpoint_{processed_counter:06d}.csv"
    )
    checkpoint_path = ctx.checkpoint_dir / checkpoint_name
    write_csv_atomic(manifest, checkpoint_path)
    write_csv_atomic(manifest, ctx.checkpoint_dir / "checkpoint_latest.csv")

    if ctx.gcs_client is not None:
        for path in [
            ctx.run_config_json, ctx.run_manifest_csv,
            checkpoint_path, ctx.checkpoint_dir / "checkpoint_latest.csv",
        ]:
            upload_file(
                path, ctx.gcs_client, gcs_prefix=ctx.gcs_run_prefix,
                base_dir=ctx.run_dir, skip_unchanged=True,
            )

    return manifest


# ============================================================
# VALIDACIÓN FINAL Y CONSOLIDACIÓN EXACTA
# ============================================================

def finalize_and_consolidate(
    ctx, target_audio_ids, df_segments, expected_counts, fingerprints,
):
    """
    Verifica que existan outputs válidos para todos los audios y, si es así,
    consolida el CSV final de segmentos + el resumen por audio.

    Devuelve (run_is_complete, df_transcribed, df_summary, df_final_audit).
    """
    final_audit_rows = []
    for audio_id_base in target_audio_ids:
        valid, reason = validate_per_audio_output(
            ctx, audio_id_base, df_segments, expected_counts, fingerprints,
            return_reason=True,
        )
        final_audit_rows.append({"audio_id_base": audio_id_base, "valid": valid, "reason": reason})

    df_final_audit = pd.DataFrame(final_audit_rows)
    missing_or_invalid = df_final_audit.loc[~df_final_audit["valid"]]

    if len(missing_or_invalid):
        return False, None, None, df_final_audit

    per_audio_frames = []
    summary_rows = []
    for audio_id_base in target_audio_ids:
        per_audio_frames.append(pd.read_csv(per_audio_output_path(ctx, audio_id_base)))
        meta_path = per_audio_meta_path(ctx, audio_id_base)
        if meta_path.exists():
            summary_rows.append(json.loads(meta_path.read_text(encoding="utf-8")))

    df_transcribed = pd.concat(per_audio_frames, ignore_index=True)
    df_transcribed = df_transcribed.sort_values(
        ["audio_id_base", "start", "end", "segment_uid"]
    ).reset_index(drop=True)

    if len(df_transcribed) != len(df_segments):
        raise ValueError(
            f"Filas consolidadas ({len(df_transcribed)}) != "
            f"segmentos de diarización ({len(df_segments)})."
        )
    if df_transcribed["segment_uid"].duplicated().any():
        raise ValueError("El consolidado contiene segment_uid duplicados.")
    if set(df_transcribed["segment_uid"]) != set(df_segments["segment_uid"]):
        raise ValueError("El consolidado no contiene exactamente los segmentos esperados.")

    df_summary = pd.DataFrame(summary_rows).sort_values("audio_id_base").reset_index(drop=True)
    if len(df_summary) != len(target_audio_ids):
        raise ValueError("Falta al menos un resumen por audio; no se publica la corrida.")
    if df_summary["audio_id_base"].duplicated().any():
        raise ValueError("El resumen contiene audio_id_base duplicados.")

    write_csv_atomic(df_transcribed, ctx.run_transcribed_segments_csv)
    write_csv_atomic(df_summary, ctx.run_transcription_summary_csv)

    save_processing_checkpoint(
        ctx, target_audio_ids, df_segments, expected_counts, fingerprints,
        processed_counter=len(target_audio_ids), final=True,
    )

    if ctx.gcs_client is not None:
        for path in [ctx.run_transcribed_segments_csv, ctx.run_transcription_summary_csv]:
            upload_file(
                path, ctx.gcs_client, gcs_prefix=ctx.gcs_run_prefix,
                base_dir=ctx.run_dir, skip_unchanged=True,
            )

    return True, df_transcribed, df_summary, df_final_audit


# ============================================================
# PUBLICACIÓN CANÓNICA (LOCAL Y GCS)
# ============================================================

def publish_local_canonical(ctx, target_audio_ids, df_transcribed, delete_old=True):
    """Copia los outputs del run activo a las rutas canónicas de la fase 05."""
    import shutil

    def _remove(path):
        path = Path(path)
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists() or path.is_symlink():
            path.unlink()

    if delete_old:
        for path in [
            ctx.canonical_per_audio_dir, ctx.canonical_words_dir, ctx.canonical_asr_dir,
            ctx.canonical_text_dir, ctx.canonical_all_segments_csv, ctx.canonical_summary_csv,
            ctx.canonical_final_segments_csv, ctx.canonical_final_summary_csv,
            ctx.canonical_active_run_json,
        ]:
            _remove(path)

    shutil.copytree(ctx.per_audio_dir, ctx.canonical_per_audio_dir)
    shutil.copytree(ctx.per_audio_words_dir, ctx.canonical_words_dir)
    shutil.copytree(ctx.per_audio_asr_dir, ctx.canonical_asr_dir)
    shutil.copytree(ctx.per_audio_text_dir, ctx.canonical_text_dir)

    # Los dos nombres finales son aliases de compatibilidad del mismo DataFrame.
    shutil.copy2(ctx.run_transcribed_segments_csv, ctx.canonical_all_segments_csv)
    shutil.copy2(ctx.run_transcription_summary_csv, ctx.canonical_summary_csv)
    shutil.copy2(ctx.run_transcribed_segments_csv, ctx.canonical_final_segments_csv)
    shutil.copy2(ctx.run_transcription_summary_csv, ctx.canonical_final_summary_csv)

    active_run = {
        "run_id": ctx.run_id,
        "run_signature": ctx.run_signature,
        "model_size": ctx.model_size,
        "transcription_method": TRANSCRIPTION_METHOD,
        "published_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_audios": len(target_audio_ids),
        "n_segments": len(df_transcribed),
    }
    write_json_atomic(active_run, ctx.canonical_active_run_json)

    active_files = sorted(ctx.canonical_per_audio_dir.glob("*_transcribed_segments.csv"))
    active_ids = [normalize_audio_id(path.name) for path in active_files]

    if len(active_files) != len(target_audio_ids):
        raise RuntimeError("La carpeta activa no contiene un archivo por audio.")
    if len(active_ids) != len(set(active_ids)):
        raise RuntimeError("La carpeta activa contiene IDs duplicados.")
    if set(active_ids) != set(target_audio_ids):
        raise RuntimeError("La carpeta activa no coincide con el universo esperado.")

    return active_run


def publish_gcs_canonical(ctx):
    """Sincroniza toda la carpeta de transcripción con GCS (solo lo cambiado)."""
    if ctx.gcs_client is None:
        print("Publicación GCS omitida.")
        return
    upload_directory(
        local_dir=TRANSCRIPTION_ROOT,
        gcs_prefix=ctx.gcs_root_prefix,
        gcs_client=ctx.gcs_client,
        skip_unchanged=True,
    )
    print("Outputs activos sincronizados con GCS.")


def ensure_canonical_outputs_in_gcs(gcs_client):
    """
    Sube a GCS la carpeta canónica de la fase 05, sin necesitar un ``ctx``.

    Se llama siempre al final del notebook, independientemente de si la fase
    se ejecutó o se saltó por tener outputs locales completos, para
    garantizar que lo que hay en local también quede en GCS (caso típico:
    un compañero corrió la fase localmente pero nunca subió el resultado).
    ``skip_unchanged=True`` hace que esto sea barato si ya estaba subido.
    """
    if gcs_client is None:
        print("Publicación GCS omitida (sin cliente).")
        return
    if not TRANSCRIPTION_ROOT.exists():
        print("No hay carpeta de transcripción local que sincronizar.")
        return
    result = upload_directory(
        local_dir=TRANSCRIPTION_ROOT,
        gcs_prefix=GCS_TRANSCRIPTION_PREFIX,
        gcs_client=gcs_client,
        skip_unchanged=True,
    )
    print("Sincronización con GCS asegurada (incondicional).")
    return result


# ============================================================
# RESTAURACIÓN DE FASE COMPLETA (salto de fase)
# ============================================================

def restore_phase_outputs_from_gcs(gcs_client):
    """
    Restaura desde GCS los outputs de la fase 05 (carpeta canónica completa).

    Se usa al inicio del notebook para poder saltar la fase si ya está hecha.
    """
    ensure_phase05_directories()
    download_directory(
        local_dir=TRANSCRIPTION_ROOT,
        gcs_prefix=GCS_UNAV_ROOT,
        gcs_client=gcs_client,
        base_dir=DATA_DIR,
    )


def phase_outputs_complete() -> bool:
    """Indica si los outputs finales de la fase 05 ya existen localmente."""
    return (
        TRANSCRIPTION_FINAL_SEGMENTS_CSV.exists()
        and TRANSCRIPTION_FINAL_SUMMARY_CSV.exists()
        and TRANSCRIPTION_ACTIVE_RUN_JSON.exists()
    )


def load_existing_phase_outputs() -> dict:
    """Carga los outputs finales ya existentes de la fase 05 (sin reejecutar)."""
    return {
        "df_transcribed": pd.read_csv(TRANSCRIPTION_FINAL_SEGMENTS_CSV),
        "df_transcription_summary": pd.read_csv(TRANSCRIPTION_FINAL_SUMMARY_CSV),
        "active_run": json.loads(TRANSCRIPTION_ACTIVE_RUN_JSON.read_text(encoding="utf-8")),
        "reused": True,
    }
