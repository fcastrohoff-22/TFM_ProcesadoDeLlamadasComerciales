"""Fase 07B: reconocimiento afectivo desde audio y análisis prosódico.

La carga/restauración desde GCS y las visualizaciones permanecen visibles en el
notebook. Este módulo contiene funciones reutilizables para preparar segmentos,
aplicar el modelo SER, extraer prosodia, calcular scores y construir agregados.
Los modelos pesados se importan de forma diferida para no cargarlos cuando los
outputs ya están completos.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from src.io_utils import write_csv_atomic


# ============================================================
# UTILIDADES GENERALES
# ============================================================


def first_existing(paths: Iterable[Path]) -> Path | None:
    """Devuelve el primer archivo existente y no vacío."""
    for path in paths:
        candidate = Path(path)
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    return None


def detect_col(
    df: pd.DataFrame,
    candidates: Iterable[str],
    required: bool = False,
    label: str = "columna",
) -> str | None:
    """Detecta la primera columna disponible entre varias candidatas."""
    for column in candidates:
        if column in df.columns:
            return column
    if required:
        raise ValueError(
            f"No se encontró {label}. Candidatas: {list(candidates)}. "
            f"Columnas disponibles: {list(df.columns)[:80]}"
        )
    return None


def outputs_complete(paths: Iterable[Path]) -> bool:
    """Comprueba que todos los outputs requeridos existan y no estén vacíos."""
    return all(
        Path(path).exists() and Path(path).stat().st_size > 0
        for path in paths
    )


def sec_to_mmss(value: object) -> str:
    """Convierte segundos a MM:SS.ss."""
    if pd.isna(value):
        return ""
    seconds_total = float(value)
    minutes = int(seconds_total // 60)
    seconds = seconds_total - 60 * minutes
    return f"{minutes:02d}:{seconds:05.2f}"


def make_interval_label(start: object, end: object) -> str:
    """Construye el intervalo legible utilizado en auditoría."""
    return f"{sec_to_mmss(start)} - {sec_to_mmss(end)}"


def robust_zscore(series: pd.Series) -> pd.Series:
    """Z-score robusto basado en mediana y MAD, igual que el original."""
    numeric = pd.to_numeric(series, errors="coerce")
    median = numeric.median()
    mad = (numeric - median).abs().median()
    if pd.isna(mad) or mad == 0:
        std = numeric.std()
        if pd.isna(std) or std == 0:
            return pd.Series(0.0, index=numeric.index)
        return (numeric - numeric.mean()) / std
    return 0.6745 * (numeric - median) / mad


def safe_minmax(series: pd.Series) -> pd.Series:
    """Escalado robusto entre percentiles 1 y 99, igual que el original."""
    numeric = pd.to_numeric(series, errors="coerce")
    low = numeric.quantile(0.01)
    high = numeric.quantile(0.99)
    if pd.isna(low) or pd.isna(high) or high <= low:
        return pd.Series(np.nan, index=numeric.index)
    clipped = numeric.clip(low, high)
    return (clipped - low) / (high - low)


def normalize_audio_stem(value: object) -> str | None:
    """Normaliza nombres de audio sin alterar su identificador base."""
    if pd.isna(value):
        return None
    name = Path(str(value)).name
    return name[:-4] if name.lower().endswith(".wav") else name


def choose_role_col(df: pd.DataFrame) -> str | None:
    """Detecta la columna de rol proxy disponible."""
    return detect_col(
        df,
        [
            "official_role_proxy",
            "role_proxy",
            "assigned_role",
            "probable_role",
            "role",
        ],
        required=False,
        label="role proxy",
    )


# ============================================================
# PREPARACIÓN DE SEGMENTOS
# ============================================================


def load_and_prepare_segments(
    proxy_segment_csv: Path,
    fallback_segments_csv: Path,
) -> tuple[pd.DataFrame, dict[str, str | None], Path]:
    """Carga segmentos y crea las columnas estándar de la fase 07B."""
    input_path = first_existing([proxy_segment_csv, fallback_segments_csv])
    if input_path is None:
        raise FileNotFoundError(
            "No se encontró segment_level_proxy_groundtruth.csv ni "
            "all_final_merged_segments.csv."
        )

    df_segments = pd.read_csv(input_path)
    audio_file_col = detect_col(
        df_segments,
        ["audio_file", "filename", "file", "audio_name"],
        required=False,
        label="audio_file",
    )
    audio_stem_col = detect_col(
        df_segments,
        ["audio_stem", "audio_id", "audio_base", "file_stem"],
        required=False,
        label="audio_stem",
    )
    start_col = detect_col(
        df_segments,
        ["start", "start_sec", "start_time"],
        required=True,
        label="start",
    )
    end_col = detect_col(
        df_segments,
        ["end", "end_sec", "end_time"],
        required=True,
        label="end",
    )
    speaker_col = detect_col(
        df_segments,
        ["speaker_final", "speaker", "speaker_relabel"],
        required=False,
        label="speaker",
    )
    role_col = choose_role_col(df_segments)
    overlap_col = detect_col(
        df_segments,
        ["overlap_ratio", "overlap", "overlap_ratio_final"],
        required=False,
        label="overlap_ratio",
    )
    rms_input_col = detect_col(
        df_segments,
        ["rms_dbfs", "rms", "mean_rms_dbfs"],
        required=False,
        label="rms_dbfs",
    )

    if audio_stem_col is None and audio_file_col is None:
        raise ValueError("No se encontró ninguna columna de audio.")

    out = df_segments.copy()
    source_audio = audio_stem_col or audio_file_col
    out["audio_stem_norm"] = out[source_audio].apply(normalize_audio_stem)
    if audio_file_col is None:
        out["audio_file_norm"] = out["audio_stem_norm"].astype(str) + ".wav"
    else:
        out["audio_file_norm"] = out[audio_file_col].astype(str).apply(
            lambda value: Path(value).name
        )

    out["speaker_for_prosody"] = (
        out[speaker_col] if speaker_col is not None else np.nan
    )
    out["role_proxy_for_prosody"] = (
        out[role_col] if role_col is not None else np.nan
    )
    out["interval"] = [
        make_interval_label(start, end)
        for start, end in zip(out[start_col], out[end_col])
    ]

    columns = {
        "audio_file": audio_file_col,
        "audio_stem": audio_stem_col,
        "start": start_col,
        "end": end_col,
        "speaker": speaker_col,
        "role": role_col,
        "overlap": overlap_col,
        "rms_input": rms_input_col,
    }
    return out, columns, input_path


def filter_prosody_candidates(
    df_segments: pd.DataFrame,
    columns: Mapping[str, str | None],
    min_segment_duration_sec: float,
    max_segment_duration_sec: float,
    max_overlap_ratio_prosody: float,
    min_rms_dbfs_input: float,
    require_role_proxy: bool,
    max_segments_debug: int | None = None,
) -> pd.DataFrame:
    """Aplica los filtros originales y asigna prosody_row_id reproducible."""
    start_col = str(columns["start"])
    end_col = str(columns["end"])
    overlap_col = columns.get("overlap")
    rms_input_col = columns.get("rms_input")

    work = df_segments.copy()
    work["duration_for_prosody"] = (
        pd.to_numeric(work[end_col], errors="coerce")
        - pd.to_numeric(work[start_col], errors="coerce")
    )

    mask = work["duration_for_prosody"].between(
        min_segment_duration_sec,
        max_segment_duration_sec,
        inclusive="both",
    )
    if overlap_col is not None:
        mask &= (
            pd.to_numeric(work[overlap_col], errors="coerce").fillna(0)
            <= max_overlap_ratio_prosody
        )
    if rms_input_col is not None:
        mask &= (
            pd.to_numeric(work[rms_input_col], errors="coerce").fillna(-999)
            >= min_rms_dbfs_input
        )
    if require_role_proxy:
        role_values = work["role_proxy_for_prosody"]
        mask &= role_values.notna()
        mask &= ~role_values.astype(str).str.lower().isin(
            ["nan", "none", "", "no_textual_proxy"]
        )

    mask &= work["audio_stem_norm"].notna()
    normalized_stems = work["audio_stem_norm"].astype(str).str.lower()
    mask &= ~normalized_stems.eq("all")
    mask &= ~normalized_stems.str.startswith("all_")

    candidates = work[mask].copy().reset_index(drop=True)
    candidates["prosody_row_id"] = candidates.index.astype(int)

    if max_segments_debug is not None:
        candidates = candidates.head(max_segments_debug).copy().reset_index(drop=True)
        candidates["prosody_row_id"] = candidates.index.astype(int)

    return candidates


# ============================================================
# RESOLUCIÓN Y EXTRACCIÓN DE AUDIO
# ============================================================


def build_audio_index(audio_dir_candidates: Iterable[Path]) -> dict[str, Path]:
    """Indexa WAV por nombre y stem bajo las carpetas candidatas."""
    audio_index: dict[str, Path] = {}
    search_roots = [Path(path) for path in audio_dir_candidates if Path(path).exists()]
    for root in search_roots:
        for wav_path in root.rglob("*.wav"):
            audio_index.setdefault(wav_path.name, wav_path)
            audio_index.setdefault(wav_path.stem, wav_path)
    return audio_index


def resolve_audio_path(
    audio_index: Mapping[str, Path],
    audio_file: object = None,
    audio_stem: object = None,
) -> Path | None:
    """Resuelve variantes raw/clean conservadas por el pipeline original."""
    candidates: list[str] = []
    if audio_file is not None and not pd.isna(audio_file):
        name = Path(str(audio_file)).name
        candidates.append(name)
        if name.lower().endswith(".wav"):
            candidates.append(name[:-4])
    if audio_stem is not None and not pd.isna(audio_stem):
        stem = normalize_audio_stem(audio_stem)
        if stem:
            candidates.extend([stem, f"{stem}.wav"])

    expanded: list[str] = []
    for candidate in candidates:
        expanded.append(candidate)
        if not candidate.startswith("raw_"):
            expanded.append("raw_" + candidate)
            expanded.append("raw_bajas_" + candidate)
        if not candidate.endswith("_clean") and not candidate.endswith("_clean.wav"):
            if candidate.endswith(".wav"):
                expanded.append(candidate[:-4] + "_clean.wav")
            else:
                expanded.extend([candidate + "_clean", candidate + "_clean.wav"])

    for candidate in expanded:
        if candidate in audio_index:
            return Path(audio_index[candidate])
    return None


def extract_segment_audio(
    audio: np.ndarray,
    sample_rate: int,
    start_sec: object,
    end_sec: object,
) -> np.ndarray:
    """Extrae un segmento respetando los timestamps diarizados."""
    start_value = max(0.0, float(start_sec))
    end_value = max(start_value, float(end_sec))
    start_index = int(round(start_value * sample_rate))
    end_index = int(round(end_value * sample_rate))
    start_index = max(0, min(start_index, len(audio)))
    end_index = max(0, min(end_index, len(audio)))
    return audio[start_index:end_index]


# ============================================================
# MODELO SER
# ============================================================


def load_ser_pipeline(model_id: str, hf_token: str | None = None):
    """Carga el modelo SER español únicamente cuando es necesario."""
    import torch
    from transformers import pipeline

    device_index = 0 if torch.cuda.is_available() else -1
    token = hf_token.strip() if isinstance(hf_token, str) and hf_token.strip() else None
    ser_pipeline = pipeline(
        task="audio-classification",
        model=model_id,
        token=token,
        device=device_index,
    )
    return ser_pipeline, ("cuda" if device_index == 0 else "cpu")


def _load_audio(audio_path: Path, target_sr: int) -> tuple[np.ndarray, int]:
    import librosa

    return librosa.load(audio_path, sr=target_sr, mono=True)


def apply_ser_with_checkpoint(
    df_candidates: pd.DataFrame,
    columns: Mapping[str, str | None],
    audio_index: Mapping[str, Path],
    ser_pipeline,
    model_id: str,
    checkpoint_csv: Path,
    predictions_csv: Path,
    target_sr: int = 16000,
    save_every: int = 250,
    resume_from_checkpoint: bool = True,
    force: bool = False,
    max_segments: int | None = None,
) -> pd.DataFrame:
    """Aplica el modelo SER por segmento con checkpoint reanudable."""
    checkpoint_path = Path(checkpoint_csv)
    predictions_path = Path(predictions_csv)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    if predictions_path.exists() and predictions_path.stat().st_size > 0 and not force:
        return pd.read_csv(predictions_path)

    previous = pd.DataFrame()
    done_ids: set[int] = set()
    predictions: list[dict[str, object]] = []
    if resume_from_checkpoint and checkpoint_path.exists() and checkpoint_path.stat().st_size > 0 and not force:
        previous = pd.read_csv(checkpoint_path)
        if "prosody_row_id" in previous.columns:
            done_ids = set(
                pd.to_numeric(previous["prosody_row_id"], errors="coerce")
                .dropna()
                .astype(int)
                .tolist()
            )
            predictions = previous.to_dict("records")

    pending = df_candidates[
        ~df_candidates["prosody_row_id"].isin(done_ids)
    ].copy()

    if max_segments is not None:
        remaining = max(0, int(max_segments) - len(done_ids))
        if remaining <= 0:
            pending = pending.iloc[0:0].copy()
        elif "role_proxy_for_prosody" in pending.columns:
            role_count = max(1, pending["role_proxy_for_prosody"].nunique())
            per_role = max(1, remaining // role_count)
            samples = []
            for _, group in pending.groupby("role_proxy_for_prosody", dropna=False):
                samples.append(
                    group.sample(min(len(group), per_role), random_state=42)
                )
            pending = pd.concat(samples, ignore_index=True).head(remaining)
        else:
            pending = pending.sample(
                min(len(pending), remaining), random_state=42
            ).reset_index(drop=True)

    start_col = str(columns["start"])
    end_col = str(columns["end"])
    audio_cache: dict[Path, tuple[np.ndarray, int]] = {}

    try:
        from tqdm.auto import tqdm

        iterator = tqdm(
            pending.iterrows(), total=len(pending), desc="SER audio model"
        )
    except Exception:
        iterator = pending.iterrows()

    for _, row in iterator:
        audio_stem = row["audio_stem_norm"]
        audio_file = row["audio_file_norm"]
        audio_path = resolve_audio_path(audio_index, audio_file, audio_stem)
        if audio_path is None:
            continue

        try:
            if audio_path not in audio_cache:
                audio_cache[audio_path] = _load_audio(audio_path, target_sr)
            audio, sample_rate = audio_cache[audio_path]
            segment = extract_segment_audio(
                audio, sample_rate, row[start_col], row[end_col]
            )
            if len(segment) < int(0.5 * sample_rate):
                continue

            result = ser_pipeline(
                {
                    "array": segment.astype(np.float32),
                    "sampling_rate": sample_rate,
                },
                top_k=None,
            )
            if isinstance(result, list) and result and isinstance(result[0], list):
                result = result[0]
            if not isinstance(result, list) or not result:
                continue

            ordered = sorted(
                result, key=lambda item: item.get("score", 0), reverse=True
            )
            best = ordered[0]
            prediction: dict[str, object] = {
                "prosody_row_id": int(row["prosody_row_id"]),
                "audio_stem_norm": audio_stem,
                "audio_file_norm": audio_file,
                "start": row[start_col],
                "end": row[end_col],
                "interval": row.get("interval", ""),
                "speaker_for_prosody": row.get(
                    "speaker_for_prosody", np.nan
                ),
                "role_proxy_for_prosody": row.get(
                    "role_proxy_for_prosody", np.nan
                ),
                "ser_model_id": model_id,
                "ser_pred_label": best.get("label"),
                "ser_pred_score": best.get("score"),
            }
            for item in ordered:
                label = (
                    str(item.get("label", "unknown"))
                    .strip()
                    .replace(" ", "_")
                    .replace("/", "_")
                )
                prediction[f"ser_prob_{label}"] = item.get("score")
            predictions.append(prediction)
        except Exception as error:
            predictions.append(
                {
                    "prosody_row_id": int(row["prosody_row_id"]),
                    "audio_stem_norm": audio_stem,
                    "audio_file_norm": audio_file,
                    "start": row[start_col],
                    "end": row[end_col],
                    "interval": row.get("interval", ""),
                    "speaker_for_prosody": row.get(
                        "speaker_for_prosody", np.nan
                    ),
                    "role_proxy_for_prosody": row.get(
                        "role_proxy_for_prosody", np.nan
                    ),
                    "ser_model_id": model_id,
                    "ser_error": str(error)[:300],
                }
            )

        if predictions and len(predictions) % save_every == 0:
            checkpoint = pd.DataFrame(predictions).drop_duplicates(
                subset=["prosody_row_id"], keep="last"
            )
            write_csv_atomic(checkpoint, checkpoint_path)

    if predictions:
        final = pd.DataFrame(predictions).drop_duplicates(
            subset=["prosody_row_id"], keep="last"
        ).reset_index(drop=True)
    else:
        final = pd.DataFrame()

    if not final.empty:
        write_csv_atomic(final, checkpoint_path)
        write_csv_atomic(final, predictions_path)
    return final


# ============================================================
# PROSODIA
# ============================================================


def compute_prosodic_features(
    segment_audio: np.ndarray,
    sample_rate: int,
    frame_length: int = 1024,
    hop_length: int = 256,
    fmin: int = 50,
    fmax: int = 500,
) -> dict[str, object]:
    """Calcula las features acústicas originales con librosa."""
    import librosa

    output: dict[str, object] = {}
    audio = np.nan_to_num(np.asarray(segment_audio, dtype=np.float32))
    output["audio_samples"] = len(audio)
    output["audio_duration_calc"] = (
        len(audio) / sample_rate if sample_rate else np.nan
    )

    if len(audio) < int(0.20 * sample_rate):
        output["prosody_status"] = "too_short_audio"
        return output

    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    output["peak_abs"] = peak
    if peak <= 1e-6:
        output["prosody_status"] = "silent_audio"
        return output

    try:
        rms = librosa.feature.rms(
            y=audio, frame_length=frame_length, hop_length=hop_length
        )[0]
        zcr = librosa.feature.zero_crossing_rate(
            audio, frame_length=frame_length, hop_length=hop_length
        )[0]
        centroid = librosa.feature.spectral_centroid(
            y=audio,
            sr=sample_rate,
            n_fft=frame_length,
            hop_length=hop_length,
        )[0]
        bandwidth = librosa.feature.spectral_bandwidth(
            y=audio,
            sr=sample_rate,
            n_fft=frame_length,
            hop_length=hop_length,
        )[0]
        rolloff = librosa.feature.spectral_rolloff(
            y=audio,
            sr=sample_rate,
            n_fft=frame_length,
            hop_length=hop_length,
        )[0]

        rms_safe = np.maximum(rms, 1e-12)
        output["rms_audio_mean"] = float(np.mean(rms_safe))
        output["rms_audio_std"] = float(np.std(rms_safe))
        output["rms_audio_dbfs"] = float(
            20 * np.log10(np.mean(rms_safe))
        )
        output["energy_dynamic_range"] = float(
            np.percentile(rms_safe, 90) - np.percentile(rms_safe, 10)
        )
        output["zcr_mean"] = float(np.mean(zcr))
        output["zcr_std"] = float(np.std(zcr))
        output["spectral_centroid_mean"] = float(np.mean(centroid))
        output["spectral_centroid_std"] = float(np.std(centroid))
        output["spectral_bandwidth_mean"] = float(np.mean(bandwidth))
        output["spectral_rolloff_mean"] = float(np.mean(rolloff))

        try:
            f0, voiced_flag, _ = librosa.pyin(
                audio,
                fmin=fmin,
                fmax=fmax,
                sr=sample_rate,
                frame_length=frame_length,
                hop_length=hop_length,
            )
            valid_f0 = f0[~np.isnan(f0)] if f0 is not None else np.array([])
            output["voiced_ratio"] = (
                float(np.mean(voiced_flag))
                if voiced_flag is not None and len(voiced_flag)
                else np.nan
            )
            if len(valid_f0) > 0:
                output["pitch_mean"] = float(np.nanmean(valid_f0))
                output["pitch_std"] = float(np.nanstd(valid_f0))
                output["pitch_median"] = float(np.nanmedian(valid_f0))
                output["pitch_range_p90_p10"] = float(
                    np.nanpercentile(valid_f0, 90)
                    - np.nanpercentile(valid_f0, 10)
                )
            else:
                output.update(
                    {
                        "pitch_mean": np.nan,
                        "pitch_std": np.nan,
                        "pitch_median": np.nan,
                        "pitch_range_p90_p10": np.nan,
                    }
                )
        except Exception as error:
            output.update(
                {
                    "pitch_error": str(error)[:200],
                    "pitch_mean": np.nan,
                    "pitch_std": np.nan,
                    "pitch_median": np.nan,
                    "pitch_range_p90_p10": np.nan,
                    "voiced_ratio": np.nan,
                }
            )

        output["prosody_status"] = "ok"
        return output
    except Exception as error:
        output["prosody_status"] = "feature_error"
        output["prosody_error"] = str(error)[:300]
        return output


def extract_prosodic_features_with_checkpoint(
    df_candidates: pd.DataFrame,
    columns: Mapping[str, str | None],
    audio_index: Mapping[str, Path],
    checkpoint_csv: Path,
    target_sr: int = 16000,
    frame_length: int = 1024,
    hop_length: int = 256,
    fmin: int = 50,
    fmax: int = 500,
    save_every_audios: int = 25,
    resume_from_checkpoint: bool = True,
    force: bool = False,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Extrae features por audio completo y guarda checkpoints por lote."""
    checkpoint_path = Path(checkpoint_csv)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    done_ids: set[int] = set()
    if resume_from_checkpoint and checkpoint_path.exists() and checkpoint_path.stat().st_size > 0 and not force:
        previous = pd.read_csv(checkpoint_path)
        if "prosody_row_id" in previous.columns:
            done_ids = set(
                pd.to_numeric(previous["prosody_row_id"], errors="coerce")
                .dropna()
                .astype(int)
                .tolist()
            )
            rows = previous.to_dict("records")

    remaining = df_candidates[
        ~df_candidates["prosody_row_id"].isin(done_ids)
    ].copy()
    unique_audio = remaining[
        ["audio_file_norm", "audio_stem_norm"]
    ].drop_duplicates()

    start_col = str(columns["start"])
    end_col = str(columns["end"])
    missing_audio: list[str] = []
    processed_audio_count = 0

    try:
        from tqdm.auto import tqdm

        iterator = tqdm(
            unique_audio.iterrows(),
            total=len(unique_audio),
            desc="Features prosódicas",
        )
    except Exception:
        iterator = unique_audio.iterrows()

    for audio_number, (_, audio_row) in enumerate(iterator, start=1):
        audio_file = audio_row["audio_file_norm"]
        audio_stem = audio_row["audio_stem_norm"]
        audio_path = resolve_audio_path(audio_index, audio_file, audio_stem)
        audio_segments = remaining[
            remaining["audio_stem_norm"] == audio_stem
        ].copy()

        if audio_path is None:
            missing_audio.append(str(audio_stem))
            for _, segment_row in audio_segments.iterrows():
                base = segment_row.to_dict()
                base.update(
                    {
                        "resolved_audio_path": None,
                        "prosody_status": "audio_not_found",
                    }
                )
                rows.append(base)
        else:
            try:
                audio, sample_rate = _load_audio(audio_path, target_sr)
                processed_audio_count += 1
            except Exception as error:
                for _, segment_row in audio_segments.iterrows():
                    base = segment_row.to_dict()
                    base.update(
                        {
                            "resolved_audio_path": str(audio_path),
                            "prosody_status": "audio_load_error",
                            "prosody_error": str(error)[:300],
                        }
                    )
                    rows.append(base)
                continue

            for _, segment_row in audio_segments.iterrows():
                segment_audio = extract_segment_audio(
                    audio,
                    sample_rate,
                    segment_row[start_col],
                    segment_row[end_col],
                )
                features = compute_prosodic_features(
                    segment_audio,
                    sample_rate,
                    frame_length=frame_length,
                    hop_length=hop_length,
                    fmin=fmin,
                    fmax=fmax,
                )
                base = segment_row.to_dict()
                base.update(features)
                base["resolved_audio_path"] = str(audio_path)
                base["feature_sr"] = sample_rate
                rows.append(base)

        if rows and audio_number % save_every_audios == 0:
            checkpoint = pd.DataFrame(rows).drop_duplicates(
                subset=["prosody_row_id"], keep="last"
            )
            write_csv_atomic(checkpoint, checkpoint_path)

    if not rows:
        raise ValueError("No se generaron filas de features prosódicas.")

    features_df = pd.DataFrame(rows).drop_duplicates(
        subset=["prosody_row_id"], keep="last"
    ).reset_index(drop=True)
    write_csv_atomic(features_df, checkpoint_path)

    diagnostics = {
        "processed_audio_count": processed_audio_count,
        "missing_audio_count": len(set(missing_audio)),
        "missing_audio": sorted(set(missing_audio)),
        "segments_with_features": len(features_df),
    }
    return features_df, diagnostics


def compute_prosodic_scores(df_prosody_segments: pd.DataFrame) -> pd.DataFrame:
    """Calcula los cuatro scores y estados proxy con las fórmulas originales."""
    scores = df_prosody_segments.copy().reset_index(drop=True)
    if "prosody_row_id" not in scores.columns:
        scores["prosody_row_id"] = scores.index.astype(int)

    ok_mask = scores["prosody_status"].eq("ok")
    feature_columns = [
        "rms_audio_dbfs",
        "rms_audio_std",
        "energy_dynamic_range",
        "pitch_std",
        "pitch_range_p90_p10",
        "spectral_centroid_mean",
        "spectral_rolloff_mean",
        "zcr_mean",
    ]
    for column in feature_columns:
        scores[f"z_{column}"] = (
            robust_zscore(scores[column])
            if column in scores.columns
            else np.nan
        )

    raw_arousal = (
        0.35 * scores.get("z_rms_audio_dbfs", 0)
        + 0.20 * scores.get("z_energy_dynamic_range", 0)
        + 0.20 * scores.get("z_pitch_range_p90_p10", 0)
        + 0.15 * scores.get("z_spectral_centroid_mean", 0)
        + 0.10 * scores.get("z_zcr_mean", 0)
    )
    raw_tension = (
        0.25 * scores.get("z_pitch_std", 0)
        + 0.25 * scores.get("z_pitch_range_p90_p10", 0)
        + 0.20 * scores.get("z_rms_audio_std", 0)
        + 0.20 * scores.get("z_spectral_rolloff_mean", 0)
        + 0.10 * scores.get("z_zcr_mean", 0)
    )
    raw_intensity = (
        0.60 * scores.get("z_rms_audio_dbfs", 0)
        + 0.25 * scores.get("z_energy_dynamic_range", 0)
        + 0.15 * scores.get("z_spectral_centroid_mean", 0)
    )

    scores["arousal_proxy_score"] = safe_minmax(raw_arousal)
    scores["tension_proxy_score"] = safe_minmax(raw_tension)
    scores["intensity_proxy_score"] = safe_minmax(raw_intensity)
    scores["calm_proxy_score"] = 1 - scores["arousal_proxy_score"]

    score_columns = [
        "arousal_proxy_score",
        "tension_proxy_score",
        "intensity_proxy_score",
        "calm_proxy_score",
    ]
    for column in score_columns:
        scores.loc[~ok_mask, column] = np.nan

    conditions = [
        scores["tension_proxy_score"] >= 0.75,
        scores["arousal_proxy_score"] >= 0.75,
        scores["calm_proxy_score"] >= 0.75,
    ]
    choices = [
        "alta_tension_prosodica",
        "alta_activacion",
        "calma_prosodica",
    ]
    scores["prosodic_state_proxy"] = np.select(
        conditions, choices, default="neutral_prosodico"
    )
    scores.loc[~ok_mask, "prosodic_state_proxy"] = "sin_score"
    return add_demo_compatibility_columns(scores)



def add_demo_compatibility_columns(df_scores: pd.DataFrame) -> pd.DataFrame:
    """Añade aliases compatibles con la demo sin alterar las columnas científicas."""
    scores = df_scores.copy()
    aliases = {
        "prosodic_state": "prosodic_state_proxy",
        "arousal_score": "arousal_proxy_score",
        "tension_score": "tension_proxy_score",
        "intensity_score": "intensity_proxy_score",
        "calm_score": "calm_proxy_score",
        "ser_label": "ser_pred_label",
    }
    for alias, source in aliases.items():
        if source not in scores.columns:
            continue
        if alias not in scores.columns:
            scores[alias] = scores[source]
        else:
            scores[alias] = scores[alias].where(scores[alias].notna(), scores[source])
    return scores

def merge_ser_predictions(
    df_scores: pd.DataFrame,
    df_ser_predictions: pd.DataFrame,
    merge_into_existing: bool = True,
) -> pd.DataFrame:
    """Integra SER por prosody_row_id sin duplicar columnas antiguas."""
    scores = df_scores.copy().reset_index(drop=True)
    if "prosody_row_id" not in scores.columns:
        scores["prosody_row_id"] = scores.index.astype(int)
    if df_ser_predictions.empty:
        return add_demo_compatibility_columns(scores)

    ser_columns = [column for column in scores.columns if column.startswith("ser_")]
    if ser_columns and not merge_into_existing:
        return scores
    if ser_columns:
        scores = scores.drop(columns=ser_columns)

    drop_columns = [
        column
        for column in [
            "audio_stem_norm",
            "audio_file_norm",
            "start",
            "end",
            "interval",
            "speaker_for_prosody",
            "role_proxy_for_prosody",
        ]
        if column in df_ser_predictions.columns
    ]
    ser_to_merge = df_ser_predictions.drop(columns=drop_columns).drop_duplicates(
        subset=["prosody_row_id"], keep="last"
    )
    merged = scores.merge(ser_to_merge, on="prosody_row_id", how="left")
    return add_demo_compatibility_columns(merged)


# ============================================================
# AGREGACIONES Y CRUCE CON TEXTO
# ============================================================


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [
        "_".join(str(part) for part in column if str(part) != "").strip("_")
        if isinstance(column, tuple)
        else column
        for column in out.columns
    ]
    return out


def build_prosody_aggregates(df_scores: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Construye agregados por llamada, rol y speaker con esquema original."""
    aggregation: dict[str, list[str]] = {}
    for column, functions in {
        "duration_for_prosody": ["count", "sum", "mean"],
        "arousal_proxy_score": ["mean", "median", "max"],
        "tension_proxy_score": ["mean", "median", "max"],
        "intensity_proxy_score": ["mean", "median", "max"],
        "calm_proxy_score": ["mean", "median"],
        "rms_audio_dbfs": ["mean", "median"],
        "pitch_mean": ["mean", "median"],
        "pitch_std": ["mean", "median"],
        "voiced_ratio": ["mean"],
        "ser_pred_score": ["mean", "median"],
    }.items():
        if column in df_scores.columns:
            aggregation[column] = functions
    for column in [
        name for name in df_scores.columns if name.startswith("ser_prob_")
    ]:
        aggregation[column] = ["mean"]

    if not aggregation:
        raise ValueError("No hay columnas numéricas disponibles para agregación.")

    def grouped(group_columns: list[str]) -> pd.DataFrame:
        result = (
            df_scores.groupby(group_columns, dropna=False)
            .agg(aggregation)
            .pipe(_flatten_columns)
            .reset_index()
        )
        return result.rename(
            columns={
                "duration_for_prosody_count": "n_segments_prosody",
                "duration_for_prosody_sum": "total_seconds_prosody",
            }
        )

    return {
        "call": grouped(["audio_stem_norm", "audio_file_norm"]),
        "call_role": grouped(
            [
                "audio_stem_norm",
                "audio_file_norm",
                "role_proxy_for_prosody",
            ]
        ),
        "call_speaker": grouped(
            [
                "audio_stem_norm",
                "audio_file_norm",
                "speaker_for_prosody",
            ]
        ),
        "role": grouped(["role_proxy_for_prosody"]),
    }


def build_audio_text_comparison(
    df_scores: pd.DataFrame,
    df_text_sentiment: pd.DataFrame,
    start_col: str,
    end_col: str,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Une audio y texto conservando todas las columnas necesarias para 07C/demo."""
    if df_scores.empty or df_text_sentiment.empty:
        return pd.DataFrame(), {"matched_text": 0, "rows": 0}

    text_audio_col = detect_col(
        df_text_sentiment,
        ["audio_stem_norm", "audio_stem", "audio_file", "filename"],
        required=False,
        label="audio textual",
    )
    text_start_col = detect_col(
        df_text_sentiment,
        ["start", "start_sec", "start_time"],
        required=False,
        label="start textual",
    )
    text_end_col = detect_col(
        df_text_sentiment,
        ["end", "end_sec", "end_time"],
        required=False,
        label="end textual",
    )
    if text_audio_col is None or text_start_col is None or text_end_col is None:
        return pd.DataFrame(), {"matched_text": 0, "rows": 0}

    text = df_text_sentiment.copy()
    audio = df_scores.copy()
    text["audio_stem_norm"] = text[text_audio_col].apply(normalize_audio_stem)
    text["start_round"] = pd.to_numeric(
        text[text_start_col], errors="coerce"
    ).round(3)
    text["end_round"] = pd.to_numeric(
        text[text_end_col], errors="coerce"
    ).round(3)
    audio["start_round"] = pd.to_numeric(
        audio[start_col], errors="coerce"
    ).round(3)
    audio["end_round"] = pd.to_numeric(
        audio[end_col], errors="coerce"
    ).round(3)

    text_columns = ["audio_stem_norm", "start_round", "end_round"]
    for column in [
        "sentiment_uid",
        "sentiment_label_raw",
        "sentiment_label",
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

    text_unique = text[text_columns].drop_duplicates(
        ["audio_stem_norm", "start_round", "end_round"]
    )
    comparison = audio.merge(
        text_unique,
        on=["audio_stem_norm", "start_round", "end_round"],
        how="left",
        suffixes=("", "_text"),
    )

    label_column = detect_col(
        comparison,
        ["sentiment_label", "sentiment_textual_label"],
        required=False,
    )
    numeric_column = detect_col(
        comparison,
        ["sentiment_numeric"],
        required=False,
    )
    matched = pd.Series(False, index=comparison.index)
    if label_column is not None:
        matched |= comparison[label_column].notna()
    if numeric_column is not None:
        matched |= pd.to_numeric(
            comparison[numeric_column], errors="coerce"
        ).notna()

    diagnostics = {
        "matched_text": int(matched.sum()),
        "rows": len(comparison),
        "unique_audio": int(comparison["audio_stem_norm"].nunique()),
    }
    return comparison, diagnostics


def build_prosody_summary(
    df_segments: pd.DataFrame,
    df_candidates: pd.DataFrame,
    df_scores: pd.DataFrame,
    df_ser_predictions: pd.DataFrame,
    model_id: str,
    run_pretrained_ser_model: bool = True,
) -> pd.DataFrame:
    """Construye el resumen compacto original para memoria."""
    summary: dict[str, object] = {
        "n_segments_input": len(df_segments),
        "n_segments_candidates": len(df_candidates),
        "n_segments_audio_affect_output": len(df_scores),
        "n_segments_prosody_ok": (
            int(df_scores["prosody_status"].eq("ok").sum())
            if "prosody_status" in df_scores.columns
            else np.nan
        ),
        "n_audios_audio_affect": int(df_scores["audio_stem_norm"].nunique()),
        "n_roles": int(
            df_scores["role_proxy_for_prosody"].nunique(dropna=True)
        ),
        "ser_model_used": bool(
            run_pretrained_ser_model and not df_ser_predictions.empty
        ),
        "ser_model_id": model_id if run_pretrained_ser_model else None,
        "n_ser_predictions": int(len(df_ser_predictions)),
    }
    for column in [
        "arousal_proxy_score",
        "tension_proxy_score",
        "intensity_proxy_score",
        "calm_proxy_score",
        "ser_pred_score",
    ]:
        if column in df_scores.columns:
            summary[f"mean_{column}"] = float(
                df_scores[column].mean(skipna=True)
            )
            summary[f"median_{column}"] = float(
                df_scores[column].median(skipna=True)
            )

    if "ser_pred_label" in df_scores.columns:
        counts = df_scores["ser_pred_label"].value_counts(dropna=True)
        if len(counts) > 0:
            summary["most_common_ser_label"] = counts.index[0]
            summary["most_common_ser_label_n"] = int(counts.iloc[0])
    return pd.DataFrame([summary])


# ============================================================
# CARGA Y GUARDADO
# ============================================================


def load_prosody_outputs(paths: Mapping[str, Path]) -> dict[str, pd.DataFrame]:
    """Carga outputs existentes por nombre lógico."""
    loaded: dict[str, pd.DataFrame] = {}
    for name, path in paths.items():
        candidate = Path(path)
        if candidate.exists() and candidate.stat().st_size > 0:
            loaded[name] = pd.read_csv(candidate)
    return loaded


def save_prosody_outputs(
    df_scores: pd.DataFrame,
    aggregates: Mapping[str, pd.DataFrame],
    summary: pd.DataFrame,
    paths: Mapping[str, Path],
    comparison: pd.DataFrame | None = None,
) -> list[Path]:
    """Guarda los outputs originales de la fase 07B."""
    df_scores = add_demo_compatibility_columns(df_scores)
    dataframes: dict[str, pd.DataFrame] = {
        "segments": df_scores,
        "call": aggregates["call"],
        "call_role": aggregates["call_role"],
        "call_speaker": aggregates["call_speaker"],
        "role": aggregates["role"],
        "summary": summary,
    }
    if comparison is not None and not comparison.empty:
        dataframes["comparison"] = comparison

    saved: list[Path] = []
    for name, dataframe in dataframes.items():
        path = Path(paths[name])
        write_csv_atomic(dataframe, path)
        saved.append(path)
    return saved
