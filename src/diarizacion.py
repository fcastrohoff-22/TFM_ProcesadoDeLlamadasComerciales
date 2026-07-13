"""Funciones de diarización, filtrado, selección de anchors y consolidación.

Las utilidades de CSV viven en ``src.io_utils`` y todo el transporte de
archivos hacia GCS vive en ``src.storage_io``. Este módulo se centra en la
lógica de diarización propiamente dicha.
"""

from pathlib import Path

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import soundfile as sf  # type: ignore
import torch  # type: ignore

from src.config import (
    OUTPUT_DIR,
    GCS_DIARIZATION_OUTPUT_PREFIX,
    DIARIZATION_SUMMARY_CSV,
    DIARIZATION_ALL_REGULAR_SEGMENTS_CSV,
    DIARIZATION_ALL_SCORED_SEGMENTS_CSV,
    DIARIZATION_ALL_VALID_SEGMENTS_CSV,
    DIARIZATION_ALL_ANCHOR_SEGMENTS_CSV,
)

from src.io_utils import (
    write_csv_atomic,
    read_csv_robust,
    csv_is_usable,
)

from src.storage_io import (
    upload_file,
    download_file_if_exists,
)


# ============================================================
# COLUMNAS DE SALIDA
# ============================================================

DIARIZATION_SEGMENT_COLUMNS = [
    "audio_file",
    "audio_stem",
    "start",
    "end",
    "duration",
    "speaker",
]

SCORED_SEGMENT_COLUMNS = [
    "segment_id_raw",
    "audio_file",
    "audio_stem",
    "start",
    "end",
    "duration",
    "speaker",
    "rms_dbfs",
    "overlap_duration_sec",
    "overlap_ratio",
    "valid_export",
    "valid_anchor",
    "drop_reasons",
    "anchor_reasons",
]

ANCHOR_SEGMENT_COLUMNS = SCORED_SEGMENT_COLUMNS + ["anchor_rank"]

SUMMARY_COLUMNS = [
    "audio_file",
    "audio_stem",
    "sample_rate",
    "duration_sec",
    "diarization_mode",
    "n_regular_segments",
    "n_scored_segments",
    "n_valid_segments",
    "n_anchor_segments",
    "n_speakers",
    "n_overlap_regions",
    "regular_csv_path",
    "raw_csv_path",
    "clean_csv_path",
    "anchors_csv_path",
    "rttm_path",
]


# ============================================================
# CARGA Y PREPARACIÓN DEL AUDIO
# ============================================================


def load_audio_as_mono(audio_path: Path):
    audio, sr = sf.read(audio_path, always_2d=True)
    waveform = torch.from_numpy(audio.T).float()
    waveform_mono = waveform.mean(dim=0, keepdim=True)
    audio_mono = waveform_mono.squeeze(0).numpy()
    duration_sec = len(audio_mono) / sr

    return waveform_mono, audio_mono, sr, duration_sec


def annotation_to_df(annotation, audio_path: Path):
    rows = []

    for turn, _, speaker in annotation.itertracks(yield_label=True):
        rows.append(
            {
                "audio_file": audio_path.name,
                "audio_stem": audio_path.stem,
                "start": float(turn.start),
                "end": float(turn.end),
                "duration": float(turn.end - turn.start),
                "speaker": speaker,
            }
        )

    # Mantiene las columnas aunque no haya segmentos.
    df = pd.DataFrame(rows, columns=DIARIZATION_SEGMENT_COLUMNS)

    if not df.empty:
        df = df.sort_values(["start", "end"]).reset_index(drop=True)

    return df


# ============================================================
# UNIÓN Y SOLAPAMIENTO
# ============================================================


def merge_adjacent_same_speaker(
    df_segments: pd.DataFrame,
    max_gap_sec: float,
):
    if df_segments.empty:
        return df_segments.copy()

    df = df_segments.sort_values(["start", "end"]).reset_index(drop=True)

    merged_rows = []
    current = df.iloc[0].to_dict()

    for _, row in df.iloc[1:].iterrows():
        gap = float(row["start"]) - float(current["end"])
        same_speaker = row["speaker"] == current["speaker"]

        if same_speaker and gap <= max_gap_sec:
            current["end"] = max(
                float(current["end"]),
                float(row["end"]),
            )
            current["duration"] = float(
                current["end"] - current["start"]
            )
        else:
            merged_rows.append(current.copy())
            current = row.to_dict()

    merged_rows.append(current.copy())

    merged_df = pd.DataFrame(merged_rows)

    if not merged_df.empty:
        merged_df = (
            merged_df
            .sort_values(["start", "end"])
            .reset_index(drop=True)
        )

    return merged_df


def compute_overlap_intervals(df_regular: pd.DataFrame):
    if df_regular.empty:
        return []

    events = []

    for _, row in df_regular.iterrows():
        start = float(row["start"])
        end = float(row["end"])

        if end <= start:
            continue

        events.append((start, 1))
        events.append((end, -1))

    # Un final en el mismo instante que otro inicio no cuenta como overlap.
    events.sort(key=lambda x: (x[0], x[1]))

    overlap_intervals = []
    active = 0
    prev_t = None

    for t, delta in events:
        if prev_t is not None and t > prev_t and active > 1:
            overlap_intervals.append((prev_t, t))

        active += delta
        prev_t = t

    merged = []

    for start, end in overlap_intervals:
        if not merged:
            merged.append([start, end])
        elif start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    return [
        (float(start), float(end))
        for start, end in merged
    ]


def interval_overlap_duration(
    start: float,
    end: float,
    intervals,
):
    total = 0.0

    for overlap_start, overlap_end in intervals:
        intersection_start = max(start, overlap_start)
        intersection_end = min(end, overlap_end)

        if intersection_end > intersection_start:
            total += intersection_end - intersection_start

    return total


# ============================================================
# PUNTUACIÓN Y FILTRADO
# ============================================================


def rms_dbfs(x):
    x = np.asarray(x, dtype=np.float32)

    if x.size == 0:
        return -120.0

    rms = np.sqrt(np.mean(np.square(x)))

    if rms <= 1e-10:
        return -120.0

    return float(20.0 * np.log10(rms))


def build_scored_segments(
    df_segments: pd.DataFrame,
    audio_mono,
    sr: int,
    overlap_intervals,
    *,
    min_segment_duration_sec: float,
    min_rms_dbfs: float,
    min_anchor_duration_sec: float,
    max_overlap_ratio_for_anchor: float,
    initial_exclude_sec_for_anchors: float,
):
    if df_segments.empty:
        return pd.DataFrame(columns=SCORED_SEGMENT_COLUMNS)

    rows = []

    for segment_index, (_, row) in enumerate(
        df_segments.iterrows(),
        start=1,
    ):
        start = float(row["start"])
        end = float(row["end"])
        duration = float(row["duration"])

        start_sample = max(0, int(round(start * sr)))
        end_sample = min(
            len(audio_mono),
            int(round(end * sr)),
        )

        if end_sample <= start_sample:
            segment_audio = np.array([], dtype=np.float32)
        else:
            segment_audio = audio_mono[start_sample:end_sample]

        segment_rms_dbfs = rms_dbfs(segment_audio)

        overlap_duration_sec = interval_overlap_duration(
            start,
            end,
            overlap_intervals,
        )

        overlap_ratio = (
            overlap_duration_sec / duration
            if duration > 0
            else 0.0
        )

        export_reasons = []

        if duration < min_segment_duration_sec:
            export_reasons.append("short")

        if segment_rms_dbfs < min_rms_dbfs:
            export_reasons.append("low_energy")

        valid_export = len(export_reasons) == 0

        anchor_reasons = list(export_reasons)

        if duration < min_anchor_duration_sec:
            anchor_reasons.append("short_anchor")

        if overlap_ratio > max_overlap_ratio_for_anchor:
            anchor_reasons.append("overlap")

        if start < initial_exclude_sec_for_anchors:
            anchor_reasons.append("initial_window")

        anchor_reasons = sorted(set(anchor_reasons))
        valid_anchor = len(anchor_reasons) == 0

        output_row = row.to_dict()

        output_row["segment_id_raw"] = int(segment_index)
        output_row["rms_dbfs"] = float(segment_rms_dbfs)
        output_row["overlap_duration_sec"] = float(
            overlap_duration_sec
        )
        output_row["overlap_ratio"] = float(overlap_ratio)
        output_row["valid_export"] = bool(valid_export)
        output_row["valid_anchor"] = bool(valid_anchor)
        output_row["drop_reasons"] = ";".join(export_reasons)
        output_row["anchor_reasons"] = ";".join(anchor_reasons)

        rows.append(output_row)

    df_scored = pd.DataFrame(rows)

    df_scored = (
        df_scored
        .sort_values(["segment_id_raw"])
        .reset_index(drop=True)
    )

    return df_scored


def select_anchor_segments(
    df_scored: pd.DataFrame,
    anchors_per_speaker: int,
):
    if df_scored.empty:
        return pd.DataFrame(columns=ANCHOR_SEGMENT_COLUMNS)

    anchors = df_scored[df_scored["valid_anchor"]].copy()

    if anchors.empty:
        return pd.DataFrame(columns=ANCHOR_SEGMENT_COLUMNS)

    anchors = anchors.sort_values(
        by=[
            "speaker",
            "duration",
            "rms_dbfs",
            "overlap_ratio",
            "start",
        ],
        ascending=[True, False, False, True, True],
    ).copy()

    anchors["anchor_rank"] = (
        anchors
        .groupby("speaker")
        .cumcount()
        + 1
    )

    anchors = anchors[
        anchors["anchor_rank"] <= anchors_per_speaker
    ].reset_index(drop=True)

    return anchors.reindex(columns=ANCHOR_SEGMENT_COLUMNS)


# ============================================================
# DIARIZACIÓN DE UN AUDIO
# ============================================================


def diarize_audio(
    audio_path: Path,
    pipeline,
    *,
    num_speakers: int,
    use_exclusive_diarization: bool,
    max_gap_merge_sec: float,
    min_segment_duration_sec: float,
    min_rms_dbfs: float,
    min_anchor_duration_sec: float,
    max_overlap_ratio_for_anchor: float,
    initial_exclude_sec_for_anchors: float,
    anchors_per_speaker: int,
):
    waveform_mono, audio_mono, sr, duration_sec = (
        load_audio_as_mono(audio_path)
    )

    output = pipeline(
        {
            "waveform": waveform_mono,
            "sample_rate": sr,
        },
        num_speakers=num_speakers,
    )

    diarization_regular = output.speaker_diarization

    diarization_exclusive = getattr(
        output,
        "exclusive_speaker_diarization",
        None,
    )

    # Se utiliza únicamente para calcular el solapamiento.
    df_regular = annotation_to_df(
        diarization_regular,
        audio_path,
    )

    if (
        use_exclusive_diarization
        and diarization_exclusive is not None
    ):
        diarization_used = diarization_exclusive
        diarization_mode = "exclusive"
    else:
        diarization_used = diarization_regular
        diarization_mode = "regular"

    df_used = annotation_to_df(
        diarization_used,
        audio_path,
    )

    df_used_merged = merge_adjacent_same_speaker(
        df_used,
        max_gap_merge_sec,
    )

    overlap_intervals = compute_overlap_intervals(
        df_regular
    )

    df_scored = build_scored_segments(
        df_used_merged,
        audio_mono,
        sr,
        overlap_intervals,
        min_segment_duration_sec=min_segment_duration_sec,
        min_rms_dbfs=min_rms_dbfs,
        min_anchor_duration_sec=min_anchor_duration_sec,
        max_overlap_ratio_for_anchor=(
            max_overlap_ratio_for_anchor
        ),
        initial_exclude_sec_for_anchors=(
            initial_exclude_sec_for_anchors
        ),
    )

    df_valid = (
        df_scored[df_scored["valid_export"]]
        .copy()
        .reset_index(drop=True)
    )

    df_anchors = select_anchor_segments(
        df_scored,
        anchors_per_speaker,
    )

    return {
        "diarization_regular": diarization_regular,
        "diarization_used": diarization_used,
        "diarization_mode": diarization_mode,
        "df_regular": df_regular,
        "df_used": df_used,
        "df_used_merged": df_used_merged,
        "df_scored": df_scored,
        "df_valid": df_valid,
        "df_anchors": df_anchors,
        "overlap_intervals": overlap_intervals,
        "sr": sr,
        "duration_sec": duration_sec,
        "audio_mono": audio_mono,
    }

# ============================================================
# GUARDADO DE RESULTADOS POR AUDIO
# ============================================================

def save_diarization_outputs(
    audio_path: Path,
    result: dict,
    output_dir: Path,
):
    raw_csv_path = (
        output_dir / f"{audio_path.stem}_raw.csv"
    )

    clean_csv_path = (
        output_dir / f"{audio_path.stem}.csv"
    )

    anchors_csv_path = (
        output_dir / f"{audio_path.stem}_anchors.csv"
    )

    rttm_path = (
        output_dir / f"{audio_path.stem}.rttm"
    )

    write_csv_atomic(
        result["df_scored"],
        raw_csv_path,
        columns=SCORED_SEGMENT_COLUMNS,
    )

    write_csv_atomic(
        result["df_valid"],
        clean_csv_path,
        columns=SCORED_SEGMENT_COLUMNS,
    )

    write_csv_atomic(
        result["df_anchors"],
        anchors_csv_path,
        columns=ANCHOR_SEGMENT_COLUMNS,
    )

    temporary_rttm_path = rttm_path.with_suffix(
        rttm_path.suffix + ".tmp"
    )

    with open(
        temporary_rttm_path,
        "w",
        encoding="utf-8",
    ) as file:
        result["diarization_used"].write_rttm(file)

    temporary_rttm_path.replace(rttm_path)

    return (
        raw_csv_path,
        clean_csv_path,
        anchors_csv_path,
        rttm_path,
    )

# ============================================================
# RUTAS Y VALIDEZ DE OUTPUTS POR AUDIO
# ============================================================

def get_audio_output_paths(
    audio_path: Path,
    output_dir: Path = OUTPUT_DIR,
):
    """
    Conserva los nombres históricos de los outputs por audio.

    - *_raw.csv: segmentos puntuados.
    - .csv: segmentos válidos.
    - *_anchors.csv: anchors.
    - *_regular.csv: diarización regular para overlap.
    - .rttm: diarización utilizada.
    """
    stem = audio_path.stem

    return {
        "regular": output_dir / f"{stem}_regular.csv",
        "scored": output_dir / f"{stem}_raw.csv",
        "valid": output_dir / f"{stem}.csv",
        "anchors": output_dir / f"{stem}_anchors.csv",
        "rttm": output_dir / f"{stem}.rttm",
    }

def required_outputs_exist(paths: dict):
    """Comprueba que todos los outputs necesarios sean legibles."""
    csv_checks = {
        "scored": SCORED_SEGMENT_COLUMNS,
        "valid": SCORED_SEGMENT_COLUMNS,
        "anchors": ANCHOR_SEGMENT_COLUMNS,
        "regular": DIARIZATION_SEGMENT_COLUMNS,
    }

    for key, columns in csv_checks.items():
        if not csv_is_usable(
            paths[key],
            required_columns=columns,
        ):
            return False

    if (
        not paths["rttm"].exists()
        or paths["rttm"].stat().st_size == 0
    ):
        return False

    return True


def remove_bad_audio_outputs(paths: dict):
    """Elimina restos de archivos de cero bytes."""
    for local_path in paths.values():
        if local_path.exists():
            try:
                if local_path.stat().st_size == 0:
                    local_path.unlink()
            except Exception:
                pass


# ============================================================
# CHECKPOINTS Y REANUDACIÓN POR AUDIO
# ============================================================


def try_restore_audio_outputs_from_gcs(
    paths: dict,
    gcs_client,
    gcs_prefix: str = GCS_DIARIZATION_OUTPUT_PREFIX,
    output_dir: Path = OUTPUT_DIR,
):
    """Restaura desde GCS los outputs de un audio que falten localmente."""
    restored_any = False

    for local_path in paths.values():
        if not local_path.exists() or local_path.stat().st_size == 0:
            restored = download_file_if_exists(
                local_path,
                gcs_client,
                gcs_prefix=gcs_prefix,
                base_dir=output_dir,
            )
            restored_any = restored or restored_any

    return restored_any


def upload_audio_outputs_to_gcs(
    paths: dict,
    gcs_client,
    gcs_prefix: str = GCS_DIARIZATION_OUTPUT_PREFIX,
    output_dir: Path = OUTPUT_DIR,
):
    """Sube a GCS todos los outputs de un audio."""
    for local_path in paths.values():
        upload_file(
            local_path,
            gcs_client,
            gcs_prefix=gcs_prefix,
            base_dir=output_dir,
        )


def build_summary_row_from_outputs(
    audio_path: Path,
    paths: dict,
    mode: str,
):
    df_regular = read_csv_robust(
        paths["regular"],
        columns=DIARIZATION_SEGMENT_COLUMNS,
    )
    df_scored = read_csv_robust(
        paths["scored"],
        columns=SCORED_SEGMENT_COLUMNS,
    )
    df_valid = read_csv_robust(
        paths["valid"],
        columns=SCORED_SEGMENT_COLUMNS,
    )
    df_anchors = read_csv_robust(
        paths["anchors"],
        columns=ANCHOR_SEGMENT_COLUMNS,
    )

    if not df_regular.empty:
        overlap_intervals = compute_overlap_intervals(df_regular)
        number_overlap_regions = len(overlap_intervals)
    else:
        number_overlap_regions = 0

    duration_sec = np.nan
    if not df_scored.empty and "end" in df_scored.columns:
        duration_sec = float(df_scored["end"].max())

    number_speakers = (
        df_scored["speaker"].nunique()
        if (
            "speaker" in df_scored.columns
            and not df_scored.empty
        )
        else 0
    )

    return {
        "audio_file": audio_path.name,
        "audio_stem": audio_path.stem,
        "sample_rate": np.nan,
        "duration_sec": duration_sec,
        "diarization_mode": mode,
        "n_regular_segments": len(df_regular),
        "n_scored_segments": len(df_scored),
        "n_valid_segments": len(df_valid),
        "n_anchor_segments": len(df_anchors),
        "n_speakers": number_speakers,
        "n_overlap_regions": number_overlap_regions,
        "regular_csv_path": str(paths["regular"]),
        "raw_csv_path": str(paths["scored"]),
        "clean_csv_path": str(paths["valid"]),
        "anchors_csv_path": str(paths["anchors"]),
        "rttm_path": str(paths["rttm"]),
    }


def rebuild_consolidated_outputs(
    wav_files,
    gcs_client,
    *,
    gcs_prefix: str = GCS_DIARIZATION_OUTPUT_PREFIX,
    output_dir: Path = OUTPUT_DIR,
    summary_csv: Path = DIARIZATION_SUMMARY_CSV,
    all_regular_segments_csv: Path = (
        DIARIZATION_ALL_REGULAR_SEGMENTS_CSV
    ),
    all_scored_segments_csv: Path = (
        DIARIZATION_ALL_SCORED_SEGMENTS_CSV
    ),
    all_valid_segments_csv: Path = (
        DIARIZATION_ALL_VALID_SEGMENTS_CSV
    ),
    all_anchor_segments_csv: Path = (
        DIARIZATION_ALL_ANCHOR_SEGMENTS_CSV
    ),
):
    summary_rows = []
    regular_frames = []
    scored_frames = []
    valid_frames = []
    anchor_frames = []

    for audio_path in wav_files:
        paths = get_audio_output_paths(
            audio_path,
            output_dir=output_dir,
        )

        if not required_outputs_exist(paths):
            continue

        summary_rows.append(
            build_summary_row_from_outputs(
                audio_path,
                paths,
                mode="from_checkpoint",
            )
        )

        df_regular = read_csv_robust(
            paths["regular"],
            columns=DIARIZATION_SEGMENT_COLUMNS,
        )
        df_scored = read_csv_robust(
            paths["scored"],
            columns=SCORED_SEGMENT_COLUMNS,
        )
        df_valid = read_csv_robust(
            paths["valid"],
            columns=SCORED_SEGMENT_COLUMNS,
        )
        df_anchors = read_csv_robust(
            paths["anchors"],
            columns=ANCHOR_SEGMENT_COLUMNS,
        )

        if not df_regular.empty:
            regular_frames.append(df_regular)
        if not df_scored.empty:
            scored_frames.append(df_scored)
        if not df_valid.empty:
            valid_frames.append(df_valid)
        if not df_anchors.empty:
            anchor_frames.append(df_anchors)

    df_summary_local = pd.DataFrame(
        summary_rows,
        columns=SUMMARY_COLUMNS,
    )
    df_all_regular_local = (
        pd.concat(regular_frames, ignore_index=True)
        if regular_frames
        else pd.DataFrame(columns=DIARIZATION_SEGMENT_COLUMNS)
    )
    df_all_scored_local = (
        pd.concat(scored_frames, ignore_index=True)
        if scored_frames
        else pd.DataFrame(columns=SCORED_SEGMENT_COLUMNS)
    )
    df_all_valid_local = (
        pd.concat(valid_frames, ignore_index=True)
        if valid_frames
        else pd.DataFrame(columns=SCORED_SEGMENT_COLUMNS)
    )
    df_all_anchor_local = (
        pd.concat(anchor_frames, ignore_index=True)
        if anchor_frames
        else pd.DataFrame(columns=ANCHOR_SEGMENT_COLUMNS)
    )

    write_csv_atomic(
        df_summary_local,
        summary_csv,
        columns=SUMMARY_COLUMNS,
    )
    write_csv_atomic(
        df_all_regular_local,
        all_regular_segments_csv,
        columns=DIARIZATION_SEGMENT_COLUMNS,
    )
    write_csv_atomic(
        df_all_scored_local,
        all_scored_segments_csv,
        columns=SCORED_SEGMENT_COLUMNS,
    )
    write_csv_atomic(
        df_all_valid_local,
        all_valid_segments_csv,
        columns=SCORED_SEGMENT_COLUMNS,
    )
    write_csv_atomic(
        df_all_anchor_local,
        all_anchor_segments_csv,
        columns=ANCHOR_SEGMENT_COLUMNS,
    )

    for path in [
        summary_csv,
        all_regular_segments_csv,
        all_scored_segments_csv,
        all_valid_segments_csv,
        all_anchor_segments_csv,
    ]:
        upload_file(
            path,
            gcs_client,
            gcs_prefix=gcs_prefix,
            base_dir=output_dir,
        )

    return (
        df_summary_local,
        df_all_regular_local,
        df_all_scored_local,
        df_all_valid_local,
        df_all_anchor_local,
    )

