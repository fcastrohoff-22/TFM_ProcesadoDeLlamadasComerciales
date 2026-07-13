"""Funciones para reetiquetado de segmentos mediante embeddings de voz."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio

from src.config import (
    OUTPUT_DIR,
    FINAL_RELABEL_DIR,
    EMBEDDING_VECTOR_CSV_DIR,
    EMBEDDING_SAMPLE_RATE,
    RELABEL_MIN_MARGIN,
    SAVE_EMBEDDING_VECTOR_CSVS,
    RELABEL_SUMMARY_CSV,
    ALL_FINAL_SEGMENTS_CSV,
    ALL_FINAL_MERGED_SEGMENTS_CSV,
    ALL_ANCHOR_EMBEDDINGS_CSV,
    ALL_CHANGED_SEGMENTS_CSV,
    ALL_ANCHOR_EMBEDDING_VECTORS_CSV,
    ALL_SEGMENT_EMBEDDING_VECTORS_CSV,
)

from src.diarizacion import (
    SCORED_SEGMENT_COLUMNS,
    ANCHOR_SEGMENT_COLUMNS,
    load_audio_as_mono,
)

from src.io_utils import (
    read_csv_robust,
    csv_is_usable,
    write_csv_atomic,
)

# ============================================================
# NORMALIZACIÓN Y DISTANCIA
# ============================================================

def l2_normalize(vec: np.ndarray):
    vec = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(vec)

    if norm <= 1e-12:
        return vec

    return vec / norm


def cosine_distance(vec_a: np.ndarray, vec_b: np.ndarray):
    vec_a = l2_normalize(vec_a)
    vec_b = l2_normalize(vec_b)

    return float(1.0 - np.dot(vec_a, vec_b))


# ============================================================
# EXTRACCIÓN DE EMBEDDINGS
# ============================================================

def get_segment_waveform_for_embedding(
    audio_mono,
    sr: int,
    start: float,
    end: float,
):
    start_sample = max(0, int(round(start * sr)))
    end_sample = min(len(audio_mono), int(round(end * sr)))

    if end_sample <= start_sample:
        return None

    segment_audio = audio_mono[start_sample:end_sample]

    waveform = (
        torch.from_numpy(segment_audio)
        .float()
        .unsqueeze(0)
        .unsqueeze(0)
    )

    if sr != EMBEDDING_SAMPLE_RATE:
        waveform = torchaudio.functional.resample(
            waveform,
            sr,
            EMBEDDING_SAMPLE_RATE,
        )

    return waveform


def get_segment_embedding(
    audio_mono,
    sr: int,
    start: float,
    end: float,
    embedding_model,
):
    waveform = get_segment_waveform_for_embedding(
        audio_mono,
        sr,
        start,
        end,
    )

    if waveform is None or waveform.shape[-1] == 0:
        return None

    embedding = embedding_model(waveform)
    embedding = np.asarray(embedding).reshape(-1)

    if not np.all(np.isfinite(embedding)):
        return None

    return l2_normalize(embedding)


# ============================================================
# EXPORTACIÓN DE EMBEDDINGS VECTORIALES
# ============================================================

def embedding_dataframe_to_wide(
    df_emb: pd.DataFrame,
    embedding_col: str = "embedding",
):
    """
    Convierte una columna de arrays en columnas
    emb_0000, emb_0001, ...
    """
    if (
        df_emb is None
        or df_emb.empty
        or embedding_col not in df_emb.columns
    ):
        return pd.DataFrame()

    df_valid = df_emb[df_emb[embedding_col].notna()].copy()

    if df_valid.empty:
        return pd.DataFrame()

    emb_matrix = np.vstack(
        df_valid[embedding_col].to_list()
    ).astype(np.float32)

    emb_cols = [
        f"emb_{i:04d}"
        for i in range(emb_matrix.shape[1])
    ]

    df_meta = (
        df_valid
        .drop(columns=[embedding_col])
        .reset_index(drop=True)
    )

    df_vectors = pd.DataFrame(
        emb_matrix,
        columns=emb_cols,
    )

    return pd.concat(
        [df_meta, df_vectors],
        axis=1,
    )


def write_embedding_vectors_csv(
    df_emb: pd.DataFrame,
    output_path: Path,
    embedding_col: str = "embedding",
):
    """
    Guarda los embeddings vectoriales en formato CSV ancho.
    """
    if not SAVE_EMBEDDING_VECTOR_CSVS:
        return pd.DataFrame()

    df_wide = embedding_dataframe_to_wide(
        df_emb,
        embedding_col=embedding_col,
    )

    if df_wide.empty:
        return df_wide

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_csv_atomic(
        df_wide,
        output_path,
    )

    return df_wide


# ============================================================
# CENTROIDES DE ANCHORS
# ============================================================

def build_anchor_centroids(
    df_anchors: pd.DataFrame,
    audio_mono,
    sr: int,
    embedding_model,
):
    if df_anchors.empty:
        return pd.DataFrame(), {}

    anchor_rows = []

    for _, row in df_anchors.iterrows():
        embedding = get_segment_embedding(
            audio_mono=audio_mono,
            sr=sr,
            start=float(row["start"]),
            end=float(row["end"]),
            embedding_model=embedding_model,
        )

        if embedding is None:
            continue

        output_row = row.to_dict()
        output_row["embedding"] = embedding

        anchor_rows.append(output_row)

    if not anchor_rows:
        return pd.DataFrame(), {}

    df_anchor_emb = pd.DataFrame(anchor_rows)
    centroids = {}

    for speaker, group in df_anchor_emb.groupby("speaker"):
        embedding_stack = np.stack(
            group["embedding"].to_list(),
            axis=0,
        )

        centroid = np.mean(
            embedding_stack,
            axis=0,
        )

        centroids[speaker] = l2_normalize(
            centroid
        )

    return df_anchor_emb, centroids


# ============================================================
# REETIQUETADO DE SEGMENTOS
# ============================================================

def relabel_valid_segments_with_centroids(
    df_valid: pd.DataFrame,
    centroids: dict,
    audio_mono,
    sr: int,
    embedding_model,
):
    if df_valid.empty:
        return df_valid.copy(), pd.DataFrame()

    centroid_speakers = sorted(centroids.keys())

    rows = []
    segment_embedding_rows = []

    for _, row in df_valid.iterrows():
        speaker_original = row["speaker"]

        embedding = get_segment_embedding(
            audio_mono=audio_mono,
            sr=sr,
            start=float(row["start"]),
            end=float(row["end"]),
            embedding_model=embedding_model,
        )

        output_row = row.to_dict()
        output_row["speaker_original"] = speaker_original

        if embedding is None:
            output_row["speaker_final"] = speaker_original
            output_row["relabel_source"] = "original_no_embedding"
            output_row["best_distance"] = np.nan
            output_row["second_distance"] = np.nan
            output_row["distance_margin"] = np.nan

            for speaker in centroid_speakers:
                output_row[f"dist_{speaker}"] = np.nan

            rows.append(output_row)
            continue

        distance_map = {
            speaker: cosine_distance(
                embedding,
                centroid,
            )
            for speaker, centroid in centroids.items()
        }

        ordered_distances = sorted(
            distance_map.items(),
            key=lambda item: item[1],
        )

        best_speaker, best_distance = ordered_distances[0]

        if len(ordered_distances) > 1:
            second_distance = ordered_distances[1][1]
            distance_margin = second_distance - best_distance
        else:
            second_distance = np.nan
            distance_margin = np.nan

        if (
            np.isnan(distance_margin)
            or distance_margin >= RELABEL_MIN_MARGIN
        ):
            speaker_final = best_speaker
            relabel_source = "centroid"
        else:
            speaker_final = speaker_original
            relabel_source = "original_low_margin"

        output_row["speaker_final"] = speaker_final
        output_row["relabel_source"] = relabel_source
        output_row["best_distance"] = float(best_distance)

        output_row["second_distance"] = (
            float(second_distance)
            if np.isfinite(second_distance)
            else np.nan
        )

        output_row["distance_margin"] = (
            float(distance_margin)
            if np.isfinite(distance_margin)
            else np.nan
        )

        for speaker in centroid_speakers:
            output_row[f"dist_{speaker}"] = float(
                distance_map[speaker]
            )

        embedding_output_row = output_row.copy()
        embedding_output_row["embedding"] = embedding

        segment_embedding_rows.append(
            embedding_output_row
        )

        rows.append(output_row)

    df_relabel = (
        pd.DataFrame(rows)
        .sort_values(["segment_id_raw"])
        .reset_index(drop=True)
    )

    if segment_embedding_rows:
        df_segment_emb = (
            pd.DataFrame(segment_embedding_rows)
            .sort_values(["segment_id_raw"])
            .reset_index(drop=True)
        )
    else:
        df_segment_emb = pd.DataFrame()

    return df_relabel, df_segment_emb


# ============================================================
# MERGE FINAL
# ============================================================

def merge_adjacent_same_final_speaker(
    df_segments: pd.DataFrame,
    max_gap_sec: float,
):
    if df_segments.empty:
        return df_segments.copy()

    df = (
        df_segments
        .sort_values(["start", "end"])
        .reset_index(drop=True)
    )

    merged_rows = []

    current = df.iloc[0].to_dict()
    current["merged_n_segments"] = 1

    current["segment_ids_raw"] = (
        [int(current["segment_id_raw"])]
        if pd.notna(current.get("segment_id_raw"))
        else []
    )

    for _, row in df.iloc[1:].iterrows():
        gap = float(row["start"]) - float(current["end"])

        same_speaker = (
            row["speaker_final"]
            == current["speaker_final"]
        )

        if same_speaker and gap <= max_gap_sec:
            current["end"] = max(
                float(current["end"]),
                float(row["end"]),
            )

            current["duration"] = float(
                current["end"] - current["start"]
            )

            current["merged_n_segments"] += 1

            if pd.notna(row.get("segment_id_raw")):
                current["segment_ids_raw"].append(
                    int(row["segment_id_raw"])
                )

        else:
            current["segment_ids_raw"] = ",".join(
                str(segment_id)
                for segment_id in current["segment_ids_raw"]
            )

            merged_rows.append(current.copy())

            current = row.to_dict()
            current["merged_n_segments"] = 1

            current["segment_ids_raw"] = (
                [int(current["segment_id_raw"])]
                if pd.notna(current.get("segment_id_raw"))
                else []
            )

    current["segment_ids_raw"] = ",".join(
        str(segment_id)
        for segment_id in current["segment_ids_raw"]
    )

    merged_rows.append(current.copy())

    return (
        pd.DataFrame(merged_rows)
        .sort_values(["start", "end"])
        .reset_index(drop=True)
    )

# ============================================================
# EJECUCIÓN DEL REETIQUETADO POR AUDIO
# ============================================================

def run_relabeling_by_audio(
    wav_files,
    embedding_model,
    max_gap_merge_sec: float,
    output_dir: Path = OUTPUT_DIR,
    final_relabel_dir: Path = FINAL_RELABEL_DIR,
    embedding_vector_csv_dir: Path = EMBEDDING_VECTOR_CSV_DIR,
):
    relabel_summary_rows = []
    all_final_segment_rows = []
    all_final_merged_rows = []
    all_anchor_embedding_rows = []
    all_changed_rows = []
    all_anchor_embedding_vector_rows = []
    all_segment_embedding_vector_rows = []

    for i, audio_path in enumerate(wav_files, start=1):
        print(
            f"[{i}/{len(wav_files)}] "
            f"Relabeling con centroides: {audio_path.name}"
        )

        valid_csv_path = output_dir / f"{audio_path.stem}.csv"
        anchors_csv_path = output_dir / f"{audio_path.stem}_anchors.csv"

        if not valid_csv_path.exists():
            print("   -> no existe CSV válido, se omite")
            continue

        if not anchors_csv_path.exists():
            print("   -> no existe CSV de anchors, se omite")
            continue

        df_valid = read_csv_robust(
            valid_csv_path,
            columns=SCORED_SEGMENT_COLUMNS,
        )

        df_anchors = read_csv_robust(
            anchors_csv_path,
            columns=ANCHOR_SEGMENT_COLUMNS,
        )

        _, audio_mono, sr, duration_sec = load_audio_as_mono(audio_path)

        # 1. Embeddings y centroides de anchors
        df_anchor_emb, centroids = build_anchor_centroids(
            df_anchors=df_anchors,
            audio_mono=audio_mono,
            sr=sr,
            embedding_model=embedding_model,
        )

        n_anchor_embeddings = len(df_anchor_emb)
        n_anchor_speakers = len(centroids)

        if not df_anchor_emb.empty:
            tmp_anchor = df_anchor_emb.drop(columns=["embedding"]).copy()
            tmp_anchor["audio_file"] = audio_path.name
            all_anchor_embedding_rows.append(tmp_anchor)

            df_anchor_emb_for_vectors = df_anchor_emb.copy()
            df_anchor_emb_for_vectors["audio_file"] = audio_path.name

            anchor_embeddings_vectors_csv = (
                embedding_vector_csv_dir
                / f"{audio_path.stem}_anchor_embeddings_vectors.csv"
            )

            df_anchor_vectors = write_embedding_vectors_csv(
                df_anchor_emb_for_vectors,
                anchor_embeddings_vectors_csv,
            )

            if not df_anchor_vectors.empty:
                all_anchor_embedding_vector_rows.append(
                    df_anchor_vectors
                )

        # Se necesitan dos centroides para comparar speakers.
        if n_anchor_speakers < 2:
            relabel_summary_rows.append({
                "audio_file": audio_path.name,
                "duration_sec": duration_sec,
                "n_valid_in": len(df_valid),
                "n_anchor_in": len(df_anchors),
                "n_anchor_embeddings": n_anchor_embeddings,
                "n_anchor_speakers": n_anchor_speakers,
                "n_changed_segments": 0,
                "n_final_segments": len(df_valid),
                "n_final_merged_segments": np.nan,
                "mean_distance_margin": np.nan,
                "status": "skipped_not_enough_anchor_speakers",
            })

            print(
                "   -> omitido: no hay anchors suficientes "
                "para los dos speakers"
            )
            continue

        # 2. Reetiquetado de segmentos válidos
        df_final_segments, df_segment_emb = (
            relabel_valid_segments_with_centroids(
                df_valid=df_valid,
                centroids=centroids,
                audio_mono=audio_mono,
                sr=sr,
                embedding_model=embedding_model,
            )
        )

        if not df_segment_emb.empty:
            df_segment_emb_for_vectors = df_segment_emb.copy()
            df_segment_emb_for_vectors["audio_file"] = audio_path.name

            segment_embeddings_vectors_csv = (
                embedding_vector_csv_dir
                / f"{audio_path.stem}_segment_embeddings_vectors.csv"
            )

            df_segment_vectors = write_embedding_vectors_csv(
                df_segment_emb_for_vectors,
                segment_embeddings_vectors_csv,
            )

            if not df_segment_vectors.empty:
                all_segment_embedding_vector_rows.append(
                    df_segment_vectors
                )

        # 3. Merge final con speaker_final
        df_final_merged = merge_adjacent_same_final_speaker(
            df_segments=df_final_segments,
            max_gap_sec=max_gap_merge_sec,
        )

        # 4. Segmentos que cambiaron de speaker
        df_changed = df_final_segments[
            df_final_segments["speaker_final"]
            != df_final_segments["speaker_original"]
        ].copy()

        # 5. Salidas por audio
        final_segments_csv = (
            final_relabel_dir
            / f"{audio_path.stem}_final_segments.csv"
        )

        final_merged_csv = (
            final_relabel_dir
            / f"{audio_path.stem}_final_merged.csv"
        )

        anchor_embeddings_csv = (
            final_relabel_dir
            / f"{audio_path.stem}_anchor_embeddings.csv"
        )

        changed_csv = (
            final_relabel_dir
            / f"{audio_path.stem}_changed_segments.csv"
        )

        df_final_segments.to_csv(final_segments_csv, index=False)
        df_final_merged.to_csv(final_merged_csv, index=False)
        df_changed.to_csv(changed_csv, index=False)

        if not df_anchor_emb.empty:
            (
                df_anchor_emb
                .drop(columns=["embedding"])
                .to_csv(anchor_embeddings_csv, index=False)
            )

        # 6. Métricas de resumen
        n_changed_segments = int(len(df_changed))

        mean_distance_margin = (
            float(df_final_segments["distance_margin"].dropna().mean())
            if df_final_segments["distance_margin"].notna().any()
            else np.nan
        )

        relabel_summary_rows.append({
            "audio_file": audio_path.name,
            "duration_sec": duration_sec,
            "n_valid_in": len(df_valid),
            "n_anchor_in": len(df_anchors),
            "n_anchor_embeddings": n_anchor_embeddings,
            "n_anchor_speakers": n_anchor_speakers,
            "n_changed_segments": n_changed_segments,
            "n_final_segments": len(df_final_segments),
            "n_final_merged_segments": len(df_final_merged),
            "mean_distance_margin": mean_distance_margin,
            "final_segments_csv": str(final_segments_csv),
            "final_merged_csv": str(final_merged_csv),
            "anchor_embeddings_csv": str(anchor_embeddings_csv),
            "changed_csv": str(changed_csv),
            "status": "ok",
        })

        all_final_segment_rows.append(
            df_final_segments.assign(audio_file=audio_path.name)
        )

        all_final_merged_rows.append(
            df_final_merged.assign(audio_file=audio_path.name)
        )

        all_changed_rows.append(
            df_changed.assign(audio_file=audio_path.name)
        )

        print(f"   -> segments changed: {n_changed_segments}")
        print(f"   -> final merged segments: {len(df_final_merged)}")

    return (
        relabel_summary_rows,
        all_final_segment_rows,
        all_final_merged_rows,
        all_anchor_embedding_rows,
        all_changed_rows,
        all_anchor_embedding_vector_rows,
        all_segment_embedding_vector_rows,
    )

# ============================================================
# CONSOLIDACIÓN Y GUARDADO FINAL
# ============================================================

def consolidate_relabeling_outputs(
    relabel_summary_rows,
    all_final_segment_rows,
    all_final_merged_rows,
    all_anchor_embedding_rows,
    all_changed_rows,
    all_anchor_embedding_vector_rows,
    all_segment_embedding_vector_rows,
):
    df_relabel_summary = pd.DataFrame(relabel_summary_rows)

    df_all_final_segments = (
        pd.concat(all_final_segment_rows, ignore_index=True)
        if all_final_segment_rows
        else pd.DataFrame()
    )

    df_all_final_merged = (
        pd.concat(all_final_merged_rows, ignore_index=True)
        if all_final_merged_rows
        else pd.DataFrame()
    )

    df_all_anchor_embeddings = (
        pd.concat(all_anchor_embedding_rows, ignore_index=True)
        if all_anchor_embedding_rows
        else pd.DataFrame()
    )

    df_all_changed_segments = (
        pd.concat(all_changed_rows, ignore_index=True)
        if all_changed_rows
        else pd.DataFrame()
    )

    df_all_anchor_embedding_vectors = (
        pd.concat(all_anchor_embedding_vector_rows, ignore_index=True)
        if all_anchor_embedding_vector_rows
        else pd.DataFrame()
    )

    df_all_segment_embedding_vectors = (
        pd.concat(all_segment_embedding_vector_rows, ignore_index=True)
        if all_segment_embedding_vector_rows
        else pd.DataFrame()
    )

    write_csv_atomic(df_relabel_summary, RELABEL_SUMMARY_CSV)
    write_csv_atomic(df_all_final_segments, ALL_FINAL_SEGMENTS_CSV)
    write_csv_atomic(df_all_final_merged, ALL_FINAL_MERGED_SEGMENTS_CSV)
    write_csv_atomic(df_all_anchor_embeddings, ALL_ANCHOR_EMBEDDINGS_CSV)
    write_csv_atomic(df_all_changed_segments, ALL_CHANGED_SEGMENTS_CSV)

    if SAVE_EMBEDDING_VECTOR_CSVS:
        if not df_all_anchor_embedding_vectors.empty:
            write_csv_atomic(
                df_all_anchor_embedding_vectors,
                ALL_ANCHOR_EMBEDDING_VECTORS_CSV,
            )

        if not df_all_segment_embedding_vectors.empty:
            write_csv_atomic(
                df_all_segment_embedding_vectors,
                ALL_SEGMENT_EMBEDDING_VECTORS_CSV,
            )

    print("\nRelabeling completado")
    print("Resumen:", RELABEL_SUMMARY_CSV)
    print("Final segments:", ALL_FINAL_SEGMENTS_CSV)
    print("Final merged:", ALL_FINAL_MERGED_SEGMENTS_CSV)
    print("Anchor embeddings:", ALL_ANCHOR_EMBEDDINGS_CSV)
    print("Changed:", ALL_CHANGED_SEGMENTS_CSV)

    if SAVE_EMBEDDING_VECTOR_CSVS:
        print(
            "Anchor embedding vectors CSV:",
            ALL_ANCHOR_EMBEDDING_VECTORS_CSV,
        )
        print(
            "Segment embedding vectors CSV:",
            ALL_SEGMENT_EMBEDDING_VECTORS_CSV,
        )

    return (
        df_relabel_summary,
        df_all_final_segments,
        df_all_final_merged,
        df_all_anchor_embeddings,
        df_all_changed_segments,
        df_all_anchor_embedding_vectors,
        df_all_segment_embedding_vectors,
    )

# ============================================================
# PIPELINE COMPLETO Y REUTILIZACIÓN DE OUTPUTS EXISTENTES
# ============================================================

def load_existing_relabeling_outputs():
    """
    Carga los CSV consolidados existentes.

    Devuelve None si faltan los outputs principales o no son legibles.
    """
    required_outputs = [
        RELABEL_SUMMARY_CSV,
        ALL_FINAL_SEGMENTS_CSV,
        ALL_FINAL_MERGED_SEGMENTS_CSV,
    ]

    if not all(csv_is_usable(path) for path in required_outputs):
        return None

    return {
        "df_relabel_summary": read_csv_robust(RELABEL_SUMMARY_CSV),
        "df_all_final_segments": read_csv_robust(ALL_FINAL_SEGMENTS_CSV),
        "df_all_final_merged": read_csv_robust(
            ALL_FINAL_MERGED_SEGMENTS_CSV
        ),
        "df_all_anchor_embeddings": read_csv_robust(
            ALL_ANCHOR_EMBEDDINGS_CSV
        ),
        "df_all_changed_segments": read_csv_robust(
            ALL_CHANGED_SEGMENTS_CSV
        ),
        "df_all_anchor_embedding_vectors": read_csv_robust(
            ALL_ANCHOR_EMBEDDING_VECTORS_CSV
        ),
        "df_all_segment_embedding_vectors": read_csv_robust(
            ALL_SEGMENT_EMBEDDING_VECTORS_CSV
        ),
    }


def run_relabeling_pipeline(
    wav_files,
    embedding_model,
    max_gap_merge_sec: float,
    force_relabel: bool = False,
):
    """
    Ejecuta el reetiquetado completo y consolida sus resultados.

    Si force_relabel=False y los consolidados principales ya existen,
    los carga sin volver a procesar los audios.
    """
    if not force_relabel:
        existing_outputs = load_existing_relabeling_outputs()

        if existing_outputs is not None:
            print(
                "Outputs consolidados de reetiquetado encontrados. "
                "Se reutilizan sin volver a procesar los audios."
            )
            return existing_outputs

    print("Ejecutando reetiquetado por embeddings...")

    relabeling_rows = run_relabeling_by_audio(
        wav_files=wav_files,
        embedding_model=embedding_model,
        max_gap_merge_sec=max_gap_merge_sec,
    )

    consolidated_outputs = consolidate_relabeling_outputs(
        *relabeling_rows
    )

    (
        df_relabel_summary,
        df_all_final_segments,
        df_all_final_merged,
        df_all_anchor_embeddings,
        df_all_changed_segments,
        df_all_anchor_embedding_vectors,
        df_all_segment_embedding_vectors,
    ) = consolidated_outputs

    return {
        "df_relabel_summary": df_relabel_summary,
        "df_all_final_segments": df_all_final_segments,
        "df_all_final_merged": df_all_final_merged,
        "df_all_anchor_embeddings": df_all_anchor_embeddings,
        "df_all_changed_segments": df_all_changed_segments,
        "df_all_anchor_embedding_vectors": (
            df_all_anchor_embedding_vectors
        ),
        "df_all_segment_embedding_vectors": (
            df_all_segment_embedding_vectors
        ),
    }

# ============================================================
# TABLA MAESTRA DE VALIDACIÓN POR AUDIO
# ============================================================

def _ensure_segment_id_raw_in_raw(df_raw: pd.DataFrame):
    df_raw = df_raw.copy()

    if "segment_id_raw" in df_raw.columns:
        return df_raw

    df_raw["segment_id_raw"] = np.arange(1, len(df_raw) + 1)
    return df_raw


def _attach_segment_id_from_raw(
    df_raw: pd.DataFrame,
    df_other: pd.DataFrame,
):
    df_raw_tmp = df_raw.copy()
    df_other_tmp = df_other.copy()

    if "segment_id_raw" in df_other_tmp.columns:
        return df_other_tmp

    for col in ["start", "end"]:
        if col in df_raw_tmp.columns:
            df_raw_tmp[col] = df_raw_tmp[col].round(3)

        if col in df_other_tmp.columns:
            df_other_tmp[col] = df_other_tmp[col].round(3)

    if (
        "duration" in df_raw_tmp.columns
        and "duration" in df_other_tmp.columns
    ):
        df_raw_tmp["duration"] = df_raw_tmp["duration"].round(3)
        df_other_tmp["duration"] = df_other_tmp["duration"].round(3)
        merge_keys = ["start", "end", "duration", "speaker"]
    else:
        merge_keys = ["start", "end", "speaker"]

    df_raw_ids = (
        df_raw_tmp[merge_keys + ["segment_id_raw"]]
        .drop_duplicates()
    )

    return df_other_tmp.merge(
        df_raw_ids,
        on=merge_keys,
        how="left",
    )


def validation_input_paths(
    audio_name: str,
    output_dir: Path = OUTPUT_DIR,
    final_relabel_dir: Path = FINAL_RELABEL_DIR,
):
    """
    Devuelve las rutas de los CSV necesarios para la tabla de validación.

    Es la única fuente de verdad de qué archivos se requieren, para que
    tanto ``validation_inputs_exist`` como ``build_validation_table``
    (y el notebook) trabajen sobre la misma lista.
    """
    audio_name = str(audio_name).strip()

    if audio_name.lower().endswith(".wav"):
        audio_stem = Path(audio_name).stem
        audio_file = audio_name
    else:
        audio_stem = audio_name
        audio_file = f"{audio_stem}.wav"

    return {
        "audio_stem": audio_stem,
        "audio_file": audio_file,
        "raw": output_dir / f"{audio_stem}_raw.csv",
        "valid": output_dir / f"{audio_stem}.csv",
        "anchors": output_dir / f"{audio_stem}_anchors.csv",
        "final_segments": (
            final_relabel_dir / f"{audio_stem}_final_segments.csv"
        ),
        "final_merged": (
            final_relabel_dir / f"{audio_stem}_final_merged.csv"
        ),
        "changed": (
            final_relabel_dir / f"{audio_stem}_changed_segments.csv"
        ),
    }


def validation_inputs_exist(
    audio_name: str,
    output_dir: Path = OUTPUT_DIR,
    final_relabel_dir: Path = FINAL_RELABEL_DIR,
):
    """Indica si existen todos los CSV necesarios para la tabla maestra."""
    paths = validation_input_paths(
        audio_name,
        output_dir=output_dir,
        final_relabel_dir=final_relabel_dir,
    )

    csv_keys = [
        "raw",
        "valid",
        "anchors",
        "final_segments",
        "final_merged",
        "changed",
    ]

    return all(paths[key].exists() for key in csv_keys)


def build_validation_table(
    audio_name: str,
    output_dir: Path = OUTPUT_DIR,
    final_relabel_dir: Path = FINAL_RELABEL_DIR,
):
    """
    Construye la tabla maestra de validación de un audio
    usando exclusivamente los CSV ya existentes.
    """
    input_paths = validation_input_paths(
        audio_name,
        output_dir=output_dir,
        final_relabel_dir=final_relabel_dir,
    )

    audio_stem = input_paths["audio_stem"]
    audio_file = input_paths["audio_file"]

    raw_csv_path = input_paths["raw"]
    valid_csv_path = input_paths["valid"]
    anchors_csv_path = input_paths["anchors"]
    final_segments_csv_path = input_paths["final_segments"]
    final_merged_csv_path = input_paths["final_merged"]
    changed_csv_path = input_paths["changed"]

    required_paths = [
        raw_csv_path,
        valid_csv_path,
        anchors_csv_path,
        final_segments_csv_path,
        final_merged_csv_path,
        changed_csv_path,
    ]

    missing = [
        str(path)
        for path in required_paths
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Faltan archivos para construir la tabla maestra:\n"
            + "\n".join(missing)
        )

    df_raw = read_csv_robust(
        raw_csv_path,
        columns=SCORED_SEGMENT_COLUMNS,
    )
    df_valid = read_csv_robust(
        valid_csv_path,
        columns=SCORED_SEGMENT_COLUMNS,
    )
    df_anchors = read_csv_robust(
        anchors_csv_path,
        columns=ANCHOR_SEGMENT_COLUMNS,
    )
    df_final_segments = read_csv_robust(
        final_segments_csv_path
    )
    df_final_merged = read_csv_robust(
        final_merged_csv_path
    )
    df_changed = read_csv_robust(
        changed_csv_path
    )

    df_raw = _ensure_segment_id_raw_in_raw(df_raw)
    df_valid = _attach_segment_id_from_raw(df_raw, df_valid)
    df_anchors = _attach_segment_id_from_raw(df_raw, df_anchors)

    df = df_raw.copy()

    rename_map = {
        "speaker": "speaker_raw",
        "start": "start_raw",
        "end": "end_raw",
        "duration": "duration_raw",
        "rms_dbfs": "rms_dbfs_raw",
        "overlap_ratio": "overlap_ratio_raw",
        "valid_export": "valid_export_raw",
        "valid_anchor": "valid_anchor_raw",
        "drop_reasons": "drop_reasons_raw",
        "anchor_reasons": "anchor_reasons_raw",
    }

    df = df.rename(
        columns={
            old: new
            for old, new in rename_map.items()
            if old in df.columns
        }
    )

    if "audio_file" not in df.columns:
        df["audio_file"] = audio_file

    # Segmentos válidos
    df_valid_small = (
        df_valid[["segment_id_raw"]]
        .drop_duplicates()
        .copy()
    )
    df_valid_small["is_valid_segment"] = True

    df = df.merge(
        df_valid_small,
        on="segment_id_raw",
        how="left",
    )
    df["is_valid_segment"] = (
        df["is_valid_segment"].fillna(False)
    )

    # Anchors
    keep_anchor_cols = ["segment_id_raw"]

    for col in ["anchor_rank", "anchor_score", "speaker"]:
        if col in df_anchors.columns:
            keep_anchor_cols.append(col)

    df_anchor_small = (
        df_anchors[keep_anchor_cols]
        .drop_duplicates(subset=["segment_id_raw"])
        .copy()
    )
    df_anchor_small["is_anchor"] = True

    if "speaker" in df_anchor_small.columns:
        df_anchor_small = df_anchor_small.rename(
            columns={"speaker": "speaker_anchor"}
        )

    df = df.merge(
        df_anchor_small,
        on="segment_id_raw",
        how="left",
    )
    df["is_anchor"] = df["is_anchor"].fillna(False)

    # Resultado del reetiquetado
    keep_final_cols = ["segment_id_raw"]

    for col in [
        "speaker_original",
        "speaker_final",
        "relabel_source",
        "best_distance",
        "second_distance",
        "distance_margin",
    ]:
        if col in df_final_segments.columns:
            keep_final_cols.append(col)

    df_final_small = (
        df_final_segments[keep_final_cols]
        .drop_duplicates(subset=["segment_id_raw"])
        .copy()
    )

    df = df.merge(
        df_final_small,
        on="segment_id_raw",
        how="left",
    )

    df["was_reclassified"] = (
        df["speaker_original"].notna()
        & df["speaker_final"].notna()
        & (df["speaker_original"] != df["speaker_final"])
    )

    # Segmentos incluidos en changed
    df_changed_small = (
        df_changed[["segment_id_raw"]]
        .drop_duplicates()
        .copy()
    )
    df_changed_small["is_in_changed_sheet"] = True

    df = df.merge(
        df_changed_small,
        on="segment_id_raw",
        how="left",
    )
    df["is_in_changed_sheet"] = (
        df["is_in_changed_sheet"].fillna(False)
    )

    # Mapeo al grupo final fusionado
    merge_map_rows = []

    df_merged_tmp = (
        df_final_merged
        .copy()
        .reset_index(drop=True)
    )
    df_merged_tmp["merge_group_id"] = (
        df_merged_tmp.index + 1
    )

    for _, row in df_merged_tmp.iterrows():
        ids_str = (
            str(row["segment_ids_raw"])
            if "segment_ids_raw" in row
            else ""
        )

        if not ids_str or ids_str == "nan":
            continue

        ids_list = [
            int(value.strip())
            for value in ids_str.split(",")
            if value.strip()
        ]

        for raw_id in ids_list:
            merge_map_rows.append({
                "segment_id_raw": raw_id,
                "merge_group_id": row["merge_group_id"],
                "merge_ids_raw": ids_str,
                "merge_n_segments": (
                    row["merged_n_segments"]
                    if "merged_n_segments" in row
                    else np.nan
                ),
                "merge_start": (
                    row["start"]
                    if "start" in row
                    else np.nan
                ),
                "merge_end": (
                    row["end"]
                    if "end" in row
                    else np.nan
                ),
                "merge_duration": (
                    row["duration"]
                    if "duration" in row
                    else np.nan
                ),
                "merge_speaker_final": (
                    row["speaker_final"]
                    if "speaker_final" in row
                    else np.nan
                ),
            })

    df_merge_map = pd.DataFrame(merge_map_rows)

    if not df_merge_map.empty:
        df = df.merge(
            df_merge_map,
            on="segment_id_raw",
            how="left",
        )
    else:
        df["merge_group_id"] = np.nan
        df["merge_ids_raw"] = np.nan
        df["merge_n_segments"] = np.nan
        df["merge_start"] = np.nan
        df["merge_end"] = np.nan
        df["merge_duration"] = np.nan
        df["merge_speaker_final"] = np.nan

    df["is_in_final_merge"] = df["merge_group_id"].notna()

    return (
        df
        .sort_values("segment_id_raw")
        .reset_index(drop=True)
    )