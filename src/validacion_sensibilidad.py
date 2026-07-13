"""Funciones de análisis de sensibilidad del margen de reetiquetado.

Este módulo da soporte al Notebook 02. No recalcula diarización ni
embeddings: opera sobre los CSV ``*_final_segments.csv`` que ya generó
el Notebook 01, reutilizando las columnas de distancia a centroides
(``dist_SPEAKER_*``) para simular distintos valores de margen.
"""

from pathlib import Path

import numpy as np  # type: ignore
import pandas as pd  # type: ignore


def find_final_segment_files(final_relabel_dir: Path):
    """
    Busca los CSV por audio generados por el Notebook 01.

    Excluye el archivo consolidado global para no duplicar segmentos.
    """
    return sorted(
        path
        for path in final_relabel_dir.glob("*_final_segments.csv")
        if path.name != "all_final_segments.csv"
    )


def infer_best_speaker_from_distances(df: pd.DataFrame):
    """
    Recalcula el speaker más cercano a partir de columnas dist_SPEAKER_*.

    Estas distancias ya fueron calculadas por el Notebook 01 con
    embeddings; aquí solo se derivan el mejor speaker y el margen.
    """
    df = df.copy()

    dist_cols = [
        column
        for column in df.columns
        if column.startswith("dist_")
    ]

    if len(dist_cols) < 2:
        raise ValueError(
            "No hay suficientes columnas dist_SPEAKER_* "
            "para recalcular márgenes."
        )

    dist_values = df[dist_cols].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    best_col = dist_values.idxmin(axis=1)
    best_distance = dist_values.min(axis=1)

    second_distance = dist_values.apply(
        lambda row: (
            row.nsmallest(2).iloc[1]
            if row.notna().sum() >= 2
            else np.nan
        ),
        axis=1,
    )

    df["best_speaker_recomputed"] = best_col.str.replace(
        "dist_",
        "",
        regex=False,
    )
    df["best_distance_recomputed"] = best_distance
    df["second_distance_recomputed"] = second_distance
    df["distance_margin_recomputed"] = (
        second_distance - best_distance
    )

    return df


def apply_relabel_margin(df: pd.DataFrame, margin: float):
    """
    Aplica un margen de reetiquetado sin recalcular embeddings.

    Si la ventaja del mejor speaker supera el margen, usa el mejor
    speaker; en caso contrario conserva el speaker original.
    """
    df = df.copy()

    if "speaker_original" in df.columns:
        original_col = "speaker_original"
    elif "speaker" in df.columns:
        original_col = "speaker"
    else:
        raise ValueError(
            "No existe speaker_original ni speaker en el CSV."
        )

    valid_margin = df["distance_margin_recomputed"].notna()
    enough_margin = df["distance_margin_recomputed"] >= margin

    df["tested_relabel_margin"] = margin
    df["speaker_final_margin"] = np.where(
        valid_margin & enough_margin,
        df["best_speaker_recomputed"],
        df[original_col],
    )

    df["was_reclassified_margin"] = (
        df["speaker_final_margin"] != df[original_col]
    )

    df["relabel_source_margin"] = np.where(
        valid_margin & enough_margin,
        "centroid_margin_ok",
        "original_low_margin_or_missing",
    )

    df["original_speaker_col_used"] = original_col

    return df


def count_merged_segments(
    df: pd.DataFrame,
    speaker_col: str = "speaker_final_margin",
    max_gap_sec: float = 0.50,
):
    """
    Cuenta cuántos segmentos quedarían tras unir segmentos consecutivos
    del mismo speaker final separados por una pausa corta.

    No guarda los segmentos unidos; solo los cuenta.
    """
    if df.empty:
        return 0

    required_cols = ["start", "end", speaker_col]
    missing = [
        column
        for column in required_cols
        if column not in df.columns
    ]

    if missing:
        return np.nan

    df = df.sort_values(["start", "end"]).reset_index(drop=True)

    n_merged = 0
    current_end = None
    current_speaker = None

    for _, row in df.iterrows():
        start = float(row["start"])
        end = float(row["end"])
        speaker = row[speaker_col]

        if current_end is None:
            n_merged = 1
            current_end = end
            current_speaker = speaker
            continue

        gap = start - current_end

        if speaker == current_speaker and gap <= max_gap_sec:
            current_end = max(current_end, end)
        else:
            n_merged += 1
            current_end = end
            current_speaker = speaker

    return n_merged
