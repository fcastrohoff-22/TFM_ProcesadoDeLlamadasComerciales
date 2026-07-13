"""Consolidación y auditoría de segmentos finales de diarización.

Este módulo da soporte al Notebook 04. Trabaja exclusivamente con los CSV
``*_final_merged.csv`` generados por el Notebook 01: detecta duplicidades
por identidad normalizada, selecciona un archivo canónico por audio,
consolida los segmentos y guarda las mismas salidas históricas.
"""

from pathlib import Path
import shutil

import pandas as pd  # type: ignore

from src.identidad_audio import normalize_consolidation_audio_id
from src.io_utils import csv_is_usable, read_csv_robust, write_csv_atomic


def find_final_merged_files(final_dir: Path):
    """Localiza los CSV individuales ``*_final_merged.csv``."""
    return sorted(final_dir.glob("*_final_merged.csv"))


def build_file_audit(final_dir: Path):
    """Construye la auditoría de archivos disponibles y sus ids normalizados."""
    final_merged_files = find_final_merged_files(final_dir)

    audit_files = pd.DataFrame({
        "source_csv": [path.name for path in final_merged_files],
        "path": [str(path) for path in final_merged_files],
        "mtime": [path.stat().st_mtime for path in final_merged_files],
    })

    if audit_files.empty:
        audit_files = pd.DataFrame(
            columns=[
                "source_csv",
                "path",
                "mtime",
                "audio_id_base",
                "has_temp_prefix",
            ]
        )
        return audit_files

    audit_files["audio_id_base"] = (
        audit_files["source_csv"]
        .apply(normalize_consolidation_audio_id)
    )

    audit_files["has_temp_prefix"] = (
        audit_files["source_csv"]
        .str.startswith(
            ("raw_", "raw_bajas_", "raw_comercial_")
        )
    )

    return audit_files


def build_duplicate_file_summary(audit_files: pd.DataFrame):
    """Resume los ids normalizados asociados a más de un CSV."""
    if audit_files.empty:
        return pd.DataFrame(
            columns=["audio_id_base", "n_csv", "ejemplos"]
        )

    return (
        audit_files
        .groupby("audio_id_base")
        .agg(
            n_csv=("source_csv", "nunique"),
            ejemplos=(
                "source_csv",
                lambda values: " | ".join(
                    sorted(values.astype(str).head(10))
                ),
            ),
        )
        .reset_index()
        .query("n_csv > 1")
        .sort_values("n_csv", ascending=False)
    )


def select_canonical_files(audit_files: pd.DataFrame):
    """
    Selecciona un CSV canónico por audio.

    Prioriza archivos sin prefijo temporal y, en caso de empate,
    conserva el más reciente.
    """
    if audit_files.empty:
        empty = audit_files.copy()
        return empty, empty

    canonical_files = (
        audit_files
        .sort_values(
            ["audio_id_base", "has_temp_prefix", "mtime"],
            ascending=[True, True, False],
        )
        .drop_duplicates("audio_id_base", keep="first")
        .reset_index(drop=True)
    )

    dropped_files = audit_files[
        ~audit_files["source_csv"].isin(
            canonical_files["source_csv"]
        )
    ].copy()

    return canonical_files, dropped_files


def consolidate_canonical_files(
    canonical_files: pd.DataFrame,
):
    """Lee y concatena los CSV canónicos conservando las columnas históricas."""
    if canonical_files.empty:
        raise FileNotFoundError(
            "No se encontraron archivos *_final_merged.csv "
            "para consolidar."
        )

    frames = []

    for file_path in canonical_files["path"]:
        file_path = Path(file_path)
        dataframe = pd.read_csv(file_path)
        dataframe["source_csv"] = file_path.name
        dataframe["audio_id_base"] = (
            dataframe["audio_file"]
            .apply(normalize_consolidation_audio_id)
        )
        frames.append(dataframe)

    df_segments = pd.concat(frames, ignore_index=True)

    if "valid_export" in df_segments.columns:
        df_segments = df_segments[
            df_segments["valid_export"] == True  # noqa: E712
        ].copy()

    df_segments["audio_id_base"] = (
        df_segments["audio_file"]
        .apply(normalize_consolidation_audio_id)
    )

    return (
        df_segments
        .sort_values(["audio_id_base", "start", "end"])
        .reset_index(drop=True)
    )


def build_segment_audit(df_segments: pd.DataFrame):
    """Detecta ids normalizados asociados a múltiples audios o CSV."""
    required_columns = [
        "audio_file",
        "audio_id_base",
        "source_csv",
    ]

    if (
        df_segments.empty
        or not set(required_columns).issubset(
            df_segments.columns
        )
    ):
        return pd.DataFrame(
            columns=[
                "audio_id_base",
                "n_audio_files",
                "n_source_csv",
                "audio_files",
                "source_csvs",
            ]
        )

    return (
        df_segments[required_columns]
        .drop_duplicates()
        .groupby("audio_id_base")
        .agg(
            n_audio_files=("audio_file", "nunique"),
            n_source_csv=("source_csv", "nunique"),
            audio_files=(
                "audio_file",
                lambda values: " | ".join(
                    sorted(values.astype(str).head(10))
                ),
            ),
            source_csvs=(
                "source_csv",
                lambda values: " | ".join(
                    sorted(values.astype(str).head(10))
                ),
            ),
        )
        .reset_index()
        .query("n_audio_files > 1 or n_source_csv > 1")
        .sort_values(
            ["n_source_csv", "n_audio_files"],
            ascending=False,
        )
    )


def consolidation_outputs_exist(
    output_path: Path,
    dedup_output_path: Path,
    audit_path: Path,
    dropped_path: Path,
):
    """Comprueba que las cuatro salidas principales sean legibles."""
    return all(
        csv_is_usable(path)
        for path in [
            output_path,
            dedup_output_path,
            audit_path,
            dropped_path,
        ]
    )


def load_existing_consolidation_outputs(
    output_path: Path,
    audit_path: Path,
    dropped_path: Path,
):
    """Carga outputs existentes y reconstruye solo las vistas de auditoría."""
    df_segments = read_csv_robust(output_path)
    audit_files = read_csv_robust(audit_path)
    dropped_files = read_csv_robust(dropped_path)

    dupes_files = build_duplicate_file_summary(
        audit_files
    )
    canonical_files, _ = select_canonical_files(
        audit_files
    )
    audit_segments = build_segment_audit(
        df_segments
    )

    return {
        "audit_files": audit_files,
        "dupes_files": dupes_files,
        "canonical_files": canonical_files,
        "dropped_files": dropped_files,
        "df_segments": df_segments,
        "audit_segments": audit_segments,
        "reused_existing_outputs": True,
    }


def save_consolidation_outputs(
    df_segments: pd.DataFrame,
    audit_files: pd.DataFrame,
    dropped_files: pd.DataFrame,
    *,
    output_path: Path,
    dedup_output_path: Path,
    audit_path: Path,
    dropped_path: Path,
    backup_path: Path,
):
    """Guarda las mismas salidas y el mismo backup del notebook original."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if output_path.exists():
        shutil.copy2(
            output_path,
            backup_path,
        )

    write_csv_atomic(
        df_segments,
        output_path,
    )
    write_csv_atomic(
        df_segments,
        dedup_output_path,
    )
    write_csv_atomic(
        audit_files,
        audit_path,
    )
    write_csv_atomic(
        dropped_files,
        dropped_path,
    )


def run_consolidation_pipeline(
    *,
    final_dir: Path,
    output_path: Path,
    dedup_output_path: Path,
    audit_path: Path,
    dropped_path: Path,
    backup_path: Path,
    force: bool = False,
):
    """
    Ejecuta o reutiliza la consolidación completa.

    Con ``force=False`` carga las salidas restauradas si las cuatro
    principales existen y son legibles.
    """
    if (
        not force
        and consolidation_outputs_exist(
            output_path,
            dedup_output_path,
            audit_path,
            dropped_path,
        )
    ):
        print(
            "Outputs consolidados encontrados. "
            "Se reutilizan sin reconstruir la consolidación."
        )
        return load_existing_consolidation_outputs(
            output_path,
            audit_path,
            dropped_path,
        )

    audit_files = build_file_audit(
        final_dir
    )
    dupes_files = build_duplicate_file_summary(
        audit_files
    )
    canonical_files, dropped_files = (
        select_canonical_files(
            audit_files
        )
    )
    df_segments = consolidate_canonical_files(
        canonical_files
    )
    audit_segments = build_segment_audit(
        df_segments
    )

    save_consolidation_outputs(
        df_segments,
        audit_files,
        dropped_files,
        output_path=output_path,
        dedup_output_path=dedup_output_path,
        audit_path=audit_path,
        dropped_path=dropped_path,
        backup_path=backup_path,
    )

    return {
        "audit_files": audit_files,
        "dupes_files": dupes_files,
        "canonical_files": canonical_files,
        "dropped_files": dropped_files,
        "df_segments": df_segments,
        "audit_segments": audit_segments,
        "reused_existing_outputs": False,
    }
