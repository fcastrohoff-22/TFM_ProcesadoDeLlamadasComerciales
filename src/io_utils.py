"""Utilidades genéricas de lectura y escritura de CSV.

Este módulo no depende de Google Cloud Storage ni de ninguna fase
concreta del pipeline. Reúne las utilidades de archivo que son
reutilizables por cualquier notebook (diarización, transcripción,
metadata, etc.).
"""

from pathlib import Path

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from pandas.errors import EmptyDataError  # type: ignore


def write_csv_atomic(
    df: pd.DataFrame,
    path: Path,
    columns=None,
):
    """
    Escribe un CSV de forma segura.

    Mantiene el encabezado aunque no haya filas y escribe primero
    en un archivo temporal antes de reemplazar el archivo final.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    df_to_write = (
        df.copy()
        if df is not None
        else pd.DataFrame()
    )

    if columns is not None:
        for column in columns:
            if column not in df_to_write.columns:
                df_to_write[column] = pd.Series(
                    dtype="object"
                )

        extra_columns = [
            column
            for column in df_to_write.columns
            if column not in columns
        ]

        df_to_write = df_to_write[
            columns + extra_columns
        ]

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    df_to_write.to_csv(
        temporary_path,
        index=False,
    )

    temporary_path.replace(path)

    return path


def read_csv_robust(
    path: Path,
    columns=None,
):
    """
    Lee un CSV de forma tolerante.

    Si no existe, tiene cero bytes o no puede leerse, devuelve
    un DataFrame vacío con las columnas esperadas.
    """
    if columns is None:
        columns = []

    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=columns)

    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame(columns=columns)
    except Exception as error:
        print(
            f"Advertencia: no se pudo leer "
            f"{path.name}: {error}"
        )
        return pd.DataFrame(columns=columns)


def csv_is_usable(
    path: Path,
    required_columns=None,
):
    """
    Un CSV con cero filas es válido si contiene encabezado.
    Un CSV de cero bytes no se considera válido.
    """
    if required_columns is None:
        required_columns = []

    if not path.exists() or path.stat().st_size == 0:
        return False

    try:
        dataframe_header = pd.read_csv(path, nrows=0)
    except Exception:
        return False

    return all(
        column in dataframe_header.columns
        for column in required_columns
    )


def safe_rate(numerator, denominator):
    """División protegida: devuelve NaN si el denominador es cero o falsy."""
    return numerator / denominator if denominator else np.nan


def write_text_atomic(text, path: Path, encoding: str = "utf-8"):
    """Escribe texto de forma atómica."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(str(text), encoding=encoding)
    temporary_path.replace(path)
    return path


def write_json_atomic(data, path: Path):
    """Escribe JSON de forma atómica y legible."""
    import json

    return write_text_atomic(
        json.dumps(data, ensure_ascii=False, indent=2, default=str),
        path,
    )
