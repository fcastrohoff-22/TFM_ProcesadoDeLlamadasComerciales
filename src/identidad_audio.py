"""Normalización y parseo de identidad a partir de nombres de audio.

Distintas fases del pipeline reciben nombres de archivo con sufijos y
prefijos diferentes (``raw_915..._clean.wav``, ``915..._final_merged.csv``,
etc.). Este módulo reúne las utilidades que convierten esos nombres en un
identificador base común.

Se conservan como funciones separadas —en lugar de forzar una sola— porque
cada fase del pipeline original aplicaba una normalización ligeramente
distinta y unificarlas a ciegas podría alterar resultados ya validados.
"""

import re
from pathlib import Path

import numpy as np  # type: ignore
import pandas as pd  # type: ignore


def parse_audio_identity(audio_file_or_stem):
    """
    Extrae (source_dataset, audio_id) de un nombre de audio.

    Reconoce formas como ``raw_915..._clean`` o ``raw_bajas_915...``.
    Usada en la fase 03 (evaluación interna de diarización).
    """
    name = str(audio_file_or_stem)
    stem = Path(name).stem

    m = re.match(r"^(raw_bajas|raw)_([0-9]+)(?:_clean)?$", stem)
    if m:
        return m.group(1), m.group(2)

    m = re.match(r"^([0-9]+)(?:_clean)?$", stem)
    if m:
        return np.nan, m.group(1)

    m = re.search(r"([0-9]{10,})", stem)
    if m:
        source = (
            "raw_bajas"
            if "raw_bajas" in stem
            else ("raw" if "raw_" in stem else np.nan)
        )
        return source, m.group(1)

    return np.nan, np.nan


def add_audio_keys(
    df,
    audio_col_candidates=(
        "audio_file",
        "audio_stem",
        "audio_base",
        "file_stem",
        "filename",
    ),
):
    """
    Añade columnas source_dataset_parsed / audio_id_parsed a un DataFrame,
    detectando automáticamente la columna de nombre de audio.
    """
    df = df.copy()
    audio_col = None

    for col in audio_col_candidates:
        if col in df.columns:
            audio_col = col
            break

    if audio_col is None:
        return df

    parsed = df[audio_col].apply(parse_audio_identity)
    df["source_dataset_parsed"] = parsed.apply(lambda x: x[0])
    df["audio_id_parsed"] = (
        parsed.apply(lambda x: x[1]).astype("string")
    )

    return df


def normalize_audio_id(x):
    """
    Normaliza nombres de audio/CSV a un id base común.

    Trata ``raw_915..._clean.wav``, ``915..._clean.wav`` y
    ``raw_bajas_915..._final_merged.csv`` como el mismo audio base.
    Usada en la fase 04 (consolidación de segmentos).
    """
    x = Path(str(x)).name

    for ext in [".wav", ".csv"]:
        if x.lower().endswith(ext):
            x = x[: -len(ext)]

    for suffix in [
        "_final_merged",
        "_final_segments",
        "_transcribed_segments",
        "_raw",
        "_clean",
    ]:
        if x.endswith(suffix):
            x = x[: -len(suffix)]

    for prefix in ["raw_bajas_", "raw_comercial_", "raw_"]:
        if x.startswith(prefix):
            x = x[len(prefix):]
            break

    return x


def normalize_audio_stem(v):
    """
    Normaliza un nombre de audio a su stem base, quitando extensión de
    audio y sufijos de fase.

    Usada en las fases de sentimiento/fusión (07 y 07C).
    """
    if pd.isna(v):
        return ""

    v = Path(str(v)).name
    v = re.sub(
        r"\.(wav|mp3|m4a|flac|ogg)$",
        "",
        v,
        flags=re.IGNORECASE,
    )

    for suffix in [
        "_final_segments",
        "_final_merged",
        "_transcribed_segments",
        "_raw",
        "_clean",
    ]:
        if v.endswith(suffix):
            v = v[: -len(suffix)]

    return v

def normalize_consolidation_audio_id(value):
    """
    Normaliza nombres de audio y CSV para la consolidación de la fase 04.

    Conserva exactamente las reglas del Notebook 04 original, incluyendo
    la eliminación repetida de prefijos temporales.
    """
    value = Path(str(value)).name

    value = re.sub(r"\.csv$", "", value)
    value = re.sub(r"_final_merged$", "", value)

    value = re.sub(r"\.wav$", "", value)
    value = re.sub(r"_clean$", "", value)

    prefixes = [
        "raw_bajas_",
        "raw_comercial_",
        "raw_",
        "bajas_",
        "comercial_",
    ]

    changed = True

    while changed:
        changed = False

        for prefix in prefixes:
            if value.startswith(prefix):
                value = value[len(prefix):]
                changed = True

    return value

