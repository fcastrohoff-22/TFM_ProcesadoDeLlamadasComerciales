"""Evaluación interna de diarización, relabeling y embeddings.

Este módulo da soporte al Notebook 03. Reúne la lógica reutilizable de
carga/diagnóstico de embeddings, integración de metadata y evaluación proxy.
No modifica los outputs principales de diarización ni de relabeling.
"""

import random
import re
from pathlib import Path

import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from sklearn.metrics import roc_auc_score, roc_curve  # type: ignore

from src.diarizacion import load_audio_as_mono
from src.identidad_audio import add_audio_keys
from src.io_utils import read_csv_robust, write_csv_atomic
from src.reetiquetado_embeddings import get_segment_embedding
from src.storage_io import download_uri_to_local


# ============================================================
# METADATA Y CLAVES DE AUDIO
# ============================================================


def prepare_inventory_for_merge(df_inventory: pd.DataFrame):
    """Prepara el inventario privado para integrarlo con tablas por audio."""
    if df_inventory is None or df_inventory.empty:
        return pd.DataFrame()

    df_inventory = df_inventory.copy()

    for column in ["source_dataset", "audio_id"]:
        if column in df_inventory.columns:
            df_inventory[column] = (
                df_inventory[column]
                .astype(str)
                .str.strip()
            )

    keep_columns = [
        "source_dataset",
        "audio_id",
        "audio_hash",
        "customer_hash",
        "agent_hash",
        "customer_id",
        "agent_id",
        "brand_ds",
        "duration_seconds",
        "silence_ratio",
        "nonsilent_ratio",
    ]
    keep_columns = [
        column
        for column in keep_columns
        if column in df_inventory.columns
    ]

    if not {"source_dataset", "audio_id"}.issubset(keep_columns):
        return pd.DataFrame()

    return (
        df_inventory[keep_columns]
        .drop_duplicates(
            ["source_dataset", "audio_id"],
            keep="first",
        )
        .copy()
    )


def merge_inventory_metadata(
    df: pd.DataFrame,
    df_inventory_small: pd.DataFrame,
):
    """Integra metadata privada usando las claves parseadas del audio."""
    if (
        df is None
        or df.empty
        or df_inventory_small is None
        or df_inventory_small.empty
    ):
        return df

    if (
        "source_dataset_parsed" not in df.columns
        or "audio_id_parsed" not in df.columns
    ):
        return df

    return df.merge(
        df_inventory_small,
        left_on=[
            "source_dataset_parsed",
            "audio_id_parsed",
        ],
        right_on=[
            "source_dataset",
            "audio_id",
        ],
        how="left",
        suffixes=("", "_inventory"),
    )


# ============================================================
# MÉTRICAS INTERNAS DE SEGMENTOS
# ============================================================


def build_audio_quality_table(df_summary: pd.DataFrame):
    """Añade ratios de segmentos válidos y anchors por audio."""
    df_quality_audio = df_summary.copy()

    if {
        "n_scored_segments",
        "n_valid_segments",
    }.issubset(df_quality_audio.columns):
        df_quality_audio["valid_segment_ratio"] = (
            df_quality_audio["n_valid_segments"]
            / df_quality_audio["n_scored_segments"].replace(0, np.nan)
        )

    if {
        "n_scored_segments",
        "n_anchor_segments",
    }.issubset(df_quality_audio.columns):
        df_quality_audio["anchor_segment_ratio"] = (
            df_quality_audio["n_anchor_segments"]
            / df_quality_audio["n_scored_segments"].replace(0, np.nan)
        )

    return df_quality_audio


def build_overlap_threshold_summary(
    df_scored_segments: pd.DataFrame,
    thresholds,
):
    """Resume la distribución de overlap para distintos umbrales."""
    rows = []

    for threshold in thresholds:
        rows.append({
            "threshold": threshold,
            "n_segments_leq": int(
                (
                    df_scored_segments["overlap_ratio"]
                    <= threshold
                ).sum()
            ),
            "pct_segments_leq": float(
                (
                    df_scored_segments["overlap_ratio"]
                    <= threshold
                ).mean()
            ),
            "n_segments_gt": int(
                (
                    df_scored_segments["overlap_ratio"]
                    > threshold
                ).sum()
            ),
            "pct_segments_gt": float(
                (
                    df_scored_segments["overlap_ratio"]
                    > threshold
                ).mean()
            ),
        })

    return pd.DataFrame(rows)


def select_voiceprint_segments(
    df_segments: pd.DataFrame,
    *,
    min_duration: float,
    max_overlap_ratio: float,
    min_rms_dbfs: float,
):
    """Selecciona segmentos candidatos para huella de voz."""
    required_columns = [
        "duration",
        "overlap_ratio",
        "rms_dbfs",
    ]
    missing = [
        column
        for column in required_columns
        if column not in df_segments.columns
    ]

    if missing:
        return pd.DataFrame(), missing, None

    speaker_column = (
        "speaker_final"
        if "speaker_final" in df_segments.columns
        else "speaker"
    )

    if speaker_column not in df_segments.columns:
        return pd.DataFrame(), [speaker_column], None

    df_voiceprint_segments = df_segments[
        (df_segments["duration"] >= min_duration)
        & (
            df_segments["overlap_ratio"]
            <= max_overlap_ratio
        )
        & (
            df_segments["rms_dbfs"]
            >= min_rms_dbfs
        )
        & df_segments[speaker_column].notna()
    ].copy()

    return df_voiceprint_segments, [], speaker_column


# ============================================================
# EMBEDDINGS VECTORIALES
# ============================================================


def detect_embedding_columns(df: pd.DataFrame):
    """Detecta columnas vectoriales explícitas de embeddings."""
    if df is None or df.empty:
        return []

    prefix_candidates = [
        "emb_",
        "embedding_",
        "dim_",
        "xvector_",
        "ecapa_",
    ]

    for prefix in prefix_candidates:
        columns = [
            column
            for column in df.columns
            if str(column).startswith(prefix)
        ]

        if len(columns) >= 16:
            return columns

    known_numeric_not_embedding = {
        "start",
        "end",
        "duration",
        "rms_dbfs",
        "overlap_duration_sec",
        "overlap_ratio",
        "segment_id_raw",
        "anchor_rank",
        "duration_sec",
        "n_speakers",
        "n_regular_segments",
        "n_scored_segments",
        "n_valid_segments",
        "n_anchor_segments",
        "n_overlap_regions",
        "best_distance",
        "second_distance",
        "distance_margin",
        "dist_SPEAKER_00",
        "dist_SPEAKER_01",
    }

    numeric_columns = (
        df.select_dtypes(include=[np.number])
        .columns
        .tolist()
    )

    numeric_columns = [
        column
        for column in numeric_columns
        if column not in known_numeric_not_embedding
    ]

    return (
        numeric_columns
        if len(numeric_columns) >= 32
        else []
    )


def load_embedding_vector_cache(candidate_paths):
    """Carga el primer cache vectorial disponible."""
    for path in candidate_paths:
        if path.exists() and path.stat().st_size > 0:
            dataframe = read_csv_robust(path)

            if not dataframe.empty:
                return add_audio_keys(dataframe), path

    return pd.DataFrame(), None


def _candidate_clean_audio_names(audio_file):
    audio_file = str(audio_file)
    stem = Path(audio_file).stem
    suffix = Path(audio_file).suffix or ".wav"

    candidate_names = []

    for candidate_stem in [
        stem,
        stem.replace("pipelineA_", ""),
    ]:
        candidate_names.append(
            candidate_stem + suffix
        )

        if not candidate_stem.endswith("_clean"):
            candidate_names.append(
                candidate_stem + "_clean" + suffix
            )

    match = re.search(r"([0-9]{10,})", stem)

    if match:
        audio_id = match.group(1)
        candidate_names.extend([
            f"raw_{audio_id}_clean.wav",
            f"raw_bajas_{audio_id}_clean.wav",
            f"pipelineA_raw_{audio_id}_clean.wav",
            f"pipelineA_raw_bajas_{audio_id}_clean.wav",
        ])

    return list(dict.fromkeys(candidate_names))


def resolve_clean_audio_path(
    audio_file,
    *,
    clean_audio_dir: Path,
    clean_audio_gcs_prefix: str,
    gcs_client,
):
    """Localiza un audio limpio y lo restaura de GCS si falta."""
    candidate_names = _candidate_clean_audio_names(audio_file)

    for candidate_name in candidate_names:
        local_path = clean_audio_dir / candidate_name

        if local_path.exists() and local_path.stat().st_size > 0:
            return local_path

    for candidate_name in candidate_names:
        local_path = clean_audio_dir / candidate_name
        source_uri = (
            clean_audio_gcs_prefix.rstrip("/")
            + "/"
            + candidate_name
        )

        download_uri_to_local(
            source_uri,
            local_path,
            gcs_client,
            force=False,
        )

        if local_path.exists() and local_path.stat().st_size > 0:
            return local_path

    return None


def rebuild_anchor_embeddings(
    df_anchor_source: pd.DataFrame,
    *,
    embedding_model,
    clean_audio_dir: Path,
    clean_audio_gcs_prefix: str,
    gcs_client,
    output_path: Path,
    max_anchors=None,
):
    """
    Reconstruye opcionalmente embeddings de anchors y guarda el cache ancho.

    Esta función solo se usa cuando el CSV vectorial del Notebook 01 no está
    disponible y la reconstrucción se activa explícitamente.
    """
    df_anchor_source = df_anchor_source.copy()

    if max_anchors is not None:
        df_anchor_source = (
            df_anchor_source
            .head(int(max_anchors))
            .copy()
        )

    required_columns = {
        "audio_file",
        "start",
        "end",
    }
    missing = required_columns - set(df_anchor_source.columns)

    if missing:
        raise ValueError(
            "Faltan columnas para reconstruir embeddings: "
            f"{missing}"
        )

    rows = []
    audio_cache = {}

    for index, row in (
        df_anchor_source
        .reset_index(drop=True)
        .iterrows()
    ):
        if index % 100 == 0:
            print(
                f"Procesando anchor "
                f"{index + 1}/{len(df_anchor_source)}",
                end="\r",
            )

        audio_file = row["audio_file"]

        if audio_file not in audio_cache:
            audio_path = resolve_clean_audio_path(
                audio_file,
                clean_audio_dir=clean_audio_dir,
                clean_audio_gcs_prefix=(
                    clean_audio_gcs_prefix
                ),
                gcs_client=gcs_client,
            )

            if audio_path is None:
                audio_cache[audio_file] = None
            else:
                _, audio_mono, sample_rate, _ = (
                    load_audio_as_mono(audio_path)
                )
                audio_cache[audio_file] = (
                    audio_mono,
                    sample_rate,
                )

        cached_audio = audio_cache[audio_file]

        if cached_audio is None:
            continue

        audio_mono, sample_rate = cached_audio

        embedding = get_segment_embedding(
            audio_mono=audio_mono,
            sr=sample_rate,
            start=float(row["start"]),
            end=float(row["end"]),
            embedding_model=embedding_model,
        )

        if embedding is None:
            continue

        output_row = row.to_dict()

        for dimension, value in enumerate(embedding):
            output_row[
                f"emb_{dimension:04d}"
            ] = float(value)

        rows.append(output_row)

    print()

    df_anchor_embeddings_vectors = pd.DataFrame(rows)

    if not df_anchor_embeddings_vectors.empty:
        write_csv_atomic(
            df_anchor_embeddings_vectors,
            output_path,
        )

    return df_anchor_embeddings_vectors


def build_speaker_centroids(
    df_anchor_embeddings: pd.DataFrame,
    embedding_columns,
):
    """Construye un centroide por audio y speaker."""
    if (
        df_anchor_embeddings is None
        or df_anchor_embeddings.empty
        or not embedding_columns
    ):
        return pd.DataFrame()

    speaker_column = (
        "speaker_final"
        if "speaker_final" in df_anchor_embeddings.columns
        else (
            "speaker"
            if "speaker" in df_anchor_embeddings.columns
            else None
        )
    )

    if speaker_column is None:
        raise ValueError(
            "No se encontró columna de speaker "
            "para construir centroides."
        )

    group_columns = [
        "audio_file",
        speaker_column,
    ]

    for column in [
        "source_dataset_parsed",
        "audio_id_parsed",
        "audio_hash",
        "customer_hash",
        "agent_hash",
    ]:
        if (
            column in df_anchor_embeddings.columns
            and column not in group_columns
        ):
            group_columns.append(column)

    return (
        df_anchor_embeddings
        .groupby(
            group_columns,
            dropna=False,
        )[embedding_columns]
        .mean()
        .reset_index()
        .rename(
            columns={
                speaker_column: "speaker_label"
            }
        )
    )


# ============================================================
# EVALUACIÓN PROXY
# ============================================================


def cosine_similarity_matrix(matrix_a, matrix_b):
    matrix_a = np.asarray(
        matrix_a,
        dtype=np.float32,
    )
    matrix_b = np.asarray(
        matrix_b,
        dtype=np.float32,
    )

    matrix_a_norm = matrix_a / np.clip(
        np.linalg.norm(
            matrix_a,
            axis=1,
            keepdims=True,
        ),
        1e-10,
        None,
    )

    matrix_b_norm = matrix_b / np.clip(
        np.linalg.norm(
            matrix_b,
            axis=1,
            keepdims=True,
        ),
        1e-10,
        None,
    )

    return matrix_a_norm @ matrix_b_norm.T


def max_similarity_between_audio_speakers(
    df_centroids: pd.DataFrame,
    audio_a,
    audio_b,
    embedding_columns,
):
    matrix_a = df_centroids[
        df_centroids["audio_file"] == audio_a
    ][embedding_columns].values

    matrix_b = df_centroids[
        df_centroids["audio_file"] == audio_b
    ][embedding_columns].values

    if len(matrix_a) == 0 or len(matrix_b) == 0:
        return np.nan

    similarities = cosine_similarity_matrix(
        matrix_a,
        matrix_b,
    )

    return float(np.nanmax(similarities))


def build_proxy_pairs(
    df_centroids: pd.DataFrame,
    identity_column: str,
    embedding_columns,
    *,
    max_pairs_per_identity: int = 20,
    max_negative_pairs: int = 20000,
):
    """Construye pares positivos y negativos para una identidad proxy."""
    df = df_centroids[
        df_centroids[identity_column].notna()
    ].copy()

    identity_audio = (
        df[
            [
                identity_column,
                "audio_file",
            ]
        ]
        .drop_duplicates()
        .groupby(identity_column)["audio_file"]
        .apply(list)
        .to_dict()
    )

    positive_rows = []

    for identity, audios in identity_audio.items():
        audios = list(set(audios))

        if len(audios) < 2:
            continue

        possible_pairs = []

        for first_index in range(len(audios)):
            for second_index in range(
                first_index + 1,
                len(audios),
            ):
                possible_pairs.append(
                    (
                        audios[first_index],
                        audios[second_index],
                    )
                )

        random.shuffle(possible_pairs)

        for audio_a, audio_b in (
            possible_pairs[
                :max_pairs_per_identity
            ]
        ):
            similarity = (
                max_similarity_between_audio_speakers(
                    df,
                    audio_a,
                    audio_b,
                    embedding_columns,
                )
            )

            if not np.isnan(similarity):
                positive_rows.append({
                    "identity": identity,
                    "audio_a": audio_a,
                    "audio_b": audio_b,
                    "similarity": similarity,
                    "label": 1,
                })

    identities = [
        identity
        for identity, audios in identity_audio.items()
        if len(audios) >= 1
    ]

    negative_rows = []
    attempts = 0

    while (
        len(negative_rows) < max_negative_pairs
        and attempts < max_negative_pairs * 20
        and len(identities) >= 2
    ):
        attempts += 1

        identity_a, identity_b = random.sample(
            identities,
            2,
        )

        if identity_a == identity_b:
            continue

        audio_a = random.choice(
            identity_audio[identity_a]
        )
        audio_b = random.choice(
            identity_audio[identity_b]
        )

        similarity = max_similarity_between_audio_speakers(
            df,
            audio_a,
            audio_b,
            embedding_columns,
        )

        if not np.isnan(similarity):
            negative_rows.append({
                "identity": (
                    f"{identity_a}__vs__{identity_b}"
                ),
                "audio_a": audio_a,
                "audio_b": audio_b,
                "similarity": similarity,
                "label": 0,
            })

    return pd.DataFrame(
        positive_rows + negative_rows
    )


def summarize_proxy_pairs(
    df_pairs: pd.DataFrame,
    title: str,
):
    """Calcula el resumen numérico de una evaluación proxy."""
    if df_pairs is None or df_pairs.empty:
        return None

    y_true = df_pairs["label"].values
    y_score = df_pairs["similarity"].values

    roc_auc = (
        roc_auc_score(y_true, y_score)
        if len(np.unique(y_true)) == 2
        else np.nan
    )

    return {
        "evaluation": title,
        "n_pairs": len(df_pairs),
        "n_positive_pairs": int(
            (df_pairs["label"] == 1).sum()
        ),
        "n_negative_pairs": int(
            (df_pairs["label"] == 0).sum()
        ),
        "mean_similarity_positive": (
            df_pairs[
                df_pairs["label"] == 1
            ]["similarity"].mean()
        ),
        "mean_similarity_negative": (
            df_pairs[
                df_pairs["label"] == 0
            ]["similarity"].mean()
        ),
        "median_similarity_positive": (
            df_pairs[
                df_pairs["label"] == 1
            ]["similarity"].median()
        ),
        "median_similarity_negative": (
            df_pairs[
                df_pairs["label"] == 0
            ]["similarity"].median()
        ),
        "roc_auc": roc_auc,
    }


def plot_proxy_pair_evaluation(
    df_pairs: pd.DataFrame,
    title: str,
):
    """Muestra histogramas y curva ROC para una evaluación proxy."""
    if df_pairs is None or df_pairs.empty:
        return

    y_true = df_pairs["label"].values
    y_score = df_pairs["similarity"].values

    roc_auc = (
        roc_auc_score(y_true, y_score)
        if len(np.unique(y_true)) == 2
        else np.nan
    )

    plt.figure(figsize=(10, 5))
    plt.hist(
        df_pairs[
            df_pairs["label"] == 1
        ]["similarity"],
        bins=40,
        alpha=0.6,
        label="misma identidad",
    )
    plt.hist(
        df_pairs[
            df_pairs["label"] == 0
        ]["similarity"],
        bins=40,
        alpha=0.6,
        label="distinta identidad",
    )
    plt.xlabel(
        "Similitud coseno máxima "
        "entre speakers de dos audios"
    )
    plt.ylabel("Frecuencia")
    plt.title(title)
    plt.legend()
    plt.show()

    if len(np.unique(y_true)) == 2:
        false_positive_rate, true_positive_rate, _ = (
            roc_curve(y_true, y_score)
        )

        plt.figure(figsize=(6, 6))
        plt.plot(
            false_positive_rate,
            true_positive_rate,
            label=f"AUC = {roc_auc:.3f}",
        )
        plt.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
        )
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC - {title}")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()
