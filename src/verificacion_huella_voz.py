"""Fase 09: verificación por pares e identificación open-set de huella de voz."""

from pathlib import Path
from itertools import combinations
import os
import json
import hashlib
import tempfile
from datetime import datetime, timezone
import random
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

from src.config import (
    PROJECT_DIR,
    DATA_DIR,
    GCS_UNAV_ROOT,
    GCS_VOICEPRINT_PREFIX,
    EMBEDDING_VECTOR_CSV_DIR,
    PROXY_GROUNDTRUTH_DIR,
    TRANSCRIPTION_ROOT,
    VOICEPRINT_DIR,
    VOICEPRINT_FIGURES_DIR,
    VOICEPRINT_CHECKPOINT_DIR,
    VOICEPRINT_SEGMENT_EMBEDDINGS_CSV,
    VOICEPRINT_ANCHOR_EMBEDDINGS_CSV,
    VOICEPRINT_ROLE_MAPPING_CSV,
    VOICEPRINT_SEGMENT_PROXY_CSV,
    TRANSCRIPTION_FINAL_SEGMENTS_CSV,
    TRANSCRIPTION_ALL_SEGMENTS_CSV,
    VOICEPRINT_FINAL_SUMMARY_CSV,
    VOICEPRINT_OPEN_SET_SUMMARY_CSV,
    VOICEPRINT_SUCCESS_JSON,
    ensure_phase09_directories,
)
from src.storage_io import (
    upload_file,
    upload_directory,
    download_file_if_exists,
    download_directory,
)

warnings.filterwarnings("ignore")

_GCS_CLIENT = None


def stable_hash_dict(payload: dict) -> str:
    """Genera un hash reproducible de una configuración serializable."""
    serialized = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def run_verificacion_huella_voz(
    gcs_client,
    force: bool = False,
):
    """Ejecuta o restaura la evaluación completa de huella de voz."""
    ensure_phase09_directories()

    download_directory(
        local_dir=VOICEPRINT_DIR,
        gcs_prefix=GCS_UNAV_ROOT,
        gcs_client=gcs_client,
        base_dir=DATA_DIR,
    )

    if (
        not force
        and VOICEPRINT_SUCCESS_JSON.exists()
        and VOICEPRINT_OPEN_SET_SUMMARY_CSV.exists()
        and VOICEPRINT_FINAL_SUMMARY_CSV.exists()
    ):
        print("Outputs completos de huella de voz restaurados. No se reconstruyen.")
        return {
            "final_summary": pd.read_csv(VOICEPRINT_FINAL_SUMMARY_CSV),
            "open_set_summary": pd.read_csv(VOICEPRINT_OPEN_SET_SUMMARY_CSV),
            "manifest": json.loads(VOICEPRINT_SUCCESS_JSON.read_text(encoding="utf-8")),
            "reused": True,
        }

    download_directory(
        local_dir=EMBEDDING_VECTOR_CSV_DIR,
        gcs_prefix=GCS_UNAV_ROOT,
        gcs_client=gcs_client,
        base_dir=DATA_DIR,
    )
    download_directory(
        local_dir=PROXY_GROUNDTRUTH_DIR,
        gcs_prefix=GCS_UNAV_ROOT,
        gcs_client=gcs_client,
        base_dir=DATA_DIR,
    )
    download_directory(
        local_dir=TRANSCRIPTION_ROOT,
        gcs_prefix=GCS_UNAV_ROOT,
        gcs_client=gcs_client,
        base_dir=DATA_DIR,
    )

    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    FINAL_RELABEL_DIR = DATA_DIR / "diarization_outputs" / "final_relabel"
    EMBEDDING_VECTOR_DIR = EMBEDDING_VECTOR_CSV_DIR
    PROXY_OUTPUT_DIR = PROXY_GROUNDTRUTH_DIR
    TRANSCRIPTION_OUTPUT_DIR = TRANSCRIPTION_ROOT
    FIGURES_DIR = VOICEPRINT_FIGURES_DIR
    CHECKPOINT_DIR = VOICEPRINT_CHECKPOINT_DIR

    SEGMENT_EMBEDDINGS_PATH = VOICEPRINT_SEGMENT_EMBEDDINGS_CSV
    ANCHOR_EMBEDDINGS_PATH = VOICEPRINT_ANCHOR_EMBEDDINGS_CSV
    ROLE_MAPPING_PATH = VOICEPRINT_ROLE_MAPPING_CSV
    SEGMENT_PROXY_PATH = VOICEPRINT_SEGMENT_PROXY_CSV
    TRANSCRIPTION_CANDIDATES = [
        TRANSCRIPTION_FINAL_SEGMENTS_CSV,
        TRANSCRIPTION_ALL_SEGMENTS_CSV,
        TRANSCRIPTION_ROOT / "transcribed_segments_final.csv",
    ]

    MIN_SEGMENT_DURATION_SEC = 1.50
    MAX_SEGMENT_DURATION_SEC = 20.00
    MAX_OVERLAP_RATIO = 0.05
    MIN_RMS_DBFS = -40.0
    MIN_WORDS_PER_SEGMENT = 0

    MIN_SEGMENTS_PER_AUDIO_PERSON = 1
    MIN_SECONDS_PER_AUDIO_PERSON = 1.50
    MIN_SAMPLES_PER_IDENTITY = 2
    MIN_TOTAL_SECONDS_PER_IDENTITY = 10

    AGENT_TEST_SIZE = 0.30
    USE_CLIENTS_IN_CALIBRATION = False

    MAX_POSITIVE_PAIRS_PER_IDENTITY = 500
    NEGATIVE_MULTIPLIER = 3
    MAX_NEGATIVE_PAIRS = 200_000
    THRESHOLD_STRATEGY = "youden"

    FORCE_REBUILD = force
    RESTORE_FROM_GCS = True
    UPLOAD_TO_GCS = True
    GCS_BUCKET_NAME = "catedras_audio_detection"
    GCS_PROJECT_PREFIX = "pipelineA/procesados_UNAV"

    OPEN_SET_MIN_SAMPLES_PER_IDENTITY = 3
    OPEN_SET_MIN_TOTAL_SECONDS_PER_IDENTITY = 10.0
    MIN_ENROLLMENT_SAMPLES = 2
    MIN_QUERY_SAMPLES = 1
    QUERY_FRACTION = 0.30
    UNKNOWN_IDENTITY_FRACTION = 0.20
    TEST_KNOWN_IDENTITY_FRACTION = 0.30
    MIN_IDENTITIES_FOR_FORMAL_OPEN_SET = 6
    OPEN_SET_THRESHOLD_STRATEGY = "eer"

    BUILD_CLIENT_PROFILES = True
    EMBEDDING_MODEL_LABEL = "embedding_precalculado_notebook_01"
    NOTEBOOK_VERSION = "09_voiceprint_pairwise_open_set_v3"


    # --- Código original de la celda 4 ---
    # ============================================================
    # CELDA 3 - UTILIDADES DE CHECKPOINT LOCAL + GCS
    # ============================================================

    _GCS_CLIENT = None


    def utc_now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()


    def get_gcs_client():
        """Crea el cliente una sola vez. Requiere autenticación disponible en la VM."""
        global _GCS_CLIENT

        if not GCS_BUCKET_NAME:
            raise ValueError(
                "GCS_BUCKET_NAME está vacío. Define TFM_GCS_BUCKET en la VM "
                "o escribe el nombre del bucket en la celda de configuración."
            )

        if _GCS_CLIENT is None:
            from google.cloud import storage
            _GCS_CLIENT = storage.Client()

        return _GCS_CLIENT


    def local_path_to_gcs_blob(local_path: Path) -> str:
        """Conserva la estructura relativa al proyecto dentro del bucket."""
        local_path = Path(local_path).resolve()

        try:
            relative = local_path.relative_to(DATA_DIR.resolve())
        except ValueError:
            relative = Path("external_checkpoints") / local_path.name

        return f"{GCS_PROJECT_PREFIX}/{relative.as_posix()}".strip("/")


    def upload_file_to_gcs(local_path: Path, overwrite: bool = True) -> str | None:
        """
        Función única de subida, deliberadamente genérica para poder compartirla
        después entre todos los notebooks.
        """
        local_path = Path(local_path)

        if not UPLOAD_TO_GCS:
            return None

        if not GCS_BUCKET_NAME:
            print(f"[GCS-SKIP] Bucket no configurado: {local_path.name}")
            return None

        client = get_gcs_client()
        bucket = client.bucket(GCS_BUCKET_NAME)
        blob_name = local_path_to_gcs_blob(local_path)
        blob = bucket.blob(blob_name)

        if blob.exists(client) and not overwrite:
            uri = f"gs://{GCS_BUCKET_NAME}/{blob_name}"
            print(f"[GCS-EXISTS] {uri}")
            return uri

        blob.upload_from_filename(str(local_path))
        uri = f"gs://{GCS_BUCKET_NAME}/{blob_name}"
        print(f"[GCS-UPLOAD] {uri}")
        return uri


    def download_file_from_gcs(local_path: Path, overwrite: bool = False) -> bool:
        """Restaura el checkpoint esperado usando la misma ruta relativa."""
        local_path = Path(local_path)

        if local_path.exists() and not overwrite:
            return True

        if not RESTORE_FROM_GCS or not GCS_BUCKET_NAME:
            return False

        client = get_gcs_client()
        bucket = client.bucket(GCS_BUCKET_NAME)
        blob_name = local_path_to_gcs_blob(local_path)
        blob = bucket.blob(blob_name)

        if not blob.exists(client):
            return False

        local_path.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(local_path))
        print(f"[GCS-DOWNLOAD] gs://{GCS_BUCKET_NAME}/{blob_name}")
        return True


    def ensure_local_checkpoint(local_path: Path) -> bool:
        local_path = Path(local_path)
        return local_path.exists() or download_file_from_gcs(local_path)


    def atomic_write_dataframe(df: pd.DataFrame, path: Path) -> None:
        """Evita interpretar como válido un CSV escrito parcialmente."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".tmp",
            prefix=f"{path.stem}_",
            dir=path.parent,
            delete=False,
            encoding="utf-8",
            newline="",
        ) as tmp:
            tmp_path = Path(tmp.name)

        try:
            df.to_csv(tmp_path, index=False)
            tmp_path.replace(path)
        finally:
            tmp_path.unlink(missing_ok=True)


    def atomic_write_json(payload: dict, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".tmp",
            prefix=f"{path.stem}_",
            dir=path.parent,
            delete=False,
            encoding="utf-8",
        ) as tmp:
            json.dump(payload, tmp, indent=2, ensure_ascii=False, default=str)
            tmp_path = Path(tmp.name)

        try:
            tmp_path.replace(path)
        finally:
            tmp_path.unlink(missing_ok=True)


    def save_dataframe_checkpoint(df: pd.DataFrame, path: Path, upload: bool = True) -> Path:
        atomic_write_dataframe(df, path)
        if upload:
            upload_file_to_gcs(path)
        print(f"[CHECKPOINT] {path} | {len(df):,} filas")
        return path


    def save_json_checkpoint(payload: dict, path: Path, upload: bool = True) -> Path:
        atomic_write_json(payload, path)
        if upload:
            upload_file_to_gcs(path)
        print(f"[CHECKPOINT] {path}")
        return path


    def load_dataframe_checkpoint(path: Path) -> pd.DataFrame | None:
        if not FORCE_REBUILD and ensure_local_checkpoint(path):
            df = pd.read_csv(path)
            print(f"[CHECKPOINT-LOAD] {path.name}: {len(df):,} filas")
            return df
        return None


    def save_figure_checkpoint(fig, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160, bbox_inches="tight")
        upload_file_to_gcs(path)
        print(f"[FIGURE] {path}")
        return path


    def upload_file_to_gcs(local_path: Path, overwrite: bool = True):
        local_path = Path(local_path)
        upload_file(
            local_path,
            gcs_client,
            gcs_prefix=GCS_VOICEPRINT_PREFIX,
            base_dir=VOICEPRINT_DIR,
            skip_unchanged=True,
        )
        return str(local_path)

    def download_file_from_gcs(local_path: Path, overwrite: bool = False):
        local_path = Path(local_path)
        if local_path.exists() and not overwrite:
            return True
        return download_file_if_exists(
            local_path,
            gcs_client,
            gcs_prefix=GCS_VOICEPRINT_PREFIX,
            base_dir=VOICEPRINT_DIR,
        )

    def ensure_local_checkpoint(local_path: Path):
        local_path = Path(local_path)
        return local_path.exists() or download_file_from_gcs(local_path)

    # --- Código original de la celda 6 ---
    # ============================================================
    # CELDA 3 - FUNCIONES AUXILIARES DE CARGA Y NORMALIZACIÓN
    # ============================================================

    def read_csv_required(path: Path, name: str) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"No existe {name}: {path}")
        df = pd.read_csv(path)
        print(f"{name}: {df.shape[0]:,} filas | {df.shape[1]:,} columnas")
        return df


    def read_first_existing(candidates, name: str, required: bool = False) -> pd.DataFrame:
        for p in candidates:
            if Path(p).exists():
                df = pd.read_csv(p)
                print(f"{name}: {df.shape[0]:,} filas | {df.shape[1]:,} columnas | {p.name}")
                return df
        if required:
            raise FileNotFoundError(f"No se encontró archivo para {name}. Candidatos: {candidates}")
        print(f"{name}: no disponible")
        return pd.DataFrame()


    def get_embedding_columns(df: pd.DataFrame):
        return sorted([c for c in df.columns if c.startswith("emb_")])


    def normalize_audio_key(value):
        if pd.isna(value):
            return np.nan
        value = str(value).strip()
        return Path(value).stem


    def choose_existing_col(df: pd.DataFrame, candidates, required=False, label="columna"):
        for c in candidates:
            if c in df.columns:
                return c
        if required:
            raise ValueError(f"No se encontró {label}. Candidatas: {candidates}. Columnas: {list(df.columns)[:50]}")
        return None


    def add_time_key(df: pd.DataFrame, start_col="start", end_col="end", decimals=3):
        df = df.copy()
        if start_col in df.columns and end_col in df.columns:
            df["time_key"] = (
                df[start_col].astype(float).round(decimals).astype(str)
                + "_"
                + df[end_col].astype(float).round(decimals).astype(str)
            )
        return df


    def l2_normalize_matrix(X):
        X = np.asarray(X, dtype=np.float32)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms <= 1e-12] = 1.0
        return X / norms


    def cosine_from_normalized(a, b):
        return float(np.dot(a, b))
    # --- Código original de la celda 7 ---
    # ============================================================
    # CELDA 4 - FUNCIONES DE PARES, UMBRAL, MÉTRICAS Y GRÁFICOS
    # ============================================================

    def build_verification_pairs(
        df_samples: pd.DataFrame,
        emb_cols,
        max_positive_pairs_per_identity=MAX_POSITIVE_PAIRS_PER_IDENTITY,
        negative_multiplier=NEGATIVE_MULTIPLIER,
        max_negative_pairs=MAX_NEGATIVE_PAIRS,
        random_state=RANDOM_SEED,
    ):
        # Cada fila de df_samples representa una muestra audio-persona.
        rng = np.random.default_rng(random_state)
        df = df_samples.reset_index(drop=True).copy()

        if df.empty or df["person_id"].nunique() < 2:
            return pd.DataFrame()

        X = l2_normalize_matrix(df[emb_cols].values)
        person_values = df["person_id"].astype(str).values
        role_values = df["role_proxy"].astype(str).values if "role_proxy" in df.columns else np.array(["UNKNOWN"] * len(df))
        sample_ids = df["sample_id"].astype(str).values
        audio_values = df["audio_key"].astype(str).values if "audio_key" in df.columns else np.array([""] * len(df))

        rows = []

        # Positivos: misma identidad. Se priorizan audios distintos.
        for person_id, idxs in df.groupby("person_id").indices.items():
            idxs = list(idxs)
            if len(idxs) < 2:
                continue

            all_pairs = [(i, j) for i, j in combinations(idxs, 2) if audio_values[i] != audio_values[j]]
            if len(all_pairs) == 0:
                all_pairs = list(combinations(idxs, 2))

            if len(all_pairs) > max_positive_pairs_per_identity:
                chosen_idx = rng.choice(len(all_pairs), size=max_positive_pairs_per_identity, replace=False)
                all_pairs = [all_pairs[k] for k in chosen_idx]

            for i, j in all_pairs:
                rows.append({
                    "sample_id_a": sample_ids[i],
                    "sample_id_b": sample_ids[j],
                    "person_id_a": person_values[i],
                    "person_id_b": person_values[j],
                    "role_a": role_values[i],
                    "role_b": role_values[j],
                    "audio_a": audio_values[i],
                    "audio_b": audio_values[j],
                    "same_identity": 1,
                    "similarity": cosine_from_normalized(X[i], X[j]),
                })

        n_pos = len(rows)
        if n_pos == 0:
            return pd.DataFrame(rows)

        # Negativos: identidades distintas. Se muestrea para evitar explosión combinatoria.
        target_neg = min(max_negative_pairs, n_pos * negative_multiplier)
        neg_rows = []
        attempts = 0
        max_attempts = max(target_neg * 20, 10_000)

        while len(neg_rows) < target_neg and attempts < max_attempts:
            attempts += 1
            i, j = rng.choice(len(df), size=2, replace=False)
            if person_values[i] == person_values[j]:
                continue
            neg_rows.append({
                "sample_id_a": sample_ids[i],
                "sample_id_b": sample_ids[j],
                "person_id_a": person_values[i],
                "person_id_b": person_values[j],
                "role_a": role_values[i],
                "role_b": role_values[j],
                "audio_a": audio_values[i],
                "audio_b": audio_values[j],
                "same_identity": 0,
                "similarity": cosine_from_normalized(X[i], X[j]),
            })

        rows.extend(neg_rows)
        return pd.DataFrame(rows)


    def compute_eer_threshold(y_true, scores):
        fpr, tpr, thresholds = roc_curve(y_true, scores)
        fnr = 1 - tpr
        idx = int(np.nanargmin(np.abs(fpr - fnr)))
        eer = float((fpr[idx] + fnr[idx]) / 2)
        threshold = float(thresholds[idx])
        return eer, threshold, fpr, tpr, thresholds


    def choose_threshold_from_pairs(df_pairs, strategy="youden"):
        y_true = df_pairs["same_identity"].astype(int).values
        scores = df_pairs["similarity"].astype(float).values
        fpr, tpr, thresholds = roc_curve(y_true, scores)
        fnr = 1 - tpr

        if strategy == "eer":
            idx = int(np.nanargmin(np.abs(fpr - fnr)))
        else:
            idx = int(np.nanargmax(tpr - fpr))

        eer_idx = int(np.nanargmin(np.abs(fpr - fnr)))

        return {
            "threshold": float(thresholds[idx]),
            "strategy": strategy,
            "eer": float((fpr[eer_idx] + fnr[eer_idx]) / 2),
            "eer_threshold": float(thresholds[eer_idx]),
            "fpr": fpr,
            "tpr": tpr,
            "thresholds": thresholds,
        }


    def evaluate_pairs(df_pairs, threshold=None, label="dataset"):
        if df_pairs is None or df_pairs.empty:
            return {
                "dataset": label,
                "n_pairs": 0,
                "n_positive": 0,
                "n_negative": 0,
                "auc": np.nan,
                "eer": np.nan,
                "threshold": threshold,
                "accuracy": np.nan,
                "precision": np.nan,
                "recall": np.nan,
                "f1": np.nan,
            }

        y_true = df_pairs["same_identity"].astype(int).values
        scores = df_pairs["similarity"].astype(float).values

        if len(np.unique(y_true)) < 2:
            auc = np.nan
            eer = np.nan
        else:
            auc = float(roc_auc_score(y_true, scores))
            eer, _, _, _, _ = compute_eer_threshold(y_true, scores)

        if threshold is None:
            threshold = choose_threshold_from_pairs(df_pairs, strategy=THRESHOLD_STRATEGY)["threshold"] if len(np.unique(y_true)) == 2 else np.nan

        if pd.isna(threshold):
            accuracy = precision = recall = f1 = np.nan
        else:
            y_pred = (scores >= threshold).astype(int)
            accuracy = float(accuracy_score(y_true, y_pred))
            precision = float(precision_score(y_true, y_pred, zero_division=0))
            recall = float(recall_score(y_true, y_pred, zero_division=0))
            f1 = float(f1_score(y_true, y_pred, zero_division=0))

        return {
            "dataset": label,
            "n_pairs": int(len(df_pairs)),
            "n_positive": int((y_true == 1).sum()),
            "n_negative": int((y_true == 0).sum()),
            "auc": auc,
            "eer": eer,
            "threshold": threshold,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }


    def plot_similarity_distribution(df_pairs, title, output_path=None):
        if df_pairs is None or df_pairs.empty:
            print("No hay pares para graficar:", title)
            return
        plt.figure(figsize=(8, 5))
        df_pos = df_pairs[df_pairs["same_identity"] == 1]
        df_neg = df_pairs[df_pairs["same_identity"] == 0]
        if not df_neg.empty:
            plt.hist(df_neg["similarity"], bins=40, alpha=0.6, label="Impostor")
        if not df_pos.empty:
            plt.hist(df_pos["similarity"], bins=40, alpha=0.6, label="Genuine")
        plt.title(title)
        plt.xlabel("Similitud coseno")
        plt.ylabel("Número de pares")
        plt.legend()
        plt.grid(True, alpha=0.3)
        if output_path is not None:
            plt.savefig(output_path, dpi=160, bbox_inches="tight")
            upload_file_to_gcs(output_path)
        plt.show()


    def plot_roc_curve(df_pairs, title, output_path=None):
        if df_pairs is None or df_pairs.empty or df_pairs["same_identity"].nunique() < 2:
            print("No hay clases suficientes para ROC:", title)
            return
        y_true = df_pairs["same_identity"].astype(int).values
        scores = df_pairs["similarity"].astype(float).values
        fpr, tpr, _ = roc_curve(y_true, scores)
        auc = roc_auc_score(y_true, scores)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--", label="Azar")
        plt.title(title)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.grid(True, alpha=0.3)
        if output_path is not None:
            plt.savefig(output_path, dpi=160, bbox_inches="tight")
            upload_file_to_gcs(output_path)
        plt.show()
    # --- Código original de la celda 9 ---
    # ============================================================
    # CELDA 5 - CARGA DE INPUTS PRINCIPALES
    # ============================================================

    # Embeddings de segmentos generados por Notebook 01 corregido
    df_embeddings = read_csv_required(SEGMENT_EMBEDDINGS_PATH, "Embeddings de segmentos")

    # Mapping speaker_final -> AGENT / CLIENT generado por Notebook 06 corregido
    df_role_mapping = read_csv_required(ROLE_MAPPING_PATH, "Mapping speaker-rol proxy")

    # Proxy segment-level opcional, útil para auditoría
    if SEGMENT_PROXY_PATH.exists():
        df_segment_proxy = pd.read_csv(SEGMENT_PROXY_PATH)
        print(f"Proxy por segmento: {df_segment_proxy.shape[0]:,} filas | {df_segment_proxy.shape[1]:,} columnas")
    else:
        df_segment_proxy = pd.DataFrame()
        print("Proxy por segmento: no disponible")

    # Transcripción opcional para métricas de palabras/texto.
    df_transcriptions = read_first_existing(TRANSCRIPTION_CANDIDATES, "Transcripción segmentada", required=False)

    emb_cols = get_embedding_columns(df_embeddings)
    print("\nColumnas de embedding detectadas:", len(emb_cols))

    if len(emb_cols) == 0:
        raise ValueError("No se detectaron columnas emb_0000, emb_0001, etc. Revisa el output del Notebook 01.")

    print("Dimensión vectorial:", len(emb_cols))
    display(df_embeddings.head(3))
    display(df_role_mapping.head(3))
    # --- Código original de la celda 10 ---
    # ============================================================
    # CELDA 6 - NORMALIZACIÓN DE CLAVES Y COLUMNAS DE ROL
    # ============================================================

    # Normalizar embeddings
    df_embeddings = df_embeddings.copy()

    if "audio_file" not in df_embeddings.columns and "audio_stem" in df_embeddings.columns:
        df_embeddings["audio_file"] = df_embeddings["audio_stem"].astype(str) + ".wav"

    if "audio_stem" not in df_embeddings.columns and "audio_file" in df_embeddings.columns:
        df_embeddings["audio_stem"] = df_embeddings["audio_file"].apply(normalize_audio_key)

    df_embeddings["audio_key"] = df_embeddings["audio_file"].apply(normalize_audio_key)

    speaker_col_emb = choose_existing_col(
        df_embeddings,
        ["speaker_final", "speaker_relabel", "speaker", "label"],
        required=True,
        label="speaker en embeddings",
    )

    if speaker_col_emb != "speaker_final":
        df_embeddings["speaker_final"] = df_embeddings[speaker_col_emb]

    # Normalizar mapping de roles
    df_role_mapping = df_role_mapping.copy()

    if "audio_file" not in df_role_mapping.columns and "audio_stem" in df_role_mapping.columns:
        df_role_mapping["audio_file"] = df_role_mapping["audio_stem"].astype(str) + ".wav"

    if "audio_stem" not in df_role_mapping.columns and "audio_file" in df_role_mapping.columns:
        df_role_mapping["audio_stem"] = df_role_mapping["audio_file"].apply(normalize_audio_key)

    df_role_mapping["audio_key"] = df_role_mapping["audio_file"].apply(normalize_audio_key)

    role_col_mapping = choose_existing_col(
        df_role_mapping,
        ["probable_role", "assigned_role", "official_role_proxy", "role_proxy", "role"],
        required=True,
        label="rol proxy en mapping",
    )

    if role_col_mapping != "role_proxy":
        df_role_mapping["role_proxy"] = df_role_mapping[role_col_mapping]

    speaker_col_mapping = choose_existing_col(
        df_role_mapping,
        ["speaker_final", "speaker", "speaker_label"],
        required=True,
        label="speaker en mapping",
    )

    if speaker_col_mapping != "speaker_final":
        df_role_mapping["speaker_final"] = df_role_mapping[speaker_col_mapping]

    # Mantener columnas necesarias del mapping
    mapping_cols = [
        "audio_key",
        "audio_file",
        "speaker_final",
        "role_proxy",
        "role_confidence",
        "proxy_confidence",
        "role_mapping_status",
        "agent_hash",
        "customer_hash",
        "brand_ds",
        "duration_min",
    ]
    mapping_cols = [c for c in mapping_cols if c in df_role_mapping.columns]

    df_role_mapping_small = (
        df_role_mapping[mapping_cols]
        .drop_duplicates(subset=["audio_key", "speaker_final"], keep="first")
        .copy()
    )

    print("Speaker usado en embeddings:", speaker_col_emb)
    print("Rol usado en mapping:", role_col_mapping)
    print("Filas mapping únicas audio-speaker:", len(df_role_mapping_small))
    display(df_role_mapping_small.head())
    # --- Código original de la celda 11 ---
    # ============================================================
    # CELDA 7 - UNIÓN DE EMBEDDINGS CON ROLES PROXY
    # ============================================================

    # Merge principal: cada segmento con embedding hereda el rol proxy del speaker en ese audio.
    df_voice_segments = df_embeddings.merge(
        df_role_mapping_small,
        on=["audio_key", "speaker_final"],
        how="left",
        suffixes=("", "_mapping"),
    )

    # Resolver rol y person_id
    df_voice_segments["role_proxy"] = df_voice_segments["role_proxy"].astype(str).str.upper().str.strip()

    df_voice_segments["person_id"] = np.where(
        df_voice_segments["role_proxy"].eq("AGENT"),
        df_voice_segments.get("agent_hash"),
        np.where(
            df_voice_segments["role_proxy"].eq("CLIENT"),
            df_voice_segments.get("customer_hash"),
            np.nan,
        ),
    )

    # Métricas opcionales de transcripción: n_words/status.
    if not df_transcriptions.empty:
        df_t = df_transcriptions.copy()
        if "audio_file" not in df_t.columns and "audio_stem" in df_t.columns:
            df_t["audio_file"] = df_t["audio_stem"].astype(str) + ".wav"
        df_t["audio_key"] = df_t["audio_file"].apply(normalize_audio_key)
        df_voice_segments = add_time_key(df_voice_segments, "start", "end")
        df_t = add_time_key(df_t, "start", "end")

        t_cols = ["audio_key", "time_key", "speaker_final", "n_words", "n_chars", "transcription_status"]
        t_cols = [c for c in t_cols if c in df_t.columns]

        if {"audio_key", "time_key"}.issubset(t_cols):
            merge_keys = ["audio_key", "time_key"]
            if "speaker_final" in t_cols and "speaker_final" in df_voice_segments.columns:
                merge_keys.append("speaker_final")
            df_voice_segments = df_voice_segments.merge(
                df_t[t_cols].drop_duplicates(subset=merge_keys),
                on=merge_keys,
                how="left",
                suffixes=("", "_transcription"),
            )

    # Resumen de cobertura
    print("Segmentos con embeddings:", len(df_embeddings))
    print("Segmentos con rol proxy unido:", df_voice_segments["role_proxy"].isin(["AGENT", "CLIENT"]).sum())
    print("Segmentos con person_id:", df_voice_segments["person_id"].notna().sum())
    print("Audios con roles:", df_voice_segments.loc[df_voice_segments["person_id"].notna(), "audio_key"].nunique())

    coverage = (
        df_voice_segments
        .assign(has_person=df_voice_segments["person_id"].notna())
        .groupby("role_proxy", dropna=False)
        .agg(
            n_segments=("audio_key", "size"),
            n_with_person=("has_person", "sum"),
            n_audios=("audio_key", "nunique"),
            n_persons=("person_id", "nunique"),
        )
        .reset_index()
    )
    display(coverage)
    display(df_voice_segments.head(3))
    # --- Código original de la celda 13 ---
    # ============================================================
    # CELDA 8 - FILTROS DE CALIDAD PARA SEGMENTOS DE HUELLA
    # ============================================================

    df_vp = df_voice_segments.copy()

    # Asegurar columnas numéricas
    for c in ["duration", "overlap_ratio", "rms_dbfs", "n_words"]:
        if c in df_vp.columns:
            df_vp[c] = pd.to_numeric(df_vp[c], errors="coerce")

    mask = pd.Series(True, index=df_vp.index)
    mask &= df_vp["role_proxy"].isin(["AGENT", "CLIENT"])
    mask &= df_vp["person_id"].notna()

    if "duration" in df_vp.columns:
        mask &= df_vp["duration"].between(MIN_SEGMENT_DURATION_SEC, MAX_SEGMENT_DURATION_SEC, inclusive="both")

    if "overlap_ratio" in df_vp.columns:
        mask &= df_vp["overlap_ratio"].fillna(0) <= MAX_OVERLAP_RATIO

    if "rms_dbfs" in df_vp.columns:
        mask &= df_vp["rms_dbfs"].fillna(-999) >= MIN_RMS_DBFS

    if "valid_export" in df_vp.columns:
        valid_export = df_vp["valid_export"]
        if valid_export.dtype == object:
            valid_export = valid_export.astype(str).str.lower().isin(["true", "1", "yes", "ok"])
        mask &= valid_export.fillna(False).astype(bool)

    if MIN_WORDS_PER_SEGMENT > 0 and "n_words" in df_vp.columns:
        mask &= df_vp["n_words"].fillna(0) >= MIN_WORDS_PER_SEGMENT

    df_voiceprint_segments = df_vp[mask].copy().reset_index(drop=True)

    # Normalizar embeddings por seguridad
    X_norm = l2_normalize_matrix(df_voiceprint_segments[emb_cols].values)
    df_voiceprint_segments[emb_cols] = X_norm

    print("Segmentos antes de filtros:", len(df_vp))
    print("Segmentos candidatos huella:", len(df_voiceprint_segments))
    print("Porcentaje conservado:", round(len(df_voiceprint_segments) / len(df_vp) * 100, 2), "%")
    print("Audios candidatos:", df_voiceprint_segments["audio_key"].nunique())
    print("Identidades candidatas:", df_voiceprint_segments["person_id"].nunique())

    display(
        df_voiceprint_segments.groupby("role_proxy")
        .agg(
            n_segments=("person_id", "size"),
            n_audios=("audio_key", "nunique"),
            n_persons=("person_id", "nunique"),
            total_duration=("duration", "sum") if "duration" in df_voiceprint_segments.columns else ("person_id", "size"),
        )
        .reset_index()
    )

    VOICEPRINT_SEGMENTS_CSV = VOICEPRINT_DIR / "voiceprint_segments_candidates.csv"
    save_dataframe_checkpoint(df_voiceprint_segments, VOICEPRINT_SEGMENTS_CSV)
    print("Guardado:", VOICEPRINT_SEGMENTS_CSV)
    # --- Código original de la celda 14 ---
    # ============================================================
    # CELDA 9 - CREAR MUESTRAS POR AUDIO-PERSONA
    # ============================================================

    # Una muestra = centroide promedio de una identidad dentro de un audio.
    group_cols = ["person_id", "role_proxy", "audio_key"]
    optional_group_cols = ["agent_hash", "customer_hash", "brand_ds"]
    for c in optional_group_cols:
        if c in df_voiceprint_segments.columns:
            group_cols.append(c)

    agg_dict = {"speaker_final": "first"}
    if "audio_file" in df_voiceprint_segments.columns:
        agg_dict["audio_file"] = "first"
    if "duration" in df_voiceprint_segments.columns:
        agg_dict["duration"] = ["sum", "mean", "count"]
    else:
        agg_dict["person_id"] = "size"
    if "overlap_ratio" in df_voiceprint_segments.columns:
        agg_dict["overlap_ratio"] = "mean"
    if "rms_dbfs" in df_voiceprint_segments.columns:
        agg_dict["rms_dbfs"] = "mean"
    if "n_words" in df_voiceprint_segments.columns:
        agg_dict["n_words"] = "sum"

    df_meta_sample = df_voiceprint_segments.groupby(group_cols, dropna=False).agg(agg_dict)
    df_meta_sample.columns = ["_".join([str(x) for x in col if str(x) != ""]).strip("_") for col in df_meta_sample.columns]
    df_meta_sample = df_meta_sample.reset_index()

    rename_map = {
        "duration_sum": "sample_duration_sec",
        "duration_mean": "mean_segment_duration_sec",
        "duration_count": "n_segments",
        "person_id_size": "n_segments",
        "overlap_ratio_mean": "mean_overlap_ratio",
        "rms_dbfs_mean": "mean_rms_dbfs",
        "n_words_sum": "sample_n_words",
        "speaker_final_first": "speaker_final",
        "audio_file_first": "audio_file",
    }
    df_meta_sample = df_meta_sample.rename(columns={k: v for k, v in rename_map.items() if k in df_meta_sample.columns})

    emb_matrix = (
        df_voiceprint_segments
        .groupby(group_cols, dropna=False)[emb_cols]
        .mean()
        .reset_index()
    )

    df_samples = df_meta_sample.merge(emb_matrix, on=group_cols, how="left")
    df_samples[emb_cols] = l2_normalize_matrix(df_samples[emb_cols].values)

    mask_sample = pd.Series(True, index=df_samples.index)
    if "n_segments" in df_samples.columns:
        mask_sample &= df_samples["n_segments"].fillna(0) >= MIN_SEGMENTS_PER_AUDIO_PERSON
    if "sample_duration_sec" in df_samples.columns:
        mask_sample &= df_samples["sample_duration_sec"].fillna(0) >= MIN_SECONDS_PER_AUDIO_PERSON

    df_samples = df_samples[mask_sample].copy().reset_index(drop=True)
    df_samples["sample_id"] = [f"S{i:06d}" for i in range(len(df_samples))]

    print("Muestras audio-persona:", len(df_samples))
    print("Audios:", df_samples["audio_key"].nunique())
    print("Identidades:", df_samples["person_id"].nunique())

    display(
        df_samples.groupby("role_proxy")
        .agg(
            n_samples=("sample_id", "size"),
            n_audios=("audio_key", "nunique"),
            n_persons=("person_id", "nunique"),
            mean_segments_per_sample=("n_segments", "mean") if "n_segments" in df_samples.columns else ("sample_id", "size"),
            mean_duration_sec=("sample_duration_sec", "mean") if "sample_duration_sec" in df_samples.columns else ("sample_id", "size"),
        )
        .reset_index()
    )

    SAMPLES_CSV = VOICEPRINT_DIR / "voiceprint_audio_person_samples.csv"
    save_dataframe_checkpoint(df_samples, SAMPLES_CSV)
    print("Guardado:", SAMPLES_CSV)
    # --- Código original de la celda 15 ---
    # ============================================================
    # CELDA 10 - RESUMEN DE IDENTIDADES REPETIDAS
    # ============================================================

    identity_agg = {
        "sample_id": "count",
        "audio_key": "nunique",
    }
    if "n_segments" in df_samples.columns:
        identity_agg["n_segments"] = "sum"
    if "sample_duration_sec" in df_samples.columns:
        identity_agg["sample_duration_sec"] = "sum"

    identity_summary = (
        df_samples
        .groupby(["role_proxy", "person_id"], dropna=False)
        .agg(identity_agg)
        .reset_index()
        .rename(columns={
            "sample_id": "n_samples",
            "audio_key": "n_audios",
            "n_segments": "total_segments",
            "sample_duration_sec": "total_duration_sec",
        })
    )

    if "total_duration_sec" not in identity_summary.columns:
        identity_summary["total_duration_sec"] = np.nan
    if "total_segments" not in identity_summary.columns:
        identity_summary["total_segments"] = np.nan

    identity_summary["eligible_verification"] = (
        (identity_summary["n_samples"] >= MIN_SAMPLES_PER_IDENTITY)
        & (identity_summary["total_duration_sec"].fillna(MIN_TOTAL_SECONDS_PER_IDENTITY) >= MIN_TOTAL_SECONDS_PER_IDENTITY)
    )

    IDENTITY_SUMMARY_CSV = VOICEPRINT_DIR / "voiceprint_identity_summary.csv"
    save_dataframe_checkpoint(identity_summary, IDENTITY_SUMMARY_CSV)

    print("Identidades totales:", len(identity_summary))
    print("Identidades elegibles para verificación:", int(identity_summary["eligible_verification"].sum()))

    display(
        identity_summary.groupby("role_proxy")
        .agg(
            n_persons=("person_id", "nunique"),
            n_eligible=("eligible_verification", "sum"),
            mean_samples=("n_samples", "mean"),
            median_samples=("n_samples", "median"),
            max_samples=("n_samples", "max"),
        )
        .reset_index()
    )

    display(
        identity_summary
        .sort_values(["eligible_verification", "n_samples", "total_duration_sec"], ascending=[False, False, False])
        .head(20)
    )
    # --- Código original de la celda 17 ---
    # ============================================================
    # CELDA 11 - SPLIT POR IDENTIDAD
    # ============================================================

    eligible = identity_summary[identity_summary["eligible_verification"]].copy()

    eligible_agents = eligible[eligible["role_proxy"].eq("AGENT")]["person_id"].dropna().astype(str).unique().tolist()
    eligible_clients = eligible[eligible["role_proxy"].eq("CLIENT")]["person_id"].dropna().astype(str).unique().tolist()

    print("Agentes elegibles:", len(eligible_agents))
    print("Clientes repetidos elegibles:", len(eligible_clients))

    if len(eligible_agents) >= 4:
        agent_calib_ids, agent_test_ids = train_test_split(
            eligible_agents,
            test_size=AGENT_TEST_SIZE,
            random_state=RANDOM_SEED,
        )
    elif len(eligible_agents) >= 2:
        agent_calib_ids = eligible_agents[:-1]
        agent_test_ids = eligible_agents[-1:]
    else:
        agent_calib_ids = eligible_agents
        agent_test_ids = []

    if USE_CLIENTS_IN_CALIBRATION and len(eligible_clients) >= 4:
        client_calib_ids, client_test_ids = train_test_split(
            eligible_clients,
            test_size=0.50,
            random_state=RANDOM_SEED,
        )
    else:
        client_calib_ids = []
        client_test_ids = eligible_clients

    split_rows = []
    for pid in agent_calib_ids:
        split_rows.append({"person_id": pid, "role_proxy": "AGENT", "split": "calibration"})
    for pid in agent_test_ids:
        split_rows.append({"person_id": pid, "role_proxy": "AGENT", "split": "test_agent"})
    for pid in client_calib_ids:
        split_rows.append({"person_id": pid, "role_proxy": "CLIENT", "split": "calibration"})
    for pid in client_test_ids:
        split_rows.append({"person_id": pid, "role_proxy": "CLIENT", "split": "test_client"})

    df_identity_split = pd.DataFrame(split_rows)

    calib_set = set(df_identity_split.loc[df_identity_split["split"].eq("calibration"), "person_id"])
    test_set = set(df_identity_split.loc[df_identity_split["split"].ne("calibration"), "person_id"])
    intersection = calib_set.intersection(test_set)

    print("Identidades calibración:", len(calib_set))
    print("Identidades test:", len(test_set))
    print("Intersección calibración/test:", len(intersection))

    if intersection:
        raise ValueError("Hay fuga de identidad entre calibración y test.")

    IDENTITY_SPLIT_CSV = VOICEPRINT_DIR / "voiceprint_identity_split.csv"
    save_dataframe_checkpoint(df_identity_split, IDENTITY_SPLIT_CSV)
    print("Guardado:", IDENTITY_SPLIT_CSV)

    display(df_identity_split.groupby(["role_proxy", "split"]).size().reset_index(name="n_identities"))
    # --- Código original de la celda 19 ---
    # ============================================================
    # CELDA 12 - CREAR DATASETS DE CALIBRACIÓN Y PRUEBA
    # ============================================================

    def subset_samples_by_ids(ids, role=None):
        ids = set([str(x) for x in ids])
        df = df_samples[df_samples["person_id"].astype(str).isin(ids)].copy()
        if role is not None:
            df = df[df["role_proxy"].eq(role)].copy()
        return df.reset_index(drop=True)

    # Calibración: agentes, y opcionalmente clientes si se activa la bandera.
    df_samples_agent_calib = subset_samples_by_ids(agent_calib_ids, role="AGENT")
    df_samples_agent_test = subset_samples_by_ids(agent_test_ids, role="AGENT")
    df_samples_client_test = subset_samples_by_ids(client_test_ids, role="CLIENT")

    if USE_CLIENTS_IN_CALIBRATION:
        df_samples_client_calib = subset_samples_by_ids(client_calib_ids, role="CLIENT")
        df_samples_calibration = pd.concat([df_samples_agent_calib, df_samples_client_calib], ignore_index=True)
    else:
        df_samples_calibration = df_samples_agent_calib.copy()

    print("Muestras calibración:", len(df_samples_calibration), "| identidades:", df_samples_calibration["person_id"].nunique())
    print("Muestras test agentes:", len(df_samples_agent_test), "| identidades:", df_samples_agent_test["person_id"].nunique())
    print("Muestras test clientes:", len(df_samples_client_test), "| identidades:", df_samples_client_test["person_id"].nunique())
    # --- Código original de la celda 20 ---
    # ============================================================
    # CELDA 13 - GENERAR PARES GENUINE / IMPOSTOR
    # ============================================================

    df_pairs_calibration = build_verification_pairs(
        df_samples_calibration,
        emb_cols,
        random_state=RANDOM_SEED,
    )

    df_pairs_agent_test = build_verification_pairs(
        df_samples_agent_test,
        emb_cols,
        random_state=RANDOM_SEED + 1,
    )

    df_pairs_client_test = build_verification_pairs(
        df_samples_client_test,
        emb_cols,
        random_state=RANDOM_SEED + 2,
    )

    print("Pares calibración:", len(df_pairs_calibration))
    print("Pares test agentes:", len(df_pairs_agent_test))
    print("Pares test clientes:", len(df_pairs_client_test))

    PAIRS_CALIBRATION_CSV = VOICEPRINT_DIR / "voiceprint_pairs_calibration.csv"
    PAIRS_AGENT_TEST_CSV = VOICEPRINT_DIR / "voiceprint_pairs_test_agents.csv"
    PAIRS_CLIENT_TEST_CSV = VOICEPRINT_DIR / "voiceprint_pairs_test_clients.csv"

    save_dataframe_checkpoint(df_pairs_calibration, PAIRS_CALIBRATION_CSV)
    save_dataframe_checkpoint(df_pairs_agent_test, PAIRS_AGENT_TEST_CSV)
    save_dataframe_checkpoint(df_pairs_client_test, PAIRS_CLIENT_TEST_CSV)

    print("Guardados:")
    print(PAIRS_CALIBRATION_CSV)
    print(PAIRS_AGENT_TEST_CSV)
    print(PAIRS_CLIENT_TEST_CSV)

    display(df_pairs_calibration.head())
    # --- Código original de la celda 22 ---
    # ============================================================
    # CELDA 14 - CALIBRAR UMBRAL CON IDENTIDADES DE CALIBRACIÓN
    # ============================================================

    if df_pairs_calibration.empty or df_pairs_calibration["same_identity"].nunique() < 2:
        raise ValueError("No hay pares suficientes para calibrar umbral. Revisa identidades repetidas de agentes.")

    threshold_info = choose_threshold_from_pairs(df_pairs_calibration, strategy=THRESHOLD_STRATEGY)
    VOICEPRINT_THRESHOLD = threshold_info["threshold"]

    print("Estrategia de umbral:", threshold_info["strategy"])
    print("Umbral calibrado:", round(VOICEPRINT_THRESHOLD, 4))
    print("EER en calibración:", round(threshold_info["eer"], 4))

    threshold_summary = pd.DataFrame([{
        "threshold_strategy": THRESHOLD_STRATEGY,
        "voiceprint_threshold": VOICEPRINT_THRESHOLD,
        "eer_threshold": threshold_info["eer_threshold"],
        "calibration_eer": threshold_info["eer"],
        "n_calibration_pairs": len(df_pairs_calibration),
        "n_calibration_positive": int((df_pairs_calibration["same_identity"] == 1).sum()),
        "n_calibration_negative": int((df_pairs_calibration["same_identity"] == 0).sum()),
    }])

    THRESHOLD_CSV = VOICEPRINT_DIR / "voiceprint_threshold_summary.csv"
    save_dataframe_checkpoint(threshold_summary, THRESHOLD_CSV)
    display(threshold_summary)
    # --- Código original de la celda 23 ---
    # ============================================================
    # CELDA 15 - MÉTRICAS EN CALIBRACIÓN, AGENTES NO VISTOS Y CLIENTES REPETIDOS
    # ============================================================

    metrics_rows = []
    metrics_rows.append(evaluate_pairs(df_pairs_calibration, VOICEPRINT_THRESHOLD, label="calibration_agents"))
    metrics_rows.append(evaluate_pairs(df_pairs_agent_test, VOICEPRINT_THRESHOLD, label="test_agents_unseen"))
    metrics_rows.append(evaluate_pairs(df_pairs_client_test, VOICEPRINT_THRESHOLD, label="test_clients_repeated"))

    df_metrics = pd.DataFrame(metrics_rows)

    METRICS_CSV = VOICEPRINT_DIR / "voiceprint_metrics_summary.csv"
    save_dataframe_checkpoint(df_metrics, METRICS_CSV)

    print("Métricas guardadas en:", METRICS_CSV)
    display(df_metrics)
    # --- Código original de la celda 24 ---
    # ============================================================
    # CELDA 16 - MATRICES DE CONFUSIÓN AL UMBRAL CALIBRADO
    # ============================================================

    def confusion_table(df_pairs, threshold, label):
        if df_pairs is None or df_pairs.empty or pd.isna(threshold):
            return pd.DataFrame()
        y_true = df_pairs["same_identity"].astype(int).values
        y_pred = (df_pairs["similarity"].astype(float).values >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
        return pd.DataFrame(
            cm,
            index=["true_genuine", "true_impostor"],
            columns=["pred_genuine", "pred_impostor"],
        ).assign(dataset=label).reset_index().rename(columns={"index": "true_label"})

    cm_calib = confusion_table(df_pairs_calibration, VOICEPRINT_THRESHOLD, "calibration_agents")
    cm_agent = confusion_table(df_pairs_agent_test, VOICEPRINT_THRESHOLD, "test_agents_unseen")
    cm_client = confusion_table(df_pairs_client_test, VOICEPRINT_THRESHOLD, "test_clients_repeated")

    df_confusion = pd.concat([cm_calib, cm_agent, cm_client], ignore_index=True)

    CONFUSION_CSV = VOICEPRINT_DIR / "voiceprint_confusion_matrices.csv"
    save_dataframe_checkpoint(df_confusion, CONFUSION_CSV)

    display(df_confusion)
    # --- Código original de la celda 26 ---
    # ============================================================
    # CELDA 17 - DISTRIBUCIONES DE SIMILITUD
    # ============================================================

    plot_similarity_distribution(
        df_pairs_calibration,
        "Distribución de similitud - calibración con agentes",
        FIGURES_DIR / "similarity_distribution_calibration_agents.png",
    )

    plot_similarity_distribution(
        df_pairs_agent_test,
        "Distribución de similitud - agentes no vistos",
        FIGURES_DIR / "similarity_distribution_test_agents.png",
    )

    plot_similarity_distribution(
        df_pairs_client_test,
        "Distribución de similitud - clientes repetidos",
        FIGURES_DIR / "similarity_distribution_test_clients.png",
    )
    # --- Código original de la celda 27 ---
    # ============================================================
    # CELDA 18 - CURVAS ROC
    # ============================================================

    plot_roc_curve(
        df_pairs_calibration,
        "Curva ROC - calibración con agentes",
        FIGURES_DIR / "roc_calibration_agents.png",
    )

    plot_roc_curve(
        df_pairs_agent_test,
        "Curva ROC - agentes no vistos",
        FIGURES_DIR / "roc_test_agents.png",
    )

    plot_roc_curve(
        df_pairs_client_test,
        "Curva ROC - clientes repetidos",
        FIGURES_DIR / "roc_test_clients.png",
    )
    # --- Código original de la celda 28 ---
    # ============================================================
    # CELDA 19 - VISUALIZACIÓN DE REPETICIÓN DE IDENTIDADES
    # ============================================================

    plt.figure(figsize=(8, 5))
    plot_df = identity_summary.copy()
    plot_df["n_samples_capped"] = plot_df["n_samples"].clip(upper=20)
    for role, group in plot_df.groupby("role_proxy"):
        plt.hist(group["n_samples_capped"], bins=20, alpha=0.6, label=role)
    plt.title("Repetición de identidades por rol")
    plt.xlabel("Número de muestras por identidad, truncado en 20")
    plt.ylabel("Número de identidades")
    plt.legend()
    plt.grid(True, alpha=0.3)
    IDENTITY_REPETITION_FIGURE = FIGURES_DIR / "identity_repetition_by_role.png"
    plt.savefig(IDENTITY_REPETITION_FIGURE, dpi=160, bbox_inches="tight")
    upload_file_to_gcs(IDENTITY_REPETITION_FIGURE)
    plt.show()
    # --- Código original de la celda 30 ---
    # ============================================================
    # CELDA 20 - EJEMPLOS DE PARES MÁS Y MENOS SIMILARES
    # ============================================================

    def show_pair_examples(df_pairs, title, n=10):
        if df_pairs is None or df_pairs.empty:
            print("No hay pares para:", title)
            return
        print("=" * 80)
        print(title)
        print("=" * 80)
        cols = [
            "same_identity",
            "similarity",
            "person_id_a",
            "person_id_b",
            "role_a",
            "role_b",
            "audio_a",
            "audio_b",
            "sample_id_a",
            "sample_id_b",
        ]
        cols = [c for c in cols if c in df_pairs.columns]
        display(df_pairs.sort_values("similarity", ascending=False)[cols].head(n))
        display(df_pairs.sort_values("similarity", ascending=True)[cols].head(n))

    show_pair_examples(df_pairs_agent_test, "Pares de agentes no vistos")
    show_pair_examples(df_pairs_client_test, "Pares de clientes repetidos")
    # --- Código original de la celda 32 ---
    # ============================================================
    # CELDA 21 - RESUMEN FINAL EJECUTIVO
    # ============================================================

    summary_rows = []
    summary_rows.append({"métrica": "Segmentos con embeddings", "valor": len(df_embeddings)})
    summary_rows.append({"métrica": "Segmentos candidatos para huella", "valor": len(df_voiceprint_segments)})
    summary_rows.append({"métrica": "Muestras audio-persona", "valor": len(df_samples)})
    summary_rows.append({"métrica": "Identidades totales", "valor": identity_summary["person_id"].nunique()})
    summary_rows.append({"métrica": "Identidades elegibles", "valor": int(identity_summary["eligible_verification"].sum())})
    summary_rows.append({"métrica": "Agentes elegibles", "valor": len(eligible_agents)})
    summary_rows.append({"métrica": "Clientes repetidos elegibles", "valor": len(eligible_clients)})
    summary_rows.append({"métrica": "Umbral calibrado", "valor": round(float(VOICEPRINT_THRESHOLD), 4)})

    for _, row in df_metrics.iterrows():
        summary_rows.append({"métrica": f"AUC - {row['dataset']}", "valor": None if pd.isna(row["auc"]) else round(float(row["auc"]), 4)})
        summary_rows.append({"métrica": f"EER - {row['dataset']}", "valor": None if pd.isna(row["eer"]) else round(float(row["eer"]), 4)})
        summary_rows.append({"métrica": f"F1 - {row['dataset']}", "valor": None if pd.isna(row["f1"]) else round(float(row["f1"]), 4)})

    df_final_summary = pd.DataFrame(summary_rows)

    FINAL_SUMMARY_CSV = VOICEPRINT_DIR / "voiceprint_final_summary_for_memory.csv"
    save_dataframe_checkpoint(df_final_summary, FINAL_SUMMARY_CSV)

    display(df_final_summary)
    print("Resumen guardado en:", FINAL_SUMMARY_CSV)

    print("\nInterpretación base:")
    print(
        "La huella de voz se evaluó como verificación de hablante usando embeddings precomputados. "
        "El umbral se calibró con identidades de agentes y se aplicó a agentes no vistos y clientes repetidos, "
        "evitando que una misma identidad apareciera simultáneamente en calibración y prueba."
    )
    # --- Código original de la celda 35 ---
    # ============================================================
    # CELDA 22 - PREPARAR MUESTRAS PARA OPEN-SET
    # ============================================================

    # Copia independiente: no modifica los outputs anteriores.
    df_samples_open_set = df_samples.copy()
    df_samples_open_set['source_identity_id'] = (
        df_samples_open_set['person_id'].astype(str)
    )

    identity_summary_open_set = (
        df_samples_open_set
        .groupby(['role_proxy', 'person_id', 'source_identity_id'], dropna=False)
        .agg(
            n_samples=('sample_id', 'size'),
            n_calls=('audio_key', 'nunique'),
            total_segments=('n_segments', 'sum'),
            total_duration_sec=('sample_duration_sec', 'sum'),
            mean_sample_duration_sec=('sample_duration_sec', 'mean'),
        )
        .reset_index()
    )

    identity_summary_open_set['eligible_profile'] = (
        identity_summary_open_set['n_samples'].ge(
            OPEN_SET_MIN_SAMPLES_PER_IDENTITY
        )
        & identity_summary_open_set['total_duration_sec'].ge(
            OPEN_SET_MIN_TOTAL_SECONDS_PER_IDENTITY
        )
    )

    OPEN_SET_IDENTITY_SUMMARY_PATH = (
        VOICEPRINT_DIR / 'voiceprint_identity_summary_open_set.csv'
    )
    save_dataframe_checkpoint(
        identity_summary_open_set,
        OPEN_SET_IDENTITY_SUMMARY_PATH,
    )

    display(
        identity_summary_open_set.groupby('role_proxy')
        .agg(
            identities=('person_id', 'nunique'),
            eligible=('eligible_profile', 'sum'),
            median_calls=('n_calls', 'median'),
            max_calls=('n_calls', 'max'),
        )
        .reset_index()
    )

    # --- Código original de la celda 37 ---
    # ============================================================
    # CELDA 9 - FUNCIONES DE SPLIT
    # ============================================================

    def deterministic_identity_partition(identity_ids, random_state=RANDOM_SEED):
        ids = sorted({str(x) for x in identity_ids})
        rng = np.random.default_rng(random_state)
        ids = list(rng.permutation(ids))
        n_ids = len(ids)

        if n_ids < MIN_IDENTITIES_FOR_FORMAL_OPEN_SET:
            raise ValueError(
                f"Solo hay {n_ids} agentes elegibles. Se necesitan al menos "
                f"{MIN_IDENTITIES_FOR_FORMAL_OPEN_SET} para separar calibración, "
                "test conocido y test unknown."
            )

        n_unknown = max(1, int(round(n_ids * UNKNOWN_IDENTITY_FRACTION)))
        n_unknown = min(n_unknown, n_ids - 4)
        unknown_ids = ids[:n_unknown]
        known_pool = ids[n_unknown:]

        n_test_known = max(2, int(round(len(known_pool) * TEST_KNOWN_IDENTITY_FRACTION)))
        n_test_known = min(n_test_known, len(known_pool) - 2)

        return {
            "calibration_known": sorted(known_pool[n_test_known:]),
            "test_known": sorted(known_pool[:n_test_known]),
            "test_unknown": sorted(unknown_ids),
        }


    def split_identity_calls(samples, identity_ids, group_label, random_state):
        rng = np.random.default_rng(random_state)
        rows = []

        subset = samples[
            samples["person_id"].astype(str).isin(set(map(str, identity_ids)))
        ].copy()

        for person_id, group in subset.groupby("person_id", sort=True):
            group = group.sort_values(["audio_key", "sample_id"]).copy()
            indices = list(rng.permutation(group.index.to_list()))
            n_samples = len(indices)

            n_query = max(MIN_QUERY_SAMPLES, int(round(n_samples * QUERY_FRACTION)))
            n_query = min(n_query, n_samples - MIN_ENROLLMENT_SAMPLES)

            if n_query < MIN_QUERY_SAMPLES:
                raise ValueError(
                    f"{person_id} no permite separar enrollment/query: "
                    f"{n_samples} muestras."
                )

            query_indices = set(indices[:n_query])

            for index in indices:
                row = group.loc[index]
                rows.append({
                    "sample_id": row["sample_id"],
                    "person_id": str(person_id),
                    "source_identity_id": row["source_identity_id"],
                    "role_proxy": row["role_proxy"],
                    "audio_key": row["audio_key"],
                    "identity_group": group_label,
                    "sample_split": "query" if index in query_indices else "enrollment",
                })

        return pd.DataFrame(rows)


    def build_open_set_split(samples, agent_identity_ids):
        partition = deterministic_identity_partition(agent_identity_ids)

        calibration = split_identity_calls(
            samples,
            partition["calibration_known"],
            "calibration_known",
            RANDOM_SEED + 11,
        )
        test_known = split_identity_calls(
            samples,
            partition["test_known"],
            "test_known",
            RANDOM_SEED + 22,
        )

        unknown = samples[
            samples["person_id"].astype(str).isin(set(partition["test_unknown"]))
        ][[
            "sample_id", "person_id", "source_identity_id",
            "role_proxy", "audio_key",
        ]].copy()
        unknown["identity_group"] = "test_unknown"
        unknown["sample_split"] = "query"

        split = pd.concat([calibration, test_known, unknown], ignore_index=True)

        if not split[
            split["identity_group"].eq("test_unknown")
            & split["sample_split"].eq("enrollment")
        ].empty:
            raise AssertionError("Una identidad unknown apareció en enrollment.")

        return split, partition

    # --- Código original de la celda 38 ---
    # ============================================================
    # CELDA 10 - CREAR O CARGAR SPLIT
    # ============================================================

    SPLIT_CHECKPOINT = CHECKPOINT_DIR / "04_voiceprint_open_set_split.csv"
    PARTITION_CHECKPOINT = CHECKPOINT_DIR / "04_voiceprint_identity_partition.json"

    df_open_set_split = load_dataframe_checkpoint(SPLIT_CHECKPOINT)

    if (
        df_open_set_split is None
        or FORCE_REBUILD
        or not ensure_local_checkpoint(PARTITION_CHECKPOINT)
    ):
        eligible_agents = (
            identity_summary_open_set[
                identity_summary_open_set["role_proxy"].eq("AGENT")
                & identity_summary_open_set["eligible_profile"].astype(bool)
            ]["person_id"]
            .dropna()
            .astype(str)
            .tolist()
        )

        df_open_set_split, identity_partition = build_open_set_split(
            df_samples_open_set,
            eligible_agents,
        )

        save_dataframe_checkpoint(df_open_set_split, SPLIT_CHECKPOINT)
        save_json_checkpoint(identity_partition, PARTITION_CHECKPOINT)
    else:
        with open(PARTITION_CHECKPOINT, "r", encoding="utf-8") as file:
            identity_partition = json.load(file)

    print({key: len(value) for key, value in identity_partition.items()})

    display(
        df_open_set_split.groupby(["identity_group", "sample_split"])
        .agg(
            identities=("person_id", "nunique"),
            samples=("sample_id", "size"),
            calls=("audio_key", "nunique"),
        )
        .reset_index()
    )

    # --- Código original de la celda 40 ---
    # ============================================================
    # CELDA 11 - FUNCIONES DE PERFILES
    # ============================================================

    def calculate_within_profile_similarity(sample_embeddings, centroid):
        similarities = cosine_similarity_matrix(
            sample_embeddings,
            centroid.reshape(1, -1),
        ).reshape(-1)

        return {
            "within_similarity_mean": float(np.mean(similarities)),
            "within_similarity_std": float(np.std(similarities)),
            "within_similarity_min": float(np.min(similarities)),
            "within_similarity_max": float(np.max(similarities)),
        }


    def build_speaker_profiles(enrollment_samples, embedding_columns, profile_set_name):
        rows = []

        for person_id, group in enrollment_samples.groupby("person_id", sort=True):
            matrix = l2_normalize_matrix(
                group[embedding_columns].to_numpy(dtype=np.float32)
            )
            centroid = l2_normalize_matrix(matrix.mean(axis=0).reshape(1, -1)).reshape(-1)
            consistency = calculate_within_profile_similarity(matrix, centroid)

            row = {
                "profile_id": str(person_id),
                "source_identity_id": group["source_identity_id"].iloc[0],
                "role_proxy": group["role_proxy"].iloc[0],
                "profile_set": profile_set_name,
                "n_enrollment_samples": int(len(group)),
                "n_enrollment_calls": int(group["audio_key"].nunique()),
                "n_enrollment_segments": int(group["n_segments"].fillna(0).sum()),
                "total_enrollment_duration_sec": float(
                    group["sample_duration_sec"].fillna(0).sum()
                ),
                "mean_sample_duration_sec": float(
                    group["sample_duration_sec"].fillna(0).mean()
                ),
                "mean_overlap_ratio": (
                    float(group["mean_overlap_ratio"].mean())
                    if "mean_overlap_ratio" in group.columns else np.nan
                ),
                "mean_rms_dbfs": (
                    float(group["mean_rms_dbfs"].mean())
                    if "mean_rms_dbfs" in group.columns else np.nan
                ),
                "embedding_dim": len(embedding_columns),
                "embedding_model": EMBEDDING_MODEL_LABEL,
                "notebook_version": NOTEBOOK_VERSION,
                **consistency,
            }
            row.update({
                column: float(value)
                for column, value in zip(embedding_columns, centroid)
            })
            rows.append(row)

        profiles = pd.DataFrame(rows)
        if not profiles.empty:
            profiles[embedding_columns] = l2_normalize_matrix(
                profiles[embedding_columns].to_numpy(dtype=np.float32)
            )
        return profiles


    def get_samples_for_split(samples, split, identity_group, sample_split):
        ids = split[
            split["identity_group"].eq(identity_group)
            & split["sample_split"].eq(sample_split)
        ]["sample_id"]

        return samples[
            samples["sample_id"].isin(set(ids))
        ].copy().reset_index(drop=True)

    # --- Código original de la celda 41 ---
    # ============================================================
    # CELDA 12 - PERFILES DE CALIBRACIÓN Y TEST
    # ============================================================

    CALIBRATION_PROFILES_CHECKPOINT = CHECKPOINT_DIR / "05_agent_profiles_calibration.csv"
    TEST_PROFILES_CHECKPOINT = CHECKPOINT_DIR / "05_agent_profiles_test.csv"

    calibration_profiles = load_dataframe_checkpoint(CALIBRATION_PROFILES_CHECKPOINT)
    test_profiles = load_dataframe_checkpoint(TEST_PROFILES_CHECKPOINT)

    if calibration_profiles is None:
        calibration_enrollment = get_samples_for_split(
            df_samples_open_set, df_open_set_split, "calibration_known", "enrollment"
        )
        calibration_profiles = build_speaker_profiles(
            calibration_enrollment,
            emb_cols,
            "calibration",
        )
        save_dataframe_checkpoint(
            calibration_profiles,
            CALIBRATION_PROFILES_CHECKPOINT,
        )

    if test_profiles is None:
        test_enrollment = get_samples_for_split(
            df_samples_open_set, df_open_set_split, "test_known", "enrollment"
        )
        test_profiles = build_speaker_profiles(
            test_enrollment,
            emb_cols,
            "test",
        )
        save_dataframe_checkpoint(test_profiles, TEST_PROFILES_CHECKPOINT)

    print("Perfiles calibración:", len(calibration_profiles))
    print("Perfiles test:", len(test_profiles))

    display(
        calibration_profiles[[
            "profile_id",
            "n_enrollment_calls",
            "n_enrollment_segments",
            "total_enrollment_duration_sec",
            "within_similarity_mean",
            "within_similarity_min",
        ]].head()
    )

    # --- Código original de la celda 43 ---
    # ============================================================
    # CELDA 13 - FUNCIONES DE IDENTIFICACIÓN Y UMBRAL
    # ============================================================

    def score_queries_against_profiles(
        query_samples,
        profiles,
        embedding_columns,
        query_group,
    ):
        if query_samples.empty:
            return pd.DataFrame(), pd.DataFrame()
        if profiles.empty:
            raise ValueError(f"No hay perfiles para consultar {query_group}.")

        query_matrix = query_samples[embedding_columns].to_numpy(dtype=np.float32)
        profile_matrix = profiles[embedding_columns].to_numpy(dtype=np.float32)
        similarities = cosine_similarity_matrix(query_matrix, profile_matrix)
        profile_ids = profiles["profile_id"].astype(str).to_numpy()
        profile_source_ids = profiles["source_identity_id"].astype(str).to_numpy()
        profile_id_set = set(profile_ids)

        score_rows = []
        prediction_rows = []

        for qpos, (_, query) in enumerate(query_samples.reset_index(drop=True).iterrows()):
            query_scores = similarities[qpos]
            order = np.argsort(-query_scores)
            best_idx = int(order[0])
            second_idx = int(order[1]) if len(order) > 1 else None

            best_profile_id = profile_ids[best_idx]
            best_source_identity_id = profile_source_ids[best_idx]
            best_score = float(query_scores[best_idx])

            if second_idx is None:
                second_profile_id = None
                second_source_identity_id = None
                second_score = np.nan
                margin = np.nan
            else:
                second_profile_id = profile_ids[second_idx]
                second_source_identity_id = profile_source_ids[second_idx]
                second_score = float(query_scores[second_idx])
                margin = best_score - second_score

            true_person_id = str(query["person_id"])
            true_is_enrolled = true_person_id in profile_id_set

            prediction_rows.append({
                "query_group": query_group,
                "sample_id": query["sample_id"],
                "audio_key": query["audio_key"],
                "true_person_id": true_person_id,
                "true_source_identity_id": query.get("source_identity_id", pd.NA),
                "true_is_enrolled": bool(true_is_enrolled),
                "best_profile_id": best_profile_id,
                "best_source_identity_id": best_source_identity_id,
                "best_similarity": best_score,
                "second_profile_id": second_profile_id,
                "second_source_identity_id": second_source_identity_id,
                "second_similarity": second_score,
                "top1_top2_margin": margin,
                "top1_correct_before_threshold": (
                    bool(true_is_enrolled)
                    and best_profile_id == true_person_id
                ),
            })

            for ppos, profile_id in enumerate(profile_ids):
                score_rows.append({
                    "query_group": query_group,
                    "sample_id": query["sample_id"],
                    "audio_key": query["audio_key"],
                    "true_person_id": true_person_id,
                    "true_is_enrolled": bool(true_is_enrolled),
                    "candidate_profile_id": str(profile_id),
                    "same_identity": int(
                        true_is_enrolled and str(profile_id) == true_person_id
                    ),
                    "similarity": float(query_scores[ppos]),
                })

        return pd.DataFrame(score_rows), pd.DataFrame(prediction_rows)


    def choose_verification_threshold(score_table, strategy="eer"):
        if score_table.empty:
            raise ValueError("No hay scores para calibrar el umbral.")

        y_true = score_table["same_identity"].astype(int).to_numpy()
        scores = score_table["similarity"].astype(float).to_numpy()

        if len(np.unique(y_true)) < 2:
            raise ValueError("La calibración necesita scores genuine e impostor.")

        fpr, tpr, thresholds = roc_curve(y_true, scores)
        fnr = 1 - tpr

        eer_idx = int(np.nanargmin(np.abs(fpr - fnr)))
        youden_idx = int(np.nanargmax(tpr - fpr))
        selected_idx = eer_idx if strategy.lower() == "eer" else youden_idx

        return {
            "strategy": strategy.lower(),
            "acceptance_threshold": float(thresholds[selected_idx]),
            "eer": float((fpr[eer_idx] + fnr[eer_idx]) / 2),
            "eer_threshold": float(thresholds[eer_idx]),
            "youden_threshold": float(thresholds[youden_idx]),
            "calibration_auc": float(roc_auc_score(y_true, scores)),
            "n_scores": int(len(score_table)),
            "n_genuine_scores": int((y_true == 1).sum()),
            "n_impostor_scores": int((y_true == 0).sum()),
        }


    def apply_open_set_decision(predictions, threshold):
        output = predictions.copy()
        output["decision"] = np.where(
            output["best_similarity"].ge(threshold),
            "KNOWN",
            "UNKNOWN",
        )
        output["predicted_person_id"] = np.where(
            output["decision"].eq("KNOWN"),
            output["best_profile_id"],
            pd.NA,
        )
        output["predicted_source_identity_id"] = np.where(
            output["decision"].eq("KNOWN"),
            output["best_source_identity_id"],
            pd.NA,
        )
        output["provisional_unknown_id"] = np.where(
            output["decision"].eq("UNKNOWN"),
            "UNKNOWN::" + output["sample_id"].astype(str),
            pd.NA,
        )
        output["identification_correct"] = False
        known_truth = output["true_is_enrolled"].fillna(False).astype(bool)

        output.loc[known_truth, "identification_correct"] = (
            output.loc[known_truth, "decision"].eq("KNOWN")
            & output.loc[known_truth, "predicted_person_id"]
                .eq(output.loc[known_truth, "true_person_id"])
                .fillna(False)
        )

        output.loc[~known_truth, "identification_correct"] = (
            output.loc[~known_truth, "decision"].eq("UNKNOWN")
        )

        output["identification_correct"] = (
            output["identification_correct"].fillna(False).astype(bool)
        )
        return output

    # --- Código original de la celda 44 ---
    # ============================================================
    # CELDA 14 - CALIBRAR UMBRAL CON QUERIES NO VISTAS
    # ============================================================

    CALIBRATION_SCORES_CHECKPOINT = (
        CHECKPOINT_DIR / "06_calibration_query_profile_scores.csv"
    )
    CALIBRATION_PREDICTIONS_CHECKPOINT = (
        CHECKPOINT_DIR / "06_calibration_top1_predictions.csv"
    )
    THRESHOLD_CHECKPOINT = VOICEPRINT_DIR / "voiceprint_thresholds.json"

    calibration_scores = load_dataframe_checkpoint(CALIBRATION_SCORES_CHECKPOINT)
    calibration_predictions = load_dataframe_checkpoint(
        CALIBRATION_PREDICTIONS_CHECKPOINT
    )

    if calibration_scores is None or calibration_predictions is None:
        calibration_queries = get_samples_for_split(
            df_samples_open_set,
            df_open_set_split,
            "calibration_known",
            "query",
        )

        calibration_scores, calibration_predictions = (
            score_queries_against_profiles(
                calibration_queries,
                calibration_profiles,
                emb_cols,
                "calibration_known",
            )
        )

        save_dataframe_checkpoint(
            calibration_scores,
            CALIBRATION_SCORES_CHECKPOINT,
        )
        save_dataframe_checkpoint(
            calibration_predictions,
            CALIBRATION_PREDICTIONS_CHECKPOINT,
        )

    if FORCE_REBUILD or not ensure_local_checkpoint(THRESHOLD_CHECKPOINT):
        open_set_threshold_info = choose_verification_threshold(
            calibration_scores,
            OPEN_SET_THRESHOLD_STRATEGY,
        )
        open_set_threshold_info.update({
            "created_at_utc": utc_now_iso(),
            "embedding_model": EMBEDDING_MODEL_LABEL,
            "embedding_dim": len(emb_cols),
            "notebook_version": NOTEBOOK_VERSION,
            "calibration_identity_count": int(
                calibration_profiles["profile_id"].nunique()
            ),
        })
        save_json_checkpoint(open_set_threshold_info, THRESHOLD_CHECKPOINT)
    else:
        with open(THRESHOLD_CHECKPOINT, "r", encoding="utf-8") as file:
            open_set_threshold_info = json.load(file)

    OPEN_SET_THRESHOLD = float(open_set_threshold_info["acceptance_threshold"])

    print(
        "Umbral:",
        round(OPEN_SET_THRESHOLD, 4),
        "| estrategia:",
        open_set_threshold_info["strategy"],
        "| EER:",
        round(float(open_set_threshold_info["eer"]), 4),
        "| AUC:",
        round(float(open_set_threshold_info["calibration_auc"]), 4),
    )

    # --- Código original de la celda 45 ---
    # ============================================================
    # CELDA 15 - TEST KNOWN Y TEST UNKNOWN
    # ============================================================

    TEST_KNOWN_SCORES_CHECKPOINT = (
        CHECKPOINT_DIR / "07_test_known_query_profile_scores.csv"
    )
    TEST_KNOWN_PREDICTIONS_CHECKPOINT = (
        CHECKPOINT_DIR / "07_test_known_top1_predictions.csv"
    )
    TEST_UNKNOWN_SCORES_CHECKPOINT = (
        CHECKPOINT_DIR / "07_test_unknown_query_profile_scores.csv"
    )
    TEST_UNKNOWN_PREDICTIONS_CHECKPOINT = (
        CHECKPOINT_DIR / "07_test_unknown_top1_predictions.csv"
    )

    test_known_scores = load_dataframe_checkpoint(TEST_KNOWN_SCORES_CHECKPOINT)
    test_known_predictions = load_dataframe_checkpoint(
        TEST_KNOWN_PREDICTIONS_CHECKPOINT
    )
    test_unknown_scores = load_dataframe_checkpoint(
        TEST_UNKNOWN_SCORES_CHECKPOINT
    )
    test_unknown_predictions = load_dataframe_checkpoint(
        TEST_UNKNOWN_PREDICTIONS_CHECKPOINT
    )

    if test_known_scores is None or test_known_predictions is None:
        test_known_queries = get_samples_for_split(
            df_samples_open_set,
            df_open_set_split,
            "test_known",
            "query",
        )
        test_known_scores, test_known_predictions = (
            score_queries_against_profiles(
                test_known_queries,
                test_profiles,
                emb_cols,
                "test_known",
            )
        )
        save_dataframe_checkpoint(
            test_known_scores,
            TEST_KNOWN_SCORES_CHECKPOINT,
        )
        save_dataframe_checkpoint(
            test_known_predictions,
            TEST_KNOWN_PREDICTIONS_CHECKPOINT,
        )

    if test_unknown_scores is None or test_unknown_predictions is None:
        test_unknown_queries = get_samples_for_split(
            df_samples_open_set,
            df_open_set_split,
            "test_unknown",
            "query",
        )
        test_unknown_scores, test_unknown_predictions = (
            score_queries_against_profiles(
                test_unknown_queries,
                test_profiles,
                emb_cols,
                "test_unknown",
            )
        )
        save_dataframe_checkpoint(
            test_unknown_scores,
            TEST_UNKNOWN_SCORES_CHECKPOINT,
        )
        save_dataframe_checkpoint(
            test_unknown_predictions,
            TEST_UNKNOWN_PREDICTIONS_CHECKPOINT,
        )

    df_open_set_predictions = pd.concat(
        [
            apply_open_set_decision(
                test_known_predictions,
                OPEN_SET_THRESHOLD,
            ),
            apply_open_set_decision(
                test_unknown_predictions,
                OPEN_SET_THRESHOLD,
            ),
        ],
        ignore_index=True,
    )

    OPEN_SET_PREDICTIONS_PATH = (
        VOICEPRINT_DIR / "open_set_identification_predictions.csv"
    )
    save_dataframe_checkpoint(
        df_open_set_predictions,
        OPEN_SET_PREDICTIONS_PATH,
    )

    display(
        df_open_set_predictions[[
            "query_group",
            "audio_key",
            "true_source_identity_id",
            "best_profile_id",
            "best_source_identity_id",
            "best_similarity",
            "top1_top2_margin",
            "decision",
            "identification_correct",
        ]].head(20)
    )

    # --- Código original de la celda 46 ---
    # ============================================================
    # CELDA 16 - MÉTRICAS
    # ============================================================

    def safe_ratio(numerator, denominator):
        return float(numerator / denominator) if denominator else np.nan


    def evaluate_score_table(score_table, label, threshold):
        if score_table.empty:
            return {"dataset": label, "n_scores": 0}

        y_true = score_table["same_identity"].astype(int).to_numpy()
        scores = score_table["similarity"].astype(float).to_numpy()
        y_pred = scores >= threshold

        if len(np.unique(y_true)) == 2:
            auc = float(roc_auc_score(y_true, scores))
            fpr, tpr, _ = roc_curve(y_true, scores)
            fnr = 1 - tpr
            eer_idx = int(np.nanargmin(np.abs(fpr - fnr)))
            eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2)
        else:
            auc = np.nan
            eer = np.nan

        return {
            "dataset": label,
            "n_scores": int(len(score_table)),
            "n_genuine": int((y_true == 1).sum()),
            "n_impostor": int((y_true == 0).sum()),
            "threshold": float(threshold),
            "auc": auc,
            "eer": eer,
            "verification_accuracy": float(accuracy_score(y_true, y_pred)),
            "verification_precision": float(
                precision_score(y_true, y_pred, zero_division=0)
            ),
            "verification_recall": float(
                recall_score(y_true, y_pred, zero_division=0)
            ),
            "verification_f1": float(
                f1_score(y_true, y_pred, zero_division=0)
            ),
        }


    known_rows = df_open_set_predictions[
        df_open_set_predictions["true_is_enrolled"].astype(bool)
    ].copy()

    unknown_rows = df_open_set_predictions[
        ~df_open_set_predictions["true_is_enrolled"].astype(bool)
    ].copy()

    identification_metrics = {
        "created_at_utc": utc_now_iso(),
        "threshold": OPEN_SET_THRESHOLD,
        "n_known_queries": int(len(known_rows)),
        "n_unknown_queries": int(len(unknown_rows)),
        "known_top1_accuracy_before_rejection": safe_ratio(
            int(known_rows["top1_correct_before_threshold"].sum()),
            len(known_rows),
        ),
        "known_identification_rate": safe_ratio(
            int(known_rows["identification_correct"].sum()),
            len(known_rows),
        ),
        "known_false_rejection_rate": safe_ratio(
            int(known_rows["decision"].eq("UNKNOWN").sum()),
            len(known_rows),
        ),
        "unknown_rejection_rate": safe_ratio(
            int(unknown_rows["decision"].eq("UNKNOWN").sum()),
            len(unknown_rows),
        ),
        "unknown_false_acceptance_rate": safe_ratio(
            int(unknown_rows["decision"].eq("KNOWN").sum()),
            len(unknown_rows),
        ),
        "overall_open_set_accuracy": safe_ratio(
            int(df_open_set_predictions["identification_correct"].sum()),
            len(df_open_set_predictions),
        ),
    }

    verification_metrics = pd.DataFrame([
        evaluate_score_table(
            calibration_scores,
            "calibration_known",
            OPEN_SET_THRESHOLD,
        ),
        evaluate_score_table(
            test_known_scores,
            "test_known",
            OPEN_SET_THRESHOLD,
        ),
    ])

    IDENTIFICATION_METRICS_PATH = (
        VOICEPRINT_DIR / "open_set_identification_metrics.json"
    )
    VERIFICATION_METRICS_PATH = (
        VOICEPRINT_DIR / "voiceprint_verification_metrics.csv"
    )

    save_json_checkpoint(
        identification_metrics,
        IDENTIFICATION_METRICS_PATH,
    )
    save_dataframe_checkpoint(
        verification_metrics,
        VERIFICATION_METRICS_PATH,
    )

    display(verification_metrics)
    display(pd.DataFrame([identification_metrics]))

    # --- Código original de la celda 47 ---
    # ============================================================
    # CELDA 17 - MATRIZ DE DECISIÓN KNOWN / UNKNOWN
    # ============================================================

    decision_true = np.where(
        df_open_set_predictions["true_is_enrolled"].astype(bool),
        "KNOWN",
        "UNKNOWN",
    )
    decision_pred = df_open_set_predictions["decision"].astype(str)

    decision_matrix = confusion_matrix(
        decision_true,
        decision_pred,
        labels=["KNOWN", "UNKNOWN"],
    )

    df_decision_matrix = pd.DataFrame(
        decision_matrix,
        index=["true_known", "true_unknown"],
        columns=["pred_known", "pred_unknown"],
    ).reset_index(names="true_class")

    DECISION_MATRIX_PATH = (
        VOICEPRINT_DIR / "open_set_decision_confusion_matrix.csv"
    )
    save_dataframe_checkpoint(
        df_decision_matrix,
        DECISION_MATRIX_PATH,
    )

    display(df_decision_matrix)

    # --- Código original de la celda 49 ---
    # ============================================================
    # CELDA 18 - BASE OPERACIONAL
    # ============================================================

    AGENT_PROFILES_OPERATIONAL_PATH = (
        VOICEPRINT_DIR / "agent_profiles_operational.csv"
    )
    CLIENT_PROFILES_OPERATIONAL_PATH = (
        VOICEPRINT_DIR / "client_profiles_operational.csv"
    )

    agent_profiles_operational = load_dataframe_checkpoint(
        AGENT_PROFILES_OPERATIONAL_PATH
    )

    if agent_profiles_operational is None:
        eligible_agent_ids = set(
            identity_summary_open_set[
                identity_summary_open_set["role_proxy"].eq("AGENT")
                & identity_summary_open_set["eligible_profile"].astype(bool)
            ]["person_id"].astype(str)
        )

        agent_samples_operational = df_samples_open_set[
            df_samples_open_set["person_id"].astype(str).isin(eligible_agent_ids)
        ].copy()

        agent_profiles_operational = build_speaker_profiles(
            agent_samples_operational,
            emb_cols,
            "operational_all_available_calls",
        )

        save_dataframe_checkpoint(
            agent_profiles_operational,
            AGENT_PROFILES_OPERATIONAL_PATH,
        )

    client_profiles_operational = pd.DataFrame()

    if BUILD_CLIENT_PROFILES:
        client_profiles_operational = load_dataframe_checkpoint(
            CLIENT_PROFILES_OPERATIONAL_PATH
        )

        if client_profiles_operational is None:
            eligible_client_ids = set(
                identity_summary_open_set[
                    identity_summary_open_set["role_proxy"].eq("CLIENT")
                    & identity_summary_open_set["eligible_profile"].astype(bool)
                ]["person_id"].astype(str)
            )

            client_samples_operational = df_samples_open_set[
                df_samples_open_set["person_id"].astype(str).isin(eligible_client_ids)
            ].copy()

            client_profiles_operational = build_speaker_profiles(
                client_samples_operational,
                emb_cols,
                "operational_all_available_calls",
            )

            save_dataframe_checkpoint(
                client_profiles_operational,
                CLIENT_PROFILES_OPERATIONAL_PATH,
            )

    print("Perfiles operacionales de agentes:", len(agent_profiles_operational))
    print("Perfiles operacionales de clientes:", len(client_profiles_operational))

    # --- Código original de la celda 50 ---
    # ============================================================
    # CELDA 19 - METADATA DEL MODELO DE HUELLA
    # ============================================================

    config_snapshot = {
        "notebook_version": NOTEBOOK_VERSION,
        "embedding_model": EMBEDDING_MODEL_LABEL,
        "embedding_dim": len(emb_cols),
        "segment_filters": {
            "min_duration_sec": MIN_SEGMENT_DURATION_SEC,
            "max_duration_sec": MAX_SEGMENT_DURATION_SEC,
            "max_overlap_ratio": MAX_OVERLAP_RATIO,
            "min_rms_dbfs": MIN_RMS_DBFS,
            "min_words": MIN_WORDS_PER_SEGMENT,
        },
        "sample_filters": {
            "min_segments_per_audio_person": MIN_SEGMENTS_PER_AUDIO_PERSON,
            "min_seconds_per_audio_person": MIN_SECONDS_PER_AUDIO_PERSON,
        },
        "profile_filters": {
            "min_samples_per_identity": OPEN_SET_MIN_SAMPLES_PER_IDENTITY,
            "min_total_seconds_per_identity": OPEN_SET_MIN_TOTAL_SECONDS_PER_IDENTITY,
        },
        "threshold": open_set_threshold_info,
    }

    model_metadata = {
        "created_at_utc": utc_now_iso(),
        "notebook_version": NOTEBOOK_VERSION,
        "config_hash": stable_hash_dict(config_snapshot),
        "embedding_source_path": str(SEGMENT_EMBEDDINGS_PATH),
        "role_mapping_source_path": str(ROLE_MAPPING_PATH),
        "embedding_model": EMBEDDING_MODEL_LABEL,
        "embedding_dim": len(emb_cols),
        "embedding_columns": emb_cols,
        "distance_metric": "cosine_similarity",
        "acceptance_threshold": OPEN_SET_THRESHOLD,
        "threshold_strategy": open_set_threshold_info["strategy"],
        "agent_profiles_path": str(AGENT_PROFILES_OPERATIONAL_PATH),
        "client_profiles_path": (
            str(CLIENT_PROFILES_OPERATIONAL_PATH)
            if BUILD_CLIENT_PROFILES
            else None
        ),
        "n_agent_profiles": int(len(agent_profiles_operational)),
        "n_client_profiles": int(len(client_profiles_operational)),
        "identification_output": {
            "known": "source_identity_id (agent_hash o customer_hash)",
            "unknown": None,
            "provisional_unknown_id": "UNKNOWN::<sample_id>",
        },
        "important_note": (
            "UNKNOWN no debe incorporarse automáticamente a la base oficial. "
            "Requiere confirmación externa o varias observaciones consistentes."
        ),
    }

    MODEL_METADATA_PATH = (
        VOICEPRINT_DIR / "voiceprint_model_metadata.json"
    )
    save_json_checkpoint(
        model_metadata,
        MODEL_METADATA_PATH,
    )

    display(pd.DataFrame([{
        "config_hash": model_metadata["config_hash"],
        "embedding_dim": model_metadata["embedding_dim"],
        "threshold": model_metadata["acceptance_threshold"],
        "agent_profiles": model_metadata["n_agent_profiles"],
        "client_profiles": model_metadata["n_client_profiles"],
    }]))

    # --- Código original de la celda 51 ---
    # ============================================================
    # CELDA 20 - CONTRATO DE INFERENCIA
    # ============================================================

    def identify_query_samples(
        query_samples,
        profiles_path=AGENT_PROFILES_OPERATIONAL_PATH,
        threshold_path=THRESHOLD_CHECKPOINT,
    ):
        profiles = pd.read_csv(profiles_path)

        with open(threshold_path, "r", encoding="utf-8") as file:
            threshold_payload = json.load(file)

        threshold = float(threshold_payload["acceptance_threshold"])

        query_emb_cols = get_embedding_columns(query_samples)
        profile_emb_cols = get_embedding_columns(profiles)

        if query_emb_cols != profile_emb_cols:
            raise ValueError(
                "Las dimensiones o nombres de embedding de la query "
                "no coinciden con los perfiles guardados."
            )

        inference_queries = query_samples.copy()

        if "person_id" not in inference_queries.columns:
            inference_queries["person_id"] = "UNLABELED"
        if "source_identity_id" not in inference_queries.columns:
            inference_queries["source_identity_id"] = pd.NA

        _, predictions = score_queries_against_profiles(
            inference_queries,
            profiles,
            profile_emb_cols,
            "external_inference",
        )

        return apply_open_set_decision(predictions, threshold)


    # Prueba del contrato con una muestra existente.
    # Esta celda verifica formato y dimensiones, no sustituye la evaluación formal.
    demo_query = df_samples_open_set[df_samples_open_set["role_proxy"].eq("AGENT")].head(1).copy()
    demo_result = identify_query_samples(demo_query)

    display(
        demo_result[[
            "sample_id",
            "best_profile_id",
            "best_source_identity_id",
            "best_similarity",
            "top1_top2_margin",
            "decision",
        ]]
    )

    # --- Código original de la celda 53 ---
    # ============================================================
    # CELDA 21 - DISTRIBUCIÓN DE CALIBRACIÓN
    # ============================================================

    fig, ax = plt.subplots(figsize=(8, 5))

    genuine = calibration_scores[
        calibration_scores["same_identity"].eq(1)
    ]["similarity"]
    impostor = calibration_scores[
        calibration_scores["same_identity"].eq(0)
    ]["similarity"]

    ax.hist(impostor, bins=40, alpha=0.60, label="Impostor")
    ax.hist(genuine, bins=40, alpha=0.60, label="Genuine")
    ax.axvline(
        OPEN_SET_THRESHOLD,
        linestyle="--",
        linewidth=2,
        label=f"Umbral = {OPEN_SET_THRESHOLD:.3f}",
    )
    ax.set_title("Calibración de huella de voz")
    ax.set_xlabel("Similitud coseno")
    ax.set_ylabel("Número de comparaciones")
    ax.legend()
    ax.grid(True, alpha=0.3)

    save_figure_checkpoint(
        fig,
        FIGURES_DIR / "voiceprint_calibration_similarity_distribution.png",
    )
    plt.show()

    # --- Código original de la celda 54 ---
    # ============================================================
    # CELDA 22 - KNOWN VS UNKNOWN
    # ============================================================

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(
        known_rows["best_similarity"].dropna(),
        bins=30,
        alpha=0.60,
        label="Queries conocidas",
    )
    ax.hist(
        unknown_rows["best_similarity"].dropna(),
        bins=30,
        alpha=0.60,
        label="Queries unknown",
    )
    ax.axvline(
        OPEN_SET_THRESHOLD,
        linestyle="--",
        linewidth=2,
        label=f"Umbral = {OPEN_SET_THRESHOLD:.3f}",
    )
    ax.set_title("Decisión open-set sobre el mejor perfil")
    ax.set_xlabel("Mejor similitud encontrada")
    ax.set_ylabel("Número de queries")
    ax.legend()
    ax.grid(True, alpha=0.3)

    save_figure_checkpoint(
        fig,
        FIGURES_DIR / "open_set_best_similarity_known_vs_unknown.png",
    )
    plt.show()

    # --- Código original de la celda 55 ---
    # ============================================================
    # CELDA 23 - CURVA ROC
    # ============================================================

    y_true_calibration = calibration_scores["same_identity"].astype(int).to_numpy()
    score_values_calibration = calibration_scores["similarity"].astype(float).to_numpy()

    fpr, tpr, _ = roc_curve(y_true_calibration, score_values_calibration)
    auc_value = roc_auc_score(y_true_calibration, score_values_calibration)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUC = {auc_value:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Azar")
    ax.set_title("ROC de verificación vocal")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    ax.grid(True, alpha=0.3)

    save_figure_checkpoint(
        fig,
        FIGURES_DIR / "voiceprint_calibration_roc.png",
    )
    plt.show()

    # --- Código original de la celda 56 ---
    # ============================================================
    # CELDA 24 - RESUMEN Y MANIFEST DE FINALIZACIÓN
    # ============================================================

    summary_rows = [
        {"metric": "segmentos_candidatos", "value": len(df_voiceprint_segments)},
        {"metric": "muestras_audio_persona", "value": len(df_samples_open_set)},
        {
            "metric": "agentes_elegibles",
            "value": int(
                identity_summary_open_set[
                    identity_summary_open_set["role_proxy"].eq("AGENT")
                    & identity_summary_open_set["eligible_profile"].astype(bool)
                ]["person_id"].nunique()
            ),
        },
        {
            "metric": "clientes_elegibles",
            "value": int(
                identity_summary_open_set[
                    identity_summary_open_set["role_proxy"].eq("CLIENT")
                    & identity_summary_open_set["eligible_profile"].astype(bool)
                ]["person_id"].nunique()
            ),
        },
        {"metric": "umbral_aceptacion", "value": OPEN_SET_THRESHOLD},
        {"metric": "auc_calibracion", "value": open_set_threshold_info["calibration_auc"]},
        {"metric": "eer_calibracion", "value": open_set_threshold_info["eer"]},
        {
            "metric": "known_identification_rate",
            "value": identification_metrics["known_identification_rate"],
        },
        {
            "metric": "unknown_rejection_rate",
            "value": identification_metrics["unknown_rejection_rate"],
        },
        {
            "metric": "open_set_accuracy",
            "value": identification_metrics["overall_open_set_accuracy"],
        },
        {
            "metric": "perfiles_operacionales_agentes",
            "value": len(agent_profiles_operational),
        },
        {
            "metric": "perfiles_operacionales_clientes",
            "value": len(client_profiles_operational),
        },
    ]

    df_open_set_final_summary = pd.DataFrame(summary_rows)

    FINAL_SUMMARY_PATH = (
        VOICEPRINT_DIR / "voiceprint_open_set_final_summary.csv"
    )
    save_dataframe_checkpoint(
        df_open_set_final_summary,
        FINAL_SUMMARY_PATH,
    )

    required_final_outputs = [
        AGENT_PROFILES_OPERATIONAL_PATH,
        THRESHOLD_CHECKPOINT,
        MODEL_METADATA_PATH,
        OPEN_SET_PREDICTIONS_PATH,
        IDENTIFICATION_METRICS_PATH,
        VERIFICATION_METRICS_PATH,
    ]

    missing_final_outputs = [
        str(path)
        for path in required_final_outputs
        if not Path(path).exists()
    ]

    run_manifest = {
        "status": "completed" if not missing_final_outputs else "incomplete",
        "completed_at_utc": utc_now_iso(),
        "notebook_version": NOTEBOOK_VERSION,
        "config_hash": model_metadata["config_hash"],
        "required_outputs": [str(path) for path in required_final_outputs],
        "missing_outputs": missing_final_outputs,
        "summary": {
            row["metric"]: row["value"]
            for row in summary_rows
        },
    }

    RUN_MANIFEST_PATH = (
        VOICEPRINT_DIR / "_SUCCESS_voiceprint_open_set.json"
    )
    save_json_checkpoint(
        run_manifest,
        RUN_MANIFEST_PATH,
    )

    display(df_open_set_final_summary)

    print("\nArchivos principales para el notebook de inferencia:")
    print("1.", AGENT_PROFILES_OPERATIONAL_PATH)
    print("2.", THRESHOLD_CHECKPOINT)
    print("3.", MODEL_METADATA_PATH)
    print("4.", RUN_MANIFEST_PATH)


    upload_directory(
        local_dir=VOICEPRINT_DIR,
        gcs_prefix=GCS_VOICEPRINT_PREFIX,
        gcs_client=gcs_client,
        skip_unchanged=True,
    )

    result = {"reused": False}
    if VOICEPRINT_FINAL_SUMMARY_CSV.exists():
        result["final_summary"] = pd.read_csv(VOICEPRINT_FINAL_SUMMARY_CSV)
    if VOICEPRINT_OPEN_SET_SUMMARY_CSV.exists():
        result["open_set_summary"] = pd.read_csv(VOICEPRINT_OPEN_SET_SUMMARY_CSV)
    if VOICEPRINT_SUCCESS_JSON.exists():
        result["manifest"] = json.loads(VOICEPRINT_SUCCESS_JSON.read_text(encoding="utf-8"))
    return result
