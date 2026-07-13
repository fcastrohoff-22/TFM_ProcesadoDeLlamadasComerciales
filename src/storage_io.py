"""Transporte genérico de archivos entre el disco local y el almacenamiento remoto.

Este módulo concentra toda la mecánica de subida y descarga. No contiene
lógica de ninguna fase concreta del pipeline (diarización, transcripción,
etc.): recibe siempre las rutas y el cliente por parámetro.

Diseño por esquema de URI
-------------------------
Las funciones públicas (``download_to_local``, ``upload_from_local``, ...)
inspeccionan el prefijo del destino y despachan a la implementación
correcta. Hoy solo está implementado el backend de Google Cloud Storage
(``gs://``). El día que un dataset viva en disco local, solo hay que
completar las ramas correspondientes de este módulo: los notebooks y el
resto de módulos no cambian, porque siempre llaman a la interfaz pública.
"""

import base64
import hashlib
from pathlib import Path

from src.config import split_gcs_uri


# ============================================================
# UTILIDADES INTERNAS
# ============================================================


def _is_gcs_uri(uri: str) -> bool:
    return str(uri).startswith("gs://")




def join_gcs_uri(prefix: str, relative_path: str) -> str:
    """Une un prefijo GCS y una ruta relativa sin duplicar barras."""
    return f"{str(prefix).rstrip('/')}/{str(relative_path).lstrip('/')}"


def delete_gcs_prefix(gcs_prefix: str, gcs_client) -> int:
    """Elimina todos los objetos contenidos bajo un prefijo exacto."""
    bucket_name, prefix = split_gcs_uri(gcs_prefix)
    blobs = list(gcs_client.list_blobs(bucket_name, prefix=prefix))
    for blob in blobs:
        blob.delete()
    return len(blobs)


def delete_gcs_uri(gcs_uri: str, gcs_client) -> bool:
    """Elimina un objeto GCS concreto cuando existe."""
    bucket_name, blob_path = split_gcs_uri(gcs_uri)
    blob = gcs_client.bucket(bucket_name).blob(blob_path)
    if blob.exists():
        blob.delete()
        return True
    return False


def _file_md5_base64(file_path: Path):
    """Calcula el MD5 local en el formato utilizado por GCS."""
    digest = hashlib.md5()

    with file_path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return base64.b64encode(
        digest.digest()
    ).decode("ascii")


def _local_matches_blob(local_path: Path, blob) -> bool:
    """
    Indica si el archivo local coincide con el blob remoto.

    Compara primero el tamaño y, si GCS expone el hash MD5, también
    el contenido. Si no hay hash disponible, se conforma con el tamaño.
    """
    if (
        not local_path.exists()
        or local_path.stat().st_size != blob.size
    ):
        return False

    if blob.md5_hash:
        return _file_md5_base64(local_path) == blob.md5_hash

    return True


# ============================================================
# TRANSPORTE DE UN ÚNICO ARCHIVO
# ============================================================


def blob_for_local_path(
    local_path: Path,
    gcs_client,
    gcs_prefix: str,
    base_dir: Path,
):
    """
    Devuelve el blob de GCS que corresponde a un archivo local,
    conservando su ruta relativa respecto a ``base_dir``.
    """
    bucket_name, prefix = split_gcs_uri(gcs_prefix)
    relative_path = local_path.relative_to(base_dir).as_posix()
    blob_path = f"{prefix}{relative_path}"
    bucket_obj = gcs_client.bucket(bucket_name)

    return bucket_obj.blob(blob_path), bucket_name, blob_path


def upload_file(
    local_path: Path,
    gcs_client,
    gcs_prefix: str,
    base_dir: Path,
    skip_unchanged: bool = True,
):
    """
    Sube un único archivo a GCS conservando su ruta relativa a ``base_dir``.

    No sube archivos inexistentes ni vacíos. Cuando ``skip_unchanged=True``,
    compara tamaño y checksum y evita transferir un archivo idéntico.
    Devuelve ``True`` solo cuando se realizó una subida.
    """
    local_path = Path(local_path)

    if not local_path.exists() or local_path.stat().st_size == 0:
        return False

    blob, _, _ = blob_for_local_path(
        local_path,
        gcs_client,
        gcs_prefix=gcs_prefix,
        base_dir=base_dir,
    )

    if skip_unchanged and blob.exists():
        blob.reload()
        if _local_matches_blob(local_path, blob):
            return False

    blob.upload_from_filename(str(local_path))
    return True


def download_file_if_exists(
    local_path: Path,
    gcs_client,
    gcs_prefix: str,
    base_dir: Path,
):
    """
    Descarga un archivo desde GCS si existe y no está vacío.

    Escribe primero en un archivo temporal y solo reemplaza el destino
    final si la descarga tiene contenido. Devuelve True si se descargó.
    """
    blob, _, _ = blob_for_local_path(
        local_path,
        gcs_client,
        gcs_prefix=gcs_prefix,
        base_dir=base_dir,
    )

    if blob.exists():
        # No se descargan blobs vacíos porque son checkpoints corruptos.
        if getattr(blob, "size", None) == 0:
            return False

        local_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = local_path.with_suffix(
            local_path.suffix + ".download"
        )
        blob.download_to_filename(str(temporary_path))

        if (
            temporary_path.exists()
            and temporary_path.stat().st_size > 0
        ):
            temporary_path.replace(local_path)
            return True

        if temporary_path.exists():
            temporary_path.unlink()

    return False



def download_uri_to_local(
    source_uri: str,
    local_path: Path,
    gcs_client,
    force: bool = False,
):
    """
    Descarga un archivo concreto desde una URI ``gs://``.

    No vuelve a descargarlo cuando el archivo local coincide con el blob
    remoto, salvo que ``force=True``. Escribe primero en un temporal.
    """
    if not _is_gcs_uri(source_uri):
        raise NotImplementedError(
            "Actualmente solo se admite un origen gs://."
        )

    bucket_name, blob_path = split_gcs_uri(source_uri)
    blob = gcs_client.bucket(bucket_name).blob(blob_path)

    if not blob.exists() or getattr(blob, "size", None) == 0:
        return False

    if not force and _local_matches_blob(local_path, blob):
        return False

    local_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = local_path.with_suffix(
        local_path.suffix + ".download"
    )

    blob.download_to_filename(str(temporary_path))

    if (
        temporary_path.exists()
        and temporary_path.stat().st_size > 0
    ):
        temporary_path.replace(local_path)
        return True

    if temporary_path.exists():
        temporary_path.unlink()

    return False


def ensure_local_file(
    local_path: Path,
    gcs_client,
    gcs_prefix: str,
    base_dir: Path,
    required: bool = True,
):
    """
    Garantiza que un archivo exista localmente.

    Si falta o está vacío, intenta restaurarlo desde GCS. Devuelve ``True``
    cuando el archivo queda disponible. Si ``required=True`` y no aparece,
    lanza ``FileNotFoundError``.
    """
    available = (
        local_path.exists()
        and local_path.stat().st_size > 0
    )

    if not available:
        download_file_if_exists(
            local_path,
            gcs_client,
            gcs_prefix=gcs_prefix,
            base_dir=base_dir,
        )
        available = (
            local_path.exists()
            and local_path.stat().st_size > 0
        )

    if required and not available:
        raise FileNotFoundError(
            f"No se encontró ni localmente ni en GCS: {local_path}"
        )

    return available


# ============================================================
# TRANSPORTE DE CARPETAS COMPLETAS
# ============================================================


def upload_directory(
    local_dir: Path,
    gcs_prefix: str,
    gcs_client,
    clear_output_fn=None,
    skip_unchanged: bool = True,
):
    """
    Sube recursivamente una carpeta local a GCS.

    Cuando skip_unchanged=True, no vuelve a subir los archivos que ya
    existen en GCS con el mismo tamaño y checksum.
    """
    bucket_name, prefix = split_gcs_uri(gcs_prefix)
    bucket = gcs_client.bucket(bucket_name)

    files_to_upload = sorted(
        path
        for path in local_dir.rglob("*")
        if path.is_file()
    )

    remote_blobs = {
        blob.name: blob
        for blob in gcs_client.list_blobs(
            bucket_name,
            prefix=prefix,
        )
        if not blob.name.endswith("/")
    }

    total_files = len(files_to_upload)
    uploaded = 0
    skipped = 0

    for index, local_path in enumerate(
        files_to_upload,
        start=1,
    ):
        relative_path = (
            local_path
            .relative_to(local_dir)
            .as_posix()
        )

        blob_path = f"{prefix}{relative_path}"
        remote_blob = remote_blobs.get(blob_path)

        unchanged = (
            skip_unchanged
            and remote_blob is not None
            and _local_matches_blob(local_path, remote_blob)
        )

        if unchanged:
            skipped += 1
            continue

        if clear_output_fn is not None:
            clear_output_fn(wait=True)

        print(
            f"Subiendo outputs "
            f"{index}/{total_files}: {relative_path}"
        )

        bucket.blob(blob_path).upload_from_filename(
            str(local_path)
        )

        uploaded += 1

    if clear_output_fn is not None:
        clear_output_fn(wait=True)

    print("Subida final completada.")
    print("Archivos locales revisados:", total_files)
    print("Archivos subidos:", uploaded)
    print("Archivos omitidos sin cambios:", skipped)
    print("Destino:", gcs_prefix)

    return {
        "total": total_files,
        "uploaded": uploaded,
        "skipped": skipped,
    }


def download_directory(
    local_dir: Path,
    gcs_prefix: str,
    gcs_client,
    base_dir: Path,
    clear_output_fn=None,
):
    """
    Restaura desde GCS una subcarpeta de ``base_dir``.

    Conserva exactamente la misma estructura de carpetas y nombres.
    Los archivos locales idénticos no se descargan de nuevo.
    """
    bucket_name, prefix = split_gcs_uri(gcs_prefix)

    relative_dir = (
        local_dir
        .relative_to(base_dir)
        .as_posix()
        .strip("/")
    )

    remote_prefix = (
        f"{prefix}{relative_dir}/"
        if relative_dir
        else prefix
    )

    blobs = [
        blob
        for blob in gcs_client.list_blobs(
            bucket_name,
            prefix=remote_prefix,
        )
        if (
            not blob.name.endswith("/")
            and getattr(blob, "size", 0)
        )
    ]

    restored = 0
    skipped = 0
    total_files = len(blobs)

    for index, blob in enumerate(blobs, start=1):
        relative_path = blob.name[len(prefix):]
        local_path = base_dir / relative_path

        if _local_matches_blob(local_path, blob):
            skipped += 1
            continue

        if clear_output_fn is not None:
            clear_output_fn(wait=True)

        print(
            f"Restaurando outputs "
            f"{index}/{total_files}: {relative_path}"
        )

        local_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary_path = local_path.with_name(
            local_path.name + ".download"
        )

        blob.download_to_filename(
            str(temporary_path)
        )

        if (
            temporary_path.exists()
            and temporary_path.stat().st_size > 0
        ):
            temporary_path.replace(local_path)
            restored += 1
        elif temporary_path.exists():
            temporary_path.unlink()

    if clear_output_fn is not None:
        clear_output_fn(wait=True)

    print("Restauración desde GCS completada.")
    print("Archivos encontrados:", total_files)
    print("Archivos descargados:", restored)
    print("Archivos locales ya vigentes:", skipped)

    return {
        "found": total_files,
        "downloaded": restored,
        "skipped": skipped,
    }


# ============================================================
# DESCARGA DE UN PREFIJO DE ORIGEN (audios de entrada, etc.)
# ============================================================


def download_prefix_to_local(
    source_uri: str,
    local_dir: Path,
    gcs_client,
    suffix: str = None,
    force: bool = False,
    clear_output_fn=None,
):
    """
    Descarga a ``local_dir`` todos los archivos bajo ``source_uri``.

    Pensada para traer datos de entrada (por ejemplo, audios limpios).
    Con force=False no vuelve a descargar los que ya existen localmente.
    Filtra por extensión si se indica ``suffix`` (por ejemplo, ".wav").
    """
    local_dir.mkdir(parents=True, exist_ok=True)

    if not _is_gcs_uri(source_uri):
        raise NotImplementedError(
            "Backend local aún no implementado. "
            "Actualmente solo se admite un origen gs://."
        )

    bucket_name, prefix = split_gcs_uri(source_uri)

    blobs = [
        blob
        for blob in gcs_client.list_blobs(bucket_name, prefix=prefix)
        if (
            suffix is None
            or blob.name.lower().endswith(suffix.lower())
        )
    ]

    total = len(blobs)
    downloaded = 0
    skipped = 0

    for index, blob in enumerate(blobs, start=1):
        local_path = local_dir / Path(blob.name).name

        if not force and local_path.exists():
            skipped += 1
            continue

        if clear_output_fn is not None:
            clear_output_fn(wait=True)

        print(
            f"Descargando {index}/{total}: {local_path.name}"
        )

        blob.download_to_filename(str(local_path))
        downloaded += 1

    if clear_output_fn is not None:
        clear_output_fn(wait=True)

    print(
        f"Descarga completada desde: gs://{bucket_name}/{prefix}"
    )
    print("Archivos encontrados:", total)
    print("Archivos descargados:", downloaded)
    print("Archivos ya vigentes localmente:", skipped)
    print("Destino local:", local_dir)

    return {
        "found": total,
        "downloaded": downloaded,
        "skipped": skipped,
    }
