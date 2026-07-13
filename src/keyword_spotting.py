"""Fase 08: keyword spotting y detección de temas críticos.

La lógica científica de esta fase conserva el diccionario, la normalización,
la detección por subcadenas, las agregaciones y el score del notebook original.
El notebook se encarga de la restauración desde GCS, la inspección visual y la
sincronización final.
"""

from __future__ import annotations

from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd

from src.config import (
    CALL_KEYWORDS_SENTIMENT_CSV,
    CALL_LEVEL_KEYWORDS_CSV,
    SEGMENTS_WITH_KEYWORDS_CSV,
    TOP_CRITICAL_CALLS_CSV,
    TRANSCRIPTION_ALL_SEGMENTS_CSV,
    TRANSCRIPTION_PER_AUDIO_DIR,
    ensure_phase08_directories,
)
from src.io_utils import csv_is_usable, read_csv_robust, write_csv_atomic


def normalize_text(text: object) -> str:
    """Normaliza texto como en el notebook original."""
    if text is None or pd.isna(text):
        return ""

    normalized = str(text).lower().strip()
    normalized = unicodedata.normalize("NFKD", normalized)
    normalized = "".join(
        char for char in normalized if not unicodedata.combining(char)
    )
    normalized = re.sub(r"[^\w\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


CRITICAL_KEYWORDS = {
    
    # 1. Riesgo de baja, cancelación o fuga del cliente
    "baja_cancelacion": [
        "baja",
        "dar de baja",
        "darme de baja",
        "darse de baja",
        "quiero la baja",
        "quiero darme de baja",
        "solicitar baja",
        "tramitar baja",
        "cancelar",
        "cancelacion",
        "anular",
        "rescindir",
        "rescision",
        "finalizar contrato",
        "romper contrato",
        "quitar el servicio",
        "quitar la linea",
        "quitar la fibra",
        "quitar internet",
        "no quiero seguir",
        "no quiero continuar",
        "me quiero ir",
        "me voy",
        "irme",
        "cambiar de compañia",
        "cambio de compañia",
        "otra compañia",
        "me cambio",
        "portabilidad",
        "hacer portabilidad",
        "solicitar portabilidad",
        "retencion",
        "departamento de bajas",
        "baja del servicio",
        "baja de la linea",
        "baja de fibra",
        "baja movil",
        "baja internet",
        "baixa",
        "donar de baixa",
        "vull donar-me de baixa",
        "cancel lar",
        "portabilitat"
    ],

    # 2. Facturación, cobros, pagos y disputas económicas
    "facturacion_cobros": [
        "factura",
        "facturas",
        "facturacion",
        "recibo",
        "recibos",
        "cobro",
        "cobros",
        "me han cobrado",
        "me habeis cobrado",
        "me cobraron",
        "cobrado de mas",
        "cobro indebido",
        "cargo",
        "cargos",
        "cargo duplicado",
        "doble cargo",
        "importe",
        "importe incorrecto",
        "cantidad incorrecta",
        "cuota",
        "tarifa",
        "precio",
        "subida de precio",
        "me ha subido",
        "me habeis subido",
        "mas caro",
        "muy caro",
        "descuento no aplicado",
        "promocion no aplicada",
        "oferta no aplicada",
        "no me han aplicado",
        "devolucion",
        "devolver dinero",
        "reembolso",
        "abono",
        "regularizacion",
        "pago",
        "pagos",
        "pagar",
        "pagado",
        "impago",
        "deuda",
        "deudas",
        "pendiente de pago",
        "domiciliacion",
        "domiciliado",
        "cuenta bancaria",
        "iban",
        "tarjeta",
        "datáfono",
        "datofono",
        "fraude en factura",
        "factura incorrecta",
        "factura mal",
        "no entiendo la factura",
        "explicar la factura",
        "detallar la factura",
        "factura més alta",
        "factura mes alta",
        "cobrament",
        "pagament",
        "rebut",
        "devolucio",
        "devolució"
    ],

    # 3. Incidencias técnicas de servicio
    "incidencia_tecnica": [
        "incidencia",
        "incidencias",
        "averia",
        "averias",
        "problema tecnico",
        "fallo tecnico",
        "fallo",
        "no funciona",
        "no me funciona",
        "no va",
        "no tengo servicio",
        "sin servicio",
        "sin internet",
        "no tengo internet",
        "internet no funciona",
        "sin conexion",
        "sin conexión",
        "no conecta",
        "no se conecta",
        "se corta",
        "se me corta",
        "cortes",
        "microcortes",
        "lento",
        "muy lento",
        "lentitud",
        "baja velocidad",
        "velocidad baja",
        "no carga",
        "no navega",
        "wifi",
        "wi fi",
        "wiffi",
        "router",
        "rúter",
        "reiniciar router",
        "fibra",
        "fibra optica",
        "instalacion",
        "instalador",
        "tecnico",
        "tecnica",
        "visita tecnica",
        "cita tecnica",
        "ont",
        "cable",
        "cableado",
        "roseta",
        "linea",
        "sin linea",
        "movil",
        "datos moviles",
        "sin datos",
        "no tengo datos",
        "cobertura",
        "sin cobertura",
        "mala cobertura",
        "llamadas cortadas",
        "no puedo llamar",
        "no puedo recibir llamadas",
        "sms",
        "sim",
        "tarjeta sim",
        "duplicado sim",
        "esim",
        "apn",
        "television",
        "tv",
        "decodificador",
        "desco",
        "canales",
        "no se ve",
        "pantalla negra",
        "avaria",
        "no funciona internet",
        "sense internet",
        "sense servei",
        "cobertura baixa",
        "router no funciona"
    ],

    # 4. Reclamaciones, quejas y escalado formal
    "reclamacion_queja": [
        "reclamacion",
        "reclamaciones",
        "reclamar",
        "quiero reclamar",
        "poner una reclamacion",
        "abrir reclamacion",
        "numero de reclamacion",
        "estado de reclamacion",
        "queja",
        "quejas",
        "quejarme",
        "me quiero quejar",
        "denuncia",
        "denunciar",
        "voy a denunciar",
        "consumo",
        "oficina de consumo",
        "facua",
        "ocu",
        "hoja de reclamaciones",
        "atencion al cliente",
        "mal servicio",
        "servicio pesimo",
        "servicio pésimo",
        "trato recibido",
        "no me solucionan",
        "nadie me soluciona",
        "no me ayudan",
        "nadie me ayuda",
        "he llamado varias veces",
        "llevo llamando",
        "no me hacen caso",
        "supervisor",
        "responsable",
        "pasame con un responsable",
        "pasame con un supervisor",
        "quiero hablar con un responsable",
        "escalar",
        "escalarlo",
        "reclamacio",
        "queixa",
        "atencio al client",
        "full de reclamacions"
    ],

    # 5. Insatisfacción fuerte, enfado o urgencia
    "insatisfaccion_urgencia": [
        "enfadado",
        "enfadada",
        "muy enfadado",
        "muy enfadada",
        "cabreado",
        "cabreada",
        "molesto",
        "molesta",
        "indignado",
        "indignada",
        "cansado",
        "cansada",
        "harto",
        "harta",
        "estoy harto",
        "estoy harta",
        "fatal",
        "horrible",
        "vergüenza",
        "verguenza",
        "es una vergüenza",
        "esto es una vergüenza",
        "no puede ser",
        "no hay derecho",
        "inaceptable",
        "esto es inaceptable",
        "esto es increible",
        "increible",
        "desesperado",
        "desesperada",
        "urgente",
        "urgencia",
        "lo necesito ya",
        "cuanto antes",
        "de inmediato",
        "no puedo mas",
        "no puedo más",
        "estoy desesperado",
        "estoy desesperada",
        "llevo esperando",
        "llevo dias",
        "llevo semanas",
        "llevo meses",
        "otra vez",
        "siempre igual",
        "ya esta bien",
        "ya está bien",
        "no me parece normal",
        "me parece fatal",
        "no tiene sentido",
        "malisimo",
        "malísimo",
        "pésimo",
        "pesimo",
        "nefasto",
        "molt enfadat",
        "molt enfadada",
        "estic fart",
        "estic farta",
        "urgent",
        "vergonya"
    ],

    # 6. Fraude, seguridad, identidad y accesos
    "fraude_seguridad": [
        "fraude",
        "fraudulento",
        "estafa",
        "estafado",
        "estafada",
        "suplantacion",
        "suplantacion de identidad",
        "identidad",
        "robo de identidad",
        "no soy yo",
        "no he sido yo",
        "yo no he contratado",
        "contrato falso",
        "firma falsa",
        "han contratado a mi nombre",
        "sin mi permiso",
        "sin autorizacion",
        "no he autorizado",
        "autorizacion",
        "titular",
        "cambio de titular",
        "titularidad",
        "dni",
        "documento",
        "contraseña",
        "password",
        "clave",
        "codigo de seguridad",
        "codigo sms",
        "sms de seguridad",
        "acceso",
        "no puedo acceder",
        "area cliente",
        "app",
        "mi cuenta",
        "cuenta bloqueada",
        "bloqueado",
        "phishing",
        "correo sospechoso",
        "mensaje sospechoso",
        "seguridad",
        "frau",
        "estafa",
        "suplantacio",
        "identitat",
        "contrasenya",
        "acces"
    ],

    # 7. Retención comercial, ofertas, competencia y permanencia
    "retencion_comercial": [
        "oferta",
        "ofertas",
        "descuento",
        "descuentos",
        "promocion",
        "promociones",
        "me ofrecieron",
        "me ofrecen",
        "me prometieron",
        "prometido",
        "no cumplen",
        "no se ha aplicado",
        "precio final",
        "condiciones",
        "contrato",
        "permanencia",
        "penalizacion",
        "penalizar",
        "penalizacion por baja",
        "compromiso de permanencia",
        "competencia",
        "otra compañia",
        "otra operadora",
        "me sale mas barato",
        "mas barato",
        "más barato",
        "mejor oferta",
        "contraoferta",
        "renovar",
        "renovacion",
        "fidelizacion",
        "cliente antiguo",
        "nuevo cliente",
        "promocio",
        "descompte",
        "permanencia",
        "penalitzacio"
    ],

    # 8. Esperas, derivaciones y experiencia operativa deficiente
    "espera_derivacion": [
        "espera",
        "esperando",
        "en espera",
        "llevo esperando",
        "mucho tiempo esperando",
        "me han dejado esperando",
        "transferir",
        "transferencia",
        "me transfieren",
        "me pasan",
        "me han pasado",
        "pasarme",
        "derivar",
        "derivacion",
        "departamento",
        "otro departamento",
        "me mandan a otro departamento",
        "me pasan de un lado a otro",
        "repetir",
        "tengo que repetir",
        "repetirlo todo",
        "ya lo he explicado",
        "ya he llamado",
        "he llamado varias veces",
        "varias llamadas",
        "nadie sabe",
        "no saben",
        "me cuelgan",
        "se ha cortado la llamada",
        "no me atienden",
        "llamada anterior",
        "esperant",
        "m han passat",
        "departament",
        "he trucat varies vegades"
    ],

    # 9. Instalaciones, citas técnicas y logística de técnicos
    "instalacion_citas": [
        "instalacion",
        "instalar",
        "instalador",
        "tecnico",
        "cita",
        "cita tecnica",
        "visita",
        "visita tecnica",
        "no ha venido",
        "no se ha presentado",
        "me han dejado plantado",
        "reprogramar",
        "cambiar cita",
        "anular cita",
        "confirmar cita",
        "franja horaria",
        "domicilio",
        "direccion",
        "mudanza",
        "traslado",
        "cambio de domicilio",
        "alta de fibra",
        "instal·lacio",
        "tecnic",
        "cita tecnica",
        "canvi de domicili"
    ],

    # 10. Alta, contratación y activación del servicio
    "alta_contratacion": [
        "alta",
        "dar de alta",
        "contratar",
        "contratacion",
        "nuevo contrato",
        "activacion",
        "activar",
        "no activado",
        "pendiente de activar",
        "pedido",
        "estado del pedido",
        "numero de pedido",
        "seguimiento del pedido",
        "router no llega",
        "sim no llega",
        "tarjeta no llega",
        "envio",
        "entrega",
        "mensajeria",
        "alta fibra",
        "alta movil",
        "alta internet",
        "contractacio",
        "activar servei",
        "alta servei"
    ],

    # 11. Datos personales, contrato y cambios administrativos
    "gestion_administrativa": [
        "datos personales",
        "cambiar datos",
        "modificar datos",
        "email",
        "correo",
        "telefono contacto",
        "direccion",
        "domicilio",
        "cambio de titular",
        "titular",
        "titularidad",
        "autorizado",
        "autorizada",
        "representante",
        "dni",
        "nie",
        "cif",
        "iban",
        "cuenta bancaria",
        "contrato",
        "copia del contrato",
        "condiciones del contrato",
        "documentacion",
        "firma",
        "signatura",
        "dades personals",
        "canvi de titular",
        "compte bancari"
    ],

    # 12. Casos de vulnerabilidad o impacto sensible
    "vulnerabilidad_impacto": [
        "persona mayor",
        "mi madre",
        "mi padre",
        "mis padres",
        "mayor",
        "anciano",
        "anciana",
        "dependiente",
        "discapacidad",
        "hospital",
        "medico",
        "médico",
        "teleasistencia",
        "trabajo desde casa",
        "teletrabajo",
        "necesito internet para trabajar",
        "necesito la linea",
        "no puedo trabajar",
        "negocio",
        "empresa",
        "tienda",
        "bar",
        "restaurante",
        "urgente por trabajo",
        "persona gran",
        "discapacitat",
        "hospital",
        "teletreball"
    ]
}

# La normalización y eliminación de duplicados conserva el comportamiento
# original y se realiza una sola vez al importar el módulo.
CRITICAL_KEYWORDS = {
    theme: sorted(
        set(
            normalize_text(keyword)
            for keyword in keywords
            if str(keyword).strip()
        )
    )
    for theme, keywords in CRITICAL_KEYWORDS.items()
}

THEME_NAMES = tuple(CRITICAL_KEYWORDS.keys())


def keyword_output_paths() -> dict[str, Path]:
    """Devuelve los outputs originales de la fase 08."""
    return {
        "segments": SEGMENTS_WITH_KEYWORDS_CSV,
        "calls": CALL_LEVEL_KEYWORDS_CSV,
        "top_calls": TOP_CRITICAL_CALLS_CSV,
        "combined": CALL_KEYWORDS_SENTIMENT_CSV,
    }


def keyword_outputs_complete() -> bool:
    """Comprueba los tres outputs obligatorios originales."""
    return all(
        [
            csv_is_usable(
                SEGMENTS_WITH_KEYWORDS_CSV,
                required_columns=[
                    "audio_file",
                    "text",
                    "has_critical_keyword",
                    "total_keyword_matches",
                ],
            ),
            csv_is_usable(
                CALL_LEVEL_KEYWORDS_CSV,
                required_columns=[
                    "audio_file",
                    "keyword_criticality_score",
                    "keyword_criticality_percentile",
                ],
            ),
            csv_is_usable(
                TOP_CRITICAL_CALLS_CSV,
                required_columns=[
                    "audio_file",
                    "keyword_criticality_score",
                ],
            ),
        ]
    )


def load_keyword_outputs() -> dict[str, pd.DataFrame]:
    """Carga de forma tolerante los outputs existentes de la fase."""
    paths = keyword_output_paths()
    return {
        name: read_csv_robust(path)
        for name, path in paths.items()
        if path.exists() and path.stat().st_size > 0
    }


def _resolve_text_column(df: pd.DataFrame) -> str:
    for column in ("text", "text_whisper", "transcription", "transcript"):
        if column in df.columns:
            return column
    raise KeyError(
        "No se encontró una columna textual. Se esperaba una de: "
        "text, text_whisper, transcription o transcript."
    )


def prepare_transcription_segments(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza el esquema mínimo requerido sin alterar las columnas originales."""
    if df.empty:
        raise ValueError("La tabla de transcripciones está vacía.")

    prepared = df.copy()
    text_column = _resolve_text_column(prepared)
    if text_column != "text":
        prepared["text"] = prepared[text_column]

    required = ["audio_file", "start", "end"]
    missing = [column for column in required if column not in prepared.columns]
    if missing:
        raise KeyError(f"Faltan columnas obligatorias de transcripción: {missing}")

    prepared["text"] = prepared["text"].fillna("").astype(str).str.strip()
    prepared["start"] = pd.to_numeric(prepared["start"], errors="coerce")
    prepared["end"] = pd.to_numeric(prepared["end"], errors="coerce")

    if "duration" not in prepared.columns:
        prepared["duration"] = prepared["end"] - prepared["start"]
    else:
        prepared["duration"] = pd.to_numeric(
            prepared["duration"], errors="coerce"
        )
        missing_duration = prepared["duration"].isna()
        prepared.loc[missing_duration, "duration"] = (
            prepared.loc[missing_duration, "end"]
            - prepared.loc[missing_duration, "start"]
        )

    prepared["n_chars"] = prepared["text"].str.len()
    prepared["n_words"] = prepared["text"].apply(
        lambda value: len(value.split()) if value else 0
    )
    prepared = prepared.sort_values(
        ["audio_file", "start", "end"], na_position="last"
    ).reset_index(drop=True)
    return prepared


def consolidate_transcription_files(
    per_audio_dir: Path = TRANSCRIPTION_PER_AUDIO_DIR,
    consolidated_path: Path = TRANSCRIPTION_ALL_SEGMENTS_CSV,
) -> tuple[pd.DataFrame, list[Path]]:
    """Consolida los CSV por audio exactamente como el notebook original."""
    transcribed_files = sorted(
        Path(per_audio_dir).glob("*_transcribed_segments.csv")
    )
    if not transcribed_files:
        raise FileNotFoundError(
            "No se encontraron archivos *_transcribed_segments.csv en "
            f"{per_audio_dir}."
        )

    frames = []
    for file_path in transcribed_files:
        frame = pd.read_csv(file_path)
        frame["transcription_csv"] = file_path.name
        frames.append(frame)

    consolidated = prepare_transcription_segments(
        pd.concat(frames, ignore_index=True)
    )
    write_csv_atomic(consolidated, Path(consolidated_path))
    return consolidated, transcribed_files


def load_transcription_segments(
    per_audio_dir: Path = TRANSCRIPTION_PER_AUDIO_DIR,
    consolidated_path: Path = TRANSCRIPTION_ALL_SEGMENTS_CSV,
    force_consolidate: bool = False,
) -> tuple[pd.DataFrame, str, int]:
    """Carga el consolidado o lo reconstruye desde los archivos por audio."""
    per_audio_files = sorted(
        Path(per_audio_dir).glob("*_transcribed_segments.csv")
    )

    if force_consolidate or not csv_is_usable(
        Path(consolidated_path),
        required_columns=["audio_file", "start", "end"],
    ):
        consolidated, files = consolidate_transcription_files(
            per_audio_dir=per_audio_dir,
            consolidated_path=consolidated_path,
        )
        return consolidated, "per_audio", len(files)

    consolidated = prepare_transcription_segments(
        pd.read_csv(consolidated_path)
    )
    return consolidated, "consolidated", len(per_audio_files)


def find_matches_by_theme(text_norm: object, keywords: list[str]) -> list[str]:
    """Aplica la búsqueda simple por subcadena utilizada por el original."""
    if text_norm is None or pd.isna(text_norm):
        return []

    text_value = str(text_norm)
    return sorted(
        set(
            keyword
            for keyword in keywords
            if keyword and keyword in text_value
        )
    )


def apply_keyword_spotting(
    df: pd.DataFrame,
    critical_keywords: dict[str, list[str]] = CRITICAL_KEYWORDS,
) -> pd.DataFrame:
    """Añade matches, conteos y flags por tema a nivel de segmento."""
    result = prepare_transcription_segments(df)
    result["text_norm"] = result["text"].apply(normalize_text)

    for theme, keywords in critical_keywords.items():
        matched_col = f"kw_{theme}_matched"
        count_col = f"kw_{theme}_count"
        flag_col = f"kw_{theme}_flag"

        result[matched_col] = result["text_norm"].apply(
            lambda text, values=keywords: find_matches_by_theme(text, values)
        )
        result[count_col] = result[matched_col].apply(len)
        result[flag_col] = result[count_col] > 0

    flag_columns = [
        f"kw_{theme}_flag" for theme in critical_keywords
    ]
    count_columns = [
        f"kw_{theme}_count" for theme in critical_keywords
    ]

    result["n_critical_themes"] = result[flag_columns].sum(axis=1)
    result["total_keyword_matches"] = result[count_columns].sum(axis=1)
    result["has_critical_keyword"] = result["total_keyword_matches"] > 0
    result["critical_themes_detected"] = result.apply(
        lambda row: [
            theme
            for theme in critical_keywords
            if bool(row[f"kw_{theme}_flag"])
        ],
        axis=1,
    )
    return result


def build_theme_summary(
    df_kw: pd.DataFrame,
    critical_keywords: dict[str, list[str]] = CRITICAL_KEYWORDS,
) -> pd.DataFrame:
    """Resume cobertura y coincidencias por tema."""
    rows = []
    for theme in critical_keywords:
        flag_col = f"kw_{theme}_flag"
        count_col = f"kw_{theme}_count"
        rows.append(
            {
                "theme": theme,
                "n_segments": int(df_kw[flag_col].sum()),
                "pct_segments": round(df_kw[flag_col].mean() * 100, 2),
                "total_keyword_matches": int(df_kw[count_col].sum()),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("n_segments", ascending=False)
        .reset_index(drop=True)
    )


def build_call_level_keywords(
    df_kw: pd.DataFrame,
    critical_keywords: dict[str, list[str]] = CRITICAL_KEYWORDS,
) -> pd.DataFrame:
    """Reproduce la agregación y el score original a nivel de llamada."""
    base_call_agg = (
        df_kw.groupby("audio_file")
        .agg(
            n_segments=("text", "count"),
            n_segments_with_text=("n_chars", lambda values: (values > 0).sum()),
            n_segments_with_keywords=("has_critical_keyword", "sum"),
            total_keyword_matches=("total_keyword_matches", "sum"),
            n_theme_hits=("n_critical_themes", "sum"),
            total_duration=("duration", "sum"),
            total_words=("n_words", "sum"),
        )
        .reset_index()
    )

    base_call_agg["pct_segments_with_keywords"] = (
        base_call_agg["n_segments_with_keywords"]
        / base_call_agg["n_segments"]
    ).round(4)

    for theme in critical_keywords:
        flag_col = f"kw_{theme}_flag"
        count_col = f"kw_{theme}_count"
        theme_call = (
            df_kw.groupby("audio_file")
            .agg(
                **{
                    f"{theme}_segments": (flag_col, "sum"),
                    f"{theme}_matches": (count_col, "sum"),
                }
            )
            .reset_index()
        )
        base_call_agg = base_call_agg.merge(
            theme_call, on="audio_file", how="left"
        )

    themes_by_audio = (
        df_kw.groupby("audio_file", sort=False)
        .apply(
            lambda group: [
                theme
                for theme in critical_keywords
                if group[f"kw_{theme}_flag"].any()
            ],
        )
        .rename("critical_themes_detected")
        .reset_index()
    )
    base_call_agg = base_call_agg.merge(
        themes_by_audio, on="audio_file", how="left"
    )
    base_call_agg["critical_themes_detected"] = (
        base_call_agg["critical_themes_detected"].apply(
            lambda value: value if isinstance(value, list) else []
        )
    )
    base_call_agg["n_distinct_critical_themes"] = (
        base_call_agg["critical_themes_detected"].apply(len)
    )

    base_call_agg["keyword_criticality_score"] = (
        base_call_agg["pct_segments_with_keywords"] * 3
        + np.log1p(base_call_agg["total_keyword_matches"])
        + base_call_agg["n_distinct_critical_themes"] * 0.5
    )
    base_call_agg["keyword_criticality_percentile"] = (
        base_call_agg["keyword_criticality_score"].rank(pct=True)
    ).round(4)

    return base_call_agg.sort_values(
        "keyword_criticality_score", ascending=False
    ).reset_index(drop=True)


def select_top_critical_calls(
    call_level_keywords: pd.DataFrame,
    top_n: int = 30,
) -> pd.DataFrame:
    """Selecciona las 30 llamadas originales con mayor score."""
    return call_level_keywords.head(top_n).copy()


def combine_keywords_with_sentiment(
    call_level_keywords: pd.DataFrame,
    call_sentiment: pd.DataFrame,
) -> pd.DataFrame:
    """Conserva la fusión opcional original 60 % keywords / 40 % negatividad."""
    required = {"audio_file", "avg_sentiment"}
    missing = required.difference(call_sentiment.columns)
    if missing:
        raise KeyError(
            "El agregado de sentimiento no contiene las columnas "
            f"requeridas: {sorted(missing)}"
        )

    combined = call_level_keywords.merge(
        call_sentiment, on="audio_file", how="left"
    )
    combined["negative_sentiment_component"] = (
        -pd.to_numeric(combined["avg_sentiment"], errors="coerce").fillna(0)
    ).clip(lower=0)
    combined["combined_criticality_score"] = (
        combined["keyword_criticality_percentile"] * 0.6
        + combined["negative_sentiment_component"] * 0.4
    )
    return combined.sort_values(
        "combined_criticality_score", ascending=False
    ).reset_index(drop=True)


def build_group_theme_summary(
    df_kw: pd.DataFrame,
    group_column: str,
    critical_keywords: dict[str, list[str]] = CRITICAL_KEYWORDS,
) -> pd.DataFrame:
    """Construye una tabla interpretable por rol o corpus sin crear outputs nuevos."""
    if group_column not in df_kw.columns:
        return pd.DataFrame()

    rows = []
    grouped = df_kw.dropna(subset=[group_column]).groupby(group_column)
    for group_value, group in grouped:
        for theme in critical_keywords:
            flag_col = f"kw_{theme}_flag"
            rows.append(
                {
                    group_column: group_value,
                    "theme": theme,
                    "n_segments": len(group),
                    "n_segments_with_theme": int(group[flag_col].sum()),
                    "pct_segments_with_theme": round(
                        group[flag_col].mean() * 100, 2
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_final_summary(
    df_kw: pd.DataFrame,
    call_level_keywords: pd.DataFrame,
    theme_summary: pd.DataFrame,
) -> dict[str, object]:
    """Genera el mismo resumen final del notebook original."""
    return {
        "n_audios_analyzed": int(df_kw["audio_file"].nunique()),
        "n_segments_analyzed": int(len(df_kw)),
        "n_segments_with_text": int((df_kw["n_chars"] > 0).sum()),
        "n_segments_with_keywords": int(df_kw["has_critical_keyword"].sum()),
        "pct_segments_with_keywords": round(
            df_kw["has_critical_keyword"].mean() * 100, 2
        ),
        "n_calls_with_keywords": int(
            (call_level_keywords["n_segments_with_keywords"] > 0).sum()
        ),
        "pct_calls_with_keywords": round(
            (
                call_level_keywords["n_segments_with_keywords"] > 0
            ).mean()
            * 100,
            2,
        ),
        "most_frequent_theme": (
            theme_summary.iloc[0]["theme"]
            if not theme_summary.empty
            else None
        ),
    }


def save_keyword_outputs(
    df_kw: pd.DataFrame,
    call_level_keywords: pd.DataFrame,
    top_critical_calls: pd.DataFrame,
    combined: pd.DataFrame | None = None,
) -> dict[str, Path]:
    """Guarda únicamente los outputs originales de la fase 08."""
    ensure_phase08_directories()
    write_csv_atomic(df_kw, SEGMENTS_WITH_KEYWORDS_CSV)
    write_csv_atomic(call_level_keywords, CALL_LEVEL_KEYWORDS_CSV)
    write_csv_atomic(top_critical_calls, TOP_CRITICAL_CALLS_CSV)

    saved = {
        "segments": SEGMENTS_WITH_KEYWORDS_CSV,
        "calls": CALL_LEVEL_KEYWORDS_CSV,
        "top_calls": TOP_CRITICAL_CALLS_CSV,
    }
    if combined is not None:
        write_csv_atomic(combined, CALL_KEYWORDS_SENTIMENT_CSV)
        saved["combined"] = CALL_KEYWORDS_SENTIMENT_CSV
    return saved
