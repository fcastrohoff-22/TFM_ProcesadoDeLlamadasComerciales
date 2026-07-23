# TFM — Procesamiento multimodal de llamadas comerciales

Este repositorio contiene el pipeline desarrollado para el Trabajo Fin de Máster orientado al procesamiento de llamadas comerciales mediante técnicas de audio, procesamiento del lenguaje natural y aprendizaje automático.

El flujo integra inventariado y limpieza de audios, diarización, reetiquetado mediante embeddings, transcripción automática, asignación proxy de roles, análisis de sentimiento textual, análisis afectivo y prosódico, detección de temas críticos y verificación de huella de voz.

Repositorio del proyecto:

<https://github.com/fcastrohoff-22/TFM_ProcesadoDeLlamadasComerciales.git>

## Estructura del repositorio

```text
.
├── notebooks/                 # Notebooks refactorizados y ejecutables
├── notebooks_legacy/          # Versiones originales, solo para trazabilidad
├── src/                       # Lógica reutilizable del pipeline
├── data/                      # Inputs restaurados, checkpoints y outputs locales
├── demos/                     # Módulos y notebooks de demostración, si aplica
├── requirements.txt           # Dependencias Python acumuladas
├── environment.yml            # Entorno Conda base
├── .env                       # Secretos locales; no se versiona
└── README.md
```

### `notebooks/`

Contiene las versiones definitivas y refactorizadas. Son los únicos notebooks que forman parte del flujo reproducible del proyecto.

### `notebooks_legacy/`

Contiene las versiones originales utilizadas durante el desarrollo. Se conservan únicamente para facilitar la trazabilidad, la comparación metodológica y la revisión histórica del trabajo.

Los notebooks de esta carpeta:

- no forman parte del orden de ejecución;
- pueden contener estructuras, rutas o celdas anteriores a la refactorización;
- no deben utilizarse para reproducir los resultados finales;
- no sustituyen a los notebooks ubicados en `notebooks/`.

### `src/`

Contiene la lógica de procesamiento separada de la orquestación de los notebooks. Entre otros, incluye módulos para:

- configuración centralizada de rutas y parámetros;
- lectura, restauración y sincronización con Google Cloud Storage;
- inventariado y limpieza de audio;
- diarización y embeddings;
- validación y reetiquetado;
- consolidación de segmentos;
- transcripción con Whisper;
- integración con metadata y asignación proxy de roles;
- sentimiento textual;
- emoción acústica y prosodia;
- fusión audio–texto;
- keyword spotting;
- verificación e identificación por huella de voz.

Los notebooks mantienen visibles los controles, decisiones de reutilización, resúmenes, tablas y visualizaciones. La lógica pesada o reutilizable permanece en `src/`.

## Requisitos previos

Para ejecutar el proyecto se recomienda:

- Conda o Miniconda;
- Python 3.11;
- JupyterLab o Jupyter Notebook;
- `ffmpeg` y `libsndfile` disponibles en el entorno;
- acceso autorizado al proyecto de Google Cloud utilizado por el TFM;
- un token de Hugging Face con acceso a los modelos utilizados;
- espacio local suficiente para restaurar audios, checkpoints y resultados.

Las fases de diarización, transcripción, análisis afectivo y huella de voz pueden requerir tiempos de ejecución elevados. El uso de GPU puede reducir significativamente algunos tiempos, aunque el pipeline también contempla ejecución en CPU.

## Instalación del entorno

Todos los comandos deben ejecutarse desde la raíz del repositorio.

### 1. Clonar el repositorio

```bash
git clone https://github.com/fcastrohoff-22/TFM_ProcesadoDeLlamadasComerciales.git
cd TFM_ProcesadoDeLlamadasComerciales
```

### 2. Crear el entorno Conda

El archivo `environment.yml` define el nombre del entorno, la versión de Python y las dependencias base del sistema.

```bash
conda env create -f environment.yml
```

Activar el entorno:

```bash
conda activate tfm_huelladevoz
```

Si el entorno ya existe y se ha actualizado el archivo `environment.yml`:

```bash
conda env update -f environment.yml --prune
conda activate tfm_huelladevoz
```

### 3. Instalar las dependencias Python

El archivo `requirements.txt` contiene las dependencias acumuladas de los notebooks y módulos del proyecto.

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Aunque cada notebook incluye al inicio una instalación acumulada mediante `%pip install -r requirements.txt`, se recomienda realizar esta instalación una vez al crear el entorno.

### 4. Registrar el kernel de Jupyter

```bash
python -m ipykernel install \
  --user \
  --name tfm_huelladevoz \
  --display-name "Python (TFM_HuellaDeVoz)"
```

Iniciar Jupyter desde la raíz del repositorio:

```bash
jupyter lab
```

En cada notebook debe seleccionarse:

```text
Kernel > Change Kernel > Python (TFM_HuellaDeVoz)
```

Ejecutar Jupyter desde la raíz es importante para que los notebooks encuentren correctamente `requirements.txt`, `src/` y las rutas definidas en `config.py`.

## Configuración del archivo `.env`

Debe crearse manualmente un archivo llamado `.env` en la raíz del repositorio.

```env
HF_TOKEN=tu_token_de_hugging_face
TFM_ANONYMIZATION_SALT=tu_salt_privado_y_estable
```

### `HF_TOKEN`

Se utiliza para descargar y cargar los modelos de Hugging Face que requieren autenticación, especialmente los modelos empleados en diarización y embeddings vocales.

Antes de ejecutar el pipeline debe comprobarse que la cuenta asociada al token tenga acceso a los modelos utilizados y que se hayan aceptado, cuando corresponda, sus condiciones de uso.

### `TFM_ANONYMIZATION_SALT`

Se utiliza para generar identificadores anonimizados reproducibles.

Debe mantenerse el mismo valor utilizado durante la generación de los outputs existentes. Cambiarlo produce hashes diferentes y puede romper la correspondencia entre metadata, audios, agentes, clientes y resultados previamente almacenados.

### Seguridad

El archivo `.env`:

- no debe subirse a GitHub;
- no debe copiarse dentro de notebooks;
- no debe incluirse en ZIP de entrega;
- no debe utilizarse para guardar rutas de GCS;
- no debe compartirse junto con los outputs.

Las rutas, buckets, prefijos y parámetros no secretos se gestionan desde `src/config.py`.

## Acceso a Google Cloud

El pipeline utiliza Google Cloud Storage para audios, checkpoints y resultados, y Google BigQuery para determinadas fuentes de metadata.

La ejecución completa solo es posible desde una máquina o máquina virtual cuya identidad tenga permisos sobre los recursos privados del proyecto. El bucket y las tablas no son públicos.

### Ejecución desde una VM autorizada de Google Cloud

En una VM configurada con una cuenta de servicio autorizada, las bibliotecas de Google utilizan normalmente las credenciales asociadas a la propia máquina:

```python
from google.cloud import storage

gcs_client = storage.Client()
```

No es necesario almacenar un archivo de credenciales dentro del repositorio.

### Ejecución desde un equipo local autorizado

Una persona que ya tenga permisos IAM sobre el proyecto puede configurar credenciales de aplicación mediante Google Cloud CLI:

```bash
gcloud auth application-default login
gcloud config set project mm-bi-catedras-upm
```

La autenticación local no concede permisos por sí sola. La cuenta utilizada debe tener acceso al bucket, a los objetos requeridos y, para las fases correspondientes, a BigQuery.

### Ejecución sin acceso a GCS o BigQuery

Sin permisos sobre los recursos privados no es posible reproducir el pipeline completo ni descargar los audios originales.

No obstante, el código, la metodología, los notebooks, las figuras y los outputs anonimizados que se hayan versionado en el repositorio pueden revisarse directamente en GitHub. Los datos privados, audios originales, identificadores reales y credenciales no se publican.

## Orden de ejecución

Los notebooks deben ejecutarse desde `notebooks/` y en el siguiente orden:

| Fase | Notebook | Objetivo principal |
|---:|---|---|
| 00 | `00_EDA_PreparacionDatos_py.ipynb` | Inventario, metadata inicial, validación y limpieza de audios. |
| 01 | `01_diarizacion_locutores_py.ipynb` | Diarización, detección de segmentos, anchors, embeddings y reetiquetado. |
| 02 | `02_validacion_interna_sensibilidad_diarizacion_py.ipynb` | Validación de solapamiento y sensibilidad de parámetros internos. |
| 03 | `03_evaluacion_interna_diarizacion_relabeling_py.ipynb` | Evaluación interna de diarización, anchors, reetiquetado y embeddings. |
| 04 | `04_consolidacion_segmentos_diarizacion_py.ipynb` | Consolidación, deduplicación y auditoría de segmentos finales. |
| 05 | `05_transcripcion_contextual_py.ipynb` | Transcripción con Whisper y alineación temporal con segmentos diarizados. |
| 06 | `06_metadata_transcripcion_groundtruth_proxy_py.ipynb` | Integración con metadata oficial y asignación proxy de roles agente–cliente. |
| 07 | `07_analisis_sentimiento_audio_texto_py.ipynb` | Sentimiento textual, emoción acústica, prosodia y fusión audio–texto. |
| 08 | `08_keyword_spotting_temas_criticos_py.ipynb` | Detección de temas y palabras clave críticas por segmento y llamada. |
| 09 | `09_verificacion_huella_voz_py.ipynb` | Verificación pairwise e identificación open-set mediante huella de voz. |

Las fases 02 y 03 son principalmente analíticas y de validación. No sustituyen la diarización ni deberían modificar los outputs científicos de la fase 01.

La fase 07 unifica en un solo notebook las antiguas fases separadas de sentimiento textual, análisis afectivo de audio y fusión audio–texto. Internamente conserva tres módulos independientes:

```text
src/sentimiento_textual.py
src/afectivo_audio.py
src/fusion_audio_texto.py
```

## Restauración, checkpoints y reutilización

La carpeta local `data/` puede estar vacía al iniciar el proyecto en una VM nueva autorizada.

Antes de recalcular una fase, los notebooks refactorizados intentan:

1. restaurar desde GCS los inputs necesarios;
2. restaurar outputs y checkpoints previos de la propia fase;
3. comprobar si los resultados finales ya están completos;
4. reutilizar los resultados válidos;
5. evitar cargar modelos pesados cuando no es necesario;
6. sincronizar únicamente los outputs correspondientes a la fase ejecutada.

Las banderas de control suelen tener nombres como:

```python
FORCE_RESTORE = False
FORCE_RECALCULATE = False
FORCE_TRANSCRIPTION = False
FORCE_OPEN_SET = False
```

Con los valores por defecto, el pipeline prioriza la reanudación y evita repetir inferencias costosas. Las banderas `FORCE_*` solo deben activarse cuando se desea reconstruir explícitamente una fase.

Los procesos largos guardan checkpoints por audio o por lote cuando la metodología de la fase lo requiere. Las sincronizaciones con GCS utilizan `skip_unchanged=True` cuando está disponible para no volver a subir archivos idénticos.

## Configuración centralizada

Las rutas, prefijos y nombres de outputs se definen en:

```text
src/config.py
```

Las operaciones generales con Google Cloud Storage se concentran en:

```text
src/storage_io.py
```

Las escrituras atómicas y utilidades de archivos se concentran en:

```text
src/io_utils.py
```

La normalización de identificadores de audio se concentra en:

```text
src/identidad_audio.py
```

No deben duplicarse rutas de GCS, nombres de archivos o secretos dentro de los notebooks.

## Modelos y técnicas principales

El pipeline utiliza, entre otros:

- `pyannote.audio` para diarización de locutores;
- embeddings vocales de Pyannote/Wespeaker para reetiquetado y huella de voz;
- `faster-whisper` para transcripción automática en español;
- RoBERTuito/Pysentimiento para sentimiento textual;
- `UMUTeam/w2v-bert-emotion-es` para emoción acústica;
- variables prosódicas extraídas con `librosa`;
- reglas lingüísticas para keyword spotting;
- similitud coseno, evaluación pairwise e identificación open-set para huella de voz.

Los modelos, thresholds, filtros, semillas, criterios de calidad y reglas científicas están definidos en los notebooks y módulos correspondientes. No deben modificarse únicamente para lograr una ejecución más rápida o métricas aparentemente mejores.

## Outputs

Los resultados locales se organizan dentro de `data/` por fase. Entre otros, se generan:

- inventarios y resultados de limpieza;
- segmentos diarizados y reetiquetados;
- embeddings;
- auditorías y análisis de sensibilidad;
- transcripciones por audio y consolidadas;
- asignaciones proxy de rol;
- sentimiento textual;
- emoción acústica y variables prosódicas;
- fusión audio–texto;
- keyword spotting y rankings de llamadas críticas;
- métricas, pares, perfiles y predicciones de huella de voz;
- figuras y checkpoints.

Los outputs completos se sincronizan con prefijos independientes de GCS para cada fase. La estructura de carpetas y los nombres de los archivos forman parte del contrato del pipeline y no deben alterarse sin revisar previamente las fases posteriores y las demos.

Para facilitar la revisión del TFM sin acceso a la infraestructura privada, pueden versionarse en GitHub únicamente outputs anonimizados, resúmenes, métricas y figuras que no contengan información confidencial. Los audios originales, metadata privada y archivos con identificadores reales deben permanecer fuera del repositorio.

## Demos y visualización de resultados

Después de completar o restaurar las fases 00–09 pueden ejecutarse las demos del proyecto:

- demo end-to-end de una llamada individual;
- dashboard global de resultados;
- visualizaciones destinadas a la memoria y a la defensa.

Las demos consumen los outputs existentes del pipeline y pueden restaurarlos desde GCS cuando la máquina tiene acceso. No recalculan los modelos científicos ni deben modificar objetos del bucket.

Los módulos de demo pueden generar archivos HTML locales para conservar una versión interactiva o estática de las visualizaciones. Estos HTML pueden abrirse en el navegador y, cuando no contienen información privada, pueden publicarse como parte de los materiales de revisión.

## Reproducibilidad y limitaciones

El repositorio permite reproducir la estructura del pipeline, sus transformaciones, controles, modelos y outputs siempre que se disponga de:

- los permisos necesarios sobre GCS y BigQuery;
- el mismo `TFM_ANONYMIZATION_SALT`;
- acceso a los modelos de Hugging Face;
- recursos de cómputo y almacenamiento suficientes.

La falta de acceso a la infraestructura privada limita la reproducción end-to-end, pero no impide revisar:

- el código fuente;
- el diseño por fases;
- las decisiones metodológicas;
- los parámetros;
- las métricas y figuras anonimizadas publicadas;
- la estructura de los outputs.

Los resultados de diarización, asignación de roles, sentimiento y huella de voz deben interpretarse como una validación técnica y exploratoria. No constituyen por sí mismos un sistema biométrico o de monitorización listo para producción.

## Solución de problemas frecuentes

### El kernel no aparece en Jupyter

```bash
conda activate tfm_huelladevoz
python -m ipykernel install \
  --user \
  --name tfm_huelladevoz \
  --display-name "Python (TFM_HuellaDeVoz)"
```

Reiniciar JupyterLab y seleccionar el kernel nuevamente.

### `ModuleNotFoundError: No module named 'src'`

Jupyter debe iniciarse desde la raíz del repositorio. También debe comprobarse que el notebook esté dentro de `notebooks/` y no se esté ejecutando una copia aislada desde otra carpeta.

### No se encuentra `requirements.txt`

Abrir Jupyter desde la raíz del repositorio. Los notebooks resuelven la ruta suponiendo que se ejecutan desde la raíz o desde `notebooks/`.

### Error de autenticación de Google Cloud

Comprobar:

- que la VM o cuenta local esté autenticada;
- que las credenciales sean Application Default Credentials válidas;
- que la identidad tenga permisos IAM sobre el bucket y BigQuery;
- que el proyecto activo sea el correcto.

### `403`, `401` o error al descargar un modelo de Hugging Face

Comprobar:

- que `HF_TOKEN` existe en `.env`;
- que el token es válido;
- que la cuenta tiene acceso al modelo;
- que se aceptaron las condiciones del modelo correspondiente.

### Los identificadores anonimizados no coinciden con outputs anteriores

Comprobar que `TFM_ANONYMIZATION_SALT` sea exactamente el mismo utilizado durante la generación de esos outputs.

### Error al cargar audio o librerías de sonido

Actualizar el entorno desde `environment.yml` y comprobar que `ffmpeg` y `libsndfile` estén instalados dentro del entorno Conda.

### Una fase vuelve a cargar un modelo pesado

Comprobar:

- que los outputs finales estén completos;
- que se hayan restaurado desde el prefijo GCS correcto;
- que las banderas `FORCE_*` estén en `False`;
- que los checkpoints no estén incompletos o corruptos.

## Privacidad y seguridad

Este proyecto procesa llamadas comerciales y puede involucrar información sensible. Por ello:

- los audios originales no deben publicarse;
- los identificadores reales no deben incluirse en GitHub;
- los outputs públicos deben estar anonimizados;
- los tokens y salts deben permanecer en `.env`;
- las credenciales de Google Cloud no deben guardarse en el repositorio;
- los notebooks de `notebooks_legacy/` deben revisarse antes de cualquier publicación externa;
- solo las versiones refactorizadas de `notebooks/` deben considerarse parte de la entrega reproducible.

## Referencia rápida de instalación

```bash
git clone https://github.com/fcastrohoff-22/TFM_ProcesadoDeLlamadasComerciales.git
cd TFM_ProcesadoDeLlamadasComerciales

conda env create -f environment.yml
conda activate tfm_huelladevoz

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python -m ipykernel install \
  --user \
  --name tfm_huelladevoz \
  --display-name "Python (TFM_HuellaDeVoz)"

jupyter lab
```

Después de crear `.env`, seleccionar el kernel `Python (TFM_HuellaDeVoz)` y ejecutar los notebooks refactorizados en orden desde la fase 00 hasta la fase 09.
