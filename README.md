# Detección de apnea del sueño a partir de ECG nocturno

Sistema de detección de apnea del sueño a partir de un electrocardiograma (ECG) de una sola derivación, mediante procesamiento de señales y aprendizaje automático. Clasifica cada minuto de un registro nocturno como **apnea** o **respiración normal**, y deriva un índice global por sujeto para tamizaje.

El sistema se entrena y valida sobre la base **Apnea-ECG** (PhysioNet) y se somete a una **validación externa** independiente sobre la base **UCDDB** (University College Dublin), de otro centro, población y derivación.

> Trabajo final de la materia **16.63 Procesamiento de Señales e Imágenes Biomédicas** — Instituto Tecnológico de Buenos Aires (ITBA).

---

## Características principales

- **Acondicionamiento** de la señal (filtrado Butterworth, remuestreo, homogeneización entre bases).
- **Detección de QRS** con el algoritmo de Pan-Tompkins implementado desde cero y limpieza de la serie R-R.
- **Extracción de características** de variabilidad de la frecuencia cardíaca (HRV, temporal y frecuencial vía Lomb-Scargle), de la respiración derivada del ECG (EDR) y de la **transformada wavelet** de la serie R-R.
- **Análisis de discriminabilidad** con d de Cohen, AUC individual, correlación y **PCA**.
- **Clasificación por minuto** con modelos de aprendizaje automático de familias distintas (regresión logística, SVM, random forest, gradient boosting) y un **ensemble**, con validación cruzada agrupada por sujeto (sin fuga de información).
- **Clasificación por sujeto** a partir de un índice de apnea estimado (AHI).
- **Validación externa** cross-database sobre UCDDB.
- **Interfaz gráfica** con dos vistas: técnica (análisis) y clínica (tamizaje).

---

## Estructura del proyecto

```
TPS_Apnea_ML/
├── src/
│   ├── __init__.py
│   ├── pipeline.py              # acondicionamiento, QRS y limpieza R-R (comun a ambas bases)
│   └── features.py             # calculo de features HRV + EDR + wavelet
│
├── 00_cargar_ucd.py            # carga UCDDB (lector EDF propio) -> cache_ucd/
├── 01_visualizacion_datos.py   # exploracion comparativa entre bases
├── 02_analisis_espectral.py    # PSD (Welch), justificacion de filtros
├── 03_preprocesamiento_y_qrs.py# inspeccion detallada de QRS y limpieza R-R
├── 03b_procesar_todos.py       # procesa ambas bases por lotes -> cache/, cache_ucd_proc/
├── 04_features_por_minuto.py   # features por minuto de ambas bases
├── 04b_analisis_features.py    # discriminabilidad, correlacion, PCA
├── 05_clasificacion_ml.py      # entrenamiento y validacion interna (per-minuto)
├── 06_evaluacion.py            # validacion externa (UCDDB) + clasificacion per-sujeto
├── interfaz_apnea.py           # interfaz grafica (dos solapas)
│
├── requirements.txt
└── README.md
```

Las bases de datos y los archivos intermedios (`cache/`) **no** se incluyen en el repositorio (ver más abajo).

---

## Instalación

Requiere **Python 3.11 o superior**.

```bash
# 1. Clonar el repositorio y entrar en la carpeta
cd TPS_Apnea_ML

# 2. (Recomendado) crear un entorno virtual
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux / macOS:
source venv/bin/activate

# 3. Instalar las dependencias
python -m pip install -r requirements.txt
```

### Nota para Windows con Python 3.14

El proyecto está pensado para funcionar en Python 3.14 en Windows. `PyWavelets` y `PySide6` se instalan directamente desde wheels precompilados (no requieren compilar). **No** se usa `pyedflib` (que requiere Visual C++ y no compila en 3.14): la lectura de los archivos EDF de UCDDB se hace con un lector propio en `numpy`.

Si tenés más de una instalación de Python, asegurate de instalar y correr con el mismo intérprete (por ejemplo usando `python -m pip ...` y seleccionando ese intérprete en tu editor).

---

## Bases de datos

Las bases **no** se incluyen en el repositorio (son varios GB). Hay que descargarlas de PhysioNet y ubicarlas en la raíz del proyecto:

### Apnea-ECG (base de desarrollo)

- Descarga: https://physionet.org/content/apnea-ecg/1.0.0/
- Ubicar en: `apnea-ecg-database-1.0.0/`
- Se usan los registros del conjunto de aprendizaje: `a01`–`a20`, `b01`–`b05`, `c01`–`c10`.

### UCDDB (base de validación externa)

- Descarga: https://physionet.org/content/ucddb/1.0.0/
- Ubicar los archivos `.rec`, `_respevt.txt` y `_stage.txt` en: `files/`
- Se usan los 25 registros `ucddb002`–`ucddb028`.

Estructura esperada tras la descarga:

```
TPS_Apnea_ML/
├── apnea-ecg-database-1.0.0/
│   ├── a01.dat, a01.hea, a01.apn, ...
│   └── additional-information.txt
└── files/
    ├── ucddb002.rec, ucddb002_respevt.txt, ucddb002_stage.txt
    └── ...
```

---

## Uso: ejecutar el pipeline

Los scripts están numerados y deben correrse **en orden** desde la raíz del proyecto. Cada uno deja resultados intermedios en las carpetas de caché (`cache/` para Apnea-ECG, `cache_ucd/` y `cache_ucd_proc/` para UCDDB), que los pasos siguientes reutilizan.

```bash
# 1. Cargar UCDDB (lee los EDF y genera el cache con el ECG remuestreado a 100 Hz)
python 00_cargar_ucd.py

# 2. (Opcional) exploracion y verificacion de compatibilidad entre bases
python 01_visualizacion_datos.py
python 02_analisis_espectral.py
python 03_preprocesamiento_y_qrs.py

# 3. Procesar ambas bases: QRS + limpieza R-R (por lotes)
python 03b_procesar_todos.py

# 4. Calcular las features por minuto de ambas bases
python 04_features_por_minuto.py

# 5. Analisis de discriminabilidad (figuras + ranking de features)
python 04b_analisis_features.py

# 6. Entrenamiento y validacion interna (clasificacion per-minuto)
python 05_clasificacion_ml.py

# 7. Validacion externa (UCDDB) + clasificacion per-sujeto
python 06_evaluacion.py
```

Los pasos mínimos indispensables antes de abrir la interfaz son **00, 03b, 04, 05 y 06**. Los pasos 01, 02 y 03 son de inspección y generan las figuras del informe, pero no son necesarios para la interfaz.

> **Tiempos**: los pasos 04, 05 y 06 son los más lentos (recargan y filtran el ECG completo de cada sujeto, y el 05 entrena varios modelos incluido un SVM). Pueden tardar varios minutos cada uno.

---

## Uso: interfaz gráfica

Una vez generados los caches (pasos 00, 03b, 04, 05 y 06), abrir la interfaz:

```bash
python interfaz_apnea.py
```

La interfaz tiene un **selector de registro compartido** y dos solapas:

- **Vista técnica** (ambas bases): ECG con las ondas R detectadas, tacograma de la serie R-R con los intervalos descartados, características por minuto coloreadas según la predicción del modelo, y tabla de clasificación por minuto.
- **Vista clínica** (solo Apnea-ECG): línea de tiempo de la noche con la predicción del modelo y la etiqueta real, datos del paciente, índice de apnea estimado y panel de explicabilidad de las características. No aplica a UCDDB por no disponer de datos clínicos del paciente.

---

## Resultados principales

| Evaluación | Métrica | Valor |
|---|---|---|
| Per-minuto, validación interna (Apnea-ECG) | AUC (ensemble) | 0,93 |
| Per-sujeto (Apnea-ECG, A vs C) | Sujetos correctos | 28 / 29 |
| Per-minuto, validación externa (UCDDB) | AUC | 0,69 |

Las características de la **transformada wavelet** resultaron las más discriminantes del conjunto. La caída de rendimiento en la validación externa cuantifica la brecha de dominio entre bases de distinto centro y derivación.

---

## Archivos de caché generados

| Carpeta / archivo | Generado por | Contenido |
|---|---|---|
| `cache_ucd/<record>.npz` | `00` | ECG de UCDDB (100 Hz) + etiquetas |
| `cache/<record>.npz`, `cache_ucd_proc/<record>.npz` | `03b` | picos R, series R-R, flags de outliers |
| `cache/features_apnea.csv`, `cache_ucd_proc/features_ucd.csv` | `04` | features por minuto |
| `cache/ranking_features.csv` | `04b` | discriminabilidad por feature |
| `cache/oof_predicciones.csv` | `05` | predicción por minuto (out-of-fold) |
| `cache/importancias.csv`, `cache/modelo_final.joblib` | `05` | importancias y modelo entrenado |
| `cache_ucd_proc/predicciones_ucd.csv` | `06` | predicción del modelo sobre UCDDB |
| `cache/resultado_per_sujeto.csv` | `06` | clasificación por sujeto |

---

## Notas metodológicas

- **Sin fuga de información**: la validación cruzada agrupa por sujeto (`StratifiedGroupKFold`), y el escalado e imputación se ajustan solo con los datos de entrenamiento de cada partición.
- **Clase borderline (B)**: se incluye en el entrenamiento por minuto (cada minuto tiene etiqueta válida) pero se excluye de la clasificación por sujeto (ambigua a nivel de sujeto).
- **Comentarios del código** en español.

---

## Autoras

Grupo 6 — Álvarez Taboada, V.; Losinno, M. P.; Gowland, D.
