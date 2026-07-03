# -*- coding: utf-8 -*-
"""
Pipeline general para procesar registros de ECG (Apnea-ECG y UCD)
=================================================================

Nucleo comun de procesamiento: filtros del ECG + Pan-Tompkins + limpieza
temporal de la serie RR. La logica de PROCESAMIENTO (procesar_ecg) es agnostica
a la base: recibe un array de ECG + fs y no le importa de donde salio. La logica
de CARGA es la que difiere por base:
  - Apnea-ECG: se carga con wfdb (.dat/.hea/.apn/.qrs)  -> procesar_registro()
  - UCD: se carga del cache que genera 00_cargar_ucd.py -> procesar_registro_ucd()

Ambos wrappers delegan en la MISMA procesar_ecg(), garantizando que las dos
bases se procesen exactamente igual (condicion necesaria para la validacion
cross-database).

Decisiones tecnicas y justificacion: ver scripts 02_analisis_espectral.py
y 03_preprocesamiento_y_qrs.py.
"""

import os
import numpy as np
import scipy.signal as sg
import wfdb


# =============================================================================
# Constantes del pipeline
# =============================================================================

# Filtros generales
FC_PASAALTOS = 0.5
FC_PASABAJOS = 40.0
ORDEN_BUTTER = 4

# Pan-Tompkins
PT_BANDA_BAJA = 5.0
PT_BANDA_ALTA = 15.0
PT_ORDEN_BANDA = 2
PT_VENTANA_INT_MS = 150
PT_REFRACTARIO_MS = 200
PT_ADAPT_ALPHA = 0.3
PT_GUARDA_BORDE_MS = 200
# Ventana del umbral adaptativo LOCAL. El umbral se recalcula cada
# PT_VENTANA_UMBRAL_SEG segundos y se suaviza para evitar saltos abruptos.
PT_VENTANA_UMBRAL_SEG = 30
PT_SUAVIZADO_UMBRAL_SEG = 5

# Filtros temporales sobre RR
RR_MIN_FISIOL = 0.3
RR_MAX_FISIOL = 2.0
# Malik al 30% (antes 20%): durante la apnea hay variacion ciclica de la FC
# (CVHR) con cambios latido-a-latido que pueden superar el 20% sin ser
# artefactos. Con 20% se marcaban como outliers latidos que son justamente la
# senal de apnea que queremos conservar. 30% respeta esa fisiologia.
MALIK_UMBRAL = 0.30
MEDIANA_VENTANA = 5
MEDIANA_UMBRAL = 0.30


# =============================================================================
# Etapa A: filtros del ECG
# =============================================================================

def disenar_pasaaltos(fc, fs, orden=ORDEN_BUTTER):
    return sg.butter(orden, fc/(fs/2), btype='highpass')


def disenar_pasabajos(fc, fs, orden=ORDEN_BUTTER):
    return sg.butter(orden, fc/(fs/2), btype='lowpass')


def filtrar_ecg_general(ecg, fs,
                        fc_hp=FC_PASAALTOS, fc_lp=FC_PASABAJOS,
                        orden=ORDEN_BUTTER):
    """HP + LP Butterworth con filtfilt (fase cero, no corre los QRS)."""
    b_hp, a_hp = disenar_pasaaltos(fc_hp, fs, orden)
    b_lp, a_lp = disenar_pasabajos(fc_lp, fs, orden)
    return sg.filtfilt(b_lp, a_lp, sg.filtfilt(b_hp, a_hp, ecg))


# =============================================================================
# Etapa B: Pan-Tompkins
# =============================================================================

def pt_paso1_bandpass(x, fs, f_low=PT_BANDA_BAJA, f_high=PT_BANDA_ALTA,
                       orden=PT_ORDEN_BANDA):
    b, a = sg.butter(orden, [f_low/(fs/2), f_high/(fs/2)], btype='band')
    return sg.filtfilt(b, a, x)


def pt_paso2_derivada(x, fs):
    """Derivada centrada con kernel de Pan-Tompkins (1/8)*[1,2,0,-2,-1]."""
    h = np.array([1, 2, 0, -2, -1]) * fs / 8.0
    return np.convolve(x, h, mode='same')


def pt_paso3_cuadrado(x):
    return x ** 2


def pt_paso4_integrador(x, fs, ventana_ms=PT_VENTANA_INT_MS):
    N = max(1, int(ventana_ms * fs / 1000))
    return np.convolve(x, np.ones(N)/N, mode='same')


def pt_detectar_picos(integrada, fs, refractario_ms=PT_REFRACTARIO_MS,
                       alpha=PT_ADAPT_ALPHA,
                       guarda_borde_ms=PT_GUARDA_BORDE_MS,
                       ventana_umbral_seg=PT_VENTANA_UMBRAL_SEG,
                       suavizado_umbral_seg=PT_SUAVIZADO_UMBRAL_SEG):
    """Deteccion adaptativa LOCAL.

    Para cada ventana de `ventana_umbral_seg` segundos, calcula:
        umbral_local = mediana_ventana + alpha * (P99_ventana - mediana_ventana)

    Los umbrales se suavizan despues con una convolucion de
    `suavizado_umbral_seg` segundos para evitar saltos abruptos entre
    ventanas. Asi el detector se adapta a cambios de amplitud a lo largo del
    registro (electrodos flojos, cambios posturales, ruido transitorio).

    Si el registro es muy corto para hacer ventanas, cae a un umbral global.

    Devuelve (picos, umbrales) donde `umbrales` es un array del mismo largo
    que `integrada` con el umbral usado en cada muestra.
    """
    N = len(integrada)
    distancia = int(refractario_ms * fs / 1000)
    borde = int(guarda_borde_ms * fs / 1000)
    win_n = int(ventana_umbral_seg * fs)

    if win_n <= 0 or N < 2 * win_n:
        # Fallback: umbral global
        noise = float(np.median(integrada))
        peak = float(np.percentile(integrada, 99))
        umbrales = np.full(N, noise + alpha * (peak - noise))
    else:
        # Umbral en ventanas no solapadas
        umbrales = np.zeros(N)
        for i in range(0, N, win_n):
            j = min(i + win_n, N)
            seg = integrada[i:j]
            noise = float(np.median(seg))
            peak = float(np.percentile(seg, 99))
            umbrales[i:j] = noise + alpha * (peak - noise)
        # Suavizar transiciones entre ventanas
        suav_n = int(suavizado_umbral_seg * fs)
        if suav_n > 1:
            kernel = np.ones(suav_n) / suav_n
            umbrales = np.convolve(umbrales, kernel, mode='same')

    # Enmascarar bordes y encontrar todos los picos respetando refractario
    integrada_busqueda = integrada.copy()
    if borde > 0:
        integrada_busqueda[:borde] = -np.inf
        integrada_busqueda[-borde:] = -np.inf
    todos_picos, _ = sg.find_peaks(integrada_busqueda, distance=distancia)
    # Filtrar por el umbral local en la ubicacion de cada pico candidato
    picos = todos_picos[integrada[todos_picos] > umbrales[todos_picos]]
    return picos, umbrales


def pt_refinar_a_R(picos_int, ecg_filtrado, fs, ventana_ms=75):
    """Para cada pico sobre la integrada, busca max local del ECG en +-75 ms."""
    half_w = int(ventana_ms * fs / 1000)
    picos_R = []
    for p in picos_int:
        i0 = max(0, p - half_w)
        i1 = min(len(ecg_filtrado), p + half_w + 1)
        picos_R.append(i0 + int(np.argmax(ecg_filtrado[i0:i1])))
    return np.array(picos_R, dtype=int)


def pan_tompkins(ecg_filtrado, fs):
    """Pan-Tompkins completo. Devuelve dict con las 4 etapas + picos R."""
    bp = pt_paso1_bandpass(ecg_filtrado, fs)
    der = pt_paso2_derivada(bp, fs)
    cua = pt_paso3_cuadrado(der)
    integ = pt_paso4_integrador(cua, fs)
    picos_int, umbral = pt_detectar_picos(integ, fs)
    picos_R = pt_refinar_a_R(picos_int, ecg_filtrado, fs)
    return {'bandpass': bp, 'derivada': der, 'cuadrado': cua,
            'integrada': integ, 'picos_int': picos_int, 'picos_R': picos_R,
            'umbral': umbral}


def comparar_detecciones(picos_propios, picos_referencia, fs, tol_ms=100):
    """TP / FP / FN con tolerancia temporal."""
    tol = int(tol_ms * fs / 1000)
    referencia = np.asarray(picos_referencia)
    propios = np.asarray(picos_propios)
    matched_ref = np.zeros(len(referencia), dtype=bool)
    matched_propio = np.zeros(len(propios), dtype=bool)
    for i, r in enumerate(referencia):
        if len(propios) == 0:
            break
        diffs = np.abs(propios - r)
        j = int(np.argmin(diffs))
        if diffs[j] <= tol and not matched_propio[j]:
            matched_ref[i] = True
            matched_propio[j] = True
    TP = int(matched_ref.sum())
    FN = int((~matched_ref).sum())
    FP = int((~matched_propio).sum())
    sens = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    return {'TP': TP, 'FP': FP, 'FN': FN,
            'sensibilidad': sens, 'precision': prec}


# =============================================================================
# Etapa C: filtros temporales sobre la serie RR
# =============================================================================

def serie_rr(picos, fs):
    return np.diff(picos) / fs


def filtro_rango_fisiologico(rr, rr_min=RR_MIN_FISIOL, rr_max=RR_MAX_FISIOL):
    return (rr < rr_min) | (rr > rr_max)


def filtro_malik(rr, umbral=MALIK_UMBRAL):
    """Filtro de cambio relativo."""
    out = np.zeros(len(rr), dtype=bool)
    for i in range(1, len(rr)):
        if rr[i-1] > 0 and abs(rr[i] - rr[i-1]) / rr[i-1] > umbral:
            out[i] = True
    return out


def filtro_mediana_local(rr, ventana=MEDIANA_VENTANA, umbral=MEDIANA_UMBRAL):
    """Outliers respecto a la mediana de la ventana local (excluyendo el propio)."""
    out = np.zeros(len(rr), dtype=bool)
    half = ventana // 2
    for i in range(len(rr)):
        i0 = max(0, i - half)
        i1 = min(len(rr), i + half + 1)
        idx_vecinos = list(range(i0, i)) + list(range(i+1, i1))
        if len(idx_vecinos) < 2:
            continue
        m = float(np.median(rr[idx_vecinos]))
        if m > 0 and abs(rr[i] - m) / m > umbral:
            out[i] = True
    return out


def interpolar_nan(rr):
    rr_out = rr.copy().astype(float)
    nans = np.isnan(rr_out)
    if not nans.any() or nans.all():
        return rr_out
    idx = np.arange(len(rr_out))
    rr_out[nans] = np.interp(idx[nans], idx[~nans], rr_out[~nans])
    return rr_out


def limpiar_rr(rr):
    """Aplica los tres filtros y devuelve la serie limpia + los flags.
    """
    flag_rango = filtro_rango_fisiologico(rr)
    flag_malik = filtro_malik(rr)
    flag_mediana = filtro_mediana_local(rr)
    flag_total = flag_rango | flag_malik | flag_mediana

    rr_nan = rr.copy().astype(float)
    rr_nan[flag_total] = np.nan
    rr_interp = interpolar_nan(rr_nan)

    return rr_interp, {
        'rango': flag_rango,
        'malik': flag_malik,
        'mediana': flag_mediana,
        'total': flag_total,
    }


# =============================================================================
# Funciones de I/O y de alto nivel
# =============================================================================

def cargar_ecg(record_name, data_dir, sampfrom=0, sampto=None):
    """Carga el ECG y los metadatos del registro.
    """
    path = os.path.join(data_dir, record_name)
    if sampto is None:
        signal, fields = wfdb.rdsamp(path, sampfrom=sampfrom)
    else:
        signal, fields = wfdb.rdsamp(path, sampfrom=sampfrom, sampto=sampto)
    return signal[:, 0], fields['fs'], fields


def cargar_segmento(record_name, minuto, duracion_seg, data_dir):
    """Wrapper convenience: carga `duracion_seg` segundos desde `minuto`."""
    fs_aprox = 100
    sampfrom = int(minuto * 60 * fs_aprox)
    sampto = sampfrom + int(duracion_seg * fs_aprox)
    return cargar_ecg(record_name, data_dir,
                       sampfrom=sampfrom, sampto=sampto)


def cargar_anotaciones_qrs(record_name, data_dir):
    """Carga las anotaciones .qrs (machine-generated). Devuelve array de
    indices de muestra. Lanza FileNotFoundError si no existe."""
    path = os.path.join(data_dir, record_name)
    ann = wfdb.rdann(path, 'qrs')
    return ann.sample


def cargar_anotaciones_apn(record_name, data_dir):
    """Carga las anotaciones .apn (apnea por minuto). Devuelve (samples, symbols).
    Solo existen para el learning set (a*, b*, c*). Lanza FileNotFoundError
    para los test (x*)."""
    path = os.path.join(data_dir, record_name)
    ann = wfdb.rdann(path, 'apn')
    return ann.sample, np.array(ann.symbol)


def procesar_ecg(ecg_crudo, fs, filtrar=True):
    """Procesamiento de un ECG YA CARGADO, agnostico a la base de datos.

    Este es el nucleo comun: recibe un array de ECG (venga de Apnea-ECG via
    wfdb, o de UCD via el cache del 00) y su fs, y aplica filtros generales +
    Pan-Tompkins + limpieza de RR. Al no depender de como se cargo la senal,
    garantiza que las dos bases se procesan EXACTAMENTE igual.

    Parameters
    ----------
    ecg_crudo : np.ndarray   senal de ECG (1D).
    fs : int                 frecuencia de muestreo.
    filtrar : bool           si False, asume que ecg_crudo ya viene filtrado
                             (no vuelve a aplicar HP+LP). Default True.

    Returns
    -------
    dict con 'fs', 'duracion_s', 'ecg_crudo', 'ecg_filtrado', 'pt',
    'rr_crudo', 'rr_interp', 'flags'.
    """
    ecg_crudo = np.asarray(ecg_crudo, dtype=float)
    ecg_filtrado = filtrar_ecg_general(ecg_crudo, fs) if filtrar else ecg_crudo
    pt = pan_tompkins(ecg_filtrado, fs)
    rr = serie_rr(pt['picos_R'], fs)
    rr_interp, flags = limpiar_rr(rr)

    return {
        'fs': fs,
        'duracion_s': len(ecg_crudo) / fs,
        'ecg_crudo': ecg_crudo,
        'ecg_filtrado': ecg_filtrado,
        'pt': pt,
        'rr_crudo': rr,
        'rr_interp': rr_interp,
        'flags': flags,
    }


def procesar_registro(record_name, data_dir, sampfrom=0, sampto=None,
                      comparar_qrs=True):
    """Pipeline completo sobre un registro de APNEA-ECG (wrapper con wfdb).

    Carga el ECG (o segmento) de Apnea-ECG y delega el procesamiento en
    procesar_ecg(). Ademas, si existen, compara contra las anotaciones .qrs.

    Parameters
    ----------
    record_name : str         (ej 'a01')
    data_dir : str
    sampfrom, sampto : int    para procesar solo un segmento (default todo)
    comparar_qrs : bool       si True intenta comparar contra .qrs de la base

    Returns
    -------
    dict (ver procesar_ecg) + 'sampfrom', y si corresponde 'comparacion_qrs'
    y 'qrs_ref'.
    """
    ecg_crudo, fs, _ = cargar_ecg(record_name, data_dir,
                                   sampfrom=sampfrom, sampto=sampto)
    result = procesar_ecg(ecg_crudo, fs, filtrar=True)
    result['sampfrom'] = sampfrom

    if comparar_qrs:
        try:
            qrs_ref = cargar_anotaciones_qrs(record_name, data_dir)
            # alinear al frame del segmento si procesamos un trozo
            if sampto is not None:
                m = (qrs_ref >= sampfrom) & (qrs_ref < sampto)
                qrs_ref = qrs_ref[m] - sampfrom
            result['comparacion_qrs'] = comparar_detecciones(
                result['pt']['picos_R'], qrs_ref, fs)
            result['qrs_ref'] = qrs_ref
        except (FileNotFoundError, Exception):
            result['comparacion_qrs'] = None
            result['qrs_ref'] = None

    return result


def procesar_registro_ucd(record_name, cache_dir='cache_ucd'):
    """Pipeline completo sobre un registro de UCD (wrapper con el cache del 00).

    Carga el ECG a 100 Hz desde cache_ucd/<record>.npz (generado por
    00_cargar_ucd.py) y delega el procesamiento en la MISMA procesar_ecg() que
    usa Apnea-ECG. UCD no tiene anotaciones .qrs de referencia, asi que no hay
    comparacion de QRS (comparacion_qrs = None).

    Devuelve ademas 'labels' (etiqueta A/N por minuto) para uso posterior.
    """
    path = os.path.join(cache_dir, f'{record_name}.npz')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'No existe {path}. Corre primero 00_cargar_ucd.py.')
    data = np.load(path, allow_pickle=True)
    ecg_crudo = data['ecg'].astype(float)
    fs = int(data['fs'])

    result = procesar_ecg(ecg_crudo, fs, filtrar=True)
    result['sampfrom'] = 0
    result['comparacion_qrs'] = None
    result['qrs_ref'] = None
    result['labels'] = data['labels']
    return result


# =============================================================================
# Utilidades para batch
# =============================================================================

def listar_registros(data_dir, incluir_respiracion=False):
    """Lista los registros principales (no los rNNr de respiracion).

    Returns
    -------
    list ordenada de nombres base (ej ['a01', 'a02', ..., 'x35'])
    """
    archivos_hea = sorted(f for f in os.listdir(data_dir) if f.endswith('.hea'))
    registros = []
    for f in archivos_hea:
        nombre = f[:-4]
        # filtrar archivos auxiliares de respiracion (rNNr y rNNer)
        if not incluir_respiracion and (nombre.endswith('r') or nombre.endswith('er')):
            continue
        registros.append(nombre)
    return sorted(registros)


def clasificar_grupo(record_name):
    """Devuelve 'apnea', 'borderline', 'control', 'test' u 'otros'."""
    if not record_name:
        return 'otros'
    c = record_name[0].lower()
    return {'a': 'apnea', 'b': 'borderline',
            'c': 'control', 'x': 'test'}.get(c, 'otros')


def listar_registros_ucd(cache_dir='cache_ucd'):
    """Lista los registros de UCD disponibles en el cache (por sus .npz).

    Requiere haber corrido 00_cargar_ucd.py antes. Devuelve lista ordenada
    (ej ['ucddb002', 'ucddb003', ...]).
    """
    if not os.path.isdir(cache_dir):
        raise FileNotFoundError(
            f'No existe {cache_dir}. Corre primero 00_cargar_ucd.py.')
    recs = sorted(
        f[:-4] for f in os.listdir(cache_dir)
        if f.endswith('.npz'))
    return recs


def resumen_registro(record_name, result):
    """Construye un dict resumen con metricas escalares de un registro
    procesado, listo para agregar a una tabla pandas."""
    rr = result['rr_crudo']
    rr_interp = result['rr_interp']
    flags = result['flags']
    n_rr = len(rr)

    fila = {
        'record': record_name,
        'grupo': clasificar_grupo(record_name),
        'duracion_h': result['duracion_s'] / 3600,
        'fs': result['fs'],
        'n_qrs': len(result['pt']['picos_R']),
        'fc_mediana_lpm': float(60 / np.median(rr_interp)) if len(rr_interp) > 0 else None,
        'rr_mediano_s': float(np.median(rr_interp)) if len(rr_interp) > 0 else None,
        'rr_std_s': float(np.std(rr_interp)) if len(rr_interp) > 0 else None,
        'n_outliers_rango': int(flags['rango'].sum()),
        'n_outliers_malik': int(flags['malik'].sum()),
        'n_outliers_mediana': int(flags['mediana'].sum()),
        'n_outliers_total': int(flags['total'].sum()),
        'pct_outliers': float(100 * flags['total'].sum() / n_rr) if n_rr > 0 else None,
    }

    comp = result.get('comparacion_qrs')
    if comp is not None:
        fila.update({
            'qrs_sens_pct': 100 * comp['sensibilidad'],
            'qrs_prec_pct': 100 * comp['precision'],
            'qrs_tp': comp['TP'],
            'qrs_fp': comp['FP'],
            'qrs_fn': comp['FN'],
        })
    else:
        fila.update({
            'qrs_sens_pct': None, 'qrs_prec_pct': None,
            'qrs_tp': None, 'qrs_fp': None, 'qrs_fn': None,
        })
    return fila