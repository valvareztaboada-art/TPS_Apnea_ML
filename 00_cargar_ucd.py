# -*- coding: utf-8 -*-
"""
Carga y homogeneizacion de la base UCD (St. Vincent's / Dublin)
================================================================

Esta base es el TEST EXTERNO del trabajo. A diferencia de Apnea-ECG, viene en
formato EDF y a 128 Hz, con las anotaciones de eventos respiratorios en un .txt
aparte (hora de inicio + duracion de cada evento). Este script:

  1. Lee el ECG mono-derivacion del .rec (canal 'ECG', 128 Hz).
  2. Lo lleva a 100 Hz para que sea comparable con Apnea-ECG (mismo terreno de
     features). Como 128->100 no es factor entero, se usa resample_poly.
  3. Parsea el _respevt.txt y construye una etiqueta binaria por minuto
     (A = algun evento respiratorio en ese minuto, N = ninguno), replicando
     el esquema minuto-a-minuto de Apnea-ECG.
  4. (Opcional) lee el _stage.txt con las etapas de sueno. NO se usan para
     filtrar (decision conservadora: tratamos toda la noche igual que
     Apnea-ECG, que no trae etapas), pero se dejan disponibles.

Decisiones de etiquetado (acordadas por el grupo):
  - Cualquier tipo de evento respiratorio (HYP-*, APNEA-* obstructiva/central/
    mixta, y respiracion periodica PB/CS si estuviera marcada) cuenta como A.
    Es la misma definicion amplia de "disordered breathing" de Penzel.
  - No se filtran los minutos de vigilia (Wake).

Salida por sujeto: un dict con el ECG a 100 Hz y el vector de labels por minuto,
en el mismo formato que despues consume el pipeline (filtros + Pan-Tompkins +
features). Opcionalmente se cachea en cache_ucd/<record>.npz.

Verificado sobre ucddb002:
  - EDF start 00:11:04, duracion 374.5 min, canal ECG en indice 5 a 128 Hz.
  - respevt con tipos HYP-C/O/M y APNEA-O/M; ultimo evento en min 294 (dentro).
  - stage: 748 epocas de 30 s = 374 min (coincide con la duracion del EDF).
"""

import os
import re
import sys
from datetime import datetime, timedelta

import numpy as np
from math import gcd

from scipy.signal import resample_poly


# =============================================================================
# Lector EDF minimo (numpy puro, sin dependencias externas)
# =============================================================================
# UCD viene en formato EDF (.rec). No usamos pyedflib ni mne para no sumar
# dependencias que ademas fallan al compilar en Windows + Python nuevo. El
# formato EDF es simple (header de texto de tamano fijo + datos int16), asi que
# lo parseamos con numpy. Clave: EDF permite frecuencias distintas por canal;
# leemos el canal de ECG a SU frecuencia real (128 Hz), sin mezclarlo con los
# canales lentos (a diferencia de wfdb.read_edf, que remuestrea todo a la fs
# base y destruiria la resolucion del ECG).
# Spec: https://www.edfplus.info/specs/edf.html

def _leer_canal_edf(path, nombre_canal='ECG'):
    """Lee un canal especifico de un EDF respetando su frecuencia real.

    Returns
    -------
    signal : np.ndarray (en unidades fisicas, p.ej. mV)
    fs     : int (frecuencia real del canal)
    start_dt : datetime o None (inicio del registro)
    dur_s  : float (duracion en segundos)
    """
    with open(path, 'rb') as f:
        raw = f.read()

    def s(o, n):
        return raw[o:o + n].decode('latin-1').strip()

    startdate = s(168, 8)   # dd.mm.yy
    starttime = s(176, 8)   # hh.mm.ss
    n_bytes_header = int(s(184, 8))
    n_records = int(s(236, 8))
    dur_record = float(s(244, 8))   # segundos por record
    n_signals = int(s(252, 4))

    base = 256
    labels = [s(base + i * 16, 16) for i in range(n_signals)]
    # saltar transducer(80) y phys_dim(8) por senal
    off = base + n_signals * 16 + n_signals * 80 + n_signals * 8
    phys_min = [float(s(off + i * 8, 8)) for i in range(n_signals)]
    off += n_signals * 8
    phys_max = [float(s(off + i * 8, 8)) for i in range(n_signals)]
    off += n_signals * 8
    dig_min = [float(s(off + i * 8, 8)) for i in range(n_signals)]
    off += n_signals * 8
    dig_max = [float(s(off + i * 8, 8)) for i in range(n_signals)]
    off += n_signals * 8
    off += n_signals * 80    # prefiltering
    n_samps = [int(s(off + i * 8, 8)) for i in range(n_signals)]

    # indice del canal buscado (match exacto y luego parcial)
    idx = None
    for i, lab in enumerate(labels):
        if lab.strip().upper() == nombre_canal.upper():
            idx = i
            break
    if idx is None:
        for i, lab in enumerate(labels):
            if nombre_canal.upper() in lab.strip().upper():
                idx = i
                break
    if idx is None:
        raise ValueError(f'Canal {nombre_canal!r} no encontrado. Canales: {labels}')

    # datos: int16 little-endian, organizados por records
    data = np.frombuffer(raw[n_bytes_header:], dtype='<i2')
    samps_per_record = sum(n_samps)
    total = samps_per_record * n_records
    data = data[:total].reshape(n_records, samps_per_record)

    ch_off = sum(n_samps[:idx])
    ch_data = data[:, ch_off:ch_off + n_samps[idx]].reshape(-1).astype(float)

    # digital -> fisico
    gain = (phys_max[idx] - phys_min[idx]) / (dig_max[idx] - dig_min[idx])
    offset = phys_max[idx] - gain * dig_max[idx]
    ch_phys = ch_data * gain + offset

    fs = int(round(n_samps[idx] / dur_record))
    dur_s = n_records * dur_record

    try:
        dd, mm, yy = startdate.split('.')
        hh, mi, ss = starttime.split('.')
        yy = int(yy)
        yy = 2000 + yy if yy < 85 else 1900 + yy
        start_dt = datetime(yy, int(mm), int(dd), int(hh), int(mi), int(ss))
    except Exception:
        start_dt = None

    return ch_phys, fs, start_dt, dur_s


# =============================================================================
# Configuracion
# =============================================================================

# Carpeta donde estan los archivos de UCD (los .rec, _respevt.txt, _stage.txt).
# En la compu del grupo la base esta en la carpeta 'files/'.
DATA_DIR_UCD = 'files'

CACHE_DIR_UCD = 'cache_ucd'

FS_OBJETIVO = 100          # Hz, para igualar a Apnea-ECG
NOMBRE_CANAL_ECG = 'ECG'   # label del canal en el EDF (verificado en ucddb002)

# Duracion de la epoca de las etapas de sueno (Rechtschaffen & Kales)
EPOCA_STAGE_SEG = 30


# =============================================================================
# Lectura del ECG desde el EDF
# =============================================================================

def cargar_ecg_ucd(record, data_dir=DATA_DIR_UCD):
    """Carga el ECG mono-derivacion de un registro UCD desde su .rec (EDF).

    Usa el lector EDF propio (numpy puro). Se queda con el canal 'ECG'
    (derivacion V2 modificada del PSG) a su frecuencia real (128 Hz).

    Returns
    -------
    ecg : np.ndarray   Senal de ECG cruda a la fs original (128 Hz).
    fs_orig : int      Frecuencia de muestreo original.
    start_dt : datetime  Inicio del registro (ancla para las anotaciones).
    dur_s : float      Duracion del registro en segundos.
    """
    path = os.path.join(data_dir, f'{record}.rec')
    if not os.path.exists(path):
        raise FileNotFoundError(f'No existe {path}')

    ecg, fs_orig, start_dt, dur_s = _leer_canal_edf(path, NOMBRE_CANAL_ECG)
    return ecg, fs_orig, start_dt, dur_s


def resamplear_a_100(ecg, fs_orig, fs_obj=FS_OBJETIVO):
    """Lleva el ECG de fs_orig a fs_obj.

    128 -> 100 no es factor entero (1.28), asi que se usa resample_poly, que
    aplica un filtro anti-aliasing FIR internamente. up/down se calculan
    reduciendo la fraccion fs_obj/fs_orig.

    Nota: el filtrado Butterworth 0.5-40 Hz del pipeline se aplica DESPUES,
    en el pipeline comun, igual que para Apnea-ECG. resample_poly ya evita
    aliasing en la banda que nos interesa (< 50 Hz).
    """
    if fs_orig == fs_obj:
        return ecg.astype(float), fs_obj
    g = gcd(fs_obj, fs_orig)
    up = fs_obj // g
    down = fs_orig // g
    ecg_rs = resample_poly(ecg.astype(float), up, down)
    return ecg_rs, fs_obj


# =============================================================================
# Parseo de las anotaciones de eventos respiratorios (_respevt.txt)
# =============================================================================

# Una linea de datos empieza con hora HH:MM:SS, seguida del tipo de evento y
# la duracion en segundos. Ejemplo:
#   00:29:13  HYP-C             16       89.9    4.1     -     -      64.7   -5.7
_RE_EVENTO = re.compile(
    r'^\s*(\d{2}):(\d{2}):(\d{2})\s+(\S+)\s+(?:(PB|CS)\s+)?(\d+)')


def parsear_respevt(record, data_dir=DATA_DIR_UCD):
    """Lee el _respevt.txt y devuelve una lista de eventos.

    Cada evento es un dict con:
        hora   : datetime.time  (hora del dia del inicio)
        tipo   : str            (HYP-C, APNEA-O, etc.)
        dur_s  : int            (duracion en segundos)

    Ojo: la primera columna es HORA DEL DIA, no segundos desde el inicio del
    registro. El anclaje al registro se hace despues con el start del EDF.
    """
    path = os.path.join(data_dir, f'{record}_respevt.txt')
    if not os.path.exists(path):
        raise FileNotFoundError(f'No existe {path}')

    eventos = []
    with open(path, 'r', encoding='latin-1') as fh:
        for ln in fh:
            m = _RE_EVENTO.match(ln)
            if not m:
                continue  # lineas de encabezado o vacias
            hh, mm, ss, tipo, _pbcs, dur = m.groups()
            eventos.append({
                'hora': (int(hh), int(mm), int(ss)),
                'tipo': tipo,
                'dur_s': int(dur),
            })
    return eventos


def eventos_a_labels_por_minuto(eventos, start_dt, dur_s):
    """Convierte la lista de eventos a un vector binario de labels por minuto.

    Parameters
    ----------
    eventos : list de dicts (salida de parsear_respevt).
    start_dt : datetime
        Inicio del registro (del EDF). Ancla para pasar hora-del-dia a offset.
    dur_s : float
        Duracion del registro en segundos.

    Returns
    -------
    labels : np.ndarray de shape (n_minutos,), dtype='<U1'
        'A' si algun evento solapa ese minuto, 'N' si no.

    Regla de solapamiento: un evento que va de [t_ini, t_ini+dur) marca como A
    todos los minutos que toca (desde floor(t_ini/60) hasta floor((t_fin)/60)).
    """
    n_min = int(dur_s // 60)
    labels = np.array(['N'] * n_min, dtype='<U1')

    start_time = start_dt

    for ev in eventos:
        hh, mm, ss = ev['hora']
        # construir el datetime del evento usando la MISMA fecha de inicio.
        ev_dt = start_time.replace(hour=hh, minute=mm, second=ss, microsecond=0)
        # si el evento "cae antes" del inicio, es porque cruzo la medianoche:
        # le sumamos un dia.
        if ev_dt < start_time:
            ev_dt = ev_dt + timedelta(days=1)

        t_ini = (ev_dt - start_time).total_seconds()
        t_fin = t_ini + ev['dur_s']

        if t_ini < 0 or t_ini >= dur_s:
            continue  # fuera del registro (defensivo)

        min_ini = int(t_ini // 60)
        min_fin = int(min(t_fin, dur_s - 1) // 60)
        for m in range(min_ini, min_fin + 1):
            if 0 <= m < n_min:
                labels[m] = 'A'

    return labels


# =============================================================================
# Etapas de sueno (opcional, NO se usan para filtrar)
# =============================================================================

def cargar_stages(record, data_dir=DATA_DIR_UCD):
    """Lee el _stage.txt (una etapa por epoca de 30 s).

    Codigos: 0=Wake 1=REM 2=S1 3=S2 4=S3 5=S4 6=Artefacto 7=Indeterminado.
    Se devuelve tal cual; la decision del grupo es NO usarlas para filtrar,
    pero quedan disponibles por si se quieren para analisis.
    """
    path = os.path.join(data_dir, f'{record}_stage.txt')
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='latin-1') as fh:
        stages = [int(l.strip()) for l in fh if l.strip().isdigit()]
    return np.array(stages, dtype=int)


# =============================================================================
# Funcion principal por sujeto
# =============================================================================

def cargar_registro_ucd(record, data_dir=DATA_DIR_UCD, resamplear=True):
    """Carga completa de un sujeto UCD: ECG (a 100 Hz) + labels por minuto.

    Returns
    -------
    dict con:
        record      : str
        fs          : int (100 si resamplear=True, si no la original)
        ecg         : np.ndarray  (ECG listo para el pipeline)
        duracion_s  : float
        labels      : np.ndarray de 'A'/'N' por minuto
        stages      : np.ndarray o None
        start_dt    : datetime
        n_eventos   : int
    """
    ecg_crudo, fs_orig, start_dt, dur_s = cargar_ecg_ucd(record, data_dir)

    if resamplear:
        ecg, fs = resamplear_a_100(ecg_crudo, fs_orig)
        # la duracion no cambia; recomputamos por consistencia
        dur_s = len(ecg) / fs
    else:
        ecg, fs = ecg_crudo.astype(float), fs_orig

    eventos = parsear_respevt(record, data_dir)
    labels = eventos_a_labels_por_minuto(eventos, start_dt, dur_s)
    stages = cargar_stages(record, data_dir)

    return {
        'record': record,
        'fs': fs,
        'ecg': ecg,
        'duracion_s': dur_s,
        'labels': labels,
        'stages': stages,
        'start_dt': start_dt,
        'n_eventos': len(eventos),
    }


def listar_registros_ucd(data_dir=DATA_DIR_UCD):
    """Lista los records UCD disponibles (por sus .rec)."""
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f'No existe la carpeta {data_dir}')
    recs = sorted(
        f[:-4] for f in os.listdir(data_dir) if f.lower().endswith('.rec'))
    return recs


# =============================================================================
# Main: procesa y cachea, con reporte
# =============================================================================

def main():
    print('=' * 70)
    print('Carga de la base UCD (test externo)')
    print('=' * 70)

    try:
        registros = listar_registros_ucd(DATA_DIR_UCD)
    except FileNotFoundError as e:
        print(f'ERROR: {e}')
        print(f'Ajusta DATA_DIR_UCD (ahora = {DATA_DIR_UCD!r}).')
        return

    print(f'Carpeta          : {DATA_DIR_UCD}')
    print(f'Registros        : {len(registros)}')
    print(f'fs objetivo      : {FS_OBJETIVO} Hz')
    print()

    os.makedirs(CACHE_DIR_UCD, exist_ok=True)

    filas = []
    for i, r in enumerate(registros, 1):
        try:
            res = cargar_registro_ucd(r)
            n_min = len(res['labels'])
            n_a = int((res['labels'] == 'A').sum())
            n_n = int((res['labels'] == 'N').sum())
            pct_a = 100 * n_a / max(1, n_min)

            np.savez_compressed(
                os.path.join(CACHE_DIR_UCD, f'{r}.npz'),
                fs=res['fs'],
                ecg=res['ecg'].astype(np.float32),
                duracion_s=res['duracion_s'],
                labels=res['labels'],
                stages=res['stages'] if res['stages'] is not None
                        else np.array([], dtype=int),
            )

            print(f'[{i:2d}/{len(registros)}] {r:10s} '
                  f'dur={res["duracion_s"]/60:6.1f} min  '
                  f'eventos={res["n_eventos"]:4d}  '
                  f'A={n_a:4d} ({pct_a:5.1f}%)  N={n_n:4d}')

            filas.append({
                'record': r, 'dur_min': res['duracion_s'] / 60,
                'n_eventos': res['n_eventos'], 'n_min': n_min,
                'n_apnea': n_a, 'n_normal': n_n, 'pct_apnea': pct_a,
            })
        except Exception as e:
            print(f'[{i:2d}/{len(registros)}] {r:10s} ERROR: {e}')

    if filas:
        import pandas as pd
        df = pd.DataFrame(filas)
        df.to_csv(os.path.join(CACHE_DIR_UCD, 'resumen_ucd.csv'), index=False)
        print()
        print('=' * 70)
        print(f'Cache guardado en {CACHE_DIR_UCD}/  (+ resumen_ucd.csv)')
        print(f'Total minutos    : {df["n_min"].sum()}')
        print(f'  apnea (A)      : {df["n_apnea"].sum()} '
              f'({100*df["n_apnea"].sum()/df["n_min"].sum():.1f}%)')
        print(f'  normal (N)     : {df["n_normal"].sum()}')
        print('=' * 70)


if __name__ == '__main__':
    main()