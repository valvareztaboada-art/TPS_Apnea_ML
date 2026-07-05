# -*- coding: utf-8 -*-
"""
Exploracion inicial de las DOS bases de datos (Apnea-ECG + UCD)
================================================================

Objetivo de este script: verificar que las DOS bases se cargan bien y comparar
visualmente que las señales son compatibles antes de procesar. Busca confirmar que:

  - el ECG de UCD (tras pasar por 00_cargar_ucd.py: EDF -> 100 Hz) se parece
    al de Apnea-ECG,
  - las anotaciones minuto-a-minuto caen donde esperamos en ambas,
  - el sistema de features va a poder tratarlas igual.

--------------------------------------------------------------------------------
Base 1 - Apnea-ECG (Penzel et al., PhysioNet), la de DESARROLLO:
  - 70 registros nocturnos, ECG single-lead a 100 Hz, ~7-10 h.
  - Learning set (35): a01-a20, b01-b05, c01-c10 (tienen .apn 'N'/'A').
  - Test set (35): x01-x35 (sin .apn -> por eso NO lo usamos).
  - Se lee con wfdb (.dat/.hea/.apn/.qrs).

Base 2 - UCD / St. Vincent's Dublin (PhysioNet), el TEST EXTERNO:
  - 25 registros de PSG completa, ECG (derivacion V2 modificada) a 128 Hz.
  - Formato EDF (.rec). Anotaciones de eventos en _respevt.txt (hora + duracion).
  - Poblacion clinica (casi todos con apnea) -> se usa para test PER-MINUTO.
  - Se carga con 00_cargar_ucd.py (EDF -> canal ECG -> remuestreo a 100 Hz ->
    labels 'A'/'N' por minuto).

"""

# =============================================================================
# 1. Importamos las librerias
# =============================================================================
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

import wfdb

# Reutilizamos el loader de UCD que ya validamos (00_cargar_ucd.py).
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# import por nombre de archivo (empieza con numero, no se puede import directo)
import importlib.util
_spec = importlib.util.spec_from_file_location(
    'cargar_ucd', os.path.join(HERE, '00_cargar_ucd.py'))
ucd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ucd)


# =============================================================================
# 2. Configuracion: donde estan las dos bases
# =============================================================================
# Apnea-ECG en su carpeta original; UCD en la carpeta 'files'.

DATA_DIR_APNEA = 'apnea-ecg-database-1.0.0'
DATA_DIR_UCD = 'files'
CACHE_DIR_UCD = 'cache_ucd'   # generado por 00_cargar_ucd.py

# Registros de ejemplo a comparar (uno de cada base).
# a02 es un sujeto apneico de Apnea-ECG; ucddb002 uno de UCD.
REG_APNEA = 'a02'
REG_UCD = 'ucddb002'

FS = 100   # ambas bases quedan a 100 Hz (UCD ya remuestreado por el 00)

print('=' * 70)
print('Configuracion')
print('=' * 70)
print(f'Apnea-ECG dir : {DATA_DIR_APNEA}  (existe: {os.path.isdir(DATA_DIR_APNEA)})')
print(f'UCD dir       : {DATA_DIR_UCD}  (existe: {os.path.isdir(DATA_DIR_UCD)})')
print(f'Registro Apnea-ECG de ejemplo : {REG_APNEA}')
print(f'Registro UCD de ejemplo       : {REG_UCD}')


# =============================================================================
# 3. Inventario: que registros tenemos en cada base?
# =============================================================================

def inventariar_apnea(data_dir):
    """Agrupa los registros de Apnea-ECG por tipo (a/b/c/x)."""
    if not os.path.isdir(data_dir):
        return {}
    heas = sorted(f for f in os.listdir(data_dir) if f.endswith('.hea'))
    regs = [f[:-4] for f in heas
            if not f.endswith('r.hea') and not f.endswith('er.hea')]
    grupos = {'apnea (a)': [], 'borderline (b)': [], 'control (c)': [],
              'test (x)': [], 'otros': []}
    for r in regs:
        if r.startswith('a'):
            grupos['apnea (a)'].append(r)
        elif r.startswith('b'):
            grupos['borderline (b)'].append(r)
        elif r.startswith('c'):
            grupos['control (c)'].append(r)
        elif r.startswith('x'):
            grupos['test (x)'].append(r)
        else:
            grupos['otros'].append(r)
    return grupos


print()
print('=' * 70)
print('Inventario de las bases')
print('=' * 70)

grupos_apnea = inventariar_apnea(DATA_DIR_APNEA)
print('APNEA-ECG:')
for nombre, lista in grupos_apnea.items():
    if lista:
        print(f'  {nombre:16s} ({len(lista):2d}): {lista}')

try:
    regs_ucd = ucd.listar_registros_ucd(DATA_DIR_UCD)
    print(f'\nUCD ({len(regs_ucd)}): {regs_ucd}')
except FileNotFoundError as e:
    print(f'\nUCD: no encontrada ({e})')
    regs_ucd = []


# =============================================================================
# 4. Cargamos un registro de cada base
# =============================================================================
# Apnea-ECG con wfdb; UCD con el loader del 00 (ya remuestreado a 100 Hz).

print()
print('=' * 70)
print('Carga de los registros de ejemplo')
print('=' * 70)

# ---- Apnea-ECG ----
path_apnea = os.path.join(DATA_DIR_APNEA, REG_APNEA)
sig_apnea, fields_apnea = wfdb.rdsamp(path_apnea)
ecg_apnea = sig_apnea[:, 0]
fs_apnea = fields_apnea['fs']
print(f'{REG_APNEA:8s} (Apnea-ECG): fs={fs_apnea} Hz, '
      f'{len(ecg_apnea)} muestras = {len(ecg_apnea)/fs_apnea/60:.1f} min, '
      f'canal={fields_apnea["sig_name"]}')

# ---- UCD ----
# IMPORTANTE: leemos del cache que genero 00_cargar_ucd.py (ECG ya remuestreado
# a 100 Hz), NO reprocesamos el EDF. Asi hay una unica fuente de verdad para el
# ECG de UCD: el 00 es el unico que toca el .rec y decide el remuestreo; todos
# los demas scripts leen el resultado ya cocinado (mismo patron que Apnea-ECG,
# que lee del cache/ que genera el 03b).
cache_path = os.path.join(CACHE_DIR_UCD, f'{REG_UCD}.npz')
if not os.path.exists(cache_path):
    raise FileNotFoundError(
        f'No existe {cache_path}. Corre primero 00_cargar_ucd.py para generar '
        f'el cache de UCD (ECG remuestreado a 100 Hz + labels por minuto).')

_data_ucd = np.load(cache_path, allow_pickle=True)
ecg_ucd = _data_ucd['ecg'].astype(float)
fs_ucd = int(_data_ucd['fs'])
labels_ucd = _data_ucd['labels']
print(f'{REG_UCD:8s} (UCD):       fs={fs_ucd} Hz, '
      f'{len(ecg_ucd)} muestras = {len(ecg_ucd)/fs_ucd/60:.1f} min, '
      f'canal=ECG (V2 mod.), leido de {CACHE_DIR_UCD}/')

# Estadisticas basicas comparativas
print()
print(f'{"":18s}{"Apnea-ECG":>14s}{"UCD":>14s}')
print(f'{"media [mV]":18s}{ecg_apnea.mean():>14.4f}{ecg_ucd.mean():>14.4f}')
print(f'{"std [mV]":18s}{ecg_apnea.std():>14.4f}{ecg_ucd.std():>14.4f}')
print(f'{"min [mV]":18s}{ecg_apnea.min():>14.4f}{ecg_ucd.min():>14.4f}')
print(f'{"max [mV]":18s}{ecg_apnea.max():>14.4f}{ecg_ucd.max():>14.4f}')


# =============================================================================
# 5. Visualizacion comparativa de un segmento de ECG (una de cada, en paralelo)
# =============================================================================
# Un segmento corto de cada base, lado a lado, para confirmar que ambos "se ven"
# como ECG y tienen morfologia comparable.

T_INICIO_MIN = 30      # arrancamos a los 30 min (evita transitorios de inicio)
DURACION_SEG = 15

def segmento(ecg, fs, t_inicio_min, dur_seg):
    i0 = int(t_inicio_min * 60 * fs)
    i1 = i0 + int(dur_seg * fs)
    i1 = min(i1, len(ecg))
    t = np.arange(i0, i1) / fs
    return t, ecg[i0:i1]

t_a, seg_a = segmento(ecg_apnea, fs_apnea, T_INICIO_MIN, DURACION_SEG)
t_u, seg_u = segmento(ecg_ucd, fs_ucd, T_INICIO_MIN, DURACION_SEG)

fig, axes = plt.subplots(2, 1, figsize=(16, 6), sharex=False)
axes[0].plot(t_a, seg_a, linewidth=0.8, color='C0')
axes[0].set_title(f'Apnea-ECG — {REG_APNEA} — {DURACION_SEG} s desde min {T_INICIO_MIN}')
axes[0].set_ylabel('ECG [mV]'); axes[0].grid(True, alpha=0.3)
axes[1].plot(t_u, seg_u, linewidth=0.8, color='C1')
axes[1].set_title(f'UCD — {REG_UCD} — {DURACION_SEG} s desde min {T_INICIO_MIN} '
                  f'(remuestreado 128->100 Hz)')
axes[1].set_ylabel('ECG [mV]'); axes[1].set_xlabel('Tiempo [s]')
axes[1].grid(True, alpha=0.3)
fig.suptitle('Comparacion de un segmento de ECG entre las dos bases')
plt.tight_layout(); plt.show()


# =============================================================================
# 6. Anotaciones de apnea minuto a minuto (una de cada base)
# =============================================================================
# Apnea-ECG: .apn con simbolos 'N'/'A' por minuto (via wfdb).
# UCD: labels 'A'/'N' por minuto que arma el 00 a partir del _respevt.txt.

print()
print('=' * 70)
print('Anotaciones de apnea por minuto')
print('=' * 70)

# ---- Apnea-ECG ----
ann = wfdb.rdann(path_apnea, 'apn')
lab_apnea = np.array([1 if s == 'A' else 0 for s in ann.symbol])
print(f'{REG_APNEA} (Apnea-ECG): {len(lab_apnea)} minutos, '
      f'{int(lab_apnea.sum())} A ({100*lab_apnea.mean():.1f}%), '
      f'{int((1-lab_apnea).sum())} N')

# ---- UCD ----
lab_ucd = (labels_ucd == 'A').astype(int)
print(f'{REG_UCD} (UCD):       {len(lab_ucd)} minutos, '
      f'{int(lab_ucd.sum())} A ({100*lab_ucd.mean():.1f}%), '
      f'{int((1-lab_ucd).sum())} N')


# =============================================================================
# 7. Linea de tiempo de apnea a lo largo de la noche (las dos en paralelo)
# =============================================================================

fig, axes = plt.subplots(2, 1, figsize=(18, 4), sharex=False)

tmin_a = np.arange(len(lab_apnea))
axes[0].step(tmin_a, lab_apnea, where='post', color='C0')
axes[0].fill_between(tmin_a, 0, lab_apnea, step='post', alpha=0.3, color='C0')
axes[0].set_yticks([0, 1]); axes[0].set_yticklabels(['N', 'A'])
axes[0].set_title(f'Apnea-ECG — {REG_APNEA} — anotaciones por minuto')
axes[0].grid(True, alpha=0.3)

tmin_u = np.arange(len(lab_ucd))
axes[1].step(tmin_u, lab_ucd, where='post', color='C1')
axes[1].fill_between(tmin_u, 0, lab_ucd, step='post', alpha=0.3, color='C1')
axes[1].set_yticks([0, 1]); axes[1].set_yticklabels(['N', 'A'])
axes[1].set_title(f'UCD — {REG_UCD} — anotaciones por minuto (desde _respevt.txt)')
axes[1].set_xlabel('Tiempo [min]')
axes[1].grid(True, alpha=0.3)

fig.suptitle('Distribucion temporal de la apnea a lo largo de la noche')
plt.tight_layout(); plt.show()


# =============================================================================
# 8. ECG + anotacion en el mismo panel (una de cada base, tramo largo)
# =============================================================================
# Vista integrada: se ve como la senal convive con los minutos apneicos.

def vista_integrada(ax_ecg, ax_apn, ecg, fs, labels, titulo, color,
                    t_inicio_min=30, dur_min=30):
    j0 = int(t_inicio_min * 60 * fs)
    j1 = j0 + int(dur_min * 60 * fs)
    j1 = min(j1, len(ecg))
    paso = 10  # decimar solo para visualizar
    t = np.arange(j0, j1, paso) / fs / 60
    ax_ecg.plot(t, ecg[j0:j1:paso], linewidth=0.5, color=color)
    ax_ecg.set_ylabel('ECG [mV]'); ax_ecg.grid(True, alpha=0.3)
    ax_ecg.set_title(titulo)

    m0, m1 = t_inicio_min, min(t_inicio_min + dur_min, len(labels))
    tm = np.arange(m0, m1)
    ax_apn.step(tm, labels[m0:m1], where='post', color=color)
    ax_apn.fill_between(tm, 0, labels[m0:m1], step='post', alpha=0.3, color=color)
    ax_apn.set_yticks([0, 1]); ax_apn.set_yticklabels(['N', 'A'])
    ax_apn.set_xlabel('Tiempo [min]'); ax_apn.grid(True, alpha=0.3)

fig, axes = plt.subplots(4, 1, figsize=(18, 9),
                         gridspec_kw={'height_ratios': [3, 1, 3, 1]})
vista_integrada(axes[0], axes[1], ecg_apnea, fs_apnea, lab_apnea,
                f'Apnea-ECG — {REG_APNEA}', 'C0')
vista_integrada(axes[2], axes[3], ecg_ucd, fs_ucd, lab_ucd,
                f'UCD — {REG_UCD}', 'C1')
fig.suptitle('ECG (decimado solo para visualizar) y anotaciones de apnea — ambas bases')
plt.tight_layout(); plt.show()


# =============================================================================
# 9. Cierre
# =============================================================================
print()
print('=' * 70)
print('Fin del script.')

