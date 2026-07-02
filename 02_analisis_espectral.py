# -*- coding: utf-8 -*-
"""
Analisis espectral comparativo: Apnea-ECG vs UCD
=================================================

Objetivo: verificar cuantitativamente que las dos bases son compatibles en el
dominio de la frecuencia, y que los MISMOS filtros (Butterworth HP 0.5 + LP 40)
sirven para ambas. Es el espejo del 02 original, pero superponiendo un registro
de cada base en cada PSD.

Tres preguntas que responde:
  1. El offset DC de UCD (~0.5 mV que vimos en el 01): aparece como energia en
     ~0 Hz. Confirmamos que el pasa-altos de 0.5 Hz lo elimina.
  2. Red electrica a 50 Hz: Apnea-ECG (100 Hz) ya la tenia filtrada por Nyquist.
     UCD original a 128 Hz (Nyquist 64) PUDO registrar la red -> hay que mirar
     si aparece un pico en 50 Hz, para decidir si UCD necesita un notch ademas
     del Butterworth. (Ojo: tras remuestrear a 100 Hz, 50 Hz queda justo en el
     nuevo Nyquist; el resample_poly ya aplica anti-aliasing, pero verificamos.)
  3. La banda del QRS (5-25 Hz): confirmar que cae en el mismo lugar en ambas
     bases (misma morfologia -> features comparables).

Nota metodologica: para que la comparacion sea JUSTA, ambas senales se leen ya
a 100 Hz (Apnea-ECG nativo; UCD del cache que genero el 00, ya remuestreado).
Asi el eje de frecuencia llega al mismo Nyquist (50 Hz) en las dos.
"""

import os
import sys

import numpy as np
import scipy.signal as sg
import matplotlib.pyplot as plt
import wfdb


# =============================================================================
# Configuracion
# =============================================================================

DATA_DIR_APNEA = 'apnea-ecg-database-1.0.0'
CACHE_DIR_UCD = 'cache_ucd'        # generado por 00_cargar_ucd.py

REG_APNEA = 'a01'
REG_UCD = 'ucddb002'

FS = 100                            # ambas bases a 100 Hz
MINUTO_REFERENCIA = 30              # segmento del medio, evita transitorios
DURACION_SEG = 60


# =============================================================================
# Funciones auxiliares
# =============================================================================

def cargar_segmento_apnea(registro, minuto, dur_seg, data_dir=DATA_DIR_APNEA):
    """Carga un segmento de un registro de Apnea-ECG con wfdb."""
    path = os.path.join(data_dir, registro)
    sampfrom = int(minuto * 60 * FS)
    sampto = sampfrom + int(dur_seg * FS)
    signal, fields = wfdb.rdsamp(path, sampfrom=sampfrom, sampto=sampto)
    return signal[:, 0], fields['fs']


def cargar_segmento_ucd(registro, minuto, dur_seg, cache_dir=CACHE_DIR_UCD):
    """Carga un segmento de UCD desde el cache (ECG ya a 100 Hz)."""
    path = os.path.join(cache_dir, f'{registro}.npz')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'No existe {path}. Corre primero 00_cargar_ucd.py.')
    data = np.load(path, allow_pickle=True)
    ecg = data['ecg'].astype(float)
    fs = int(data['fs'])
    i0 = int(minuto * 60 * fs)
    i1 = i0 + int(dur_seg * fs)
    i1 = min(i1, len(ecg))
    return ecg[i0:i1], fs


def psd_welch(x, fs, nperseg=1024):
    """PSD por Welch. nperseg=1024 a 100 Hz -> ventanas de ~10 s.

    Resolucion ~0.098 Hz/bin, suficiente para ver baseline wander. Overlap 50%.
    """
    return sg.welch(x, fs=fs, nperseg=min(nperseg, len(x)), detrend=False)


def potencia_en_banda(f, Pxx, f_min, f_max):
    """Integra la PSD en [f_min, f_max]."""
    mask = (f >= f_min) & (f <= f_max)
    if not mask.any():
        return 0.0
    integrar = getattr(np, 'trapezoid', None) or np.trapz
    return float(integrar(Pxx[mask], f[mask]))


# =============================================================================
# Seccion 1: cargamos un segmento de cada base
# =============================================================================

print('=' * 70)
print('Analisis espectral comparativo Apnea-ECG vs UCD')
print(f'  segmento: {DURACION_SEG} s desde el minuto {MINUTO_REFERENCIA}')
print('=' * 70)

ecg_apnea, fs_a = cargar_segmento_apnea(REG_APNEA, MINUTO_REFERENCIA, DURACION_SEG)
ecg_ucd, fs_u = cargar_segmento_ucd(REG_UCD, MINUTO_REFERENCIA, DURACION_SEG)

print(f'{REG_APNEA} (Apnea-ECG): {len(ecg_apnea)} muestras, fs={fs_a} Hz, '
      f'media={ecg_apnea.mean():+.4f} mV, std={ecg_apnea.std():.4f} mV')
print(f'{REG_UCD} (UCD):       {len(ecg_ucd)} muestras, fs={fs_u} Hz, '
      f'media={ecg_ucd.mean():+.4f} mV, std={ecg_ucd.std():.4f} mV')

# NOTA: detrend=False a proposito, para VER el offset DC en la PSD (si
# usaramos detrend='constant' lo estariamos borrando justo lo que queremos ver).
f_a, P_a = psd_welch(ecg_apnea, fs_a)
f_u, P_u = psd_welch(ecg_ucd, fs_u)


# =============================================================================
# Seccion 2: PSD completa - las dos bases superpuestas
# =============================================================================

plt.figure(figsize=(16, 6))
plt.semilogy(f_a, P_a, label=f'{REG_APNEA} (Apnea-ECG)', linewidth=1.3, color='C0')
plt.semilogy(f_u, P_u, label=f'{REG_UCD} (UCD)', linewidth=1.3, color='C1')
plt.axvspan(0, 0.5, alpha=0.15, color='red', label='Baseline wander (0-0.5 Hz)')
plt.axvspan(5, 25, alpha=0.10, color='green', label='Banda QRS (5-25 Hz)')
plt.axvline(50, color='orange', linestyle='--', alpha=0.7,
            label='Red (50 Hz = Nyquist)')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [mV^2 / Hz] (escala log)')
plt.title('PSD comparada: Apnea-ECG vs UCD (senal cruda, ambas a 100 Hz)')
plt.legend(loc='upper right')
plt.grid(True, which='both', alpha=0.3)
plt.xlim(0, FS / 2 + 2)
plt.tight_layout(); plt.show()


# =============================================================================
# Seccion 3: zoom baja frecuencia (0-2 Hz) - el offset DC y el baseline wander
# =============================================================================
# Aca se ve el offset de UCD (energia concentrada cerca de 0 Hz) y como el corte
# de 0.5 Hz lo deja afuera.

plt.figure(figsize=(15, 5))
plt.semilogy(f_a, P_a, label=f'{REG_APNEA} (Apnea-ECG)', linewidth=1.5,
             marker='.', markersize=4, color='C0')
plt.semilogy(f_u, P_u, label=f'{REG_UCD} (UCD)', linewidth=1.5,
             marker='.', markersize=4, color='C1')
plt.axvspan(0, 0.5, alpha=0.2, color='red', label='Se elimina con HP 0.5 Hz')
plt.axvline(0.5, color='red', linestyle='--', alpha=0.6, label='Corte HP (0.5 Hz)')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [mV^2 / Hz]')
plt.title('Zoom 0-2 Hz: offset DC (UCD) y baseline wander. '
          'El HP de 0.5 Hz recorta esta zona.')
plt.xlim(0, 2)
plt.legend()
plt.grid(True, which='both', alpha=0.3)
plt.tight_layout(); plt.show()


# =============================================================================
# Seccion 4: zoom alta frecuencia (35-50 Hz) - la red electrica
# =============================================================================
# Pregunta clave: aparece pico de red a 50 Hz en UCD? Apnea-ECG (100 Hz) ya la
# tenia filtrada. Si UCD muestra un pico marcado en 50 Hz, habria que sumar un
# notch; si no, el LP de 40 Hz alcanza.

plt.figure(figsize=(15, 5))
plt.semilogy(f_a, P_a, label=f'{REG_APNEA} (Apnea-ECG)', linewidth=1.5,
             marker='.', markersize=5, color='C0')
plt.semilogy(f_u, P_u, label=f'{REG_UCD} (UCD)', linewidth=1.5,
             marker='.', markersize=5, color='C1')
plt.axvline(50, color='orange', linestyle='--', alpha=0.7, label='Red (50 Hz)')
plt.axvline(40, color='gray', linestyle=':', alpha=0.6, label='Corte LP (40 Hz)')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [mV^2 / Hz]')
plt.title('Zoom 35-50 Hz: hay interferencia de red en UCD?')
plt.xlim(35, FS / 2 + 1)
plt.legend()
plt.grid(True, which='both', alpha=0.3)
plt.tight_layout(); plt.show()


# =============================================================================
# Seccion 5: tabla de potencias por banda (comparativa)
# =============================================================================

print()
print('=' * 70)
print('Potencias integradas por banda (mV^2)')
print('=' * 70)
print(f"{'Base':<20} {'0-0.5 Hz':>12} {'5-25 Hz (QRS)':>16} "
      f"{'45-50 Hz':>12} {'ratio BL/QRS':>14}")
print('-' * 76)
for nombre, f, P in [(f'{REG_APNEA} (Apnea)', f_a, P_a),
                     (f'{REG_UCD} (UCD)', f_u, P_u)]:
    p_bl = potencia_en_banda(f, P, 0, 0.5)
    p_qrs = potencia_en_banda(f, P, 5, 25)
    p_50 = potencia_en_banda(f, P, 45, 50)
    ratio = p_bl / p_qrs if p_qrs > 0 else float('inf')
    print(f"{nombre:<20} {p_bl:>12.5f} {p_qrs:>16.5f} {p_50:>12.5f} {ratio:>14.3f}")
print()
print('Lectura:')
print('  - ratio BL/QRS alto en UCD (por el offset DC) es esperable; el HP 0.5 lo limpia.')
print('  - si P(45-50 Hz) de UCD NO es mucho mayor que la de Apnea, no hace falta notch.')


# =============================================================================
# Seccion 6: verificacion del filtrado - antes/despues sobre las dos bases
# =============================================================================
# Aplicamos el mismo Butterworth HP 0.5 + LP 40 a ambas y confirmamos que
# quedan centradas en 0 y comparables. Esto ADELANTA lo que hace el pipeline.

def filtrar_butter(x, fs, fc_hp=0.5, fc_lp=40, orden=4):
    b_hp, a_hp = sg.butter(orden, fc_hp / (fs / 2), btype='high')
    b_lp, a_lp = sg.butter(orden, fc_lp / (fs / 2), btype='low')
    y = sg.filtfilt(b_hp, a_hp, x)
    y = sg.filtfilt(b_lp, a_lp, y)
    return y

ecg_a_filt = filtrar_butter(ecg_apnea, fs_a)
ecg_u_filt = filtrar_butter(ecg_ucd, fs_u)

print()
print('=' * 70)
print('Efecto del filtrado (Butterworth HP 0.5 + LP 40, filtfilt)')
print('=' * 70)
print(f"{'Base':<20} {'media antes':>12} {'media desp':>12} "
      f"{'std antes':>12} {'std desp':>12}")
print('-' * 70)
for nombre, cr, fi in [(f'{REG_APNEA}', ecg_apnea, ecg_a_filt),
                       (f'{REG_UCD}', ecg_ucd, ecg_u_filt)]:
    print(f"{nombre:<20} {cr.mean():>12.4f} {fi.mean():>12.4f} "
          f"{cr.std():>12.4f} {fi.std():>12.4f}")
print()
print('Esperado: la media despues del filtro ~0 en AMBAS (el HP mata el offset).')

# Visualizacion antes/despues (una fila por base)
t_a = np.arange(len(ecg_apnea)) / fs_a
t_u = np.arange(len(ecg_ucd)) / fs_u
seg = 10  # mostrar 10 s
na = int(seg * fs_a); nu = int(seg * fs_u)

fig, axes = plt.subplots(2, 2, figsize=(16, 7))
axes[0, 0].plot(t_a[:na], ecg_apnea[:na], color='C0', linewidth=0.8)
axes[0, 0].set_title(f'{REG_APNEA} (Apnea-ECG) - CRUDO')
axes[0, 0].set_ylabel('mV'); axes[0, 0].grid(True, alpha=0.3)
axes[0, 1].plot(t_a[:na], ecg_a_filt[:na], color='C0', linewidth=0.8)
axes[0, 1].set_title(f'{REG_APNEA} - FILTRADO'); axes[0, 1].grid(True, alpha=0.3)
axes[1, 0].plot(t_u[:nu], ecg_ucd[:nu], color='C1', linewidth=0.8)
axes[1, 0].set_title(f'{REG_UCD} (UCD) - CRUDO (offset ~0.5 mV)')
axes[1, 0].set_ylabel('mV'); axes[1, 0].set_xlabel('Tiempo [s]')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 1].plot(t_u[:nu], ecg_u_filt[:nu], color='C1', linewidth=0.8)
axes[1, 1].set_title(f'{REG_UCD} - FILTRADO (centrado en 0)')
axes[1, 1].set_xlabel('Tiempo [s]'); axes[1, 1].grid(True, alpha=0.3)
fig.suptitle('Efecto del filtrado: el offset de UCD desaparece, ambas quedan comparables')
plt.tight_layout(); plt.show()


# =============================================================================
# Seccion 7: PSD despues del filtrado - confirmacion final
# =============================================================================
# Tras filtrar, las dos PSD deberian verse muy parecidas en la banda util.

f_af, P_af = psd_welch(ecg_a_filt, fs_a)
f_uf, P_uf = psd_welch(ecg_u_filt, fs_u)

plt.figure(figsize=(16, 5))
plt.semilogy(f_af, P_af, label=f'{REG_APNEA} filtrado', linewidth=1.3, color='C0')
plt.semilogy(f_uf, P_uf, label=f'{REG_UCD} filtrado', linewidth=1.3, color='C1')
plt.axvspan(5, 25, alpha=0.10, color='green', label='Banda QRS')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [mV^2 / Hz] (log)')
plt.title('PSD tras el filtrado: ambas bases quedan comparables en la banda util')
plt.legend(loc='upper right')
plt.grid(True, which='both', alpha=0.3)
plt.xlim(0, FS / 2 + 2)
plt.tight_layout(); plt.show()


print()
print('=' * 70)
print('Fin. Si tras el filtrado las PSD se parecen y no hay pico de red en UCD,')
print('los mismos filtros sirven para las dos bases -> features comparables.')
print('=' * 70)
