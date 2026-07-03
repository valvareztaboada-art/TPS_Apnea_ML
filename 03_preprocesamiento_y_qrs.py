# -*- coding: utf-8 -*-
"""
Preprocesamiento + Pan-Tompkins: UCD en detalle + comparacion entre bases
==========================================================================

Script EXPLORATORIO para generar las figuras del informe. Como el detalle de
Pan-Tompkins sobre Apnea-ECG ya se mostro en el TP anterior (sujeto a01), aca:

  1. Mostramos Pan-Tompkins ETAPA POR ETAPA sobre la base NUEVA (UCD), para
     evidenciar que el mismo detector se adapta bien a la otra base (otra
     derivacion, QRS predominantemente positivo).
  2. Comparamos las dos bases lado a lado:
       - ECG filtrado con los picos R detectados,
       - tacograma R-R con los intervalos descartados por la limpieza.

Usa el pipeline refactorizado (src/pipeline.py), que procesa ambas bases con
la MISMA logica (procesar_ecg). Apnea-ECG se carga con wfdb; UCD del cache que
genera 00_cargar_ucd.py.
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from src.pipeline import (
    # nucleo y wrappers
    procesar_registro,          # Apnea-ECG (wfdb)
    procesar_registro_ucd,      # UCD (cache)
    pan_tompkins,
    filtrar_ecg_general,
    serie_rr, limpiar_rr,
    # constantes (para los titulos / anotaciones)
    FC_PASAALTOS, FC_PASABAJOS,
    PT_BANDA_BAJA, PT_BANDA_ALTA, PT_VENTANA_INT_MS,
    PT_REFRACTARIO_MS, PT_ADAPT_ALPHA,
    MALIK_UMBRAL, MEDIANA_UMBRAL,
)


# =============================================================================
# Configuracion
# =============================================================================

DATA_DIR_APNEA = 'apnea-ecg-database-1.0.0'
CACHE_DIR_UCD = 'cache_ucd'

REG_APNEA = 'c04'
REG_UCD = 'ucddb002'

# Ventana para visualizar las etapas de Pan-Tompkins (segundos)
VENTANA_VIZ_SEG = 10
# Desde que minuto tomar la ventana de visualizacion
MINUTO_VIZ = 30

# Sujeto de Apnea-ECG con outliers de RR para la validacion manual de ectopicos
# (la devolucion pedia verificar si los RR descartados eran ectopicos reales,
# error de deteccion, o apnea que no habia que sacar). 
REG_OUTLIERS = 'c04'
# cuantos RR descartados inspeccionar en detalle (los de mayor desviacion)
N_ECTOPICOS_VIZ = 6
# ventana de ECG alrededor de cada RR descartado (segundos a cada lado)
VENTANA_ECTOPICO_SEG = 3


# =============================================================================
# Procesamiento de un registro de cada base (pipeline completo)
# =============================================================================

print('=' * 70)
print('Procesamiento exploratorio + comparacion entre bases')
print('=' * 70)
print(f'  Apnea-ECG : {REG_APNEA}')
print(f'  UCD       : {REG_UCD}')
print(f'  Malik={int(MALIK_UMBRAL*100)}%  Mediana local={int(MEDIANA_UMBRAL*100)}%')
print()

# --- Apnea-ECG (todo el registro, con comparacion .qrs) ---
res_a = procesar_registro(REG_APNEA, DATA_DIR_APNEA, comparar_qrs=True)
print(f'{REG_APNEA} (Apnea-ECG): fs={res_a["fs"]} Hz, '
      f'{res_a["duracion_s"]/60:.1f} min, '
      f'{len(res_a["pt"]["picos_R"])} QRS')
if res_a.get('comparacion_qrs'):
    c = res_a['comparacion_qrs']
    print(f'  vs .qrs: sens={100*c["sensibilidad"]:.1f}% '
          f'prec={100*c["precision"]:.1f}%')

# --- UCD (del cache) ---
res_u = procesar_registro_ucd(REG_UCD, CACHE_DIR_UCD)
print(f'{REG_UCD} (UCD): fs={res_u["fs"]} Hz, '
      f'{res_u["duracion_s"]/60:.1f} min, '
      f'{len(res_u["pt"]["picos_R"])} QRS '
      f'(UCD no tiene .qrs de referencia)')
print()


# =============================================================================
# FIGURA 1: Pan-Tompkins etapa por etapa sobre UCD
# =============================================================================
# Espejo de lo que el TP anterior mostraba para a01, pero sobre la base nueva.
# Confirma que las 5 etapas (bandpass -> derivada -> cuadrado -> integrador ->
# deteccion con umbral local + floor) funcionan sobre la derivacion V2 de UCD.

fs_u = res_u['fs']
pt_u = res_u['pt']
ecg_u_filt = res_u['ecg_filtrado']

i0 = int(MINUTO_VIZ * 60 * fs_u)
i1 = i0 + int(VENTANA_VIZ_SEG * fs_u)
i1 = min(i1, len(ecg_u_filt))
t = np.arange(i0, i1) / fs_u

fig, axes = plt.subplots(5, 1, figsize=(16, 12), sharex=True)
axes[0].plot(t, ecg_u_filt[i0:i1], color='C1')
axes[0].set_ylabel('ECG filtrado')
axes[1].plot(t, pt_u['bandpass'][i0:i1], color='C0')
axes[1].set_ylabel(f'1) Bandpass\n{PT_BANDA_BAJA:.0f}-{PT_BANDA_ALTA:.0f} Hz')
axes[2].plot(t, pt_u['derivada'][i0:i1], color='C2')
axes[2].set_ylabel('2) Derivada')
axes[3].plot(t, pt_u['cuadrado'][i0:i1], color='C3')
axes[3].set_ylabel('3) Cuadrado')
axes[4].plot(t, pt_u['integrada'][i0:i1], color='C4')
axes[4].set_ylabel(f'4) Integrador\n{PT_VENTANA_INT_MS} ms')

# picos sobre integrador + umbral local (con floor)
mI = (pt_u['picos_int'] >= i0) & (pt_u['picos_int'] < i1)
axes[4].plot(pt_u['picos_int'][mI] / fs_u,
             pt_u['integrada'][pt_u['picos_int'][mI]],
             'rv', label='picos sobre integrador')
axes[4].plot(t, pt_u['umbral'][i0:i1], 'k--', alpha=0.6,
             label='umbral local')
axes[4].legend(loc='upper right')

# R refinados sobre el ECG filtrado
mR = (pt_u['picos_R'] >= i0) & (pt_u['picos_R'] < i1)
axes[0].plot(pt_u['picos_R'][mR] / fs_u,
             ecg_u_filt[pt_u['picos_R'][mR]],
             'ro', markersize=6, label='R refinados')
axes[0].legend(loc='upper right')

for ax in axes:
    ax.grid(True, alpha=0.3)
axes[-1].set_xlabel('Tiempo [s]')
fig.suptitle(f'Pan-Tompkins etapa por etapa sobre UCD ({REG_UCD}) '
             f'- {VENTANA_VIZ_SEG} s desde min {MINUTO_VIZ}')
plt.tight_layout(); plt.show()


# =============================================================================
# FIGURA 2: ECG filtrado + R detectados, las dos bases lado a lado
# =============================================================================
fs_a = res_a['fs']
ecg_a_filt = res_a['ecg_filtrado']
pt_a = res_a['pt']

# misma ventana temporal (en segundos) para ambas
def ventana(ecg, fs, min_ini, dur_seg):
    a = int(min_ini * 60 * fs)
    b = min(a + int(dur_seg * fs), len(ecg))
    return a, b

a0, a1 = ventana(ecg_a_filt, fs_a, MINUTO_VIZ, VENTANA_VIZ_SEG)
u0, u1 = ventana(ecg_u_filt, fs_u, MINUTO_VIZ, VENTANA_VIZ_SEG)

fig, axes = plt.subplots(2, 1, figsize=(16, 6))
# Apnea-ECG
ta = np.arange(a0, a1) / fs_a
axes[0].plot(ta, ecg_a_filt[a0:a1], color='C0', linewidth=0.9)
mRa = (pt_a['picos_R'] >= a0) & (pt_a['picos_R'] < a1)
axes[0].plot(pt_a['picos_R'][mRa] / fs_a, ecg_a_filt[pt_a['picos_R'][mRa]],
             'ro', markersize=6, label='R detectados')
axes[0].set_title(f'Apnea-ECG - {REG_APNEA} - ECG filtrado + R detectados')
axes[0].set_ylabel('mV'); axes[0].legend(loc='upper right')
axes[0].grid(True, alpha=0.3)
# UCD
tu = np.arange(u0, u1) / fs_u
axes[1].plot(tu, ecg_u_filt[u0:u1], color='C1', linewidth=0.9)
mRu = (pt_u['picos_R'] >= u0) & (pt_u['picos_R'] < u1)
axes[1].plot(pt_u['picos_R'][mRu] / fs_u, ecg_u_filt[pt_u['picos_R'][mRu]],
             'ro', markersize=6, label='R detectados')
axes[1].set_title(f'UCD - {REG_UCD} - ECG filtrado + R detectados')
axes[1].set_ylabel('mV'); axes[1].set_xlabel('Tiempo [s]')
axes[1].legend(loc='upper right'); axes[1].grid(True, alpha=0.3)
fig.suptitle('Deteccion de R: el mismo detector en las dos bases')
plt.tight_layout(); plt.show()


# =============================================================================
# FIGURA 3: tacograma R-R limpio, las dos bases lado a lado
# =============================================================================
# Muestra la serie RR completa de cada base, con los intervalos descartados
# por la limpieza marcados. Confirma que la serie queda fisiologica en ambas.

def tacograma(ax, res, fs, titulo, color):
    picos = res['pt']['picos_R']
    rr = res['rr_crudo']
    rr_interp = res['rr_interp']
    flags = res['flags']
    t_rr = picos[1:] / fs / 60  # cada RR ubicado en el pico que lo cierra
    ax.plot(t_rr, rr, '.', markersize=1.5, color=color, alpha=0.5,
            label='RR crudo')
    if flags['total'].any():
        # clip visual a 2 s para que las cruces de RR gigantes se vean en el borde
        rr_plot = np.clip(rr[flags['total']], 0, 1.98)
        ax.plot(t_rr[flags['total']], rr_plot, 'x',
                color='red', markersize=4,
                label=f'descartados (n={int(flags["total"].sum())})')
    ax.plot(t_rr, rr_interp, '-', color=color, linewidth=0.4, alpha=0.8)
    fc = 60 / np.median(rr_interp)
    ax.set_title(f'{titulo}  (FC med {fc:.0f} lpm, '
                 f'{100*flags["total"].sum()/len(rr):.2f}% descartados)')
    ax.set_ylabel('RR [s]'); ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=8)
    ax.set_ylim(0, 2)

fig, axes = plt.subplots(2, 1, figsize=(16, 7))
tacograma(axes[0], res_a, fs_a, f'Apnea-ECG - {REG_APNEA}', 'C0')
tacograma(axes[1], res_u, fs_u, f'UCD - {REG_UCD}', 'C1')
axes[1].set_xlabel('Tiempo [min]')
fig.suptitle('Tacograma R-R limpio: serie fisiologica en ambas bases')
plt.tight_layout(); plt.show()


# =============================================================================
# FIGURA 4: validacion manual de los RR descartados (ectopicos?)
# =============================================================================
# Responde la devolucion: para un sujeto con outliers, inspeccionamos cada RR
# descartado sobre el ECG con su contexto, para juzgar si era (a) un ectopico
# real, (b) un error de deteccion de QRS, o (c) un latido normal en zona de
# apnea que NO habia que descartar. Cruzamos ademas con la anotacion .apn de
# ese minuto para saber si el descarte cayo en un minuto apneico.

print(f'Validacion de ectopicos sobre {REG_OUTLIERS} ...')
try:
    from src.pipeline import cargar_anotaciones_apn
    res_o = procesar_registro(REG_OUTLIERS, DATA_DIR_APNEA, comparar_qrs=False)
    fs_o = res_o['fs']
    ecg_o = res_o['ecg_filtrado']
    picos_o = res_o['pt']['picos_R']
    rr_o = res_o['rr_crudo']
    flags_o = res_o['flags']
    rr_med = float(np.median(rr_o))

    # anotaciones de apnea por minuto (para saber si el descarte cae en apnea)
    try:
        samp_apn, sym_apn = cargar_anotaciones_apn(REG_OUTLIERS, DATA_DIR_APNEA)
        apn_por_min = {int(s / fs_o / 60): sym for s, sym in zip(samp_apn, sym_apn)}
    except Exception:
        apn_por_min = {}

    def clasificar_descarte(i_rr):
        """Heuristica para etiquetar el TIPO de RR descartado.

        - 'hueco/deteccion': RR mucho mayor que la mediana -> Pan-Tompkins
          perdio uno o mas latidos (no es un intervalo real).
        - 'ectopico': RR corto junto a un RR largo (prematuro + pausa
          compensatoria), patron tipico de latido ectopico.
        - 'CVHR?': RR moderadamente desviado en minuto apneico -> podria ser
          variacion ciclica de la FC (senal de apnea), NO artefacto.
        - 'otro': el resto.
        """
        rr_i = rr_o[i_rr]
        rr_prev = rr_o[i_rr - 1] if i_rr > 0 else rr_med
        rr_next = rr_o[i_rr + 1] if i_rr + 1 < len(rr_o) else rr_med
        centro_s = picos_o[i_rr + 1] if i_rr + 1 < len(picos_o) else picos_o[i_rr]
        minuto = int(centro_s / fs_o / 60)
        es_apnea = apn_por_min.get(minuto) == 'A'

        if rr_i > 1.8 * rr_med:
            return 'hueco/deteccion', minuto, es_apnea
        corto = rr_i < 0.7 * rr_med
        largo_vecino = (rr_prev > 1.2 * rr_med) or (rr_next > 1.2 * rr_med)
        largo = rr_i > 1.3 * rr_med
        corto_vecino = (rr_prev < 0.8 * rr_med) or (rr_next < 0.8 * rr_med)
        if (corto and largo_vecino) or (largo and corto_vecino):
            return 'ectopico', minuto, es_apnea
        if es_apnea and 0.7 * rr_med <= rr_i <= 1.5 * rr_med:
            return 'CVHR?', minuto, es_apnea
        return 'otro', minuto, es_apnea

    idx_desc = np.where(flags_o['total'])[0]
    print(f'  {REG_OUTLIERS}: {len(idx_desc)} RR descartados de {len(rr_o)} '
          f'({100*len(idx_desc)/len(rr_o):.2f}%)')

    if len(idx_desc) == 0:
        print(f'  {REG_OUTLIERS} no tiene RR descartados; probar otro REG_OUTLIERS.')
    else:
        desviacion = np.abs(rr_o[idx_desc] - rr_med)
        orden = np.argsort(desviacion)[::-1]
        idx_sel = idx_desc[orden[:N_ECTOPICOS_VIZ]]

        n = len(idx_sel)
        ncol = 3
        nrow = int(np.ceil(n / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(16, 3.4 * nrow))
        axes = np.atleast_1d(axes).ravel()

        half = int(VENTANA_ECTOPICO_SEG * fs_o)
        for k, i_rr in enumerate(idx_sel):
            ax = axes[k]
            # los DOS R que definen el intervalo descartado
            pR_ini = picos_o[i_rr]
            pR_fin = picos_o[min(i_rr + 1, len(picos_o) - 1)]
            # centrar en el PUNTO MEDIO del intervalo, para ver el intervalo
            # completo (clave en huecos largos por deteccion perdida)
            centro = (pR_ini + pR_fin) // 2
            s0 = max(0, centro - half)
            s1 = min(len(ecg_o), centro + half)
            t_seg = np.arange(s0, s1) / fs_o
            ax.plot(t_seg, ecg_o[s0:s1], color='C0', linewidth=0.8)

            # todos los R detectados en la ventana (negro)
            mR = (picos_o >= s0) & (picos_o < s1)
            ax.plot(picos_o[mR] / fs_o, ecg_o[picos_o[mR]], 'ko',
                    markersize=4, alpha=0.4)
            # los 2 R del intervalo descartado (rojo) + linea entre ellos
            for pj in (pR_ini, pR_fin):
                if s0 <= pj < s1:
                    ax.plot(pj / fs_o, ecg_o[pj], 'ro', markersize=9)
            ax.plot([pR_ini / fs_o, pR_fin / fs_o],
                    [ecg_o[pR_ini], ecg_o[pR_fin]], 'r--', alpha=0.5,
                    label='intervalo descartado')

            crits = []
            if flags_o['rango'][i_rr]:
                crits.append('rango')
            if flags_o['malik'][i_rr]:
                crits.append('Malik')
            if flags_o['mediana'][i_rr]:
                crits.append('mediana')
            tipo, minuto, es_apnea = clasificar_descarte(i_rr)
            apn = 'A' if es_apnea else apn_por_min.get(minuto, '?')
            ax.set_title(f'[{tipo}] RR={rr_o[i_rr]:.2f}s (med {rr_med:.2f}s)\n'
                         f'{"+".join(crits)} | min {minuto} apn={apn}',
                         fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('t [s]')
            ax.legend(loc='upper right', fontsize=7)

        for k in range(n, len(axes)):
            axes[k].axis('off')

        fig.suptitle(f'Validacion manual de RR descartados - {REG_OUTLIERS}\n'
                     f'(rojo = los 2 R del intervalo; [tipo] = clasificacion '
                     f'heuristica; apn=A = minuto apneico)')
        plt.tight_layout(); plt.show()

        # =====================================================================
        # Desglose por TIPO y por apnea (responde: se come la CVHR?)
        # =====================================================================
        from collections import Counter
        tipos = Counter()
        tipos_en_apnea = Counter()
        for i_rr in idx_desc:
            tipo, minuto, es_apnea = clasificar_descarte(i_rr)
            tipos[tipo] += 1
            if es_apnea:
                tipos_en_apnea[tipo] += 1

        print()
        print(f'  Desglose de los {len(idx_desc)} RR descartados por tipo:')
        for tipo in ['hueco/deteccion', 'ectopico', 'CVHR?', 'otro']:
            n_t = tipos.get(tipo, 0)
            n_ta = tipos_en_apnea.get(tipo, 0)
            print(f'    {tipo:18s}: {n_t:4d}  (de esos, {n_ta} en apnea)')
        print()
        n_cvhr = tipos.get('CVHR?', 0)
        if n_cvhr > 0:
            print(f'  ATENCION: {n_cvhr} descartes clasificados como CVHR? '
                  f'(RR moderado en minuto apneico).')
            print(f'  Podrian ser senal de apnea que el cleaning se come. Revisar '
                  f'los paneles [CVHR?].')
        else:
            print(f'  Ningun descarte quedo como CVHR?: el cleaning saca '
                  f'huecos/ectopicos, no senal de apnea. Bien.')
except Exception as e:
    import traceback
    print(f'  no se pudo hacer la validacion de ectopicos: {e}')
    traceback.print_exc()
print()


# =============================================================================
# Resumen
# =============================================================================
print('=' * 70)
print('Resumen comparativo')
print('=' * 70)
print(f"{'':16s}{'Apnea-ECG':>16s}{'UCD':>16s}")
print(f"{'registro':16s}{REG_APNEA:>16s}{REG_UCD:>16s}")
print(f"{'fs [Hz]':16s}{res_a['fs']:>16d}{res_u['fs']:>16d}")
print(f"{'duracion [min]':16s}{res_a['duracion_s']/60:>16.1f}"
      f"{res_u['duracion_s']/60:>16.1f}")
print(f"{'QRS':16s}{len(res_a['pt']['picos_R']):>16d}"
      f"{len(res_u['pt']['picos_R']):>16d}")
fc_a = 60/np.median(res_a['rr_interp'])
fc_u = 60/np.median(res_u['rr_interp'])
print(f"{'FC med [lpm]':16s}{fc_a:>16.1f}{fc_u:>16.1f}")
pa = 100*res_a['flags']['total'].sum()/len(res_a['rr_crudo'])
pu = 100*res_u['flags']['total'].sum()/len(res_u['rr_crudo'])
print(f"{'% descartado':16s}{pa:>16.2f}{pu:>16.2f}")
print('=' * 70)
print('Si el detector marca bien los R en UCD y el tacograma queda fisiologico,')
print('el pipeline esta listo para procesar ambas bases en batch (03b).')
print('=' * 70)