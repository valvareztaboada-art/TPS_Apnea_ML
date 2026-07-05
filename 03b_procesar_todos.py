# -*- coding: utf-8 -*-
"""
Procesamiento completo de TODOS los sujetos: Apnea-ECG + UCD
============================================================

Procesa las DOS bases aplicando el MISMO pipeline (filtros + Pan-Tompkins +
limpieza de RR, via procesar_ecg). Guarda un .npz por sujeto y un resumen.csv
por base.

  Apnea-ECG  -> cache/       (learning set a*/b*/c*; el test x* se descarta)
  UCD        -> cache_ucd/   (los 25 registros; ya tenian su cache del 00,
                             pero aca se re-genera CON el pipeline procesado:
                             picos R, serie RR, flags, y ademas labels/minuto)

Nota sobre el cache de UCD: el 00_cargar_ucd.py genero un cache con el ECG
remuestreado + labels. Este 03b lo SOBREESCRIBE con la version procesada
(picos R, RR, flags) que necesita el 04. Se conservan los labels por minuto.

Salida por sujeto (<cache>/<record>.npz):
  - fs, duracion_s
  - picos_R          : indices de muestra de cada R detectado
  - rr_crudo         : serie RR original (segundos)
  - rr_interp        : serie RR limpia + interpolada
  - flag_rango / flag_malik / flag_mediana / flag_total : booleanos
  - qrs_ref          : (solo Apnea-ECG) anotaciones .qrs de referencia
  - labels           : (solo UCD) etiqueta 'A'/'N' por minuto
  - base             : 'apnea' o 'ucd'
No se guarda la señal cruda ni la filtrada.
"""

import os
import sys
import time
import traceback

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from src.pipeline import (
    procesar_registro,          # Apnea-ECG (wfdb)
    procesar_registro_ucd,      # UCD (cache del 00)
    listar_registros,           # lista .hea de Apnea-ECG
    listar_registros_ucd,       # lista .npz de UCD
    clasificar_grupo,
    resumen_registro,
)


def guardar_npz(path, **arrays):
    """Guarda un .npz comprimido de forma robusta en Windows.

    Escribe primero a un archivo temporal y despues reemplaza el destino
    (os.replace es atomico). Esto evita el 'Errno 22 Invalid argument' que
    aparece en Windows al intentar sobreescribir un .npz que quedo bloqueado
    de una corrida previa (p.ej. ucddb025 al re-generar el cache del 00).
    """
    tmp = path + '.tmp'
    # limpiar un .tmp viejo si quedo colgado
    if os.path.exists(tmp):
        try:
            os.remove(tmp)
        except OSError:
            pass
    np.savez_compressed(tmp, **arrays)
    # np.savez_compressed agrega .npz si no lo tiene -> normalizar
    if not os.path.exists(tmp) and os.path.exists(tmp + '.npz'):
        tmp = tmp + '.npz'
    os.replace(tmp, path)


# =============================================================================
# Configuracion
# =============================================================================

DATA_DIR_APNEA = 'apnea-ecg-database-1.0.0'
CACHE_APNEA = 'cache'

# UCD tiene DOS caches distintos:
#   - CACHE_UCD_RAW: lo genera 00_cargar_ucd.py (ECG remuestreado + labels).
#     Es la ENTRADA de este script. NO se toca.
#   - CACHE_UCD: lo genera este 03b (picos R + RR + flags + labels), sin ECG.
#     Es la SALIDA, la que consume el 04. Asi correr el 03b varias veces no
#     rompe nada (no pisa el cache que tiene el ECG).
CACHE_UCD_RAW = 'cache_ucd'
CACHE_UCD = 'cache_ucd_proc'

# Que bases procesar
PROCESAR_APNEA = True
PROCESAR_UCD = True

# Apnea-ECG: procesar solo el learning set (a/b/c). El test set (x) no tiene
# anotaciones utiles y lo descartamos del trabajo. Ademas, como es clasificacion
# binaria, la clase B (borderline) se puede excluir aca o mas adelante; por
# ahora la procesamos y se filtra en el paso de ML.
GRUPOS_APNEA = ('apnea', 'borderline', 'control')   # excluye 'test' (x*)

# Para pruebas: correr solo un subconjunto (ej ['a01','c01','ucddb002']).
SUBSET = None


# =============================================================================
# Procesamiento de una base
# =============================================================================

def procesar_base_apnea(registros, cache_dir):
    """Procesa registros de Apnea-ECG y guarda .npz + devuelve filas resumen."""
    os.makedirs(cache_dir, exist_ok=True)
    filas = []
    for i, r in enumerate(registros, 1):
        t0 = time.time()
        print(f'  [APNEA {i:2d}/{len(registros)}] {r:6s} ...', end=' ', flush=True)
        try:
            result = procesar_registro(r, DATA_DIR_APNEA, comparar_qrs=True)
            guardar_npz(
                os.path.join(cache_dir, f'{r}.npz'),
                fs=result['fs'],
                duracion_s=result['duracion_s'],
                picos_R=result['pt']['picos_R'],
                rr_crudo=result['rr_crudo'],
                rr_interp=result['rr_interp'],
                flag_rango=result['flags']['rango'],
                flag_malik=result['flags']['malik'],
                flag_mediana=result['flags']['mediana'],
                flag_total=result['flags']['total'],
                qrs_ref=result.get('qrs_ref') if result.get('qrs_ref') is not None
                         else np.array([], dtype=int),
                base='apnea',
            )
            fila = resumen_registro(r, result)
            fila['base'] = 'apnea'
            fila['procesado_ok'] = True
            fila['error'] = None
            fila['tiempo_s'] = time.time() - t0
            filas.append(fila)

            n_qrs = len(result['pt']['picos_R'])
            pct = 100 * result['flags']['total'].sum() / max(1, len(result['rr_crudo']))
            print(f'ok ({n_qrs:5d} QRS, {pct:.2f}% out, {time.time()-t0:.1f}s)')
        except Exception as e:
            filas.append({'record': r, 'base': 'apnea',
                          'grupo': clasificar_grupo(r),
                          'procesado_ok': False, 'error': str(e),
                          'tiempo_s': time.time() - t0})
            print(f'ERROR: {e}')
            traceback.print_exc(limit=2)
    return filas


def procesar_base_ucd(registros, cache_raw, cache_out):
    """Procesa registros de UCD.

    Lee el ECG del cache_raw (generado por 00_cargar_ucd.py) y escribe el
    resultado procesado (picos, RR, flags, labels) en cache_out. No sobreescribe
    el cache_raw, asi el script es idempotente (se puede correr varias veces).
    """
    os.makedirs(cache_out, exist_ok=True)
    filas = []
    for i, r in enumerate(registros, 1):
        t0 = time.time()
        print(f'  [UCD   {i:2d}/{len(registros)}] {r:10s} ...', end=' ', flush=True)
        try:
            result = procesar_registro_ucd(r, cache_raw)
            labels = result.get('labels', np.array([], dtype='<U1'))
            guardar_npz(
                os.path.join(cache_out, f'{r}.npz'),
                fs=result['fs'],
                duracion_s=result['duracion_s'],
                picos_R=result['pt']['picos_R'],
                rr_crudo=result['rr_crudo'],
                rr_interp=result['rr_interp'],
                flag_rango=result['flags']['rango'],
                flag_malik=result['flags']['malik'],
                flag_mediana=result['flags']['mediana'],
                flag_total=result['flags']['total'],
                qrs_ref=np.array([], dtype=int),   # UCD no tiene .qrs
                labels=labels,
                base='ucd',
            )
            # resumen (UCD no tiene grupo a/b/c; usamos % apnea del sujeto)
            rr = result['rr_crudo']
            rr_interp = result['rr_interp']
            flags = result['flags']
            n_a = int((labels == 'A').sum()) if len(labels) else 0
            n_min = len(labels)
            fila = {
                'record': r, 'base': 'ucd', 'grupo': 'ucd',
                'duracion_h': result['duracion_s'] / 3600,
                'fs': result['fs'],
                'n_qrs': len(result['pt']['picos_R']),
                'fc_mediana_lpm': float(60 / np.median(rr_interp)) if len(rr_interp) else None,
                'rr_mediano_s': float(np.median(rr_interp)) if len(rr_interp) else None,
                'rr_std_s': float(np.std(rr_interp)) if len(rr_interp) else None,
                'n_outliers_rango': int(flags['rango'].sum()),
                'n_outliers_malik': int(flags['malik'].sum()),
                'n_outliers_mediana': int(flags['mediana'].sum()),
                'n_outliers_total': int(flags['total'].sum()),
                'pct_outliers': float(100 * flags['total'].sum() / len(rr)) if len(rr) else None,
                'n_min': n_min, 'n_apnea': n_a,
                'pct_apnea': float(100 * n_a / n_min) if n_min else None,
                'qrs_sens_pct': None, 'qrs_prec_pct': None,
                'qrs_tp': None, 'qrs_fp': None, 'qrs_fn': None,
                'procesado_ok': True, 'error': None,
                'tiempo_s': time.time() - t0,
            }
            filas.append(fila)
            pct = 100 * flags['total'].sum() / max(1, len(rr))
            print(f'ok ({len(result["pt"]["picos_R"]):5d} QRS, {pct:.2f}% out, '
                  f'{n_a}/{n_min} A, {time.time()-t0:.1f}s)')
        except Exception as e:
            filas.append({'record': r, 'base': 'ucd', 'grupo': 'ucd',
                          'procesado_ok': False, 'error': str(e),
                          'tiempo_s': time.time() - t0})
            print(f'ERROR: {e}')
            traceback.print_exc(limit=2)
    return filas


# =============================================================================
# Main
# =============================================================================

def main():
    t_inicio = time.time()
    todas_filas = []

    # -------- Apnea-ECG --------
    if PROCESAR_APNEA:
        print('=' * 70)
        print('BASE 1: Apnea-ECG (learning set)')
        print('=' * 70)
        registros_a = listar_registros(DATA_DIR_APNEA)
        registros_a = [r for r in registros_a
                       if clasificar_grupo(r) in GRUPOS_APNEA]
        if SUBSET is not None:
            registros_a = [r for r in registros_a if r in SUBSET]
        print(f'Registros Apnea-ECG a procesar: {len(registros_a)} '
              f'(grupos: {GRUPOS_APNEA})')
        print(f'Cache: {CACHE_APNEA}/')
        print()
        todas_filas += procesar_base_apnea(registros_a, CACHE_APNEA)
        print()

    # -------- UCD --------
    if PROCESAR_UCD:
        print('=' * 70)
        print('BASE 2: UCD (test externo)')
        print('=' * 70)
        try:
            registros_u = listar_registros_ucd(CACHE_UCD_RAW)
        except FileNotFoundError as e:
            print(f'ERROR: {e}')
            print('Corre primero 00_cargar_ucd.py para generar el cache de UCD.')
            registros_u = []
        if SUBSET is not None:
            registros_u = [r for r in registros_u if r in SUBSET]
        print(f'Registros UCD a procesar: {len(registros_u)}')
        print(f'Cache entrada (del 00): {CACHE_UCD_RAW}/')
        print(f'Cache salida (procesado): {CACHE_UCD}/')
        print()
        todas_filas += procesar_base_ucd(registros_u, CACHE_UCD_RAW, CACHE_UCD)
        print()

    # -------- Resumen global --------
    df = pd.DataFrame(todas_filas)
    # guardar un resumen por base
    if PROCESAR_APNEA:
        df[df['base'] == 'apnea'].to_csv(
            os.path.join(CACHE_APNEA, 'resumen.csv'), index=False)
    if PROCESAR_UCD:
        df[df['base'] == 'ucd'].to_csv(
            os.path.join(CACHE_UCD, 'resumen.csv'), index=False)

    print('=' * 70)
    print(f'Procesamiento completo en {time.time()-t_inicio:.1f} s')
    if PROCESAR_APNEA:
        print(f'  Apnea-ECG -> {CACHE_APNEA}/resumen.csv')
    if PROCESAR_UCD:
        print(f'  UCD       -> {CACHE_UCD}/resumen.csv')
    print('=' * 70)

    # tabla por grupo (Apnea-ECG)
    ok = df[df['procesado_ok']].copy()
    if PROCESAR_APNEA and (ok['base'] == 'apnea').any():
        oka = ok[ok['base'] == 'apnea']
        print()
        print('Apnea-ECG - resumen por grupo:')
        agg = oka.groupby('grupo').agg(
            n_sujetos=('record', 'count'),
            duracion_h_media=('duracion_h', 'mean'),
            fc_mediana=('fc_mediana_lpm', 'mean'),
            pct_outliers_med=('pct_outliers', 'median'),
            pct_outliers_max=('pct_outliers', 'max'),
            qrs_sens=('qrs_sens_pct', 'mean'),
            qrs_prec=('qrs_prec_pct', 'mean'),
        ).round(2)
        print(agg.to_string())

    # tabla UCD
    if PROCESAR_UCD and (ok['base'] == 'ucd').any():
        oku = ok[ok['base'] == 'ucd']
        print()
        print('UCD - resumen:')
        print(f'  sujetos          : {len(oku)}')
        print(f'  FC mediana media : {oku["fc_mediana_lpm"].mean():.1f} lpm')
        print(f'  % outliers (med) : {oku["pct_outliers"].median():.2f}%')
        print(f'  % outliers (max) : {oku["pct_outliers"].max():.2f}%')
        print(f'  minutos totales  : {int(oku["n_min"].sum())}')
        print(f'  minutos apnea    : {int(oku["n_apnea"].sum())} '
              f'({100*oku["n_apnea"].sum()/oku["n_min"].sum():.1f}%)')

    # alertas de sujetos a revisar (ambas bases)
    print()
    print('Sujetos para revisar (outliers > 5% o sens < 95%):')
    sosp = ok[(ok['pct_outliers'] > 5) |
              (ok['qrs_sens_pct'].notna() & (ok['qrs_sens_pct'] < 95))]
    if len(sosp) == 0:
        print('  ninguno, todo dentro de rangos esperados')
    else:
        cols = [c for c in ['record', 'base', 'grupo', 'pct_outliers',
                            'qrs_sens_pct', 'qrs_prec_pct'] if c in sosp.columns]
        print(sosp[cols].to_string(index=False))

    # errores
    n_err = (~df['procesado_ok']).sum()
    if n_err > 0:
        print()
        print(f'ERRORES: {n_err} sujetos fallaron')
        print(df[~df['procesado_ok']][['record', 'base', 'error']].to_string(index=False))


if __name__ == '__main__':
    main()