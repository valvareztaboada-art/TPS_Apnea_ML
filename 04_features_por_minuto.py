# -*- coding: utf-8 -*-
"""
Calculo de features por minuto: Apnea-ECG + UCD
================================================

Lee los caches procesados por 03b y calcula features HRV + EDR + wavelet por
minuto, para las DOS bases. Genera un CSV de features por base.

  Apnea-ECG -> cache/features_apnea.csv     (desarrollo: a/b/c)
  UCD       -> cache_ucd_proc/features_ucd.csv  (test externo)

Decisiones (acordadas):
  - Se EXCLUYEN sujetos con > PCT_OUTLIERS_MAX % de RR no fisiologicos
    (fallo sistematico de deteccion o arritmia sostenida que invalida la HRV).
    Con 20% cae solo c04 (Apnea-ECG); ningun UCD supera el umbral.
  - Se calculan features de TODOS los grupos validos (a/b/c). La clase B NO se
    excluye aca: cada minuto de B tiene su etiqueta A/N valida y sirve para el
    clasificador per-minuto. La decision de usar B o no (y de excluirla para la
    evaluacion per-sujeto) se toma en el 05. Por eso guardamos 'record', 'grupo'
    y 'base' en cada fila.
  - EDR necesita el ECG (amplitud de R): se recarga al vuelo (wfdb para Apnea,
    cache RAW del 00 para UCD), se filtra igual, y se calculan las amplitudes.

Pre-requisito: haber corrido 00_cargar_ucd.py y 03b_procesar_todos.py.
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
    listar_registros,
    listar_registros_ucd,
    clasificar_grupo,
    cargar_anotaciones_apn,
    cargar_ecg,
    filtrar_ecg_general,
)
from src.features import features_por_minuto, amplitudes_R


# =============================================================================
# Configuracion
# =============================================================================

DATA_DIR_APNEA = 'apnea-ecg-database-1.0.0'
CACHE_APNEA = 'cache'

CACHE_UCD_RAW = 'cache_ucd'          # tiene el ECG (del 00)
CACHE_UCD_PROC = 'cache_ucd_proc'    # tiene picos/RR/labels (del 03b)

# Umbral de exclusion por RR no fisiologicos (fallo de deteccion / arritmia)
PCT_OUTLIERS_MAX = 20.0

PROCESAR_APNEA = True
PROCESAR_UCD = True

# Para pruebas: solo un subconjunto (ej ['a01','c01','ucddb002']).
SUBSET = None


# =============================================================================
# Helpers
# =============================================================================

def pct_outliers(flag_total, rr_crudo):
    n = len(rr_crudo)
    return 100.0 * int(np.sum(flag_total)) / n if n else 0.0


def ecg_amplitudes_apnea(record, picos_R, fs):
    """Carga el ECG de Apnea-ECG con wfdb, lo filtra y da amplitudes de R."""
    ecg_raw, _fs, _ = cargar_ecg(record, DATA_DIR_APNEA)
    ecg_filt = filtrar_ecg_general(ecg_raw, fs)
    return amplitudes_R(ecg_filt, picos_R, fs)


def ecg_amplitudes_ucd(record, picos_R, fs, cache_raw=CACHE_UCD_RAW):
    """Carga el ECG de UCD del cache RAW del 00, lo filtra y da amplitudes."""
    path = os.path.join(cache_raw, f'{record}.npz')
    data = np.load(path, allow_pickle=True)
    if 'ecg' not in data.files:
        # el cache raw deberia tener el ecg; si no, no hay EDR
        return None
    ecg_raw = data['ecg'].astype(float)
    ecg_filt = filtrar_ecg_general(ecg_raw, fs)
    return amplitudes_R(ecg_filt, picos_R, fs)


# =============================================================================
# Procesamiento de una base
# =============================================================================

def procesar_features_apnea(registros):
    dfs = []
    excluidos = []
    for i, r in enumerate(registros, 1):
        cache_path = os.path.join(CACHE_APNEA, f'{r}.npz')
        if not os.path.exists(cache_path):
            print(f'  [{i:2d}/{len(registros)}] {r:6s}: sin cache, salteo')
            continue
        t0 = time.time()
        print(f'  [{i:2d}/{len(registros)}] {r:6s} ...', end=' ', flush=True)
        try:
            data = np.load(cache_path, allow_pickle=True)
            pct = pct_outliers(data['flag_total'], data['rr_crudo'])
            if pct > PCT_OUTLIERS_MAX:
                excluidos.append((r, pct))
                print(f'EXCLUIDO ({pct:.1f}% outliers > {PCT_OUTLIERS_MAX}%)')
                continue

            picos_R = data['picos_R']
            fs = int(data['fs'])
            amps_R = ecg_amplitudes_apnea(r, picos_R, fs)

            df = features_por_minuto(
                picos_R=picos_R, rr_interp=data['rr_interp'], fs=fs,
                duracion_s=float(data['duracion_s']),
                amplitudes_picos=amps_R, incluir_wavelet=True)
            df['record'] = r
            df['grupo'] = clasificar_grupo(r)
            df['base'] = 'apnea'

            # etiquetas .apn
            try:
                samples, symbols = cargar_anotaciones_apn(r, DATA_DIR_APNEA)
                labels = {int(s / fs / 60): sym for s, sym in zip(samples, symbols)}
                df['label'] = df['minute'].map(labels)
            except (FileNotFoundError, Exception):
                df['label'] = None

            dfs.append(df)
            n_lbl = int(df['label'].isin(['A', 'N']).sum())
            print(f'ok ({len(df)} min, {n_lbl} label, {time.time()-t0:.1f}s)')
        except Exception as e:
            print(f'ERROR: {e}')
            traceback.print_exc(limit=2)
    return dfs, excluidos


def procesar_features_ucd(registros):
    dfs = []
    excluidos = []
    for i, r in enumerate(registros, 1):
        cache_path = os.path.join(CACHE_UCD_PROC, f'{r}.npz')
        if not os.path.exists(cache_path):
            print(f'  [{i:2d}/{len(registros)}] {r:10s}: sin cache, salteo')
            continue
        t0 = time.time()
        print(f'  [{i:2d}/{len(registros)}] {r:10s} ...', end=' ', flush=True)
        try:
            data = np.load(cache_path, allow_pickle=True)
            pct = pct_outliers(data['flag_total'], data['rr_crudo'])
            if pct > PCT_OUTLIERS_MAX:
                excluidos.append((r, pct))
                print(f'EXCLUIDO ({pct:.1f}% outliers > {PCT_OUTLIERS_MAX}%)')
                continue

            picos_R = data['picos_R']
            fs = int(data['fs'])
            amps_R = ecg_amplitudes_ucd(r, picos_R, fs)

            df = features_por_minuto(
                picos_R=picos_R, rr_interp=data['rr_interp'], fs=fs,
                duracion_s=float(data['duracion_s']),
                amplitudes_picos=amps_R, incluir_wavelet=True)
            df['record'] = r
            df['grupo'] = 'ucd'
            df['base'] = 'ucd'

            # etiquetas por minuto (del cache: 'A'/'N')
            labels = data['labels']
            # alinear al numero de minutos del df
            lab_map = {m: labels[m] for m in range(min(len(labels), len(df)))}
            df['label'] = df['minute'].map(lab_map)

            dfs.append(df)
            n_lbl = int(df['label'].isin(['A', 'N']).sum())
            print(f'ok ({len(df)} min, {n_lbl} label, {time.time()-t0:.1f}s)')
        except Exception as e:
            print(f'ERROR: {e}')
            traceback.print_exc(limit=2)
    return dfs, excluidos


# =============================================================================
# Reporte comparativo Normal vs Apnea
# =============================================================================

def reporte_features(df, titulo):
    train = df[df['label'].isin(['A', 'N'])].copy()
    if len(train) == 0:
        return
    print(f'\n{titulo} - Normal vs Apnea (medias):')
    cols = ['mean_hr', 'sdnn', 'rmssd', 'lf_power', 'hf_power', 'lf_hf_ratio',
            'cvhr_power', 'cvhr_norm']
    edr = [c for c in ['edr_resp_power', 'edr_apnea_power', 'edr_resp_norm',
                       'edr_apnea_norm', 'edr_apnea_resp_ratio']
           if c in train.columns]
    wav = [c for c in train.columns if c.startswith('wav_')]
    cols = [c for c in cols + edr + wav if c in train.columns]
    comp = train.groupby('label')[cols].mean().round(4)
    print(comp.to_string())


# =============================================================================
# Main
# =============================================================================

def main():
    t_inicio = time.time()

    if PROCESAR_APNEA:
        print('=' * 70)
        print('BASE 1: Apnea-ECG - features')
        print('=' * 70)
        regs = listar_registros(DATA_DIR_APNEA)
        regs = [r for r in regs if clasificar_grupo(r) in
                ('apnea', 'borderline', 'control')]
        if SUBSET is not None:
            regs = [r for r in regs if r in SUBSET]
        print(f'Sujetos: {len(regs)} | umbral exclusion: {PCT_OUTLIERS_MAX}% outliers')
        print()
        dfs_a, excl_a = procesar_features_apnea(regs)
        if dfs_a:
            big_a = pd.concat(dfs_a, ignore_index=True)
            base_cols = ['record', 'grupo', 'base', 'minute', 'label']
            feat_cols = [c for c in big_a.columns if c not in base_cols]
            big_a = big_a[base_cols + feat_cols]
            out = os.path.join(CACHE_APNEA, 'features_apnea.csv')
            big_a.to_csv(out, index=False)
            print(f'\n  guardado: {out}  ({len(big_a)} filas, {big_a.shape[1]} cols)')
            if excl_a:
                print(f'  excluidos por outliers: {excl_a}')
            reporte_features(big_a, 'Apnea-ECG')
        print()

    if PROCESAR_UCD:
        print('=' * 70)
        print('BASE 2: UCD - features')
        print('=' * 70)
        try:
            regs_u = listar_registros_ucd(CACHE_UCD_PROC)
        except FileNotFoundError as e:
            print(f'ERROR: {e}\nCorre 03b_procesar_todos.py primero.')
            regs_u = []
        if SUBSET is not None:
            regs_u = [r for r in regs_u if r in SUBSET]
        print(f'Sujetos: {len(regs_u)} | umbral exclusion: {PCT_OUTLIERS_MAX}% outliers')
        print()
        dfs_u, excl_u = procesar_features_ucd(regs_u)
        if dfs_u:
            big_u = pd.concat(dfs_u, ignore_index=True)
            base_cols = ['record', 'grupo', 'base', 'minute', 'label']
            feat_cols = [c for c in big_u.columns if c not in base_cols]
            big_u = big_u[base_cols + feat_cols]
            out = os.path.join(CACHE_UCD_PROC, 'features_ucd.csv')
            big_u.to_csv(out, index=False)
            print(f'\n  guardado: {out}  ({len(big_u)} filas, {big_u.shape[1]} cols)')
            if excl_u:
                print(f'  excluidos por outliers: {excl_u}')
            reporte_features(big_u, 'UCD')
        print()

    print('=' * 70)
    print(f'Done en {time.time()-t_inicio:.1f} s')
    print('=' * 70)


if __name__ == '__main__':
    main()
