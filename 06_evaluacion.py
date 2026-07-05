# -*- coding: utf-8 -*-
"""
Evaluacion final: test externo (UCD) + clasificacion per-sujeto (Apnea-ECG)
============================================================================

  PARTE 1 - Test externo per-minuto en UCD (cross-database):
    Entrena con TODO Apnea-ECG (learning set, incluida B para per-minuto) y
    predice sobre UCD. Reporta la caida por cambio de dominio. Compara:
      - modelo REDUCIDO con EDR   (el oficial)
      - modelo SIN EDR            

  PARTE 2 - Clasificacion per-sujeto (solo Apnea-ECG, solo A y C):
    Usa las predicciones out-of-fold del 05 (sin leakage) para estimar, por
    sujeto, la fraccion de minutos apneicos -> "AHI estimado". Calibra un umbral
    (maximizando balanced accuracy) para clasificar sujeto apnea/control.
    La clase B se EXCLUYE aca (ambigua a nivel sujeto), como se acordo.

Requisitos: cache/features_apnea.csv, cache_ucd_proc/features_ucd.csv,
cache/oof_predicciones.csv (los generan el 04 y el 05).
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (roc_auc_score, f1_score, confusion_matrix,
                             roc_curve, precision_recall_fscore_support,
                             balanced_accuracy_score)


# =============================================================================
# Configuracion
# =============================================================================

CACHE = 'cache'
CACHE_UCD = 'cache_ucd_proc'
FEAT_APNEA = os.path.join(CACHE, 'features_apnea.csv')
FEAT_UCD = os.path.join(CACHE_UCD, 'features_ucd.csv')
OOF_CSV = os.path.join(CACHE, 'oof_predicciones.csv')

RANDOM_STATE = 42

# set reducido (del 05). Con EDR:
FEATS_REDUCIDO = ['edr_apnea_power', 'wav_energy_L5', 'wav_energy_L3',
                  'edr_apnea_norm', 'edr_resp_power', 'wav_entropy',
                  'edr_apnea_resp_ratio', 'cvhr_power', 'cvhr_norm', 'sd_hr',
                  'vlf_power', 'sdnn', 'edr_resp_norm', 'wav_energy_L1',
                  'lf_power', 'wav_energy_L4']
FEATS_EDR = ['edr_resp_power', 'edr_apnea_power', 'edr_resp_norm',
             'edr_apnea_norm', 'edr_apnea_resp_ratio']
FEATS_SIN_EDR = [f for f in FEATS_REDUCIDO if f not in FEATS_EDR]


def hacer_modelo():
    """Ensemble (mismo que gano en el 05) con imputacion + escalado."""
    def pipe(clf):
        return Pipeline([('imputer', SimpleImputer(strategy='median')),
                         ('scaler', StandardScaler()), ('clf', clf)])
    estimadores = [
        ('LogReg', pipe(LogisticRegression(max_iter=1000, class_weight='balanced',
                                           random_state=RANDOM_STATE))),
        ('RandomForest', pipe(RandomForestClassifier(
            n_estimators=200, class_weight='balanced', n_jobs=-1,
            random_state=RANDOM_STATE))),
        ('GradBoost', pipe(GradientBoostingClassifier(
            n_estimators=200, random_state=RANDOM_STATE))),
        ('SVM-RBF', pipe(SVC(kernel='rbf', class_weight='balanced',
                             probability=True, random_state=RANDOM_STATE))),
    ]
    return VotingClassifier(estimators=estimadores, voting='soft', n_jobs=-1)


def metricas_binarias(y, proba, umbral=0.5):
    pred = (proba >= umbral).astype(int)
    p, r, f, _ = precision_recall_fscore_support(y, pred, average=None,
                                                 labels=[0, 1], zero_division=0)
    return {
        'auc': roc_auc_score(y, proba),
        'f1_macro': f1_score(y, pred, average='macro'),
        'f1_apnea': f1_score(y, pred, pos_label=1, zero_division=0),
        'rec_A': r[1], 'prec_A': p[1], 'rec_N': r[0],
        'pred': pred,
    }


def umbral_youden(y, proba):
    """Umbral del punto de Youden (maximiza TPR - FPR sobre la ROC)."""
    fpr, tpr, thr = roc_curve(y, proba)
    j = tpr - fpr
    return float(thr[np.argmax(j)])


def umbral_max_f1(y, proba):
    """Umbral que maximiza F1-macro, buscando sobre una grilla."""
    grid = np.linspace(0.05, 0.95, 91)
    mejor_u, mejor_f1 = 0.5, 0
    for u in grid:
        f = f1_score(y, (proba >= u).astype(int), average='macro',
                     zero_division=0)
        if f > mejor_f1:
            mejor_f1, mejor_u = f, u
    return float(mejor_u)


# =============================================================================
# PARTE 1: test externo en UCD
# =============================================================================

def parte1_test_ucd():
    print('=' * 70)
    print('PARTE 1: Test externo per-minuto en UCD (cross-database)')
    print('=' * 70)
 
    df_a = pd.read_csv(FEAT_APNEA)
    df_a = df_a[df_a['label'].isin(['A', 'N'])].copy()   # incluye B
    ya = (df_a['label'] == 'A').astype(int).values
 
    df_u = pd.read_csv(FEAT_UCD)
    df_u = df_u[df_u['label'].isin(['A', 'N'])].copy()
    yu = (df_u['label'] == 'A').astype(int).values
 
    print(f'Train (Apnea-ECG): {len(df_a)} min | Test (UCD): {len(df_u)} min')
    print()
 
    resultados = {}
    for nombre, feats in [('reducido CON EDR', FEATS_REDUCIDO),
                          ('SIN EDR', FEATS_SIN_EDR)]:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            modelo = hacer_modelo()
            modelo.fit(df_a[feats].values, ya)          # entrena con Apnea-ECG
            proba_u = modelo.predict_proba(df_u[feats].values)[:, 1]  # predice UCD
        met = metricas_binarias(yu, proba_u)
        resultados[nombre] = {'proba': proba_u, 'met': met, 'feats': feats}
 
        # Guardar las predicciones per-minuto del modelo OFICIAL (reducido CON
        # EDR) sobre UCD, para que la interfaz las muestre. Mismas columnas que
        # oof_predicciones.csv de Apnea-ECG, para leerlas igual en las dos bases.
        if nombre == 'reducido CON EDR':
            pred_ucd = df_u[['record', 'minute', 'label']].copy()
            pred_ucd['y_true'] = yu
            pred_ucd['proba_apnea'] = proba_u
            pred_ucd['pred'] = (proba_u >= 0.5).astype(int)
            pred_ucd.to_csv(os.path.join(CACHE_UCD, 'predicciones_ucd.csv'),
                            index=False)
 
        print(f'[{nombre}] ({len(feats)} feats)')
        print(f'   umbral 0.5      -> AUC={met["auc"]:.3f}  F1-macro={met["f1_macro"]:.3f}  '
              f'recall_A={met["rec_A"]:.3f}  prec_A={met["prec_A"]:.3f}')
        # umbral recalibrado para UCD (Youden y max-F1). NOTA: recalibrar mirando
        # UCD usa el test, asi que esto es "rendimiento ALCANZABLE con
        # recalibracion del umbral", no un resultado ciego. Se reporta como tal.
        u_youden = umbral_youden(yu, proba_u)
        u_f1 = umbral_max_f1(yu, proba_u)
        met_y = metricas_binarias(yu, proba_u, umbral=u_youden)
        met_f = metricas_binarias(yu, proba_u, umbral=u_f1)
        resultados[nombre]['met_youden'] = met_y
        resultados[nombre]['u_youden'] = u_youden
        print(f'   umbral Youden ({u_youden:.2f}) -> F1-macro={met_y["f1_macro"]:.3f}  '
              f'recall_A={met_y["rec_A"]:.3f}  prec_A={met_y["prec_A"]:.3f}')
        print(f'   umbral maxF1  ({u_f1:.2f}) -> F1-macro={met_f["f1_macro"]:.3f}  '
              f'recall_A={met_f["rec_A"]:.3f}  prec_A={met_f["prec_A"]:.3f}')
        print()
 
    # comparar con el rendimiento interno (referencia del 05)
    print('Interpretacion:')
    auc_con = resultados['reducido CON EDR']['met']['auc']
    auc_sin = resultados['SIN EDR']['met']['auc']
    print(f'  - AUC interno (Apnea-ECG, del 05): ~0.93')
    print(f'  - AUC externo UCD con EDR : {auc_con:.3f}  (caida por cambio de dominio)')
    print(f'  - AUC externo UCD sin EDR : {auc_sin:.3f}')
    if auc_sin > auc_con:
        print(f'  -> SIN EDR transfiere MEJOR (+{auc_sin-auc_con:.3f}): el EDR no '
              f'generaliza bien entre derivaciones distintas.')
    else:
        print(f'  -> el EDR sigue ayudando en UCD ({auc_con-auc_sin:+.3f}): conserva '
              f'poder discriminante transferible (contra la hipotesis inicial).')
    print(f'  - El AUC (independiente del umbral) mide el ordenamiento; el recall '
          f'bajo con umbral 0.5 indica que el')
    print(f'    corte esta descalibrado para UCD. Con umbral recalibrado (Youden/'
          f'maxF1) el recall sube: el modelo')
    print(f'    ordena bien los minutos, pero el punto de decision optimo difiere '
          f'entre bases.')
    print()
    return df_u, yu, resultados
 

# =============================================================================
# PARTE 2: clasificacion per-sujeto (Apnea-ECG, solo A y C)
# =============================================================================

def calibrar_umbral_ahi(ahi, es_apnea):
    """Busca el umbral de AHI que maximiza balanced accuracy per-sujeto."""
    candidatos = np.linspace(ahi.min(), ahi.max(), 200)
    mejor_u, mejor_ba = candidatos[0], 0
    for u in candidatos:
        pred = (ahi >= u).astype(int)
        ba = balanced_accuracy_score(es_apnea, pred)
        if ba > mejor_ba:
            mejor_ba, mejor_u = ba, u
    return mejor_u, mejor_ba


def parte2_per_sujeto():
    print('=' * 70)
    print('PARTE 2: Clasificacion per-sujeto (Apnea-ECG, solo A y C)')
    print('=' * 70)

    oof = pd.read_csv(OOF_CSV)
    # solo A y C a nivel SUJETO (excluir borderline)
    oof = oof[oof['grupo'].isin(['apnea', 'control'])].copy()

    # AHI estimado = % de minutos predichos como apnea por sujeto
    # (proxy del AHI; el AHI real es eventos/hora pero la fraccion de minutos
    #  apneicos es monotona con el, sirve para clasificar)
    por_sujeto = oof.groupby('record').agg(
        grupo=('grupo', 'first'),
        n_min=('pred', 'count'),
        min_apnea_pred=('pred', 'sum'),
        min_apnea_real=('y_true', 'sum'),
    ).reset_index()
    por_sujeto['ahi_estimado'] = 100 * por_sujeto['min_apnea_pred'] / por_sujeto['n_min']
    por_sujeto['ahi_real_pct'] = 100 * por_sujeto['min_apnea_real'] / por_sujeto['n_min']
    por_sujeto['es_apnea'] = (por_sujeto['grupo'] == 'apnea').astype(int)

    print(f'Sujetos: {len(por_sujeto)} '
          f'(A={int(por_sujeto["es_apnea"].sum())}, '
          f'C={int((1-por_sujeto["es_apnea"]).sum())})')

    ahi = por_sujeto['ahi_estimado'].values
    es_apnea = por_sujeto['es_apnea'].values
    umbral, ba = calibrar_umbral_ahi(ahi, es_apnea)

    por_sujeto['pred_sujeto'] = (ahi >= umbral).astype(int)
    cm = confusion_matrix(es_apnea, por_sujeto['pred_sujeto'])
    acc = (por_sujeto['pred_sujeto'] == es_apnea).mean()

    print(f'\nUmbral AHI calibrado: {umbral:.1f}% de minutos apneicos')
    print(f'Balanced accuracy per-sujeto: {ba:.3f}')
    print(f'Accuracy per-sujeto: {acc:.3f}')
    print(f'\nMatriz de confusion per-sujeto:')
    print(f'              pred_C  pred_A')
    print(f'  real_C    {cm[0,0]:6d}  {cm[0,1]:6d}')
    print(f'  real_A    {cm[1,0]:6d}  {cm[1,1]:6d}')

    # sujetos mal clasificados
    mal = por_sujeto[por_sujeto['pred_sujeto'] != por_sujeto['es_apnea']]
    if len(mal):
        print(f'\nSujetos mal clasificados ({len(mal)}):')
        print(mal[['record', 'grupo', 'ahi_estimado', 'ahi_real_pct']].to_string(index=False))
    print()
    return por_sujeto, umbral


# =============================================================================
# Main + figuras
# =============================================================================

def main():
    for f in [FEAT_APNEA, FEAT_UCD, OOF_CSV]:
        if not os.path.exists(f):
            print(f'ERROR: falta {f}. Corre 04 y 05 primero.')
            return

    df_u, yu, res_ucd = parte1_test_ucd()
    por_sujeto, umbral = parte2_per_sujeto()

    # -------------------------------------------------------------------------
    # FIGURAS
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # ROC en UCD: con vs sin EDR, marcando el punto de operacion recalibrado
    ax = axes[0]
    for nombre, r in res_ucd.items():
        fpr, tpr, thr = roc_curve(yu, r['proba'])
        linea, = ax.plot(fpr, tpr, label=f"{nombre} (AUC={r['met']['auc']:.3f})")
        # marcar el punto de Youden
        u_y = r.get('u_youden')
        if u_y is not None:
            idx = np.argmin(np.abs(thr - u_y))
            ax.plot(fpr[idx], tpr[idx], 'o', color=linea.get_color(),
                    markersize=9, markeredgecolor='k')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4)
    ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
    ax.set_title('Test externo UCD (o = punto Youden recalibrado)')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # matriz de confusion UCD (modelo oficial)
    ax = axes[1]
    cm = confusion_matrix(yu, res_ucd['reducido CON EDR']['met']['pred'])
    im = ax.imshow(cm, cmap='Blues')
    for (r, c), v in np.ndenumerate(cm):
        ax.text(c, r, str(v), ha='center', va='center',
                color='white' if v > cm.max()/2 else 'black', fontsize=13)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['N', 'A'])
    ax.set_yticks([0, 1]); ax.set_yticklabels(['N', 'A'])
    ax.set_xlabel('Predicho'); ax.set_ylabel('Real')
    ax.set_title('Matriz confusion UCD (per-minuto)')

    # per-sujeto: AHI estimado vs real, coloreado por clase
    ax = axes[2]
    for grupo, color, marker in [('apnea', 'C3', 'o'), ('control', 'C0', 's')]:
        sub = por_sujeto[por_sujeto['grupo'] == grupo]
        ax.scatter(sub['ahi_real_pct'], sub['ahi_estimado'],
                   c=color, marker=marker, s=60, label=grupo, alpha=0.8)
    ax.axhline(umbral, color='k', linestyle='--', alpha=0.6,
               label=f'umbral ({umbral:.0f}%)')
    ax.set_xlabel('% minutos apnea REAL')
    ax.set_ylabel('% minutos apnea PREDICHO (AHI est.)')
    ax.set_title('Per-sujeto: AHI estimado vs real')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    plt.tight_layout(); plt.show()

    # guardar resumen per-sujeto
    por_sujeto.to_csv(os.path.join(CACHE, 'resultado_per_sujeto.csv'), index=False)

    print('=' * 70)
    print('Resumen final para el informe:')
    print(f'  PER-MINUTO interno (Apnea-ECG): AUC ~0.93 (del 05)')
    print(f'  PER-MINUTO externo (UCD) con EDR: AUC {res_ucd["reducido CON EDR"]["met"]["auc"]:.3f}')
    print(f'  PER-MINUTO externo (UCD) sin EDR: AUC {res_ucd["SIN EDR"]["met"]["auc"]:.3f}')
    print(f'  PER-SUJETO (Apnea-ECG A vs C): balanced acc en la calibracion')
    print('=' * 70)


if __name__ == '__main__':
    main()