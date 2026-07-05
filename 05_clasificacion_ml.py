# -*- coding: utf-8 -*-
"""
Clasificacion ML per-minuto (Apnea-ECG): entrenamiento y validacion
====================================================================

Entrena y valida clasificadores para detectar apnea minuto a minuto, sobre el
learning set (Apnea-ECG). El test externo (UCD) y la evaluacion per-sujeto van
en el 06.

Puntos metodologicos:
  - StratifiedGroupKFold con grupo = sujeto. Los minutos de un sujeto estan
    correlacionados; si el mismo sujeto cae en train y validacion, las metricas
    se inflan (data leakage). Agrupar por sujeto lo evita.
  - La clase B (borderline) SE USA en el entrenamiento per-minuto: cada minuto
    tiene su etiqueta A/N valida, sea o no borderline el sujeto. La ambiguedad
    de B es a nivel de SUJETO (sano/no sano), y eso solo afecta al 06.
  - Desbalance: class_weight='balanced' y se reportan F1-macro, AUC, matriz de
    confusion, precision/recall por clase (no solo accuracy).

Experimentos que corre:
  1. Varios clasificadores (LogReg, RandomForest, GradientBoosting, SVM-RBF) +
     un ensemble (VotingClassifier), con el set de features COMPLETO.
  2. Set COMPLETO vs REDUCIDO (seleccion automatica por importancia) para el
     mejor modelo.
  3. CON EDR vs SIN EDR, para medir cuanto depende el modelo del EDR (que
     transfiere peor a UCD por ser otra derivacion).

Salidas:
  - cache/oof_predicciones.csv : predicciones out-of-fold por minuto (para el 06)
  - cache/modelo_final.joblib   : el mejor modelo reentrenado con todo el train
  - cache/importancias.csv      : importancia de cada feature
  - metricas por pantalla + figuras (ROC, matriz de confusion, importancias)
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              VotingClassifier)
from sklearn.svm import SVC
from sklearn.metrics import (roc_auc_score, f1_score, confusion_matrix,
                             classification_report, roc_curve,
                             precision_recall_fscore_support)
from sklearn.inspection import permutation_importance

import joblib


# =============================================================================
# Configuracion
# =============================================================================

CACHE = 'cache'
FEATURES_CSV = os.path.join(CACHE, 'features_apnea.csv')

N_SPLITS = 5
RANDOM_STATE = 42

# Features a EXCLUIR siempre (AUC < 0.55 en el 04b: no discriminan).
FEATURES_EXCLUIR = ['rmssd', 'nn50', 'pnn50', 'hf_power', 'mean_rr', 'mean_hr',
                    'n_beats']

# Features EDR (para el experimento con/sin EDR)
FEATURES_EDR = ['edr_resp_power', 'edr_apnea_power', 'edr_resp_norm',
                'edr_apnea_norm', 'edr_apnea_resp_ratio']

NO_FEATURES = ['record', 'grupo', 'base', 'minute', 'label']


# =============================================================================
# Definicion de clasificadores
# =============================================================================

def hacer_pipeline(clf):
    """Envuelve un clasificador con imputacion + escalado (sin leakage).

    El imputer y el scaler se ajustan solo con el train de cada fold porque
    van DENTRO del Pipeline que cross_val_predict reentrena en cada split.
    """
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('clf', clf),
    ])


def clasificadores():
    """Devuelve un dict nombre -> pipeline. class_weight balanced donde aplica."""
    return {
        'LogReg': hacer_pipeline(LogisticRegression(
            max_iter=1000, class_weight='balanced', random_state=RANDOM_STATE)),
        'RandomForest': hacer_pipeline(RandomForestClassifier(
            n_estimators=200, class_weight='balanced', n_jobs=-1,
            random_state=RANDOM_STATE)),
        'GradBoost': hacer_pipeline(GradientBoostingClassifier(
            n_estimators=200, random_state=RANDOM_STATE)),
        'SVM-RBF': hacer_pipeline(SVC(
            kernel='rbf', class_weight='balanced', probability=True,
            random_state=RANDOM_STATE)),
    }


def ensemble(estimadores_base):
    """VotingClassifier (soft) con los pipelines base."""
    return VotingClassifier(
        estimators=[(n, p) for n, p in estimadores_base.items()],
        voting='soft', n_jobs=-1)


# =============================================================================
# Evaluacion por CV agrupada
# =============================================================================

def evaluar_cv(pipe, X, y, groups, n_splits=N_SPLITS):
    """CV agrupada por sujeto. Devuelve (proba_oof, pred_oof, metricas)."""
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                              random_state=RANDOM_STATE)
    # probabilidades out-of-fold
    proba = cross_val_predict(pipe, X, y, groups=groups, cv=cv,
                              method='predict_proba', n_jobs=-1)[:, 1]
    pred = (proba >= 0.5).astype(int)
    met = {
        'auc': roc_auc_score(y, proba),
        'f1_macro': f1_score(y, pred, average='macro'),
        'f1_apnea': f1_score(y, pred, pos_label=1),
    }
    p, r, f, _ = precision_recall_fscore_support(y, pred, average=None,
                                                 labels=[0, 1])
    met['prec_N'], met['prec_A'] = p[0], p[1]
    met['rec_N'], met['rec_A'] = r[0], r[1]
    return proba, pred, met


# =============================================================================
# Main
# =============================================================================

def main():
    if not os.path.exists(FEATURES_CSV):
        print(f'ERROR: falta {FEATURES_CSV}. Corre 04_features_por_minuto.py.')
        return

    df = pd.read_csv(FEATURES_CSV)
    # minutos etiquetados A/N; B SE INCLUYE (per-minuto)
    df = df[df['label'].isin(['A', 'N'])].copy().reset_index(drop=True)
    y = (df['label'] == 'A').astype(int).values
    groups = df['record'].values

    all_feats = [c for c in df.columns if c not in NO_FEATURES]
    feats_full = [f for f in all_feats if f not in FEATURES_EXCLUIR]
    feats_no_edr = [f for f in feats_full if f not in FEATURES_EDR]

    print('=' * 70)
    print('Clasificacion ML per-minuto (Apnea-ECG)')
    print('=' * 70)
    print(f'Minutos: {len(df)}  (A={int(y.sum())}, N={int((1-y).sum())})')
    print(f'Sujetos: {df["record"].nunique()} '
          f'(incluye B para entrenar per-minuto)')
    print(f'Features (full): {len(feats_full)}  |  sin EDR: {len(feats_no_edr)}')
    print(f'Excluidas (AUC<0.55): {FEATURES_EXCLUIR}')
    print()

    X_full = df[feats_full].values

    # -------------------------------------------------------------------------
    # EXPERIMENTO 1: comparar clasificadores (set completo)
    # -------------------------------------------------------------------------
    print('-' * 70)
    print('EXP 1: comparacion de clasificadores (features completas)')
    print('-' * 70)
    print(f"{'modelo':14s} {'AUC':>6s} {'F1-mac':>7s} {'F1-apn':>7s} "
          f"{'recA':>6s} {'precA':>6s}")
    resultados = {}
    clfs = clasificadores()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for nombre, pipe in clfs.items():
            proba, pred, met = evaluar_cv(pipe, X_full, y, groups)
            resultados[nombre] = {'proba': proba, 'pred': pred, 'met': met}
            print(f"{nombre:14s} {met['auc']:6.3f} {met['f1_macro']:7.3f} "
                  f"{met['f1_apnea']:7.3f} {met['rec_A']:6.3f} {met['prec_A']:6.3f}")

        # ensemble
        ens = ensemble(clasificadores())
        proba_e, pred_e, met_e = evaluar_cv(ens, X_full, y, groups)
        resultados['Ensemble'] = {'proba': proba_e, 'pred': pred_e, 'met': met_e}
        print(f"{'Ensemble':14s} {met_e['auc']:6.3f} {met_e['f1_macro']:7.3f} "
              f"{met_e['f1_apnea']:7.3f} {met_e['rec_A']:6.3f} {met_e['prec_A']:6.3f}")

    # mejor modelo por AUC
    mejor = max(resultados, key=lambda k: resultados[k]['met']['auc'])
    print(f'\nMejor modelo (AUC): {mejor}')
    print()

    # -------------------------------------------------------------------------
    # EXPERIMENTO 2: CON EDR vs SIN EDR (con el mejor modelo base)
    # -------------------------------------------------------------------------
    print('-' * 70)
    print('EXP 2: impacto del EDR (mismo modelo, con vs sin features EDR)')
    print('-' * 70)
    # usar RandomForest como modelo estable para esta comparacion
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        pipe_con = hacer_pipeline(RandomForestClassifier(
            n_estimators=200, class_weight='balanced', n_jobs=-1,
            random_state=RANDOM_STATE))
        _, _, met_con = evaluar_cv(pipe_con, df[feats_full].values, y, groups)

        pipe_sin = hacer_pipeline(RandomForestClassifier(
            n_estimators=200, class_weight='balanced', n_jobs=-1,
            random_state=RANDOM_STATE))
        _, _, met_sin = evaluar_cv(pipe_sin, df[feats_no_edr].values, y, groups)
    print(f"{'config':14s} {'AUC':>6s} {'F1-mac':>7s} {'recA':>6s}")
    print(f"{'con EDR':14s} {met_con['auc']:6.3f} {met_con['f1_macro']:7.3f} "
          f"{met_con['rec_A']:6.3f}")
    print(f"{'sin EDR':14s} {met_sin['auc']:6.3f} {met_sin['f1_macro']:7.3f} "
          f"{met_sin['rec_A']:6.3f}")
    delta = met_con['auc'] - met_sin['auc']
    print(f'\nEl EDR aporta {delta:+.3f} de AUC. '
          f'{"Depende bastante del EDR -> ojo con la caida en UCD." if delta > 0.02 else "Poca dependencia del EDR -> deberia transferir bien a UCD."}')
    print()

    # -------------------------------------------------------------------------
    # EXPERIMENTO 3: importancias + set reducido
    # -------------------------------------------------------------------------
    print('-' * 70)
    print('EXP 3: importancia de features (RandomForest sobre todo el train)')
    print('-' * 70)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        rf_full = hacer_pipeline(RandomForestClassifier(
            n_estimators=300, class_weight='balanced', n_jobs=-1,
            random_state=RANDOM_STATE))
        rf_full.fit(X_full, y)
    importancias = rf_full.named_steps['clf'].feature_importances_
    imp_df = pd.DataFrame({'feature': feats_full, 'importancia': importancias})
    imp_df = imp_df.sort_values('importancia', ascending=False).reset_index(drop=True)
    imp_df.to_csv(os.path.join(CACHE, 'importancias.csv'), index=False)
    print(imp_df.to_string(index=False))

    # set reducido: top features que acumulan 90% de la importancia
    imp_df['acum'] = imp_df['importancia'].cumsum()
    feats_reducido = imp_df[imp_df['acum'] <= 0.90]['feature'].tolist()
    if len(feats_reducido) < 3:
        feats_reducido = imp_df['feature'].head(8).tolist()
    print(f'\nSet reducido (90% de importancia): {len(feats_reducido)} features')
    print(f'  {feats_reducido}')

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        pipe_red = hacer_pipeline(RandomForestClassifier(
            n_estimators=200, class_weight='balanced', n_jobs=-1,
            random_state=RANDOM_STATE))
        _, _, met_red = evaluar_cv(pipe_red, df[feats_reducido].values, y, groups)
    print(f"\n{'config':14s} {'AUC':>6s} {'F1-mac':>7s}")
    print(f"{'completo':14s} {met_con['auc']:6.3f} {met_con['f1_macro']:7.3f}")
    print(f"{'reducido':14s} {met_red['auc']:6.3f} {met_red['f1_macro']:7.3f}")
    print(f'  -> {"el reducido rinde igual o mejor: mas simple e interpretable" if met_red["auc"] >= met_con["auc"] - 0.01 else "el completo rinde mejor: conviene mantener todas"}')
    print()

    # -------------------------------------------------------------------------
    # Guardar OOF del mejor modelo + modelo final entrenado con todo
    # -------------------------------------------------------------------------
    proba_mejor = resultados[mejor]['proba']
    oof = df[['record', 'grupo', 'minute', 'label']].copy()
    oof['y_true'] = y
    oof['proba_apnea'] = proba_mejor
    oof['pred'] = (proba_mejor >= 0.5).astype(int)
    oof.to_csv(os.path.join(CACHE, 'oof_predicciones.csv'), index=False)

    # modelo final: reentrenar el mejor con TODO el train (para el 06/UCD)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if mejor == 'Ensemble':
            modelo_final = ensemble(clasificadores())
        else:
            modelo_final = clasificadores()[mejor]
        modelo_final.fit(X_full, y)
    joblib.dump({'modelo': modelo_final, 'features': feats_full,
                 'features_reducido': feats_reducido},
                os.path.join(CACHE, 'modelo_final.joblib'))
    print(f'Guardado: oof_predicciones.csv, importancias.csv, modelo_final.joblib')
    print()

    # -------------------------------------------------------------------------
    # FIGURAS
    # -------------------------------------------------------------------------
    # ROC de todos los modelos
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.5))
    for nombre, res in resultados.items():
        fpr, tpr, _ = roc_curve(y, res['proba'])
        ax1.plot(fpr, tpr, label=f"{nombre} (AUC={res['met']['auc']:.3f})")
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.4)
    ax1.set_xlabel('FPR'); ax1.set_ylabel('TPR')
    ax1.set_title('Curvas ROC (CV agrupada por sujeto)')
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

    # matriz de confusion del mejor
    cm = confusion_matrix(y, resultados[mejor]['pred'])
    im = ax2.imshow(cm, cmap='Blues')
    for (r, c), v in np.ndenumerate(cm):
        ax2.text(c, r, str(v), ha='center', va='center',
                 color='white' if v > cm.max()/2 else 'black', fontsize=14)
    ax2.set_xticks([0, 1]); ax2.set_xticklabels(['N', 'A'])
    ax2.set_yticks([0, 1]); ax2.set_yticklabels(['N', 'A'])
    ax2.set_xlabel('Predicho'); ax2.set_ylabel('Real')
    ax2.set_title(f'Matriz de confusion - {mejor}')

    # importancias top 12
    top = imp_df.head(12).iloc[::-1]
    ax3.barh(top['feature'], top['importancia'], color='C0')
    ax3.set_xlabel('Importancia')
    ax3.set_title('Feature importances (RandomForest)')
    ax3.grid(True, alpha=0.3, axis='x')
    plt.tight_layout(); plt.show()

    print('=' * 70)
    print('Resumen para el informe:')
    print(f'  - Mejor modelo per-minuto: {mejor} (AUC={resultados[mejor]["met"]["auc"]:.3f})')
    print(f'  - EDR aporta {delta:+.3f} AUC (relevante para anticipar UCD)')
    print(f'  - Set reducido ({len(feats_reducido)} feats) AUC={met_red["auc"]:.3f} '
          f'vs completo {met_con["auc"]:.3f}')
    print('  - Las predicciones OOF alimentan el per-sujeto (06)')
    print('=' * 70)


if __name__ == '__main__':
    main()
