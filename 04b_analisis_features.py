# -*- coding: utf-8 -*-
"""
Analisis de discriminabilidad de features 
===================================================================

Lee cache/features_apnea.csv (learning set) y analiza cuanto discrimina cada
feature entre Apnea (A) y Normal (N). Vemos:
    -> Cohen's d (diferencia de medias / desvio combinado) y AUC por feature.
    -> boxplots / violines por clase + histogramas superpuestos.
    -> se cuantifica con AUC (cercano a 0.5 = mucho solapamiento) y se muestra.

Ademas:
  - Heatmap de correlacion entre features (redundancia -> justifica PCA).
  - PCA 2D de las clases (separabilidad visual + varianza explicada).
  - Ranking de features por poder discriminante -> insumo para el 05.

Todo el analisis es sobre APNEA-ECG (el learning set, donde entrenamos). Las
wavelet se evaluan junto con las demas: sobreviven solo si discriminan.

Salidas: figuras en pantalla + cache/ranking_features.csv con d de Cohen y AUC.
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    _HAY_SNS = True
except ImportError:
    _HAY_SNS = False

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score


# =============================================================================
# Configuracion
# =============================================================================

CACHE_APNEA = 'cache'
FEATURES_CSV = os.path.join(CACHE_APNEA, 'features_apnea.csv')

# columnas que NO son features
NO_FEATURES = ['record', 'grupo', 'base', 'minute', 'label', 'n_beats']

# incluir clase B en el analisis? La decision final es del 05; aca por defecto
# analizamos solo A vs C (clases limpias) para que el ranking sea claro.
INCLUIR_B = False


# =============================================================================
# Metricas de discriminabilidad
# =============================================================================

def cohens_d(x_a, x_n):
    """d de Cohen: diferencia de medias estandarizada por el desvio combinado.

    Incorpora la dispersion (no solo la media). |d|>0.8 = efecto grande,
    0.5 medio, 0.2 chico.
    """
    x_a = x_a[~np.isnan(x_a)]
    x_n = x_n[~np.isnan(x_n)]
    if len(x_a) < 2 or len(x_n) < 2:
        return np.nan
    na, nn = len(x_a), len(x_n)
    va, vn = np.var(x_a, ddof=1), np.var(x_n, ddof=1)
    s_pool = np.sqrt(((na - 1) * va + (nn - 1) * vn) / (na + nn - 2))
    if s_pool == 0:
        return np.nan
    return (np.mean(x_a) - np.mean(x_n)) / s_pool


def auc_feature(x, y):
    """AUC de una feature individual como clasificador. 0.5 = no discrimina."""
    m = ~np.isnan(x)
    if m.sum() < 10 or len(np.unique(y[m])) < 2:
        return np.nan
    try:
        auc = roc_auc_score(y[m], x[m])
        return max(auc, 1 - auc)   # simetrico: no importa la direccion
    except Exception:
        return np.nan


# =============================================================================
# Carga
# =============================================================================

print('=' * 70)
print('Analisis de discriminabilidad de features (Apnea-ECG)')
print('=' * 70)

if not os.path.exists(FEATURES_CSV):
    print(f'ERROR: no existe {FEATURES_CSV}. Corre 04_features_por_minuto.py.')
    sys.exit(1)

df = pd.read_csv(FEATURES_CSV)
print(f'Filas totales: {len(df)}')

# quedarse con minutos etiquetados A/N
df = df[df['label'].isin(['A', 'N'])].copy()
if not INCLUIR_B:
    df = df[df['grupo'] != 'borderline'].copy()
    print('Clase B excluida del analisis (solo A vs C).')

print(f'Minutos analizados: {len(df)}  '
      f'(A={int((df["label"]=="A").sum())}, N={int((df["label"]=="N").sum())})')

feature_cols = [c for c in df.columns if c not in NO_FEATURES]
print(f'Features a evaluar: {len(feature_cols)}')
print()

y = (df['label'] == 'A').astype(int).values


# =============================================================================
# Ranking por Cohen's d y AUC
# =============================================================================

filas = []
for col in feature_cols:
    x = df[col].values.astype(float)
    xa = df.loc[df['label'] == 'A', col].values.astype(float)
    xn = df.loc[df['label'] == 'N', col].values.astype(float)
    filas.append({
        'feature': col,
        'media_A': np.nanmean(xa),
        'media_N': np.nanmean(xn),
        'std_A': np.nanstd(xa),
        'std_N': np.nanstd(xn),
        'cohens_d': cohens_d(xa, xn),
        'auc': auc_feature(x, y),
    })

ranking = pd.DataFrame(filas)
ranking['abs_d'] = ranking['cohens_d'].abs()
ranking = ranking.sort_values('auc', ascending=False).reset_index(drop=True)

out_rank = os.path.join(CACHE_APNEA, 'ranking_features.csv')
ranking.to_csv(out_rank, index=False)

print('Ranking de features por poder discriminante (AUC):')
print('-' * 70)
print(f"{'feature':24s} {'AUC':>6s} {'|d|':>6s} {'media_A':>10s} {'media_N':>10s}")
print('-' * 70)
for _, r in ranking.iterrows():
    marca = ''
    if r['auc'] >= 0.70:
        marca = ' ***'
    elif r['auc'] >= 0.60:
        marca = ' *'
    d = r['cohens_d']
    print(f"{r['feature']:24s} {r['auc']:6.3f} {abs(d):6.2f} "
          f"{r['media_A']:10.4f} {r['media_N']:10.4f}{marca}")
print('-' * 70)
print('*** AUC>=0.70 (buena)  * AUC>=0.60 (moderada)  sin marca: debil')
print(f'ranking guardado en {out_rank}')
print()

# separar features clasicas vs wavelet para el comentario
wav_feats = ranking[ranking['feature'].str.startswith('wav_')]
if len(wav_feats):
    mejor_wav = wav_feats.iloc[0]
    print(f'Mejor feature wavelet: {mejor_wav["feature"]} (AUC={mejor_wav["auc"]:.3f}). '
          f'{"Compite con las clasicas." if mejor_wav["auc"] >= 0.60 else "Debil vs las clasicas."}')
print()


# =============================================================================
# FIGURA 1: boxplots/violines de las top features por clase
# =============================================================================
# Responde: analisis visual de distribuciones + mostrar el desvio, no solo media

top_n = min(8, len(ranking))
top_feats = ranking['feature'].head(top_n).tolist()

fig, axes = plt.subplots(2, 4, figsize=(18, 9))
axes = axes.ravel()
for i, feat in enumerate(top_feats):
    ax = axes[i]
    data_a = df.loc[df['label'] == 'A', feat].dropna()
    data_n = df.loc[df['label'] == 'N', feat].dropna()
    if _HAY_SNS:
        parts = ax.violinplot([data_n.values, data_a.values],
                              showmeans=False, showmedians=True)
        for pc, col in zip(parts['bodies'], ['C0', 'C3']):
            pc.set_facecolor(col)
            pc.set_alpha(0.6)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Normal', 'Apnea'])
    else:
        ax.boxplot([data_n, data_a], showfliers=False)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Normal', 'Apnea'])
    auc = ranking.loc[ranking['feature'] == feat, 'auc'].values[0]
    d = ranking.loc[ranking['feature'] == feat, 'cohens_d'].values[0]
    ax.set_title(f'{feat}\nAUC={auc:.3f}  d={d:.2f}', fontsize=10)
    ax.grid(True, alpha=0.3)
for i in range(top_n, len(axes)):
    axes[i].axis('off')
fig.suptitle('Distribucion por clase de las top-8 features '
             '(el solapamiento se ve en el ancho de cada campana)')
plt.tight_layout(); plt.show()


# =============================================================================
# FIGURA 2: histogramas superpuestos (mostrar el solapamiento explicitamente)
# =============================================================================
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
axes = axes.ravel()
for i, feat in enumerate(top_feats[:6]):
    ax = axes[i]
    data_a = df.loc[df['label'] == 'A', feat].dropna()
    data_n = df.loc[df['label'] == 'N', feat].dropna()
    # rango comun
    lo = np.nanpercentile(df[feat], 1)
    hi = np.nanpercentile(df[feat], 99)
    bins = np.linspace(lo, hi, 50)
    ax.hist(data_n, bins=bins, alpha=0.5, color='C0', label='Normal', density=True)
    ax.hist(data_a, bins=bins, alpha=0.5, color='C3', label='Apnea', density=True)
    auc = ranking.loc[ranking['feature'] == feat, 'auc'].values[0]
    ax.set_title(f'{feat} (AUC={auc:.3f})', fontsize=10)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
fig.suptitle('Histogramas superpuestos A vs N: la superposicion de campanas '
             'es la razon por la que un umbral unico no separa bien')
plt.tight_layout(); plt.show()


# =============================================================================
# FIGURA 3: heatmap de correlacion entre features (redundancia)
# =============================================================================
corr = df[feature_cols].dropna().corr()
etiquetas = list(corr.columns)          # orden real de la matriz
M = corr.values
 
fig, ax = plt.subplots(figsize=(13, 11))
im = ax.imshow(M, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
ax.set_xticks(range(len(etiquetas)))
ax.set_xticklabels(etiquetas, rotation=90, fontsize=7)
ax.set_yticks(range(len(etiquetas)))
ax.set_yticklabels(etiquetas, fontsize=7)
# anotar el valor en cada celda (chico) para poder verificar
for i in range(len(etiquetas)):
    for j in range(len(etiquetas)):
        v = M[i, j]
        if not np.isnan(v):
            ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                    fontsize=5,
                    color='white' if abs(v) > 0.6 else '#333333')
plt.colorbar(im, ax=ax, shrink=0.7, label='Correlacion (Pearson)')
ax.set_title('Correlacion entre features '
             '(|r| alto -> redundancia; cercano a 0 -> independiente)')
plt.tight_layout(); plt.show()
 
# reportar pares muy correlacionados
print('Pares de features muy correlacionadas (|r| > 0.85):')
pares = []
for i in range(len(feature_cols)):
    for j in range(i + 1, len(feature_cols)):
        r = corr.iloc[i, j]
        if abs(r) > 0.85:
            pares.append((feature_cols[i], feature_cols[j], r))
if pares:
    for a, b, r in sorted(pares, key=lambda x: -abs(x[2])):
        print(f'  {a:22s} <-> {b:22s} r={r:+.3f}')
else:
    print('  ninguno (poca redundancia)')
print()


# =============================================================================
# FIGURA 4: PCA 2D de las clases
# =============================================================================
# Separabilidad visual + cuanta varianza explican las componentes.

X = df[feature_cols].copy()
# imputar NaN con la mediana para el PCA (solo para visualizar)
X = X.fillna(X.median())
Xs = StandardScaler().fit_transform(X)

pca = PCA(n_components=min(10, len(feature_cols)))
Z = pca.fit_transform(Xs)
var = pca.explained_variance_ratio_

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
# scatter 2D
idx = np.random.default_rng(0).permutation(len(Z))  # mezclar para no tapar
ax1.scatter(Z[idx, 0], Z[idx, 1], c=y[idx], cmap='coolwarm', s=4, alpha=0.3)
ax1.set_xlabel(f'PC1 ({100*var[0]:.1f}% var)')
ax1.set_ylabel(f'PC2 ({100*var[1]:.1f}% var)')
ax1.set_title('PCA 2D: Apnea (rojo) vs Normal (azul)')
ax1.grid(True, alpha=0.3)
# varianza explicada acumulada
ax2.plot(range(1, len(var) + 1), np.cumsum(var) * 100, 'o-')
ax2.axhline(95, color='red', linestyle='--', alpha=0.6, label='95%')
ax2.set_xlabel('N componentes')
ax2.set_ylabel('Varianza acumulada [%]')
ax2.set_title('Varianza explicada acumulada')
ax2.legend(); ax2.grid(True, alpha=0.3)
fig.suptitle('PCA: separabilidad de clases y redundancia del feature set')
plt.tight_layout(); plt.show()

n95 = int(np.searchsorted(np.cumsum(var) * 100, 95) + 1)
print(f'PCA: se necesitan {n95} de {len(feature_cols)} componentes para el 95% '
      f'de la varianza.')
print(f'  (si n95 << n_features, hay redundancia -> PCA/seleccion tiene sentido)')
print()


# =============================================================================
# Resumen para el informe
# =============================================================================
print('=' * 70)
print('Resumen')
print('=' * 70)
buenas = ranking[ranking['auc'] >= 0.60]['feature'].tolist()
debiles = ranking[ranking['auc'] < 0.55]['feature'].tolist()
print(f'Features con buen poder discriminante (AUC>=0.60): {buenas}')
print(f'Features debiles (AUC<0.55): {debiles}')
wav_buenas = [f for f in buenas if f.startswith('wav_')]
print(f'Wavelet que sobreviven (AUC>=0.60): {wav_buenas if wav_buenas else "ninguna"}')
print()
print('Estas metricas alimentan la decision de que features usar en el 05.')
print('=' * 70)
