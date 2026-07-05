# -*- coding: utf-8 -*-
"""
Interfaz de visualizacion de apnea del sueno — dos solapas
===========================================================

Interfaz interactiva (PySide6 + pyqtgraph) que integra dos vistas sobre el
mismo sistema de deteccion de apnea, con un selector de registro compartido:

  SOLAPA TECNICA (perfil ingenieria / analisis de datos)
    - ECG con las ondas R detectadas.
    - Tacograma RR con los intervalos descartados (outliers) marcados,
      sincronizado con el ECG.
    - Panel de features por minuto (z-normalizadas) con el fondo coloreado
      segun la prediccion del modelo y marcas en los minutos de apnea real.
    - Tabla de todos los minutos con features y prediccion del modelo.
    - Funciona para las dos bases: Apnea-ECG y UCDDB.

  SOLAPA MEDICA (perfil clinico / tamizaje) — solo Apnea-ECG
    - Monitor de la noche: ECG animado, timeline de doble tira (prediccion
      del modelo vs etiqueta real), datos del paciente, resumen a nivel
      sujeto (AHI estimado) y explicabilidad de features del minuto.
    - Para UCDDB no se muestra: no se dispone de datos clinicos del paciente
      (edad, sexo, AHI de referencia), por lo que la vista clinica no aplica.

Pipeline nuevo (ML). Lee del cache:
  - cache/<record>.npz              (03b: picos_R, rr_crudo, rr_interp, flags, fs)
  - cache/features_apnea.csv        (04)
  - cache/oof_predicciones.csv      (05: prediccion del modelo por minuto, OOF)
  - cache/ranking_features.csv      (04b: media_A / media_N por feature)
  - cache/importancias.csv          (05: orden de features por importancia)
  - cache_ucd/<record>.npz          (00: ECG + labels de UCDDB)
  - cache_ucd_proc/<record>.npz     (03b: picos/rr/flags de UCDDB)
  - cache_ucd_proc/features_ucd.csv (04)
  - cache_ucd_proc/predicciones_ucd.csv (06: prediccion del modelo en UCDDB) [opcional]

Correr desde la carpeta del proyecto: python interfaz_apnea.py
"""

import os
import sys

import numpy as np
import pandas as pd

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QSplitter, QComboBox, QTabWidget,
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTableView, QFormLayout,
    QHeaderView, QGroupBox, QGridLayout, QStatusBar, QMessageBox, QTableWidget,
    QTableWidgetItem, QAbstractItemView,
)
import pyqtgraph as pg

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from src.pipeline import cargar_ecg, filtrar_ecg_general, clasificar_grupo


# =============================================================================
# Constantes y rutas de cache
# =============================================================================

DATA_DIR = 'apnea-ecg-database-1.0.0'      # registros wfdb de Apnea-ECG
CACHE_DIR = 'cache'                          # cache Apnea-ECG (03b, 04, 05)
CACHE_UCD_RAW = 'cache_ucd'                  # cache UCD con ECG (00)
CACHE_UCD_PROC = 'cache_ucd_proc'            # cache UCD procesado (03b, 04)
ADDITIONAL_INFO_FILE = 'additional-information.txt'

FS = 100

CLASS_COLORS = {
    'A': (200, 60, 60),     # rojo  - apnea
    'B': (220, 150, 50),    # naranja - borderline
    'C': (60, 160, 80),     # verde - control
}
APNEA_RGB = (200, 60, 60)
NORMAL_RGB = (210, 235, 210)

pg.setConfigOption('background', '#fafafa')
pg.setConfigOption('foreground', '#222222')
pg.setConfigOption('antialias', True)


FEATURE_LABELS = {
    'mean_rr': 'RR medio (s)', 'sdnn': 'SDNN (s)', 'rmssd': 'RMSSD (s)',
    'nn50': 'NN50', 'pnn50': 'pNN50 (%)',
    'mean_hr': 'FC media (lpm)', 'sd_hr': 'SD de la FC (lpm)',
    'vlf_power': 'Potencia VLF', 'lf_power': 'Potencia LF', 'hf_power': 'Potencia HF',
    'total_power': 'Potencia total', 'lf_hf_ratio': 'Ratio LF/HF',
    'lf_norm': 'LF normalizada', 'hf_norm': 'HF normalizada',
    'cvhr_power': 'Potencia CVHR', 'cvhr_norm': 'CVHR normalizada',
    'wav_energy_L1': 'Energia wavelet L1', 'wav_energy_L2': 'Energia wavelet L2',
    'wav_energy_L3': 'Energia wavelet L3', 'wav_energy_L4': 'Energia wavelet L4',
    'wav_energy_L5': 'Energia wavelet L5', 'wav_entropy': 'Entropia wavelet',
    'edr_resp_power': 'EDR potencia respiratoria', 'edr_apnea_power': 'EDR potencia apneica',
    'edr_resp_norm': 'EDR resp. normalizada', 'edr_apnea_norm': 'EDR apneica normalizada',
    'edr_apnea_resp_ratio': 'EDR ratio apnea/resp',
}


# =============================================================================
# Helpers de carga
# =============================================================================

def _cargar_info_paciente(record, data_dir, path=ADDITIONAL_INFO_FILE):
    """Datos del paciente desde additional-information.txt (solo Apnea-ECG)."""
    ruta = os.path.join(data_dir, path)
    if not os.path.exists(ruta):
        return None
    with open(ruta, 'r', encoding='utf-8', errors='ignore') as f:
        for linea in f:
            partes = linea.split()
            if len(partes) < 12 or partes[0] != record:
                continue
            try:
                return {
                    'duracion_min': int(partes[1]),
                    'no_apnea_min': int(partes[2]),
                    'apnea_min': int(partes[3]),
                    'horas_con_apnea': int(partes[4]),
                    'AI': float(partes[5]), 'HI': float(partes[6]), 'AHI': float(partes[7]),
                    'edad': int(partes[8]), 'sexo': partes[9],
                    'altura_cm': int(partes[10]), 'peso_kg': int(partes[11]),
                }
            except ValueError:
                return None
    return None


def es_ucd(record):
    """True si el record es de la base UCDDB (por prefijo)."""
    return str(record).startswith('ucddb')


def cargar_ecg_ucd(record, cache_raw=CACHE_UCD_RAW):
    """Carga el ECG de UCD desde el cache RAW del 00 (ya remuestreado a 100 Hz)."""
    path = os.path.join(cache_raw, f'{record}.npz')
    data = np.load(path, allow_pickle=True)
    if 'ecg' not in data.files:
        return None
    return data['ecg'].astype(float)


# =============================================================================
# Modelo Qt para la tabla de minutos (solapa tecnica)
# =============================================================================

class TablaMinutosModel(QtCore.QAbstractTableModel):
    """Tabla con features y la prediccion del modelo por minuto."""

    COLUMNS = [
        ('minute', 'Min'),
        ('label', 'Real'),
        ('cvhr_norm', 'cvhr_norm'),
        ('lf_hf_ratio', 'LF/HF'),
        ('wav_energy_L5', 'wav_L5'),
        ('edr_apnea_resp_ratio', 'EDR ap/resp'),
        ('proba_apnea', 'p(apnea)'),
        ('pred', 'Modelo'),
    ]

    def __init__(self, df, parent=None):
        super().__init__(parent)
        self.df = df.reset_index(drop=True)

    def rowCount(self, parent=QtCore.QModelIndex()):
        return len(self.df)

    def columnCount(self, parent=QtCore.QModelIndex()):
        return len(self.COLUMNS)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return self.COLUMNS[section][1]
        if role == Qt.DisplayRole and orientation == Qt.Vertical:
            return str(section)
        return None

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        col_key = self.COLUMNS[index.column()][0]
        val = self.df.iloc[index.row()].get(col_key) if col_key in self.df.columns else None

        if role == Qt.DisplayRole:
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return '-'
            if col_key == 'pred':
                return 'APNEA' if int(val) == 1 else 'Normal'
            if isinstance(val, float):
                return f'{val:.3f}'
            if isinstance(val, (int, np.integer)):
                return str(int(val))
            return str(val)

        if role == Qt.TextAlignmentRole:
            return int(Qt.AlignCenter)

        if role == Qt.BackgroundRole:
            if col_key == 'label':
                if val == 'A':
                    return QtGui.QColor(255, 225, 225)
                if val == 'N':
                    return QtGui.QColor(225, 245, 225)
            if col_key == 'pred':
                if val == 1:
                    return QtGui.QColor(255, 225, 225)
                if val == 0:
                    return QtGui.QColor(225, 245, 225)
        return None


# =============================================================================
# Barra de comparacion (solapa medica) — valor vs media normal/apnea
# =============================================================================

class BarraComparacion(QtWidgets.QWidget):
    """Ubica el valor de una feature en una barra entre la media tipica de
    minutos normales (izquierda, verde) y de minutos con apnea (derecha, rojo)."""

    def __init__(self, valor, media_normal, media_apnea, parent=None):
        super().__init__(parent)
        self.valor = valor
        self.media_normal = media_normal
        self.media_apnea = media_apnea
        self.setMinimumHeight(34)

    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        w, h = self.width(), self.height()
        x0, x1 = 6, w - 6
        barra_y = h // 2 - 8
        barra_h = 8

        grad = QtGui.QLinearGradient(x0, 0, x1, 0)
        grad.setColorAt(0.0, QtGui.QColor(*NORMAL_RGB))
        grad.setColorAt(1.0, QtGui.QColor(*APNEA_RGB))
        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(grad)
        p.drawRoundedRect(x0, barra_y, x1 - x0, barra_h, 4, 4)

        if (self.media_normal is not None and self.media_apnea is not None
                and not pd.isna(self.valor) and self.media_apnea != self.media_normal):
            frac = (self.valor - self.media_normal) / (self.media_apnea - self.media_normal)
            frac = min(max(frac, -0.12), 1.12)
            x = x0 + frac * (x1 - x0)
            p.setBrush(QtGui.QColor('#f0f0f0'))
            p.setPen(QtGui.QPen(QtGui.QColor('#15171d'), 1.5))
            p.drawEllipse(QtCore.QPointF(x, barra_y + barra_h / 2), 6, 6)

        p.setPen(QtGui.QColor('#9098a8'))
        f = p.font()
        f.setPointSize(8)
        p.setFont(f)
        txt_n = f'{self.media_normal:.2g}' if self.media_normal is not None else '-'
        txt_a = f'{self.media_apnea:.2g}' if self.media_apnea is not None else '-'
        p.drawText(QtCore.QRectF(x0, barra_y + barra_h + 3, 60, 14), int(QtCore.Qt.AlignLeft), txt_n)
        p.drawText(QtCore.QRectF(x1 - 60, barra_y + barra_h + 3, 60, 14), int(QtCore.Qt.AlignRight), txt_a)


# =============================================================================
# SOLAPA TECNICA
# =============================================================================

class SolapaTecnica(QWidget):
    """Vista tecnica: ECG + tacograma + features + tabla, para ambas bases."""

    def __init__(self, datos, parent=None):
        super().__init__(parent)
        self.datos = datos          # referencia al contenedor de datos compartidos
        self._record = None
        self._cache = None
        self._ecg = None
        self._fs = FS
        self._feat = None
        self._marca_minuto = None
        self._build_ui()

    def _build_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(8)

        # --- Panel izquierdo: resumen + prediccion ---
        left = QWidget()
        left.setMinimumWidth(250)
        left.setMaximumWidth(300)
        ll = QVBoxLayout(left)
        ll.setSpacing(8)

        gb_resumen = QGroupBox('Resumen del registro')
        gl = QGridLayout(gb_resumen)
        gl.setVerticalSpacing(4)
        self.lbl_grupo = QLabel('-')
        self.lbl_clase_real = QLabel('-')
        self.lbl_duracion = QLabel('-')
        self.lbl_fc_media = QLabel('-')
        self.lbl_n_outliers = QLabel('-')
        for i, (k, v) in enumerate([
            ('Base', self.lbl_grupo),
            ('Clase real', self.lbl_clase_real),
            ('Duracion', self.lbl_duracion),
            ('FC media', self.lbl_fc_media),
            ('Outliers RR', self.lbl_n_outliers),
        ]):
            gl.addWidget(QLabel(f'{k}:'), i, 0)
            gl.addWidget(v, i, 1)
        ll.addWidget(gb_resumen)

        # Prediccion del modelo (nivel sujeto)
        gb_pred = QGroupBox('Prediccion del modelo (ML)')
        gp = QGridLayout(gb_pred)
        gp.setVerticalSpacing(4)
        self.lbl_min_apnea_real = QLabel('-')
        self.lbl_min_apnea_pred = QLabel('-')
        self.lbl_ahi_est = QLabel('-')
        for i, (k, v) in enumerate([
            ('Min. apnea (real)', self.lbl_min_apnea_real),
            ('Min. apnea (modelo)', self.lbl_min_apnea_pred),
            ('AHI estimado', self.lbl_ahi_est),
        ]):
            gp.addWidget(QLabel(f'{k}:'), i, 0)
            gp.addWidget(v, i, 1)
        ll.addWidget(gb_pred)

        ll.addStretch()

        legend = QLabel(
            '<small><b>Leyenda</b><br>'
            '<span style="background-color: rgb(255,225,225);'
            ' padding: 1px 4px;">Apnea</span> &nbsp;'
            '<span style="background-color: rgb(225,245,225);'
            ' padding: 1px 4px;">Normal</span></small>'
        )
        legend.setWordWrap(True)
        ll.addWidget(legend)
        main_layout.addWidget(left)

        # --- Panel derecho: plots + tabla ---
        splitter = QSplitter(Qt.Vertical)
        splitter.setChildrenCollapsible(False)

        self.plot_ecg = pg.PlotWidget(title='ECG con R detectados')
        self.plot_ecg.setLabel('left', 'ECG (mV)')
        self.plot_ecg.setLabel('bottom', 'Tiempo (s)')
        self.plot_ecg.setDownsampling(auto=True, mode='peak')
        self.plot_ecg.setClipToView(True)
        self.plot_ecg.showGrid(x=True, y=True, alpha=0.3)
        splitter.addWidget(self.plot_ecg)

        self.plot_rr = pg.PlotWidget(title='Tacograma RR (outliers marcados)')
        self.plot_rr.setLabel('left', 'RR (s)')
        self.plot_rr.setLabel('bottom', 'Tiempo (s)')
        self.plot_rr.showGrid(x=True, y=True, alpha=0.3)
        self.plot_rr.setXLink(self.plot_ecg)
        splitter.addWidget(self.plot_rr)

        self.plot_features = pg.PlotWidget(
            title='Features por minuto (fondo = prediccion del modelo)')
        self.plot_features.setLabel('left', 'Feature (z-norm robusto)')
        self.plot_features.setLabel('bottom', 'Minuto')
        self.plot_features.showGrid(x=True, y=True, alpha=0.3)
        self.plot_features.addLegend(offset=(10, 10))
        splitter.addWidget(self.plot_features)

        self.tabla = QTableView()
        self.tabla.setSelectionBehavior(QTableView.SelectRows)
        self.tabla.setSelectionMode(QTableView.SingleSelection)
        self.tabla.verticalHeader().setVisible(False)
        self.tabla.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.tabla.clicked.connect(self._on_tabla_click)
        splitter.addWidget(self.tabla)

        splitter.setSizes([220, 160, 280, 220])
        main_layout.addWidget(splitter, stretch=1)

    # ---------------------------------------------------------------------
    def cargar_registro(self, record):
        """Carga un registro de cualquiera de las dos bases en la vista tecnica."""
        self._record = record
        ucd = es_ucd(record)

        # cache npz (picos, rr, flags)
        cache_dir = CACHE_UCD_PROC if ucd else CACHE_DIR
        cache_path = os.path.join(cache_dir, f'{record}.npz')
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f'No se encontro {cache_path}.')
        self._cache = np.load(cache_path, allow_pickle=True)
        self._fs = int(self._cache['fs'])

        # ECG filtrado
        if ucd:
            ecg_raw = cargar_ecg_ucd(record)
            self._ecg = filtrar_ecg_general(ecg_raw, self._fs) if ecg_raw is not None else None
        else:
            ecg_raw, _, _ = cargar_ecg(record, DATA_DIR)
            self._ecg = filtrar_ecg_general(ecg_raw, self._fs)

        # features + predicciones de este registro (merge)
        feats = self.datos.features_ucd if ucd else self.datos.features_apnea
        preds = self.datos.pred_ucd if ucd else self.datos.pred_apnea
        self._feat = feats[feats['record'] == record].copy()
        if preds is not None and len(preds):
            cols = [c for c in ['minute', 'proba_apnea', 'pred', 'y_true'] if c in preds.columns]
            pr = preds[preds['record'] == record][cols]
            self._feat = self._feat.merge(pr, on='minute', how='left')
        self._feat = self._feat.sort_values('minute').reset_index(drop=True)

        self._actualizar_resumen(record, ucd)
        self._dibujar_ecg_y_rr()
        self._dibujar_features()
        self._llenar_tabla()

    def _actualizar_resumen(self, record, ucd):
        self.lbl_grupo.setText('UCDDB' if ucd else 'Apnea-ECG')
        if ucd:
            self.lbl_clase_real.setText('(sin clase de sujeto)')
        else:
            g = clasificar_grupo(record)
            self.lbl_clase_real.setText({'apnea': 'A', 'borderline': 'B',
                                         'control': 'C'}.get(g, '-'))
        dur = float(self._cache['duracion_s']) / 60
        self.lbl_duracion.setText(f'{dur:.0f} min')
        # FC media a partir de rr_interp
        rr = self._cache['rr_interp']
        rr = rr[(rr > 0.3) & (rr < 2.0)]
        if len(rr):
            self.lbl_fc_media.setText(f'{60/np.mean(rr):.0f} lpm')
        n_out = int(np.sum(self._cache['flag_total']))
        n_tot = len(self._cache['rr_crudo'])
        self.lbl_n_outliers.setText(f'{n_out} / {n_tot} ({100*n_out/n_tot:.1f}%)')

        # prediccion nivel sujeto
        if 'pred' in self._feat.columns:
            lab = self._feat['label']
            n_real = int((lab == 'A').sum())
            n_pred = int((self._feat['pred'] == 1).sum())
            n_min = len(self._feat)
            self.lbl_min_apnea_real.setText(f'{n_real} min')
            self.lbl_min_apnea_pred.setText(f'{n_pred} min')
            self.lbl_ahi_est.setText(f'{100*n_pred/n_min:.1f}% de la noche')

    def _dibujar_ecg_y_rr(self):
        self.plot_ecg.clear()
        self.plot_rr.clear()
        if self._ecg is None:
            return
        t = np.arange(len(self._ecg)) / self._fs
        self.plot_ecg.plot(t, self._ecg, pen=pg.mkPen('#2f5d9e', width=1))
        picos = np.asarray(self._cache['picos_R']).astype(int)
        picos = picos[picos < len(self._ecg)]
        self.plot_ecg.plot(t[picos], self._ecg[picos], pen=None, symbol='o',
                           symbolBrush=(220, 40, 40), symbolPen=pg.mkPen('w', width=1),
                           symbolSize=7)
        # tacograma
        picos_R = np.asarray(self._cache['picos_R']).astype(int)
        rr = np.asarray(self._cache['rr_crudo'])
        flag = np.asarray(self._cache['flag_total']).astype(bool)
        if len(picos_R) >= 2 and len(rr) >= 1:
            t_rr = picos_R[1:len(rr)+1] / self._fs
            n = min(len(t_rr), len(rr), len(flag))
            t_rr, rr, flag = t_rr[:n], rr[:n], flag[:n]
            self.plot_rr.plot(t_rr, rr, pen=pg.mkPen('#666', width=1))
            if flag.any():
                self.plot_rr.plot(t_rr[flag], rr[flag], pen=None, symbol='x',
                                  symbolBrush=(220, 40, 40), symbolSize=9)
            # Fijar el eje Y a un rango fisiologico. La serie cruda (rr_crudo)
            # contiene intervalos no fisiologicos (detecciones perdidas/dobles)
            # que llegan a miles de segundos; sin acotar el eje, aplastan los RR
            # normales (~0.8 s) en una linea plana. Los limitamos a [0, 2] s.
            self.plot_rr.setYRange(0, 2.0, padding=0.05)
            self.plot_rr.disableAutoRange(axis=pg.ViewBox.YAxis)

    def _dibujar_features(self):
        self.plot_features.clear()
        if self._feat is None or len(self._feat) == 0:
            return
        minutos = self._feat['minute'].values
        # fondo por prediccion del modelo
        if 'pred' in self._feat.columns:
            for _, row in self._feat.iterrows():
                if row.get('pred') == 1:
                    reg = pg.LinearRegionItem(
                        values=[row['minute'] - 0.5, row['minute'] + 0.5],
                        brush=(255, 210, 210, 80), movable=False)
                    reg.setZValue(-10)
                    self.plot_features.addItem(reg)
        # features z-norm
        colores = {'cvhr_norm': '#1f77b4', 'lf_hf_ratio': '#ff7f0e',
                   'wav_energy_L5': '#2ca02c'}
        for feat, color in colores.items():
            if feat not in self._feat.columns:
                continue
            x = self._feat[feat].values.astype(float)
            med = np.nanmedian(x)
            mad = np.nanmedian(np.abs(x - med)) + 1e-9
            z = (x - med) / (1.4826 * mad)
            self.plot_features.plot(minutos, z, pen=pg.mkPen(color, width=2),
                                    name=FEATURE_LABELS.get(feat, feat))
        # marcas de apnea real
        if 'label' in self._feat.columns:
            ap = self._feat[self._feat['label'] == 'A']['minute'].values
            for m in ap:
                ln = pg.InfiniteLine(pos=m, angle=90,
                                     pen=pg.mkPen((200, 60, 60, 120), width=1, style=Qt.DotLine))
                self.plot_features.addItem(ln)
        self._marca_minuto = pg.InfiniteLine(pos=0, angle=90,
                                             pen=pg.mkPen('#111', width=2))
        self.plot_features.addItem(self._marca_minuto)

    def _llenar_tabla(self):
        self.tabla.setModel(TablaMinutosModel(self._feat))

    def _on_tabla_click(self, index):
        row = index.row()
        if self._feat is None or row >= len(self._feat):
            return
        minuto = int(self._feat.iloc[row]['minute'])
        # zoom al minuto en ECG/tacograma
        self.plot_ecg.setXRange(minuto * 60, (minuto + 1) * 60, padding=0.02)
        if self._marca_minuto is not None:
            self._marca_minuto.setPos(minuto)


# =============================================================================
# SOLAPA MEDICA (solo Apnea-ECG)
# =============================================================================

class SolapaMedica(QWidget):
    """Vista clinica/tamizaje: timeline de la noche, datos del paciente,
    resumen a nivel sujeto y explicabilidad de features del minuto.
    Solo aplica a Apnea-ECG (UCDDB no tiene datos clinicos de paciente)."""

    N_FEATURES_PANEL = 12

    def __init__(self, datos, parent=None):
        super().__init__(parent)
        self.datos = datos
        self._record = None
        self._ecg = None
        self._feat = None
        self._pred = None
        self._min_sel = None
        self._build_ui()

    def _build_ui(self):
        self.stack = QtWidgets.QStackedLayout(self)

        # Pagina 0: aviso para UCDDB
        self.aviso = QLabel(
            'La vista clínica no está disponible para registros de UCDDB.\n\n'
            'No se dispone de los datos clínicos del paciente (edad, sexo, '
            'AHI de referencia) necesarios para el tamizaje médico.\n'
            'Seleccioná un registro de Apnea-ECG (a/b/c) para ver esta solapa.')
        self.aviso.setAlignment(Qt.AlignCenter)
        self.aviso.setWordWrap(True)
        self.aviso.setStyleSheet('color:#667; font-size:14px; padding:40px;')
        w_aviso = QWidget(); la = QVBoxLayout(w_aviso); la.addWidget(self.aviso)
        self.stack.addWidget(w_aviso)

        # Pagina 1: la vista medica
        w_med = QWidget()
        main = QHBoxLayout(w_med)

        # izquierda: ECG del minuto + timeline
        left = QWidget()
        ll = QVBoxLayout(left)
        self.lbl_fc = QLabel('Minuto seleccionado')
        self.lbl_fc.setAlignment(Qt.AlignCenter)
        self.lbl_fc.setStyleSheet('font-size: 20px; font-weight: bold; color:#c02020;')
        ll.addWidget(self.lbl_fc)

        self.plot_ecg = pg.PlotWidget(title='ECG del minuto seleccionado')
        self.plot_ecg.setLabel('left', 'ECG (mV)')
        self.plot_ecg.setLabel('bottom', 'Tiempo (s)')
        self.plot_ecg.showGrid(x=True, y=True, alpha=0.3)
        ll.addWidget(self.plot_ecg, stretch=1)

        ll.addWidget(QLabel('<b>Noche completa</b> — click para inspeccionar un minuto'))
        self.lbl_resumen_timeline = QLabel('')
        ll.addWidget(self.lbl_resumen_timeline)
        ll.addWidget(QLabel('Predicción del modelo'))
        self.timeline_pred = self._crear_tira(eje_x=False)
        ll.addWidget(self.timeline_pred)
        ll.addWidget(QLabel('Etiqueta clínica real'))
        self.timeline_real = self._crear_tira(eje_x=True)
        ll.addWidget(self.timeline_real)
        main.addWidget(left, stretch=1)

        # derecha: paciente + resumen + features
        right = QWidget()
        right.setMaximumWidth(380)
        rl = QVBoxLayout(right)

        self.gb_paciente = QGroupBox('Paciente')
        fp = QFormLayout(self.gb_paciente)
        self.lbl_edad = QLabel('-'); self.lbl_sexo = QLabel('-')
        self.lbl_altura = QLabel('-'); self.lbl_peso = QLabel('-')
        self.lbl_ahi = QLabel('-'); self.lbl_dur = QLabel('-')
        fp.addRow('Edad:', self.lbl_edad); fp.addRow('Sexo:', self.lbl_sexo)
        fp.addRow('Altura:', self.lbl_altura); fp.addRow('Peso:', self.lbl_peso)
        fp.addRow('AHI / AI / HI real:', self.lbl_ahi)
        fp.addRow('Duración:', self.lbl_dur)
        rl.addWidget(self.gb_paciente)

        gb_res = QGroupBox('Resumen del registro (nivel sujeto)')
        fr = QFormLayout(gb_res)
        self.lbl_res_real = QLabel('-'); self.lbl_res_pred = QLabel('-')
        self.lbl_res_ahi = QLabel('-')
        fr.addRow('Apnea real (clínica):', self.lbl_res_real)
        fr.addRow('Apnea predicha (modelo):', self.lbl_res_pred)
        fr.addRow('AHI estimado (modelo):', self.lbl_res_ahi)
        nota = QLabel('* Referencia clínica: AHI ≥ 5 sugiere apnea. Es la tasa '
                      'de minutos predichos, no un diagnóstico.')
        nota.setWordWrap(True); nota.setStyleSheet('color:#889; font-size:10px;')
        fr.addRow(nota)
        rl.addWidget(gb_res)

        gb_min = QGroupBox('Minuto seleccionado')
        fm = QFormLayout(gb_min)
        self.lbl_min_num = QLabel('-'); self.lbl_min_real = QLabel('-')
        self.lbl_min_pred = QLabel('-')
        fm.addRow('Minuto:', self.lbl_min_num)
        fm.addRow('Etiqueta real:', self.lbl_min_real)
        fm.addRow('Predicción modelo:', self.lbl_min_pred)
        rl.addWidget(gb_min)

        gb_feat = QGroupBox('¿Por qué apnea? — features vs. referencia')
        vf = QVBoxLayout(gb_feat)
        self.tbl_feat = QTableWidget(0, 3)
        self.tbl_feat.setHorizontalHeaderLabels(['Feature', 'Valor', 'Normal ⟷ Apnea'])
        self.tbl_feat.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.tbl_feat.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.tbl_feat.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.tbl_feat.verticalHeader().setVisible(False)
        self.tbl_feat.verticalHeader().setDefaultSectionSize(36)
        self.tbl_feat.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tbl_feat.setSelectionMode(QAbstractItemView.NoSelection)
        vf.addWidget(self.tbl_feat)
        rl.addWidget(gb_feat, stretch=1)

        main.addWidget(right)
        self.stack.addWidget(w_med)

    def _crear_tira(self, eje_x, alto=44):
        w = pg.PlotWidget()
        w.setMinimumHeight(alto); w.setMaximumHeight(alto + 6)
        w.getPlotItem().hideAxis('left')
        if eje_x:
            w.setLabel('bottom', 'Minuto')
        else:
            w.getPlotItem().hideAxis('bottom')
        w.setMouseEnabled(x=False, y=False)
        return w

    # ---------------------------------------------------------------------
    def cargar_registro(self, record):
        self._record = record
        if es_ucd(record):
            self.stack.setCurrentIndex(0)   # aviso
            return
        self.stack.setCurrentIndex(1)

        # ECG
        ecg_raw, _, _ = cargar_ecg(record, DATA_DIR)
        self._ecg = filtrar_ecg_general(ecg_raw, FS)

        # features + predicciones de este registro
        feats = self.datos.features_apnea
        preds = self.datos.pred_apnea
        self._feat = feats[feats['record'] == record].sort_values('minute').reset_index(drop=True)
        self._pred = preds[preds['record'] == record].sort_values('minute').reset_index(drop=True) \
            if preds is not None else pd.DataFrame()
        self._feat_idx = self._feat.set_index('minute')

        # paciente
        info = _cargar_info_paciente(record, DATA_DIR)
        if info:
            self.lbl_edad.setText(f"{info['edad']} años")
            self.lbl_sexo.setText('Masculino' if info['sexo'] == 'M' else 'Femenino')
            self.lbl_altura.setText(f"{info['altura_cm']} cm")
            self.lbl_peso.setText(f"{info['peso_kg']} kg")
            self.lbl_ahi.setText(f"{info['AHI']:.1f} / {info['AI']:.1f} / {info['HI']:.1f}")
            self.lbl_dur.setText(f"{info['duracion_min']} min")
        else:
            for l in [self.lbl_edad, self.lbl_sexo, self.lbl_altura,
                      self.lbl_peso, self.lbl_ahi, self.lbl_dur]:
                l.setText('sin datos')

        self._construir_timeline()
        # mostrar el primer minuto con apnea predicha, o el 0
        if len(self._pred) and (self._pred['pred'] == 1).any():
            m0 = int(self._pred[self._pred['pred'] == 1]['minute'].iloc[0])
        else:
            m0 = int(self._feat['minute'].iloc[0]) if len(self._feat) else 0
        self._mostrar_minuto(m0)

    def _construir_timeline(self):
        for tl in (self.timeline_pred, self.timeline_real):
            tl.clear()
        if len(self._pred) == 0:
            return
        preds = self._pred['pred'].fillna(0).astype(int).values
        hay_real = 'y_true' in self._pred.columns
        reales = self._pred['y_true'].fillna(0).astype(int).values if hay_real else None
        min0 = int(self._pred['minute'].min()); n = len(preds)
        self._min0, self._n = min0, n

        self._pintar_tira(self.timeline_pred, preds, min0, n)
        self.cursor_pred = pg.InfiniteLine(angle=90, pen=pg.mkPen('#1040c0', width=2))
        self.timeline_pred.addItem(self.cursor_pred)
        self.timeline_pred.scene().sigMouseClicked.connect(
            lambda ev: self._click_tira(ev, self.timeline_pred))

        self.cursores = [self.cursor_pred]
        if hay_real:
            self._pintar_tira(self.timeline_real, reales, min0, n)
            self.cursor_real = pg.InfiniteLine(angle=90, pen=pg.mkPen('#1040c0', width=2))
            self.timeline_real.addItem(self.cursor_real)
            self.timeline_real.scene().sigMouseClicked.connect(
                lambda ev: self._click_tira(ev, self.timeline_real))
            self.cursores.append(self.cursor_real)
            self.timeline_real.show()
        else:
            self.timeline_real.hide()

        n_pa = int(preds.sum())
        txt = f'{n_pa}/{n} min predichos apnea ({100*n_pa/n:.1f}%)'
        self.lbl_res_pred.setText(f'{n_pa}/{n} min ({100*n_pa/n:.1f}%)')
        horas = len(self._ecg) / FS / 3600
        self.lbl_res_ahi.setText(f'{n_pa/horas:.1f} min-apnea/h' if horas > 0 else '-')
        if hay_real:
            n_ra = int(reales.sum())
            txt += f'   |   {n_ra}/{n} min reales ({100*n_ra/n:.1f}%)'
            self.lbl_res_real.setText(f'{n_ra}/{n} min ({100*n_ra/n:.1f}%)')
        else:
            self.lbl_res_real.setText('sin datos')
        self.lbl_resumen_timeline.setText(txt)

    def _pintar_tira(self, widget, valores, min0, n):
        img = np.zeros((n, 1, 4), dtype=np.ubyte)
        for k, v in enumerate(valores):
            c = APNEA_RGB if v == 1 else NORMAL_RGB
            img[k, 0] = (c[0], c[1], c[2], 255)
        item = pg.ImageItem(img)
        item.setRect(QtCore.QRectF(min0, 0, n, 1))
        widget.addItem(item)
        widget.setYRange(0, 1, padding=0)
        widget.setXRange(min0, min0 + n, padding=0)

    def _click_tira(self, event, widget):
        if event.button() != Qt.LeftButton:
            return
        vb = widget.getPlotItem().vb
        pos = vb.mapSceneToView(event.scenePos())
        m = int(round(pos.x()))
        m = int(np.clip(m, self._min0, self._min0 + self._n - 1))
        self._mostrar_minuto(m)

    def _mostrar_minuto(self, minuto):
        self._min_sel = minuto
        for c in getattr(self, 'cursores', []):
            c.setPos(minuto + 0.5)
        # ECG del minuto
        self.plot_ecg.clear()
        if self._ecg is not None:
            i0, i1 = minuto * 60 * FS, (minuto + 1) * 60 * FS
            i1 = min(i1, len(self._ecg))
            if i0 < i1:
                seg = self._ecg[i0:i1]
                t = np.arange(len(seg)) / FS
                self.plot_ecg.plot(t, seg, pen=pg.mkPen('#2f5d9e', width=1))
        self.lbl_fc.setText(f'Minuto {minuto}')
        self.lbl_min_num.setText(str(minuto))
        # etiqueta y prediccion
        fila = self._pred[self._pred['minute'] == minuto]
        if len(fila):
            fila = fila.iloc[0]
            yt = fila.get('y_true', np.nan)
            real = 'APNEA' if yt == 1 else 'Normal'
            self.lbl_min_real.setText(real)
            self.lbl_min_real.setStyleSheet(
                'font-weight:bold;color:#c02020' if real == 'APNEA' else 'font-weight:bold;color:#207020')
            pr = int(fila.get('pred', 0)); proba = fila.get('proba_apnea', np.nan)
            txt = f"{'APNEA' if pr == 1 else 'Normal'} (p={proba:.2f})"
            self.lbl_min_pred.setText(txt)
            self.lbl_min_pred.setStyleSheet(
                'font-weight:bold;color:#c02020' if pr == 1 else 'font-weight:bold;color:#207020')
        self._llenar_tabla_features(minuto)

    def _llenar_tabla_features(self, minuto):
        self.tbl_feat.setRowCount(0)
        if minuto not in self._feat_idx.index:
            return
        fila = self._feat_idx.loc[minuto]
        ranking = self.datos.ranking
        orden = self.datos.orden_features[:self.N_FEATURES_PANEL]
        for feat in orden:
            if feat not in fila.index:
                continue
            val = fila[feat]
            r = self.tbl_feat.rowCount()
            self.tbl_feat.insertRow(r)
            self.tbl_feat.setItem(r, 0, QTableWidgetItem(FEATURE_LABELS.get(feat, feat)))
            self.tbl_feat.setItem(r, 1, QTableWidgetItem('-' if pd.isna(val) else f'{val:.3g}'))
            mn = ma = None
            if ranking is not None and feat in ranking.index:
                ref = ranking.loc[feat]
                mn, ma = float(ref['media_N']), float(ref['media_A'])
            self.tbl_feat.setCellWidget(r, 2, BarraComparacion(val, mn, ma))


# =============================================================================
# Contenedor de datos compartidos (se cargan una sola vez)
# =============================================================================

class DatosCompartidos:
    """Carga y guarda los CSV del cache que comparten las dos solapas."""

    def __init__(self):
        # Apnea-ECG
        self.features_apnea = self._leer(os.path.join(CACHE_DIR, 'features_apnea.csv'))
        self.pred_apnea = self._leer(os.path.join(CACHE_DIR, 'oof_predicciones.csv'))
        # UCDDB (opcionales)
        self.features_ucd = self._leer(os.path.join(CACHE_UCD_PROC, 'features_ucd.csv'))
        self.pred_ucd = self._leer(os.path.join(CACHE_UCD_PROC, 'predicciones_ucd.csv'))
        # referencia normal/apnea + orden por importancia
        self.ranking = self._leer(os.path.join(CACHE_DIR, 'ranking_features.csv'))
        if self.ranking is not None and 'feature' in self.ranking.columns:
            self.ranking = self.ranking.set_index('feature')
        imp = self._leer(os.path.join(CACHE_DIR, 'importancias.csv'))
        if imp is not None:
            imp = imp.sort_values('importancia', ascending=False)
            self.orden_features = imp['feature'].tolist()
        else:
            self.orden_features = list(FEATURE_LABELS.keys())

    @staticmethod
    def _leer(path):
        return pd.read_csv(path) if os.path.exists(path) else None

    def lista_registros(self):
        """Todos los registros disponibles, Apnea-ECG primero y UCDDB despues."""
        regs = []
        if self.features_apnea is not None:
            regs += sorted(self.features_apnea['record'].unique().tolist())
        if self.features_ucd is not None:
            regs += sorted(self.features_ucd['record'].unique().tolist())
        return regs


# =============================================================================
# Ventana principal con las dos solapas
# =============================================================================

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle('Detección de apnea del sueño — visualización')
        self.resize(1500, 900)

        self.datos = DatosCompartidos()
        if self.datos.features_apnea is None:
            QMessageBox.critical(self, 'Error',
                f'No se encontró {CACHE_DIR}/features_apnea.csv.\n'
                'Correr el pipeline (04, 05) antes de abrir la interfaz.')
            sys.exit(1)

        central = QWidget()
        self.setCentralWidget(central)
        v = QVBoxLayout(central)

        # --- Barra superior: selector compartido ---
        top = QHBoxLayout()
        top.addWidget(QLabel('<b>Registro:</b>'))
        self.combo = QComboBox()
        self.combo.addItems(self.datos.lista_registros())
        self.combo.currentTextChanged.connect(self._cambiar_registro)
        top.addWidget(self.combo)
        top.addWidget(QLabel('  (a/b/c = Apnea-ECG · ucddb* = UCDDB)'))
        top.addStretch()
        v.addLayout(top)

        # --- Solapas ---
        self.tabs = QTabWidget()
        self.solapa_tecnica = SolapaTecnica(self.datos)
        self.solapa_medica = SolapaMedica(self.datos)
        self.tabs.addTab(self.solapa_tecnica, 'Vista técnica')
        self.tabs.addTab(self.solapa_medica, 'Vista clínica')
        v.addWidget(self.tabs)

        self.setStatusBar(QStatusBar())

        if self.combo.count():
            self._cambiar_registro(self.combo.currentText())

    def _cambiar_registro(self, record):
        if not record:
            return
        self.statusBar().showMessage(f'Cargando {record}...')
        QApplication.processEvents()
        try:
            self.solapa_tecnica.cargar_registro(record)
            self.solapa_medica.cargar_registro(record)
            self.statusBar().showMessage(f'{record} — listo')
        except Exception as e:
            self.statusBar().showMessage(f'Error: {e}')
            QMessageBox.warning(self, 'Error', f'No se pudo cargar {record}:\n{e}')


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()