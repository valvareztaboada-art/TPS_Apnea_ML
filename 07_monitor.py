# -*- coding: utf-8 -*-
"""
Monitor ECG en tiempo real — R + FC + timeline de la noche + explicabilidad
============================================================================
- ECG scrolleando con R marcados, FC latido a latido.
- Timeline de toda la noche con DOS franjas (arriba: prediccion del modelo,
  abajo: etiqueta real/clinica) — permite ver de un vistazo falsos
  positivos/negativos. Es CLICKEABLE: clickeando un minuto la interfaz
  saltea el reproductor ahi mismo (y se pausa) para poder inspeccionarlo.
- Botones para pausar/reanudar y para saltar directo al minuto de apnea
  anterior/siguiente (predicha).
- Panel de datos del paciente (edad, sexo, altura, peso, AHI/AI/HI reales)
  leidos de additional-information.txt.
- Panel de features del minuto que se esta viendo: valor actual vs. la
  media tipica en minutos normales y en minutos con apnea (ranking_features
  .csv), ordenadas por importancia en el modelo final (importancias.csv).
  Responde "por que el modelo cree que ac hay apnea".
Todo sale del cache (picos_R, features_apnea.csv, oof_predicciones.csv,
ranking_features.csv, importancias.csv).
Correr desde la MISMA carpeta que interfaz_apnea.py.
"""

import os
import sys

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets


HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from src.pipeline import cargar_ecg, filtrar_ecg_general


# ------------------------------- Config -------------------------------
DATA_DIR  = 'apnea-ecg-database-1.0.0'
CACHE_DIR = 'cache'
RECORD    = 'c01'
FS        = 100
VENTANA_S = 20
DT_MS     = 40
VELOCIDAD = 1.0        # 1.0 = tiempo real. Subilo para ver el cursor barrer la noche
N_PROMEDIO_FC   = 5    # latidos para suavizar la FC
N_FEATURES_PANEL = 21  # cuantas features mostrar en el panel (ordenadas por importancia)

APNEA_RGB  = (200, 60, 60)     # rojo
NORMAL_RGB = (210, 235, 210)   # verde claro

ADDITIONAL_INFO_FILE = 'additional-information.txt'

ESTILO_BOTON_NAV = """
QPushButton {
    background-color: #2b2f3a;
    color: #e8e8e8;
    border: 1px solid #454d61;
    border-radius: 8px;
    padding: 8px 16px;
    font-size: 13px;
    font-weight: 600;
}
QPushButton:hover  { background-color: #394158; border-color: #5a6480; }
QPushButton:pressed { background-color: #1f232c; }
"""

ESTILO_BOTON_PLAY = """
QPushButton {
    background-color: #c02020;
    color: white;
    border: none;
    border-radius: 24px;
    font-size: 20px;
}
QPushButton:hover   { background-color: #d43333; }
QPushButton:pressed { background-color: #9c1a1a; }
"""

class BarraComparacion(QtWidgets.QWidget):
    """Ubica el valor actual de una feature en una barra entre la media
    tipica de minutos normales (izquierda, verde) y de minutos con apnea
    (derecha, rojo). Mucho mas facil de leer de un vistazo que 3 numeros."""

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
        p.drawText(QtCore.QRectF(x0, barra_y + barra_h + 3, 60, 14), QtCore.Qt.AlignLeft, txt_n)
        p.drawText(QtCore.QRectF(x1 - 60, barra_y + barra_h + 3, 60, 14), QtCore.Qt.AlignRight, txt_a)


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


def _cargar_info_paciente(record, data_dir, path=ADDITIONAL_INFO_FILE):
    """Parsea additional-information.txt y devuelve un dict con los datos
    del paciente para `record`, o None si no esta (p.ej. registros x*)."""
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
                    'apnea_min':    int(partes[3]),
                    'horas_con_apnea': int(partes[4]),
                    'AI': float(partes[5]), 'HI': float(partes[6]), 'AHI': float(partes[7]),
                    'edad': int(partes[8]), 'sexo': partes[9],
                    'altura_cm': int(partes[10]), 'peso_kg': int(partes[11]),
                }
            except ValueError:
                return None
    return None


class Monitor(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f'Monitor ECG — noche completa — {RECORD}')
        self.resize(1400, 720)

        # --- ECG filtrado ---
        ecg_raw, _, _ = cargar_ecg(RECORD, DATA_DIR)
        self.ecg = filtrar_ecg_general(ecg_raw, FS)
        self.t   = np.arange(len(self.ecg)) / FS

        # --- Picos R desde el cache ---
        cache = np.load(os.path.join(CACHE_DIR, f'{RECORD}.npz'))
        self.picos = np.asarray(cache['picos_R']).astype(int)

        # --- Predicciones por minuto (out-of-fold) de este registro ---
        pred = pd.read_csv(os.path.join(CACHE_DIR, 'oof_predicciones.csv'))
        self.pred = pred[pred['record'] == RECORD].sort_values('minute').reset_index(drop=True)
        self.pred_idx = self.pred.set_index('minute') if len(self.pred) else self.pred

        # --- Features por minuto de este registro ---
        feats = pd.read_csv(os.path.join(CACHE_DIR, 'features_apnea.csv'))
        self.features = feats[feats['record'] == RECORD].sort_values('minute').reset_index(drop=True)
        self.feat_idx = self.features.set_index('minute') if len(self.features) else self.features

        # --- Referencia global normal/apnea por feature + importancia del modelo ---
        self.ranking = pd.read_csv(os.path.join(CACHE_DIR, 'ranking_features.csv')).set_index('feature')
        importancias = pd.read_csv(os.path.join(CACHE_DIR, 'importancias.csv'))
        importancias = importancias.sort_values('importancia', ascending=False)
        self.orden_features = [f for f in importancias['feature'].tolist()][:N_FEATURES_PANEL]

        # --- Datos del paciente ---
        self.info_paciente = _cargar_info_paciente(RECORD, DATA_DIR)

        self._minuto_mostrado = None

        # ================= Layout =================
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        outer = QtWidgets.QHBoxLayout(central)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        outer.addWidget(splitter)

        splitter.addWidget(self._crear_panel_izquierdo())
        splitter.addWidget(self._crear_panel_derecho())
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)

        self._llenar_panel_paciente()
        self._construir_timeline()

        # --- Estado del reproductor ---
        self.N    = int(VENTANA_S * FS)
        self.step = max(1, int(FS * DT_MS / 1000 * VELOCIDAD))
        self.i    = self.N

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._tick)
        self.timer.start(DT_MS)

        self._draw_frame()

    # ---------------------------------------------------------------------
    # Construccion de la UI
    # ---------------------------------------------------------------------
    def _crear_panel_izquierdo(self):
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)

        # FC arriba
        self.lbl_fc = QtWidgets.QLabel('-- bpm')
        self.lbl_fc.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_fc.setStyleSheet(
            'font-size: 34px; font-weight: bold; color: #c02020; padding: 4px;')
        layout.addWidget(self.lbl_fc)

        # ECG en el medio — zoom horizontal libre (rueda / arrastre). El eje
        # Y se reescala solo, siempre, para ajustarse a lo que este visible
        # en pantalla (nunca queda espacio muerto ni se recorta la señal).
        self.plot = pg.PlotWidget()
        self.plot.setLabel('left', 'ECG (mV)')
        self.plot.setLabel('bottom', 'Tiempo (s)')
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.plot.setMouseEnabled(x=True, y=False)
        layout.addWidget(self.plot, stretch=1)

        self.curve = self.plot.plot(pen=pg.mkPen('#2f5d9e', width=1))
        self.curve_r = self.plot.plot(
            pen=None, symbol='o', symbolBrush=(220, 40, 40),
            symbolPen=pg.mkPen('w', width=1), symbolSize=11)

        self.plot.getViewBox().sigXRangeChanged.connect(
            lambda vb, rng: self._auto_y_range(rng))

        # Controles: anterior / pausa-reanuda / siguiente, agrupados y centrados
        controles = QtWidgets.QHBoxLayout()
        controles.setSpacing(14)

        self.btn_prev = QtWidgets.QPushButton('◀  Apnea anterior')
        self.btn_prev.setStyleSheet(ESTILO_BOTON_NAV)
        self.btn_prev.setCursor(QtCore.Qt.PointingHandCursor)
        self.btn_prev.clicked.connect(lambda: self._saltar_apnea(-1))

        self.btn_play = QtWidgets.QPushButton('⏸')
        self.btn_play.setFixedSize(48, 48)
        self.btn_play.setStyleSheet(ESTILO_BOTON_PLAY)
        self.btn_play.setCursor(QtCore.Qt.PointingHandCursor)
        self.btn_play.setToolTip('Pausar')
        self.btn_play.clicked.connect(self._toggle_play)

        self.btn_next = QtWidgets.QPushButton('Apnea siguiente  ▶')
        self.btn_next.setStyleSheet(ESTILO_BOTON_NAV)
        self.btn_next.setCursor(QtCore.Qt.PointingHandCursor)
        self.btn_next.clicked.connect(lambda: self._saltar_apnea(+1))

        controles.addStretch(1)
        controles.addWidget(self.btn_prev)
        controles.addWidget(self.btn_play)
        controles.addWidget(self.btn_next)
        controles.addStretch(1)
        layout.addLayout(controles)

        # Timeline: DOS TIRAS separadas y bien identificadas, cada una clickeable
        layout.addWidget(QtWidgets.QLabel(
            '<b>Noche completa</b> — click en cualquiera de las dos tiras para saltar a ese minuto'))
        self.lbl_resumen_timeline = QtWidgets.QLabel('')
        layout.addWidget(self.lbl_resumen_timeline)

        lbl_pred = QtWidgets.QLabel('Predicción del modelo')
        lbl_pred.setStyleSheet('font-weight:600; color:#d0d4de;')
        layout.addWidget(lbl_pred)
        self.timeline_pred = self._crear_tira_timeline(eje_x=False)
        layout.addWidget(self.timeline_pred)

        lbl_real = QtWidgets.QLabel('Etiqueta clínica real')
        lbl_real.setStyleSheet('font-weight:600; color:#d0d4de; margin-top:4px;')
        layout.addWidget(lbl_real)
        self.timeline_real = self._crear_tira_timeline(eje_x=True)
        layout.addWidget(self.timeline_real)

        return panel

    def _crear_tira_timeline(self, eje_x, alto_min=38, alto_max=46):
        w = pg.PlotWidget()
        w.setMinimumHeight(alto_min)
        w.setMaximumHeight(alto_max)
        w.getPlotItem().hideAxis('left')
        if eje_x:
            w.setLabel('bottom', 'Minuto')
        else:
            w.getPlotItem().hideAxis('bottom')
        w.setMouseEnabled(x=False, y=False)
        return w

    def _crear_panel_derecho(self):
        panel = QtWidgets.QWidget()
        panel.setMaximumWidth(380)
        layout = QtWidgets.QVBoxLayout(panel)

        # --- Datos del paciente ---
        grupo_paciente = QtWidgets.QGroupBox(f'Paciente — registro {RECORD}')
        form = QtWidgets.QFormLayout(grupo_paciente)
        self.lbl_pac_edad = QtWidgets.QLabel('-')
        self.lbl_pac_sexo = QtWidgets.QLabel('-')
        self.lbl_pac_altura = QtWidgets.QLabel('-')
        self.lbl_pac_peso = QtWidgets.QLabel('-')
        self.lbl_pac_ahi = QtWidgets.QLabel('-')
        self.lbl_pac_duracion = QtWidgets.QLabel('-')
        form.addRow('Edad:', self.lbl_pac_edad)
        form.addRow('Sexo:', self.lbl_pac_sexo)
        form.addRow('Altura:', self.lbl_pac_altura)
        form.addRow('Peso:', self.lbl_pac_peso)
        form.addRow('AHI / AI / HI real:', self.lbl_pac_ahi)
        form.addRow('Duracion registro:', self.lbl_pac_duracion)
        layout.addWidget(grupo_paciente)

        # --- Resumen agregado a nivel sujeto: cuenta TODOS los minutos
        # predichos como apnea, aunque sean pocos y el registro sea en su
        # mayoria normal (para no "perder" apneas aisladas / falsos positivos).
        grupo_resumen = QtWidgets.QGroupBox('Resumen del registro (nivel sujeto)')
        form_r = QtWidgets.QFormLayout(grupo_resumen)
        self.lbl_resumen_real = QtWidgets.QLabel('-')
        self.lbl_resumen_pred = QtWidgets.QLabel('-')
        self.lbl_resumen_ahi = QtWidgets.QLabel('-')
        form_r.addRow('Apnea real (clinica):', self.lbl_resumen_real)
        form_r.addRow('Apnea predicha (modelo):', self.lbl_resumen_pred)
        form_r.addRow('AHI estimado (modelo):', self.lbl_resumen_ahi)
        nota = QtWidgets.QLabel(
            '* Referencia clínica: AHI ≥ 5 sugiere apnea. El umbral calibrado '
            'por sujeto (paso 6 del proyecto) todavía no está implementado; '
            'esto es solo la tasa de minutos predichos, no un diagnóstico.')
        nota.setWordWrap(True)
        nota.setStyleSheet('color:#8890a0; font-size:10px;')
        form_r.addRow(nota)
        layout.addWidget(grupo_resumen)

        # --- Minuto seleccionado ---
        grupo_minuto = QtWidgets.QGroupBox('Minuto en pantalla')
        form2 = QtWidgets.QFormLayout(grupo_minuto)
        self.lbl_min_num = QtWidgets.QLabel('-')
        self.lbl_min_real = QtWidgets.QLabel('-')
        self.lbl_min_pred = QtWidgets.QLabel('-')
        form2.addRow('Minuto:', self.lbl_min_num)
        form2.addRow('Etiqueta real:', self.lbl_min_real)
        form2.addRow('Prediccion modelo:', self.lbl_min_pred)
        layout.addWidget(grupo_minuto)

        # --- Features del minuto vs. referencia normal/apnea ---
        grupo_feat = QtWidgets.QGroupBox('¿Por que apnea? — features vs. referencia global')
        v = QtWidgets.QVBoxLayout(grupo_feat)
        self.tbl_features = QtWidgets.QTableWidget(0, 3)
        self.tbl_features.setHorizontalHeaderLabels(['Feature', 'Valor', 'Normal  ⟷  Apnea'])
        self.tbl_features.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        self.tbl_features.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        self.tbl_features.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.Stretch)
        self.tbl_features.verticalHeader().setVisible(False)
        self.tbl_features.verticalHeader().setDefaultSectionSize(36)
        self.tbl_features.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.tbl_features.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.tbl_features.setToolTip(
            'El punto ubica el valor actual entre el promedio típico en minutos '
            'normales (verde, izquierda) y en minutos con apnea (rojo, derecha).')
        v.addWidget(self.tbl_features)
        layout.addWidget(grupo_feat, stretch=1)

        return panel

    def _llenar_panel_paciente(self):
        info = self.info_paciente
        if info is None:
            self.lbl_pac_edad.setText('sin datos')
            return
        self.lbl_pac_edad.setText(f"{info['edad']} años")
        self.lbl_pac_sexo.setText('Masculino' if info['sexo'] == 'M' else 'Femenino')
        self.lbl_pac_altura.setText(f"{info['altura_cm']} cm")
        self.lbl_pac_peso.setText(f"{info['peso_kg']} kg")
        self.lbl_pac_ahi.setText(f"{info['AHI']:.1f} / {info['AI']:.1f} / {info['HI']:.1f}")
        self.lbl_pac_duracion.setText(f"{info['duracion_min']} min "
                                       f"({info['apnea_min']} min con apnea, "
                                       f"{info['horas_con_apnea']} h con apnea)")

    # ---------------------------------------------------------------------
    def _construir_timeline(self):
        """Pinta las dos tiras (predicho / real), con cursor sincronizado y
        clicks para saltar de minuto. Tambien llena el resumen a nivel
        sujeto: CUALQUIER minuto predicho como apnea se cuenta ahi, aunque
        sea uno solo y el registro sea en su mayoria normal."""
        if len(self.pred) == 0 or 'pred' not in self.pred.columns:
            return
        preds = self.pred['pred'].fillna(0).astype(int).values
        hay_real = 'y_true' in self.pred.columns
        reales = self.pred['y_true'].fillna(0).astype(int).values if hay_real else None
        min0 = int(self.pred['minute'].min())
        n = len(preds)
        self._min0_timeline = min0
        self._n_timeline = n

        self._pintar_tira(self.timeline_pred, preds, min0, n)
        self.cursor_pred = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('#1040c0', width=2))
        self.timeline_pred.addItem(self.cursor_pred)
        self.timeline_pred.scene().sigMouseClicked.connect(
            lambda ev: self._click_en_tira(ev, self.timeline_pred))
        self.cursores = [self.cursor_pred]

        if hay_real:
            self._pintar_tira(self.timeline_real, reales, min0, n)
            self.cursor_real = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('#1040c0', width=2))
            self.timeline_real.addItem(self.cursor_real)
            self.timeline_real.scene().sigMouseClicked.connect(
                lambda ev: self._click_en_tira(ev, self.timeline_real))
            self.cursores.append(self.cursor_real)
        else:
            self.timeline_real.hide()

        n_pred_apnea = int(preds.sum())
        texto = f'{n_pred_apnea}/{n} min predichos como apnea ({100*n_pred_apnea/n:.1f}%)'
        if hay_real:
            n_real_apnea = int(reales.sum())
            texto += f'   |   {n_real_apnea}/{n} min reales con apnea ({100*n_real_apnea/n:.1f}%)'
        self.lbl_resumen_timeline.setText(texto)

        # --- Resumen a nivel sujeto (panel derecho) ---
        horas = len(self.ecg) / FS / 3600
        ahi_estimado = n_pred_apnea / horas if horas > 0 else float('nan')
        self.lbl_resumen_pred.setText(f'{n_pred_apnea}/{n} min ({100*n_pred_apnea/n:.1f}%)')
        self.lbl_resumen_ahi.setText(f'{ahi_estimado:.1f} min-apnea/h')
        if hay_real:
            n_real_apnea = int(reales.sum())
            self.lbl_resumen_real.setText(f'{n_real_apnea}/{n} min ({100*n_real_apnea/n:.1f}%)')
        else:
            self.lbl_resumen_real.setText('sin datos')

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

    def _click_en_tira(self, event, widget):
        if event.button() != QtCore.Qt.LeftButton:
            return
        vb = widget.getPlotItem().vb
        pos = vb.mapSceneToView(event.scenePos())
        self._ir_a_minuto(pos.x())

    # ---------------------------------------------------------------------
    # Navegacion
    # ---------------------------------------------------------------------
    def _ir_a_minuto(self, minuto):
        min0 = getattr(self, '_min0_timeline', 0)
        n = getattr(self, '_n_timeline', None)
        minuto = int(round(minuto))
        if n is not None:
            minuto = int(np.clip(minuto, min0, min0 + n - 1))
        else:
            minuto = max(0, minuto)

        idx = int(np.clip(minuto * 60 * FS, self.N, len(self.ecg) - 1))
        self.i = idx
        self._pausar()
        self._minuto_mostrado = None   # fuerza refresco del panel
        self._draw_frame()

    def _saltar_apnea(self, direccion):
        if not hasattr(self.pred_idx, 'index') or 'pred' not in getattr(self.pred_idx, 'columns', []):
            return
        minutos_apnea = sorted(self.pred_idx.index[self.pred_idx['pred'] == 1].tolist())
        if not minutos_apnea:
            return
        actual = self._minuto_mostrado if self._minuto_mostrado is not None else 0
        if direccion > 0:
            candidatos = [m for m in minutos_apnea if m > actual]
            objetivo = candidatos[0] if candidatos else minutos_apnea[0]
        else:
            candidatos = [m for m in minutos_apnea if m < actual]
            objetivo = candidatos[-1] if candidatos else minutos_apnea[-1]
        self._ir_a_minuto(objetivo)

    def _toggle_play(self):
        if self.timer.isActive():
            self._pausar()
        else:
            self._reanudar()

    def _pausar(self):
        self.timer.stop()
        self.btn_play.setText('▶')
        self.btn_play.setToolTip('Reanudar')

    def _reanudar(self):
        self.timer.start(DT_MS)
        self.btn_play.setText('⏸')
        self.btn_play.setToolTip('Pausar')

    # ---------------------------------------------------------------------
    # Reproduccion
    # ---------------------------------------------------------------------
    def _tick(self):
        self.i += self.step
        if self.i >= len(self.ecg):
            self.i = self.N
        self._draw_frame()

    def _draw_frame(self):
        lo = self.i - self.N
        x = self.t[lo:self.i]
        y = self.ecg[lo:self.i]
        self.curve.setData(x, y)
        self.plot.setXRange(x[0], x[-1], padding=0)

        # R en la ventana
        a = np.searchsorted(self.picos, lo)
        b = np.searchsorted(self.picos, self.i)
        pk = self.picos[a:b]
        self.curve_r.setData(self.t[pk], self.ecg[pk])

        # Cursor en ambas tiras de la timeline (minuto actual)
        if hasattr(self, 'cursores'):
            minuto_pos = self.i / FS / 60.0
            for c in self.cursores:
                c.setPos(minuto_pos)

        self._actualizar_fc()

        minuto_actual = int(self.i / FS / 60)
        if minuto_actual != self._minuto_mostrado:
            self._actualizar_panel_minuto(minuto_actual)
            self._minuto_mostrado = minuto_actual

    def _auto_y_range(self, rng):
        """Reescala el eje Y del ECG para que siempre se ajuste a lo que
        esta visible en el eje X (zoomear no deja espacio muerto ni recorta)."""
        x0, x1 = rng
        i0 = max(0, int(x0 * FS))
        i1 = min(len(self.ecg), int(x1 * FS) + 1)
        seg = self.ecg[i0:i1] if i1 > i0 else self.ecg
        y_lo, y_hi = np.percentile(seg, [0.5, 99.5])
        margen = 0.2 * (y_hi - y_lo) if y_hi > y_lo else 0.1
        self.plot.setYRange(y_lo - margen, y_hi + margen, padding=0)

    def _actualizar_fc(self):
        b = np.searchsorted(self.picos, self.i)
        if b < 2:
            return
        ultimos = self.picos[max(0, b - (N_PROMEDIO_FC + 1)):b]
        rr = np.diff(ultimos) / FS
        rr = rr[(rr > 0.3) & (rr < 2.0)]
        if len(rr) == 0:
            return
        self.lbl_fc.setText(f'{60.0 / np.mean(rr):.0f} bpm')

    # ---------------------------------------------------------------------
    # Panel de explicabilidad (minuto + features)
    # ---------------------------------------------------------------------
    def _actualizar_panel_minuto(self, minuto):
        hh, mm = divmod(minuto, 60)
        self.lbl_min_num.setText(f'{minuto}  ({hh:02d}h{mm:02d}m)')

        if minuto in self.pred_idx.index:
            fila = self.pred_idx.loc[minuto]
            etiqueta = fila.get('label', '?')
            texto_real = 'APNEA' if etiqueta == 'A' else ('Normal' if etiqueta == 'N' else '?')
            self.lbl_min_real.setText(texto_real)
            self.lbl_min_real.setStyleSheet(
                'font-weight:bold;color:#c02020' if texto_real == 'APNEA' else 'font-weight:bold;color:#207020')

            pred = int(fila.get('pred', 0))
            proba = fila.get('proba_apnea', np.nan)
            texto_pred = f"{'APNEA' if pred == 1 else 'Normal'} (p={proba:.2f})"
            self.lbl_min_pred.setText(texto_pred)
            self.lbl_min_pred.setStyleSheet(
                'font-weight:bold;color:#c02020' if pred == 1 else 'font-weight:bold;color:#207020')
        else:
            self.lbl_min_real.setText('sin datos')
            self.lbl_min_pred.setText('sin datos')

        self._llenar_tabla_features(minuto)

    def _llenar_tabla_features(self, minuto):
        tabla = self.tbl_features
        tabla.setRowCount(0)
        if minuto not in self.feat_idx.index:
            return
        fila = self.feat_idx.loc[minuto]

        for feat in self.orden_features:
            if feat not in fila.index:
                continue
            val = fila[feat]
            r = tabla.rowCount()
            tabla.insertRow(r)
            tabla.setItem(r, 0, QtWidgets.QTableWidgetItem(FEATURE_LABELS.get(feat, feat)))
            tabla.setItem(r, 1, QtWidgets.QTableWidgetItem('-' if pd.isna(val) else f'{val:.3g}'))

            media_n = media_a = None
            if feat in self.ranking.index:
                ref = self.ranking.loc[feat]
                media_n, media_a = float(ref['media_N']), float(ref['media_A'])
            tabla.setCellWidget(r, 2, BarraComparacion(val, media_n, media_a))


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = Monitor()
    win.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
