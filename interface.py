import sys
import os
import json
import numpy as np
import joblib
import onnxruntime as ort

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QGroupBox, QFrame, QGridLayout, QProgressBar
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ================= ПУТИ =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Папки с моделями
CNN_DIR = os.path.join(BASE_DIR, "CNN")
CNN_LSTM_DIR = os.path.join(BASE_DIR, "CNN_LSTM")
CNN_BIGRU_DIR = os.path.join(BASE_DIR, "CNN_BiGRU")
CNN_STACKED_DIR = os.path.join(BASE_DIR, "CNN_StackedLSTM")

RESULTS_PATH = os.path.join(RESULTS_DIR, "all_results.json")
LE_PATH = os.path.join(DATA_DIR, "label_encoder.pkl")
X_TEST_PATH = os.path.join(DATA_DIR, "X_test.npy")
Y_TEST_PATH = os.path.join(RESULTS_DIR, "y_test.npy")
NPZ_PATH = os.path.join(DATA_DIR, "CWRU_48k_load_1_CNN_data.npz")

# ================= ДАННЫЕ =================
X_test_spec = np.load(X_TEST_PATH)
y_test = np.load(Y_TEST_PATH)
le = joblib.load(LE_PATH)
CLASS_NAMES = {i: str(n) for i, n in enumerate(le.classes_)}

npz_data = np.load(NPZ_PATH)
X_raw = npz_data["data"].astype(np.float64)
y_raw = npz_data["labels"]

FS = 12000
RPM = 1797

# ================= СООТВЕТСТВИЕ ТЕСТ ↔ СЫРЫЕ =================
test_to_raw = []
for i in range(len(X_test_spec)):
    test_sig = X_test_spec[i].flatten()
    best_idx, best_dist = 0, float('inf')
    for j in range(len(X_raw)):
        raw_sig = X_raw[j].flatten()
        dist = np.mean((test_sig - raw_sig) ** 2)
        if dist < best_dist:
            best_dist, best_idx = dist, j
    test_to_raw.append(best_idx)

# ================= ПРИЗНАКИ =================
if X_raw.ndim == 3:
    rms_all = np.sqrt(np.mean(X_raw ** 2, axis=(1, 2)))
    mean_ax = np.mean(X_raw, axis=(1, 2), keepdims=True)
    std_ax = np.std(X_raw, axis=(1, 2))
    kurt_all = np.mean((X_raw - mean_ax) ** 4, axis=(1, 2)) / (std_ax ** 4 + 1e-10)
else:
    rms_all = np.sqrt(np.mean(X_raw ** 2, axis=1))
    mean_ax = np.mean(X_raw, axis=1, keepdims=True)
    std_ax = np.std(X_raw, axis=1)
    kurt_all = np.mean((X_raw - mean_ax) ** 4, axis=1) / (std_ax ** 4 + 1e-10)

type_all = np.array([s.split("_")[0] for s in y_raw])
TYPE_NAMES_RU = {"Ball": "Шарик", "IR": "Внутр. кольцо", "OR": "Наруж. кольцо", "Normal": "Норма"}
TYPE_COLORS = {"Ball": "#e74c3c", "IR": "#3498db", "OR": "#2ecc71", "Normal": "#f39c12"}

# ================= ТОЧНЫЕ ДИАПАЗОНЫ НОРМЫ =================
NORMAL_RMS = (0.059, 0.073)
NORMAL_KURT = (2.48, 3.30)
NORMAL_PEAK = (2.56, 3.62)


# ================= ЗАГРУЗЧИК МОДЕЛЕЙ =================
class ModelLoader:
    def __init__(self):
        self.models = {}
        self.input_names = {}

        # ТОЛЬКО 4 СВЕРТОЧНЫЕ МОДЕЛИ С ВАШИМИ ИМЕНАМИ ФАЙЛОВ
        onnx_models = [
            ("CNN", CNN_DIR, "cnn_model.onnx"),
            ("CNN-LSTM", CNN_LSTM_DIR, "cnn_lstm_model.onnx"),
            ("CNN-BiGRU", CNN_BIGRU_DIR, "cnn_bigru_model.onnx"),
            ("CNN-Stacked LSTM", CNN_STACKED_DIR, "best_model (3).onnx"),  # ВАШ ФАЙЛ!
        ]

        print("=" * 50)
        print("ЗАГРУЗКА 4 СВЕРТОЧНЫХ МОДЕЛЕЙ")
        print("=" * 50)

        for name, model_dir, file_name in onnx_models:
            path = os.path.join(model_dir, file_name)

            # Если файл не найден, пробуем альтернативные имена
            if not os.path.exists(path):
                # Для CNN-Stacked LSTM пробуем другие имена
                if name == "CNN-Stacked LSTM":
                    alt_names = [
                        "best_model.onnx",
                        "cnn_stacked_lstm_model.onnx",
                        "cnn_stacked_lstm.onnx",
                        "stacked_lstm.onnx",
                        "best_model (1).onnx",
                        "best_model (2).onnx",
                        "best_model (3).onnx",
                    ]
                    found = False
                    for alt in alt_names:
                        alt_path = os.path.join(model_dir, alt)
                        if os.path.exists(alt_path):
                            path = alt_path
                            found = True
                            break
                    if not found:
                        print(f"❌ {name} не найдена в {model_dir}")
                        continue
                else:
                    print(f"❌ {name} не найдена: {path}")
                    continue

            try:
                session = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
                self.models[name] = session
                self.input_names[name] = session.get_inputs()[0].name
                print(f"✅ {name} загружена из {os.path.basename(path)}")
            except Exception as e:
                print(f"❌ Ошибка загрузки {name}: {e}")

        print("=" * 50)
        print(f"✅ Загружено: {len(self.models)} моделей")
        print("=" * 50)

    def predict(self, model_name, spec_2d):
        if model_name not in self.models:
            return None, None

        session = self.models[model_name]
        input_name = self.input_names[model_name]

        try:
            # Определяем формат входа
            input_shape = session.get_inputs()[0].shape

            if len(input_shape) == 4:
                # CNN модель (batch, height, width, channels)
                X = spec_2d.reshape(1, 32, 32, 1).astype(np.float32)
            else:
                # Рекуррентные модели (batch, timesteps, features)
                X = spec_2d.reshape(1, 32, 32).astype(np.float32)

            out = session.run(None, {input_name: X})[0]
            cls = int(np.argmax(out, axis=1)[0])
            probs = out[0]
            return cls, probs

        except Exception as e:
            print(f"Ошибка [{model_name}]: {e}")
            return None, None


# ================= ИНТЕРФЕЙС =================
class Dashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Диагностика подшипников — 4 сверточные модели")
        self.setGeometry(50, 50, 1550, 880)

        self.loader = ModelLoader()

        with open(RESULTS_PATH, "r") as f:
            self.results = json.load(f)

        self.current_sample = 38
        self.spec_2d = X_test_spec[self.current_sample]
        self.true_label = int(y_test[self.current_sample])
        self.raw_idx = test_to_raw[self.current_sample]
        self.raw_signal = X_raw[self.raw_idx].flatten()
        self.true_name = CLASS_NAMES.get(self.true_label, "?")

        self.fr = RPM / 60
        self.bpfo = 3.58 * self.fr
        self.bpfi = 5.42 * self.fr
        self.bsf = 2.33 * self.fr
        self.ftf = 0.4 * self.fr

        self.init_ui()

    def status_color(self, value, low, high):
        if low <= value <= high:
            return "#27ae60", "✓ В норме"
        elif value < low * 0.5 or value > high * 1.5:
            return "#e74c3c", "⚠ Отклонение"
        else:
            return "#f39c12", "⚠ Пограничное"

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main = QHBoxLayout(central)
        main.setSpacing(10)

        # ===== ЛЕВАЯ ПАНЕЛЬ: ГРАФИКИ =====
        left = QVBoxLayout()

        sig_group = QGroupBox("Вибросигнал и спектр Фурье")
        sig_group.setFont(QFont("Segoe UI", 11, QFont.Bold))
        sig_layout = QVBoxLayout(sig_group)
        self.fig_sig, (self.ax_time, self.ax_fft) = plt.subplots(2, 1, figsize=(7.5, 5))
        self.fig_sig.tight_layout(pad=2.5)
        self.canvas_sig = FigureCanvas(self.fig_sig)
        sig_layout.addWidget(self.canvas_sig)
        left.addWidget(sig_group, 3)

        rms_group = QGroupBox("Распределение дефектов (RMS и Куртозис)")
        rms_group.setFont(QFont("Segoe UI", 11, QFont.Bold))
        rms_layout = QVBoxLayout(rms_group)
        self.fig_rms, self.ax_rms = plt.subplots(figsize=(7.5, 3.5))
        self.fig_rms.tight_layout(pad=2)
        self.canvas_rms = FigureCanvas(self.fig_rms)
        rms_layout.addWidget(self.canvas_rms)
        left.addWidget(rms_group, 2)

        main.addLayout(left, 3)

        # ===== ПРАВАЯ ПАНЕЛЬ =====
        right = QVBoxLayout()
        right.setSpacing(6)

        # --- Диагноз ---
        diag_group = QGroupBox("Результат диагностики")
        diag_group.setFont(QFont("Segoe UI", 13, QFont.Bold))
        diag_layout = QVBoxLayout(diag_group)
        self.status_label = QLabel("Выберите модель и образец")
        self.status_label.setFont(QFont("Segoe UI", 18, QFont.Bold))
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: #333; padding: 10px; background: #f5f5f5; border-radius: 8px;")
        diag_layout.addWidget(self.status_label)
        self.info_label = QLabel("")
        self.info_label.setFont(QFont("Segoe UI", 11))
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setStyleSheet("color: #666;")
        diag_layout.addWidget(self.info_label)

        self.analysis_label = QLabel("")
        self.analysis_label.setFont(QFont("Segoe UI", 9, QFont.Bold))
        self.analysis_label.setAlignment(Qt.AlignCenter)
        self.analysis_label.setWordWrap(True)
        self.analysis_label.setStyleSheet("padding: 4px;")
        diag_layout.addWidget(self.analysis_label)

        right.addWidget(diag_group)

        # --- Параметры испытаний ---
        test_group = QGroupBox("Режим испытаний")
        test_group.setFont(QFont("Segoe UI", 10, QFont.Bold))
        test_layout = QGridLayout(test_group)
        test_layout.setSpacing(3)

        self.lbl_sample_info = QLabel(f"Образец №{self.current_sample}")
        self.lbl_sample_info.setFont(QFont("Consolas", 10, QFont.Bold))
        test_layout.addWidget(QLabel("Образец:"), 0, 0)
        test_layout.addWidget(self.lbl_sample_info, 0, 1)

        self.lbl_true_class = QLabel(self.true_name)
        self.lbl_true_class.setFont(QFont("Consolas", 10, QFont.Bold))
        test_layout.addWidget(QLabel("Класс дефекта:"), 1, 0)
        test_layout.addWidget(self.lbl_true_class, 1, 1)

        self.lbl_rpm = QLabel(f"{RPM} об/мин")
        self.lbl_rpm.setFont(QFont("Consolas", 10, QFont.Bold))
        test_layout.addWidget(QLabel("Скорость вращения:"), 2, 0)
        test_layout.addWidget(self.lbl_rpm, 2, 1)

        self.lbl_load = QLabel("0 л.с. (без нагрузки)")
        self.lbl_load.setFont(QFont("Consolas", 10, QFont.Bold))
        test_layout.addWidget(QLabel("Нагрузка на вал:"), 3, 0)
        test_layout.addWidget(self.lbl_load, 3, 1)

        self.lbl_fr = QLabel(f"{self.fr:.1f} Гц")
        self.lbl_fr.setFont(QFont("Consolas", 10, QFont.Bold))
        test_layout.addWidget(QLabel("Частота вращения (Fr):"), 4, 0)
        test_layout.addWidget(self.lbl_fr, 4, 1)

        right.addWidget(test_group)

        # --- Характерные частоты ---
        freq_group = QGroupBox("Характерные частоты дефектов")
        freq_group.setFont(QFont("Segoe UI", 10, QFont.Bold))
        freq_layout = QGridLayout(freq_group)
        freq_layout.setSpacing(3)

        self.lbl_bpfo = QLabel(f"{self.bpfo:.1f} Гц")
        self.lbl_bpfo.setFont(QFont("Consolas", 10, QFont.Bold))
        freq_layout.addWidget(QLabel("Наружное кольцо (BPFO):"), 0, 0)
        freq_layout.addWidget(self.lbl_bpfo, 0, 1)

        self.lbl_bpfi = QLabel(f"{self.bpfi:.1f} Гц")
        self.lbl_bpfi.setFont(QFont("Consolas", 10, QFont.Bold))
        freq_layout.addWidget(QLabel("Внутреннее кольцо (BPFI):"), 1, 0)
        freq_layout.addWidget(self.lbl_bpfi, 1, 1)

        self.lbl_bsf = QLabel(f"{self.bsf:.1f} Гц")
        self.lbl_bsf.setFont(QFont("Consolas", 10, QFont.Bold))
        freq_layout.addWidget(QLabel("Шарики (BSF):"), 2, 0)
        freq_layout.addWidget(self.lbl_bsf, 2, 1)

        self.lbl_ftf = QLabel(f"{self.ftf:.1f} Гц")
        self.lbl_ftf.setFont(QFont("Consolas", 10, QFont.Bold))
        freq_layout.addWidget(QLabel("Сепаратор (FTF):"), 3, 0)
        freq_layout.addWidget(self.lbl_ftf, 3, 1)

        right.addWidget(freq_group)

        # --- Параметры вибросигнала с индикаторами ---
        sig_met_group = QGroupBox("Параметры вибросигнала")
        sig_met_group.setFont(QFont("Segoe UI", 10, QFont.Bold))
        sig_met_layout = QGridLayout(sig_met_group)
        sig_met_layout.setSpacing(4)

        self.lbl_rms_val = QLabel("—")
        self.lbl_rms_val.setFont(QFont("Consolas", 10, QFont.Bold))
        sig_met_layout.addWidget(QLabel("RMS:"), 0, 0)
        sig_met_layout.addWidget(self.lbl_rms_val, 0, 1)
        self.rms_bar = QProgressBar()
        self.rms_bar.setMaximum(150)
        sig_met_layout.addWidget(self.rms_bar, 0, 2)
        self.lbl_rms_status = QLabel("")
        sig_met_layout.addWidget(self.lbl_rms_status, 0, 3)

        self.lbl_kurt_val = QLabel("—")
        self.lbl_kurt_val.setFont(QFont("Consolas", 10, QFont.Bold))
        sig_met_layout.addWidget(QLabel("Куртозис:"), 1, 0)
        sig_met_layout.addWidget(self.lbl_kurt_val, 1, 1)
        self.kurt_bar = QProgressBar()
        self.kurt_bar.setMaximum(150)
        sig_met_layout.addWidget(self.kurt_bar, 1, 2)
        self.lbl_kurt_status = QLabel("")
        sig_met_layout.addWidget(self.lbl_kurt_status, 1, 3)

        self.lbl_peak_val = QLabel("—")
        self.lbl_peak_val.setFont(QFont("Consolas", 10, QFont.Bold))
        sig_met_layout.addWidget(QLabel("Пик-фактор:"), 2, 0)
        sig_met_layout.addWidget(self.lbl_peak_val, 2, 1)
        self.peak_bar = QProgressBar()
        self.peak_bar.setMaximum(150)
        sig_met_layout.addWidget(self.peak_bar, 2, 2)
        self.lbl_peak_status = QLabel("")
        sig_met_layout.addWidget(self.lbl_peak_status, 2, 3)

        right.addWidget(sig_met_group)

        # --- Управление ---
        ctrl_group = QGroupBox("Управление")
        ctrl_group.setFont(QFont("Segoe UI", 10, QFont.Bold))
        ctrl_layout = QVBoxLayout(ctrl_group)
        ctrl_layout.addWidget(QLabel("Образец:"))
        self.sample_spin = QComboBox()
        self.sample_spin.addItems([str(i) for i in range(len(X_test_spec))])
        self.sample_spin.setCurrentText(str(self.current_sample))
        self.sample_spin.setMinimumHeight(26)
        self.sample_spin.currentTextChanged.connect(self.change_sample)
        ctrl_layout.addWidget(self.sample_spin)

        ctrl_layout.addWidget(QLabel("Модель:"))
        self.selector = QComboBox()
        # ТОЛЬКО 4 СВЕРТОЧНЫЕ МОДЕЛИ
        model_names = ["CNN", "CNN-LSTM", "CNN-BiGRU", "CNN-Stacked LSTM"]
        # Проверяем, какие модели загружены
        available = [m for m in model_names if m in self.loader.models]
        if available:
            self.selector.addItems(available)
        else:
            self.selector.addItems(model_names)
        self.selector.setMinimumHeight(26)
        self.selector.currentTextChanged.connect(self.run_diagnosis)
        ctrl_layout.addWidget(self.selector)

        self.metrics_label = QLabel("")
        self.metrics_label.setFont(QFont("Consolas", 8))
        self.metrics_label.setStyleSheet("color: #888; padding-top: 2px;")
        ctrl_layout.addWidget(self.metrics_label)

        right.addWidget(ctrl_group)

        # --- Обозначения ---
        help_group = QGroupBox("Обозначения")
        help_group.setFont(QFont("Segoe UI", 9, QFont.Bold))
        help_layout = QVBoxLayout(help_group)
        help_text = QLabel(
            "🟢 Зелёный — в норме\n"
            "🟡 Жёлтый — пограничное\n"
            "🔴 Красный — дефект\n\n"
            "Классы: Ball (шарик), IR (внутр. кольцо),\n"
            "OR (наруж. кольцо), Normal (норма)\n"
            "007/014/021 — диаметр дефекта в дюймах\n"
            "(0.18 / 0.36 / 0.53 мм)"
        )
        help_text.setFont(QFont("Segoe UI", 7))
        help_text.setStyleSheet("color: #555;")
        help_layout.addWidget(help_text)
        right.addWidget(help_group)

        main.addLayout(right, 2)

        self.update_plots()
        self.run_diagnosis()

    def change_sample(self, text):
        self.current_sample = int(text)
        self.spec_2d = X_test_spec[self.current_sample]
        self.true_label = int(y_test[self.current_sample])
        self.true_name = CLASS_NAMES.get(self.true_label, "?")
        self.raw_idx = test_to_raw[self.current_sample]
        self.raw_signal = X_raw[self.raw_idx].flatten()

        self.lbl_sample_info.setText(f"Образец №{self.current_sample}")
        self.lbl_true_class.setText(self.true_name)

        self.update_plots()
        self.run_diagnosis()

    def fft(self, x):
        x = np.asarray(x).flatten()
        x = x - np.mean(x)
        fft_vals = np.fft.fft(x)
        mag = np.abs(fft_vals)[:len(x) // 2]
        freqs = np.fft.fftfreq(len(x), d=1 / FS)[:len(x) // 2]
        return freqs, mag

    def update_plots(self):
        self.ax_time.clear()
        self.ax_fft.clear()
        sig = self.raw_signal
        t = np.arange(len(sig)) / FS

        rms_val = np.sqrt(np.mean(sig ** 2))

        # --- Временной график ---
        self.ax_time.axhline(y=NORMAL_RMS[1], color='orange', linestyle='--', alpha=0.5,
                             label=f'Верх нормы RMS ({NORMAL_RMS[1]:.3f})')
        self.ax_time.axhline(y=-NORMAL_RMS[1], color='orange', linestyle='--', alpha=0.5)
        self.ax_time.axhline(y=NORMAL_RMS[0], color='green', linestyle=':', alpha=0.4)
        self.ax_time.axhline(y=-NORMAL_RMS[0], color='green', linestyle=':', alpha=0.4)

        self.ax_time.plot(t, sig, linewidth=0.8, color='#2196F3')
        self.ax_time.set_title(f"Вибросигнал — {len(sig)} точек, ~{len(sig) / FS * 1000:.0f} мс", fontsize=12,
                               fontweight='bold')
        self.ax_time.set_xlabel("Время, с")
        self.ax_time.set_ylabel("Амплитуда")
        self.ax_time.set_ylim(-1.5, 1.5)
        self.ax_time.grid(True, alpha=0.3, linestyle='--')
        self.ax_time.legend(fontsize=6, loc='upper right')

        # --- Спектр ---
        f, m = self.fft(sig)
        mask = f <= 800
        self.ax_fft.plot(f[mask], m[mask], linewidth=0.8, color='#4CAF50')

        for freq, name, col in [
            (self.bpfo, "BPFO", "#e74c3c"),
            (self.bpfi, "BPFI", "#3498db"),
            (self.bsf, "BSF", "#f39c12"),
            (self.ftf, "FTF", "#9b59b6"),
        ]:
            self.ax_fft.axvspan(freq - 5, freq + 5, alpha=0.1, color=col)
            self.ax_fft.axvline(freq, color=col, linestyle='--', alpha=0.6, linewidth=1)
            self.ax_fft.text(freq, 480, name, color=col, fontsize=7, ha='center')

        self.ax_fft.set_title("Спектр Фурье (0–800 Гц) с зонами дефектов", fontsize=12, fontweight='bold')
        self.ax_fft.set_xlabel("Частота, Гц")
        self.ax_fft.set_ylabel("Амплитуда")
        self.ax_fft.set_ylim(0, 500)
        self.ax_fft.grid(True, alpha=0.3, linestyle='--')
        self.canvas_sig.draw_idle()

        # --- RMS vs Куртозис ---
        self.ax_rms.clear()
        for t_eng, t_ru in TYPE_NAMES_RU.items():
            mask = type_all == t_eng
            self.ax_rms.scatter(rms_all[mask], kurt_all[mask],
                                c=TYPE_COLORS[t_eng], label=t_ru, alpha=0.4, s=12, edgecolors='none')

        idx = self.raw_idx
        self.ax_rms.scatter(rms_all[idx], kurt_all[idx],
                            c='black', s=100, edgecolors='yellow', linewidth=2, zorder=10, label='Образец')

        rect = Rectangle((NORMAL_RMS[0], NORMAL_KURT[0]),
                         NORMAL_RMS[1] - NORMAL_RMS[0], NORMAL_KURT[1] - NORMAL_KURT[0],
                         linewidth=2, edgecolor='#27ae60', facecolor='#27ae60', alpha=0.1, label='Зона нормы')
        self.ax_rms.add_patch(rect)

        self.ax_rms.set_title("Пространство признаков (RMS и Куртозис)", fontsize=12, fontweight='bold')
        self.ax_rms.set_xlabel("RMS (среднеквадратичная амплитуда)")
        self.ax_rms.set_ylabel("Куртозис")
        self.ax_rms.set_xlim(0, 1.5)
        self.ax_rms.set_ylim(1, 40)
        self.ax_rms.legend(fontsize=7, loc='upper right')
        self.ax_rms.grid(True, alpha=0.3, linestyle='--')
        self.canvas_rms.draw_idle()

        # --- Индикаторы ---
        kurt_val = np.mean((sig - np.mean(sig)) ** 4) / (np.std(sig) ** 4 + 1e-10)
        peak_val = np.max(np.abs(sig)) / (rms_val + 1e-10)

        self.lbl_rms_val.setText(f"{rms_val:.3f}")
        self.lbl_kurt_val.setText(f"{kurt_val:.2f}")
        self.lbl_peak_val.setText(f"{peak_val:.2f}")

        # RMS
        color, status = self.status_color(rms_val, NORMAL_RMS[0], NORMAL_RMS[1])
        self.rms_bar.setValue(int(min(rms_val, 1.5) * 100))
        self.rms_bar.setStyleSheet(f"QProgressBar::chunk {{ background: {color}; }}")
        self.lbl_rms_status.setText(status)
        self.lbl_rms_status.setStyleSheet(f"color: {color}; font-weight: bold; font-size: 9px;")

        # Куртозис
        color, status = self.status_color(kurt_val, NORMAL_KURT[0], NORMAL_KURT[1])
        self.kurt_bar.setValue(int(min(kurt_val / 10 * 100, 150)))
        self.kurt_bar.setStyleSheet(f"QProgressBar::chunk {{ background: {color}; }}")
        self.lbl_kurt_status.setText(status)
        self.lbl_kurt_status.setStyleSheet(f"color: {color}; font-weight: bold; font-size: 9px;")

        # Пик-фактор
        color, status = self.status_color(peak_val, NORMAL_PEAK[0], NORMAL_PEAK[1])
        self.peak_bar.setValue(int(min(peak_val / 10 * 100, 150)))
        self.peak_bar.setStyleSheet(f"QProgressBar::chunk {{ background: {color}; }}")
        self.lbl_peak_status.setText(status)
        self.lbl_peak_status.setStyleSheet(f"color: {color}; font-weight: bold; font-size: 9px;")

        # --- Анализ ---
        faults_found = []
        for freq, name in [
            (self.bpfo, "наружного кольца (BPFO)"),
            (self.bpfi, "внутреннего кольца (BPFI)"),
            (self.bsf, "шарика (BSF)"),
        ]:
            idx_f = np.argmin(np.abs(f - freq))
            if m[idx_f] > np.mean(m) * 1.5:
                faults_found.append(f"  • Пик на частоте {name}: {freq:.1f} Гц (амплитуда {m[idx_f]:.1f})")

        if faults_found:
            analysis = "Обнаружены признаки дефекта:\n" + "\n".join(faults_found)
        else:
            analysis = "✓ Выраженных пиков на характерных частотах не обнаружено"

        if rms_val > NORMAL_RMS[1]:
            analysis += f"\n  • RMS ({rms_val:.3f}) выше нормы (>{NORMAL_RMS[1]})"
        if kurt_val > NORMAL_KURT[1]:
            analysis += f"\n  • Куртозис ({kurt_val:.1f}) выше нормы (>{NORMAL_KURT[1]}) — есть ударные импульсы"

        self.analysis_label.setText(analysis)
        if faults_found:
            self.analysis_label.setStyleSheet(
                "color: #c0392b; padding: 4px; background: #fdf2f2; border-radius: 4px; font-size: 9px;")
        else:
            self.analysis_label.setStyleSheet(
                "color: #27ae60; padding: 4px; background: #f0fff0; border-radius: 4px; font-size: 9px;")

    def run_diagnosis(self):
        model_name = self.selector.currentText()
        true_name = CLASS_NAMES.get(self.true_label, "?")

        # Получаем метрики из all_results.json
        json_map = {
            "CNN": "CNN",
            "CNN-LSTM": "CNN_LSTM",
            "CNN-BiGRU": "CNN_BiGRU",
            "CNN-Stacked LSTM": "CNN_StackedLSTM"
        }
        json_name = json_map.get(model_name, model_name)
        r = self.results.get(json_name, {})

        self.metrics_label.setText(
            f"Accuracy: {r.get('accuracy', 0):.4f}  "
            f"Precision: {r.get('precision', 0):.4f}  "
            f"Recall: {r.get('recall', 0):.4f}  "
            f"F1: {r.get('f1_score', 0):.4f}  "
            f"Loss: {r.get('loss', 0):.4f}"
        )

        cls, probs = self.loader.predict(model_name, self.spec_2d)

        if cls is None:
            self.status_label.setText("Ошибка модели")
            self.status_label.setStyleSheet(
                "color: #e74c3c; font-size: 18px; font-weight: bold; padding: 10px; background: #fff3cd; border-radius: 8px;")
            self.info_label.setText(f"Истинная метка: {true_name}")
            return

        pred_name = CLASS_NAMES.get(cls, "?")
        ok = cls == self.true_label
        conf = f"Уверенность: {probs[cls]:.1%}" if probs is not None else ""

        if ok:
            color, bg = "#155724", "#d4edda"
            icon = "✅"
        else:
            color, bg = "#721c24", "#f8d7da"
            icon = "❌"

        self.status_label.setText(f"{icon} {pred_name}")
        self.status_label.setStyleSheet(
            f"color: {color}; font-size: 18px; font-weight: bold; padding: 10px; background: {bg}; border-radius: 8px;")
        self.info_label.setText(f"Истинная: {true_name}  |  {conf}  |  Модель: {model_name}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    win = Dashboard()
    win.show()
    sys.exit(app.exec_())