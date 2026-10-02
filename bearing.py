import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.stats import kurtosis
from google.colab import files

# ==============================================================================
# 1. ЗАГРУЗКА ДАННЫХ
# ==============================================================================
print("📁 Выберите ваш .npz файл (CWRU dataset)")
uploaded = files.upload()
file_path = list(uploaded.keys())[0]
data = np.load(file_path, allow_pickle=True)

X = data['data']
y = data['labels']
y_str = np.array([str(label) for label in y])

# ==============================================================================
# 2. ПАРАМЕТРЫ (Настройка разрешения)
# ==============================================================================
FS = 12000  # Частота дискретизации
# Увеличиваем длину окна для FFT до 8192 или более для высокой четкости
N_FFT = 8192

# Теоретические частоты для справки (1797 RPM)
fault_freqs = {
    'Shaft (1x)': 29.95,
    'BSF (Ball)': 70.1,
    'BPFO (Outer)': 107.4,
    'BPFI (Inner)': 162.2
}

# ==============================================================================
# 3. ПОСТРОЕНИЕ УЛУЧШЕННОГО ГРАФИКА
# ==============================================================================
plt.figure(figsize=(16, 9))

# Цветовая схема: яркие дефекты и спокойная норма
colors = {
    'Normal': '#27ae60',  # Зеленый
    'Ball_014': '#f1c40f',  # Желтый
    'IR_014': '#e67e22',  # Оранжевый
    'OR_014': '#c0392b'  # Красный
}

target_classes = ['Normal', 'Ball_014', 'IR_014', 'OR_014']

for cls in target_classes:
    indices = np.where(y_str == cls)[0]
    if len(indices) > 0:
        # Берем достаточно длинный фрагмент сигнала для высокого разрешения
        sig_raw = X[indices[0]].flatten()

        # Если сигнал короче N_FFT, берем весь, если длиннее - отрезаем кусок
        n = min(len(sig_raw), N_FFT)
        sig = sig_raw[:n]
        sig = sig - np.mean(sig)  # Удаление постоянной составляющей

        # FFT расчет
        yf = fft(sig)
        xf = fftfreq(n, 1 / FS)

        # Амплитудный спектр (нормализованный на длину)
        amplitude = 2.0 / n * np.abs(yf[:n // 2])
        freqs = xf[:n // 2]

        # Маска для отображения только интересующей нас области (0 - 300 Гц)
        mask = (freqs >= 0) & (freqs <= 300)

        plt.plot(freqs[mask], amplitude[mask], label=f"{cls}",
                 color=colors.get(cls, 'gray'), linewidth=1.2, alpha=0.9)

# Добавляем вертикальные линии частот дефектов
for name, f_val in fault_freqs.items():
    plt.axvline(x=f_val, color='black', linestyle='--', alpha=0.4, linewidth=1)
    plt.text(f_val + 0.5, plt.ylim()[1] * 0.85, name, rotation=90,
             verticalalignment='center', fontsize=10, fontweight='bold')

# Настройка сетки и оформления
plt.title("Прецизионный спектральный анализ подшипников (CWRU)\nВысокое частотное разрешение (N_FFT=8192)", fontsize=16)
plt.xlabel("Частота (Гц)", fontsize=13)
plt.ylabel("Амплитуда (Ускорение g)", fontsize=13)
plt.legend(title="Состояние подшипника", fontsize=11, loc='upper right')
plt.grid(True, which='major', color='#CCCCCC', linestyle='--')
plt.grid(True, which='minor', color='#EEEEEE', linestyle=':', alpha=0.5)
plt.minorticks_on()
plt.xlim(0, 300)

plt.tight_layout()
plt.show()

# ==============================================================================
# 4. ВЫВОД ДИАГНОСТИЧЕСКИХ МЕТРИК
# ==============================================================================
print("\n" + "=" * 60)
print(f"{'Тип сигнала':<15} | {'RMS (Энергия)':<15} | {'Kurtosis (Пики)':<15}")
print("-" * 60)

for cls in target_classes:
    indices = np.where(y_str == cls)[0]
    if len(indices) > 0:
        sig = X[indices[0]].flatten()
        print(f"{cls:<15} | {np.sqrt(np.mean(sig ** 2)):<15.4f} | {kurtosis(sig):<15.4f}")
print("=" * 60)