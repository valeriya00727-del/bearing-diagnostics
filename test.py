import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

npz_data = np.load(os.path.join(DATA_DIR, "CWRU_48k_load_1_CNN_data.npz"))
X = npz_data["data"]
y = npz_data["labels"]

le = joblib.load(os.path.join(DATA_DIR, "label_encoder.pkl"))
class_names = le.classes_

# Перевод дюймов в миллиметры
def inches_to_mm(inches):
    return round(inches * 25.4, 2)

def get_size_in_mm(label):
    parts = label.split('_')
    if len(parts) == 2 and parts[1].isdigit():
        inches = int(parts[1]) / 1000
        return f"{inches_to_mm(inches)} мм"
    return ""

# Русские названия для заголовков
ru_names = {}
for label in class_names:
    if label == "Normal":
        ru_names[label] = "Норма"
    elif label.startswith("Ball"):
        size_mm = get_size_in_mm(label)
        ru_names[label] = f"Дефект тела качения (шарика)\n{size_mm}"
    elif label.startswith("IR"):
        size_mm = get_size_in_mm(label)
        ru_names[label] = f"Дефект внутреннего кольца\n{size_mm}"
    elif label.startswith("OR"):
        size_mm = get_size_in_mm(label)
        ru_names[label] = f"Дефект наружного кольца\n{size_mm}"
    else:
        ru_names[label] = label

fig, axes = plt.subplots(2, 5, figsize=(20, 8))
fig.suptitle("Спектрограммы вибрационных сигналов подшипников (CWRU)", fontsize=14, fontweight='bold')

for idx, label in enumerate(class_names):
    sample_idx = np.where(y == label)[0][0]
    S = X[sample_idx]

    S = np.log1p(np.abs(S))
    S = (S - S.min()) / (S.max() - S.min() + 1e-8)

    row, col = idx // 5, idx % 5

    im = axes[row, col].imshow(S, aspect='auto', cmap='turbo', origin='lower')
    axes[row, col].set_title(ru_names.get(label, label), fontsize=9, fontweight='bold')
    axes[row, col].set_xlabel("Время (усл. ед.)", fontsize=8)
    axes[row, col].set_ylabel("Частота (усл. ед.)", fontsize=8)

plt.tight_layout()
plt.savefig("spectrograms_clean.png", dpi=300, bbox_inches='tight')
plt.show()