# train_cnn_bigru.py (исправленная версия)
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from tensorflow import keras
from tensorflow.keras import layers

print("=" * 60)
print("CNN-BiGRU TRAINING (Bidirectional GRU) - FIXED")
print("=" * 60)

# ─────────────────────────────
# 📂 PATHS
# ─────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "CNN_BiGRU")

os.makedirs(MODEL_DIR, exist_ok=True)

# ─────────────────────────────
# 📦 LOAD DATA
# ─────────────────────────────
X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))

le = joblib.load(os.path.join(DATA_DIR, "label_encoder.pkl"))
num_classes = len(le.classes_)

print("Train:", X_train.shape)
print("Test :", X_test.shape)

# ─────────────────────────────
# 🧠 RESHAPE для CNN-BiGRU
# ─────────────────────────────
# Спектрограмма 32×32 → последовательность из 32 шагов
# Каждый шаг — вектор из 32 частотных компонент
X_train = X_train.reshape(-1, 32, 32)  # (samples, time_steps=32, features=32)
X_test = X_test.reshape(-1, 32, 32)

y_train_cat = keras.utils.to_categorical(y_train, num_classes)
y_test_cat = keras.utils.to_categorical(y_test, num_classes)

print("Reshaped for BiGRU:", X_train.shape)  # (3680, 32, 32)

# ─────────────────────────────
# 🧠 МОДЕЛЬ CNN-BiGRU (ИСПРАВЛЕННАЯ)
# ─────────────────────────────
model = keras.Sequential([
    # Входной слой: последовательность из 32 временных шагов
    layers.Input(shape=(32, 32)),

    # ========== СВЁРТОЧНАЯ ЧАСТЬ ==========
    # Conv1D сохраняет временную размерность
    layers.Conv1D(64, kernel_size=3, activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling1D(pool_size=2),  # 32 → 16

    layers.Conv1D(128, kernel_size=3, activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling1D(pool_size=2),  # 16 → 8

    # Conv1D без пулинга, только извлечение признаков
    layers.Conv1D(256, kernel_size=3, activation='relu', padding='same'),
    layers.BatchNormalization(),
    # ❌ Убираем GlobalMaxPooling1D! Он убивает временную размерность

    # ========== РЕКУРРЕНТНАЯ ЧАСТЬ ==========
    # BiGRU ожидает (batch, time_steps, features)
    # После Conv1D у нас размерность: (None, 8, 256) — это 3D!
    layers.Bidirectional(layers.GRU(128, return_sequences=False, dropout=0.3)),

    # ========== ПОЛНОСВЯЗНАЯ ЧАСТЬ ==========
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax')
])

# Вывод архитектуры
print("\n" + "=" * 60)
print("MODEL ARCHITECTURE")
print("=" * 60)
model.summary()

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ─────────────────────────────
# Callbacks
# ─────────────────────────────
callbacks = [
    keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True, verbose=1),
    keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, verbose=1)
]

# ─────────────────────────────
# 🚀 TRAIN
# ─────────────────────────────
history = model.fit(
    X_train, y_train_cat,
    validation_data=(X_test, y_test_cat),
    epochs=50,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

# ─────────────────────────────
# 💾 SAVE MODEL
# ─────────────────────────────
model.save(os.path.join(MODEL_DIR, "cnn_bigru_model.keras"))
print("\n✅ Model saved")

# ─────────────────────────────
# 📊 1. LEARNING CURVES
# ─────────────────────────────
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Обучающая', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Валидационная', linewidth=2)
plt.title("Точность", fontsize=14)
plt.xlabel("Эпоха")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Обучающая', linewidth=2)
plt.plot(history.history['val_loss'], label='Валидационная', linewidth=2)
plt.title("Функция потерь", fontsize=14)
plt.xlabel("Эпоха")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "cnn_bigru_learning_curves.png"), dpi=150)
plt.show()

# ─────────────────────────────
# 📊 2. CONFUSION MATRIX
# ─────────────────────────────
y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test_cat, axis=1)

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='RdBu',
            xticklabels=le.classes_,
            yticklabels=le.classes_,
            annot_kws={'size': 10})
plt.title("CNN-BiGRU Confusion Matrix", fontsize=14)
plt.xlabel("Predicted", fontsize=12)
plt.ylabel("True", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "cnn_bigru_confusion_matrix.png"), dpi=150)
plt.show()

# ─────────────────────────────
# 📊 3. CLASSIFICATION REPORT
# ─────────────────────────────
print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)
print(classification_report(y_true, y_pred, target_names=le.classes_))

# ─────────────────────────────
# 📌 FINAL METRICS
# ─────────────────────────────
loss, acc = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"\n🏆 CNN-BiGRU Accuracy: {acc:.4f}")
print(f"📉 CNN-BiGRU Loss: {loss:.4f}")

# Сохранение метрик в JSON
import json

metrics = {
    "model": "CNN-BiGRU",
    "accuracy": float(acc),
    "loss": float(loss),
    "parameters": model.count_params()
}
with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)