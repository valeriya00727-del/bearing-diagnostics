import numpy as np
import joblib
import os
import time

import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("=" * 60)
print("CNN + LSTM TRAINING")
print("=" * 60)

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
LSTM_DIR = os.path.join(BASE_DIR, "CNN_LSTM")

os.makedirs(LSTM_DIR, exist_ok=True)


X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
X_test  = np.load(os.path.join(DATA_DIR, "X_test.npy"))
y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
y_test  = np.load(os.path.join(DATA_DIR, "y_test.npy"))

le = joblib.load(os.path.join(DATA_DIR, "label_encoder.pkl"))
num_classes = len(le.classes_)

print("Train:", X_train.shape)
print("Test :", X_test.shape)


#mean = X_train.mean()
#std = X_train.std()

#X_train = (X_train - mean) / (std + 1e-8)
#X_test  = (X_test - mean) / (std + 1e-8)


X_train = X_train.reshape(X_train.shape[0], 32, 32)
X_test  = X_test.reshape(X_test.shape[0], 32, 32)

# ─────────────────────────────
# 🧠 MODEL
# ─────────────────────────────
model = keras.Sequential([
    layers.Input(shape=(32, 32)),

    layers.Conv1D(64, 3, activation='relu', padding='same'),
    layers.MaxPooling1D(),

    layers.Conv1D(128, 3, activation='relu', padding='same'),

    layers.LSTM(64),

    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),

    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ─────────────────────────────
# 🚀 TRAIN
# ─────────────────────────────
t0 = time.time()

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=20,
    batch_size=32,
    verbose=1
)

t1 = time.time()

# ─────────────────────────────
# 📊 PREDICTION
# ─────────────────────────────
y_pred = np.argmax(model.predict(X_test), axis=1)

acc = accuracy_score(y_test, y_pred)

print("\n🏆 CNN+LSTM Accuracy:", acc)
print("⏱ Time:", t1 - t0)

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

# ─────────────────────────────
# 📊 1. CONFUSION MATRIX
# ─────────────────────────────
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples')

plt.title("CNN + LSTM матрица ошибок")
plt.xlabel("Предсказано")
plt.ylabel("Истинное")

plt.tight_layout()
plt.savefig(os.path.join(LSTM_DIR, "cnn_lstm_confusion_matrix.png"))
plt.show()

# ─────────────────────────────
# 📊 2. METRICS
# ─────────────────────────────
plt.figure(figsize=(6,4))

plt.bar(
    ["Accuracy", "Loss"],
    [acc, 1-acc]
)

plt.title("CNN + LSTM Performance")
plt.ylim(0, 1)
plt.grid(axis="y")

plt.tight_layout()
plt.savefig(os.path.join(LSTM_DIR, "cnn_lstm_metrics.png"))
plt.show()

# ─────────────────────────────
# 📊 3. LEARNING CURVES
# ─────────────────────────────
plt.figure(figsize=(12,5))

# Левый график — Точность (Accuracy)
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Обучающая')
plt.plot(history.history['val_accuracy'], label='Валидационная')
plt.title("Точность")
plt.legend()
plt.grid()

# Правый график — Ошибка (Loss)
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Обучающая')
plt.plot(history.history['val_loss'], label='Валидационная')
plt.title("Ошибка (функция потерь)")
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig(os.path.join(LSTM_DIR, "cnn_lstm_learning_curves.png"))
plt.show()

# ─────────────────────────────
# 💾 SAVE MODEL
# ─────────────────────────────
model.save(os.path.join(LSTM_DIR, "cnn_lstm_model.keras"))

print("\n✅ CNN+LSTM SAVED")