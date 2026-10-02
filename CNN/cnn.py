import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from tensorflow import keras
from tensorflow.keras import layers

print("=" * 60)
print("CNN TRAINING (FULL VERSION WITH PLOTS)")
print("=" * 60)

# ─────────────────────────────
# 📂 PATHS FIX
# ─────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CNN_DIR = os.path.join(BASE_DIR, "CNN")

os.makedirs(CNN_DIR, exist_ok=True)

# ─────────────────────────────
# 📦 LOAD DATA
# ─────────────────────────────
X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
X_test  = np.load(os.path.join(DATA_DIR, "X_test.npy"))
y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
y_test  = np.load(os.path.join(DATA_DIR, "y_test.npy"))

le = joblib.load(os.path.join(DATA_DIR, "label_encoder.pkl"))
num_classes = len(le.classes_)

print("Train:", X_train.shape)
print("Test :", X_test.shape)

# ─────────────────────────────
# 🧠 RESHAPE FOR CNN
# ─────────────────────────────
X_train = X_train.reshape(-1, 32, 32, 1)
X_test  = X_test.reshape(-1, 32, 32, 1)

y_train_cat = keras.utils.to_categorical(y_train, num_classes)
y_test_cat  = keras.utils.to_categorical(y_test, num_classes)

# ─────────────────────────────
# 🧠 MODEL
# ─────────────────────────────
model = keras.Sequential([
    layers.Input(shape=(32, 32, 1)),

    layers.Conv2D(32, 3, activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    layers.Conv2D(128, 3, activation="relu", padding="same"),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ─────────────────────────────
# 🚀 TRAIN
# ─────────────────────────────
history = model.fit(
    X_train, y_train_cat,
    validation_data=(X_test, y_test_cat),
    epochs=20,
    batch_size=32,
    verbose=1
)

# ─────────────────────────────
# 💾 SAVE MODEL
# ─────────────────────────────
model.save(os.path.join(CNN_DIR, "cnn_model.keras"))
print("\n✅ Model saved")

# ─────────────────────────────
# 📊 1. LEARNING CURVES
# ─────────────────────────────
plt.figure(figsize=(12,5))

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
plt.savefig(os.path.join(CNN_DIR, "cnn_learning_curves.png"))
plt.show()

# ─────────────────────────────
# 📊 2. CONFUSION MATRIX
# ─────────────────────────────
y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test_cat, axis=1)

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_,
            yticklabels=le.classes_)

plt.title("CNN Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")

plt.tight_layout()
plt.savefig(os.path.join(CNN_DIR, "cnn_confusion_matrix.png"))
plt.show()

# ─────────────────────────────
# 📌 FINAL ACCURACY
# ─────────────────────────────
loss, acc = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"\n🏆 CNN Accuracy: {acc:.4f}")