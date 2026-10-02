# train_cnn_stacked_lstm.py

import os
import json
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix, classification_report

print("=" * 60)
print("CNN + STACKED LSTM")
print("=" * 60)

# ==========================================================
# PATHS
# ==========================================================

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "CNN_StackedLSTM")

os.makedirs(MODEL_DIR, exist_ok=True)

# ==========================================================
# LOAD DATA
# ==========================================================

X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))

y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))

le = joblib.load(os.path.join(DATA_DIR, "label_encoder.pkl"))

num_classes = len(le.classes_)

print("Train:", X_train.shape)
print("Test :", X_test.shape)

# ==========================================================
# RESHAPE
# ==========================================================

X_train = X_train.reshape(-1, 32, 32)
X_test = X_test.reshape(-1, 32, 32)

y_train_cat = keras.utils.to_categorical(y_train, num_classes)
y_test_cat = keras.utils.to_categorical(y_test, num_classes)

print("Reshaped:", X_train.shape)

# ==========================================================
# MODEL
# ==========================================================

model = keras.Sequential([
    layers.Input(shape=(32, 32)),

    # CNN
    layers.Conv1D(
        filters=64,
        kernel_size=3,
        activation="relu",
        padding="same"
    ),
    layers.BatchNormalization(),
    layers.MaxPooling1D(pool_size=2),

    layers.Conv1D(
        filters=128,
        kernel_size=3,
        activation="relu",
        padding="same"
    ),
    layers.BatchNormalization(),
    layers.MaxPooling1D(pool_size=2),

    layers.Conv1D(
        filters=256,
        kernel_size=3,
        activation="relu",
        padding="same"
    ),
    layers.BatchNormalization(),

    # Stacked LSTM
    layers.LSTM(
        128,
        return_sequences=True,
        dropout=0.3
    ),

    layers.LSTM(
        64,
        return_sequences=False,
        dropout=0.3
    ),

    # Dense
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),

    layers.Dense(64, activation="relu"),
    layers.Dropout(0.3),

    layers.Dense(num_classes, activation="softmax")
])

print("\nMODEL SUMMARY")
print("=" * 60)
model.summary()

# ==========================================================
# COMPILE
# ==========================================================

model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ==========================================================
# CALLBACKS
# ==========================================================

callbacks = [

    keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),

    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        verbose=1
    ),

    keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(
            MODEL_DIR,
            "best_model.keras"
        ),
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1
    )
]

# ==========================================================
# TRAIN
# ==========================================================

history = model.fit(
    X_train,
    y_train_cat,
    validation_data=(X_test, y_test_cat),
    epochs=50,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

# ==========================================================
# SAVE FINAL MODEL
# ==========================================================

final_model_path = os.path.join(
    MODEL_DIR,
    "cnn_stacked_lstm_model.keras"
)

model.save(final_model_path)

print(f"\n✅ Model saved:")
print(final_model_path)

# ==========================================================
# LEARNING CURVES
# ==========================================================

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("Accuracy")
plt.legend(["Train", "Validation"])
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Loss")
plt.legend(["Train", "Validation"])
plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        MODEL_DIR,
        "learning_curves.png"
    ),
    dpi=150
)

plt.close()

# ==========================================================
# EVALUATION
# ==========================================================

y_pred = np.argmax(
    model.predict(X_test),
    axis=1
)

y_true = np.argmax(
    y_test_cat,
    axis=1
)

# ==========================================================
# CONFUSION MATRIX
# ==========================================================

cm = confusion_matrix(
    y_true,
    y_pred
)

plt.figure(figsize=(10, 8))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Purples",
    xticklabels=le.classes_,
    yticklabels=le.classes_
)

plt.title("CNN + Stacked LSTM")
plt.xlabel("Predicted")
plt.ylabel("True")

plt.tight_layout()

plt.savefig(
    os.path.join(
        MODEL_DIR,
        "confusion_matrix.png"
    ),
    dpi=150
)

plt.close()

# ==========================================================
# CLASSIFICATION REPORT
# ==========================================================

print("\n")
print("=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_true,
        y_pred,
        target_names=le.classes_
    )
)

# ==========================================================
# FINAL METRICS
# ==========================================================

loss, acc = model.evaluate(
    X_test,
    y_test_cat,
    verbose=0
)

print(f"\nAccuracy : {acc:.4f}")
print(f"Loss     : {loss:.4f}")

metrics = {
    "model": "CNN_StackedLSTM",
    "accuracy": float(acc),
    "loss": float(loss),
    "parameters": int(model.count_params())
}

with open(
    os.path.join(MODEL_DIR, "metrics.json"),
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        metrics,
        f,
        indent=4
    )

print("✅ metrics.json saved")

# ==========================================================
# ONNX EXPORT
# ==========================================================

print("\n")
print("=" * 60)
print("ONNX EXPORT")
print("=" * 60)

try:
    import tensorflow as tf
    import tf2onnx

    best_model_path = os.path.join(
        MODEL_DIR,
        "best_model.keras"
    )

    best_model = tf.keras.models.load_model(
        best_model_path
    )

    input_signature = (
        tf.TensorSpec(
            shape=(None, 32, 32),
            dtype=tf.float32,
            name="input"
        ),
    )

    onnx_model, _ = tf2onnx.convert.from_keras(
        best_model,
        input_signature=input_signature,
        opset=13
    )

    onnx_path = os.path.join(
        MODEL_DIR,
        "cnn_stacked_lstm.onnx"
    )

    with open(
        onnx_path,
        "wb"
    ) as f:
        f.write(
            onnx_model.SerializeToString()
        )

    print(f"✅ ONNX saved:")
    print(onnx_path)

except ImportError:
    print(
        "❌ Install:\n"
        "pip install tf2onnx onnx onnxruntime"
    )

except Exception as e:
    print(f"❌ ONNX conversion error:\n{e}")

print("\nDONE")