import os

os.makedirs("data", exist_ok=True)
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ─────────────────────────────
# LOAD RAW DATA
# ─────────────────────────────
data = np.load("CWRU_48k_load_1_CNN_data.npz")

X = data["data"].astype(np.float32)
y_raw = data["labels"]

print("X shape:", X.shape)

# ─────────────────────────────
# LABEL ENCODING
# ─────────────────────────────
le = LabelEncoder()
y = le.fit_transform(y_raw)

# ─────────────────────────────
# SPLIT (ОДИН РАЗ!)
# ─────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ─────────────────────────────
# NORMALIZATION (GLOBAL)
# ─────────────────────────────
mean = X_train.mean()
std = X_train.std()

X_train = (X_train - mean) / (std + 1e-8)
X_test = (X_test - mean) / (std + 1e-8)

# ─────────────────────────────
# SAVE EVERYTHING
# ─────────────────────────────
np.save("data/X_train.npy", X_train)
np.save("data/X_test.npy", X_test)
np.save("data/y_train.npy", y_train)
np.save("data/y_test.npy", y_test)

np.save("data/mean.npy", mean)
np.save("data/std.npy", std)

joblib.dump(le, "data/label_encoder.pkl")

print("\n✅ DATA READY (CLEAN PIPELINE)")