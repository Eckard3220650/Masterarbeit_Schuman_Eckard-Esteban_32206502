# 04_train_ann.py 
import pandas as pd
import os
import joblib
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

#  CONFIG 
DATA_DIR = r"C:\Users\49176\Desktop\Studium\Masterarbeit_3\ModelData"
MODEL_DIR = r"C:\Users\49176\Desktop\Studium\Masterarbeit_3\Models\ANN\ANN_default"
RESULTS_DIR = r"C:\Users\49176\Desktop\Studium\Masterarbeit_3\Results\ANN"
PREPROCESSOR_PATH = os.path.join(DATA_DIR, "preprocessor.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

datasets = {
    "raw": ("X_train_raw.csv", "y_train_raw.csv"),
    "under": ("X_train_under.csv", "y_train_under.csv"),
    "smotenc": ("X_train_smotenc.csv", "y_train_smotenc.csv"),
}

#  Load Test Set 
X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
y_test = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).values.ravel()

#  Feature Names 
try:
    pre = joblib.load(PREPROCESSOR_PATH)
    feature_names = pre.get_feature_names_out()
except Exception:
    feature_names = [f"f{i}" for i in range(X_test.shape[1])]

#  Business Values 
V_TP = 30
C_FP = -5
BASELINE_THRESHOLD = 0.5

#  Model Builder 
def build_ann(input_dim):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=input_dim))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

#  Training Loop 
for name, (X_file, y_file) in datasets.items():
    print(f"[INFO] Training baseline ANN on {name.upper()}")

    # Load Data
    X_train = pd.read_csv(os.path.join(DATA_DIR, X_file))
    y_train = pd.read_csv(os.path.join(DATA_DIR, y_file)).values.ravel()

    # Build & Train Model
    model = build_ann(X_train.shape[1])
    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        callbacks=[es],
        verbose=0
    )

    # Predict
    y_prob = model.predict(X_test).ravel()
    y_pred = (y_prob >= BASELINE_THRESHOLD).astype(int)

    # Metrics
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    metrics = {
        "dataset": name,
        "threshold_used": BASELINE_THRESHOLD,
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "pr_auc": average_precision_score(y_test, y_prob),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "profit": V_TP * tp + C_FP * fp
    }

    # Save Metrics
    out_csv = os.path.join(RESULTS_DIR, f"ann_metrics_baseline_{name}.csv")
    pd.DataFrame([metrics]).to_csv(out_csv, index=False)

    # Save Model and Features
    model.save(os.path.join(MODEL_DIR, f"ann_baseline_{name}.h5"))
    pd.Series(feature_names).to_csv(
        os.path.join(MODEL_DIR, f"ann_baseline_{name}_features.csv"),
        index=False
    )

    print(f" Saved ANN model and metrics for {name}")

print("Baseline ANN models trained and saved.")
