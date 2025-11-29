# 04_train_rf.py 
 
import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix
)

#  CONFIG 
DATA_DIR = r"C:\Users\49176\Desktop\Studium\Masterarbeit_3\ModelData"
MODEL_DIR = r"C:\Users\49176\Desktop\Studium\Masterarbeit_3\Models\RF\RF_default"
RESULTS_DIR = r"C:\Users\49176\Desktop\Studium\Masterarbeit_3\Results\RF"
PREPROCESSOR_PATH = os.path.join(DATA_DIR, "preprocessor.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Dataset variants
datasets = {
    "raw": ("X_train_raw.csv", "y_train_raw.csv"),
    "under": ("X_train_under.csv", "y_train_under.csv"),
    "smotenc": ("X_train_smotenc.csv", "y_train_smotenc.csv"),
}

# Test set
X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
y_test = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).values.ravel()

# Try to get feature names from preprocessor
try:
    pre = joblib.load(PREPROCESSOR_PATH)
    feature_names = pre.get_feature_names_out()
except Exception:
    feature_names = [f"f{i}" for i in range(X_test.shape[1])]

# Business cost assumptions (same as LogReg)
V_TP = 30
C_FP = -5
BASELINE_THRESHOLD = 0.5

#  TRAIN LOOP 
for name, (X_file, y_file) in datasets.items():
    print(f"[INFO] Training baseline RF on {name.upper()}")

    # Load training data
    X_train = pd.read_csv(os.path.join(DATA_DIR, X_file))
    y_train = pd.read_csv(os.path.join(DATA_DIR, y_file)).values.ravel()

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # Predict
    y_prob = model.predict_proba(X_test)[:, 1]
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
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "profit": V_TP * tp + C_FP * fp  # Add profit calculation
    }

    # Save individual metrics CSV
    out_path = os.path.join(RESULTS_DIR, f"rf_metrics_baseline_{name}.csv")
    pd.DataFrame([metrics]).to_csv(out_path, index=False)

    # Save model and feature names
    joblib.dump(model, os.path.join(MODEL_DIR, f"rf_baseline_{name}.pkl"))
    pd.Series(feature_names).to_csv(os.path.join(MODEL_DIR, f"rf_baseline_{name}_features.csv"), index=False)

    print(f" Saved RF model and metrics for {name}")

print("Baseline Random Forest models trained and saved.")
