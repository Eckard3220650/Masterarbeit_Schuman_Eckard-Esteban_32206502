# 04_train_logreg.py 
 
import pandas as pd
import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix
)

#  CONFIG 
DATA_DIR = r"C:\Users\49176\Desktop\Studium\Masterarbeit_3\ModelData"
MODEL_DIR = r"C:\Users\49176\Desktop\Studium\Masterarbeit_3\Models\LogReg"
RESULTS_DIR = r"C:\Users\49176\Desktop\Studium\Masterarbeit_3\Results\LogReg"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Dataset variants
datasets = {
    "raw": ("X_train_raw.csv", "y_train_raw.csv"),
    "under": ("X_train_under.csv", "y_train_under.csv"),
    "smotenc": ("X_train_smotenc.csv", "y_train_smotenc.csv"),
}

# Shared test set
X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
y_test = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).values.ravel()

# Business cost settings
V_TP = 30
C_FP = -5
BASELINE_THRESHOLD = 0.5

#  TRAIN LOOP 
for variant, (X_file, y_file) in datasets.items():
    print(f"[INFO] Training Logistic Regression on {variant.upper()}...")

    X_train = pd.read_csv(os.path.join(DATA_DIR, X_file))
    y_train = pd.read_csv(os.path.join(DATA_DIR, y_file)).values.ravel()

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= BASELINE_THRESHOLD).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    metrics = {
        "dataset": variant,
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
        "profit": V_TP * tp + C_FP * fp
    }

    # Save individual metrics CSV
    out_file = os.path.join(RESULTS_DIR, f"logreg_metrics_baseline_{variant}.csv")
    pd.DataFrame([metrics]).to_csv(out_file, index=False)

    # Save model
    model_path = os.path.join(MODEL_DIR, f"logreg_baseline_{variant}.pkl")
    joblib.dump(model, model_path)

    print(f"Metrics saved to: {out_file}")
    print(f"Model saved to: {model_path}")
