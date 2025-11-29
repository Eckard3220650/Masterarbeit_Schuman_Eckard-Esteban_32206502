# 09c_threshold_tuning_rf_scaled.py
import os, json
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix, precision_recall_curve
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

#  CONFIG 
MODEL_DIR   = r"C:\Users\49176\Desktop\Studium\Masterarbeit_3\Models\RF\RF_Tuned"
DATA_DIR    = r"C:\Users\49176\Desktop\Studium\Masterarbeit_3\ModelData"
RES_THR_DIR = r"C:\Users\49176\Desktop\Studium\Masterarbeit_3\Results\RF\Thresholds"
FINAL_DIR   = r"C:\Users\49176\Desktop\Studium\Masterarbeit_3\Results\RF"
os.makedirs(RES_THR_DIR, exist_ok=True)
os.makedirs(FINAL_DIR, exist_ok=True)

#  BUSINESS PARAMETERS 
V_TP = 30   # Profit per True Positive (purchase)
C_FP = -5   # Cost per False Positive (wasted outreach)

#  DATASETS 
datasets = {
    "raw":     ("X_train_raw.csv", "y_train_raw.csv"),
    "smotenc": ("X_train_smotenc.csv", "y_train_smotenc.csv"),
    "under":   ("X_train_under.csv", "y_train_under.csv"),
}

#  TEST SET 
X_test_raw = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
y_test = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).values.ravel()

# Profit-based threshold selection 
def choose_threshold_profit(y_true, y_prob, V_TP=30, C_FP=-5, plot=False, variant="rf"):
    thresholds = np.linspace(0.01, 0.99, 100)
    profits = []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tp = ((y_true == 1) & (y_pred == 1)).sum()
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        profit = V_TP * tp + C_FP * fp
        profits.append(profit)

    best_idx = int(np.argmax(profits))
    best_threshold = float(thresholds[best_idx])

    if plot:
        plt.figure(figsize=(8, 5))
        plt.plot(thresholds, profits, label="Expected Profit (€)")
        plt.axvline(best_threshold, color='red', linestyle='--', label=f"Best Threshold = {best_threshold:.2f}")
        plt.title(f"Expected Profit Curve — {variant.upper()}")
        plt.xlabel("Threshold")
        plt.ylabel("Expected Profit (€)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(RES_THR_DIR, f"profit_curve_{variant}.png"))
        plt.close()

    return best_threshold

#  Metrics computation 
def compute_metrics(y_true, y_prob, thr):
    y_pred = (y_prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "threshold_used": float(thr),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }

#  Main Loop 
for name, (Xfile, yfile) in datasets.items():
    print(f"[INFO] Threshold tuning (validation) for RF — {name.upper()}")

    tuned_path = os.path.join(MODEL_DIR, f"rf_tuned_{name}.pkl")
    if not os.path.exists(tuned_path):
        raise FileNotFoundError(f"Tuned RF not found: {tuned_path}")

    model: RandomForestClassifier = joblib.load(tuned_path)

    X_train_raw = pd.read_csv(os.path.join(DATA_DIR, Xfile))
    y_train = pd.read_csv(os.path.join(DATA_DIR, yfile)).values.ravel()

    # Split into train/validation
    X_tr_raw, X_val_raw, y_tr, y_val = train_test_split(
        X_train_raw, y_train, test_size=0.2, stratify=y_train, random_state=42
    )

    #  SCALING 
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr_raw)
    X_val = scaler.transform(X_val_raw)
    X_train = scaler.transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    # Fit on train split
    model.fit(X_tr, y_tr)

    #  Profit-based threshold optimization 
    val_prob = model.predict_proba(X_val)[:, 1]
    t_star = choose_threshold_profit(y_val, val_prob, V_TP, C_FP, plot=True, variant=name)

    # Refit on full training set
    model.fit(X_train, y_train)

    #  Save model, scaler, threshold 
    out_dir = os.path.join(MODEL_DIR, f"final_{name}")
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(model, os.path.join(out_dir, "model_final.pkl"))
    joblib.dump(scaler, os.path.join(out_dir, "scaler.pkl"))
    with open(os.path.join(out_dir, "threshold_used.json"), "w", encoding="utf-8") as f:
        json.dump({"threshold_used": t_star}, f, indent=2)

    #  Evaluate on test set 
    test_prob = model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, test_prob, t_star)

    #  Save metrics 
    out_csv = os.path.join(FINAL_DIR, f"rf_metrics_{name}.csv")
    pd.DataFrame([{"dataset": name, **metrics}]).to_csv(out_csv, index=False)

    #  Save Precision-Recall sweep for inspection 
    p, r, thr = precision_recall_curve(y_val, val_prob)
    sweep_df = pd.DataFrame({"precision": p[:-1], "recall": r[:-1], "threshold": thr})
    sweep_df.to_csv(os.path.join(RES_THR_DIR, f"rf_val_pr_sweep_{name}.csv"), index=False)

print(" RF thresholds tuned using business profit. Models, scalers, and metrics saved.")
