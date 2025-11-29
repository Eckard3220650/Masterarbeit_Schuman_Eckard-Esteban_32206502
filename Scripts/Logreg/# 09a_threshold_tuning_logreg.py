#09a_threshold_tuning_logreg.py
import os, json
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    precision_recall_curve, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

#  CONFIG 
MODEL_DIR   = r"C:\Users\49176\Desktop\Studium\Masterarbeit_3\Models\LogReg\LogReg_Tuned"
DATA_DIR    = r"C:\Users\49176\Desktop\Studium\Masterarbeit_3\ModelData"
RESULTS_DIR = r"C:\Users\49176\Desktop\Studium\Masterarbeit_3\Results\LogReg\Thresholds"
FINAL_DIR   = r"C:\Users\49176\Desktop\Studium\Masterarbeit_3\Results\LogReg"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FINAL_DIR, exist_ok=True)

# Business values
V_TP = 30   # Profit per true positive
C_FP = -5   # Cost per false positive

datasets = {
    "raw":    ("X_train_raw.csv",    "y_train_raw.csv"),
    "smotenc": ("X_train_smotenc.csv", "y_train_smotenc.csv"),
    "under":  ("X_train_under.csv",  "y_train_under.csv"),
}

# Load test set
X_test_raw = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
y_test = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).values.ravel()

#  Profit-based threshold optimization 
def choose_threshold_profit(y_true, y_prob, V_TP=30, C_FP=-5, plot=False, variant_name="variant"):
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
        plt.title(f"Expected Profit Curve — {variant_name.upper()}")
        plt.xlabel("Threshold")
        plt.ylabel("Expected Profit (€)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"profit_curve_{variant_name}.png"))
        plt.close()

    return best_threshold

#  Evaluation on test set 
def compute_metrics(y_true, y_prob, thr):
    y_hat = (y_prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_hat).ravel()
    return dict(
        threshold_used=float(thr),
        precision=float(precision_score(y_true, y_hat, zero_division=0)),
        recall=float(recall_score(y_true, y_hat, zero_division=0)),
        f1_score=float(f1_score(y_true, y_hat, zero_division=0)),
        roc_auc=float(roc_auc_score(y_true, y_prob)),
        pr_auc=float(average_precision_score(y_true, y_prob)),
        tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn)
    )

#  Main Loop 
for name, (Xfile, yfile) in datasets.items():
    print(f"[INFO] Threshold tuning (validation) for LogReg — {name.upper()}")

    # Load tuned model parameters
    tuned_model = joblib.load(os.path.join(MODEL_DIR, f"logreg_tuned_{name}.pkl"))
    tuned_params = tuned_model.get_params()

    # Load training data
    X_train_raw = pd.read_csv(os.path.join(DATA_DIR, Xfile))
    y_train = pd.read_csv(os.path.join(DATA_DIR, yfile)).values.ravel()

    # Train/val split
    X_tr_raw, X_val_raw, y_tr, y_val = train_test_split(
        X_train_raw, y_train, test_size=0.2, stratify=y_train, random_state=42
    )

    #  SCALING
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr_raw)
    X_val = scaler.transform(X_val_raw)
    X_train = scaler.transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    # Refit model with tuned params
    valid_params = {k: v for k, v in tuned_params.items() if k in LogisticRegression().get_params()}
    clf = LogisticRegression(**valid_params)
    clf.fit(X_tr, y_tr)

    # Predict proba on validation
    val_prob = clf.predict_proba(X_val)[:, 1]
    t_star = choose_threshold_profit(y_val, val_prob, V_TP, C_FP, plot=True, variant_name=name)

    #  Save PR sweep (NEW!) 
    p, r, thr = precision_recall_curve(y_val, val_prob)
    sweep_df = pd.DataFrame({"precision": p[:-1], "recall": r[:-1], "threshold": thr})
    sweep_df.to_csv(os.path.join(RESULTS_DIR, f"logreg_val_pr_sweep_{name}.csv"), index=False)

    # Refit on full training data
    clf.fit(X_train, y_train)

    # Save final model, scaler, threshold
    final_model_dir = os.path.join(MODEL_DIR, f"final_{name}")
    os.makedirs(final_model_dir, exist_ok=True)
    joblib.dump(clf, os.path.join(final_model_dir, "model_final.pkl"))
    joblib.dump(scaler, os.path.join(final_model_dir, "scaler.pkl"))
    with open(os.path.join(final_model_dir, "threshold_used.json"), "w", encoding="utf-8") as f:
        json.dump({"threshold_used": t_star}, f, indent=2)

    # Evaluate on test set
    test_prob = clf.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, test_prob, t_star)

    # Save metrics
    out_csv = os.path.join(FINAL_DIR, f"logreg_metrics_{name}.csv")
    pd.DataFrame([{"dataset": name, **metrics}]).to_csv(out_csv, index=False)

print("Business-calibrated thresholds applied. Models + metrics + PR curves saved.")
