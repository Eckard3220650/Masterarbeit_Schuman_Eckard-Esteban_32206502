# 09d_threshold_tuning_ann.py
 
import os, json
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    precision_recall_curve
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

#  CONFIG 
MODEL_DIR   = r"C:\Users\49176\Desktop\Studium\Masterarbeit_3\Models\ANN\ANN_Tuned"
SCALER_DIR  = os.path.join(MODEL_DIR, "Scalers")
DATA_DIR    = r"C:\Users\49176\Desktop\Studium\Masterarbeit_3\ModelData"
RES_THR_DIR = r"C:\Users\49176\Desktop\Studium\Masterarbeit_3\Results\ANN\Thresholds"
FINAL_DIR   = r"C:\Users\49176\Desktop\Studium\Masterarbeit_3\Results\ANN"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(SCALER_DIR, exist_ok=True)
os.makedirs(RES_THR_DIR, exist_ok=True)
os.makedirs(FINAL_DIR, exist_ok=True)

#  BUSINESS COSTS
V_TP = 30   # profit per true positive
C_FP = -5   # cost per false positive

datasets = {
    "raw":     ("X_train_raw.csv",    "y_train_raw.csv"),
    "smotenc": ("X_train_smotenc.csv", "y_train_smotenc.csv"),
    "under":   ("X_train_under.csv",  "y_train_under.csv"),
}

X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
y_test = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).values.ravel()

#  Business threshold optimizer 
def choose_threshold_profit(y_true, y_prob, V_TP=30, C_FP=-5, plot=False, variant="ann"):
    thresholds = np.linspace(0.01, 0.99, 100)
    profits = []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tp = ((y_true == 1) & (y_pred == 1)).sum()
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        profits.append(V_TP * tp + C_FP * fp)

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

#  Metrics helper 
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
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)
    }

#  Main loop 
for name, (Xfile, yfile) in datasets.items():
    print(f"[INFO] Training and threshold tuning for ANN — {name.upper()}")

    X_train = pd.read_csv(os.path.join(DATA_DIR, Xfile))
    y_train = pd.read_csv(os.path.join(DATA_DIR, yfile)).values.ravel()

    #  Scale 
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(SCALER_DIR, f"scaler_{name}.pkl"))

    #  Train/val split 
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_scaled, y_train, test_size=0.2, stratify=y_train, random_state=42
    )

    #  Define ANN model 
    model = Sequential(name="ann_binary")
    model.add(Input(shape=(X_train_scaled.shape[1],)))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])

    #  Fit on train 
    es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    model.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=50, batch_size=64, callbacks=[es], verbose=0)

    #  Threshold tuning 
    val_prob = model.predict(X_val, verbose=0).ravel()
    t_star = choose_threshold_profit(y_val, val_prob, V_TP, C_FP, plot=True, variant=name)

    #  Retrain on full set 
    model.fit(X_train_scaled, y_train, validation_split=0.2, epochs=50, batch_size=64, callbacks=[es], verbose=0)

    #  Evaluate on TEST 
    test_prob = model.predict(X_test_scaled, verbose=0).ravel()
    metrics = compute_metrics(y_test, test_prob, t_star)
    pd.DataFrame([{"dataset": name, **metrics}]).to_csv(
        os.path.join(FINAL_DIR, f"ann_metrics_{name}.csv"), index=False
    )

    #  Save model + threshold 
    final_model_dir = os.path.join(MODEL_DIR, f"final_{name}")
    os.makedirs(final_model_dir, exist_ok=True)
    model.save(os.path.join(final_model_dir, "model.h5"))
    with open(os.path.join(final_model_dir, "threshold_used.json"), "w") as f:
        json.dump({"threshold_used": t_star}, f, indent=2)

    #  Save PR sweep 
    p, r, thr = precision_recall_curve(y_val, val_prob)
    sweep_df = pd.DataFrame({"precision": p[:-1], "recall": r[:-1], "threshold": thr})
    sweep_df.to_csv(os.path.join(RES_THR_DIR, f"ann_val_pr_sweep_{name}.csv"), index=False)

print("[OK] All ANN models trained and business thresholds applied.")
