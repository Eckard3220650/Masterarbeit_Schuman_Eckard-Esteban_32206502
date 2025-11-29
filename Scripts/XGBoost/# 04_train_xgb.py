# 04_train_xgb.py 
import pandas as pd
import os
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix
)

#  CONFIG 
DATA_DIR = r"C:\Users\49176\Desktop\Studium\Masterarbeit_3\ModelData"
MODEL_DIR = r"C:\Users\49176\Desktop\Studium\Masterarbeit_3\Models\XGB\XGB_default"
RESULTS_DIR = r"C:\Users\49176\Desktop\Studium\Masterarbeit_3\Results\XGB"
PREPROCESSOR_PATH = os.path.join(DATA_DIR, "preprocessor.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

datasets = {
    "raw": ("X_train_raw.csv", "y_train_raw.csv"),
    "under": ("X_train_under.csv", "y_train_under.csv"),
    "smotenc": ("X_train_smotenc.csv", "y_train_smotenc.csv"),
}

#  LOAD TEST SET 
X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
y_test = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).values.ravel()

#  LOAD FEATURE NAMES 
try:
    pre = joblib.load(PREPROCESSOR_PATH)
    feature_names = pre.get_feature_names_out()
except Exception:
    feature_names = [f"f{i}" for i in range(X_test.shape[1])]

#  BUSINESS COST CONFIG 
V_TP = 30
C_FP = -5
BASELINE_THRESHOLD = 0.5

#  TRAINING LOOP
for name, (X_file, y_file) in datasets.items():
    print(f"[INFO] Training baseline XGBoost on {name.upper()}")

    # Load data
    X_train = pd.read_csv(os.path.join(DATA_DIR, X_file))
    y_train = pd.read_csv(os.path.join(DATA_DIR, y_file)).values.ravel()

    # Model
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1
    )
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
        "profit": V_TP * tp + C_FP * fp
    }

    # Save per variant
    out_path = os.path.join(RESULTS_DIR, f"xgb_metrics_baseline_{name}.csv")
    pd.DataFrame([metrics]).to_csv(out_path, index=False)

    # Save model and feature names
    joblib.dump(model, os.path.join(MODEL_DIR, f"xgb_baseline_{name}.pkl"))
    pd.Series(feature_names).to_csv(
        os.path.join(MODEL_DIR, f"xgb_baseline_{name}_features.csv"), index=False
    )

    print(f" Saved model and metrics for {name}")

print("Baseline XGBoost models trained and saved.")
