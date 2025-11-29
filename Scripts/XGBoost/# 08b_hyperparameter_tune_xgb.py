# 08b_tune_xgb.py 

import os
import joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import f1_score, make_scorer
from xgboost import XGBClassifier
import numpy as np

#  CONFIG 
DATA_DIR    = r"C:\Users\49176\Desktop\Studium\Masterarbeit_3\ModelData"
MODEL_DIR   = r"C:\Users\49176\Desktop\Studium\Masterarbeit_3\Models\XGB\XGB_Tuned"
RESULTS_DIR = r"C:\Users\49176\Desktop\Studium\Masterarbeit_3\Results\XGB\XGB_Tuned"
Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

# Datasets
datasets = {
    "raw":     ("X_train_raw.csv",    "y_train_raw.csv"),
    "under":   ("X_train_under.csv",  "y_train_under.csv"),
    "smotenc": ("X_train_smotenc.csv", "y_train_smotenc.csv"),
}

#  PARAMETER GRID 
param_grid = {
    "n_estimators": [200, 400, 600],
    "max_depth": [3, 6, 9],
    "learning_rate": [0.03, 0.1],
    "subsample": [0.7, 1.0],
    "colsample_bytree": [0.7, 1.0],
    "min_child_weight": [1, 5],
    "gamma": [0, 1],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scorer = make_scorer(f1_score)

#  TUNING LOOP 
for name, (X_file, y_file) in datasets.items():
    print(f"\n[INFO] Tuning XGBoost on: {name.upper()}")

    X = pd.read_csv(os.path.join(DATA_DIR, X_file))
    y = pd.read_csv(os.path.join(DATA_DIR, y_file)).values.ravel()

    # Class imbalance ratio
    pos = (y == 1).sum()
    neg = (y == 0).sum()
    spw = (neg / max(pos, 1)) if pos > 0 else 1.0

    base_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
        scale_pos_weight=spw
    )

    grid = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring=scorer,
        cv=cv,
        n_jobs=-1,
        verbose=1,
        refit=True,
        return_train_score=False
    )

    grid.fit(X, y)

    best_model  = grid.best_estimator_
    best_params = grid.best_params_
    best_score  = grid.best_score_

    #  SAVE MODEL 
    model_path = os.path.join(MODEL_DIR, f"xgb_tuned_{name}.pkl")
    joblib.dump(best_model, model_path)

    #  SAVE RESULTS 
    summary_path = os.path.join(RESULTS_DIR, f"xgb_tuning_summary_{name}.csv")
    cv_path      = os.path.join(RESULTS_DIR, f"xgb_cv_results_{name}.csv")

    pd.DataFrame([{
        "dataset": name,
        "best_f1_score": best_score,
        "scale_pos_weight": spw,
        **best_params
    }]).to_csv(summary_path, index=False)

    pd.DataFrame(grid.cv_results_).to_csv(cv_path, index=False)

    print(f"Done tuning {name.upper()} — Best F1: {best_score:.4f} (spw = {spw:.2f})")
    print(f"Model saved:     {model_path}")
    print(f"Summary saved:   {summary_path}")
    print(f"Full CV results: {cv_path}")
