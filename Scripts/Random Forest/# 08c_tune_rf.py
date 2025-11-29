# 08c_tune_rf.py (Final – Updated for raw, under, smotenc)

import pandas as pd
import os
import joblib
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, make_scorer

#  CONFIG 
DATA_DIR    = r"C:\Users\49176\Desktop\Studium\Masterarbeit_3\ModelData"
MODEL_DIR   = r"C:\Users\49176\Desktop\Studium\Masterarbeit_3\Models\RF\RF_Tuned"
RESULTS_DIR = r"C:\Users\49176\Desktop\Studium\Masterarbeit_3\Results\RF\RF_Tuned"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Updated dataset variants
datasets = {
    "raw":     ("X_train_raw.csv",    "y_train_raw.csv"),
    "under":   ("X_train_under.csv",  "y_train_under.csv"),
    "smotenc": ("X_train_smotenc.csv", "y_train_smotenc.csv"),
}

#  PARAMETER GRID 
param_grid = {
    "n_estimators": [200, 400],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "max_features": ["sqrt", "log2", None],
    "class_weight": [None, "balanced_subsample"],
    "bootstrap": [True],
    "random_state": [42],  # for consistent CV behavior
}

# CV + Scorer
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scorer = make_scorer(f1_score)

#  MAIN LOOP 
for name, (X_file, y_file) in datasets.items():
    print(f"\n[INFO] Tuning RandomForest on: {name.upper()}")

    X = pd.read_csv(os.path.join(DATA_DIR, X_file))
    y = pd.read_csv(os.path.join(DATA_DIR, y_file)).values.ravel()

    base_model = RandomForestClassifier(n_jobs=-1)

    grid = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring=scorer,
        cv=cv,
        n_jobs=-1,
        verbose=1,
        refit=True,
        return_train_score=False,
    )

    grid.fit(X, y)

    best_model  = grid.best_estimator_
    best_params = grid.best_params_
    best_score  = grid.best_score_

    #  SAVE MODEL 
    model_path = os.path.join(MODEL_DIR, f"rf_tuned_{name}.pkl")
    joblib.dump(best_model, model_path)

    #  SAVE RESULTS 
    summary_path = os.path.join(RESULTS_DIR, f"rf_tuning_summary_{name}.csv")
    cv_path      = os.path.join(RESULTS_DIR, f"rf_cv_results_{name}.csv")

    pd.DataFrame([{
        "dataset": name,
        "best_f1_score": best_score,
        **best_params
    }]).to_csv(summary_path, index=False)

    pd.DataFrame(grid.cv_results_).to_csv(cv_path, index=False)

    print(f"Done tuning {name.upper()} — Best F1: {best_score:.4f}")
    print(f"Model saved:     {model_path}")
    print(f"Summary saved:   {summary_path}")
    print(f"Full CV results: {cv_path}")
