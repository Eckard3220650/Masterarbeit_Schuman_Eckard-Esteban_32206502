# 08d_tune_ann.py (Final — Cleaned & Tuned for raw, under, smotenc)

import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import f1_score, make_scorer
from scikeras.wrappers import KerasClassifier

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

#  CONFIG 
DATA_DIR    = r"C:\Users\49176\Desktop\Studium\Masterarbeit_3\ModelData"
MODEL_DIR   = r"C:\Users\49176\Desktop\Studium\Masterarbeit_3\Models\ANN_Tuned"
RESULTS_DIR = r"C:\Users\49176\Desktop\Studium\Masterarbeit_3\Results\ANN_Tuned"
Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

# Updated dataset list
datasets = {
    "raw":     ("X_train_raw.csv",    "y_train_raw.csv"),
    "under":   ("X_train_under.csv",  "y_train_under.csv"),
    "smotenc": ("X_train_smotenc.csv", "y_train_smotenc.csv"),
}

#  Reproducibility
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

#  MODEL BUILDER 
def create_model(hidden_layer_sizes=(64, 32), dropout_rate=0.3, learning_rate=0.001, input_dim=None):
    tf.keras.backend.clear_session()
    model = Sequential(name="ann_binary")
    model.add(Input(shape=(input_dim,)))
    model.add(Dense(hidden_layer_sizes[0], activation="relu"))
    model.add(Dropout(dropout_rate))
    model.add(Dense(hidden_layer_sizes[1], activation="relu"))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model

#  TUNING LOOP 
for name, (X_file, y_file) in datasets.items():
    print(f"\n[INFO] Tuning ANN on {name.upper()}")

    X = pd.read_csv(os.path.join(DATA_DIR, X_file))
    y = pd.read_csv(os.path.join(DATA_DIR, y_file)).values.ravel()
    input_dim = X.shape[1]

    #  EarlyStopping
    es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    est = KerasClassifier(
        model=create_model,
        input_dim=input_dim,
        validation_split=0.2,
        verbose=0,
        random_state=SEED
    )

    param_grid = {
        "model__hidden_layer_sizes": [(64, 32), (128, 64), (64, 64)],
        "model__dropout_rate": [0.2, 0.3, 0.4],
        "model__learning_rate": [0.001, 0.0005],
        "batch_size": [32, 64],
        "epochs": [50],
        "callbacks": [[es]],
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    scorer = make_scorer(f1_score)

    grid = GridSearchCV(
        estimator=est,
        param_grid=param_grid,
        scoring=scorer,
        cv=cv,
        n_jobs=1,      
        verbose=1,
        refit=True
    )

    grid.fit(X, y)

    best_estimator = grid.best_estimator_
    best_params    = grid.best_params_
    best_score     = grid.best_score_

    #  Save model (.h5)
    model_path = os.path.join(MODEL_DIR, f"ann_tuned_{name}.h5")
    best_estimator.model_.save(model_path)

    #  Save best params summary
    summary_path = os.path.join(RESULTS_DIR, f"ann_tuning_summary_{name}.csv")
    meta = pd.DataFrame([{
        "dataset": name,
        "best_f1_score": best_score,
        **{k.replace("model__", ""): v for k, v in best_params.items() if k.startswith("model__")},
        "batch_size": best_params.get("batch_size"),
        "epochs": best_params.get("epochs"),
        "seed": SEED
    }])
    meta.to_csv(summary_path, index=False)

    #  Save CV results
    pd.DataFrame(grid.cv_results_).to_csv(os.path.join(RESULTS_DIR, f"ann_cv_results_{name}.csv"), index=False)

    print(f"Tuning complete for {name.upper()} — Best F1: {best_score:.4f}")
    print(f"Model saved:     {model_path}")
    print(f"Summary saved:   {summary_path}")
