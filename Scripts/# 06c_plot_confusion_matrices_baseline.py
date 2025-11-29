# 06c_plot_confusion_matrices_baseline.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#  CONFIG 
BASE_RESULTS_DIR = r"C:\Users\49176\Desktop\Studium\Masterarbeit_3\Results"
PLOT_DIR = os.path.join(BASE_RESULTS_DIR, "ConfusionMatrices", "Baseline")
os.makedirs(PLOT_DIR, exist_ok=True)

MODELS = ["LogReg", "RF", "XGB", "ANN"]
METRICS_BASELINE_FILES = {
    "LogReg": "logreg_metrics_baseline.csv",
    "RF": "rf_metrics_baseline.csv",
    "XGB": "xgb_metrics_baseline.csv",
    "ANN": "ann_metrics_baseline.csv",
}

#  LOAD METRICS 
for model in MODELS:
    metrics_path = os.path.join(BASE_RESULTS_DIR, model, METRICS_BASELINE_FILES[model])

    if not os.path.exists(metrics_path):
        print(f"[WARN] Missing: {metrics_path}")
        continue

    df = pd.read_csv(metrics_path)

    if "dataset" not in df.columns:
        df["dataset"] = "baseline"
    if "model" not in df.columns:
        df["model"] = model

    for _, row in df.iterrows():
        try:
            cm = np.array([
                [int(row["tn"]), int(row["fp"])],
                [int(row["fn"]), int(row["tp"])]
            ])
        except Exception as e:
            print(f"[WARN] Skipping {model}: {e}")
            continue

        plt.figure(figsize=(4, 3.5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=["Pred: 0", "Pred: 1"],
                    yticklabels=["Actual: 0", "Actual: 1"])

        plt.title(f"Confusion Matrix – {model} (baseline)")
        plt.tight_layout()
        filename = f"{model.lower()}_baseline_confusion_matrix.png"
        plt.savefig(os.path.join(PLOT_DIR, filename), dpi=300)
        plt.close()
        print(f"[INFO] Saved: {filename}")

print(f"\n Baseline confusion matrices saved to: {PLOT_DIR}")
