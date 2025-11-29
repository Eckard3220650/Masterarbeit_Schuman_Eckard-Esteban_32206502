# 05_evaluate_models.py (Supports main + baseline metrics)

import pandas as pd
import os

#  CONFIG 
BASE_RESULTS_DIR = r"C:\Users\49176\Desktop\Studium\Masterarbeit_3\Results"
OUTPUT_PATH = os.path.join(BASE_RESULTS_DIR, "all_models_combined_metrics.csv")

model_dirs = {
    "LogReg": "LogReg",
    "RF": "RF",
    "XGB": "XGB",
    "ANN": "ANN"
}

required_cols = [
    "dataset", "accuracy", "precision", "recall", "f1_score",
    "roc_auc", "pr_auc", "tp", "fp", "fn", "tn"
]

df_list = []

#  LOAD METRICS FOR EACH MODEL 
for model_name, subdir in model_dirs.items():
    model_path = os.path.join(BASE_RESULTS_DIR, subdir)

    # Main metrics file
    main_metrics = os.path.join(model_path, f"{model_name.lower()}_metrics.csv")

    # Baseline metrics file (if available)
    baseline_metrics = os.path.join(model_path, f"{model_name.lower()}_metrics_baseline.csv")

    for path, tag in [(main_metrics, None), (baseline_metrics, "baseline")]:
        if not os.path.exists(path):
            print(f"[WARN] Metrics file not found: {path}")
            continue

        df = pd.read_csv(path)
        df["model"] = model_name

        if tag:
            df["dataset"] = "baseline"  # overwrite or set if missing

        # Ensure all required columns exist
        for col in required_cols:
            if col not in df.columns:
                df[col] = pd.NA

        df_list.append(df)

#  CHECK 
if not df_list:
    raise RuntimeError("No metrics files found — nothing to combine.")

#  COMBINE 
combined_df = pd.concat(df_list, ignore_index=True)

# ORGANIZE COLUMNS 
ordered_cols = ["model", "dataset"] + required_cols[1:]
combined_df = combined_df[ordered_cols]

#  SORT 
combined_df.sort_values(by=["model", "dataset"], inplace=True)

# SAVE 
combined_df.to_csv(OUTPUT_PATH, index=False)
print(f"[INFO] Combined metrics saved to: {OUTPUT_PATH}")
