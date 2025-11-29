# 09_evaluate_tunned_models.py
import os
import pandas as pd

#  CONFIG 
BASE_RESULTS_DIR = r"C:\Users\49176\Desktop\Studium\Masterarbeit_3\Results"
OUTPUT_FILE = os.path.join(BASE_RESULTS_DIR, "ensemble_model_comparison.csv")

model_folders = {
    "LogReg": "LogReg",
    "RF": "RF",
    "XGB": "XGB",
    "ANN": "ANN"
}

all_metrics = []

#  LOAD METRICS 
for model, folder in model_folders.items():
    full_path = os.path.join(BASE_RESULTS_DIR, folder)
    for fname in os.listdir(full_path):
        if fname.endswith(".csv") and "metrics" in fname:
            file_path = os.path.join(full_path, fname)
            try:
                df = pd.read_csv(file_path)
            except Exception as e:
                print(f"[WARN] Skipped {fname}: {e}")
                continue

            #  FIXED: Infer variant and status correctly 
            parts = fname.replace(".csv", "").split("_")

            if "baseline" in parts:
                status = "Baseline"
                baseline_index = parts.index("baseline")
                if baseline_index + 1 < len(parts):
                    variant = parts[baseline_index + 1]
                else:
                    variant = "raw"  # fallback if no variant part exists
            else:
                status = "Tuned"
                variant = parts[-1]

            df["Model"] = model
            df["Variant"] = variant
            df["Status"] = status

            # Normalize column names if needed
            rename_map = {
                "f1": "f1_score",
                "F1 Score": "f1_score",
                "F1": "f1_score",
                "Threshold": "threshold_used",
                "threshold": "threshold_used",
                "Threshold_used": "threshold_used"
            }
            df.rename(columns=rename_map, inplace=True)

            all_metrics.append(df)

#  COMBINE 
if not all_metrics:
    raise RuntimeError("No metrics found. Check folder paths and files.")

combined = pd.concat(all_metrics, ignore_index=True)

#  COLUMN ORDER 
desired_order = [
    "Model", "Variant", "Status", "threshold_used", "f1_score", "precision", "recall",
    "roc_auc", "pr_auc", "tp", "fp", "fn", "tn"
]
existing_cols = [col for col in desired_order if col in combined.columns]
combined = combined[existing_cols + [c for c in combined.columns if c not in existing_cols]]

# Round scores for clarity
for metric in ["threshold_used", "f1_score", "precision", "recall", "roc_auc", "pr_auc"]:
    if metric in combined.columns:
        combined[metric] = combined[metric].astype(float).round(3)

#  SAVE 
combined.to_csv(OUTPUT_FILE, index=False)
print(f"[OK] Combined model results saved to:\n{OUTPUT_FILE}")
