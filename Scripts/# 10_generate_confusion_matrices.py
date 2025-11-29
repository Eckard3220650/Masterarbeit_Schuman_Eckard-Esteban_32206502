# 10_generate_confusion_matrices_from_combined.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#  CONFIG 
COMBINED   = r"C:\Users\49176\Desktop\Studium\Masterarbeit_3\Results\ensemble_model_comparison.csv"
OUTPUT_DIR = r"C:\Users\49176\Desktop\Studium\Masterarbeit_3\Results\ConfusionMatrices"
os.makedirs(OUTPUT_DIR, exist_ok=True)

#  Load & preprocess 
df = pd.read_csv(COMBINED)

# Normalize column names
df.columns = [c.lower() for c in df.columns]

# Rename to match expectations
df = df.rename(columns={
    "threshold": "threshold_used",
})

# Ensure required columns exist
required = {"model", "variant", "tp", "fp", "fn", "tn"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing columns in combined metrics: {missing}")

# Drop rows missing confusion matrix counts
df = df.dropna(subset=["tp", "fp", "fn", "tn"]).copy()

# Ensure integer types for counts
for c in ["tp", "fp", "fn", "tn"]:
    df[c] = df[c].astype(int)

#  Fill metrics if missing or recompute (safe overwrite) 
def _metric_fill(r):
    tp, fp, fn, tn = r.tp, r.fp, r.fn, r.tn
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rcl = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * rcl / (p + rcl) if (p + rcl) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return pd.Series({
        "precision_calc": p,
        "recall_calc": rcl,
        "f1_score_calc": f1,
        "specificity": spec,
        "support": tp + fp + fn + tn
    })

metrics = df.apply(_metric_fill, axis=1)

# Merge calculated metrics safely
for col in ["precision", "recall", "f1_score"]:
    calc_col = f"{col}_calc"
    if col in df.columns:
        df[col] = df[col].fillna(metrics[calc_col])
    else:
        df[col] = metrics[calc_col]

df["specificity"] = metrics["specificity"]
df["support"] = metrics["support"]

#  Individual Confusion Matrix PNGs 
for _, r in df.iterrows():
    model = str(r["model"])
    variant = str(r["variant"])
    status = str(r.get("status", "Tuned"))
    tp, fp, fn, tn = r.tp, r.fp, r.fn, r.tn
    thr = r.get("threshold_used", None)

    cm = np.array([[tn, fp],
                   [fn, tp]])

    plt.figure(figsize=(4, 3))
    ax = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                     xticklabels=["No Purchase", "Purchase"],
                     yticklabels=["No Purchase", "Purchase"])
    title = f"{model} ({variant.upper()}, {status})"
    if pd.notna(thr):
        title += f"\nThreshold = {float(thr):.2f}"
    ax.set_title(title, pad=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, f"{model.lower()}_{status.lower()}_confmat_{variant}.png")
    out_path = out_path.replace(" ", "_")
    plt.savefig(out_path, dpi=200)
    plt.close()

#  1×3 Panel Per Model (RAW / SMOTENC / UNDER) 
order = ["raw", "smotenc", "under"]
for model in sorted(df["model"].unique()):
    for status in ["Baseline", "Tuned"]:
        sub = df[(df["model"] == model) & (df["status"] == status)]
        if sub.empty:
            continue
        sub = sub.set_index("variant")

        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
        for i, variant in enumerate(order):
            ax = axes[i]
            if variant in sub.index:
                r = sub.loc[variant]
                cm = np.array([[r.tn, r.fp],
                               [r.fn, r.tp]])
                thr = r.get("threshold_used", None)
                t = f"{model} — {variant.upper()}"
                if pd.notna(thr):
                    t += f"\nτ={float(thr):.2f}"
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                            xticklabels=["No", "Yes"], yticklabels=["No", "Yes"], ax=ax)
                ax.set_title(t, pad=8)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
            else:
                ax.axis("off")

        plt.tight_layout()
        panel_path = os.path.join(OUTPUT_DIR, f"{model.lower()}_{status.lower()}_confmat_panel.png")
        panel_path = panel_path.replace(" ", "_")
        plt.savefig(panel_path, dpi=200)
        plt.close()

#  Save summary CSV 
summary_cols = [
    "model", "variant", "status", "threshold_used",
    "tp", "fp", "fn", "tn",
    "precision", "recall", "specificity", "f1_score", "support"
]
summary = df[summary_cols].copy()
summary[["precision", "recall", "specificity", "f1_score"]] = summary[["precision", "recall", "specificity", "f1_score"]].round(3)
summary["threshold_used"] = summary["threshold_used"].round(3)

summary_path = os.path.join(OUTPUT_DIR, "final_confmat_summary.csv")
summary.to_csv(summary_path, index=False)

print(f" Saved individual confusion matrices → {OUTPUT_DIR}")
print(f" Saved summary CSV → {summary_path}")
