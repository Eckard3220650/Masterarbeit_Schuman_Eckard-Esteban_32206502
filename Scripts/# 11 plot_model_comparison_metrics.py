
#11_plot_model_comparison_metrics.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

#  CONFIG 
CSV_PATH = r"C:\Users\49176\Desktop\Studium\Masterarbeit_3\Results\ensemble_model_comparison.csv"
OUTPUT_DIR = r"C:\Users\49176\Desktop\Studium\Masterarbeit_3\Results\MetricPlots\ModelWise"
os.makedirs(OUTPUT_DIR, exist_ok=True)

#  METRICS TO PLOT 
metrics = ["f1_score", "precision", "recall", "roc_auc", "pr_auc"]

#  LOAD DATA 
df = pd.read_csv(CSV_PATH)

#  REQUIRED COLUMNS CHECK 
required_cols = {"Model", "Variant", "Status"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing columns in dataset: {missing}")

#  AGGREGATE 
grouped = df.groupby(["Model", "Variant", "Status"])[metrics].mean().reset_index()

#  PLOT EACH METRIC PER MODEL 
for model in grouped["Model"].unique():
    model_data = grouped[grouped["Model"] == model]

    for metric in metrics:
        plt.figure(figsize=(7, 5))
        ax = sns.barplot(
            data=model_data,
            x="Variant", y=metric,
            hue="Status",
            palette="Set2"
        )

        
        for p in ax.patches:
            height = p.get_height()
    
        if not pd.isna(height) and height > 0.01:
            x = p.get_x() + p.get_width() / 2
            y = height + 0.01
            s = f"{height:.2f}"
            ax.text(x, y, s, ha='center', va='bottom', fontsize=9, color='black')


        plt.title(f"{model} — {metric.upper()} by Dataset Variant and Status")
        plt.ylim(0, 1)
        plt.ylabel(metric.upper())
        plt.xlabel("Dataset Variant")
        plt.grid(axis='y', linestyle='--', alpha=0.4)
        plt.legend(title="Status")
        plt.tight_layout()

        # Save plot
        out_path = os.path.join(OUTPUT_DIR, f"{model}_{metric}.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"[✓] Saved: {out_path}")

print("Model-wise annotated metric plots generated.")
