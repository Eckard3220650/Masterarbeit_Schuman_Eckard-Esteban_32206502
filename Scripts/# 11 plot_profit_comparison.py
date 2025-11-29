#11 plot_profit_comparison
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

#  CONFIG 
CSV_PATH = r"C:\Users\49176\Desktop\Studium\Masterarbeit_3\Results\ensemble_model_comparison.csv"
OUTPUT_DIR = r"C:\Users\49176\Desktop\Studium\Masterarbeit_3\Results\MetricPlots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

#  BUSINESS VALUES 
V_TP = 30   # Value of true positive
C_FP = -5   # Cost of false positive

#  LOAD AND CHECK DATA 
df = pd.read_csv(CSV_PATH)
required_cols = {"tp", "fp", "Model", "Variant", "Status"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

#  CALCULATE PROFIT 
df["profit"] = df["tp"] * V_TP + df["fp"] * C_FP

#  AGGREGATE BY MODEL, VARIANT, STATUS 
grouped = df.groupby(["Model", "Variant", "Status"])["profit"].mean().reset_index()

#  PLOT EACH MODEL 
for model in grouped["Model"].unique():
    model_data = grouped[grouped["Model"] == model]

    plt.figure(figsize=(7, 5))
    ax = sns.barplot(
        data=model_data,
        x="Variant", y="profit",
        hue="Status",
        palette="Set2"
    )

    #  Annotate each bar with profit value 
    for p in ax.patches:
        height = p.get_height()
        if not pd.isna(height) and abs(height) > 1: 
            x = p.get_x() + p.get_width() / 2
            y = height + 50  
            ax.text(x, y, f"{height:.0f}€", ha='center', va='bottom', fontsize=9)

    plt.title(f"{model} — Profit by Dataset Variant and Status")
    plt.ylabel("Average Profit (€)")
    plt.xlabel("Dataset Variant")
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.legend(title="Status")

    #  SAVE 
    out_path = os.path.join(OUTPUT_DIR, f"{model}_profit.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[✓] Saved: {out_path}")

print("All model-wise profit plots generated.")
