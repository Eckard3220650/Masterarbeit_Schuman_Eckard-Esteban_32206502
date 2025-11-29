 #12_shap_rf_global.py
import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

#  CONFIG 
MODEL_DIR = r"C:\Users\49176\Desktop\Studium\Masterarbeit_3\Models\RF\RF_Tuned"
X_TEST_PATH = r"C:\Users\49176\Desktop\Studium\Masterarbeit_3\ModelData\X_test.csv"
PREPROCESSOR_PATH = r"C:\Users\49176\Desktop\Studium\Masterarbeit_3\ModelData\preprocessor.pkl"
OUTPUT_DIR = r"C:\Users\49176\Desktop\Studium\Masterarbeit_3\Results\SHAP\RF"

os.makedirs(OUTPUT_DIR, exist_ok=True)

#  MODELS TO EXPLAIN 
models_info = [
    {"name": "RF_raw_Tuned", "path": os.path.join(MODEL_DIR, "final_raw", "model_final.pkl")},
    {"name": "RF_smotenc_Tuned", "path": os.path.join(MODEL_DIR, "final_smotenc", "model_final.pkl")},
]

#  LOAD AND PREP TEST DATA 
df_raw = pd.read_csv(X_TEST_PATH)
if "Unnamed: 0" in df_raw.columns:
    df_raw = df_raw.drop(columns=["Unnamed: 0"])

# Load preprocessor for feature names
preprocessor = joblib.load(PREPROCESSOR_PATH)
feature_names = preprocessor.get_feature_names_out()
X_df = pd.DataFrame(df_raw.values, columns=feature_names)

#  SHAP FOR EACH MODEL 
for model_info in models_info:
    model_name = model_info["name"]
    model_path = model_info["path"]

    print(f"\n[INFO] Explaining: {model_name}")
    model = joblib.load(model_path)

    explainer = shap.Explainer(model, X_df)
    shap_values = explainer(X_df)

    # Extract SHAP values for class 1 (purchase intent)
    shap_values_np = shap_values.values[:, :, 1]  # shape: (n_samples, n_features)

    print(f"[DEBUG] SHAP shape: {shap_values_np.shape}, Features: {len(feature_names)}")

    # Compute mean absolute SHAP values per feature
    mean_shap = np.abs(shap_values_np).mean(axis=0)

    feature_importance = pd.DataFrame({
        "feature": X_df.columns,
        "mean_abs_shap": mean_shap
    })

    #  Top 20 Features: Sort descending, plot ascending for barh
    feature_importance = (
        feature_importance.sort_values("mean_abs_shap", ascending=False)
        .head(20)
        .sort_values("mean_abs_shap", ascending=True)
    )

    #  Plot SHAP Bar Chart 
    plt.figure(figsize=(8, 6))
    plt.barh(feature_importance["feature"], feature_importance["mean_abs_shap"], color="skyblue")
    plt.xlabel("Mean |SHAP value|")
    plt.title(f"Feature Importance — {model_name} (Top 20)")
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, f"{model_name}_shap_bar_top20.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved top-20 SHAP bar plot: {out_path}")

print("\n All SHAP top-20 bar plots generated successfully.")