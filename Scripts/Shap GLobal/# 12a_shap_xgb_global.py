#12a_shap_global_top_models
import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np


#  CONFIG 
MODEL_DIR = r"C:\Users\49176\Desktop\Studium\Masterarbeit_3\Models\XGB\XGB_Tuned"
X_TEST_PATH = r"C:\Users\49176\Desktop\Studium\Masterarbeit_3\ModelData\X_test.csv"
PREPROCESSOR_PATH = r"C:\Users\49176\Desktop\Studium\Masterarbeit_3\ModelData\preprocessor.pkl"
OUTPUT_DIR = r"C:\Users\49176\Desktop\Studium\Masterarbeit_3\Results\SHAP\Global"

os.makedirs(OUTPUT_DIR, exist_ok=True)

#  MODEL TO EXPLAIN 
model_name = "XGB_raw_Tuned"
model_path = os.path.join(MODEL_DIR, "final_raw", "model_final.pkl")

#  LOAD DATA & PREPROCESSOR 
X_raw = pd.read_csv(X_TEST_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)

# Apply preprocessor to assign feature names
feature_names = preprocessor.get_feature_names_out()
X_df = pd.DataFrame(X_raw.values, columns=feature_names)

#  LOAD MODEL 
print(f"[INFO] Explaining XGBoost model: {model_name}")
model = joblib.load(model_path)

#  SHAP VALUES 
explainer = shap.Explainer(model, X_df)
shap_values = explainer(X_df)

#  Extract SHAP values 
shap_values_np = shap_values.values  # shape: (n_samples, n_features)
mean_shap = np.abs(shap_values_np).mean(axis=0)


#  Build Feature Importance Table 
feature_importance = pd.DataFrame({
    "feature": feature_names,
    "mean_abs_shap": mean_shap
})

#  Top 20 Features: Sort descending, then reverse for barh order 
feature_importance = (
    feature_importance.sort_values("mean_abs_shap", ascending=False)
    .head(20)
    .sort_values("mean_abs_shap", ascending=True)
)

#  Plot Bar Chart 
plt.figure(figsize=(8, 6))
plt.barh(feature_importance["feature"], feature_importance["mean_abs_shap"], color="skyblue")
plt.xlabel("Mean |SHAP value|")
plt.title(f"Feature Importance — {model_name} (Top 20)")
plt.tight_layout()

out_path = os.path.join(OUTPUT_DIR, f"{model_name}_shap_bar_top20.png")
plt.savefig(out_path, dpi=300)
plt.close()

print(f"Saved SHAP bar plot: {out_path}")