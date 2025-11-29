# 12c_local_shap_xgb_raw.py


import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier
import numpy as np

#  CONFIG 
MODEL_PATH = r"C:\Users\49176\Desktop\Studium\Masterarbeit_3\Models\XGB\XGB_Tuned\final_raw\model_final.pkl"
X_TEST_PATH = r"C:\Users\49176\Desktop\Studium\Masterarbeit_3\ModelData\X_test.csv"
Y_TEST_PATH = r"C:\Users\49176\Desktop\Studium\Masterarbeit_3\ModelData\y_test.csv"
PREPROCESSOR_PATH = r"C:\Users\49176\Desktop\Studium\Masterarbeit_3\ModelData\preprocessor.pkl"
OUTPUT_DIR = r"C:\Users\49176\Desktop\Studium\Masterarbeit_3\Results\SHAP\Local"

os.makedirs(OUTPUT_DIR, exist_ok=True)
model_name = "XGB_raw_Tuned"

#  LOAD DATA & MODEL 
X_raw = pd.read_csv(X_TEST_PATH)
y_true = pd.read_csv(Y_TEST_PATH).values.ravel()
preprocessor = joblib.load(PREPROCESSOR_PATH)
model: XGBClassifier = joblib.load(MODEL_PATH)

expected_cols = set(preprocessor.feature_names_in_)
actual_cols = set(X_raw.columns)

missing = expected_cols - actual_cols
extra = actual_cols - expected_cols

print(f"Expected columns: {expected_cols}")
print(f"Missing columns: {missing}")
print(f"Extra columns: {extra}")


#  Transform Features 
feature_names = preprocessor.get_feature_names_out()
X_df = pd.DataFrame(X_raw.values, columns=feature_names)


#  Predictions 
y_pred = model.predict(X_df)

#  Identify TP, TN, FP, FN indices 
indices = {
    "TP": ((y_true == 1) & (y_pred == 1)),
    "TN": ((y_true == 0) & (y_pred == 0)),
    "FP": ((y_true == 0) & (y_pred == 1)),
    "FN": ((y_true == 1) & (y_pred == 0)),
}

#  SHAP EXPLAINER
explainer = shap.Explainer(model, X_df)
shap_values = explainer(X_df)

#  Generate Plots 
for label, mask in indices.items():
    idx = np.where(mask)[0][0] if mask.sum() > 0 else None
    if idx is None:
        print(f"[WARN] No {label} sample found.")
        continue

    print(f"[INFO] Plotting {label} sample (index {idx})")

    shap_val = shap_values[idx]

    # Create PNG plot with waterfall
    plt.figure()
    shap.plots.waterfall(shap_val, max_display=15, show=False)

    out_path = os.path.join(OUTPUT_DIR, f"{model_name}_local_{label}.png")
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close()

    print(f" Saved: {out_path}")

print("Local SHAP plots completed.")
