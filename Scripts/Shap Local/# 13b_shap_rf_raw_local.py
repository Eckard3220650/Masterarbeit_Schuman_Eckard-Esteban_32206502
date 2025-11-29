# 12c_local_shap_rf_raw.py 

import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import warnings

#  CONFIG 
BASE_MODEL_DIR = r"C:\Users\49176\Desktop\Studium\Masterarbeit_3\Models\RF\RF_Tuned"
X_TEST_PATH = r"C:\Users\49176\Desktop\Studium\Masterarbeit_3\ModelData\X_test_fixed.csv"
Y_TEST_PATH = r"C:\Users\49176\Desktop\Studium\Masterarbeit_3\ModelData\y_test.csv"
PREPROCESSOR_PATH = r"C:\Users\49176\Desktop\Studium\Masterarbeit_3\ModelData\preprocessor.pkl"
OUTPUT_DIR = r"C:\Users\49176\Desktop\Studium\Masterarbeit_3\Results\SHAP\Local_RF"
os.makedirs(OUTPUT_DIR, exist_ok=True)

#  Suppress feature name warnings 
warnings.filterwarnings("ignore", message="X has feature names")

#  MODEL VARIANTS 
models = {
    "RF_raw_Tuned": os.path.join(BASE_MODEL_DIR, "final_raw", "model_final.pkl"),
    "RF_smotenc_Tuned": os.path.join(BASE_MODEL_DIR, "final_smotenc", "model_final.pkl")
}

#  LOAD DATA 
X_raw = pd.read_csv(X_TEST_PATH)
y_true = pd.read_csv(Y_TEST_PATH).squeeze()
preprocessor = joblib.load(PREPROCESSOR_PATH)

#  Validate columns 
expected_cols = list(preprocessor.feature_names_in_)
missing = [col for col in expected_cols if col not in X_raw.columns]
if missing:
    raise ValueError(f"Still missing columns: {missing}")
X_raw = X_raw[expected_cols]

#  Transform raw features 
X_transformed = preprocessor.transform(X_raw)
feature_names = preprocessor.get_feature_names_out()
X_df = pd.DataFrame(X_transformed, columns=feature_names)

#  LOOP OVER MODELS 
for model_name, model_path in models.items():
    print(f"\n[INFO] Processing model: {model_name}")
    model = joblib.load(model_path)

    #  Predictions 
    y_pred = model.predict(X_df)

    #  Outcome conditions 
    conditions = {
        "TP": (y_true == 1) & (y_pred == 1),
        "TN": (y_true == 0) & (y_pred == 0),
        "FP": (y_true == 0) & (y_pred == 1),
        "FN": (y_true == 1) & (y_pred == 0)
    }

    #  SHAP Explainer 
    explainer = shap.Explainer(model, X_df)

    #  Generate SHAP plots per outcome 
    for outcome, mask in conditions.items():
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            print(f"[WARN] No {outcome} sample found for {model_name}")
            continue

        idx = idxs[0]
        print(f"[INFO] Plotting {outcome} sample (index {idx})")

        shap_values = explainer(X_df.iloc[[idx]])

        #  Construct single-class SHAP explanation (class 1) 
        shap_value = shap.Explanation(
            values=shap_values.values[0][:, 1],           # SHAP values for positive class
            base_values=shap_values.base_values[0][1],    # scalar base value
            data=X_df.iloc[[idx]].values[0],
            feature_names=feature_names
        )

        #  Plot SHAP waterfall 
        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(shap_value, max_display=15, show=False)

        # Save plot 
        filename = f"{model_name}_local_{outcome}.png"
        out_path = os.path.join(OUTPUT_DIR, filename)
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved: {out_path}")

print("\nAll local SHAP plots successfully generated for Random Forest models.")
