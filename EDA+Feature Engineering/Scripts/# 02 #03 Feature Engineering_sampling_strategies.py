import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTENC
import joblib

#  CONFIG 
RAW_CSV = r"C:\Users\49176\Desktop\Studium\Masterarbeit_3\Data\online_shoppers_intention.csv"
OUT_DIR = r"C:\Users\49176\Desktop\Studium\Masterarbeit_3\ModelData"
SEED = 42
TEST_SIZE = 0.25
TARGET = "Revenue"

CATEGORICAL = ["Month", "VisitorType", "Weekend", "OperatingSystems", "Browser", "Region", "TrafficType"]

#  LOAD RAW 
df = pd.read_csv(RAW_CSV).dropna().copy()
df[TARGET] = df[TARGET].astype(int)

X = df.drop(columns=[TARGET])
y = df[TARGET].values

numeric_cols = [c for c in X.select_dtypes(include=["float64", "int64"]).columns if c not in CATEGORICAL]
cat_cols = [c for c in CATEGORICAL if c in X.columns]
cat_indices = [X.columns.get_loc(c) for c in cat_cols]

#  SPLIT 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=TEST_SIZE, random_state=SEED
)

#  BUILD AND FIT PREPROCESSOR ON RAW TRAIN 
pre = ColumnTransformer([
    ("num", StandardScaler(), numeric_cols),
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), cat_cols)
], verbose_feature_names_out=False)

pre.fit(X_train)  # Fit only once, before any resampling
feature_names = pre.get_feature_names_out()

#  RAW (no resampling) 
X_train_raw_enc = pre.transform(X_train)
X_test_enc = pre.transform(X_test)

#  UNDERSAMPLING 
rus = RandomUnderSampler(random_state=SEED)
X_train_under, y_train_under = rus.fit_resample(X_train, y_train)
X_train_under_enc = pre.transform(pd.DataFrame(X_train_under, columns=X.columns))

#  SMOTENC 
smote_nc = SMOTENC(categorical_features=cat_indices, random_state=SEED)
X_train_smotenc, y_train_smotenc = smote_nc.fit_resample(X_train, y_train)
X_train_smotenc_enc = pre.transform(pd.DataFrame(X_train_smotenc, columns=X.columns))

#  SAVE ALL 
os.makedirs(OUT_DIR, exist_ok=True)

# RAW
pd.DataFrame(X_train_raw_enc, columns=feature_names).to_csv(os.path.join(OUT_DIR, "X_train_raw.csv"), index=False)
pd.Series(y_train).to_csv(os.path.join(OUT_DIR, "y_train_raw.csv"), index=False)

# UNDER
pd.DataFrame(X_train_under_enc, columns=feature_names).to_csv(os.path.join(OUT_DIR, "X_train_under.csv"), index=False)
pd.Series(y_train_under).to_csv(os.path.join(OUT_DIR, "y_train_under.csv"), index=False)

# SMOTENC
pd.DataFrame(X_train_smotenc_enc, columns=feature_names).to_csv(os.path.join(OUT_DIR, "X_train_smotenc.csv"), index=False)
pd.Series(y_train_smotenc).to_csv(os.path.join(OUT_DIR, "y_train_smotenc.csv"), index=False)

# TEST SET 
pd.DataFrame(X_test_enc, columns=feature_names).to_csv(os.path.join(OUT_DIR, "X_test.csv"), index=False)
pd.Series(y_test).to_csv(os.path.join(OUT_DIR, "y_test.csv"), index=False)

# Save preprocessor for reuse 
joblib.dump(pre, os.path.join(OUT_DIR, "preprocessor.pkl"))

print("[All datasets generated with consistent feature names.")
print(f"Data saved to: {OUT_DIR}")
