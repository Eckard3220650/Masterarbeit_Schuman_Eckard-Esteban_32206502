import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.manifold import TSNE
import numpy as np
import os
from math import sqrt
from scipy.stats import norm, chi2_contingency, mannwhitneyu

#  CONFIGURATION 
plt.style.use("ggplot")
sns.set_palette("dark")
pd.set_option("display.float_format", lambda x: f"{x:,.4f}")

#  PATHS 
INPUT_PATH = r"C:\Users\49176\Desktop\Studium\Masterarbeit_3\Data\online_shoppers_intention.csv"
OUTPUT_DIR = r"C:\Users\49176\Desktop\Studium\Masterarbeit_3\EDA+Feature Engineering"
os.makedirs(OUTPUT_DIR, exist_ok=True)

#  LOAD DATA 
df = pd.read_csv(INPUT_PATH)
LABEL_COL = "Revenue"
df[LABEL_COL] = df[LABEL_COL].astype(int)

#  DEFINE FEATURE TYPES 
int_categoricals = ["OperatingSystems", "Browser", "Region", "TrafficType"]
categoricals = ["Month", "VisitorType", "Weekend"] + int_categoricals
numeric_cols = [c for c in df.select_dtypes(include=["float64", "int64"]).columns
                if c not in ([LABEL_COL] + int_categoricals)]

# BASIC STATS 
print("Shape:", df.shape)
print("Missing values:\n", df.isnull().sum())
print("Revenue distribution (%):\n", df[LABEL_COL].value_counts(normalize=True) * 100)

#  CLASS IMBALANCE PLOT 
plt.figure(figsize=(5, 4))
ax = sns.countplot(x=LABEL_COL, data=df)
for p in ax.patches:
    count = int(p.get_height())
    ax.annotate(f"{count:,}", (p.get_x() + p.get_width()/2., p.get_height()),
                ha="center", va="bottom", fontsize=10, color="black")
plt.title("Revenue (Purchase) Distribution")
plt.xlabel("Revenue (1 = Purchase, 0 = No Purchase)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "revenue_distribution.png"))
plt.close()

#  SCALING NUMERIC FEATURES (ONLY CONTINUOUS) 
scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[numeric_cols] = scaler.fit_transform(df[numeric_cols])

#  FEATURE DISTRIBUTIONS WITH EFFECT SIZE 
def cliffs_delta(a, b):
    """Vargha–Delaney A12 & Cliff’s delta."""
    n1, n2 = len(a), len(b)
    U, _ = mannwhitneyu(a, b, alternative="two-sided")
    A12 = U / (n1 * n2)
    delta = 2 * A12 - 1
    return A12, delta

for feature in ["ProductRelated", "BounceRates", "PageValues"]:
    plt.figure(figsize=(7, 4))
    sns.boxplot(x=LABEL_COL, y=feature, data=df)
    a12, delta = cliffs_delta(df.loc[df[LABEL_COL]==0, feature],
                              df.loc[df[LABEL_COL]==1, feature])
    plt.title(f"{feature} by Revenue (A12={a12:.3f}, Δ={delta:.3f})")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{feature}_by_revenue.png"))
    plt.close()

#  CORRELATION HEATMAP (NUMERIC ONLY) 
corr_df = df_scaled[numeric_cols + [LABEL_COL]]
plt.figure(figsize=(14, 10))
sns.heatmap(corr_df.corr(), cmap="coolwarm", annot=False)
plt.title("Correlation Matrix (Continuous Numeric Features)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap_numeric.png"))
plt.close()

#  NUMERIC vs LABEL CORRELATIONS 
corr_series = corr_df.corr()[LABEL_COL].drop(LABEL_COL).sort_values(key=abs, ascending=False)
corr_series.to_csv(os.path.join(OUTPUT_DIR, "numeric_vs_label_correlation.csv"))
print("\nTop correlated numeric features:\n", corr_series.head(10))

#  CATEGORICAL ASSOCIATIONS (Cramér’s V)
def cramers_v(x, y):
    table = pd.crosstab(x, y)
    chi2, p, dof, exp = chi2_contingency(table, correction=False)
    n = table.sum().sum()
    phi2 = chi2 / n
    r, k = table.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1))) if min((kcorr-1), (rcorr-1)) > 0 else np.nan

cat_assoc = {col: cramers_v(df[col].astype(str), df[LABEL_COL]) for col in categoricals if col in df.columns}
pd.Series(cat_assoc).sort_values(ascending=False).to_csv(os.path.join(OUTPUT_DIR, "categorical_vs_label_cramersV.csv"))

#  CONVERSION RATE PLOTS (WITH WILSON CI) 
def wilson_ci(k, n, conf=0.95):
    z = norm.ppf(1 - (1 - conf) / 2)
    phat = k / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2*n)) / denom
    half = (z * sqrt((phat*(1-phat) + z**2/(4*n))/n)) / denom
    return center - half, center + half

def plot_conversion(df, col):
    rates = df.groupby(col)[LABEL_COL].agg(["mean", "sum", "count"]).reset_index()
    rates["ci_low"], rates["ci_high"] = zip(*[wilson_ci(row["sum"], row["count"]) for _, row in rates.iterrows()])
    rates["err_low"] = rates["mean"] - rates["ci_low"]
    rates["err_high"] = rates["ci_high"] - rates["mean"]
    plt.figure(figsize=(10, 4))
    plt.bar(rates[col].astype(str), rates["mean"],
            yerr=[rates["err_low"], rates["err_high"]], capsize=4)
    for i, n in enumerate(rates["count"]):
        plt.text(i, rates["mean"].iloc[i] + 0.005, f"n={n}", ha="center", fontsize=8)
    plt.title(f"Conversion Rate by {col}")
    plt.ylabel("Mean Revenue (Conversion Rate)")
    plt.xlabel(col)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"conversion_by_{col.lower()}.png"))
    plt.close()

for col in ["Month", "Weekend", "Region", "OperatingSystems", "Browser", "TrafficType", "VisitorType"]:
    if col in df.columns:
        plot_conversion(df, col)

#  OPTIONAL T-SNE (NUMERIC + ONE-HOT CAT)
cat_cols = [c for c in categoricals if c in df.columns]
pre = ColumnTransformer([
    ("num", StandardScaler(), numeric_cols),
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), cat_cols)
], verbose_feature_names_out=False)
X = pre.fit_transform(df[cat_cols + numeric_cols])
sample_size = min(300, X.shape[0])
idx = np.random.choice(X.shape[0], size=sample_size, replace=False)
X_sample = X[idx]
y_sample = df[LABEL_COL].iloc[idx]
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
emb = tsne.fit_transform(X_sample)
tsne_df = pd.DataFrame(emb, columns=["TSNE1", "TSNE2"])
tsne_df[LABEL_COL] = y_sample.values
plt.figure(figsize=(8, 6))
sns.scatterplot(data=tsne_df, x="TSNE1", y="TSNE2", hue=LABEL_COL, alpha=0.7, palette="Set1")
plt.title("t-SNE Clustering (One-Hot + Scaled Numeric)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "tsne_scatter_fixed.png"))
plt.close()
