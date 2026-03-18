import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import recall_score
from sklearn.decomposition import PCA
from mlxtend.frequent_patterns import apriori, association_rules

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / 'data'
OUTPUTS_DIR = PROJECT_DIR / 'outputs'

print("\n=== 1. Data Integrity Check (ETL) ===")
# Raw data zeros
df_raw = pd.read_csv(DATA_DIR / 'diabetes.csv')
zeros_insulin_before = (df_raw['Insulin'] == 0).sum()
zeros_skin_before = (df_raw['SkinThickness'] == 0).sum()

# Since 01_pipeline.py directly outputs scaled data to normalized.csv, we re-apply just the imputation to check the delta.
df_imputed = df_raw.copy()
zero_impute_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for col in zero_impute_cols:
    df_imputed[col] = df_imputed[col].replace(0, pd.NA)
    class_medians = df_imputed.groupby("Outcome")[col].transform("median")
    df_imputed[col] = df_imputed[col].fillna(class_medians)

zeros_insulin_after = (df_imputed['Insulin'] == 0).sum()
zeros_skin_after = (df_imputed['SkinThickness'] == 0).sum()

print(f"Insulin 0-values: {zeros_insulin_before} (Before) -> {zeros_insulin_after} (After). Delta = {zeros_insulin_before - zeros_insulin_after} corrected.")
print(f"SkinThickness 0-values: {zeros_skin_before} (Before) -> {zeros_skin_after} (After). Delta = {zeros_skin_before - zeros_skin_after} corrected.")


print("\n=== 2. Rule Significance Check (Association) ===")
df_binned = pd.read_csv(DATA_DIR / 'binned.csv')
df_encoded = pd.get_dummies(df_binned)
df_encoded = (df_encoded > 0).astype(bool)
frequent_itemsets = apriori(df_encoded, min_support=0.1, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.5)
target_col = 'Outcome_True' if 'Outcome_True' in df_encoded.columns else 'Outcome_1'
if target_col not in df_encoded.columns:
    target_col = 'Outcome'
diabetes_rules = rules[rules['consequents'].apply(lambda x: target_col in str(x))]
highest_rules = diabetes_rules.sort_values(by='lift', ascending=False)
for _, row in highest_rules.head(3).iterrows():
    ants = ", ".join(list(row['antecedents']))
    print(f"Rule: IF [{ants}] THEN Diabetes | Lift: {row['lift']:.3f} | Confidence: {row['confidence']:.3f}")


print("\n=== 3. Dimensionality Check (Clustering) ===")
df_norm = pd.read_csv(DATA_DIR / 'normalized.csv')
X = df_norm.drop(columns=['Outcome'])
pca = PCA(n_components=2)
pca.fit(X)
var_explained = pca.explained_variance_ratio_
total_var = sum(var_explained) * 100
print(f"PC1 explains {var_explained[0]*100:.2f}% of variance.")
print(f"PC2 explains {var_explained[1]*100:.2f}% of variance.")
print(f"Total variance explained by first two components: {total_var:.2f}%.")


print("\n=== 4. Clinical Safety Check (Classification) ===")
y = df_norm['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
rf_recall = recall_score(y_test, rf.predict(X_test))

svm = SVC(probability=True, random_state=42)
svm.fit(X_train, y_train)
svm_recall = recall_score(y_test, svm.predict(X_test))

print(f"Random Forest Recall: {rf_recall:.4f}")
print(f"SVM Recall:          {svm_recall:.4f}")
print(f"Higher Recall Model:  {'Random Forest' if rf_recall > svm_recall else 'SVM'}")

print("\n=== 5. Operational Check (Web Mining) ===")
with open(OUTPUTS_DIR / 'web_mining_metrics.json', 'r') as f:
    web_metrics = json.load(f)

peak_latency = web_metrics['Peak Hour Latency (ms)']
off_peak_latency = web_metrics['Off-Peak Latency (ms)']
requests = web_metrics['Peak Hour Requests']
status = web_metrics['Success Rate (%)']

print(f"Peak Hour Requests: {requests}")
print(f"Peak Hour Avg Latency: {peak_latency} ms")
print(f"Off-Peak Avg Latency:  {off_peak_latency} ms")
print(f"System Success Rate:   {status}%")
if peak_latency > off_peak_latency * 2:
    print("Conclusion: Strong correlation detected. Latency spikes massively indicating an architectural bottleneck under peak load.")

