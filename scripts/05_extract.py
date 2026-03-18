import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / 'data'
OUTPUTS_DIR = PROJECT_DIR / 'outputs'

# 1. Apriori Rules
df_binned = pd.read_csv(DATA_DIR / 'binned.csv')
df_encoded = pd.get_dummies(df_binned)
df_encoded = (df_encoded > 0).astype(bool)
frequent_itemsets = apriori(df_encoded, min_support=0.1, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.2)
# Filter for rules where confidence > 0.5
rules = rules[rules['confidence'] > 0.5]
# Get top 3
top_3_rules = rules.sort_values(by=['lift', 'confidence', 'support'], ascending=False).head(3)

apriori_md = "#### Apriori: Top 3 Rules\n| Antecedents | Consequents | Support | Confidence | Lift |\n|---|---|---|---|---|\n"
for _, row in top_3_rules.iterrows():
    ants = ", ".join(list(row['antecedents']))
    cons = ", ".join(list(row['consequents']))
    apriori_md += f"| {ants} | {cons} | {row['support']:.3f} | {row['confidence']:.3f} | {row['lift']:.3f} |\n"

# 2. Classification F1 and AUC
df_norm = pd.read_csv(DATA_DIR / 'normalized.csv')
X = df_norm.drop('Outcome', axis=1)
y = df_norm['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'SVM': SVC(probability=True, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42)
}

class_md = "#### Classification Models\n| Model | F1-Score | ROC-AUC |\n|---|---|---|\n"
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    class_md += f"| {name} | {f1:.4f} | {auc:.4f} |\n"

# 3. Clustering Silhouette Score
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X)
sil_score = silhouette_score(X, clusters)
cluster_md = f"#### Clustering\n- **K-Means (k=2) Silhouette Score**: {sil_score:.4f}\n"

# 4. Web Mining Metrics
metrics_path = OUTPUTS_DIR / 'web_mining_metrics.json'
with open(metrics_path, 'r') as f:
    web_metrics = json.load(f)

web_md = f"""#### Web Mining (Simulated API Log Analytics)
- **Peak Load Time**: {web_metrics['Peak Hour']}:00 - {web_metrics['Peak Hour']+1}:00 ({web_metrics['Peak Hour Requests']} requests)
- **System Reliability (Success Rate)**: {web_metrics['Success Rate (%)']}%
- **Cluster Locations Found**: {web_metrics['Total Clusters Found']}
- **Most Active Clinic Cluster**: {web_metrics['Most Active Cluster ID']}
"""

# Compile to Truth Table
truth_table = f"""### Metric Truth Table: Diabetes Prediction Pipeline

{apriori_md}
{class_md}
{cluster_md}
{web_md}
"""

table_path = OUTPUTS_DIR / 'metric_truth_table.md'
with open(table_path, 'w') as f:
    f.write(truth_table)

print(f"Metric Truth Table generated at {table_path}")
