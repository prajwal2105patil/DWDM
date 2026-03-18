import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Ensure output directory exists
os.makedirs('../outputs', exist_ok=True)

# ==========================================
# 1. Association Rules (binned.csv)
# ==========================================
print("--- Association Rule Mining ---")
# Load binned data
df_binned = pd.read_csv('../data/binned.csv')

# Convert to one-hot encoded format
# This expands categorical columns into boolean columns (0/1)
df_encoded = pd.get_dummies(df_binned)

# Ensure all values are binary (0 or 1) for apriori
df_encoded = (df_encoded > 0).astype(int)

# Run Apriori algorithm
# min_support=0.1 means the itemset must appear in at least 10% of transactions
frequent_itemsets = apriori(df_encoded, min_support=0.1, use_colnames=True)

# Generate rules
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)

# Filter for rules where the consequent is 'Outcome_1' (Diabetes = True)
# Note: get_dummies usually creates 'Outcome_1' if the original column was 'Outcome'
target_col = 'Outcome_1' 
diabetes_rules = rules[rules['consequents'].apply(lambda x: target_col in str(x))]

# Print top 5 rules sorted by Lift
top_5_rules = diabetes_rules.sort_values(by='lift', ascending=False).head(5)
print(f"Top 5 Association Rules for {target_col}:")
print(top_5_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
print("\n")

# ==========================================
# 2. Clustering (normalized.csv)
# ==========================================
print("--- K-Means Clustering ---")
# Load normalized data
df_norm = pd.read_csv('../data/normalized.csv')

# Drop 'Outcome' for unsupervised learning
X = df_norm.drop(columns=['Outcome'])

# Run K-Means (k=2)
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X)
print("Clustering complete.")

# ==========================================
# 3. Visualization (PCA)
# ==========================================
# Reduce dimensions to 2D for plotting
pca = PCA(n_components=2)
pca_components = pca.fit_transform(X)

# Create DataFrame for plotting
df_pca = pd.DataFrame(data=pca_components, columns=['PC1', 'PC2'])
df_pca['Cluster'] = clusters

# Plotting
plt.figure(figsize=(10, 7))
sns.scatterplot(
    x='PC1', y='PC2', 
    hue='Cluster', 
    palette='viridis', 
    data=df_pca, 
    alpha=0.7
)
plt.title('PCA - Diabetes Dataset Clusters (K=2)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

# Save the plot
plt.savefig('../outputs/pca_plot.png')
print("Visualization saved to '../outputs/pca_plot.png'")

plt.show()