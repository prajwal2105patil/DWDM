### Metric Truth Table: Diabetes Prediction Pipeline

#### Apriori: Top 3 Rules
| Antecedents | Consequents | Support | Confidence | Lift |
|---|---|---|---|---|
| Age_Bin_Adult, Glucose_Bin_Diabetes | Outcome | 0.102 | 0.722 | 2.070 |
| Age_Bin_Adult, Glucose, Glucose_Bin_Diabetes | Outcome | 0.102 | 0.722 | 2.070 |
| Age_Bin_Adult, Glucose_Bin_Diabetes | Outcome, Glucose | 0.102 | 0.722 | 2.070 |

#### Classification Models
| Model | F1-Score | ROC-AUC |
|---|---|---|
| Random Forest | 0.8421 | 0.9336 |
| SVM | 0.7857 | 0.8883 |
| Decision Tree | 0.7895 | 0.8384 |

#### Clustering
- **K-Means (k=2) Silhouette Score**: 0.7044

#### Web Mining (Simulated API Log Analytics)
- **Peak Load Time**: 12:00 - 13:00 (60 requests)
- **System Reliability (Success Rate)**: 80.0%
- **Cluster Locations Found**: 3
- **Most Active Clinic Cluster**: 1

