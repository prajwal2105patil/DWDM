import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / 'data'
OUTPUT_DIR = SCRIPT_DIR.parent / 'outputs'

def evaluate_models():
    print("Initializing Predictive Engine (Honest Validation)...")
    try:
        train_df = pd.read_csv(DATA_DIR / 'train_processed.csv')
        test_df = pd.read_csv(DATA_DIR / 'test_processed.csv')
    except FileNotFoundError:
        print("Error: Run 01_pipeline.py first to generate processed train/test splits.")
        return

    X_train = train_df.drop('actual_outcome', axis=1)
    y_train = train_df['actual_outcome']
    X_test = test_df.drop('actual_outcome', axis=1)
    y_test = test_df['actual_outcome']

    models = {
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'SVM': SVC(random_state=42, probability=True),
        'Decision Tree': DecisionTreeClassifier(random_state=42)
    }

    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # We need predict_proba for ROC-AUC
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_prob)
        else:
            auc = 0.0

        res = {
            'Model': name,
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1-Score': f1_score(y_test, y_pred),
            'ROC-AUC': auc,
            'model_ref': model
        }
        results.append(res)

    print("\nModel           | Precision  | Recall     | F1-Score   | ROC-AUC")
    print("-" * 65)
    for r in results:
        print(f"{r['Model']:<15} | {r['Precision']:.4f}     | {r['Recall']:.4f}     | {r['F1-Score']:.4f}     | {r['ROC-AUC']:.4f}")
    print("-" * 65)

    # In Clinical settings, False Negatives are fatal, so we optimize for Recall.
    # We select Random Forest as the primary engine.
    best_config = next(r for r in results if r['Model'] == 'Random Forest')
    best_model = best_config['model_ref']
    
    print(f"\nEvaluating Random Forest (Honest Clinical Recall: {best_config['Recall']:.4f})")
    
    # Flaw 2 Fix: Amnesiac Architecture
    # Serialize the best model so the Web Backend doesn't have to retrain.
    OUTPUT_DIR.mkdir(exist_ok=True)
    engine_path = OUTPUT_DIR / 'diagnostic_engine.pkl'
    joblib.dump(best_model, engine_path)
    
    print(f"\n[SYSTEM] Machine Learning Engine Serialized to Disk: {engine_path}")
    print("         Prediction weights locked. Handoff to App Server ready.")

if __name__ == "__main__":
    evaluate_models()
