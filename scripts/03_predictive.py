# scripts/03_predictive.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, ConfusionMatrixDisplay
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / 'data'
OUTPUTS_DIR = PROJECT_DIR / 'outputs'

def run_predictive_models():
    print("Initializing Predictive Pipeline...\n")

    # 1. Load the Data from Operator 1
    try:
        df = pd.read_csv(DATA_DIR / 'normalized.csv')
    except FileNotFoundError:
        print(f"CRITICAL ERROR: '{DATA_DIR / 'normalized.csv'}' not found.")
        print("Ensure Operator 1 has placed the file in the correct directory.")
        return

    # 2. Split Features and Target
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    # Split 80% Train, 20% Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Define the Models
    models = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42), # probability=True required for AUC
        'Decision Tree': DecisionTreeClassifier(random_state=42)
    }

    results = {}
    best_f1 = -1
    best_model_name = ""
    best_model_instance = None

    # 4. Train, Evaluate, and Format Table
    print(f"{'Model':<15} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'ROC-AUC':<10}")
    print("-" * 65)

    for name, model in models.items():
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Calculate Metrics
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        # Print Row
        print(f"{name:<15} | {precision:.4f}     | {recall:.4f}     | {f1:.4f}     | {auc:.4f}")

        # Track Best Model (Optimizing for F1-Score in medical datasets)
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_model_instance = model

    print("-" * 65)
    print(f"\nBest Performing Model selected for visualization: **{best_model_name}**\n")

    # 5. Generate and Save Confusion Matrix
    print(f"Generating Confusion Matrix for {best_model_name}...")
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay.from_estimator(best_model_instance, X_test, y_test, cmap='Blues', ax=ax)
    disp.ax_.set_title(f'Confusion Matrix: {best_model_name}')

    # Save to the predefined outputs folder
    save_path = OUTPUTS_DIR / 'conf_matrix.png'
    plt.savefig(save_path)
    print(f"SUCCESS: Image saved to '{save_path}'. Handoff ready for Operator 4.")

if __name__ == "__main__":
    run_predictive_models()
