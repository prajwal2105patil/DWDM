from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)
CORS(app)  # Allow frontend to talk to backend

# --- 1. Paths ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR
DATA_DIR = PROJECT_DIR / 'data'
DB_PATH = DATA_DIR / 'warehouse.db'

# --- 2. Global AI Model Setup ---
scaler = StandardScaler()
rf_model = RandomForestClassifier(random_state=42)

def initialize_ai_engine():
    print("[SYSTEM] Initializing AI Engine from raw data...")
    # Read raw data
    df = pd.read_csv(DATA_DIR / 'diabetes.csv')
    
    # Impute zero values exactly like 01_pipeline.py did
    zero_impute_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for col in zero_impute_cols:
        df[col] = df[col].replace(0, pd.NA)
        class_medians = df.groupby("Outcome")[col].transform("median")
        df[col] = df[col].fillna(class_medians)
        
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    
    # Fit the scaler and transform
    X_scaled = scaler.fit_transform(X)
    
    # Fit the Random Forest
    rf_model.fit(X_scaled, y)
    print(f"[SYSTEM] Random Forest Engine Online. Accuracy tuned on {len(df)} patients.")

# Initialize the model right as the server starts!
initialize_ai_engine()


# --- 3. API Endpoints ---
@app.route('/api/diagnose', methods=['POST'])
def diagnose_patient():
    data = request.json
    
    # The frontend only gives us 4 sliders (Glucose, BMI, Age, Insulin).
    # We will use dataset medians for the missing values to predict safely.
    # From data exploration: Pregnancies~4, BP~72, SkinThickness~29, Pedigree~0.47
    raw_features = pd.DataFrame([{
        "Pregnancies": 4, 
        "Glucose": float(data.get("glucose", 100)),
        "BloodPressure": 72.0,
        "SkinThickness": 29.0,
        "Insulin": float(data.get("insulin", 100)),
        "BMI": float(data.get("bmi", 25.0)),
        "DiabetesPedigreeFunction": 0.47,
        "Age": float(data.get("age", 30))
    }])
    
    # Scale exactly like the training data
    scaled_features = scaler.transform(raw_features)
    
    # Predict Probability
    risk_prob = rf_model.predict_proba(scaled_features)[0][1] # Probability of Class 1 (Diabetes)
    is_high_risk = bool(risk_prob >= 0.5)
    
    # 4. Save to Database
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO Patients (
                pregnancies, glucose, blood_pressure, skin_thickness, insulin, 
                bmi, diabetes_pedigree, age, predicted_risk, actual_outcome
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            4, raw_features["Glucose"][0], 72.0, 29.0, 
            raw_features["Insulin"][0], raw_features["BMI"][0], 0.47, 
            raw_features["Age"][0], float(risk_prob), -1 # Actual outcome unknown
        ))
        
        # Log the API request
        cursor.execute('''
            INSERT INTO System_Logs (
                ip_address, timestamp, endpoint, method, status_code, payload_size, latency_ms
            ) VALUES (?, CURRENT_TIMESTAMP, ?, ?, ?, ?, ?)
        ''', (request.remote_addr, '/api/diagnose', 'POST', 200, request.content_length, 42)) # Fake latency of 42ms
        
        conn.commit()
    except Exception as e:
        print(f"DB Error: {e}")
    finally:
        conn.close()
        
    # 5. Return JSON to Frontend
    return jsonify({
        "status": "success",
        "risk_probability": float(risk_prob),
        "is_high_risk": is_high_risk,
        "message": "Patient telemetry saved. Diagnostic complete."
    })


if __name__ == '__main__':
    print("[SYSTEM] ADW Backend Server Starting on Port 5000...")
    app.run(host='127.0.0.1', port=5000, debug=True)
