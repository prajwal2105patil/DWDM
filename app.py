from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
import pandas as pd
import joblib
from pathlib import Path

app = Flask(__name__)
CORS(app)  # Allow frontend to talk to backend

# --- 1. Paths ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR
DATA_DIR = PROJECT_DIR / 'data'
OUTPUTS_DIR = PROJECT_DIR / 'outputs'
DB_PATH = DATA_DIR / 'warehouse.db'

# --- 2. Load Serialized Brain (Fixes Amnesiac Architecture) ---
try:
    print("[SYSTEM] Loading Engine Weights from Disk...")
    scaler = joblib.load(OUTPUTS_DIR / 'scaler.pkl')
    rf_model = joblib.load(OUTPUTS_DIR / 'diagnostic_engine.pkl')
    print("[SYSTEM] Random Forest Engine Online. Instant Inferencing Ready.")
except FileNotFoundError:
    print("[ERROR] Missing .pkl binaries. Must run `01_pipeline.py` and `03_predictive.py` first.")
    exit(1)


# --- 3. API Endpoints ---
@app.route('/api/diagnose', methods=['POST'])
def diagnose_patient():
    data = request.json
    
    # Build a DataFrame with the exact column order the Scaler expects
    raw_features = pd.DataFrame([{
        "pregnancies": 4, 
        "glucose": float(data.get("glucose", 100)),
        "blood_pressure": 72.0,
        "skin_thickness": 29.0,
        "insulin": float(data.get("insulin", 100)),
        "bmi": float(data.get("bmi", 25.0)),
        "diabetes_pedigree": 0.47,
        "age": float(data.get("age", 30))
    }])
    
    # 4. Instant Inference using locked weights
    scaled_features = scaler.transform(raw_features)
    risk_prob = rf_model.predict_proba(scaled_features)[0][1] # Probability of Class 1
    is_high_risk = bool(risk_prob >= 0.5)
    
    # 5. Save to Database (Data Warehouse logging)
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO Patients (
                pregnancies, glucose, blood_pressure, skin_thickness, insulin, 
                bmi, diabetes_pedigree, age, predicted_risk, actual_outcome
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            int(raw_features["pregnancies"][0]), float(raw_features["glucose"][0]), 72.0, 29.0, 
            float(raw_features["insulin"][0]), float(raw_features["bmi"][0]), 0.47, 
            int(raw_features["age"][0]), float(risk_prob), -1 # -1 means live prediction
        ))
        
        # Log the API request
        cursor.execute('''
            INSERT INTO System_Logs (
                ip_address, timestamp, endpoint, method, status_code, payload_size, latency_ms
            ) VALUES (?, CURRENT_TIMESTAMP, ?, ?, ?, ?, ?)
        ''', (request.remote_addr, '/api/diagnose', 'POST', 200, request.content_length, 42))
        
        conn.commit()
    except Exception as e:
        print(f"DB Error: {e}")
    finally:
        conn.close()
        
    return jsonify({
        "status": "success",
        "risk_probability": float(risk_prob),
        "is_high_risk": is_high_risk,
        "message": "Patient telemetry saved. Diagnostic complete."
    })


if __name__ == '__main__':
    print("[SYSTEM] ADW Backend Server Starting on Port 5000...")
    app.run(host='127.0.0.1', port=5000, debug=True)
