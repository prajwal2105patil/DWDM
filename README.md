# Autonomous Diagnostic Warehouse (ADW)

This repository contains the full-stack, production-ready pipeline for the Autonomous Diagnostic Warehouse. It operates on a robust architecture featuring a native SQLite Data Warehouse, a serialized Machine Learning inference engine, and a live dynamically updating medical telemetry dashboard.

## 🚀 Execution Guide

To run this project from a completely fresh start, execute these commands in your terminal in this exact order:

### 1. Build the Database (Data Warehouse layer)
First, we must construct the relational database schema and inject our raw historical patient records and web server logs.
```bash
python scripts/07_db_init.py
python scripts/08_db_seed.py
```

### 2. Run the Data Engine (ETL and Machine Learning)
Next, we securely query the Warehouse to extract our training data, preventing "Target Leakage" by establishing strict Train/Test boundaries before standardizing variables. Once cleaned, we train our robust Random Forest algorithm and serialize its "brain" for the Web App.
```bash
python scripts/01_pipeline.py
python scripts/03_predictive.py
```

### 3. Launch the Backend API Server
With the database seeded and the ML Engine (`.pkl`) fully serialized in your `outputs/` folder, start the live Flask server.
```bash
python app.py
```
*(Leave this terminal window open running in the background).*

### 4. Open the Interface
Finally, open the `index.html` file natively in your web browser. 

The dashboard is fully wired to your `app.py` server. Adjust the Patient Telemetry sliders and click "RUN DIAGNOSTIC" to watch your live Machine Learning model dynamically classify the patient on the cluster grid!
