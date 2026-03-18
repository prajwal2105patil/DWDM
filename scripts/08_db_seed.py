import sqlite3
import pandas as pd
import re
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / 'data'
DB_PATH = DATA_DIR / 'warehouse.db'

def seed_database():
    print("Beginning Autonomous Diagnostic Warehouse Seeding protocol...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 1. Seed Patients Telemetry
    print("Seeding [Patients] data...")
    try:
        df_patients = pd.read_csv(DATA_DIR / 'diabetes.csv')
        # We assume 0 for predicted risk initially, actual_outcome is 'Outcome'
        # The schema requires specific columns:
        # pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age, predicted_risk, actual_outcome
        inserted_patients = 0
        for _, row in df_patients.iterrows():
            cursor.execute('''
                INSERT INTO Patients (
                    pregnancies, glucose, blood_pressure, skin_thickness, insulin, 
                    bmi, diabetes_pedigree, age, predicted_risk, actual_outcome
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                row['Pregnancies'], row['Glucose'], row['BloodPressure'], 
                row['SkinThickness'], row['Insulin'], row['BMI'], 
                row['DiabetesPedigreeFunction'], row['Age'], 
                -1.0, # -1 represents 'Not calculated yet'
                int(row['Outcome'])
            ))
            inserted_patients += 1
        print(f"-> Successfully inserted {inserted_patients} patient records.")
    except Exception as e:
        print(f"Error seeding patients: {e}")

    # 2. Seed System Logs
    print("Seeding [System_Logs] data...")
    try:
        log_pattern = re.compile(
            r'(?P<ip>\d+\.\d+\.\d+\.\d+) - - \[(?P<time>[^\]]+)\] "(?P<method>[A-Z]+) (?P<endpoint>[^ ]+) HTTP/1.1" (?P<status>\d{3}) (?P<payload>\d+) (?P<latency>\d+)'
        )
        
        inserted_logs = 0
        with open(DATA_DIR / 'api_logs.txt', 'r') as f:
            for line in f:
                match = log_pattern.match(line)
                if match:
                    d = match.groupdict()
                    # Convert time format "01/Oct/2023:14:30:00 +0000" into standard SQL datetime
                    # We can use pd.to_datetime safely
                    dt_val = pd.to_datetime(d['time'], format='%d/%b/%Y:%H:%M:%S %z').strftime('%Y-%m-%d %H:%M:%S')
                    
                    cursor.execute('''
                        INSERT INTO System_Logs (
                            ip_address, timestamp, endpoint, method, status_code, payload_size, latency_ms
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        d['ip'], dt_val, d['endpoint'], d['method'],
                        int(d['status']), int(d['payload']), int(d['latency'])
                    ))
                    inserted_logs += 1
        print(f"-> Successfully inserted {inserted_logs} API traffic logs.")
    except Exception as e:
        print(f"Error seeding logs: {e}")

    conn.commit()
    conn.close()
    print("Database seeding COMPLETE!")

if __name__ == "__main__":
    seed_database()
