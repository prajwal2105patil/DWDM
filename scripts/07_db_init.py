import sqlite3
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / 'data'
DB_PATH = DATA_DIR / 'warehouse.db'

def create_database():
    print(f"Initializing Autonomous Diagnostic Warehouse Database at: {DB_PATH}")
    
    # Connect (will create the file if it doesn't exist)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 1. Patients Table
    # Stores telemetry and diagnostic predictions
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Patients (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        pregnancies INTEGER,
        glucose REAL,
        blood_pressure REAL,
        skin_thickness REAL,
        insulin REAL,
        bmi REAL,
        diabetes_pedigree REAL,
        age INTEGER,
        predicted_risk REAL,
        actual_outcome INTEGER
    )
    ''')
    
    # 2. System Logs Table (Web Mining Output)
    # Stores the API hit information securely
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS System_Logs (
        log_id INTEGER PRIMARY KEY AUTOINCREMENT,
        ip_address TEXT,
        timestamp DATETIME,
        endpoint TEXT,
        method TEXT,
        status_code INTEGER,
        payload_size INTEGER,
        latency_ms INTEGER
    )
    ''')
    
    conn.commit()
    conn.close()
    
    print("Database tables `Patients` and `System_Logs` successfully constructed!")

if __name__ == "__main__":
    create_database()
