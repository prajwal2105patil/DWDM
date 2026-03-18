import sqlite3
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
OUTPUTS_DIR = SCRIPT_DIR.parent / "outputs"
DB_PATH = DATA_DIR / "warehouse.db"

def extract_from_warehouse() -> pd.DataFrame:
    """Read true patient records from the SQLite Data Warehouse."""
    conn = sqlite3.connect(DB_PATH)
    # Exclude the -1 outcomes because those are live API predictions, not ground truth.
    df = pd.read_sql_query("SELECT * FROM Patients WHERE actual_outcome != -1", conn)
    conn.close()
    
    # Drop warehouse-only metadata (id, timestamp, predicted_risk) before ML matching
    df = df.drop(columns=['id', 'timestamp', 'predicted_risk'])
    return df

def prevent_leakage_and_impute(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Calculate medians EXACTLY on the training set to prevent Target Leakage."""
    
    # Make copies to prevent Pandas SettingWithCopy warnings
    train_imp = train_df.copy()
    test_imp = test_df.copy()
    
    zero_impute_cols = ["glucose", "blood_pressure", "skin_thickness", "insulin", "bmi"]
    
    # 1. Replace 0 with pd.NA
    for col in zero_impute_cols:
        train_imp[col] = train_imp[col].replace(0, pd.NA)
        test_imp[col] = test_imp[col].replace(0, pd.NA)
        
    # 2. Calculate Medians ONLY from training data (grouped by actual_outcome for precision)
    for col in zero_impute_cols:
        # We calculate the median mapping (Outcome -> Median value) strictly from the Train set
        train_medians = train_imp.groupby("actual_outcome")[col].median()
        
        # We apply this mapping to both Train and Test
        # .transform('median') would calculate it per dataframe, which is leakage for Test
        train_imp[col] = train_imp.apply(
            lambda row: train_medians[row['actual_outcome']] if pd.isna(row[col]) else row[col], axis=1
        )
        test_imp[col] = test_imp.apply(
            lambda row: train_medians[row['actual_outcome']] if pd.isna(row[col]) else row[col], axis=1
        )
        
    return train_imp, test_imp

def scale_and_serialize(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Fit StandardScaler strictly on the training set and transform both."""
    train_scaled = train_df.copy()
    test_scaled = test_df.copy()
    
    feature_cols = [c for c in train_scaled.columns if c != "actual_outcome"]
    
    scaler = StandardScaler()
    # Fit purely on train to prevent variance leakage
    train_scaled[feature_cols] = scaler.fit_transform(train_scaled[feature_cols])
    # Apply exactly to test
    test_scaled[feature_cols] = scaler.transform(test_scaled[feature_cols])
    
    # Serialize the scaler for the live Web Backend
    OUTPUTS_DIR.mkdir(exist_ok=True)
    joblib.dump(scaler, OUTPUTS_DIR / 'scaler.pkl')
    print(f"-> Scaler Brain Serialized to {OUTPUTS_DIR / 'scaler.pkl'}")
    
    return train_scaled, test_scaled

def main() -> None:
    print("--- ETL Phase: Pulling from Data Warehouse ---")
    df = extract_from_warehouse()
    print(f"Extracted {len(df)} validated patients from Warehouse DB.")
    
    # 1. SPLIT FIRST (Cardinal Rule of Machine Learning)
    print("-> Performing 80/20 Train/Test Split...")
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['actual_outcome'])
    
    # 2. SEQUESTERED IMPUTATION
    print("-> Imputing missing data securely on Training Set to prevent leakage...")
    train_df_imp, test_df_imp = prevent_leakage_and_impute(train_df, test_df)
    
    # 3. SEQUESTERED SCALING
    print("-> Fitting StandardScaler on Training Set and Serializing...")
    train_scaled, test_scaled = scale_and_serialize(train_df_imp, test_df_imp)
    
    # 4. SAVE PIPELINE ARTIFACTS
    train_scaled.to_csv(DATA_DIR / "train_processed.csv", index=False)
    test_scaled.to_csv(DATA_DIR / "test_processed.csv", index=False)
    
    print("ETL COMPLETE: Leakage Eliminated. Data Saved.")

if __name__ == "__main__":
    main()
