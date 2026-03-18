# Diabetes Prediction Pipeline - Progress Documentation

## 1) Project Overview
This project sets up a clean ETL pipeline for the UCI Pima Indians Diabetes dataset.

Current objective completed:
- Load source data from `data/diabetes.csv`
- Impute impossible zero values in key medical columns using class-wise medians by `Outcome`
- Produce two transformed datasets:
  - Scaled numeric dataset
  - Discretized categorical dataset

---

## 2) Repository Structure

- `.env` : Local runtime configuration (not tracked by Git)
- `.env.example` : Shareable template for environment variables
- `.gitignore` : Ignore rules for virtual env, env files, cache, and tooling artifacts
- `.venv/` : Local Python virtual environment
- `requirements.txt` : Python dependency list
- `scripts/01_pipeline.py` : Main ETL pipeline script
- `data/diabetes.csv` : Raw input dataset
- `data/normalized.csv` : Output A (scaled)
- `data/binned.csv` : Output B (discretized)

---

## 3) Environment and Dependency Setup Completed

Virtual environment:
- Created and activated local `.venv`

Dependencies installed from `requirements.txt`:
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- mlxtend

Pinned requirements file currently contains:
- pandas>=2.0
- numpy>=1.24
- scikit-learn>=1.3
- matplotlib>=3.7
- seaborn>=0.13
- mlxtend>=0.23

---

## 4) ETL Logic Implemented in scripts/01_pipeline.py

### 4.1 Load Step
- Reads CSV from `data/diabetes.csv` using pandas.

### 4.2 Imputation Step
Columns treated for impossible zero values:
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI

Method used:
1. Convert 0 to missing values.
2. Compute median within each class of `Outcome`.
3. Fill missing values with the class-specific median.

### 4.3 Transformation A - Scaled DataFrame
- Identifies all numeric columns.
- Excludes `Outcome` from scaling.
- Applies StandardScaler to the remaining continuous feature columns.
- Saves as `data/normalized.csv`.

### 4.4 Transformation B - Discretized DataFrame
No scaling is applied.

Adds categorical bins with labels:
- BMI -> Underweight, Normal, Overweight, Obese
- Age -> Young Adult, Adult, Middle Aged, Senior
- Glucose -> Normal, Prediabetes, Diabetes

Saved as `data/binned.csv`.

---

## 5) Execution and Validation Status

Pipeline execution command (PowerShell with venv interpreter):
- `& ".\.venv\Scripts\python.exe" "scripts\01_pipeline.py"`

Execution result:
- Successful
- Output files generated and verified:
  - `data/normalized.csv`
  - `data/binned.csv`

No code diagnostics errors were reported for `scripts/01_pipeline.py`.

---

## 6) Configuration Files Status

### .env
Contains runtime settings:
- DATA_PATH=data/diabetes.csv
- RANDOM_STATE=42
- TEST_SIZE=0.2

### .env.example
Template copy of the same runtime settings for sharing.

### .gitignore
Configured to ignore:
- Python cache and build artifacts
- Virtual environments (`.venv`, `venv`, `env`, `ENV`)
- Local environment files (`.env`, `.env.*`, while keeping `.env.example`)
- Notebook, test, linter, and editor artifacts

---
