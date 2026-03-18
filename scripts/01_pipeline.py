from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler


def impute_zero_values_by_outcome_median(df: pd.DataFrame) -> pd.DataFrame:
	"""Replace impossible zero values using class-wise (Outcome) medians."""
	df_imputed = df.copy()
	zero_impute_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

	for col in zero_impute_cols:
		df_imputed[col] = df_imputed[col].replace(0, pd.NA)
		class_medians = df_imputed.groupby("Outcome")[col].transform("median")
		df_imputed[col] = df_imputed[col].fillna(class_medians)

	return df_imputed


def create_scaled_dataframe(df: pd.DataFrame) -> pd.DataFrame:
	"""Scale all continuous numeric feature columns, keeping Outcome unchanged."""
	scaled_df = df.copy()
	numeric_cols = scaled_df.select_dtypes(include="number").columns.tolist()
	feature_cols = [c for c in numeric_cols if c != "Outcome"]

	scaler = StandardScaler()
	scaled_df[feature_cols] = scaler.fit_transform(scaled_df[feature_cols])
	return scaled_df


def create_binned_dataframe(df: pd.DataFrame) -> pd.DataFrame:
	"""Create categorical bins for BMI, Age, and Glucose without scaling."""
	binned_df = df.copy()

	binned_df["BMI_Bin"] = pd.cut(
		binned_df["BMI"],
		bins=[-float("inf"), 18.5, 24.9, 29.9, float("inf")],
		labels=["Underweight", "Normal", "Overweight", "Obese"],
		include_lowest=True,
	)

	binned_df["Age_Bin"] = pd.cut(
		binned_df["Age"],
		bins=[-float("inf"), 30, 45, 60, float("inf")],
		labels=["Young Adult", "Adult", "Middle Aged", "Senior"],
		include_lowest=True,
	)

	binned_df["Glucose_Bin"] = pd.cut(
		binned_df["Glucose"],
		bins=[-float("inf"), 99, 125, float("inf")],
		labels=["Normal", "Prediabetes", "Diabetes"],
		include_lowest=True,
	)

	return binned_df


def main() -> None:
	script_dir = Path(__file__).resolve().parent
	data_dir = script_dir.parent / "data"

	input_path = data_dir / "diabetes.csv"
	normalized_path = data_dir / "normalized.csv"
	binned_path = data_dir / "binned.csv"

	df = pd.read_csv(input_path)
	df_imputed = impute_zero_values_by_outcome_median(df)

	df_scaled = create_scaled_dataframe(df_imputed)
	df_binned = create_binned_dataframe(df_imputed)

	df_scaled.to_csv(normalized_path, index=False)
	df_binned.to_csv(binned_path, index=False)

	print(f"Saved scaled data to: {normalized_path}")
	print(f"Saved binned data to: {binned_path}")


if __name__ == "__main__":
	main()
