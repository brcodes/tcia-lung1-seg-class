import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import os

def patient_stratified_split(
    manifest_path="manifest.csv",
    output_path="splits.csv",
    train_size=0.8,
    random_state=42
):
    # Load manifest
    df = pd.read_csv(manifest_path)
    
    print(df.head())

    # Ensure required columns exist
    if "PatientID" not in df.columns or "Overall.Stage" not in df.columns:
        raise ValueError("manifest.csv must contain 'PatientID' and 'Overall.Stage' columns")

    # Drop rows with missing stage
    df = df.dropna(subset=["Overall.Stage"]).copy()

    # Stratified split by stage
    splitter = StratifiedShuffleSplit(
        n_splits=1, train_size=train_size, random_state=random_state
    )
    y = df["Overall.Stage"]

    train_idx, test_idx = next(splitter.split(df, y))
    
    # Initialize split column
    df["Split"] = None
    # Assign using positional indices
    df.iloc[train_idx, df.columns.get_loc("Split")] = "Train"
    df.iloc[test_idx, df.columns.get_loc("Split")] = "Test"

    # Save splits
    df.to_csv(output_path, index=False)

    # Print class counts and proportions
    for split in ["Train", "Test"]:
        subset = df[df["Split"] == split]
        counts = subset["Overall.Stage"].value_counts().sort_index()
        proportions = counts / len(subset)
        print(f"\n[{split.upper()}] n={len(subset)}")
        print("Counts:\n", counts)
        print("Proportions:\n", proportions.round(4))

if __name__ == "__main__":
    base = "../../data/"
    manifest_path = os.path.join(base, "raw", "NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv")
    output_path = os.path.join(base, "interim", "NSCLC-Radiomics-Lung1-Splits.csv")
    patient_stratified_split(manifest_path=manifest_path, output_path=output_path)