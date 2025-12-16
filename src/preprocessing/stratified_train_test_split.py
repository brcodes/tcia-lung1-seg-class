import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import os

import hashlib
import json
from datetime import datetime, timezone
import platform

def file_sha256(path):
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def split_manifest(df, source_manifest, output_path, split_name, random_state):
    """
    Generate a clinically relevant JSON manifest for a given split.
    """
    # Counts and proportions
    counts = df["Overall.Stage"].value_counts().sort_index().to_dict()
    proportions = (df["Overall.Stage"].value_counts(normalize=True)
                   .sort_index().round(4).to_dict())

    manifest = {
        "source_manifest": os.path.abspath(source_manifest),
        "output_path": os.path.abspath(output_path),
        "split_name": split_name,
        "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
        "random_state": random_state,
        "hashes": {
            "source_manifest_sha256": file_sha256(source_manifest),
            "output_split_sha256": file_sha256(output_path) if os.path.exists(output_path) else None
        },
        "patient_count": int(len(df)),
        "classification_labels": {
            "counts": counts,
            "proportions": proportions
        },
        "script_metadata": {
            "function": "patient_stratified_split",
            "version": "v1.0",
            "python_version": platform.python_version(),
            "pandas_version": pd.__version__
        }
    }

    # Save JSON alongside split file
    json_path = output_path.replace(".csv", f"_{split_name}_manifest.json")
    with open(json_path, "w") as f:
        json.dump(manifest, f, indent=4)

    print(f"[{split_name.upper()}] manifest written to {json_path}")
    return manifest

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
        
        # Generate JSON manifest
        split_manifest(subset, manifest_path, output_path, split, random_state)

if __name__ == "__main__":
    base = "../../data/"
    manifest_path = os.path.join(base, "raw", "NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv")
    output_path = os.path.join(base, "interim", "NSCLC-Radiomics-Lung1-Splits.csv")
    patient_stratified_split(manifest_path=manifest_path, output_path=output_path)
    
    