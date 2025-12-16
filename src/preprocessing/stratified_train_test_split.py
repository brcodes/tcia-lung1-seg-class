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

def split_manifest(df, source_manifest, output_path, splits_info, random_state):
    """
    Generate one clinically relevant JSON manifest for the entire splits.csv,
    with nested Train/Test metadata.
    """
    patient_count = int(len(df))
    manifest = {
        "source_manifest": os.path.abspath(source_manifest),
        "output_path": os.path.abspath(output_path),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "random_state": random_state,
        "hashes": {
            "source_manifest_sha256": file_sha256(source_manifest),
            "output_split_sha256": file_sha256(output_path) if os.path.exists(output_path) else None
        },
        "script_metadata": {
            "function": "patient_stratified_split",
            "version": "v1.0",
            "python_version": platform.python_version(),
            "pandas_version": pd.__version__
        },
        "patient_count": patient_count,
        "splits": splits_info
    }
    
    # Save JSON alongside split file
    json_path = output_path.replace(".csv", ".json")
    with open(json_path, "w") as f:
        json.dump(manifest, f, indent=4)

    print(f"Split manifest written to {json_path}")
    return manifest

def patient_stratified_split(
    manifest_path="manifest.csv",
    output_path="splits.csv",
    train_size=0.8,
    random_state=42
):
    # Load manifest
    df = pd.read_csv(manifest_path)
    
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
    
    # Prepare splits info for manifest, print
    splits_info = {}
    for split in ["Train", "Test"]:
        subset = df[df["Split"] == split]
        counts = subset["Overall.Stage"].value_counts().sort_index().to_dict()
        proportions = (subset["Overall.Stage"].value_counts(normalize=True)
                       .sort_index().round(4).to_dict())
        splits_info[split] = {
            "patient_count": int(len(subset)),
            "classification_labels": {
                "counts": counts,
                "proportions": proportions
            }
        }
        print(f"{split} info: {splits_info[split]}")
    split_manifest(df, manifest_path, output_path, splits_info, random_state)

if __name__ == "__main__":
    base = "../../data/"
    manifest_path = os.path.join(base, "raw", "NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv")
    output_path = os.path.join(base, "interim", "NSCLC-Radiomics-Lung1-Splits.csv")
    patient_stratified_split(manifest_path=manifest_path, output_path=output_path)
    
    