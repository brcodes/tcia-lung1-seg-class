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
        "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
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
        "patient_count_source": patient_count,
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
    keep_columns=None,
    dropna_columns=None,
    train_size=0.8,
    random_state=42
):
    # Load manifest
    df = pd.read_csv(manifest_path)
    source_patient_count = int(len(df))
    source_column_count = int(len(df.columns))
    
    # keep_columns=["PatientID", "Overall.Stage"],
    # dropna_columns=["Overall.Stage"],
    
    # Drop unwanted columns if
    # Drop unwanted columns if
    if keep_columns is not None:
        missing = [col for col in keep_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Columns {missing} specified in keep_columns not found in manifest.csv")
        original_columns = list(df.columns)
        df = df.loc[:, keep_columns]
        # Log remaining columns
        remaining_columns = list(df.columns)
        remaining_columns_count = int(len(remaining_columns))
        # Log dropped columns
        dropped_columns = [col for col in original_columns if col not in remaining_columns]
        dropped_columns_count = int(len(dropped_columns))
        print(f"{remaining_columns_count} Columns kept: {remaining_columns}")
        print(f"{dropped_columns_count} Columns dropped: {dropped_columns}")

    # if dropna_columns is not None:
    #     missing = [col for col in dropna_columns if col not in df.columns]
    #     if missing:
    #         raise ValueError(f"Columns {missing} specified in dropna_columns not found in manifest.csv")

    #     before_drop = len(df)
    #     na_counts_before = df[dropna_columns].isna().sum().to_dict()

    #     # Capture rows that will be dropped
    #     dropped_rows = df[df[dropna_columns].isna().any(axis=1)]
    #     dropped_ids = dropped_rows.get("patientID", dropped_rows.index).tolist()

    #     df = df.dropna(subset=dropna_columns)

    #     after_drop = len(df)
    #     dropped_total = before_drop - after_drop
    #     na_counts_after = df[dropna_columns].isna().sum().to_dict()
    #     dropped_by_col = {col: na_counts_before[col] - na_counts_after[col] for col in dropna_columns}

    #     print(f"Num rows before dropping NAs: {before_drop}")
    #     print(f"Rows dropped due to NA in any of columns {dropna_columns}: {dropped_total}")
    #     print(f"Rows dropped due to NA by column: {dropped_by_col}")
    #     print(f"Num rows after dropping NAs: {after_drop}")
    #     print(f"Dropped patientIDs (or index if no patientID column): {dropped_ids}")
        
    if dropna_columns is not None:
        missing = [col for col in dropna_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Columns {missing} specified in dropna_columns not found in manifest.csv")

        before_drop = len(df)
        na_counts_before = df[dropna_columns].isna().sum().to_dict()

        # Capture rows that will be dropped
        dropped_rows = df[df[dropna_columns].isna().any(axis=1)]
        # Collect identifiers from *all* dropna_columns
        dropped_ids = dropped_rows[dropna_columns].to_dict(orient="records")

        df = df.dropna(subset=dropna_columns)

        after_drop = len(df)
        dropped_total = before_drop - after_drop
        na_counts_after = df[dropna_columns].isna().sum().to_dict()
        dropped_by_col = {col: na_counts_before[col] - na_counts_after[col] for col in dropna_columns}

        print(f"Num rows before dropping NAs: {before_drop}")
        print(f"Rows dropped due to NA in any of columns {dropna_columns}: {dropped_total}")
        print(f"Rows dropped due to NA by column: {dropped_by_col}")
        print(f"Num rows after dropping NAs: {after_drop}")
        print("Dropped identifiers (values from dropna_columns):")
        for rec in dropped_ids:
            print(rec)

        

    # Stratified split by stage
    splitter = StratifiedShuffleSplit(
        n_splits=1, train_size=train_size, random_state=random_state
    )
    y = df["Overall.Stage"]
    # Iterator
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
            "patient_count_split": int(len(subset)),
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
    keep_columns = ["PatientID", "Overall.Stage"]
    dropna_columns = ["Overall.Stage"]
    train_size = 0.8
    random_state = 42
    patient_stratified_split(manifest_path=manifest_path, 
                             output_path=output_path, 
                             keep_columns=keep_columns, 
                             dropna_columns=dropna_columns, 
                             train_size=train_size, 
                             random_state=random_state)
    