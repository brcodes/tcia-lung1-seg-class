import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

import os
import json
from datetime import datetime, timezone
import platform

from preprocessing.clean_patient_manifest_csv import clean_patient_manifest_csv
from util import file_sha256


def write_stratified_split_metadata_json(df, source_manifest, output_path, clean_info, splits_info, random_state):
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

def stratify_split_patient_manifest_csv(
    manifest_path="manifest.csv",
    output_path="splits.csv",
    strat_column="label",
    train_size=0.8,
    random_state=42,
    **optional_clean_kwargs
):
    # Load
    df = pd.read_csv(manifest_path)
    # Optional preprocessing
    if optional_clean_kwargs:
        # Explicit, Pythonic: only call if kwargs provided
        df = clean_patient_manifest_csv(df, optional_clean_kwargs)


    # Stratified split by stage
    splitter = StratifiedShuffleSplit(
        n_splits=1, train_size=train_size, random_state=random_state
    )
    y = df[strat_column]
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
        counts = subset[strat_column].value_counts().sort_index().to_dict()
        proportions = (subset[strat_column].value_counts(normalize=True)
                       .sort_index().round(4).to_dict())
        splits_info[split] = {
            "patient_count": int(len(subset)),
            "classification_labels": {
                "counts": counts,
                "proportions": proportions
            }
        }
        print(f"{split} info: {splits_info[split]}")
    write_stratified_split_metadata_json(df, manifest_path, output_path, clean_info, splits_info, random_state)

if __name__ == "__main__":
    base = "../../data/"
    manifest_name = "NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019-patient-manifest.csv"
    raw_path = os.path.join(base, "raw", manifest_name)
    # Cleaning before split; no clean if None
    optional_clean_kwargs = {"keep_columns":["PatientID", "Overall.Stage"], 
                        "dropna_columns":["Overall.Stage"], 
                        "clean_path": os.path.join(base, "interim", manifest_name.replace(".csv", ".cleaned.csv"))}
    strat_column = "Overall.Stage"
    train_size = 0.8
    random_state = 42
    
    output_path = os.path.join(base, "interim", manifest_name.replace(".csv", ".stratified-split.csv"))
    
    stratify_split_patient_manifest_csv(manifest_path=raw_path, 
                             output_path=output_path, 
                             strat_column = strat_column,
                             train_size=train_size, 
                             random_state=random_state,
                             optional_clean_kwargs=optional_clean_kwargs)