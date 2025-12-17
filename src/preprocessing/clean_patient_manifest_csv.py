import pandas as pd
import json
import os
import hashlib
from datetime import datetime, timezone
import platform

def write_cleaning_metadata_json(df, source_manifest, output_path, splits_info, random_state):
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


def clean_manifest_csv(manifest_or_path, clean_path, keep_columns=None, dropna_columns=None, **kwargs):
    
    # When clean is a helper
    if kwargs:
        # Update this list to include other cleaning parameters as needed
        valid_kwargs = {"manifest_or_path", "clean_path", "keep_columns", "dropna_columns"}
        invalid_kwargs = set(kwargs) - valid_kwargs
        if invalid_kwargs:
            raise ValueError(f"Unexpected kwargs for clean_raw_manifest: {invalid_kwargs}")
        # Override if present in kwargs
        manifest_or_path = kwargs.get("manifest_or_path", manifest_or_path)
        clean_path = kwargs.get("clean_path", clean_path)
        keep_columns = kwargs.get("keep_columns", keep_columns)
        dropna_columns = kwargs.get("dropna_columns", dropna_columns)
        
    # Load manifest
    if not isinstance(manifest_or_path, pd.DataFrame):
        df = pd.read_csv(manifest_or_path)
    else:
        df = manifest_or_path

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
