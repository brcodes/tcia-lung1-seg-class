import pandas as pd
import json
import os
import hashlib
from datetime import datetime, timezone
import platform
from util import file_sha256

def write_cleaning_metadata_json(df, manifest_or_path, clean_path, random_state=42):
    """
    Generate one clinically relevant JSON manifest for the entire splits.csv,
    with nested Train/Test metadata.
    """
    # Prepare clean info for manifest, print
    global_clean_info = {
        "source_manifest": "dataframe in memory" if isinstance(manifest_or_path, pd.DataFrame) else os.path.abspath(manifest_or_path),
        "source_type": "DataFrame" if isinstance(manifest_or_path, pd.DataFrame) else "CSV",
        "clean_path": os.path.abspath(clean_path),
        "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
        "random_state": random_state,
        "hashes": {
            "source_manifest_sha256": file_sha256(manifest_or_path) if isinstance(manifest_or_path, str) else file_sha256(df),
            "cleaned_manifest_sha256": file_sha256(clean_path)
        },
        "script_metadata": {
            "function": "clean_patient_manifest_csv",
            "commit_hash": "ab1230c",
            "python_version": platform.python_version(),
            "pandas_version": pd.__version__
        }
    }
    # Save JSON alongside cleaned file
    json_path = clean_path.replace(".csv", ".json")
    with open(json_path, "w") as f:
        json.dump(global_clean_info, f, indent=4)


def clean_patient_manifest_csv(manifest_or_path, clean_path=None, keep_columns=None, dropna_columns=None, random_state=None, **kwargs):
    
    if clean_path is None:
        raise ValueError("clean_path must be specified to save cleaned manifest CSV.")
    
    # When clean is a helper
    if kwargs:
        helper = True
        # Dynamically get all argument names except 'kwargs'
        valid_kwargs = set(locals().keys()) - {"kwargs"}
        invalid_kwargs = set(kwargs) - valid_kwargs
        
        # valid_kwargs = {"manifest_or_path", "clean_path", "keep_columns", "dropna_columns"}
        # invalid_kwargs = set(kwargs) - valid_kwargs
        if invalid_kwargs:
            raise ValueError(f"Unexpected kwargs for clean_raw_manifest_csv: {invalid_kwargs}")
        # Override if present in kwargs
        manifest_or_path = kwargs.get("manifest_or_path", manifest_or_path)
        clean_path = kwargs.get("clean_path", clean_path)
        keep_columns = kwargs.get("keep_columns", keep_columns)
        dropna_columns = kwargs.get("dropna_columns", dropna_columns)
        
    # Load manifest
    if not isinstance(manifest_or_path, pd.DataFrame):
        # Check file presence
        if not os.path.exists(manifest_or_path):
            raise FileNotFoundError(f"Manifest path {manifest_or_path} does not exist.")
        df = pd.read_csv(manifest_or_path)
    else:
        df = manifest_or_path

    # Drop unwanted columns if
    if keep_columns is not None:
        missing = [col for col in keep_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Columns {missing} specified in keep_columns not found in manifest.csv")
        original_columns = list(df.columns)
        org_col_count = int(len(original_columns))
        df = df.loc[:, keep_columns]
        # Log remaining columns
        remaining_columns = list(df.columns)
        rem_col_count = int(len(remaining_columns))
        # Log dropped columns
        dropped_columns = [col for col in original_columns if col not in remaining_columns]
        dropped_columns_count = int(len(dropped_columns))
        print(f"{rem_col_count} Columns kept: {remaining_columns}")
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
            
    # Save cleaned manifest
    df.to_csv(clean_path, index=False)
    print(f"Cleaned manifest saved to {clean_path}")

    # Prepare clean info for manifest, print
    clean_info = {
        "columns": {
            "original": (org_col_count, original_columns),
            "kept": (rem_col_count, remaining_columns),
            "removed": (dropped_columns_count, dropped_columns)
        },
        "rows": {
            "before_drop": before_drop,
            "after_drop": after_drop,
            "dropped_total": dropped_total,
            "dropped_by_column": dropped_by_col
        }
    }
    
    global_clean_info = write_cleaning_metadata_json(df, manifest_or_path, clean_path, random_state=random_state)

    if helper is True:
        return df, global_clean_info
