import pandas as pd
import json
import os
import hashlib
from datetime import datetime, timezone
import platform
from util import file_sha256


PHI_STATUS = {
    "containsPhi": False,
    "phiLevel": "none",
    "verificationSource": "TCIA NSCLC-Radiomics Lung1 de-identified public dataset",
    "notes": "Dataset is de-identified according to TCIA documentation."
}


def write_clean_manifest_metadata(df, manifest_or_path, clean_path, local_clean_info, id_column=None, random_state=None):
    """Write JSON manifest metadata sidecar for a cleaned patient cohort manifest CSV.

    The CSV is treated as the patient cohort manifest and this function
    writes processing metadata alongside it, following an Epic-style
    naming convention (manifest vs metadata).
    """
    print("running: write_clean_manifest_metadata")
    print("\n")

    source_is_dataframe = isinstance(manifest_or_path, pd.DataFrame)
    executed_at = datetime.now(timezone.utc).isoformat()

    # Basic dataset / manifest information
    raw_manifest_path = "dataframe in memory" if source_is_dataframe else os.path.abspath(manifest_or_path)
    clean_manifest_path = os.path.abspath(clean_path)

    # Column-level summary for cleaned manifest
    column_summary = {}
    for col in df.columns:
        series = df[col]
        column_summary[col] = {
            "dtype": str(series.dtype),
            "nonNullCount": int(series.notna().sum())
        }

    patients_info = local_clean_info.get("patients", {}) if isinstance(local_clean_info, dict) else {}

    metadata = {
        "dataset": {
            "datasetId": "NSCLC-Radiomics-Lung1",
            "description": "NSCLC-Radiomics Lung1 cohort manifest cleaning",
            "sourceSystem": "TCIA"
        },
        "sourceManifest": {
            "rawManifestPath": raw_manifest_path,
            "rawRowCount": int(patients_info.get("before_drop", 0)),
            "sourceType": "DataFrame" if source_is_dataframe else "CSV"
        },
        "cleanManifest": {
            "cleanManifestPath": clean_manifest_path,
            "rowCount": int(patients_info.get("after_drop", len(df))),
            "idColumn": [id_column],
            "columnSummary": column_summary,
            "hash": file_sha256(clean_path)
        },
        "processing": {
            "pipelineName": "clean_patient_manifest",
            "pipelineVersion": "ab123dd0",
            "executedAt": executed_at,
            "randomState": random_state,
            "pythonVersion": platform.python_version(),
            "pandasVersion": pd.__version__,
            "cleaningSummary": local_clean_info
        },
        "phiStatus": PHI_STATUS
    }

    # Save JSON alongside cleaned file
    json_path = clean_path.replace(".csv", ".metadata.json")
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"Cleaning manifest metadata written to {json_path}")
    print("\n")

    return metadata


def clean_patient_manifest_csv(manifest_or_path, clean_path=None, keep_columns=None, dropna_columns=None, id_column=None, random_state=None, **_):
    print("running: clean_patient_manifest_csv")
    print("\n")

    # Basic checks
    if clean_path is None:
        raise ValueError("clean_path must be specified to save cleaned manifest CSV.")
    if random_state is None:
        print("Random state is None. Non-random cleaning steps assumed.")
        
    # Load manifest
    if not isinstance(manifest_or_path, pd.DataFrame):
        # Check file presence
        if not os.path.exists(manifest_or_path):
            raise FileNotFoundError(f"Manifest path {manifest_or_path} does not exist.")
        df = pd.read_csv(manifest_or_path)
    else:
        df = manifest_or_path

    # Drop unwanted columns if requested
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
    else:
        original_columns = list(df.columns)
        org_col_count = int(len(original_columns))
        remaining_columns = original_columns
        rem_col_count = org_col_count
        dropped_columns = []
        dropped_columns_count = 0
    
    if dropna_columns is not None:
        missing = [col for col in dropna_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Columns {missing} specified in dropna_columns not found in manifest.csv")

        before_drop = len(df)
        na_counts_before = df[dropna_columns].isna().sum().to_dict()

        # Capture rows that will be dropped
        dropped_rows = df[df[dropna_columns].isna().any(axis=1)].where(pd.notna(df), None)

        # Collect PatientID + values from dropna_columns
        dropped_records = dropped_rows[[id_column] + dropna_columns].to_dict(orient="records")

        # NEW: Collect dropped records grouped by column
        dropped_by_col_records = {
            col: dropped_rows[dropped_rows[col].isna()][[id_column] + dropna_columns].to_dict(orient="records")
            for col in dropna_columns
        }

        df = df.dropna(subset=dropna_columns)

        after_drop = len(df)
        dropped_total = before_drop - after_drop
        na_counts_after = df[dropna_columns].isna().sum().to_dict()
        dropped_by_col = {col: na_counts_before[col] - na_counts_after[col] for col in dropna_columns}

        print(f"Num rows before dropping NAs: {before_drop}")
        print(f"Rows dropped due to NA in any of columns {dropna_columns}: {dropped_total}")
        print(f"Rows dropped due to NA by column: {dropped_by_col}")
        print(f"Num rows after dropping NAs: {after_drop}")
        print("Dropped records (PatientID + values from dropna_columns):")
        for rec in dropped_records:
            print(rec)

        print("\nDropped records grouped by column:")
        for col, recs in dropped_by_col_records.items():
            print(f"Column {col}:")
            for rec in recs:
                print(rec)
                
    # Save cleaned manifest
    df.to_csv(clean_path, index=False)
    print(f"Cleaned manifest saved to {clean_path}")
    print("\n")

    # Prepare clean info for manifest, print
    local_clean_info = {
        "columns": {
            "original": {
                "count": org_col_count,
                "names": original_columns
            },
            "kept": {
                "count": rem_col_count,
                "names": remaining_columns
            },
            "removed": {
                "count": dropped_columns_count,
                "names": dropped_columns
            }
        },
        "patients": {
            "before_drop": before_drop,
            "after_drop": after_drop,
            "dropped_total": dropped_total,
            "dropped_by_column": dropped_by_col,
            "dropped_identifiers": dropped_records,
            "dropped_identifiers_by_column": dropped_by_col_records
        }
    }

    cleaning_metadata = write_clean_manifest_metadata(
        df,
        manifest_or_path,
        clean_path,
        local_clean_info,
        id_column=id_column,
        random_state=random_state
    )

    return df, cleaning_metadata
