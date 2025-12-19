import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

import os
import json
from datetime import datetime, timezone
import platform

from clean_patient_manifest_csv import clean_patient_manifest_csv, PHI_STATUS
from util import file_sha256


def write_stratified_split_metadata(df, source_manifest, output_path, clean_metadata, splits_info, strat_column, train_size, random_state):
    """Write JSON manifest metadata for a stratified split patient cohort manifest CSV.

    The CSV with split assignments is treated as the patient cohort
    manifest; this function writes accompanying processing metadata.
    """
    print("running: write_stratified_split_metadata")
    print("\n")

    executed_at = datetime.now(timezone.utc).isoformat()

    source_manifest_path = os.path.abspath(source_manifest)
    split_manifest_path = os.path.abspath(output_path)

    # Build per-split summary in a more Epic-style structure
    splits = []
    for split_name, info in splits_info.items():
        label_stats = info.get("classification_labels", {})
        splits.append({
            "name": split_name,
            "manifestPath": split_manifest_path,
            "rowCount": int(info.get("patient_count", 0)),
            "labelDistribution": {
                "counts": label_stats.get("counts", {}),
                "proportions": label_stats.get("proportions", {})
            }
        })

    metadata = {
        "dataset": {
            "datasetId": "NSCLC-Radiomics-Lung1",
            "description": "NSCLC-Radiomics Lung1 stratified split cohort manifest",
            "sourceSystem": "TCIA"
        },
        "sourceManifest": {
            "cleanManifestPath": source_manifest_path,
            "rowCount": int(len(df)),
            "hash": file_sha256(source_manifest)
        },
        "splitConfig": {
            "strategy": "stratifiedHoldout",
            "targetColumn": strat_column,
            "fractions": {
                "train": float(train_size),
                "test": float(1.0 - train_size)
            },
            "randomSeed": random_state,
            "shuffle": True
        },
        "splits": splits,
        "processing": {
            "pipelineName": "stratify_split_patient_manifest",
            "pipelineVersion": "a123dd0",
            "executedAt": executed_at,
            "pythonVersion": platform.python_version(),
            "pandasVersion": pd.__version__,
            "cleaningMetadata": clean_metadata
        },
        "phiStatus": PHI_STATUS
    }

    # Save JSON alongside split file
    json_path = output_path.replace(".csv", ".metadata.json")
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"Split manifest metadata written to {json_path}")
    print("\n")

    return metadata

def stratify_split_patient_manifest_csv(
    manifest_path="manifest.csv",
    output_path="splits.csv",
    strat_column="label",
    train_size=0.8,
    random_state=42,
    **optional_clean_kwargs
):
    print("running: stratify_split_patient_manifest_csv")
    print("\n")
    
    # Load source cohort manifest
    df = pd.read_csv(manifest_path)
    # Optional preprocessing
    if optional_clean_kwargs and optional_clean_kwargs.get('clean', None) is not None:
        # Automatically registered as helper
        kwargs = optional_clean_kwargs['optional_clean_kwargs']
        df, clean_info = clean_patient_manifest_csv(df, **kwargs)
    else:
        clean_info = None
        
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

    # Save stratified cohort manifest with split assignments
    df.to_csv(output_path, index=False)
    print(f"Stratified split manifest saved to {output_path}")
    print("\n")
    
    # Prepare splits info for manifest metadata
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
    write_stratified_split_metadata(
        df,
        manifest_path,
        output_path,
        clean_info,
        splits_info,
        strat_column,
        train_size,
        random_state,
    )

if __name__ == "__main__":
    base = "../../data/"
    manifest_name = "NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.patient-manifest.csv"
    raw_path = os.path.join(base, "raw", manifest_name)
    # Cleaning before split; no clean if None
    clean = True
    strat_column = "Overall.Stage"
    train_size = 0.8
    random_state = 42
    
    output_path = os.path.join(base, "interim", manifest_name)
    if clean:
        output_path = output_path.replace(".csv", ".cleaned.csv")
        optional_clean_kwargs = {"keep_columns":["PatientID", "Overall.Stage"], 
                            "dropna_columns":["Overall.Stage"], 
                            "id_column":"PatientID",
                            "clean_path": output_path}
    output_path = output_path.replace(".csv", ".stratified-split.csv")
    
    print("stratify_split_patient_manifest_csv parameters:")
    print(f"Manifest path: {raw_path}")
    print(f"Clean: {clean}")
    print(f"Stratification column: {strat_column}")
    print(f"Train size: {train_size}")
    print(f"Random state: {random_state}")
    print(f"Output path: {output_path}")
    print(f"Optional clean kwargs: {optional_clean_kwargs if clean else 'None'}")
    print("\n")
    
    stratify_split_patient_manifest_csv(manifest_path=raw_path, 
                             output_path=output_path, 
                             strat_column = strat_column,
                             train_size=train_size, 
                             random_state=random_state,
                             optional_clean_kwargs=optional_clean_kwargs)