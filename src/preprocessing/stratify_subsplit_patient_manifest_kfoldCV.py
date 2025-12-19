import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import json
from datetime import datetime, timezone
import platform

from clean_patient_manifest_csv import clean_patient_manifest_csv, PHI_STATUS
from util import file_sha256


def write_stratified_subsplit_kfoldCV_metadata(
    df,
    source_manifest,
    output_path,
    clean_metadata,
    strat_column,
    k_folds,
    random_state,
    folds_info,
):
    """Write JSON manifest metadata for a stratified split patient cohort manifest CSV.

    The CSV with split assignments is treated as the patient cohort
    manifest; this function writes accompanying processing metadata.
    """
    print("running: write_stratified_subsplit_kfoldCV_metadata")
    print("\n")

    executed_at = datetime.now(timezone.utc).isoformat()

    source_manifest_path = os.path.abspath(source_manifest)
    split_manifest_path = os.path.abspath(output_path)

    # Build per-fold summary (each fold has a Train/Val split)
    folds = []
    for fold in folds_info:
        fold_index = fold.get("fold", None)
        fold_splits = []
        for split_name, info in (fold.get("splits", {}) or {}).items():
            label_stats = info.get("classification_labels", {})
            fold_splits.append({
                "name": split_name,
                "manifestPath": split_manifest_path,
                "rowCount": int(info.get("patient_count", 0)),
                "labelDistribution": {
                    "counts": label_stats.get("counts", {}),
                    "proportions": label_stats.get("proportions", {})
                }
            })

        folds.append({
            "index": int(fold_index) if fold_index is not None else None,
            "splits": fold_splits
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
            "strategy": "stratifiedKFold",
            "targetColumn": strat_column,
            "kFolds": int(k_folds),
            "fractions": {
                "train": float(k_folds - 1) / k_folds,
                "test": float(1.0 / k_folds)
            },
            "randomSeed": random_state,
            "shuffle": True
        },
        "folds": folds,
        "processing": {
            "pipelineName": "stratify_subsplit_patient_manifest_kfoldCV",
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


def stratify_subsplit_patient_manifest_kfoldCV(
    manifest_path="manifest.csv",
    output_path="splits.csv",
    strat_column="label",
    k_folds=5,
    random_state=42,
    **optional_clean_kwargs
):
    print("running: stratify_subsplit_patient_manifest_kfoldCV")
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
        
    # Train subset is where we compute CV fold assignments. We then write the
    # resulting fold columns back into the full manifest, leaving Test rows as NA.
    train_mask = df["Split"] == "Train"
    df_train = df.loc[train_mask].copy()

    y = df_train[strat_column]

    splitter = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=random_state)

    # Add fold membership columns to the full manifest (default NA).
    # Only Train rows will be populated.
    fold_columns = [f"cv_fold_{i}" for i in range(1, k_folds + 1)]
    for col in fold_columns:
        df[col] = pd.NA
        df_train[col] = "train"

    folds_info = []

    for fold_index, (train_idx, test_idx) in enumerate(splitter.split(df_train, y), start=1):
        fold_col = f"cv_fold_{fold_index}"
        # `split()` yields positional indices; map them to index labels for `.loc`.
        df_train.loc[df_train.index[test_idx], fold_col] = "val"

        # Per-fold split summary for metadata
        splits_info = {}
        for split_name, idx in [("train", train_idx), ("val", test_idx)]:
            subset = df_train.iloc[idx]
            counts = subset[strat_column].value_counts().sort_index().to_dict()
            proportions = (
                subset[strat_column]
                .value_counts(normalize=True)
                .sort_index()
                .round(4)
                .to_dict()
            )
            splits_info[split_name] = {
                "patient_count": int(len(subset)),
                "classification_labels": {
                    "counts": counts,
                    "proportions": proportions,
                },
            }

        folds_info.append({
            "fold": int(fold_index),
            "splits": splits_info,
        })

    # Write fold assignments back into the full manifest (Train rows only).
    # This keeps Test rows as NA.
    if len(df_train) > 0:
        df.loc[df_train.index, fold_columns] = df_train[fold_columns]

    # Save single cohort manifest with fold columns
    df.to_csv(output_path, index=False)
    print(f"Stratified KFold manifest saved to {output_path}")

    write_stratified_subsplit_kfoldCV_metadata(
        df,
        manifest_path,
        output_path,
        clean_info,
        strat_column,
        k_folds,
        random_state,
        folds_info,
    )

    print("\n")

    return True


if __name__ == "__main__":
    base = "../../data/"
    manifest_name = "NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.patient-manifest.cleaned.stratified-splitCVTEST.csv"
    raw_path = os.path.join(base, "interim", manifest_name)
    ## Will end with the same name
    # Authoritative source of all splits
    output_path = raw_path
    '''
    Cleaning
    '''
    # Cleaning before split; no clean if None
    clean = None
    optional_clean_kwargs = {'clean':None}
    if clean:
        output_path = output_path.replace(".csv", ".cleaned.csv")
        optional_clean_kwargs = {"keep_columns":["PatientID", "Overall.Stage"], 
                            "dropna_columns":["Overall.Stage"], 
                            "id_column":"PatientID",
                            "clean_path": output_path}
    '''
    Strat, K-Fold CV SubSplit
    '''
    strat_column = "Overall.Stage"
    # k 5 folds equivalent to 80% train, 20% test
    k = 5
    random_state = 42
    
    
    print("stratify_subsplit_patient_manifest_kfoldCV parameters:")
    print(f"Manifest path: {raw_path}")
    print(f"Clean: {clean}")
    print(f"Stratification column: {strat_column}")
    print(f"K (Folds): {k}")
    print(f"Random state: {random_state}")
    print(f"Output path: {output_path}")
    print(f"Optional clean kwargs: {optional_clean_kwargs if clean else 'None'}")
    print("\n")
    
    stratify_subsplit_patient_manifest_kfoldCV(
        manifest_path=raw_path,
        output_path=output_path,
        strat_column=strat_column,
        k_folds=k,
        random_state=random_state,
        optional_clean_kwargs=optional_clean_kwargs,
    )