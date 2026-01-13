"""
main.py — Entry point for the PHI de-identification pipeline.

This script:
  1. Loads configuration
  2. Instantiates audit writers
  3. Runs the PHI de-id pipeline (steps 1–6)
  4. Writes:
       - deid_audit.jsonl
       - uid_map.parquet
       - metadata_audit.parquet
       - ct_prepro_audit.json (series-level geometry)
  5. Prints a summary

This is the top-level orchestrator for the four-audit architecture.
"""

from pathlib import Path
from collections import defaultdict

from audit_writers import (
    DeidAuditWriter,
    UIDMapCollector,
    MetadataAuditCollector,
    CTPreproAudit,
)

from get_dicom import get_dicom

from phi_deid_pipeline import (
    run_phi_deid_pipeline,
    DeidConfig,
    PathsConfig,
    Writers,
)


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

def build_config():
    """
    Central place to define paths and settings.
    Modify these for your environment.
    """
    script_root = Path(__file__).resolve().parent
    repo_root = script_root.parents[1]

    paths_cfg = PathsConfig(
        input_root=repo_root / "data" / "raw" / "NSCLC-Radiomics",
        output_root=repo_root / "data" / "de-id",
    )

    deid_cfg = DeidConfig(
        salt="YOUR_SALT_HERE",  # replace with a stable secret
        ps3_15_rules_path=script_root / "ps3_15_rules.json",
        overwrite_existing_output=False,
    )

    return repo_root, paths_cfg, deid_cfg


# ----------------------------------------------------------------------
# CT geometry aggregation (series-level)
# ----------------------------------------------------------------------

def aggregate_ct_geometry(ct_records):
    """
    Convert per-instance CT geometry into per-series summaries.
    This feeds ct_prepro_audit.json.
    """
    series_map = defaultdict(list)

    for rec in ct_records:
        series_uid = rec["series_uid"]
        series_map[series_uid].append(rec)

    aggregated = []

    for series_uid, items in series_map.items():
        # Extract study UID (consistent within series)
        study_uid = items[0]["study_uid"]

        # Collect z-positions if available
        z_positions = []
        for it in items:
            pos = it.get("image_position_patient")
            if pos and len(pos) == 3:
                z_positions.append(pos[2])

        z_positions = sorted(z_positions) if z_positions else []

        # Compute spacing stats
        if len(z_positions) >= 2:
            diffs = [z_positions[i+1] - z_positions[i] for i in range(len(z_positions)-1)]
            z_mean = sum(diffs) / len(diffs)
            z_std = (sum((d - z_mean)**2 for d in diffs) / len(diffs))**0.5
        else:
            z_mean = None
            z_std = None

        # Representative fields
        slice_thickness = items[0].get("slice_thickness")
        pixel_spacing = items[0].get("pixel_spacing")
        orientation = items[0].get("image_orientation_patient")

        # Volume extent
        if z_positions:
            z_extent = abs(z_positions[-1] - z_positions[0])
        else:
            z_extent = None

        aggregated.append({
            "series_uid": series_uid,
            "study_uid": study_uid,
            "num_slices": len(items),
            "slice_thickness": slice_thickness,
            "pixel_spacing": pixel_spacing,
            "z_spacing_mean": z_mean,
            "z_spacing_std": z_std,
            "orientation_matrix": orientation,
            "volume_extent_mm": [
                None if pixel_spacing is None else pixel_spacing[0] * 512,
                None if pixel_spacing is None else pixel_spacing[1] * 512,
                z_extent,
            ],
            "is_uniform_spacing": (z_std is not None and z_std < 0.05),
            "requires_resampling": (z_std is None or z_std >= 0.05),
            "recommended_preprocessing_pipeline": "resample_iso_1mm_clip_hu",
            "notes": None,
        })

    return aggregated


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    repo_root, paths_cfg, deid_cfg = build_config()

    dicom_selection = get_dicom(
        PatientID=["LUNG1-001"],
        StudyInstanceUID_index=1,
        SeriesInstanceUID_index=None,
        SeriesNumber=None,
        InstanceNumber=None,
        base_dir=str(paths_cfg.input_root),
    )

    if isinstance(dicom_selection, (str, Path)):
        dicom_paths = [Path(dicom_selection)]
    else:
        dicom_paths = [Path(p) for p in dicom_selection]

    if not dicom_paths:
        raise ValueError("get_dicom returned no paths; cannot run de-id pipeline.")

    print(f"Found {len(dicom_paths)} DICOMs to process via get_dicom().")

    audit_root = repo_root / "data" / "audits"
    for subdir in ["deid", "uid", "metadata", "ct_prepro"]:
        (audit_root / subdir).mkdir(parents=True, exist_ok=True)

    # Instantiate audit writers
    deid_writer = DeidAuditWriter(
        output_path=audit_root / "deid" / "deid_audit.jsonl",
        script_version="deid_pipeline_0.3.0",
        ps3_15_profile_version="Basic_Confidentiality_1.0",
    )

    uid_map = UIDMapCollector()
    metadata = MetadataAuditCollector()
    ct_prepro = CTPreproAudit(script_version="ct_geometry_0.1.0")

    writers = Writers(
        deid_writer=deid_writer,
        uid_map=uid_map,
        metadata=metadata,
    )

    # Run PHI de-identification pipeline
    result = run_phi_deid_pipeline(
        paths_cfg=paths_cfg,
        cfg=deid_cfg,
        writers=writers,
        dicom_paths=dicom_paths,
    )

    # Close deid JSONL writer
    deid_writer.close()

    # Write UID map + metadata Parquet
    uid_map.to_parquet(audit_root / "uid" / "uid_map.parquet")
    metadata.to_parquet(audit_root / "metadata" / "metadata_audit.parquet")

    # Aggregate CT geometry → write ct_prepro_audit.json
    aggregated = aggregate_ct_geometry(result["ct_geometry_records"])
    for series in aggregated:
        ct_prepro.add_series(**series)

    ct_prepro.to_json(audit_root / "ct_prepro" / "ct_prepro_audit.json")

    # Summary
    print("\n=== PHI De-ID Pipeline Complete ===")
    print(f"Processed: {result['num_processed']}")
    print(f"Failed:    {result['num_failed']}")
    print(f"CT series: {len(aggregated)}")
    print(f"Timestamp: {result['timestamp']}")
    print("Audit outputs written to ../data/audits/")
    print("===================================\n")


if __name__ == "__main__":
    main()
