Clinical‑Grade DICOM PHI De‑Identification & Audit Pipeline

A modular, deterministic, audit‑ready workflow for CT/SEG/RTSTRUCT imaging datasets
Overview

This repository implements a clinical‑grade DICOM de‑identification and audit pipeline designed for:

    HIPAA Safe Harbor compliance

    DICOM PS3.15 Appendix E Basic Attribute Confidentiality Profile

    Deterministic UID remapping

    CT geometry extraction for ML preprocessing

    SEG/RTSTRUCT → CT linkage preservation

    Fully reproducible, audit‑ready dataset creation

The pipeline is modular, reviewer‑friendly, and structured around a four‑audit architecture that keeps outputs lean, interpretable, and aligned with clinical IT expectations.
Key Features

    Deterministic salted UID remapping (non‑reversible)

    PHI removal + PS3.15 rule application

    Burned‑in text OCR scan (OpenCV + Tesseract)

    CT geometry extraction for preprocessing decisions

    SEG/RTSTRUCT reference normalization

    Filetree renaming synchronized to new UIDs

    Four separate audit artifacts for clarity and compliance

    Parquet‑based metadata for ML reproducibility

    JSONL‑based per‑instance de‑ID logs for legal defensibility

Architecture

The pipeline is organized into two layers:

    PHI De‑ID Module (scripts/phi_deid_pipeline/)

        Performs steps 1–6 of the de‑identification workflow

        Emits structured data for all audits

        Does not write audit files directly

    Audit Assembly Layer (main.py)

        Aggregates instance‑level outputs

        Writes the four audit artifacts

        Produces a final summary

This separation keeps the PHI module focused and the audits clean.
Four‑Audit Architecture

The pipeline produces four audit artifacts, each with a distinct purpose and schema.
1. De‑ID Audit

File: audits/deid/deid_audit.jsonl  
Granularity: per‑instance
Purpose: HIPAA + PS3.15 compliance evidence

Each line contains:

    tags_removed

    tags_cleaned

    private_tags_stripped

    burned_in_text_scan

    checksum

    input_path / output_path

    script_version

    timestamp

This is the legal audit trail.
2. UID Mapping Audit

File: audits/uid/uid_map.parquet  
Granularity: per UID
Purpose: deterministic identity graph

Columns:

    old_uid

    new_uid

    uid_type (STUDY / SERIES / SOP / FRAME_OF_REFERENCE / etc.)

    parent_uid

    sop_class_uid

Used for:

    file renaming

    SEG/RTSTRUCT linkage validation

    dataset lineage

3. Metadata Audit

File: audits/metadata/metadata_audit.parquet  
Granularity: per SOPInstanceUID
Purpose: ML reproducibility metadata

Includes:

    modality

    body_part_examined

    series_description

    manufacturer

    slice_thickness

    pixel_spacing

    orientation

    rows/columns

    bits_stored / bits_allocated

This is the dataset‑level metadata table used by ML pipelines.
4. CT Preprocessing Audit

File: audits/ct_prepro/ct_prepro_audit.json  
Granularity: per CT series
Purpose: preprocessing decisions + geometry summary

Includes:

    num_slices

    slice_thickness

    pixel_spacing

    z_spacing_mean / std

    orientation_matrix

    volume_extent_mm

    is_uniform_spacing

    requires_resampling

    recommended_preprocessing_pipeline

This is the audit that explains why your preprocessing pipeline behaves the way it does.
De‑Identification Workflow

The PHI module implements the following steps:

    Ingest & Inventory

        Read DICOMs

        Extract identity UIDs

        Extract referenced UIDs

        Build UID mapping table

    HIPAA Safe Harbor PHI Removal

        Remove identifiers

        Clean dates

        Remove free‑text comments

    "DICOM PS3.15 Appendix E"

        Apply Basic Attribute Confidentiality Profile (PS3.15 is Copyrighted)

        Remove, clean, or zero attributes based on local policy

        Strip all private tags

    Deterministic UID Remapping

        Salted SHA‑256 hashing

        Replace identity tags

        Replace all references (SEG, RTSTRUCT, etc.)

    Burned‑In Text Scan

        OCR detection

        Flagging for QC

    Write De‑Identified DICOMs

        Save to output tree

        Compute SHA‑256 checksum

The PHI module emits structured data for all audits but does not write audit files.
Modular Code Organization
Code

scripts/
  phi_deid_pipeline/
    pipeline.py              # orchestrates steps 1–6
    phi_rules.py             # HIPAA Safe Harbor tag lists
    ps3_15_rules.py          # loads/validates PS3.15 JSON rules
    uid_mapping.py           # deterministic UID hashing + reference replacement
    burned_in_text.py        # OCR logic
    metadata_extract.py      # metadata + CT geometry extraction
    dicom_utils.py           # shared helpers (read/write/checksum)

  audit_writers.py           # writers for the four audits
  main.py                    # top-level orchestrator

This structure keeps:

    policy separate from code

    logic separate from orchestration

    audits separate from de‑ID

Reviewers love this.
Running the Pipeline
Code

python scripts/main.py

Outputs will appear under:
Code

audits/
  deid/
  uid/
  metadata/
  ct_prepro/

PS3.15 Rules

The PS3.15 Basic Attribute Confidentiality Profile is encoded in:
Code

ps3_15_rules.json

This file defines:

    "remove" → delete

    "clean" → blank or normalize

    "zero" → neutralize

This keeps the policy transparent and editable.
Intended Use

This pipeline is designed for:

    Clinical ML research

    Model training on de‑identified CT/SEG/RTSTRUCT datasets

    Epic App Orchard / EHR integration

    Regulatory‑style auditability

    Recruiter‑friendly portfolio demonstration

It is not a replacement for a commercial de‑identification appliance, but it follows the same architectural principles.
License

You may include your preferred license here.