"""
PHI de-identification module for clinical DICOM pipeline.

Implements:
[1] Ingest & inventory
[2] HIPAA Safe Harbor PHI removal
[3] DICOM PS3.15 Appendix E profile application
[4] Deterministic UID remapping
[5] Burned-in text scan (automated QC)
[6] Write de-identified DICOMs + checksums

This module does NOT write any audit files itself.
Instead, it calls the provided writer objects:
- deid_writer.log_instance(...)
- uid_map.add_uid(...)
- metadata.add_instance(...)

CT geometry fields are returned for later aggregation into
ct_prepro_audit.json by a separate module.
"""

import json
import os
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Iterable

import numpy as np
import pydicom
from pydicom.dataset import Dataset
from pydicom.datadict import tag_for_keyword
from pydicom.uid import generate_uid

import cv2
import pytesseract


# ----------------------------------------------------------------------
# Configuration containers
# ----------------------------------------------------------------------


@dataclass
class DeidConfig:
    """
    Configuration for PHI removal and UID remapping.
    """

    salt: str  # for deterministic salted UID hashing
    ps3_15_rules_path: Path  # JSON file with PS3.15 rules
    overwrite_existing_output: bool = False
    keep_tcai_uids_as_backup: bool = True  # store old UID in private tags if needed


@dataclass
class PathsConfig:
    input_root: Path
    output_root: Path


@dataclass
class Writers:
    """
    Wrapper for external audit writers.
    """
    deid_writer: Any            # DeidAuditWriter
    uid_map: Any                # UIDMapCollector
    metadata: Any               # MetadataAuditCollector


# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------


def iso_now() -> str:
    from datetime import datetime

    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def deterministic_uid(old_uid: str, salt: str) -> str:
    """
    Deterministic, salted hash -> pseudo UID.
    This is not a true DICOM UID root; in practice you may wrap this
    into your own root + hex digest.
    """
    h = hashlib.sha256()
    h.update(salt.encode("utf-8"))
    h.update(old_uid.encode("utf-8"))
    digest = h.hexdigest()[:24]
    # Example synthetic UID root; replace with your org's root if desired
    return f"1.2.826.0.1.3680043.10.543.{int(digest, 16)}"


def load_ps3_15_rules(path: Path) -> Dict[str, Any]:
    """
    Load local JSON encoding of DICOM PS3.15 Appendix E rules.

    Expected format (example):
    {
      "remove": ["PatientName", "PatientAddress", ...],
      "clean": ["StudyDate", "PatientBirthDate", ...],
      "zero": ["InstitutionName", ...]
    }
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ----------------------------------------------------------------------
# PHI removal / PS3.15 application
# ----------------------------------------------------------------------


def remove_phi_tags(ds: Dataset) -> Tuple[List[str], List[str]]:
    """
    Apply HIPAA Safe Harbor PHI removal:
    - Remove/blank direct identifiers and free-text comments.
    Returns (tags_removed, tags_cleaned) as lists of DICOM keyword strings.
    """
    tags_removed: List[str] = []
    tags_cleaned: List[str] = []

    # Remove/blank fields that should not exist
    # (These are examples; extend to your full policy)
    remove_keywords = [
        "PatientName",
        "PatientID",
        "IssuerOfPatientID",
        "OtherPatientIDs",
        "OtherPatientNames",
        "PatientAddress",
        "PatientTelephoneNumbers",
        "PatientBirthName",
        "SocialSecurityNumber",
        "AccessionNumber",
        "InstitutionName",
        "InstitutionAddress",
        "ReferringPhysicianName",
        "PerformingPhysicianName",
        "OperatorsName",
        "PatientMotherBirthName",
        "MedicalRecordLocator",
        "InsurancePlanIdentification",
        "PatientComments",
        "StudyID",
        "StudyInstanceUID",  # will be replaced later
        "SeriesInstanceUID",  # will be replaced later
        "SOPInstanceUID",  # will be replaced later
    ]

    clean_keywords = [
        "PatientBirthDate",
        "StudyDate",
        "SeriesDate",
        "AcquisitionDate",
        "ContentDate",
    ]

    for kw in remove_keywords:
        tag = tag_for_keyword(kw)
        if tag is None:
            # Skip unknown keywords to avoid pydicom warnings
            continue
        if tag in ds:
            del ds[tag]
            tags_removed.append(kw)

    # Example cleaning: keep year only, or blank entirely
    for kw in clean_keywords:
        tag = tag_for_keyword(kw)
        if tag is None:
            continue
        if tag in ds:
            original = str(ds.get(tag, ""))
            if original:
                # Blank date-like fields to avoid invalid VR warnings (e.g., "2014")
                ds[tag].value = ""
                tags_cleaned.append(kw)
            else:
                del ds[tag]
                tags_removed.append(kw)

    # Free-text comments — blanket removal (conservative)
    comment_keywords = [
        "ImageComments",
        "DerivationDescription",
        "AcquisitionComments",
        "ProtocolName",  # optional; may contain PHI
    ]
    for kw in comment_keywords:
        tag = tag_for_keyword(kw)
        if tag is None:
            continue
        if tag in ds:
            del ds[tag]
            tags_removed.append(kw)

    return tags_removed, tags_cleaned


def apply_ps3_15_rules(ds: Dataset, rules: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """
    Apply Basic Attribute Confidentiality Profile rules from local JSON.
    Returns (tags_removed, tags_cleaned) for PS3.15 step.
    """
    tags_removed: List[str] = []
    tags_cleaned: List[str] = []

    for kw in rules.get("remove", []):
        tag = tag_for_keyword(kw)
        if tag is None:
            continue
        if tag in ds:
            del ds[tag]
            tags_removed.append(kw)

    for kw in rules.get("clean", []):
        tag = tag_for_keyword(kw)
        if tag is None:
            continue
        if tag in ds:
            # Implement your specific cleaning logic; here we blank
            ds[tag].value = ""
            tags_cleaned.append(kw)

    for kw in rules.get("zero", []):
        tag = tag_for_keyword(kw)
        if tag is None:
            continue
        if tag in ds:
            # Zero numeric / length-based fields
            elem = ds[tag]
            try:
                if isinstance(elem.value, (int, float)):
                    elem.value = 0
                elif isinstance(elem.value, (list, tuple)):
                    elem.value = [0 for _ in elem.value]
                else:
                    elem.value = ""
                tags_cleaned.append(kw)
            except Exception:
                # Fallback: remove
                del ds[tag]
                tags_removed.append(kw)

    return tags_removed, tags_cleaned


def strip_private_tags(ds: Dataset) -> bool:
    """
    Strip all private tags.
    """
    original_count = len(ds)
    ds.remove_private_tags()
    return len(ds) < original_count


# ----------------------------------------------------------------------
# UID remapping (identity graph)
# ----------------------------------------------------------------------


@dataclass
class UIDs:
    patient_id: Optional[str]
    study_uid: str
    series_uid: str
    sop_uid: str
    frame_of_reference_uid: Optional[str]


def extract_uids(ds: Dataset) -> UIDs:
    return UIDs(
        patient_id=str(ds.get("PatientID", "")) if "PatientID" in ds else None,
        study_uid=str(ds.StudyInstanceUID),
        series_uid=str(ds.SeriesInstanceUID),
        sop_uid=str(ds.SOPInstanceUID),
        frame_of_reference_uid=str(ds.FrameOfReferenceUID) if "FrameOfReferenceUID" in ds else None,
    )


def normalize_uid_graph(
    ds: Dataset,
    uids: UIDs,
    cfg: DeidConfig,
    writers: Writers,
) -> UIDs:
    """
    Create deterministic salted-mapping for all identity UIDs and
    update both identity tags and all references.
    """
    uid_types = {}
    if uids.study_uid:
        uid_types[uids.study_uid] = "STUDY"
    if uids.series_uid:
        uid_types[uids.series_uid] = "SERIES"
    if uids.sop_uid:
        uid_types[uids.sop_uid] = "SOP"
    if uids.frame_of_reference_uid:
        uid_types[uids.frame_of_reference_uid] = "FRAME_OF_REFERENCE"

    old_to_new: Dict[str, str] = {}
    for old, t in uid_types.items():
        new = deterministic_uid(old, cfg.salt)
        old_to_new[old] = new

        parent_uid = None
        if t == "SERIES":
            parent_uid = uids.study_uid
        elif t == "SOP":
            parent_uid = uids.series_uid

        sop_class_uid = str(ds.get("SOPClassUID", "")) if t == "SOP" else None

        writers.uid_map.add_uid(
            old_uid=old,
            new_uid=new,
            uid_type=t,
            parent_uid=parent_uid,
            sop_class_uid=sop_class_uid,
        )

    # Replace identity tags
    if uids.study_uid:
        ds.StudyInstanceUID = old_to_new[uids.study_uid]
    if uids.series_uid:
        ds.SeriesInstanceUID = old_to_new[uids.series_uid]
    if uids.sop_uid:
        ds.SOPInstanceUID = old_to_new[uids.sop_uid]
    if uids.frame_of_reference_uid and "FrameOfReferenceUID" in ds:
        ds.FrameOfReferenceUID = old_to_new[uids.frame_of_reference_uid]

    # Replace referenced UIDs
    replace_referenced_uids(ds, old_to_new)

    return UIDs(
        patient_id=uids.patient_id,
        study_uid=old_to_new.get(uids.study_uid, uids.study_uid),
        series_uid=old_to_new.get(uids.series_uid, uids.series_uid),
        sop_uid=old_to_new.get(uids.sop_uid, uids.sop_uid),
        frame_of_reference_uid=old_to_new.get(
            uids.frame_of_reference_uid, uids.frame_of_reference_uid
        ),
    )


def replace_referenced_uids(ds: Dataset, mapping: Dict[str, str]) -> None:
    """
    Traverse common reference sequences and replace UIDs.
    This is not exhaustive but covers key CT/SEG/RTSTRUCT relationships.
    """

    def replace_value(value: Any) -> Any:
        s = str(value)
        return mapping.get(s, s)

    # Simple attributes
    for kw in ["ReferencedStudySequence", "ReferencedSeriesSequence"]:
        if kw in ds:
            for item in ds[kw]:
                if "StudyInstanceUID" in item:
                    item.StudyInstanceUID = replace_value(item.StudyInstanceUID)
                if "SeriesInstanceUID" in item:
                    item.SeriesInstanceUID = replace_value(item.SeriesInstanceUID)

    # Referenced SOPs
    for kw in [
        "ReferencedImageSequence",
        "SourceImageSequence",
        "ReferencedInstanceSequence",
        "DerivationImageSequence",
    ]:
        if kw in ds:
            for item in ds[kw]:
                if "ReferencedSOPInstanceUID" in item:
                    item.ReferencedSOPInstanceUID = replace_value(
                        item.ReferencedSOPInstanceUID
                    )

    # SEG/RTSTRUCT linkages: we mainly care about target series / images
    if ds.get("Modality", "") in ["SEG", "RTSTRUCT"]:
        for kw in [
            "ReferencedFrameOfReferenceSequence",
            "StructureSetROISequence",
            "ROIContourSequence",
        ]:
            if kw in ds:
                for item in ds[kw]:
                    if "FrameOfReferenceUID" in item:
                        item.FrameOfReferenceUID = replace_value(
                            item.FrameOfReferenceUID
                        )
                    for inner_kw in [
                        "ReferencedStudySequence",
                        "ReferencedSeriesSequence",
                        "ContourImageSequence",
                    ]:
                        if inner_kw in item:
                            for sub in item[inner_kw]:
                                if "StudyInstanceUID" in sub:
                                    sub.StudyInstanceUID = replace_value(
                                        sub.StudyInstanceUID
                                    )
                                if "SeriesInstanceUID" in sub:
                                    sub.SeriesInstanceUID = replace_value(
                                        sub.SeriesInstanceUID
                                    )
                                if "ReferencedSOPInstanceUID" in sub:
                                    sub.ReferencedSOPInstanceUID = replace_value(
                                        sub.ReferencedSOPInstanceUID
                                    )


# ----------------------------------------------------------------------
# Burned-in text scan (OpenCV + Tesseract)
# ----------------------------------------------------------------------


def scan_burned_in_text(ds: Dataset) -> Tuple[str, Optional[str]]:
    """
    Very conservative OCR scan for burned-in text.
    Returns (status, sample_text) where status is 'passed' or 'flagged'.
    sample_text is kept for debugging only (not to be released).
    """
    if "PixelData" not in ds:
        return "passed", None

    try:
        arr = ds.pixel_array.astype(np.float32)
    except Exception:
        return "passed", None

    # Basic windowing for CT to highlight visible text
    arr = arr - arr.min()
    if arr.max() > 0:
        arr = arr / arr.max()
    arr = (arr * 255).astype(np.uint8)

    if arr.ndim == 3:
        # take a central slice if volume
        arr2d = arr[arr.shape[0] // 2]
    else:
        arr2d = arr

    # Simple threshold to highlight text
    _, thresh = cv2.threshold(arr2d, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)

    # OCR
    ocr_text = pytesseract.image_to_string(thresh)
    ocr_text_clean = ocr_text.strip()

    if ocr_text_clean:
        return "flagged", ocr_text_clean[:200]
    else:
        return "passed", None


# ----------------------------------------------------------------------
# Metadata extraction for metadata audit + CT geometry
# ----------------------------------------------------------------------


def extract_metadata(ds: Dataset) -> Dict[str, Any]:
    """
    Extract non-PHI metadata for metadata_audit.parquet.
    """
    def get_or_none(kw: str):
        return str(ds.get(kw, "")) if kw in ds else None

    modality = str(ds.get("Modality", ""))

    slice_thickness = None
    if "SliceThickness" in ds:
        try:
            slice_thickness = float(ds.SliceThickness)
        except Exception:
            slice_thickness = None

    pixel_spacing = None
    if "PixelSpacing" in ds:
        try:
            vals = [float(v) for v in ds.PixelSpacing]
            if len(vals) == 2:
                pixel_spacing = vals
        except Exception:
            pixel_spacing = None

    orientation = None
    if "ImageOrientationPatient" in ds:
        try:
            orientation = [float(v) for v in ds.ImageOrientationPatient]
        except Exception:
            orientation = None

    position = None
    if "ImagePositionPatient" in ds:
        try:
            position = [float(v) for v in ds.ImagePositionPatient]
        except Exception:
            position = None

    rows = int(ds.Rows) if "Rows" in ds else None
    columns = int(ds.Columns) if "Columns" in ds else None
    bits_stored = int(ds.BitsStored) if "BitsStored" in ds else None
    bits_allocated = int(ds.BitsAllocated) if "BitsAllocated" in ds else None

    return {
        "sop_uid": str(ds.SOPInstanceUID),
        "series_uid": str(ds.SeriesInstanceUID),
        "study_uid": str(ds.StudyInstanceUID),
        "modality": modality,
        "body_part_examined": get_or_none("BodyPartExamined"),
        "series_description": get_or_none("SeriesDescription"),
        "manufacturer": get_or_none("Manufacturer"),
        "manufacturer_model_name": get_or_none("ManufacturerModelName"),
        "slice_thickness": slice_thickness,
        "pixel_spacing": pixel_spacing,
        "image_orientation_patient": orientation,
        "image_position_patient": position,
        "rows": rows,
        "columns": columns,
        "bits_stored": bits_stored,
        "bits_allocated": bits_allocated,
    }


def extract_ct_geometry(ds: Dataset) -> Dict[str, Any]:
    """
    Extract per-instance CT geometry fields (for later series-level aggregation).
    """
    meta = extract_metadata(ds)
    return {
        "series_uid": meta["series_uid"],
        "study_uid": meta["study_uid"],
        "slice_thickness": meta["slice_thickness"],
        "pixel_spacing": meta["pixel_spacing"],
        "image_position_patient": meta["image_position_patient"],
        "image_orientation_patient": meta["image_orientation_patient"],
    }


# ----------------------------------------------------------------------
# Core per-instance processing
# ----------------------------------------------------------------------


def process_instance(
    ds: Dataset,
    in_path: Path,
    out_base: Path,
    patient_id: str,
    orig_filename: str,
    cfg: DeidConfig,
    ps3_15_rules: Dict[str, Any],
    writers: Writers,
) -> Dict[str, Any]:
    """
    Process a single DICOM instance:
    - Ingest & inventory
    - HIPAA Safe Harbor PHI removal
    - PS3.15 profile application
    - Deterministic UID remapping
    - Burned-in text scan
    - Write de-identified DICOM + checksum
    - Emit audit records via writer objects

    Returns a dict with:
    - 'ct_geometry' (if modality is CT)
    """
    modality = str(ds.get("Modality", ""))

    # Ingest UIDs
    original_uids = extract_uids(ds)

    # PHI removal (Safe Harbor)
    sh_removed, sh_cleaned = remove_phi_tags(ds)

    # PS3.15 rules
    ps_removed, ps_cleaned = apply_ps3_15_rules(ds, ps3_15_rules)

    # Strip private tags
    strip_private_tags(ds)
    private_tags_stripped = True

    # Deterministic UID remapping
    new_uids = normalize_uid_graph(ds, original_uids, cfg, writers)

    # Burned-in text scan
    burned_in_status, ocr_sample = scan_burned_in_text(ds)

    # Write de-identified DICOM
    out_path = out_base / patient_id / new_uids.study_uid / new_uids.series_uid / orig_filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(out_path, write_like_original=False)
    checksum = sha256_file(out_path)

    # Metadata for metadata audit
    meta = extract_metadata(ds)
    writers.metadata.add_instance(
        sop_uid=meta["sop_uid"],
        series_uid=meta["series_uid"],
        study_uid=meta["study_uid"],
        modality=meta["modality"],
        body_part_examined=meta["body_part_examined"],
        series_description=meta["series_description"],
        manufacturer=meta["manufacturer"],
        manufacturer_model_name=meta["manufacturer_model_name"],
        slice_thickness=meta["slice_thickness"],
        pixel_spacing=meta["pixel_spacing"],
        image_orientation_patient=meta["image_orientation_patient"],
        image_position_patient=meta["image_position_patient"],
        rows=meta["rows"],
        columns=meta["columns"],
        bits_stored=meta["bits_stored"],
        bits_allocated=meta["bits_allocated"],
    )

    # De-ID audit log
    tags_removed = sh_removed + ps_removed
    tags_cleaned = sh_cleaned + ps_cleaned

    writers.deid_writer.log_instance(
        sop_uid=meta["sop_uid"],
        input_path=in_path,
        output_path=out_path,
        modality=meta["modality"],
        tags_removed=tags_removed,
        tags_cleaned=tags_cleaned,
        private_tags_stripped=private_tags_stripped,
        burned_in_text_scan=burned_in_status,
        checksum_sha256=checksum,
    )

    # CT geometry payload for later aggregation
    ct_geometry = None
    if modality == "CT":
        ct_geometry = extract_ct_geometry(ds)
        # Optionally, you can attach OCR info, etc., if needed for QC.

    return {
        "ct_geometry": ct_geometry,
        "ocr_sample": ocr_sample if burned_in_status == "flagged" else None,
    }


# ----------------------------------------------------------------------
# Orchestrator over a directory tree
# ----------------------------------------------------------------------


def iter_dicom_files(root: Path) -> Iterable[Path]:
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(".dcm"):
                yield Path(dirpath) / fn


def run_phi_deid_pipeline(
    paths_cfg: PathsConfig,
    cfg: DeidConfig,
    writers: Writers,
    dicom_paths: Optional[Iterable[Any]] = None,
) -> Dict[str, Any]:
    """
    Run PHI de-identification over a DICOM tree.

        This function:
        - Loads PS3.15 rules
        - Iterates DICOM files (walks input_root when dicom_paths is None; otherwise uses the provided list)
        - Applies process_instance
        - Returns:
                {
                    'ct_geometry_records': List[Dict],
                    'num_processed': int,
                    'num_failed': int
                }

    The ct_geometry_records list can be passed to a separate CT
    preprocessing audit module to build ct_prepro_audit.json.
    """
    ps3_15_rules = load_ps3_15_rules(cfg.ps3_15_rules_path)

    ct_geometry_records: List[Dict[str, Any]] = []
    num_processed = 0
    num_failed = 0

    if dicom_paths is None:
        in_paths_iter: Iterable[Path] = iter_dicom_files(paths_cfg.input_root)
    elif isinstance(dicom_paths, (str, Path)):
        in_paths_iter = [Path(dicom_paths)]
    else:
        in_paths_iter = [Path(p) for p in dicom_paths]

    max_error_reports = 5  # avoid flooding stdout

    for in_path in in_paths_iter:
        try:
            ds = pydicom.dcmread(in_path)
        except Exception as exc:
            num_failed += 1
            if num_failed <= max_error_reports:
                print(f"[read_fail] {in_path}: {exc}")
            continue

        patient_id = str(ds.get("PatientID", "")) or "anon"
        orig_filename = in_path.name

        try:
            result = process_instance(
                ds=ds,
                in_path=in_path,
                out_base=paths_cfg.output_root,
                patient_id=patient_id,
                orig_filename=orig_filename,
                cfg=cfg,
                ps3_15_rules=ps3_15_rules,
                writers=writers,
            )
            if result["ct_geometry"] is not None:
                ct_geometry_records.append(result["ct_geometry"])
            num_processed += 1
        except Exception as exc:
            num_failed += 1
            if num_failed <= max_error_reports:
                import traceback

                tb = traceback.format_exc(limit=1)
                print(f"[process_fail] {in_path}: {exc}\n{tb}")
            continue

    return {
        "ct_geometry_records": ct_geometry_records,
        "num_processed": num_processed,
        "num_failed": num_failed,
        "timestamp": iso_now(),
    }
