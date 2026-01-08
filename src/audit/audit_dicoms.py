import csv
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
import os
import sys
from dataclasses import dataclass


# Define a set of DICOM tags commonly considered PHI under HIPAA
PHI_TAGS = {
    (0x0010, 0x0010): "PatientName",
    (0x0010, 0x0030): "PatientBirthDate",
    (0x0010, 0x0040): "PatientSex",
    (0x0008, 0x0020): "StudyDate",
    (0x0008, 0x0030): "StudyTime",
    (0x0008, 0x0090): "ReferringPhysicianName",
    (0x0008, 0x1010): "StationName",
    (0x0008, 0x0080): "InstitutionName",
    (0x0008, 0x0050): "AccessionNumber",
}

# Define a set of DICOM tags 
AUX_TAGS = {
    (0x0018, 0x0050): "SliceThickness"
}

# Define a set of DICOM tags 
MASK_FINDER_TAGS = {
    (0x0008, 0x103E): "SeriesDescription"
}

# Tags useful for identifying segmentation / RTSTRUCT mask objects.
MASK_TAGS = {
    (0x0008, 0x0016): "SOPClassUID",
    (0x0008, 0x0060): "Modality",
    (0x0008, 0x103E): "SeriesDescription",
    # RTSTRUCT
    (0x3006, 0x0002): "StructureSetLabel",
    (0x3006, 0x0004): "StructureSetName",
    (0x3006, 0x0020): "StructureSetROISequence",
    (0x3006, 0x0026): "ROIName",
    (0x3006, 0x0080): "RTROIObservationsSequence",
    # SEG
    (0x0062, 0x0002): "SegmentSequence",
    (0x0062, 0x0005): "SegmentLabel",
    (0x0062, 0x0008): "SegmentAlgorithmType",
    (0x0062, 0x0009): "SegmentAlgorithmName",
}

import glob
import re
from collections import defaultdict


_DEFAULT_AUDIT_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "audit_dicoms"
_DEFAULT_UNIFIED_AUDIT_JSON = _DEFAULT_AUDIT_OUTPUT_DIR / "dicom_audit.json"


_DICOM_FILENAME_RE = re.compile(r"^(?P<series>\d+)-(?P<instance>\d+)\.dcm$")


_SOP_CLASS_UID_NAMES = {
    # Common imaging storage classes (subset)
    "1.2.840.10008.5.1.4.1.1.2": "CT Image Storage",
    "1.2.840.10008.5.1.4.1.1.4": "MR Image Storage",
    "1.2.840.10008.5.1.4.1.1.128": "PET Image Storage",
    # Radiotherapy
    "1.2.840.10008.5.1.4.1.1.481.2": "RT Dose Storage",
    "1.2.840.10008.5.1.4.1.1.481.3": "RT Structure Set Storage",
    "1.2.840.10008.5.1.4.1.1.481.5": "RT Plan Storage",
    # Segmentation
    "1.2.840.10008.5.1.4.1.1.66.4": "Segmentation Storage",
}


def _sha256_hex(s: str) -> str:
    s = "" if s is None else str(s)
    return hashlib.sha256(s.encode("utf-8", errors="replace")).hexdigest()


def _fhir_id(prefix: str, stable_key: str, n: int = 24) -> str:
    """Create a stable FHIR-safe id ([A-Za-z0-9\-\.]{1,64}) derived from stable_key."""

    h = _sha256_hex(stable_key)[:n]
    return f"{prefix}-{h}"


def _fhir_now() -> str:
    # FHIR instant/dateTime should be ISO8601; use UTC Z.
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def export_mask_audit_as_fhir_r4_bundle(
    combined_mask_audit: dict,
    output_json: str | os.PathLike,
    *,
    patient_identifier_system: str = "urn:dicom:patientid",
) -> dict:
    """Export a FHIR R4 Bundle(type=collection) representing results of `find_tumor_mask_dicoms()`.

    Requirements (per user):
    - FHIR R4 strict-ish JSON
    - Bundle.type = "collection"
    - Save SeriesDescription only as `seriesDescriptionHash` (SHA-256)
    - Patient identifier system uses `urn:dicom:patientid`

    Notes:
    - DICOM UIDs are represented via Identifier.system='urn:dicom:uid' and value='urn:oid:{uid}'.
    - Filepaths are included in an Observation extension (secure environment assumption).
    """

    audits = combined_mask_audit.get("audits", []) or []
    timestamp = _fhir_now()

    patient_resources: dict[str, dict] = {}
    imagingstudy_resources: dict[tuple[str, str], dict] = {}  # (patient_id, study_uid) -> ImagingStudy
    observation_resources: list[dict] = []

    def _dicom_uid_identifier(uid: str) -> dict:
        return {"system": "urn:dicom:uid", "value": f"urn:oid:{uid}"}

    def _modality_coding(modality: str | None) -> dict | None:
        if modality in (None, "", " "):
            return None
        return {"system": "http://dicom.nema.org/resources/ontology/DCM", "code": str(modality)}

    def _ensure_series(imagingstudy: dict, series_uid: str | None, modality: str | None):
        if not series_uid:
            return
        existing = {s.get("uid") for s in imagingstudy.get("series", []) if isinstance(s, dict)}
        if series_uid in existing:
            return
        s = {"uid": str(series_uid)}
        mod = _modality_coding(modality)
        if mod:
            s["modality"] = mod
        imagingstudy.setdefault("series", []).append(s)

    # Build resources
    for a in audits:
        if not isinstance(a, dict):
            continue
        if a.get("error"):
            continue

        patient_id = a.get("patient_id")
        study_uid = a.get("study_instance_uid")
        mask_series_uid = a.get("series_instance_uid")
        referenced_series_uid = a.get("referenced_series_instance_uid")

        if not patient_id or not study_uid:
            continue

        # Patient
        if patient_id not in patient_resources:
            pid = _fhir_id("patient", str(patient_id))
            patient_resources[patient_id] = {
                "resourceType": "Patient",
                "id": pid,
                "identifier": [{"system": patient_identifier_system, "value": str(patient_id)}],
            }

        patient_ref = {"reference": f"Patient/{patient_resources[patient_id]['id']}"}

        # ImagingStudy (one per patient+study)
        study_key = (str(patient_id), str(study_uid))
        if study_key not in imagingstudy_resources:
            is_id = _fhir_id("imagingstudy", f"{patient_id}|{study_uid}")
            imagingstudy_resources[study_key] = {
                "resourceType": "ImagingStudy",
                "id": is_id,
                "status": "available",
                "subject": patient_ref,
                "identifier": [_dicom_uid_identifier(str(study_uid))],
                "started": timestamp,
                "series": [],
            }

        imagingstudy = imagingstudy_resources[study_key]
        _ensure_series(imagingstudy, mask_series_uid, a.get("modality"))
        _ensure_series(imagingstudy, referenced_series_uid, None)

        imagingstudy_ref = {"reference": f"ImagingStudy/{imagingstudy['id']}"}

        # Observation (one per audited mask-related DICOM)
        file_path = a.get("file")
        obs_id = _fhir_id("obs", f"{patient_id}|{study_uid}|{file_path}")

        series_desc = a.get("series_description")
        series_desc_hash = _sha256_hex(series_desc) if series_desc not in (None, "", " ") else None

        deep_completed = bool(a.get("deep_tag_search_completed", False))
        scan_sequences_enabled = bool(combined_mask_audit.get("config", {}).get("scan_sequences", True))

        components: list[dict] = []

        def _add_component(code: str, value_key: str, value):
            components.append(
                {
                    "code": {
                        "coding": [
                            {
                                "system": "http://example.org/fhir/CodeSystem/dicom-audit",
                                "code": code,
                            }
                        ]
                    },
                    value_key: value,
                }
            )

        if a.get("modality"):
            _add_component("modality", "valueString", str(a.get("modality")))
        if a.get("sop_class_uid"):
            _add_component("sopClassUID", "valueString", str(a.get("sop_class_uid")))
        if a.get("sop_instance_uid"):
            _add_component("sopInstanceUID", "valueString", str(a.get("sop_instance_uid")))

        _add_component("isSEG", "valueBoolean", bool(a.get("is_seg")))
        _add_component("isRTSTRUCT", "valueBoolean", bool(a.get("is_rtstruct")))
        _add_component("maskFound", "valueBoolean", bool(a.get("mask_found")))

        _add_component("scanSequencesEnabled", "valueBoolean", scan_sequences_enabled)
        _add_component("deepTagSearchCompleted", "valueBoolean", deep_completed)

        # Requested: seriesDescriptionHash (SHA-256)
        if series_desc_hash:
            _add_component("seriesDescriptionHash", "valueString", series_desc_hash)

        if mask_series_uid:
            _add_component("maskSeriesInstanceUID", "valueString", str(mask_series_uid))
        if referenced_series_uid:
            _add_component("referencedSeriesInstanceUID", "valueString", str(referenced_series_uid))

        _add_component("tumorLikeLabelCount", "valueInteger", int(a.get("tumor_like_label_count", 0) or 0))

        if deep_completed:
            labels = a.get("labels_found") or []
            tumor_labels = a.get("tumor_like_labels_found") or []

            def _extract_hashes(xs):
                out = []
                for x in xs:
                    if isinstance(x, dict) and "sha256" in x:
                        out.append(x["sha256"])
                    elif isinstance(x, str):
                        out.append(_sha256_hex(x))
                return out

            _add_component("labelHashesSHA256", "valueString", json.dumps(_extract_hashes(labels)))
            _add_component("tumorLikeLabelHashesSHA256", "valueString", json.dumps(_extract_hashes(tumor_labels)))

        obs = {
            "resourceType": "Observation",
            "id": obs_id,
            "status": "final",
            "category": [
                {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                            "code": "imaging",
                            "display": "Imaging",
                        }
                    ]
                }
            ],
            "code": {
                "coding": [
                    {
                        "system": "http://example.org/fhir/CodeSystem/dicom-audit",
                        "code": "dicom-mask-audit",
                        "display": "DICOM SEG/RTSTRUCT Mask Audit",
                    }
                ],
                "text": "DICOM SEG/RTSTRUCT Mask Audit",
            },
            "subject": patient_ref,
            "effectiveDateTime": timestamp,
            "derivedFrom": [imagingstudy_ref],
            "component": components,
        }

        # Secure-environment assumption: keep raw filepath in an extension.
        if file_path:
            obs.setdefault("extension", []).append(
                {
                    "url": "http://example.org/fhir/StructureDefinition/source-file-path",
                    "valueString": str(file_path),
                }
            )

        observation_resources.append(obs)

    entries: list[dict] = []
    for res in patient_resources.values():
        entries.append({"fullUrl": f"urn:uuid:{res['id']}", "resource": res})
    for res in imagingstudy_resources.values():
        entries.append({"fullUrl": f"urn:uuid:{res['id']}", "resource": res})
    for res in observation_resources:
        entries.append({"fullUrl": f"urn:uuid:{res['id']}", "resource": res})

    bundle = {
        "resourceType": "Bundle",
        "type": "collection",
        "timestamp": timestamp,
        "entry": entries,
    }

    Path(output_json).write_text(json.dumps(bundle, indent=2))
    return bundle


@dataclass(frozen=True)
class AuditConfig:
    """Configuration for `audit_dicom_header`.

    Defaults are set to a safer mode for clinical-ish workflows:
    - No full header dump (`include_all_headers=False`)
    - Redact values in outputs (`redact_values=True`)
    - Print only when violations exist
    """

    include_all_headers: bool = False
    scan_sequences: bool = True
    flag_private_tags: bool = True
    redact_values: bool = True
    print_only_violations: bool = True
    print_tag_details: bool = False
    max_value_preview: int = 0


@dataclass(frozen=True)
class MaskFinderConfig:
    """Configuration for `find_tumor_mask_dicoms`.

    Defaults are conservative:
    - Print only when a mask-like object is detected
    - Print only counts unless `print_tag_details=True`
    - Redact extracted label values in JSON by default
    """

    scan_sequences: bool = True
    deep_tag_search: bool = False
    redact_values: bool = True
    print_only_matches: bool = True
    print_tag_details: bool = False
    print_summary: bool = True
    max_value_preview: int = 0


@dataclass
class UnifiedAuditConfig:
    """Configuration for the unified one-and-done DICOM audit."""

    # Stage toggles
    run_metadata_summary: bool = True
    run_phi_audit: bool = True
    run_slice_thickness: bool = True
    run_mask_finder: bool = True

    # Outputs
    write_unified_json: bool = True
    write_unified_fhir_bundle: bool = True

    # Storage controls
    store_input_files: bool = True
    store_file_paths_in_audits: bool = True  # if False, redact per-file `file` fields to hashes

    # Dataset-layout assumptions for metadata summary (matches get_dicom layout)
    base_dir_for_layout: str = "../../data/raw/NSCLC-Radiomics"

    # Component configs
    header_audit_config: AuditConfig = AuditConfig()
    mask_finder_config: MaskFinderConfig = MaskFinderConfig()


@dataclass
class DicomHeaderCache:
    """In-run cache of pydicom datasets (header-only reads)."""

    datasets_by_path: dict[str, object]
    hits: int = 0
    misses: int = 0
    errors: list[dict] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


def _normalize_path_key(path: str | os.PathLike) -> str:
    try:
        return str(Path(path).resolve())
    except Exception:
        return str(path)


def dcmread_cached(path: str | os.PathLike, cache: DicomHeaderCache, *, stop_before_pixels: bool = True):
    """Read a DICOM header using a cache to avoid repeated disk IO."""

    import pydicom

    key = _normalize_path_key(path)
    if key in cache.datasets_by_path:
        cache.hits += 1
        return cache.datasets_by_path[key]

    cache.misses += 1
    ds = pydicom.dcmread(str(path), stop_before_pixels=stop_before_pixels)
    cache.datasets_by_path[key] = ds
    return ds


class Lung1PathError(ValueError):
    pass


def lung1_patient_study_series_from_path(path: str | os.PathLike) -> tuple[str, str, str]:
    """Return (patient_id, study_instance_uid, series_instance_uid_folder) from Lung1 folder layout.

    Expected layout:
    .../<PatientID>/<StudyInstanceUID>/<SeriesInstanceUID>/<instance>.dcm

    Hard-fail if PatientID doesn't contain 'LUNG1' (case-insensitive).
    """

    p = Path(path)
    try:
        series_folder = p.parents[0].name
        study_instance_uid = p.parents[1].name
        patient_id = p.parents[2].name
    except Exception:
        raise Lung1PathError(f"Path too short to extract PatientID/StudyInstanceUID/SeriesInstanceUID: {path}")

    if not patient_id or "lung1" not in patient_id.lower():
        raise Lung1PathError(f"Expected PatientID folder containing 'LUNG1' but got '{patient_id}' for path: {path}")
    if not study_instance_uid:
        raise Lung1PathError(f"Empty StudyInstanceUID folder name for path: {path}")
    if not series_folder:
        raise Lung1PathError(f"Empty SeriesInstanceUID folder name for path: {path}")

    return patient_id, study_instance_uid, series_folder


def extract_slice_thickness_mm(ds) -> float | None:
    """Extract SliceThickness (0018,0050) in mm if present and parseable."""

    # Prefer direct keyword lookup; fall back to attribute access; then explicit tag.
    raw = None
    try:
        raw = ds.get("SliceThickness", None)
    except Exception:
        raw = None

    try:
        if raw is None and hasattr(ds, "SliceThickness"):
            raw = getattr(ds, "SliceThickness")
    except Exception:
        pass

    # Explicit tag lookup
    try:
        from pydicom.tag import Tag

        thickness_tag = Tag(0x0018, 0x0050)
        if raw in (None, "", " ") and thickness_tag in ds:
            raw = ds[thickness_tag].value
    except Exception:
        pass

    if raw in (None, "", " "):
        return None

    try:
        return float(raw)
    except Exception:
        s = str(raw)
        m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
        if not m:
            return None
        try:
            return float(m.group(0))
        except Exception:
            return None


def _parse_float_value(raw) -> tuple[float | None, str | None]:
    if raw in (None, "", " "):
        return None, None
    try:
        return float(raw), None
    except Exception as e:
        try:
            return float(str(raw).strip()), None
        except Exception as e2:
            return None, f"float_parse_error: {e2 or e}"


def _parse_int_value(raw) -> tuple[int | None, str | None]:
    if raw in (None, "", " "):
        return None, None
    try:
        return int(raw), None
    except Exception as e:
        try:
            return int(float(raw)), None
        except Exception as e2:
            return None, f"int_parse_error: {e2 or e}"


def _parse_str_value(raw) -> tuple[str | None, str | None]:
    if raw in (None, "", " "):
        return None, None
    try:
        s = str(raw).strip()
    except Exception as e:
        return None, f"str_parse_error: {e}"
    if s == "":
        return None, None
    return s, None


def _parse_float_list(raw, expected_len: int | None = None) -> tuple[list[float] | None, str | None]:
    if raw in (None, "", " "):
        return None, None

    seq = None
    if isinstance(raw, (list, tuple)):
        seq = list(raw)
    elif isinstance(raw, str):
        parts = [p.strip() for p in raw.replace(",", "\\").split("\\") if p.strip() != ""]
        seq = parts if parts else [raw]
    else:
        try:
            from collections.abc import Iterable

            if isinstance(raw, Iterable) and not isinstance(raw, (bytes, bytearray)):
                seq = list(raw)
        except Exception:
            seq = None
        if seq is None:
            seq = [raw]

    values: list[float] = []
    for item in seq:
        if item in (None, "", " "):
            continue
        try:
            values.append(float(item))
        except Exception as e:
            return None, f"float_list_parse_error: {e}"

    if not values:
        return None, None
    if expected_len is not None and len(values) != expected_len:
        return None, f"expected_{expected_len}_values_got_{len(values)}"

    return values, None


def extract_ct_geometry_meta(ds) -> dict:
    """Collect CT geometry metadata needed for downstream validation."""

    pixel_spacing, pixel_spacing_err = _parse_float_list(ds.get("PixelSpacing"), expected_len=2)
    orientation, orientation_err = _parse_float_list(ds.get("ImageOrientationPatient"), expected_len=6)
    spacing_between_slices, spacing_between_slices_err = _parse_float_value(ds.get("SpacingBetweenSlices"))
    gantry_tilt, gantry_tilt_err = _parse_float_value(ds.get("GantryDetectorTilt"))
    frame_uid, frame_uid_err = _parse_str_value(ds.get("FrameOfReferenceUID"))
    rows, rows_err = _parse_int_value(ds.get("Rows"))
    columns, columns_err = _parse_int_value(ds.get("Columns"))

    return {
        "pixel_spacing_mm": {
            "value": pixel_spacing,
            "present": pixel_spacing is not None,
            "error": pixel_spacing_err,
            "source_tag": "(0028,0030)",
        },
        "image_orientation_patient": {
            "value": orientation,
            "present": orientation is not None,
            "error": orientation_err,
            "source_tag": "(0020,0037)",
        },
        "spacing_between_slices_mm": {
            "value": spacing_between_slices,
            "present": spacing_between_slices is not None,
            "error": spacing_between_slices_err,
            "source_tag": "(0018,0088)",
        },
        "gantry_detector_tilt_deg": {
            "value": gantry_tilt,
            "present": gantry_tilt is not None,
            "error": gantry_tilt_err,
            "source_tag": "(0018,1120)",
        },
        "frame_of_reference_uid": {
            "value": frame_uid,
            "present": frame_uid is not None,
            "error": frame_uid_err,
            "source_tag": "(0020,0052)",
        },
        "rows": {
            "value": rows,
            "present": rows is not None,
            "error": rows_err,
            "source_tag": "(0028,0010)",
        },
        "columns": {
            "value": columns,
            "present": columns is not None,
            "error": columns_err,
            "source_tag": "(0028,0011)",
        },
    }


def extract_hu_rescale_meta(ds) -> dict:
    """Capture rescale slope/intercept/type for HU normalization checks."""

    slope, slope_err = _parse_float_value(ds.get("RescaleSlope"))
    intercept, intercept_err = _parse_float_value(ds.get("RescaleIntercept"))
    rescale_type, type_err = _parse_str_value(ds.get("RescaleType"))

    errors = [err for err in (slope_err, intercept_err, type_err) if err]
    error_msg = "; ".join(errors) if errors else None
    valid = slope is not None and intercept is not None

    return {
        "slope": slope,
        "intercept": intercept,
        "type": rescale_type,
        "valid": bool(valid),
        "error": error_msg,
        "source_tags": {
            "slope": "(0028,1053)",
            "intercept": "(0028,1052)",
            "type": "(0028,1054)",
        },
    }


def extract_seg_grid_meta(ds) -> dict:
    """Collect SEG grid metadata (rows/cols/spacing/frame UID)."""

    rows, rows_err = _parse_int_value(ds.get("Rows"))
    columns, columns_err = _parse_int_value(ds.get("Columns"))

    pixel_spacing = None
    pixel_spacing_err = None
    spacing_between_slices = None
    spacing_between_slices_err = None

    try:
        shared = ds.get("SharedFunctionalGroupsSequence")
        if shared:
            first = shared[0]
            pm_seq = getattr(first, "PixelMeasuresSequence", None)
            if pm_seq:
                pm = pm_seq[0]
                pixel_spacing, pixel_spacing_err = _parse_float_list(getattr(pm, "PixelSpacing", None), expected_len=2)
                spacing_between_slices, spacing_between_slices_err = _parse_float_value(getattr(pm, "SpacingBetweenSlices", None))
    except Exception as e:
        if pixel_spacing_err is None:
            pixel_spacing_err = str(e)
        if spacing_between_slices_err is None:
            spacing_between_slices_err = str(e)

    if pixel_spacing is None:
        fallback, fallback_err = _parse_float_list(ds.get("PixelSpacing"), expected_len=2)
        if fallback is not None:
            pixel_spacing = fallback
        if pixel_spacing_err is None:
            pixel_spacing_err = fallback_err

    if pixel_spacing is None:
        try:
            per_frame = ds.get("PerFrameFunctionalGroupsSequence")
            if per_frame:
                first = per_frame[0]
                pm_seq = getattr(first, "PixelMeasuresSequence", None)
                if pm_seq:
                    pm = pm_seq[0]
                    pixel_spacing, pixel_spacing_err = _parse_float_list(getattr(pm, "PixelSpacing", None), expected_len=2)
                    spacing_between_slices, spacing_between_slices_err = _parse_float_value(getattr(pm, "SpacingBetweenSlices", None))
        except Exception as e:
            if pixel_spacing_err is None:
                pixel_spacing_err = str(e)
            if spacing_between_slices_err is None:
                spacing_between_slices_err = str(e)

    if spacing_between_slices is None:
        fallback_spacing, fallback_spacing_err = _parse_float_value(ds.get("SpacingBetweenSlices"))
        if fallback_spacing is not None:
            spacing_between_slices = fallback_spacing
        if spacing_between_slices_err is None:
            spacing_between_slices_err = fallback_spacing_err

    frame_uid, frame_uid_err = _parse_str_value(ds.get("FrameOfReferenceUID"))

    if frame_uid is None:
        try:
            shared = ds.get("SharedFunctionalGroupsSequence")
            if shared:
                first = shared[0]
                frame_seq = getattr(first, "FrameOfReferenceSequence", None)
                if frame_seq:
                    frame_elem = frame_seq[0]
                    frame_uid, frame_uid_err = _parse_str_value(getattr(frame_elem, "FrameOfReferenceUID", None))
        except Exception as e:
            if frame_uid_err is None:
                frame_uid_err = str(e)

    return {
        "rows": {
            "value": rows,
            "present": rows is not None,
            "error": rows_err,
            "source_tag": "(0028,0010)",
        },
        "columns": {
            "value": columns,
            "present": columns is not None,
            "error": columns_err,
            "source_tag": "(0028,0011)",
        },
        "pixel_spacing_mm": {
            "value": pixel_spacing,
            "present": pixel_spacing is not None,
            "error": pixel_spacing_err,
            "source_tag": "PixelMeasures.PixelSpacing",
        },
        "spacing_between_slices_mm": {
            "value": spacing_between_slices,
            "present": spacing_between_slices is not None,
            "error": spacing_between_slices_err,
            "source_tag": "PixelMeasures.SpacingBetweenSlices",
        },
        "frame_of_reference_uid": {
            "value": frame_uid,
            "present": frame_uid is not None,
            "error": frame_uid_err,
            "source_tag": "FrameOfReferenceUID",
        },
    }


def _compare_seg_geometry_to_ct(
    seg_info: dict | None,
    ct_geometry: dict | None,
    ct_slice_thickness_mm: float | None,
    tol_xy: float = 1e-3,
    tol_z: float = 1e-2,
) -> dict:
    """Compare SEG grid metadata against CT geometry with tolerances."""

    def _value(d: dict | None, key: str):
        if not isinstance(d, dict):
            return None
        val = d.get(key)
        if isinstance(val, dict):
            return val.get("value")
        return None

    seg_rows = _value(seg_info, "rows")
    seg_cols = _value(seg_info, "columns")
    seg_ps = _value(seg_info, "pixel_spacing_mm")
    seg_spacing_z = _value(seg_info, "spacing_between_slices_mm")
    seg_for = _value(seg_info, "frame_of_reference_uid")

    ct_rows = _value(ct_geometry, "rows")
    ct_cols = _value(ct_geometry, "columns")
    ct_ps = _value(ct_geometry, "pixel_spacing_mm")
    ct_spacing_z = _value(ct_geometry, "spacing_between_slices_mm")
    ct_for = _value(ct_geometry, "frame_of_reference_uid")

    pixel_spacing_match = None
    if seg_ps is not None and ct_ps is not None and len(seg_ps) == 2 and len(ct_ps) == 2:
        pixel_spacing_match = all(abs(seg_ps[i] - ct_ps[i]) <= tol_xy for i in range(2))

    matrix_match = None
    if seg_rows is not None and seg_cols is not None and ct_rows is not None and ct_cols is not None:
        matrix_match = seg_rows == ct_rows and seg_cols == ct_cols

    spacing_match = None
    seg_ref_spacing = seg_spacing_z
    ct_ref_spacing = ct_spacing_z if ct_spacing_z is not None else ct_slice_thickness_mm
    if seg_ref_spacing is not None and ct_ref_spacing is not None:
        spacing_match = abs(seg_ref_spacing - ct_ref_spacing) <= tol_z

    frame_match = None
    if seg_for and ct_for:
        frame_match = seg_for == ct_for

    comparisons = [v for v in (pixel_spacing_match, matrix_match, spacing_match, frame_match) if v is not None]
    overall = None
    if comparisons:
        overall = all(comparisons)

    return {
        "match": overall,
        "pixel_spacing": pixel_spacing_match,
        "matrix": matrix_match,
        "spacing_between_slices": spacing_match,
        "frame_of_reference": frame_match,
    }


_DECISION_OK = "OK"
_DECISION_WARN = "WARN"
_DECISION_FAIL = "FAILURE"


def _evaluate_preprocessing_readiness(patient_payload: dict) -> dict:
    """Replicate preprocessing validation logic on the in-memory patient payload."""

    def _entry_present(entry: dict | None, key: str) -> bool:
        if not isinstance(entry, dict):
            return False
        value = entry.get(key)
        if isinstance(value, dict):
            if value.get("present") is False:
                return False
            return value.get("value") is not None
        return value is not None

    issues: list[str] = []
    warnings: list[str] = []

    principal = patient_payload.get("principal_series") or {}
    if not principal.get("has_principal_series"):
        issues.append("principal_series_missing")
        return {
            "decision": _DECISION_FAIL,
            "issues": issues,
            "warnings": warnings,
            "segments": [],
        }

    geometry = principal.get("geometry")
    if not isinstance(geometry, dict):
        issues.append("principal_geometry_missing")
    else:
        if not _entry_present(geometry, "pixel_spacing_mm"):
            issues.append("principal_geometry.pixel_spacing_missing")
        if not _entry_present(geometry, "rows") or not _entry_present(geometry, "columns"):
            warnings.append("principal_geometry.matrix_size_missing")
        if not _entry_present(geometry, "image_orientation_patient"):
            warnings.append("principal_geometry.orientation_missing")
        if not _entry_present(geometry, "frame_of_reference_uid"):
            warnings.append("principal_geometry.frame_uid_missing")

    hu_meta = principal.get("hu_rescale")
    if not isinstance(hu_meta, dict) or not hu_meta.get("valid"):
        issues.append("principal_hu_rescale_invalid")
    elif hu_meta.get("error"):
        warnings.append(f"principal_hu_rescale_warning:{hu_meta['error']}")

    if principal.get("slice_thickness_mm") is None:
        warnings.append("principal_slice_thickness_missing")

    seg_records: list[dict] = []
    for rec in patient_payload.get("files", []) or []:
        if not isinstance(rec, dict):
            continue
        ma = rec.get("mask_audit")
        if not isinstance(ma, dict):
            continue
        if ma.get("mask_type") == "SEG":
            seg_records.append(rec)

    if not seg_records:
        issues.append("segmentation_missing")

    segments_summary: list[dict] = []
    for seg_rec in seg_records:
        ma = seg_rec.get("mask_audit") or {}
        seg_info = ma.get("seg_grid_info")
        match_info = ma.get("seg_grid_matches_ct")
        seg_file = seg_rec.get("file")

        if not isinstance(seg_info, dict):
            issues.append(f"seg_grid_info_missing:{seg_file}")
            segments_summary.append({"file": seg_file, "match": None, "details": None})
            continue

        if not isinstance(match_info, dict):
            warnings.append(f"seg_alignment_unknown:{seg_file}")
            segments_summary.append({"file": seg_file, "match": None, "details": None})
            continue

        match_state = match_info.get("match")
        if match_state is False:
            issues.append(f"seg_alignment_failed:{seg_file}")
        elif match_state is None:
            warnings.append(f"seg_alignment_indeterminate:{seg_file}")

        segments_summary.append({"file": seg_file, "match": match_state, "details": match_info})

    decision = _DECISION_OK
    if issues:
        decision = _DECISION_FAIL
    elif warnings:
        decision = _DECISION_WARN

    return {
        "decision": decision,
        "issues": issues,
        "warnings": warnings,
        "segments": segments_summary,
    }


def summarize_paths_layout(
    dicom_paths: list[str],
    *,
    base_dir: str,
    print_summary: bool = True,
) -> dict:
    """Compute the same dataset-style summary that `get_dicom()` prints, but as structured data."""

    base_path = Path(base_dir).resolve()

    patient_ids_matched: set[str] = set()
    series_folders_by_patient: dict[str, set[tuple[str, str]]] = defaultdict(set)
    dcm_count_by_series_folder: dict[tuple[str, str, str], int] = defaultdict(int)
    representative_dicom_by_series_folder: dict[tuple[str, str, str], str] = {}
    unparsed = 0

    for p in dicom_paths:
        p_path = Path(p)
        try:
            p_path = p_path.resolve()
        except Exception:
            pass

        try:
            rel = p_path.relative_to(base_path)
        except Exception:
            rel = Path(os.path.relpath(str(p_path), str(base_path)))

        parts = rel.parts
        if len(parts) < 4:
            unparsed += 1
            continue

        patient = parts[0]
        study = parts[1]
        series = parts[2]

        patient_ids_matched.add(patient)
        series_folders_by_patient[patient].add((study, series))
        dcm_count_by_series_folder[(patient, study, series)] += 1
        representative_dicom_by_series_folder.setdefault((patient, study, series), str(p_path))

    patient_ids_sorted = sorted(patient_ids_matched)
    first_pid = patient_ids_sorted[0] if patient_ids_sorted else None
    last_pid = patient_ids_sorted[-1] if patient_ids_sorted else None

    min_series = None
    max_series = None
    if series_folders_by_patient:
        per_patient_series_counts = [(len(v), pid) for pid, v in series_folders_by_patient.items()]
        per_patient_series_counts.sort(key=lambda t: (t[0], t[1]))
        min_series_count, min_series_pid = per_patient_series_counts[0]
        max_series_count, max_series_pid = max(per_patient_series_counts, key=lambda t: (t[0], t[1]))
        min_series = {"count": int(min_series_count), "patient_id": str(min_series_pid)}
        max_series = {"count": int(max_series_count), "patient_id": str(max_series_pid)}

    principal_series_items = []
    for (patient, study, series), count in dcm_count_by_series_folder.items():
        if int(count) > 2:
            principal_series_items.append((int(count), patient, study, series))
    principal_series_items.sort(key=lambda t: (t[0], t[1], t[2], t[3]))

    principal_summary = {
        "principal_series_folder_count": len(principal_series_items),
        "min_dicoms": None,
        "max_dicoms": None,
        "avg_dicoms": None,
        "slice_thickness_mm": {"min": None, "max": None, "avg": None, "values_parsed": 0},
    }

    if principal_series_items:
        min_dcms, min_pid, _min_study, min_series_uid = principal_series_items[0]
        max_dcms, max_pid, _max_study, max_series_uid = max(
            principal_series_items, key=lambda t: (t[0], t[1], t[2], t[3])
        )
        avg_dcms = sum(int(c) for (c, _p, _st, _s) in principal_series_items) / len(principal_series_items)
        principal_summary["min_dicoms"] = {"count": int(min_dcms), "patient_id": str(min_pid), "series": str(min_series_uid)}
        principal_summary["max_dicoms"] = {"count": int(max_dcms), "patient_id": str(max_pid), "series": str(max_series_uid)}
        principal_summary["avg_dicoms"] = float(avg_dcms)

        # SliceThickness across principal series (representative instance per series)
        thicknesses = []
        for (_count, pid, st, ser) in principal_series_items:
            rep = representative_dicom_by_series_folder.get((pid, st, ser))
            if not rep:
                continue
            try:
                import pydicom

                ds = pydicom.dcmread(rep, stop_before_pixels=True)
            except Exception:
                continue

            v = extract_slice_thickness_mm(ds)
            if v is not None:
                thicknesses.append(v)

        if thicknesses:
            principal_summary["slice_thickness_mm"] = {
                "min": float(min(thicknesses)),
                "max": float(max(thicknesses)),
                "avg": float(sum(thicknesses) / len(thicknesses)),
                "values_parsed": int(len(thicknesses)),
            }

    out = {
        "total_paths": int(len(dicom_paths)),
        "unparsed_paths": int(unparsed),
        "patient_ids_count": int(len(patient_ids_sorted)),
        "patient_id_first": first_pid,
        "patient_id_last": last_pid,
        "min_series_folders_per_patient": min_series,
        "max_series_folders_per_patient": max_series,
        "principal_series": principal_summary,
    }

    if print_summary:
        print(f"Total paths found: {out['total_paths']}")
        print(f"PatientIDs found: {out['patient_ids_count']}")
        print(f"1st PatientID (from sorted IDs): {out['patient_id_first'] if out['patient_id_first'] else 'UNKNOWN'}")
        print(f"Last PatientID (from sorted IDs): {out['patient_id_last'] if out['patient_id_last'] else 'UNKNOWN'}")
        if min_series:
            print(
                f"Min series folders per patient: {min_series['count']} (PatientID={min_series['patient_id']})"
            )
        else:
            print("Min series folders per patient: UNKNOWN (PatientID=UNKNOWN)")
        if max_series:
            print(
                f"Max series folders per patient: {max_series['count']} (PatientID={max_series['patient_id']})"
            )
        else:
            print("Max series folders per patient: UNKNOWN (PatientID=UNKNOWN)")

        ps = principal_summary
        print(f"Principal series folders found: {ps['principal_series_folder_count']}")
        if ps["min_dicoms"]:
            print(
                "Min DICOMs per principal series folder: "
                f"{ps['min_dicoms']['count']} (PatientID={ps['min_dicoms']['patient_id']}, SeriesInstanceUID={ps['min_dicoms']['series']})"
            )
            print(
                "Max DICOMs per principal series folder: "
                f"{ps['max_dicoms']['count']} (PatientID={ps['max_dicoms']['patient_id']}, SeriesInstanceUID={ps['max_dicoms']['series']})"
            )
            print(f"Avg DICOMs per principal series folder: {ps['avg_dicoms']:.4g}")

            th = ps["slice_thickness_mm"]
            if th["values_parsed"]:
                print(f"Min SliceThickness (mm) [principal series]: {th['min']:.4g}")
                print(f"Max SliceThickness (mm) [principal series]: {th['max']:.4g}")
                print(f"Avg SliceThickness (mm) [principal series]: {th['avg']:.4g}")
            else:
                print("Min SliceThickness (mm) [principal series]: UNKNOWN")
                print("Max SliceThickness (mm) [principal series]: UNKNOWN")
                print("Avg SliceThickness (mm) [principal series]: UNKNOWN")
        else:
            print(
                "Min DICOMs per principal series folder: UNKNOWN "
                "(PatientID=UNKNOWN, SeriesInstanceUID=UNKNOWN)"
            )
            print(
                "Max DICOMs per principal series folder: UNKNOWN "
                "(PatientID=UNKNOWN, SeriesInstanceUID=UNKNOWN)"
            )
            print("Avg DICOMs per principal series folder: UNKNOWN")
            print("Min SliceThickness (mm) [principal series]: UNKNOWN")
            print("Max SliceThickness (mm) [principal series]: UNKNOWN")
            print("Avg SliceThickness (mm) [principal series]: UNKNOWN")

    return out

def get_dicom(
        PatientID=None,
    StudyInstanceUID_index=None,
    SeriesInstanceUID_index=None,
        SeriesNumber=None,
        InstanceNumber=None,
    base_dir="../../data/raw/NSCLC-Radiomics",
):
    """
        Grab DICOM file(s) from the Lung1 dataset.

        Behavior:
        - If *all* of (PatientID, StudyInstanceUID_index, SeriesNumber, InstanceNumber) are not None,
            returns a single DICOM path (str), matching the prior behavior.
        - If *any* of those arguments are None, that argument becomes a wildcard and the function
            returns a list[str] of all matching DICOM paths.
    
    Args:
        PatientID (str|None): Patient folder name (e.g. "LUNG1-001"). None means all patients.
        StudyInstanceUID_index (int|None): Study order (1-based index, lexicographically sorted
            study directory list within each patient). None means all studies per patient.
        SeriesInstanceUID_index (int|None): Series order (1-based index, lexicographically sorted
            series directory list within each selected study). None means all series per study.
        SeriesNumber (int|None): DICOM SeriesNumber encoded in filename as "<SeriesNumber>-...".
            None means all series numbers.
        InstanceNumber (int|None): DICOM InstanceNumber encoded in filename as "...-<Instance>.dcm".
            None means all instance numbers.
        base_dir (str): Base path to NSCLC-Radiomics dataset.
    
    Returns:
        str | list[str]: One DICOM path when fully specified; otherwise a list of paths.
    """
    # Always print inputs at the start (requested).
    print("Getting DICOM(s)\n")
    print("Args:")
    print(f"PatientID: {PatientID}")
    print(f"StudyInstanceUID_index: {StudyInstanceUID_index}")
    print(f"SeriesInstanceUID_index: {SeriesInstanceUID_index}")
    print(f"SeriesNumber: {SeriesNumber}")
    print(f"InstanceNumber: {InstanceNumber}")
    print(f"base_dir: {base_dir}")

    want_list = any(
        v is None
        for v in (
            PatientID,
            StudyInstanceUID_index,
            SeriesInstanceUID_index,
            SeriesNumber,
            InstanceNumber,
        )
    )

    # Patient selection
    if PatientID is None:
        candidate_patient_dirs = sorted(glob.glob(os.path.join(base_dir, "*")))
        patient_dirs = [p for p in candidate_patient_dirs if os.path.isdir(p)]
        if not patient_dirs:
            raise FileNotFoundError(f"No patient directories found in {base_dir}")
    else:
        patient_dir = os.path.join(base_dir, PatientID)
        if not os.path.isdir(patient_dir):
            raise FileNotFoundError(f"Patient directory not found: {patient_dir}")
        patient_dirs = [patient_dir]

    matches = []

    for patient_dir in patient_dirs:
        # Select study directories under the patient folder
        study_dirs = glob.glob(os.path.join(patient_dir, "*"))
        study_dirs = [s for s in study_dirs if os.path.isdir(s)]
        if not study_dirs:
            # If PatientID is explicit, preserve the old behavior of erroring.
            if PatientID is not None:
                raise FileNotFoundError(f"No studies found for patient ds ID {PatientID}")
            continue

        # Sort study directories lexicographically (UIDs are strings, not ints)
        study_dirs.sort()

        # Select study (1-based index) or all
        if StudyInstanceUID_index is None:
            chosen_study_dirs = study_dirs
        else:
            try:
                chosen_study_dirs = [study_dirs[StudyInstanceUID_index - 1]]
            except IndexError:
                raise IndexError(
                    f"Study index {StudyInstanceUID_index} out of range "
                    f"(found {len(study_dirs)} studies) for patient {os.path.basename(patient_dir)}."
                )

        for chosen_study in chosen_study_dirs:
            series_dirs = glob.glob(os.path.join(chosen_study, "*"))
            series_dirs = [s for s in series_dirs if os.path.isdir(s)]
            if not series_dirs:
                continue
            series_dirs.sort()

            # Select series (1-based index) or all
            if SeriesInstanceUID_index is None:
                chosen_series_dirs = series_dirs
            else:
                try:
                    chosen_series_dirs = [series_dirs[SeriesInstanceUID_index - 1]]
                except IndexError:
                    raise IndexError(
                        f"Series index {SeriesInstanceUID_index} out of range "
                        f"(found {len(series_dirs)} series) for study {os.path.basename(chosen_study)} "
                        f"patient {os.path.basename(patient_dir)}."
                    )

            for series_dir in chosen_series_dirs:
                dcm_paths = glob.glob(os.path.join(series_dir, "*.dcm"))
                if not dcm_paths:
                    continue
                dcm_paths.sort()

                for dicom_path in dcm_paths:
                    name = os.path.basename(dicom_path)
                    m = _DICOM_FILENAME_RE.match(name)
                    if not m:
                        continue
                    series_num = int(m.group("series"))
                    instance_num = int(m.group("instance"))

                    if SeriesNumber is not None and series_num != SeriesNumber:
                        continue
                    if InstanceNumber is not None and instance_num != InstanceNumber:
                        continue

                    matches.append(dicom_path)

    if not matches:
        raise FileNotFoundError(
            "No DICOMs matched "
            f"PatientID={PatientID}, StudyInstanceUID_index={StudyInstanceUID_index}, "
            f"SeriesNumber={SeriesNumber}, InstanceNumber={InstanceNumber} under {base_dir}"
        )

    # Deterministic ordering
    matches.sort()

    # Print totals and patient ID span (requested) once matches are known.
    print(f"Total paths found: {len(matches)}")
    try:
        base_path = Path(base_dir).resolve()
        patient_ids_matched: set[str] = set()
        series_folders_by_patient: dict[str, set[tuple[str, str]]] = defaultdict(set)
        dcm_count_by_series_folder: dict[tuple[str, str, str], int] = defaultdict(int)
        representative_dicom_by_series_folder: dict[tuple[str, str, str], str] = {}

        for p in matches:
            p_path = Path(p).resolve()

            try:
                rel = p_path.relative_to(base_path)
            except Exception:
                # If base_dir resolution doesn't match (symlinks, etc.), fall back to string parsing.
                rel = Path(os.path.relpath(str(p_path), str(base_path)))

            parts = rel.parts
            # Expected: <patient>/<study>/<series>/<file>.dcm
            if len(parts) < 4:
                continue

            patient = parts[0]
            study = parts[1]
            series = parts[2]

            patient_ids_matched.add(patient)
            series_folders_by_patient[patient].add((study, series))
            dcm_count_by_series_folder[(patient, study, series)] += 1
            representative_dicom_by_series_folder.setdefault((patient, study, series), str(p_path))

        # Requested: print first/last PatientID in sorted order.
        patient_ids_sorted = sorted(patient_ids_matched)
        print(f"PatientIDs found: {len(patient_ids_sorted)}")
        if patient_ids_sorted:
            first_pid = patient_ids_sorted[0]
            last_pid = patient_ids_sorted[-1]
            print(f"1st PatientID (from sorted IDs): {first_pid}")
            print(f"Last PatientID (from sorted IDs): {last_pid}")
        else:
            # Should be rare, but keep behavior predictable.
            print("1st PatientID (from sorted IDs): UNKNOWN")
            print("Last PatientID (from sorted IDs): UNKNOWN")

        # Requested: min/max number of series folders (unique series dirs) per patient.
        if series_folders_by_patient:
            per_patient_series_counts = [(len(v), pid) for pid, v in series_folders_by_patient.items()]
            per_patient_series_counts.sort(key=lambda t: (t[0], t[1]))
            min_series_count, min_series_pid = per_patient_series_counts[0]
            max_series_count, max_series_pid = max(per_patient_series_counts, key=lambda t: (t[0], t[1]))
            print(f"Min series folders per patient: {min_series_count} (PatientID={min_series_pid})")
            print(f"Max series folders per patient: {max_series_count} (PatientID={max_series_pid})")
        else:
            print("Min series folders per patient: UNKNOWN (PatientID=UNKNOWN)")
            print("Max series folders per patient: UNKNOWN (PatientID=UNKNOWN)")

        # Requested: min/max number of DICOMs per *principal* series folder.
        # Definition (per user): principal series folder always has > 2 DICOMs.
        if dcm_count_by_series_folder:
            items = [
                (count, patient, study, series)
                for (patient, study, series), count in dcm_count_by_series_folder.items()
                if int(count) > 2
            ]
            if items:
                print(f"Principal series folders found: {len(items)}")
                items.sort(key=lambda t: (t[0], t[1], t[2], t[3]))
                min_dcms, min_dcms_pid, _min_dcms_study, min_dcms_series = items[0]
                max_dcms, max_dcms_pid, _max_dcms_study, max_dcms_series = max(items, key=lambda t: (t[0], t[1], t[2], t[3]))
                print(
                    f"Min DICOMs per principal series folder: {min_dcms} "
                    f"(PatientID={min_dcms_pid}, SeriesInstanceUID={min_dcms_series})"
                )
                print(
                    f"Max DICOMs per principal series folder: {max_dcms} "
                    f"(PatientID={max_dcms_pid}, SeriesInstanceUID={max_dcms_series})"
                )
                avg_dcms = sum(int(c) for (c, _p, _st, _s) in items) / len(items)
                print(f"Avg DICOMs per principal series folder: {avg_dcms:.4g}")

                # SliceThickness stats across principal series only.
                try:
                    import pydicom
                    from pydicom.tag import Tag

                    thickness_tag = Tag(0x0018, 0x0050)
                    thicknesses_mm: list[float] = []

                    for (_count, pid, st, ser) in items:
                        rep = representative_dicom_by_series_folder.get((pid, st, ser))
                        if not rep:
                            continue
                        try:
                            ds = pydicom.dcmread(rep, stop_before_pixels=True)
                        except Exception:
                            continue

                        raw = ds.get("SliceThickness", None)
                        if raw is None and hasattr(ds, "SliceThickness"):
                            raw = getattr(ds, "SliceThickness")
                        if raw in (None, "", " ") and thickness_tag in ds:
                            raw = ds[thickness_tag].value
                        if raw in (None, "", " "):
                            continue

                        try:
                            thicknesses_mm.append(float(raw))
                        except Exception:
                            s = str(raw)
                            m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
                            if not m:
                                continue
                            thicknesses_mm.append(float(m.group(0)))

                    if thicknesses_mm:
                        mn = min(thicknesses_mm)
                        mx = max(thicknesses_mm)
                        avg = sum(thicknesses_mm) / len(thicknesses_mm)
                        print(f"Min SliceThickness (mm) [principal series]: {mn:.4g}")
                        print(f"Max SliceThickness (mm) [principal series]: {mx:.4g}")
                        print(f"Avg SliceThickness (mm) [principal series]: {avg:.4g}")
                    else:
                        print("Min SliceThickness (mm) [principal series]: UNKNOWN")
                        print("Max SliceThickness (mm) [principal series]: UNKNOWN")
                        print("Avg SliceThickness (mm) [principal series]: UNKNOWN")
                except Exception:
                    print("Min SliceThickness (mm) [principal series]: UNKNOWN")
                    print("Max SliceThickness (mm) [principal series]: UNKNOWN")
                    print("Avg SliceThickness (mm) [principal series]: UNKNOWN")
            else:
                print("Principal series folders found: 0")
                print(
                    "Min DICOMs per principal series folder: UNKNOWN "
                    "(PatientID=UNKNOWN, SeriesInstanceUID=UNKNOWN)"
                )
                print(
                    "Max DICOMs per principal series folder: UNKNOWN "
                    "(PatientID=UNKNOWN, SeriesInstanceUID=UNKNOWN)"
                )
                print("Avg DICOMs per principal series folder: UNKNOWN")
                print("Min SliceThickness (mm) [principal series]: UNKNOWN")
                print("Max SliceThickness (mm) [principal series]: UNKNOWN")
                print("Avg SliceThickness (mm) [principal series]: UNKNOWN")
        else:
            print("Principal series folders found: 0")
            print(
                "Min DICOMs per principal series folder: UNKNOWN "
                "(PatientID=UNKNOWN, SeriesInstanceUID=UNKNOWN)"
            )
            print(
                "Max DICOMs per principal series folder: UNKNOWN "
                "(PatientID=UNKNOWN, SeriesInstanceUID=UNKNOWN)"
            )
            print("Avg DICOMs per principal series folder: UNKNOWN")
            print("Min SliceThickness (mm) [principal series]: UNKNOWN")
            print("Max SliceThickness (mm) [principal series]: UNKNOWN")
            print("Avg SliceThickness (mm) [principal series]: UNKNOWN")
    except Exception:
        # Do not fail the data fetch due to logging.
        print("PatientIDs found: UNKNOWN")
        print("1st PatientID (from sorted IDs): UNKNOWN")
        print("Last PatientID (from sorted IDs): UNKNOWN")
        print("Min series folders per patient: UNKNOWN (PatientID=UNKNOWN)")
        print("Max series folders per patient: UNKNOWN (PatientID=UNKNOWN)")
        print(
            "Min DICOMs per principal series folder: UNKNOWN "
            "(PatientID=UNKNOWN, SeriesInstanceUID=UNKNOWN)"
        )
        print(
            "Max DICOMs per principal series folder: UNKNOWN "
            "(PatientID=UNKNOWN, SeriesInstanceUID=UNKNOWN)"
        )
        print("Avg DICOMs per principal series folder: UNKNOWN")
        print("Principal series folders found: UNKNOWN")
        print("Min SliceThickness (mm) [principal series]: UNKNOWN")
        print("Max SliceThickness (mm) [principal series]: UNKNOWN")
        print("Avg SliceThickness (mm) [principal series]: UNKNOWN")

    if want_list:
        return matches

    if len(matches) != 1:
        raise RuntimeError(
            "Expected exactly one matching DICOM but found "
            f"{len(matches)}. First few: {matches[:5]}"
        )

    dicom_path = matches[0]
    print("Chose DICOM:")
    print(f"PatientID: {PatientID}")
    print(f"StudyInstanceUID index: {StudyInstanceUID_index}")
    print(f"SeriesNumber: {SeriesNumber}")
    print(f"InstanceNumber: {InstanceNumber}")
    print(f"path: {dicom_path}")
    return dicom_path


def summarize_patient_series_contents(
    base_dir: str = "../../data/raw/NSCLC-Radiomics",
    patient_id: str | None = None,
    series_double_check: bool = False,
    max_patients: int | None = None,
):
    """Print per-patient, per-series DICOM header summaries.

    Uses `get_dicom(..., base_dir=...)` to collect DICOM filepaths, then groups
    them into buckets by (PatientID folder, Study folder, Series folder).

    For each series bucket, prints:
    - Count of DICOM instances in that series
    - Modality (0008,0060)
    - SOPClassUID (0008,0016)

    Args:
        base_dir: Root folder containing patient directories.
        patient_id: Optional single patient folder to summarize.
        series_double_check: If True, scans *all* instances in the series and
            reports sets of observed values. If False, reads the first instance
            only (faster) and assumes series-level consistency.
        max_patients: Optional cap to limit output (useful for quick previews).
    """

    import pydicom

    dicom_paths = get_dicom(PatientID=patient_id, base_dir=base_dir)
    base_path = Path(base_dir).resolve()

    series_buckets: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    unparsed = 0

    for p in dicom_paths:
        p_path = Path(p).resolve()
        try:
            rel = p_path.relative_to(base_path)
        except Exception:
            # If base_dir resolution doesn't match (symlinks, etc.), fall back to string parsing.
            rel = Path(os.path.relpath(str(p_path), str(base_path)))

        parts = rel.parts
        # Expected: <patient>/<study>/<series>/<file>.dcm
        if len(parts) < 4:
            unparsed += 1
            continue

        patient = parts[0]
        study = parts[1]
        series = parts[2]
        series_buckets[(patient, study, series)].append(str(p_path))

    patients = sorted({k[0] for k in series_buckets.keys()})
    if max_patients is not None:
        patients = patients[:max_patients]

    print("\n=== PATIENT / SERIES DICOM SUMMARY ===")
    print(f"Base dir: {base_dir}")
    if patient_id is not None:
        print(f"Patient filter: {patient_id}")
    print(f"Patients found: {len(patients)}")
    if unparsed:
        print(f"Warning: skipped {unparsed} paths with unexpected layout")

    for patient in patients:
        patient_keys = [k for k in series_buckets.keys() if k[0] == patient]
        patient_keys.sort(key=lambda t: (t[1], t[2]))
        print("\n---")
        print(f"Patient: {patient} | Series buckets: {len(patient_keys)}")

        for (patient_k, study_k, series_k) in patient_keys:
            series_paths = series_buckets[(patient_k, study_k, series_k)]
            series_paths.sort()

            to_read = series_paths if series_double_check else [series_paths[0]]
            modalities: set[str] = set()
            sop_uids: set[str] = set()
            read_errors = 0

            for dcm_path in to_read:
                try:
                    ds = pydicom.dcmread(dcm_path, stop_before_pixels=True)
                except Exception:
                    read_errors += 1
                    continue

                mod = ds.get("Modality", None)
                sop = ds.get("SOPClassUID", None)

                modalities.add(str(mod) if mod not in (None, "", " ") else "MISSING")
                sop_uids.add(str(sop) if sop not in (None, "", " ") else "MISSING")

            sop_named = []
            for uid in sorted(sop_uids):
                name = _SOP_CLASS_UID_NAMES.get(uid)
                sop_named.append(f"{uid} ({name})" if name else uid)

            print(
                " | ".join(
                    [
                        f"study={study_k}",
                        f"series={series_k}",
                        f"instances={len(series_paths)}",
                        f"Modality={sorted(modalities)}",
                        f"SOPClassUID={sop_named}",
                        f"read_errors={read_errors}",
                    ]
                )
            )

def typical_slice_thickness(dicom_paths, double_check=False):
    import pydicom
    from pydicom.tag import Tag

    # Accept a single path or a list of paths.
    if isinstance(dicom_paths, (str, os.PathLike)):
        dicom_paths = [str(dicom_paths)]
    else:
        dicom_paths = [str(p) for p in dicom_paths]

    thicknesses_mm = []
    skipped = 0
    
    thicknesses_tag = Tag(0x0018, 0x0050)

    for dicom_path in dicom_paths:
        ds = pydicom.dcmread(dicom_path, stop_before_pixels=True)

        # Prefer direct keyword lookup; fall back to attribute access.
        raw = ds.get("SliceThickness", None)
        if raw is None and hasattr(ds, "SliceThickness"):
            raw = getattr(ds, "SliceThickness")

        # If keyword/attribute lookup failed (or is empty), fall back to explicit tag lookup.
        if raw in (None, "", " ") and thicknesses_tag in ds:
            raw = ds[thicknesses_tag].value

        if raw in (None, "", " "):
            skipped += 1
            continue

        try:
            # pydicom may give a DSfloat/DecimalString already.
            thickness_mm = float(raw)
        except Exception:
            # If user is seeing strings like "XX: 'Y'", extract numeric portion.
            s = str(raw)
            m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
            if not m:
                skipped += 1
                print(f"Could not parse SliceThickness from DICOM: {dicom_path}\n(raw='{s}')")
                continue
            thickness_mm = float(m.group(0))

        thicknesses_mm.append(thickness_mm)

    if not thicknesses_mm:
        raise ValueError(
            "No SliceThickness values found/parsed from provided DICOMs. "
            f"Inputs={len(dicom_paths)}, skipped={skipped}."
        )

    mn = min(thicknesses_mm)
    mx = max(thicknesses_mm)
    avg = sum(thicknesses_mm) / len(thicknesses_mm)

    print("\n=== SLICE THICKNESS SUMMARY (mm) ===")
    print(f"DICOMs scanned: {len(dicom_paths)}")
    print(f"Values parsed: {len(thicknesses_mm)}")
    print(f"Skipped (missing/unparseable): {skipped}")
    print(f"min: {mn:.4g}")
    print(f"max: {mx:.4g}")
    print(f"avg: {avg:.4g}")

    # Plot distribution in three bins: 0-2, 2-4, 4-6 mm.
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("matplotlib not installed; skipping plot.")
        return thicknesses_mm

    bins = [0, 2, 4, 6]
    plt.figure()
    plt.hist(thicknesses_mm, bins=bins, edgecolor="black")
    plt.xlabel("SliceThickness (mm)")
    plt.ylabel("Count")
    plt.title("SliceThickness distribution")
    plt.xticks(bins)
    plt.tight_layout()
    plt.show()

    return thicknesses_mm

def audit_dicom_header(
    dicom_path,
    output_json="dicom_header_audit.json",
    *,
    config=None,
    header_cache: DicomHeaderCache | None = None,
    write_json: bool = True,
):
    """Audit one DICOM or many DICOMs.

    - If `dicom_path` is a list/tuple, audits each path in sequence and writes
      one combined JSON file to `output_json`.
    - By default, prints only when violations are found, and avoids printing
      raw values.

    Advanced behavior can be controlled via an optional `config` argument.
    """

    import hashlib

    import pydicom

    cfg = config if config is not None else AuditConfig()

    def _hash_value(s: str) -> str:
        return hashlib.sha256(s.encode("utf-8", errors="replace")).hexdigest()

    def _maybe_redact(value: str):
        if not cfg.redact_values:
            return value
        value = "" if value is None else str(value)
        out = {"sha256": _hash_value(value)}
        if cfg.max_value_preview and cfg.max_value_preview > 0:
            out["preview"] = value[: cfg.max_value_preview]
        return out

    def _iter_elements(ds):
        for elem in ds:
            yield elem
            acknowledging_sequence = cfg.scan_sequences and getattr(elem, "VR", None) == "SQ"
            if not acknowledging_sequence:
                continue
            try:
                for item in elem.value:
                    yield from _iter_elements(item)
            except Exception:
                continue

    def _audit_one(path: str):
        if header_cache is not None:
            ds = dcmread_cached(path, header_cache, stop_before_pixels=True)
        else:
            ds = pydicom.dcmread(path, stop_before_pixels=True)

        header_dict = {} if cfg.include_all_headers else None
        phi_flags = []
        private_flags = []
        flags = []

        for elem in _iter_elements(ds):
            tag_tuple = (elem.tag.group, elem.tag.element)
            keyword = elem.keyword if elem.keyword else str(elem.tag)
            value_str = str(elem.value)

            if cfg.include_all_headers:
                header_dict[keyword] = _maybe_redact(value_str)

            is_phi_tag = tag_tuple in PHI_TAGS
            is_private = bool(getattr(elem.tag, "is_private", False))
            is_private_risk = cfg.flag_private_tags and is_private

            if is_phi_tag or is_private_risk:
                entry = {
                    "tag": f"({elem.tag.group:04X},{elem.tag.element:04X})",
                    "keyword": keyword,
                    "reason": "PHI_TAGS" if is_phi_tag else "PRIVATE_TAG",
                    "value": _maybe_redact(value_str),
                }
                flags.append(entry)
                if is_phi_tag:
                    phi_flags.append(entry)
                else:
                    private_flags.append(entry)

        phi_count = len(phi_flags)
        private_count = len(private_flags)
        total_found = phi_count + private_count

        # Print only if there are violations (and do NOT print raw values)
        if (not cfg.print_only_violations) or (total_found > 0):
            print(f"\nFile: {path}")
            print(f"TOTAL TAGS found: {total_found}")
            print(f"PHI_TAGS found: {phi_count}")
            if cfg.print_tag_details and phi_count:
                for f in phi_flags:
                    # Prefer "NAME (gggg,eeee)"; avoid duplicates if keyword is already the tag.
                    if f.get("keyword") and f["keyword"] != f["tag"]:
                        print(f"PHI_TAG {f['keyword']} {f['tag']}")
                    else:
                        print(f"PHI_TAG {f['tag']}")
            print(f"PRIVATE_TAGS found: {private_count}")
            if cfg.print_tag_details and private_count:
                for f in private_flags:
                    # Private tags often have no standard keyword; avoid printing "(tag) (tag)".
                    if f.get("keyword") and f["keyword"] != f["tag"]:
                        print(f"PRIVATE_TAG {f['keyword']} {f['tag']}")
                    else:
                        print(f"PRIVATE_TAG {f['tag']}")

        audit_output = {
            "file": str(path),
            "total_attributes": len(ds),
            "total_tag_count": total_found,
            "phi_tag_count": phi_count,
            "private_tag_count": private_count,
            "potential_phi": flags,
        }
        if cfg.include_all_headers:
            audit_output["all_headers"] = header_dict
        return audit_output

    # Normalize input to list vs single
    if isinstance(dicom_path, (list, tuple)):
        paths = [str(p) for p in dicom_path]
        per_file = []
        for path in paths:
            try:
                per_file.append(_audit_one(path))
            except Exception as e:
                per_file.append({"file": str(path), "error": str(e)})

        combined = {
            "total_files": len(paths),
            "files": [x.get("file") for x in per_file],
            "audits": per_file,
            "config": {
                "include_all_headers": cfg.include_all_headers,
                "scan_sequences": cfg.scan_sequences,
                "flag_private_tags": cfg.flag_private_tags,
                "redact_values": cfg.redact_values,
                "print_only_violations": cfg.print_only_violations,
                "print_tag_details": cfg.print_tag_details,
                "max_value_preview": cfg.max_value_preview,
            },
        }
        if write_json:
            Path(output_json).write_text(json.dumps(combined, indent=2))
        if not cfg.print_only_violations:
            print(f"\nAudit JSON exported to {output_json}")
        return combined

    single = _audit_one(str(dicom_path))
    if write_json:
        Path(output_json).write_text(json.dumps(single, indent=2))
    if not cfg.print_only_violations:
        print(f"\nAudit JSON exported to {output_json}")
    return single


def find_tumor_mask_dicoms(
    dicom_path,
    output_json="dicom_mask_audit.json",
    *,
    config: MaskFinderConfig | None = None,
    header_cache: DicomHeaderCache | None = None,
    write_json: bool = True,
    export_fhir: bool = True,
):
    """Scan one DICOM or many DICOMs and identify likely tumor mask objects.

    This function is header-only (uses `stop_before_pixels=True`) and classifies
    DICOMs as:
    - RTSTRUCT: RT Structure Set Storage and/or Modality=RTSTRUCT, with ROI sequences
    - SEG: Segmentation Storage and/or Modality=SEG, with Segment sequences

    It additionally tries to detect *tumor-related* content by examining ROI/segment
    labels (e.g. GTV/ITV/PTV/tumor) when those are present in headers.

    Accepts a single path or a list/tuple of paths; writes one combined JSON file.
    """

    import hashlib
    import re

    import pydicom
    from pydicom.tag import Tag

    cfg = config if config is not None else MaskFinderConfig()

    output_json_path = Path(output_json) if output_json is not None else None

    # Common tumor-ish label heuristics seen clinically.
    tumor_label_re = re.compile(
        r"(tumou?r|gtv|itv|ptv|lesion|mass|nodule|primary|gross)",
        re.IGNORECASE,
    )

    RTSTRUCT_UID = "1.2.840.10008.5.1.4.1.1.481.3"
    SEG_UID = "1.2.840.10008.5.1.4.1.1.66.4"

    # Tags we key off for structural confidence.
    T_SOP = Tag(0x0008, 0x0016)
    T_MOD = Tag(0x0008, 0x0060)
    T_SERIES_DESC = Tag(0x0008, 0x103E)
    T_SOP_INSTANCE_UID = Tag(0x0008, 0x0018)
    T_ROI_SEQ = Tag(0x3006, 0x0020)
    T_ROI_NAME = Tag(0x3006, 0x0026)
    T_SEG_SEQ = Tag(0x0062, 0x0002)
    T_SEG_LABEL = Tag(0x0062, 0x0005)

    # Tags used to link mask objects back to source images/series.
    T_SERIES_INSTANCE_UID = Tag(0x0020, 0x000E)
    T_REFERENCED_SOP_INSTANCE_UID = Tag(0x0008, 0x1155)

    class PatientIdPathError(ValueError):
        pass

    def _patient_and_study_from_path(path: str) -> tuple[str, str]:
        """Return (patient_id, study_instance_uid) from the Lung1 folder layout.

        Expected layout:
        .../<PatientID>/<StudyInstanceUID>/<SeriesInstanceUID>/<instance>.dcm

        PatientID is defined as the *parent folder of the StudyInstanceUID folder*.
        Hard-fail if PatientID doesn't contain 'LUNG1' (case-insensitive).
        """

        p = Path(path)
        try:
            study_instance_uid = p.parents[1].name
            patient_id = p.parents[2].name
        except Exception:
            raise PatientIdPathError(f"Path too short to extract PatientID/StudyInstanceUID: {path}")

        if not patient_id or "lung1" not in patient_id.lower():
            raise PatientIdPathError(
                f"Expected PatientID folder containing 'LUNG1' but got '{patient_id}' for path: {path}"
            )

        if not study_instance_uid:
            raise PatientIdPathError(f"Empty StudyInstanceUID folder name for path: {path}")

        return patient_id, study_instance_uid

    def _hash_value(s: str) -> str:
        return hashlib.sha256(s.encode("utf-8", errors="replace")).hexdigest()

    def _maybe_redact(value: str):
        if not cfg.redact_values:
            return value
        value = "" if value is None else str(value)
        out = {"sha256": _hash_value(value)}
        if cfg.max_value_preview and cfg.max_value_preview > 0:
            out["preview"] = value[: cfg.max_value_preview]
        return out

    def _iter_elements(ds):
        for elem in ds:
            yield elem
            if not (cfg.deep_tag_search and cfg.scan_sequences and getattr(elem, "VR", None) == "SQ"):
                continue
            try:
                for item in elem.value:
                    yield from _iter_elements(item)
            except Exception:
                continue

    def _tag_str(tag: Tag) -> str:
        return f"({tag.group:04X},{tag.element:04X})"

    def _get_str(ds, tag: Tag):
        if tag in ds:
            try:
                return str(ds[tag].value)
            except Exception:
                return str(ds[tag])
        return None

    def _collect_labels(ds):
        """Collect ROI/segment labels from RTSTRUCT/SEG headers.

        This may miss some data if it only exists inside pixel data, but should
        capture most clinically meaningful label strings.
        """
        labels = set()
        for elem in _iter_elements(ds):
            if elem.tag == T_ROI_NAME or elem.tag == T_SEG_LABEL:
                try:
                    labels.add(str(elem.value))
                except Exception:
                    labels.add(str(elem))
        return sorted(labels)

    def _collect_referenced_series_instance_uid(ds, source_path: str):
        """Return the referenced SeriesInstanceUID for a SEG/RTSTRUCT.

        Preferred: find a non-root (0020,000E) somewhere in sequences.
        Fallback: if series UID isn't present, use a referenced SOPInstanceUID (0008,1155)
        and resolve its SeriesInstanceUID by scanning DICOMs in the same patient/study.

        Assumption for this dataset: each mask refers to at most one source series.
        """

        if not cfg.scan_sequences:
            return None

        # Lightweight recursion through sequence elements.
        def _walk(dataset):
            for elem in dataset:
                yield elem
                if getattr(elem, "VR", None) != "SQ":
                    continue
                try:
                    for item in elem.value:
                        yield from _walk(item)
                except Exception:
                    continue

        try:
            root_series_uid = ds.get("SeriesInstanceUID", None)
            root_series_uid = str(root_series_uid) if root_series_uid not in (None, "", " ") else None
        except Exception:
            root_series_uid = None

        # 1) Try direct SeriesInstanceUID references in sequences.
        for elem in _walk(ds):
            try:
                if elem.tag != T_SERIES_INSTANCE_UID:
                    continue
                v = str(elem.value)
                if not v:
                    continue
                if root_series_uid and v == root_series_uid:
                    continue
                return v
            except Exception:
                continue

        # 2) Fall back to a referenced SOPInstanceUID -> resolve to SeriesInstanceUID.
        referenced_sop_uid = None
        for elem in _walk(ds):
            try:
                if elem.tag != T_REFERENCED_SOP_INSTANCE_UID:
                    continue
                v = str(elem.value)
                if v:
                    referenced_sop_uid = v
                    break
            except Exception:
                continue

        if not referenced_sop_uid:
            return None

        # Expected layout: .../<PatientID>/<StudyInstanceUID>/<SeriesInstanceUID>/<instance>.dcm
        p = Path(source_path)
        try:
            study_dir = p.parents[1]
        except Exception:
            return None

        # Search all DICOMs in the study folder for a matching SOPInstanceUID.
        # This is only used when the series UID isn't directly present in the mask header.
        try:
            for candidate in glob.glob(str(study_dir / "*" / "*.dcm")):
                try:
                    if header_cache is not None:
                        ds_src = dcmread_cached(candidate, header_cache, stop_before_pixels=True)
                    else:
                        ds_src = pydicom.dcmread(candidate, stop_before_pixels=True)
                except Exception:
                    continue
                sop_uid = ds_src.get("SOPInstanceUID", None)
                sop_uid = str(sop_uid) if sop_uid not in (None, "", " ") else None
                if sop_uid != referenced_sop_uid:
                    continue
                series_uid = ds_src.get("SeriesInstanceUID", None)
                series_uid = str(series_uid) if series_uid not in (None, "", " ") else None
                return series_uid
        except Exception:
            return None

        return None

    def _audit_one(path: str):
        if header_cache is not None:
            ds = dcmread_cached(path, header_cache, stop_before_pixels=True)
        else:
            ds = pydicom.dcmread(path, stop_before_pixels=True)

        patient_id, study_instance_uid_path = _patient_and_study_from_path(path)

        sop = _get_str(ds, T_SOP)
        sop_instance_uid = _get_str(ds, T_SOP_INSTANCE_UID)
        modality = _get_str(ds, T_MOD)
        series_desc = _get_str(ds, T_SERIES_DESC)

        # The mask object's own SeriesInstanceUID (not the referenced source series).
        # This should exist in all valid DICOMs; fall back to folder name for robustness.
        try:
            series_instance_uid = ds.get("SeriesInstanceUID", None)
            series_instance_uid = str(series_instance_uid) if series_instance_uid not in (None, "", " ") else None
        except Exception:
            series_instance_uid = None
        if not series_instance_uid:
            try:
                # Expected: .../<PatientID>/<StudyInstanceUID>/<SeriesInstanceUID>/<instance>.dcm
                series_instance_uid = Path(path).parents[0].name
            except Exception:
                series_instance_uid = None

        # Streamlined detection: this dataset is clean; we key solely off Modality.
        is_rtstruct = modality == "RTSTRUCT"
        is_seg = modality == "SEG"

        is_mask_object = bool(is_rtstruct or is_seg)
        mask_type = "RTSTRUCT" if is_rtstruct else ("SEG" if is_seg else None)

        deep_tag_search_completed = 1 if cfg.deep_tag_search else 0

        # Optional deep tag search for labels/tags/references.
        if cfg.deep_tag_search and is_mask_object:
            labels = _collect_labels(ds)
            tumor_labels = [s for s in labels if tumor_label_re.search(s or "")]
            referenced_series_instance_uid = _collect_referenced_series_instance_uid(ds, path)
        else:
            labels = []
            tumor_labels = []
            referenced_series_instance_uid = _collect_referenced_series_instance_uid(ds, path)

        # Track which MASK_TAGS were present (deep search only).
        present_mask_tags: list[dict] = []
        if cfg.deep_tag_search and is_mask_object:
            present_tag_tuples = set()
            for elem in _iter_elements(ds):
                t = (elem.tag.group, elem.tag.element)
                if t in MASK_TAGS and t not in present_tag_tuples:
                    present_tag_tuples.add(t)
                    keyword = elem.keyword if elem.keyword else MASK_TAGS.get(t, str(elem.tag))
                    present_mask_tags.append(
                        {
                            "tag": f"({elem.tag.group:04X},{elem.tag.element:04X})",
                            "keyword": keyword,
                        }
                    )

        total_mask_tag_count = len(present_mask_tags)

        mask_found = 1 if is_mask_object else 0
        seg_found = 1 if is_seg else 0
        rtstruct_found = 1 if is_rtstruct else 0

        # Print in the same concise style as your other audits.
        if (not cfg.print_only_matches) or is_mask_object:
            print(f"\nFile: {path}")
            print(f"SEG found: {1 if is_seg else 0}")
            print(f"RTSTRUCT found: {1 if is_rtstruct else 0}")
            print(f"Deep tag search completed: {deep_tag_search_completed}")
            print(f"ALT MASK TAGS found: {total_mask_tag_count}")

            if cfg.print_tag_details and total_mask_tag_count:
                for t in present_mask_tags:
                    print(f"MASK_TAG {t['keyword']} {t['tag']}")
            

            if is_mask_object:
                # Requested: print/log patient ID (folder name) immediately above referenced series info.
                print(f"PatientID: {patient_id}")
                print(f"StudyInstanceUID: {study_instance_uid_path}")
                print(f"Scan sequences: {1 if cfg.scan_sequences else 0}")
                print(f"Referenced SeriesInstanceUID: {referenced_series_instance_uid}")
                print(f"SeriesInstanceUID: {series_instance_uid}")

            

        audit_output = {
            "file": str(path),
            "patient_id": patient_id,
            "study_instance_uid": study_instance_uid_path,
            "sop_class_uid": sop,
            "sop_instance_uid": sop_instance_uid,
            "sop_class_name": _SOP_CLASS_UID_NAMES.get(sop) if sop else None,
            "modality": modality,
            "series_description": series_desc,
            "series_instance_uid": series_instance_uid,
            "referenced_series_instance_uid": referenced_series_instance_uid,
            "has_reference": bool(referenced_series_instance_uid),
            "deep_tag_search_completed": deep_tag_search_completed,
            "total_mask_tag_count": total_mask_tag_count,
            "mask_tag_names": [t["keyword"] for t in present_mask_tags],
            "mask_tags_present": present_mask_tags,
            "is_rtstruct": is_rtstruct,
            "is_seg": is_seg,
            "mask_type": mask_type,
            "mask_found": mask_found,
            "seg_found": seg_found,
            "rtstruct_found": rtstruct_found,
            "labels_found": [_maybe_redact(x) for x in labels],
            "tumor_like_labels_found": [_maybe_redact(x) for x in tumor_labels],
            "tumor_like_label_count": len(tumor_labels),
        }
        return audit_output

    # Normalize input to list vs single
    if isinstance(dicom_path, (list, tuple)):
        paths = [str(p) for p in dicom_path]
    else:
        paths = [str(dicom_path)]

    per_file = []
    for path in paths:
        try:
            per_file.append(_audit_one(path))
        except PatientIdPathError:
            raise
        except Exception as e:
            per_file.append({"file": str(path), "error": str(e)})

    seg_files = [x["file"] for x in per_file if x.get("mask_type") == "SEG"]
    rtstruct_files = [x["file"] for x in per_file if x.get("mask_type") == "RTSTRUCT"]
    mask_files = rtstruct_files + seg_files

    # Patient-level summary: which patients have no detected mask objects in the scanned files.
    all_patient_ids = sorted({a.get("patient_id") for a in per_file if a.get("patient_id")})
    patients_with_masks = {
        a.get("patient_id")
        for a in per_file
        if a.get("patient_id") and a.get("mask_type") in ("SEG", "RTSTRUCT")
    }
    patients_without_masks = sorted(set(all_patient_ids) - set(patients_with_masks))
    patients_without_masks_count = len(patients_without_masks)
    patients_without_masks_first = patients_without_masks[0] if patients_without_masks else None
    patients_without_masks_last = patients_without_masks[-1] if patients_without_masks else None

    # Patient-level summary: which patients have no SEG detected in the scanned files.
    patients_with_seg = {
        a.get("patient_id")
        for a in per_file
        if a.get("patient_id") and a.get("mask_type") == "SEG"
    }
    patients_without_seg = sorted(set(all_patient_ids) - set(patients_with_seg))
    patients_without_seg_count = len(patients_without_seg)
    patients_without_seg_first = patients_without_seg[0] if patients_without_seg else None
    patients_without_seg_last = patients_without_seg[-1] if patients_without_seg else None

    # Aggregate: referenced SeriesInstanceUID -> mask files.
    masks_by_referenced_series: dict[str, list[str]] = {}
    for a in per_file:
        if a.get("mask_type") not in ("SEG", "RTSTRUCT"):
            continue
        uid = a.get("referenced_series_instance_uid")
        if uid:
            masks_by_referenced_series.setdefault(uid, []).append(a.get("file"))

    if cfg.print_summary:
        print(f"\nTOTAL MASK FILES found: {len(mask_files)}")
        print(f"SEG files found: {len(seg_files)}")
        print(f"RTSTRUCT files found: {len(rtstruct_files)}")
        print(f"PatientIDs with NO tumor mask info found: {patients_without_masks_count}")
        print(f"1st PatientID with NO tumor mask info: {patients_without_masks_first}")
        print(f"Last PatientID with NO tumor mask info: {patients_without_masks_last}")
        print(f"PatientIDs with NO SEG found: {patients_without_seg_count}")
        print(f"1st PatientID with NO SEG found: {patients_without_seg_first}")
        print(f"Last PatientID with NO SEG found: {patients_without_seg_last}")
    combined = {
        "total_files": len(paths),
        "mask_file_count": len(mask_files),
        "seg_file_count": len(seg_files),
        "rtstruct_file_count": len(rtstruct_files),
        "patients_without_tumor_mask_info_count": patients_without_masks_count,
        "patients_without_tumor_mask_info_first_patient_id": patients_without_masks_first,
        "patients_without_tumor_mask_info_last_patient_id": patients_without_masks_last,
        "patients_without_seg_found_count": patients_without_seg_count,
        "patients_without_seg_found_first_patient_id": patients_without_seg_first,
        "patients_without_seg_found_last_patient_id": patients_without_seg_last,
        "audits": per_file,
        "config": {
            "scan_sequences": cfg.scan_sequences,
            "deep_tag_search": cfg.deep_tag_search,
            "redact_values": cfg.redact_values,
            "print_only_matches": cfg.print_only_matches,
            "print_tag_details": cfg.print_tag_details,
            "print_summary": cfg.print_summary,
            "max_value_preview": cfg.max_value_preview,
        },
    }

    if write_json:
        output_json_path.write_text(json.dumps(combined, indent=2))

    # Emit FHIR R4 Bundle (collection) alongside the native JSON.
    if export_fhir:
        if output_json_path is None:
            raise ValueError("export_fhir=True requires output_json to be a valid path")
        fhir_path = output_json_path.with_name(output_json_path.stem + ".fhir.bundle.json")
        export_mask_audit_as_fhir_r4_bundle(combined, fhir_path, patient_identifier_system="urn:dicom:patientid")
        combined["fhir_r4_bundle_file"] = str(fhir_path)
        if write_json:
            output_json_path.write_text(json.dumps(combined, indent=2))
    return combined


def export_unified_audit_as_fhir_r4_bundle(
    unified_audit: dict,
    output_json: str | os.PathLike,
    *,
    patient_identifier_system: str = "urn:dicom:patientid",
) -> dict:
    """Export a FHIR R4 Bundle(type=collection) for unified audit results."""

    timestamp = _fhir_now()

    by_patient_container = unified_audit.get("by_patient_id", {}) or {}
    if not (isinstance(by_patient_container, dict) and isinstance(by_patient_container.get("patient_id"), dict)):
        raise ValueError(
            "Unified audit must have by_patient_id shaped as: { 'patient_id': { '<PatientID>': { ... } } }"
        )
    by_patient_id = by_patient_container["patient_id"]

    patient_resources: dict[str, dict] = {}
    imagingstudy_resources: dict[tuple[str, str], dict] = {}  # (patient_id, study_uid)
    observation_resources: list[dict] = []

    def _dicom_uid_identifier(uid: str) -> dict:
        return {"system": "urn:dicom:uid", "value": f"urn:oid:{uid}"}

    def _modality_coding(modality: str | None) -> dict | None:
        if modality in (None, "", " "):
            return None
        return {"system": "http://dicom.nema.org/resources/ontology/DCM", "code": str(modality)}

    def _ensure_series(imagingstudy: dict, series_uid: str | None, modality: str | None):
        if not series_uid:
            return
        existing = {s.get("uid") for s in imagingstudy.get("series", []) if isinstance(s, dict)}
        if series_uid in existing:
            return
        s = {"uid": str(series_uid)}
        mod = _modality_coding(modality)
        if mod:
            s["modality"] = mod
        imagingstudy.setdefault("series", []).append(s)

    def _get_patient_and_study(a: dict) -> tuple[str | None, str | None]:
        pid = a.get("patient_id")
        study = a.get("study_instance_uid")
        return pid, study

    def _ensure_patient_and_study(pid: str, study_uid: str, modality: str | None, series_uid: str | None):
        if pid not in patient_resources:
            patient_resources[pid] = {
                "resourceType": "Patient",
                "id": _fhir_id("patient", str(pid)),
                "identifier": [{"system": patient_identifier_system, "value": str(pid)}],
            }

        patient_ref = {"reference": f"Patient/{patient_resources[pid]['id']}"}
        key = (str(pid), str(study_uid))
        if key not in imagingstudy_resources:
            imagingstudy_resources[key] = {
                "resourceType": "ImagingStudy",
                "id": _fhir_id("imagingstudy", f"{pid}|{study_uid}"),
                "status": "available",
                "subject": patient_ref,
                "identifier": [_dicom_uid_identifier(str(study_uid))],
                "started": timestamp,
                "series": [],
            }

        imagingstudy = imagingstudy_resources[key]
        _ensure_series(imagingstudy, series_uid, modality)
        return patient_ref, {"reference": f"ImagingStudy/{imagingstudy['id']}"}

    def _add_source_path_extension(obs: dict, file_path: str | None):
        if not file_path:
            return
        obs.setdefault("extension", []).append(
            {
                "url": "http://example.org/fhir/StructureDefinition/source-file-path",
                "valueString": str(file_path),
            }
        )

    for pid, patient_payload in by_patient_id.items():
        if not isinstance(patient_payload, dict):
            continue

        # Ensure Patient resource exists even if we emit only patient-level summary.
        if pid not in patient_resources:
            patient_resources[str(pid)] = {
                "resourceType": "Patient",
                "id": _fhir_id("patient", str(pid)),
                "identifier": [{"system": patient_identifier_system, "value": str(pid)}],
            }
        patient_ref = {"reference": f"Patient/{patient_resources[str(pid)]['id']}"}

        # Patient-level principal series summary (derived from dataset layout).
        ps = patient_payload.get("principal_series") if isinstance(patient_payload.get("principal_series"), dict) else None
        has_ps = bool(ps.get("has_principal_series")) if ps else False
        ps_uid = ps.get("series_instance_uid") if ps else None
        ps_n = ps.get("num_dicoms") if ps else None

        ps_components: list[dict] = []

        def _c_ps(code: str, value_key: str, value):
            ps_components.append(
                {
                    "code": {
                        "coding": [
                            {
                                "system": "http://example.org/fhir/CodeSystem/dicom-audit",
                                "code": code,
                            }
                        ]
                    },
                    value_key: value,
                }
            )

        _c_ps("hasPrincipalSeries", "valueBoolean", bool(has_ps))
        if ps_uid not in (None, "", " "):
            _c_ps("principalSeriesInstanceUID", "valueString", str(ps_uid))
        if ps_n is not None:
            try:
                _c_ps("principalSeriesDicomCount", "valueInteger", int(ps_n))
            except Exception:
                pass

        observation_resources.append(
            {
                "resourceType": "Observation",
                "id": _fhir_id("obs", f"principal-series|{pid}"),
                "status": "final",
                "category": [
                    {
                        "coding": [
                            {
                                "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                                "code": "imaging",
                                "display": "Imaging",
                            }
                        ]
                    }
                ],
                "code": {
                    "coding": [
                        {
                            "system": "http://example.org/fhir/CodeSystem/dicom-audit",
                            "code": "dicom-principal-series",
                            "display": "DICOM Principal Series Summary",
                        }
                    ],
                    "text": "DICOM Principal Series Summary",
                },
                "subject": patient_ref,
                "effectiveDateTime": timestamp,
                "component": ps_components,
            }
        )

        files = patient_payload.get("files", []) or []
        for row in files:
            if not isinstance(row, dict):
                continue
            file_path = row.get("file")

            study_uid = row.get("study_instance_uid")
            if not pid or not study_uid:
                continue

            dicom_meta = row.get("dicom") or {}
            modality = None
            series_uid = row.get("series_instance_uid")  # dataset layout: series folder name is SeriesInstanceUID
            sop_instance_uid = row.get("sop_instance_uid")

            mask = row.get("mask_audit") or {}
            referenced_series_uid = mask.get("referenced_series_instance_uid")

            patient_ref, imagingstudy_ref = _ensure_patient_and_study(str(pid), str(study_uid), modality, series_uid)
            if referenced_series_uid:
                imagingstudy = imagingstudy_resources[(str(pid), str(study_uid))]
                _ensure_series(imagingstudy, str(referenced_series_uid), None)

            # 1) PHI audit observation (counts only)
            header_audit = row.get("header_audit") or {}
            if isinstance(header_audit, dict) and (
                "phi_tag_count" in header_audit or "private_tag_count" in header_audit
            ):
                components = []

                def _c(code: str, value_key: str, value):
                    components.append(
                        {
                            "code": {
                                "coding": [
                                    {
                                        "system": "http://example.org/fhir/CodeSystem/dicom-audit",
                                        "code": code,
                                    }
                                ]
                            },
                            value_key: value,
                        }
                    )

                _c("phiTagCount", "valueInteger", int(header_audit.get("phi_tag_count", 0) or 0))
                _c("privateTagCount", "valueInteger", int(header_audit.get("private_tag_count", 0) or 0))
                _c("potentialPhiTagCount", "valueInteger", int(header_audit.get("total_tag_count", 0) or 0))

                obs = {
                    "resourceType": "Observation",
                    "id": _fhir_id("obs", f"phi|{pid}|{study_uid}|{file_path}"),
                    "status": "final",
                    "category": [
                        {
                            "coding": [
                                {
                                    "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                                    "code": "imaging",
                                    "display": "Imaging",
                                }
                            ]
                        }
                    ],
                    "code": {
                        "coding": [
                            {
                                "system": "http://example.org/fhir/CodeSystem/dicom-audit",
                                "code": "dicom-phi-audit",
                                "display": "DICOM PHI/Private Tag Audit",
                            }
                        ],
                        "text": "DICOM PHI/Private Tag Audit",
                    },
                    "subject": patient_ref,
                    "effectiveDateTime": timestamp,
                    "derivedFrom": [imagingstudy_ref],
                    "component": components,
                }
                _add_source_path_extension(obs, file_path)
                observation_resources.append(obs)

            # 2) Slice thickness observation (one per file when parseable)
            st = dicom_meta.get("slice_thickness") or {}
            st_mm = st.get("mm")
            if st_mm is not None:
                components = []

                def _c(code: str, value_key: str, value):
                    components.append(
                        {
                            "code": {
                                "coding": [
                                    {
                                        "system": "http://example.org/fhir/CodeSystem/dicom-audit",
                                        "code": code,
                                    }
                                ]
                            },
                            value_key: value,
                        }
                    )

                if sop_instance_uid:
                    _c("sopInstanceUID", "valueString", str(sop_instance_uid))
                if series_uid:
                    _c("seriesInstanceUID", "valueString", str(series_uid))
                if modality:
                    _c("modality", "valueString", str(modality))

                obs = {
                    "resourceType": "Observation",
                    "id": _fhir_id("obs", f"thickness|{pid}|{study_uid}|{file_path}"),
                    "status": "final",
                    "category": [
                        {
                            "coding": [
                                {
                                    "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                                    "code": "imaging",
                                    "display": "Imaging",
                                }
                            ]
                        }
                    ],
                    "code": {
                        "coding": [
                            {
                                "system": "http://example.org/fhir/CodeSystem/dicom-audit",
                                "code": "sliceThickness",
                                "display": "DICOM SliceThickness (0018,0050)",
                            }
                        ],
                        "text": "DICOM SliceThickness (0018,0050)",
                    },
                    "subject": patient_ref,
                    "effectiveDateTime": timestamp,
                    "derivedFrom": [imagingstudy_ref],
                    "valueQuantity": {
                        "value": float(st_mm),
                        "unit": "mm",
                        "system": "http://unitsofmeasure.org",
                        "code": "mm",
                    },
                    "component": components,
                }
                _add_source_path_extension(obs, file_path)
                observation_resources.append(obs)

            # 3) Mask observation (for mask-like objects only)
            if isinstance(mask, dict) and mask.get("mask_type") in ("SEG", "RTSTRUCT") and not mask.get("error"):
                series_desc = mask.get("series_description")
                series_desc_hash = _sha256_hex(series_desc) if series_desc not in (None, "", " ") else None

                deep_completed = bool(mask.get("deep_tag_search_completed", False))
                scan_sequences_enabled = bool(
                    unified_audit.get("config", {}).get("mask", {}).get("scan_sequences", True)
                )

                components = []

                def _c(code: str, value_key: str, value):
                    components.append(
                        {
                            "code": {
                                "coding": [
                                    {
                                        "system": "http://example.org/fhir/CodeSystem/dicom-audit",
                                        "code": code,
                                    }
                                ]
                            },
                            value_key: value,
                        }
                    )

                if modality:
                    _c("modality", "valueString", str(modality))
                if mask.get("sop_class_uid"):
                    _c("sopClassUID", "valueString", str(mask.get("sop_class_uid")))
                if mask.get("sop_instance_uid"):
                    _c("sopInstanceUID", "valueString", str(mask.get("sop_instance_uid")))
                _c("isSEG", "valueBoolean", bool(mask.get("is_seg")))
                _c("isRTSTRUCT", "valueBoolean", bool(mask.get("is_rtstruct")))
                _c("maskFound", "valueBoolean", bool(mask.get("mask_found")))

                _c("scanSequencesEnabled", "valueBoolean", bool(scan_sequences_enabled))
                _c("deepTagSearchCompleted", "valueBoolean", bool(deep_completed))

                if series_desc_hash:
                    _c("seriesDescriptionHash", "valueString", series_desc_hash)
                if series_uid:
                    _c("maskSeriesInstanceUID", "valueString", str(series_uid))
                if referenced_series_uid:
                    _c("referencedSeriesInstanceUID", "valueString", str(referenced_series_uid))

                _c("tumorLikeLabelCount", "valueInteger", int(mask.get("tumor_like_label_count", 0) or 0))

                if deep_completed:
                    labels = mask.get("labels_found") or []
                    tumor_labels = mask.get("tumor_like_labels_found") or []

                    def _extract_hashes(xs):
                        out = []
                        for x in xs:
                            if isinstance(x, dict) and "sha256" in x:
                                out.append(x["sha256"])
                            elif isinstance(x, str):
                                out.append(_sha256_hex(x))
                        return out

                    _c("labelHashesSHA256", "valueString", json.dumps(_extract_hashes(labels)))
                    _c("tumorLikeLabelHashesSHA256", "valueString", json.dumps(_extract_hashes(tumor_labels)))

                obs = {
                    "resourceType": "Observation",
                    "id": _fhir_id("obs", f"mask|{pid}|{study_uid}|{file_path}"),
                    "status": "final",
                    "category": [
                        {
                            "coding": [
                                {
                                    "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                                    "code": "imaging",
                                    "display": "Imaging",
                                }
                            ]
                        }
                    ],
                    "code": {
                        "coding": [
                            {
                                "system": "http://example.org/fhir/CodeSystem/dicom-audit",
                                "code": "dicom-mask-audit",
                                "display": "DICOM SEG/RTSTRUCT Mask Audit",
                            }
                        ],
                        "text": "DICOM SEG/RTSTRUCT Mask Audit",
                    },
                    "subject": patient_ref,
                    "effectiveDateTime": timestamp,
                    "derivedFrom": [imagingstudy_ref],
                    "component": components,
                }
                _add_source_path_extension(obs, file_path)
                observation_resources.append(obs)

    entries: list[dict] = []
    for res in patient_resources.values():
        entries.append({"fullUrl": f"urn:uuid:{res['id']}", "resource": res})
    for res in imagingstudy_resources.values():
        entries.append({"fullUrl": f"urn:uuid:{res['id']}", "resource": res})
    for res in observation_resources:
        entries.append({"fullUrl": f"urn:uuid:{res['id']}", "resource": res})

    bundle = {
        "resourceType": "Bundle",
        "type": "collection",
        "timestamp": timestamp,
        "entry": entries,
    }

    Path(output_json).write_text(json.dumps(bundle, indent=2))
    return bundle


def run_unified_dicom_audit(
    dicom_path,
    *,
    config: UnifiedAuditConfig | None = None,
    output_json: str | os.PathLike = _DEFAULT_UNIFIED_AUDIT_JSON,
    patient_identifier_system: str = "urn:dicom:patientid",
) -> dict:
    """Run the unified audit and write a single JSON + optional FHIR bundle."""

    cfg = config if config is not None else UnifiedAuditConfig()

    # Normalize input paths
    if isinstance(dicom_path, (list, tuple)):
        paths = [str(p) for p in dicom_path]
    else:
        paths = [str(dicom_path)]

    # Deterministic order
    paths = list(paths)
    paths.sort()

    # In-run header cache
    cache = DicomHeaderCache(datasets_by_path={})

    # Preload headers to populate cache + capture read errors early.
    for p in paths:
        try:
            dcmread_cached(p, cache, stop_before_pixels=True)
        except Exception as e:
            cache.errors.append({"file": str(p), "error": str(e)})

    # Optional dataset-style metadata summary (from filesystem layout)
    metadata_summary = None
    if cfg.run_metadata_summary:
        try:
            metadata_summary = summarize_paths_layout(paths, base_dir=cfg.base_dir_for_layout, print_summary=True)
        except Exception as e:
            metadata_summary = {"error": str(e)}

    header_combined = None
    if cfg.run_phi_audit:
        header_combined = audit_dicom_header(
            paths,
            output_json=None,
            config=cfg.header_audit_config,
            header_cache=cache,
            write_json=False,
        )

    mask_combined = None
    if cfg.run_mask_finder:
        mask_combined = find_tumor_mask_dicoms(
            paths,
            output_json=None,
            config=cfg.mask_finder_config,
            header_cache=cache,
            write_json=False,
            export_fhir=False,
        )

    # Build per-file slice thickness dicts
    thickness_by_file: dict[str, dict] = {}
    if cfg.run_slice_thickness:
        for p in paths:
            try:
                ds = dcmread_cached(p, cache, stop_before_pixels=True)
            except Exception as e:
                thickness_by_file[str(p)] = {
                    "mm": None,
                    "present": False,
                    "source_tag": "(0018,0050)",
                    "parse_error": str(e),
                }
                continue

            mm = extract_slice_thickness_mm(ds)
            thickness_by_file[str(p)] = {
                "mm": mm,
                "present": bool(mm is not None),
                "source_tag": "(0018,0050)",
                "parse_error": None,
            }

    # Index stage audits by file for merging
    header_by_file: dict[str, dict] = {}
    if header_combined and isinstance(header_combined, dict):
        for a in header_combined.get("audits", []) or []:
            if isinstance(a, dict) and a.get("file"):
                header_by_file[str(a.get("file"))] = a

    mask_by_file: dict[str, dict] = {}
    if mask_combined and isinstance(mask_combined, dict):
        for a in mask_combined.get("audits", []) or []:
            if isinstance(a, dict) and a.get("file"):
                mask_by_file[str(a.get("file"))] = a

    # Build patient-centric organization
    by_patient_id: dict[str, dict] = {}

    # Map patient -> (study, series_folder) -> file list, for principal-series & referenced-series lookups
    series_files_by_patient: dict[str, dict[tuple[str, str], list[str]]] = defaultdict(lambda: defaultdict(list))

    for p in paths:
        patient_id, study_uid, series_folder = lung1_patient_study_series_from_path(p)
        series_files_by_patient[patient_id][(study_uid, series_folder)].append(str(p))

    for p in paths:
        patient_id, study_uid, series_folder = lung1_patient_study_series_from_path(p)
        dicom_meta = {}
        if cfg.run_slice_thickness:
            dicom_meta["slice_thickness"] = thickness_by_file.get(str(p))

        sop_instance_uid = None
        modality = None
        sop_class_uid = None
        ds = None
        try:
            ds = dcmread_cached(p, cache, stop_before_pixels=True)
            sop_raw = None
            try:
                sop_raw = ds.get("SOPInstanceUID")
            except Exception:
                sop_raw = None
            if sop_raw not in (None, "", " "):
                sop_instance_uid = str(sop_raw)

            try:
                modality_raw = ds.get("Modality")
                modality = str(modality_raw).strip().upper() if modality_raw not in (None, "", " ") else None
            except Exception:
                modality = None

            try:
                sop_class = ds.get("SOPClassUID")
                sop_class_uid = str(sop_class).strip() if sop_class not in (None, "", " ") else None
            except Exception:
                sop_class_uid = None
        except Exception:
            sop_instance_uid = None
            ds = None
            modality = None
            sop_class_uid = None

        if ds is not None:
            if modality == "CT" or sop_class_uid == "1.2.840.10008.5.1.4.1.1.2":
                geometry_meta = extract_ct_geometry_meta(ds)
                hu_meta = extract_hu_rescale_meta(ds)
                dicom_meta["geometry"] = geometry_meta
                dicom_meta["hu_rescale"] = hu_meta

        filename = Path(p).name
        series_num: str | None = None
        instance_num: str | None = None

        m = _DICOM_FILENAME_RE.match(filename)
        if not m:
            # Distinguish "missing number" vs "format error" for the common X-Y.dcm layout.
            if filename.lower().endswith(".dcm") and "-" in filename:
                stem = filename[:-4]
                left, right = stem.split("-", 1)
                if left.strip() == "":
                    series_num = "series_num_missing"
                elif left.strip().isdigit():
                    series_num = left.strip()
                else:
                    series_num = "filename_format_error"

                if right.strip() == "":
                    instance_num = "instance_num_missing"
                elif right.strip().isdigit():
                    instance_num = right.strip()
                else:
                    instance_num = "filename_format_error"

                if series_num in ("series_num_missing", "filename_format_error"):
                    print(f"WARNING: series_num parse issue ({series_num}) for filename: {filename}")
                if instance_num in ("instance_num_missing", "filename_format_error"):
                    print(f"WARNING: instance_num parse issue ({instance_num}) for filename: {filename}")
            else:
                series_num = "filename_format_error"
                instance_num = "filename_format_error"
                print(f"WARNING: filename format error for series/instance parse: {filename}")
        else:
            series_num = str(m.group("series"))
            instance_num = str(m.group("instance"))

        ha = header_by_file.get(str(p))
        if isinstance(ha, dict) and "file" in ha:
            ha = {k: v for k, v in ha.items() if k != "file"}

        ma = mask_by_file.get(str(p))
        if isinstance(ma, dict):
            drop_keys = {
                "file",
                "patient_id",
                "study_instance_uid",
                "sop_instance_uid",
                "series_instance_uid",
            }
            if any(k in ma for k in drop_keys):
                ma = {k: v for k, v in ma.items() if k not in drop_keys}

            if ds is not None and (ma.get("mask_type") == "SEG" or modality == "SEG"):
                seg_grid_meta = extract_seg_grid_meta(ds)
                ma["seg_grid_info"] = seg_grid_meta

        rec = {
            "file": str(p),
            "study_instance_uid": study_uid,
            "series_instance_uid": series_folder,
            "sop_instance_uid": sop_instance_uid,
            "series_num": series_num,
            "instance_num": instance_num,
            "dicom": dicom_meta,
            "header_audit": ha,
            "mask_audit": ma,
        }

        patient_payload = by_patient_id.setdefault(
            patient_id,
            {
                "total_files": 0,
                "principal_series": {
                    "has_principal_series": False,
                    "series_instance_uid": None,
                    "num_dicoms": None,
                },
                "files": [],
                "strata": {
                    "mask_files": {"SEG": [], "RTSTRUCT": []},
                    "seg_masks_by_referenced_series_instance_uid": {},
                    "rtstruct_masks_by_referenced_series_instance_uid": {},
                    "principal_series_folders": [],
                    "referenced_series_representative_files": {},
                },
            },
        )

        patient_payload["total_files"] += 1
        patient_payload["files"].append(rec)

    flat_patient_rows: list[dict] = []

    # Build per-patient strata
    for patient_id, patient_payload in by_patient_id.items():
        files = patient_payload.get("files", [])

        # principal series folders (layout-based: series folders with >2 instances)
        principal_series = []
        rep_by_series: dict[str, str] = {}
        for (study_uid, series_folder), file_list in series_files_by_patient.get(patient_id, {}).items():
            if len(file_list) <= 2:
                continue
            file_list_sorted = sorted(file_list)
            rep = file_list_sorted[0]
            principal_series.append(
                {
                    "study_instance_uid": study_uid,
                    "series_folder": series_folder,
                    "dicom_count": int(len(file_list_sorted)),
                    "representative_file": rep,
                }
            )
            rep_by_series[series_folder] = rep

        principal_series.sort(key=lambda d: (d.get("study_instance_uid"), d.get("series_folder")))
        patient_payload["strata"]["principal_series_folders"] = principal_series

        # Patient-level principal series (single representative): pick the largest principal series folder.
        # Output shape requested:
        #   by_patient_id['patient_id'][PatientID]['principal_series'] = {
        #       'has_principal_series': bool,
        #       'series_instance_uid': str|None,
        #       'num_dicoms': int|None,
        #   }
        if principal_series:
            best = max(
                principal_series,
                key=lambda d: (
                    int(d.get("dicom_count") or 0),
                    str(d.get("study_instance_uid") or ""),
                    str(d.get("series_folder") or ""),
                ),
            )
            patient_payload["principal_series"] = {
                "has_principal_series": True,
                "series_instance_uid": best.get("series_folder"),
                "num_dicoms": int(best.get("dicom_count") or 0),
                "representative_file": best.get("representative_file"),
            }
        else:
            patient_payload["principal_series"] = {
                "has_principal_series": False,
                "series_instance_uid": None,
                "num_dicoms": None,
                "representative_file": None,
            }

        # Attach principal series geometry/hu metadata where available
        principal_geometry = None
        principal_hu = None
        principal_slice_thickness = None
        representative_file = patient_payload["principal_series"].get("representative_file")
        if representative_file:
            for rec in files:
                if rec.get("file") == representative_file:
                    dicom_meta_rec = rec.get("dicom", {})
                    principal_geometry = dicom_meta_rec.get("geometry")
                    principal_hu = dicom_meta_rec.get("hu_rescale")
                    slice_meta = dicom_meta_rec.get("slice_thickness") or {}
                    principal_slice_thickness = slice_meta.get("mm")
                    break

        patient_payload["principal_series"]["geometry"] = principal_geometry
        patient_payload["principal_series"]["hu_rescale"] = principal_hu
        patient_payload["principal_series"]["slice_thickness_mm"] = principal_slice_thickness

        # mask strata
        seg_by_ref: dict[str, list[str]] = defaultdict(list)
        rt_by_ref: dict[str, list[str]] = defaultdict(list)

        for rec in files:
            ma = rec.get("mask_audit")
            if not isinstance(ma, dict):
                continue
            mask_type = ma.get("mask_type")
            if mask_type not in ("SEG", "RTSTRUCT"):
                continue
            patient_payload["strata"]["mask_files"][mask_type].append(rec.get("file"))
            ref_uid = ma.get("referenced_series_instance_uid")
            if ref_uid:
                if mask_type == "SEG":
                    seg_by_ref[str(ref_uid)].append(rec.get("file"))
                else:
                    rt_by_ref[str(ref_uid)].append(rec.get("file"))

        # Keep these deterministic
        patient_payload["strata"]["mask_files"]["SEG"].sort()
        patient_payload["strata"]["mask_files"]["RTSTRUCT"].sort()

        patient_payload["strata"]["seg_masks_by_referenced_series_instance_uid"] = {
            k: sorted(v) for k, v in sorted(seg_by_ref.items(), key=lambda t: t[0])
        }
        patient_payload["strata"]["rtstruct_masks_by_referenced_series_instance_uid"] = {
            k: sorted(v) for k, v in sorted(rt_by_ref.items(), key=lambda t: t[0])
        }

        # referenced series representative files (for quick "principal series filenames associated with SEG")
        referenced_rep = {}
        for ref_uid in set(list(seg_by_ref.keys()) + list(rt_by_ref.keys())):
            # In this dataset layout, series folder name is SeriesInstanceUID.
            referenced_rep[ref_uid] = rep_by_series.get(ref_uid)
        patient_payload["strata"]["referenced_series_representative_files"] = referenced_rep

        # Compute SEG grid alignment vs principal CT geometry
        for rec in files:
            ma = rec.get("mask_audit")
            if not isinstance(ma, dict):
                continue
            if ma.get("mask_type") != "SEG":
                continue
            seg_info = ma.get("seg_grid_info")
            if not isinstance(seg_info, dict):
                ma["seg_grid_matches_ct"] = {
                    "match": None,
                    "pixel_spacing": None,
                    "matrix": None,
                    "spacing_between_slices": None,
                    "frame_of_reference": None,
                    "reason": "seg_grid_info_missing",
                }
                continue
            if principal_geometry:
                comparison = _compare_seg_geometry_to_ct(seg_info, principal_geometry, principal_slice_thickness)
                ma["seg_grid_matches_ct"] = comparison
            else:
                ma["seg_grid_matches_ct"] = {
                    "match": None,
                    "pixel_spacing": None,
                    "matrix": None,
                    "spacing_between_slices": None,
                    "frame_of_reference": None,
                    "reason": "principal_ct_geometry_missing",
                }

        # Eligibility signals
        principal_info = patient_payload.get("principal_series") or {}
        linkage_reasons: list[str] = []
        if not principal_info.get("has_principal_series"):
            linkage_reasons.append("principal_series_missing")
        principal_series_uid = principal_info.get("series_instance_uid")
        seg_files_linked: list[str] = []
        if principal_info.get("has_principal_series") and principal_series_uid:
            seg_link_map = patient_payload.get("strata", {}).get("seg_masks_by_referenced_series_instance_uid", {})
            seg_files_linked = list(seg_link_map.get(str(principal_series_uid), []))
            all_seg_files = patient_payload.get("strata", {}).get("mask_files", {}).get("SEG", []) or []
            if not all_seg_files:
                linkage_reasons.append("segmentation_missing")
            elif not seg_files_linked:
                linkage_reasons.append("segmentation_not_linked_to_principal_series")

        ct_seg_and_linkage = len(linkage_reasons) == 0

        preprocessing_eval = _evaluate_preprocessing_readiness(patient_payload)

        patient_payload["eligibility"] = {
            "ct_seg_and_linkage": {
                "value": bool(ct_seg_and_linkage),
                "reasons": linkage_reasons,
                "linked_seg_files": seg_files_linked,
            },
            "ct_ser_valid_for_downstream_preprocessing": preprocessing_eval,
        }

        flat_patient_rows.append(
            {
                "patient_id": patient_id,
                "ct_seg_and_linkage": "TRUE" if ct_seg_and_linkage else "FALSE",
                "ct_seg_and_linkage_reasons_json": json.dumps(linkage_reasons, ensure_ascii=True),
                "ct_seg_linked_files_json": json.dumps(seg_files_linked, ensure_ascii=True),
                "ct_ser_valid_for_downstream_preprocessing": preprocessing_eval["decision"],
                "ct_ser_issues_json": json.dumps(preprocessing_eval["issues"], ensure_ascii=True),
                "ct_ser_warnings_json": json.dumps(preprocessing_eval["warnings"], ensure_ascii=True),
                "ct_ser_segments_json": json.dumps(preprocessing_eval["segments"], ensure_ascii=True),
            }
        )

    # Summary counts
    thickness_values = [v.get("mm") for v in thickness_by_file.values() if isinstance(v, dict)]
    thickness_values = [v for v in thickness_values if v is not None]

    by_patient_id_index = by_patient_id
    by_patient_id = {"patient_id": by_patient_id_index}

    decision_counts: dict[str, int] = {_DECISION_OK: 0, _DECISION_WARN: 0, _DECISION_FAIL: 0}
    for row in flat_patient_rows:
        decision = row.get("ct_ser_valid_for_downstream_preprocessing")
        if decision in decision_counts:
            decision_counts[decision] += 1

    unified = {
        "schema_version": "dicom-unified-audit-4",
        "created_at": _fhir_now(),
        "inputs": {
            "base_dir": str(cfg.base_dir_for_layout),
            "total_files": int(len(paths)),
            "file_first": (paths[0] if (cfg.store_input_files and paths) else None),
            "file_last": (paths[-1] if (cfg.store_input_files and paths) else None),
        },
        "cache_stats": {
            "hits": int(cache.hits),
            "misses": int(cache.misses),
            "size": int(len(cache.datasets_by_path)),
            "errors": cache.errors,
        },
        "metadata_summary": metadata_summary,
        "summary": {
            "slice_thickness_value_count": int(len(thickness_values)),
            "slice_thickness_missing_count": int(len(paths) - len(thickness_values)) if cfg.run_slice_thickness else None,
            "mask_file_count": mask_combined.get("mask_file_count") if isinstance(mask_combined, dict) else None,
            "seg_file_count": mask_combined.get("seg_file_count") if isinstance(mask_combined, dict) else None,
            "rtstruct_file_count": mask_combined.get("rtstruct_file_count") if isinstance(mask_combined, dict) else None,
            "ct_seg_and_linkage_true_count": int(sum(row.get("ct_seg_and_linkage") == "TRUE" for row in flat_patient_rows)),
            "ct_seg_and_linkage_false_count": int(sum(row.get("ct_seg_and_linkage") == "FALSE" for row in flat_patient_rows)),
            "ct_ser_decision_counts": decision_counts,
        },
        "config": {
            "audit_pipeline": {
                "run_metadata_summary": cfg.run_metadata_summary,
                "run_phi_audit": cfg.run_phi_audit,
                "run_slice_thickness": cfg.run_slice_thickness,
                "run_mask_finder": cfg.run_mask_finder,
                "store_input_files": cfg.store_input_files,
                "store_file_paths_in_audits": cfg.store_file_paths_in_audits,
                "base_dir_for_layout": cfg.base_dir_for_layout,
            },
            "phi": {
                "include_all_headers": cfg.header_audit_config.include_all_headers,
                "scan_sequences": cfg.header_audit_config.scan_sequences,
                "flag_private_tags": cfg.header_audit_config.flag_private_tags,
                "redact_values": cfg.header_audit_config.redact_values,
                "print_only_violations": cfg.header_audit_config.print_only_violations,
                "print_tag_details": cfg.header_audit_config.print_tag_details,
                "max_value_preview": cfg.header_audit_config.max_value_preview,
            },
            "masks": {
                "scan_sequences": cfg.mask_finder_config.scan_sequences,
                "deep_tag_search": cfg.mask_finder_config.deep_tag_search,
                "redact_values": cfg.mask_finder_config.redact_values,
                "print_only_matches": cfg.mask_finder_config.print_only_matches,
                "print_tag_details": cfg.mask_finder_config.print_tag_details,
                "print_summary": cfg.mask_finder_config.print_summary,
                "max_value_preview": cfg.mask_finder_config.max_value_preview,
            },
        },
        "by_patient_id": by_patient_id,
    }

    # Optionally redact file paths inside per-file audits to a stable hash.
    if not cfg.store_file_paths_in_audits:
        def _hash_fp(value: str | None):
            if value in (None, "", " "):
                return None
            return {"sha256": _sha256_hex(str(value))}

        patient_container = unified.get("by_patient_id", {})
        if not (isinstance(patient_container, dict) and isinstance(patient_container.get("patient_id"), dict)):
            raise ValueError(
                "Unified audit must have by_patient_id shaped as: { 'patient_id': { '<PatientID>': { ... } } }"
            )

        for patient_payload in patient_container["patient_id"].values():
            if not isinstance(patient_payload, dict):
                continue
            for rec in patient_payload.get("files", []) or []:
                if not isinstance(rec, dict):
                    continue
                try:
                    fp = rec.get("file")
                    rec["file"] = _hash_fp(fp)
                except Exception:
                    continue
                for k in ("header_audit", "mask_audit"):
                    if isinstance(rec.get(k), dict) and "file" in rec[k]:
                        rec[k]["file"] = _hash_fp(str(rec[k]["file"]))

            strata = patient_payload.get("strata")
            if not isinstance(strata, dict):
                continue

            mask_files = strata.get("mask_files")
            if isinstance(mask_files, dict):
                for k in ("SEG", "RTSTRUCT"):
                    if isinstance(mask_files.get(k), list):
                        mask_files[k] = [_hash_fp(x) for x in mask_files[k]]

            for map_key in (
                "seg_masks_by_referenced_series_instance_uid",
                "rtstruct_masks_by_referenced_series_instance_uid",
            ):
                m = strata.get(map_key)
                if isinstance(m, dict):
                    for ref_uid, file_list in list(m.items()):
                        if isinstance(file_list, list):
                            m[ref_uid] = [_hash_fp(x) for x in file_list]

            ps = strata.get("principal_series_folders")
            if isinstance(ps, list):
                for item in ps:
                    if isinstance(item, dict) and "representative_file" in item:
                        item["representative_file"] = _hash_fp(item.get("representative_file"))

            reps = strata.get("referenced_series_representative_files")
            if isinstance(reps, dict):
                for ref_uid, rep_file in list(reps.items()):
                    reps[ref_uid] = _hash_fp(rep_file) if rep_file else None

    output_json_path = Path(output_json)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "patient_id",
        "ct_seg_and_linkage",
        "ct_seg_and_linkage_reasons_json",
        "ct_seg_linked_files_json",
        "ct_ser_valid_for_downstream_preprocessing",
        "ct_ser_issues_json",
        "ct_ser_warnings_json",
        "ct_ser_segments_json",
    ]

    unified["eligibility_table"] = flat_patient_rows

    derived_files = unified.setdefault("derived_files", {})

    flattened_written_path = None
    flatten_error: str | None = None

    try:
        import pandas as pd  # type: ignore

        df_flat = pd.DataFrame(flat_patient_rows, columns=fieldnames)
        flattened_parquet_path = output_json_path.with_name("patient_eligibility_flat.parquet")
        df_flat.to_parquet(flattened_parquet_path, index=False)
        flattened_written_path = str(flattened_parquet_path)
    except Exception as exc:  # pragma: no cover - fallback path
        flatten_error = str(exc)
        fallback_csv_path = output_json_path.with_name("patient_eligibility_flat.csv")
        with open(fallback_csv_path, "w", newline="", encoding="utf-8") as f_csv:
            writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
            writer.writeheader()
            for row in flat_patient_rows:
                writer.writerow(row)
        flattened_written_path = str(fallback_csv_path)

    if flattened_written_path:
        derived_files["patient_eligibility_flat"] = flattened_written_path
        if flatten_error:
            derived_files["patient_eligibility_flat_fallback_reason"] = flatten_error
    if cfg.write_unified_json:
        output_json_path.write_text(json.dumps(unified, indent=2))

    if cfg.write_unified_fhir_bundle:
        fhir_path = output_json_path.with_name(output_json_path.stem + ".fhir.bundle.json")
        export_unified_audit_as_fhir_r4_bundle(unified, fhir_path, patient_identifier_system=patient_identifier_system)
        unified["fhir_r4_bundle_file"] = str(fhir_path)
        if cfg.write_unified_json:
            output_json_path.write_text(json.dumps(unified, indent=2))

    return unified

if __name__ == "__main__":
    dicom_paths = get_dicom(
        PatientID="LUNG1-001",
        StudyInstanceUID_index=1,
        SeriesInstanceUID_index=None,
        SeriesNumber=None,
        InstanceNumber=None,
    )

    print("\nRunning DICOM audit pipeline")
    run_unified_dicom_audit(
        dicom_paths,
        output_json=_DEFAULT_UNIFIED_AUDIT_JSON,
        config=UnifiedAuditConfig(
            run_metadata_summary=True,
            run_phi_audit=True,
            run_slice_thickness=True,
            run_mask_finder=True,
            write_unified_json=True,
            write_unified_fhir_bundle=True,
            store_input_files=True,
            store_file_paths_in_audits=True,
            base_dir_for_layout="../../data/raw/NSCLC-Radiomics",
            header_audit_config=AuditConfig(
                include_all_headers=False,
                scan_sequences=True,
                flag_private_tags=True,
                redact_values=True,
                print_only_violations=True,
                print_tag_details=False,
                max_value_preview=0,
            ),
            mask_finder_config=MaskFinderConfig(
                scan_sequences=True,
                deep_tag_search=False,
                redact_values=True,
                print_only_matches=True,
                print_tag_details=False,
                print_summary=True,
                max_value_preview=0,
            ),
        ),
    )
            


# DICOM 
# PHI should typically not occur in PatientID here, where EHR pull would yield a surrogate ID
# PatientID 
#   (surrogate ID; in clinical settings, backed by a linkage key to the MRN)
# StudyInstanceUID
# SeriesInstanceUIDs
#    (globally unique)
# SOPInstanceUID 
#   (uninformative)