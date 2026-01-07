"""Validation helpers leveraging the DICOM audit before preprocessing."""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List

_DECISION_OK = "OK"
_DECISION_WARN = "WARN_minor"
_DECISION_FAIL = "FAIL_exclude"


def _entry_value(entry: Dict[str, Any] | None, key: str) -> Any:
    if not isinstance(entry, dict):
        return None
    value = entry.get(key)
    if isinstance(value, dict):
        return value.get("value")
    return value


def _entry_present(entry: Dict[str, Any] | None, key: str) -> bool:
    if not isinstance(entry, dict):
        return False
    value = entry.get(key)
    if isinstance(value, dict):
        if value.get("present") is False:
            return False
        return value.get("value") is not None
    return value is not None


def load_audit_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_patient(patient_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    issues: List[str] = []
    warnings: List[str] = []

    principal = payload.get("principal_series") or {}
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

    seg_records: List[Dict[str, Any]] = []
    for rec in payload.get("files", []):
        if not isinstance(rec, dict):
            continue
        ma = rec.get("mask_audit")
        if not isinstance(ma, dict):
            continue
        if ma.get("mask_type") == "SEG":
            seg_records.append(rec)

    if not seg_records:
        issues.append("segmentation_missing")

    segments_summary: List[Dict[str, Any]] = []
    for seg_rec in seg_records:
        ma = seg_rec.get("mask_audit") or {}
        seg_info = ma.get("seg_grid_info")
        match_info = ma.get("seg_grid_matches_ct")
        seg_file = seg_rec.get("file")

        if not isinstance(seg_info, dict):
            issues.append(f"seg_grid_info_missing:{seg_file}")
            segments_summary.append({"file": seg_file, "match": None})
            continue

        if not isinstance(match_info, dict):
            warnings.append(f"seg_alignment_unknown:{seg_file}")
            segments_summary.append({"file": seg_file, "match": None})
            continue

        match_state = match_info.get("match")
        if match_state is False:
            issues.append(f"seg_alignment_failed:{seg_file}")
        elif match_state is None:
            warnings.append(f"seg_alignment_indeterminate:{seg_file}")
        segments_summary.append({"file": seg_file, "match": match_state})

    decision = _DECISION_OK
    if issues:
        decision = _DECISION_FAIL
    elif warnings:
        decision = _DECISION_WARN

    return {
        "decision": decision,
        "issues": issues,
        "warnings": warnings,
        "principal_series": {
            "series_instance_uid": principal.get("series_instance_uid"),
            "representative_file": principal.get("representative_file"),
        },
        "segments": segments_summary,
    }


def build_validated_manifest(audit_data: Dict[str, Any]) -> Dict[str, Any]:
    patients_block = audit_data.get("by_patient_id") or {}
    if isinstance(patients_block, dict) and "patient_id" in patients_block:
        patients_block = patients_block["patient_id"]

    results: Dict[str, Any] = {}
    for patient_id, payload in sorted(patients_block.items()):
        results[patient_id] = evaluate_patient(patient_id, payload)

    return {
        "schema_version": "preprocessing-validation-1",
        "created_at": audit_data.get("created_at"),
        "source_audit_schema": audit_data.get("schema_version"),
        "patients": results,
    }


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate DICOM audit before preprocessing.")
    parser.add_argument("--audit-json", required=True, help="Path to dicom_audit.json produced by audit script")
    parser.add_argument("--output-json", required=True, help="Where to write the validated manifest JSON")
    parser.add_argument(
        "--fail-on-warn",
        action="store_true",
        help="Exit with non-zero status if any patient ends in WARN_minor or FAIL_exclude",
    )
    args = parser.parse_args(argv)

    audit_data = load_audit_json(args.audit_json)
    manifest = build_validated_manifest(audit_data)
    manifest["source_audit_path"] = args.audit_json

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    if args.fail_on_warn:
        if any(result.get("decision") != _DECISION_OK for result in manifest["patients"].values()):
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
