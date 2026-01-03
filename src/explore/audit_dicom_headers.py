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

def audit_dicom_header(dicom_path, output_json="dicom_header_audit.json", *, config=None):
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
        Path(output_json).write_text(json.dumps(combined, indent=2))
        if not cfg.print_only_violations:
            print(f"\nAudit JSON exported to {output_json}")
        return combined

    single = _audit_one(str(dicom_path))
    Path(output_json).write_text(json.dumps(single, indent=2))
    if not cfg.print_only_violations:
        print(f"\nAudit JSON exported to {output_json}")
    return single


def find_tumor_mask_dicoms(dicom_path, output_json="dicom_mask_audit.json", *, config: MaskFinderConfig | None = None):
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

    output_json_path = Path(output_json)

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
        "mask_files": mask_files,
        "mask_file_count": len(mask_files),
        "seg_files": seg_files,
        "seg_file_count": len(seg_files),
        "rtstruct_files": rtstruct_files,
        "rtstruct_file_count": len(rtstruct_files),
        "masks_by_referenced_series_instance_uid": masks_by_referenced_series,
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

    output_json_path.write_text(json.dumps(combined, indent=2))

    # Emit FHIR R4 Bundle (collection) alongside the native JSON.
    fhir_path = output_json_path.with_name(output_json_path.stem + ".fhir.bundle.json")
    export_mask_audit_as_fhir_r4_bundle(combined, fhir_path, patient_identifier_system="urn:dicom:patientid")
    combined["fhir_r4_bundle_file"] = str(fhir_path)
    output_json_path.write_text(json.dumps(combined, indent=2))
    return combined

if __name__ == "__main__":
    
    # dicom_paths = get_dicom(PatientID='LUNG1-001',
    #                         StudyInstanceUID_index=1,
    #                         SeriesInstanceUID_index=None,
    #                         SeriesNumber=None,
    #                         InstanceNumber=None)
    
    dicom_paths = get_dicom(PatientID=None,
                            StudyInstanceUID_index=None,
                            SeriesInstanceUID_index=None,
                            SeriesNumber=None,
                            InstanceNumber=None)
    
    audit = False
    find_masks = False
    
    # # Check slice thickness
    # dicom_paths = get_dicom(StudyInstanceUID_index=2)
    
    if audit:
        # Audit a specific DICOM
        print(f"Auditing DICOM file(s)")
        audit_dicom_header(dicom_paths, config=AuditConfig(
            include_all_headers=False,
            redact_values=True,
            print_only_violations=True,
            print_tag_details=False))

    if find_masks:
        # Find tumor mask objects (SEG / RTSTRUCT) within a patient (or broaden by setting PatientID=None).
        print(f"\nFinding tumor mask DICOM file(s)")
        find_tumor_mask_dicoms(
            dicom_paths,
            output_json="dicom_mask_audit.json",
            config=MaskFinderConfig(
                scan_sequences=True,
                deep_tag_search=False,
                redact_values=True,
                print_only_matches=True,
                print_tag_details=False,
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