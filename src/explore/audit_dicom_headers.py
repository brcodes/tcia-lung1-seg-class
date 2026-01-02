import json
from pathlib import Path
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
import os
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
        SeriesInstanceUID_index (int|None): Deprecated alias for StudyInstanceUID_index.
        SeriesNumber (int|None): DICOM SeriesNumber encoded in filename as "<SeriesNumber>-...".
            None means all series numbers.
        InstanceNumber (int|None): DICOM InstanceNumber encoded in filename as "...-<Instance>.dcm".
            None means all instance numbers.
        base_dir (str): Base path to NSCLC-Radiomics dataset.
    
    Returns:
        str | list[str]: One DICOM path when fully specified; otherwise a list of paths.
    """
    # Backward-compat: historically this arg was misnamed.
    if StudyInstanceUID_index is not None and SeriesInstanceUID_index is not None:
        raise ValueError("Provide only one of StudyInstanceUID_index or SeriesInstanceUID_index")
    if StudyInstanceUID_index is None and SeriesInstanceUID_index is not None:
        StudyInstanceUID_index = SeriesInstanceUID_index

    want_list = any(v is None for v in (PatientID, StudyInstanceUID_index, SeriesNumber, InstanceNumber))

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

            for series_dir in series_dirs:
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
            print(f"File: {path}")
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
            if not (cfg.scan_sequences and getattr(elem, "VR", None) == "SQ"):
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

    def _collect_references(ds):
        """Collect reference information to help link SEG/RTSTRUCT -> CT series.

        We conservatively extract:
        - the object's own SeriesInstanceUID (top-level)
        - any nested SeriesInstanceUID values found in sequences (often referenced source series)
        - any ReferencedSOPInstanceUID values found in sequences
        """

        try:
            root_series_uid = ds.get("SeriesInstanceUID", None)
            root_series_uid = str(root_series_uid) if root_series_uid not in (None, "", " ") else None
        except Exception:
            root_series_uid = None

        referenced_series_uids: set[str] = set()
        referenced_sop_uids: set[str] = set()

        for elem in _iter_elements(ds):
            try:
                if elem.tag == T_SERIES_INSTANCE_UID:
                    v = str(elem.value)
                    if v and v != root_series_uid:
                        referenced_series_uids.add(v)
                elif elem.tag == T_REFERENCED_SOP_INSTANCE_UID:
                    v = str(elem.value)
                    if v:
                        referenced_sop_uids.add(v)
            except Exception:
                continue

        return root_series_uid, referenced_series_uids, referenced_sop_uids

    def _audit_one(path: str):
        ds = pydicom.dcmread(path, stop_before_pixels=True)

        patient_id, study_instance_uid_path = _patient_and_study_from_path(path)

        sop = _get_str(ds, T_SOP)
        modality = _get_str(ds, T_MOD)
        series_desc = _get_str(ds, T_SERIES_DESC)

        # Presence-based signals
        has_roi_seq = T_ROI_SEQ in ds
        has_seg_seq = T_SEG_SEQ in ds

        # Strong type determination
        is_rtstruct = (sop == RTSTRUCT_UID) or (modality == "RTSTRUCT")
        is_seg = (sop == SEG_UID) or (modality == "SEG")

        # Structural confidence checks (helps catch mis-encoded modality strings)
        rtstruct_confident = is_rtstruct and has_roi_seq
        seg_confident = is_seg and has_seg_seq

        # Collect labels and try tumor-ish match
        labels = _collect_labels(ds) if (rtstruct_confident or seg_confident or is_rtstruct or is_seg) else []
        tumor_labels = [s for s in labels if tumor_label_re.search(s or "")]

        # Collect reference info (helps tie masks back to the source CT series)
        series_instance_uid, referenced_series_uids, referenced_sop_uids = _collect_references(ds)

        # Track which MASK_TAGS were actually present anywhere in the dataset.
        present_mask_tags: list[dict] = []
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

        is_mask_object = bool(rtstruct_confident or seg_confident)
        mask_type = "RTSTRUCT" if rtstruct_confident else ("SEG" if seg_confident else None)

        mask_found = 1 if is_mask_object else 0
        seg_found = 1 if seg_confident else 0
        rtstruct_found = 1 if rtstruct_confident else 0

        # Print in the same concise style as your other audits.
        if (not cfg.print_only_matches) or is_mask_object:
            print(f"File: {path}")
            print(f"TOTAL MASK TAGS found: {total_mask_tag_count}")

            # Report mask type counts as 0/1 for per-file reporting clarity.
            print(f"RTSTRUCT found: {1 if rtstruct_confident else 0}")
            print(f"SEG found: {1 if seg_confident else 0}")

            if is_mask_object:
                # Requested: print/log patient ID (folder name) immediately above referenced series info.
                print(f"PatientID: {patient_id}")
                print(f"StudyInstanceUID: {study_instance_uid_path}")
                print(f"Referenced SeriesInstanceUIDs: {sorted(referenced_series_uids)}")

            if cfg.print_tag_details and total_mask_tag_count:
                for t in present_mask_tags:
                    print(f"MASK_TAG {t['keyword']} {t['tag']}")

        audit_output = {
            "file": str(path),
            "patient_id": patient_id,
            "study_instance_uid": study_instance_uid_path,
            "sop_class_uid": sop,
            "sop_class_name": _SOP_CLASS_UID_NAMES.get(sop) if sop else None,
            "modality": modality,
            "series_description": series_desc,
            "series_instance_uid": series_instance_uid,
            "referenced_series_instance_uids": sorted(referenced_series_uids),
            "referenced_series_instance_uid_count": len(referenced_series_uids),
            "referenced_sop_instance_uid_count": len(referenced_sop_uids),
            "has_references": bool(referenced_series_uids or referenced_sop_uids),
            "total_mask_tag_count": total_mask_tag_count,
            "mask_tag_names": [t["keyword"] for t in present_mask_tags],
            "mask_tags_present": present_mask_tags,
            "is_rtstruct": rtstruct_confident,
            "is_seg": seg_confident,
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
        for uid in a.get("referenced_series_instance_uids", []) or []:
            masks_by_referenced_series.setdefault(uid, []).append(a.get("file"))

    if cfg.print_summary:
        print(f"TOTAL MASK FILES found: {len(mask_files)}")
        print(f"SEG files found: {len(seg_files)}")
        print(f"RTSTRUCT files found: {len(rtstruct_files)}")
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
            "redact_values": cfg.redact_values,
            "print_only_matches": cfg.print_only_matches,
            "print_tag_details": cfg.print_tag_details,
            "print_summary": cfg.print_summary,
            "max_value_preview": cfg.max_value_preview,
        },
    }

    output_json_path.write_text(json.dumps(combined, indent=2))
    return combined

if __name__ == "__main__":
    
    dicom_paths = get_dicom(PatientID=None,
                            StudyInstanceUID_index=1,
                            SeriesNumber=1,
                            InstanceNumber=1)
    
    audit = False
    find_masks = True
    check_thickness = False
    
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
        print(f"Finding tumor mask DICOM file(s)")
        find_tumor_mask_dicoms(
            dicom_paths,
            output_json="dicom_mask_audit.json",
            config=MaskFinderConfig(
                scan_sequences=True,
                redact_values=True,
                print_only_matches=True,
                print_tag_details=False,
            ),
        )
            
        

    #TO-DO:
    '''
    Check to make sure that
    you didn"t miss principal series bt hard coding this
    '''
    check_thickness = False
    if check_thickness:
        typical_slice_thickness(dicom_paths)

# DICOM 
# PHI should typically not occur in PatientID here, where EHR pull would yield a surrogate ID
# PatientID 
#   (surrogate ID; in clinical settings, backed by a linkage key to the MRN)
# StudyInstanceUID
# SeriesInstanceUIDs
#    (globally unique)
# SOPInstanceUID 
#   (uninformative)