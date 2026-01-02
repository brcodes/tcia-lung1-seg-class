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

def get_dicom(
        PatientID=None,
        SeriesInstanceUID_index=None,
        SeriesNumber=None,
        InstanceNumber=None,
        base_dir="../../data/raw/NSCLC-Radiomics",
):
    """
        Grab DICOM file(s) from the Lung1 dataset.

        Behavior:
        - If *all* of (PatientID, SeriesInstanceUID_index, SeriesNumber, InstanceNumber) are not None,
            returns a single DICOM path (str), matching the prior behavior.
        - If *any* of those arguments are None, that argument becomes a wildcard and the function
            returns a list[str] of all matching DICOM paths.
    
    Args:
        PatientID (str|None): Patient folder name (e.g. "LUNG1-001"). None means all patients.
        SeriesInstanceUID_index (int|None): Series order (1-based index, lexicographically sorted
            series directory list within each patient). None means all series per patient.
        SeriesNumber (int|None): DICOM SeriesNumber encoded in filename as "<SeriesNumber>-...".
            None means all series numbers.
        InstanceNumber (int|None): DICOM InstanceNumber encoded in filename as "...-<Instance>.dcm".
            None means all instance numbers.
        base_dir (str): Base path to NSCLC-Radiomics dataset.
    
    Returns:
        str | list[str]: One DICOM path when fully specified; otherwise a list of paths.
    """
    want_list = any(
        v is None
        for v in (PatientID, SeriesInstanceUID_index, SeriesNumber, InstanceNumber)
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
        # Build glob for all series under the patient/studyUID
        series_dirs = glob.glob(os.path.join(patient_dir, "*", "*"))
        series_dirs = [s for s in series_dirs if os.path.isdir(s)]
        if not series_dirs:
            # If PatientID is explicit, preserve the old behavior of erroring.
            if PatientID is not None:
                raise FileNotFoundError(f"No series found for patient ds ID {PatientID}")
            continue

        # Sort series directories lexicographically (UIDs are strings, not ints)
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
                    f"(found {len(series_dirs)} series) for patient {os.path.basename(patient_dir)}."
                )

        for chosen_series in chosen_series_dirs:
            dcm_paths = glob.glob(os.path.join(chosen_series, "*.dcm"))
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
            f"PatientID={PatientID}, SeriesInstanceUID_index={SeriesInstanceUID_index}, "
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
    print(f"SeriesInstanceUID index: {SeriesInstanceUID_index}")
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

if __name__ == "__main__":
    
    audit = True
    if audit:
        # Audit a specific DICOM
        dicom_path = get_dicom(PatientID="LUNG1-001",
                                SeriesInstanceUID_index=1,
                                SeriesNumber=1,
                                InstanceNumber=1)
        print(f"Auditing DICOM file(s)")
        print(f"Auditing DICOM file at: {dicom_path}")
        audit_dicom_header(dicom_path, config=AuditConfig(
            include_all_headers=False,
            redact_values=True,
            print_only_violations=True,
            print_tag_details=True))
            
        

    #TO-DO:
    '''
    Check to make sure that
    you didn"t miss principal series bt hard coding this
    '''
    check_thickness = False
    if check_thickness:
        # Check slice thickness
        dicom_paths = get_dicom(SeriesInstanceUID_index=2)
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