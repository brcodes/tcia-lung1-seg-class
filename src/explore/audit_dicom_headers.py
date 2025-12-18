import json
from pathlib import Path

# Define a set of DICOM tags commonly considered PHI under HIPAA
PHI_TAGS = {
    (0x0010, 0x0010): "PatientName",
    (0x0010, 0x0020): "PatientID",
    (0x0010, 0x0030): "PatientBirthDate",
    (0x0010, 0x0040): "PatientSex",
    (0x0008, 0x0020): "StudyDate",
    (0x0008, 0x0030): "StudyTime",
    (0x0008, 0x0090): "ReferringPhysicianName",
    (0x0008, 0x1010): "StationName",
    (0x0008, 0x0080): "InstitutionName",
    (0x0008, 0x0050): "AccessionNumber",
}

import glob
import os
import re


_DICOM_FILENAME_RE = re.compile(r"^(?P<series>\d+)-(?P<instance>\d+)\.dcm$")

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
            if raw is None:
                skipped += 1
                # Tag search
                # if thicknesses_tag in ds:
                #     raw = ds[thicknesses_tag].value
                #     if raw is None:
                #         skipped += 1
                        
                        # Save this in a find substring print context( ) util
                        # if double_check:
                        #     needles = ("thickness", "slice")
                        #     context_chars = 100
                        #     for elem in ds:
                        #         keyword = elem.keyword if elem.keyword else str(elem.tag)
                        #         all_key_values = f"{keyword}: {elem.value}"
                        #         haystack = all_key_values
                        #         haystack_lower = haystack.lower()

                        #         matched = next((n for n in needles if n in haystack_lower), None)
                        #         if matched is None:
                        #             continue

                        #         idx = haystack_lower.find(matched)
                        #         start = max(0, idx - context_chars)
                        #         end = min(len(haystack), idx + len(matched) + context_chars)
                        #         snippet = haystack[start:end]
                        #         prefix = "..." if start > 0 else ""
                        #         suffix = "..." if end < len(haystack) else ""

                                
                        #         print(
                        #             "Found possible SliceThickness-related info in DICOM (SliceThickness missing):\n"
                        #             f"file: {dicom_path}\n"
                        #             f"match: '{matched}'\n"
                        #             f"context: {prefix}{snippet}{suffix}"
                        #         )

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

def audit_dicom_header(dicom_path, output_json="dicom_header_audit.json"):
    import pydicom
    ds = pydicom.dcmread(dicom_path, stop_before_pixels=True)
    
    header_dict = {}
    phi_flags = []

    for elem in ds:
        # print(elem)
        tag = (elem.tag.group, elem.tag.element)
        keyword = elem.keyword if elem.keyword else str(elem.tag)
        value = str(elem.value)
        header_dict[keyword] = value

        if tag in PHI_TAGS:
            phi_flags.append({
                "tag": f"({elem.tag.group:04X},{elem.tag.element:04X})",
                "keyword": keyword,
                "value": value
            })

    # Print summary to terminal
    print("\n=== DICOM HEADER AUDIT ===")
    print(f"File: {dicom_path}")
    print(f"Total attributes: {len(ds)}")
    print(f"Potential PHI fields found: {len(phi_flags)}")
    for f in phi_flags:
        print(f"⚠️ {f['keyword']} = {f['value']}")

    # Export to JSON for audit trail
    audit_output = {
        "file": str(dicom_path),
        "total_attributes": len(ds),
        "potential_phi": phi_flags,
        "all_headers": header_dict
    }
    Path(output_json).write_text(json.dumps(audit_output, indent=2))
    print(f"\nAudit JSON exported to {output_json}")

if __name__ == "__main__":
    
    audit = False
    if audit:
        # Audit a specific DICOM
        dicom_path = get_dicom(PatientID="LUNG1-001",
                                SeriesInstanceUID_index=3,
                                InstanceNumber=2)
        print(f"Auditing DICOM file at: {dicom_path}")
        audit_dicom_header(dicom_path)
    
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