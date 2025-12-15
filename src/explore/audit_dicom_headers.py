import pydicom
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

def get_dicom(patient_name, series_index=1, dicom_filename="1-1.dcm",
              base_dir="../../data/raw/NSCLC-Radiomics"):
    """
    Grab a specific DICOM file given patient name, series order, and filename.
    
    Args:
        patient_name (str): Patient folder name (e.g. "LUNG-001").
        series_index (int): Series order (1-based index, sorted numerically).
        dicom_filename (str): DICOM filename to grab (e.g. "1-1.dcm").
        base_dir (str): Base path to NSCLC-Radiomics dataset.
    
    Returns:
        str: Path to the requested DICOM file.
    """
    # Build glob for all series under the patient/studyUID
    pattern = os.path.join(base_dir, patient_name, "*", "*")
    series_dirs = glob.glob(pattern)
    if not series_dirs:
        raise FileNotFoundError(f"No series found for patient ds ID {patient_name}")
    
    # Sort series directories lexicographically (UIDs are strings, not ints)
    # Observe that they increase in numeric order due to UID structure, each representing a timepoint
    series_dirs.sort()
    
    # Select the requested series (1-based index)
    try:
        chosen_series = series_dirs[series_index - 1]
    except IndexError:
        raise IndexError(f"Series index {series_index} out of range (found {len(series_dirs)} series).")
    
    # Build full path to requested DICOM
    dicom_path = os.path.join(chosen_series, dicom_filename)
    if not os.path.exists(dicom_path):
        raise FileNotFoundError(f"DICOM file {dicom_filename} not found in {chosen_series}")
    
    print(f"Chose {patient_name} series {series_index}, {dicom_filename}")
    print(f"path: {dicom_path}")
    
    return dicom_path

def audit_dicom_headers(dicom_path, output_json="dicom_header_audit.json"):
    ds = pydicom.dcmread(dicom_path, stop_before_pixels=True)
    
    header_dict = {}
    phi_flags = []

    for elem in ds:
        print(elem)
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

# Example usage:

dicom_path = get_dicom("LUNG1-001", series_index=1, dicom_filename="1-1.dcm")
print(f"Auditing DICOM file at: {dicom_path}")
# audit_dicom_headers(dicom_path)

# DICOM SeriesInstanceUIDs