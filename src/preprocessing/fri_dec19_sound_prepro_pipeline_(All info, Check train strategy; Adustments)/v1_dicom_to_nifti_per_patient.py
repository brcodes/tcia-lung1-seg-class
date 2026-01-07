"""
Convert per-patient DICOM slices into a single 3D NIfTI volume.

Assumptions
- You have a directory structure like:
    base_path/Patient1/**/**/slice1.dcm
    base_path/Patient1/**/**/slice2.dcm
    ...
    base_path/Patient2/**/**/slice*.dcm
- Each patient directory contains ONE "legitimate" CT series you want to convert.
  (If multiple series exist, this script picks the series with the most slices.)
- Output:
    interim_path/Patient1.nii.gz
    interim_path/Patient2.nii.gz
    ...

Why SimpleITK here:
- Clinically robust DICOM series reading (geometry, spacing, orientation).
- Correct slice ordering + handles common DICOM quirks better than rolling your own pydicom stacking.
"""

import os
import sys
import SimpleITK as sitk


def find_patient_dirs(base_path: str):
    return [
        os.path.join(base_path, d)
        for d in sorted(os.listdir(base_path))
        if os.path.isdir(os.path.join(base_path, d))
    ]


def choose_best_series(series_ids, reader: sitk.ImageSeriesReader, patient_dir: str):
    """
    If multiple series exist under a patient folder, pick the one with the most files.
    This is a pragmatic heuristic; for production you may add modality/body-part checks.
    """
    best_sid = None
    best_files = None

    for sid in series_ids:
        files = reader.GetGDCMSeriesFileNames(patient_dir, sid)
        if not files:
            continue
        if best_files is None or len(files) > len(best_files):
            best_sid = sid
            best_files = files

    return best_sid, best_files


def dicom_series_to_nifti(patient_dir: str, out_path: str) -> None:
    reader = sitk.ImageSeriesReader()
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()

    series_ids = reader.GetGDCMSeriesIDs(patient_dir)
    if not series_ids:
        raise RuntimeError(f"No DICOM series found under: {patient_dir}")

    sid, files = choose_best_series(series_ids, reader, patient_dir)
    if not files:
        raise RuntimeError(f"Found series IDs but no files for patient_dir={patient_dir}")

    reader.SetFileNames(files)
    img = reader.Execute()  # SimpleITK Image, typically int16, with geometry

    # Optional: cast to int16 explicitly (common for CT); you can change to sitk.sitkFloat32 if desired
    if img.GetPixelID() != sitk.sitkInt16:
        img = sitk.Cast(img, sitk.sitkInt16)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Write .nii.gz
    # Note: SimpleITK writes in LPS by default; this is OK as long as you're consistent.
    sitk.WriteImage(img, out_path, useCompression=True)


def main(base_path: str, interim_path: str):
    patient_dirs = find_patient_dirs(base_path)
    if not patient_dirs:
        raise SystemExit(f"No patient subdirectories found under base_path={base_path}")

    failures = []
    for pdir in patient_dirs:
        patient_id = os.path.basename(os.path.normpath(pdir))
        out_path = os.path.join(interim_path, f"{patient_id}.nii.gz")

        try:
            dicom_series_to_nifti(pdir, out_path)
            print(f"[OK] {patient_id} -> {out_path}")
        except Exception as e:
            failures.append((patient_id, str(e)))
            print(f"[FAIL] {patient_id}: {e}", file=sys.stderr)

    if failures:
        print("\nFailures:", file=sys.stderr)
        for patient_id, err in failures:
            print(f" - {patient_id}: {err}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    # Example usage:
    #   python dicom_to_nifti_per_patient.py /data/Lung1_DICOM /data/interim_nifti
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python dicom_to_nifti_per_patient.py <base_path> <interim_path>")

    base_path = sys.argv[1]
    interim_path = sys.argv[2]
    main(base_path, interim_path)