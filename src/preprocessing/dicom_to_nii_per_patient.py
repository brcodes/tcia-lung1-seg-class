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
from pathlib import Path
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


def iter_dirs_within(root: Path, max_depth: int) -> list[Path]:
    """Return directories within root (including root) up to max_depth levels down."""
    if max_depth < 0:
        return []

    out: list[Path] = []
    stack: list[tuple[Path, int]] = [(root, 0)]
    while stack:
        cur, depth = stack.pop()
        if not cur.is_dir():
            continue
        out.append(cur)
        if depth >= max_depth:
            continue
        try:
            children = [p for p in cur.iterdir() if p.is_dir()]
        except PermissionError:
            continue
        for child in sorted(children):
            stack.append((child, depth + 1))
    return out


def choose_best_series_under(study_dir: Path, *, max_depth: int = 3):
    """Choose the DICOM series under study_dir with the most files.

    Notes:
    - SimpleITK/GDCM expects a directory containing DICOM files.
    - Many datasets store files under series subfolders, so we search recursively.
    """
    reader = sitk.ImageSeriesReader()

    best_sid = None
    best_files = None
    best_dir = None

    def has_at_least_n_files(dir_path: Path, n: int) -> bool:
        count = 0
        try:
            for p in dir_path.iterdir():
                if p.is_file():
                    count += 1
                    if count >= n:
                        return True
        except PermissionError:
            return False
        return False

    for dicom_dir in iter_dirs_within(study_dir, max_depth=max_depth):
        # Skip "calibration" / non-series dirs that only contain 1-2 files.
        # This both matches your intent (ignore single-slice scans) and avoids a lot
        # of noisy GDCM warnings from probing directories that can't form a series.
        if not has_at_least_n_files(dicom_dir, 3):
            continue
        series_ids = reader.GetGDCMSeriesIDs(str(dicom_dir))
        if not series_ids:
            continue
        for sid in series_ids:
            files = reader.GetGDCMSeriesFileNames(str(dicom_dir), sid)
            if not files:
                continue
            # Explicitly skip 1-2 file series (often calibration/localizer, etc.)
            if len(files) < 3:
                continue
            if best_files is None or len(files) > len(best_files):
                best_sid = sid
                best_files = files
                best_dir = dicom_dir

    return best_sid, best_files, best_dir


def dicom_files_to_nifti(files, out_path: str) -> None:
    reader = sitk.ImageSeriesReader()
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()

    reader.SetFileNames(list(files))
    img = reader.Execute()  # SimpleITK Image, typically int16, with geometry

    if img.GetPixelID() != sitk.sitkInt16:
        img = sitk.Cast(img, sitk.sitkInt16)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sitk.WriteImage(img, out_path, useCompression=True)


def dicom_series_to_nifti(patient_dir: str | Path, out_path: str) -> None:
    reader = sitk.ImageSeriesReader()
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()

    patient_dir = str(patient_dir)

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


def dicoms_to_nifti(base_path: str, interim_path: str):
    patient_dirs = find_patient_dirs(base_path)
    if not patient_dirs:
        raise SystemExit(f"No patient subdirectories found under base_path={base_path}")

    failures = []
    for pdir in patient_dirs:
        # Patient folder has exactly one studyID subfolder
        pdir = Path(pdir)
        try:
            study_dir = next(d for d in pdir.iterdir() if d.is_dir())
        except StopIteration:
            failures.append((pdir.name, f"No studyID directory found under: {pdir}"))
            continue

        patient_id = pdir.name
        out_path = os.path.join(interim_path, f"{patient_id}TEST.nii.gz")

        try:
            sid, files, dicom_dir = choose_best_series_under(study_dir, max_depth=3)
            if not files:
                raise RuntimeError(
                    "No DICOM series found under study directory (searched recursively). "
                    f"study_dir={study_dir}"
                )

            dicom_files_to_nifti(files, out_path)
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
    # Local usage (edit these paths for your machine)
    # base_path: folder containing per-patient DICOM directories
    # interim_path: output folder for per-patient .nii.gz volumes
    base_path = "../../data/raw/NSCLC-Radiomics/"  # e.g. "data/raw/NSCLC-Radiomics"
    interim_path = "../../data/interim/nifti"
    dicoms_to_nifti(base_path, interim_path)
    # preprocessing()