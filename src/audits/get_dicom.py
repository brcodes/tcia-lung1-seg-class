import glob
import os
import re
from pathlib import Path

# Accept filenames like "1-1.dcm" or "1-anything-2.dcm"; first number is series, last is instance.
_DICOM_FILENAME_RE = re.compile(r"(?P<series>\d+)-(?:.*-)?(?P<instance>\d+)\.dcm$")


def get_dicom(
        PatientID=None,
    StudyInstanceUID_index=None,
    SeriesInstanceUID_index=None,
        SeriesNumber=None,
        InstanceNumber=None,
    base_dir="../../data/raw/NSCLC-Radiomics",
):
    """
    Grab DICOM file paths from the Lung1 dataset.

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
        str | list[str]: A single DICOM path if exactly one match is found; otherwise a list of
        all matching paths.
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
    print("\nResult:")

    # Patient selection
    if PatientID is None:
        candidate_patient_dirs = sorted(glob.glob(os.path.join(base_dir, "*")))
        patient_dirs = [p for p in candidate_patient_dirs if os.path.isdir(p)]
        if not patient_dirs:
            raise FileNotFoundError(f"No patient directories found in {base_dir}")
    else:
        if isinstance(PatientID, (list, tuple, set)):
            requested_ids = list(PatientID)
        else:
            requested_ids = [PatientID]

        patient_dirs = []
        for pid in requested_ids:
            patient_dir = os.path.join(base_dir, pid)
            if not os.path.isdir(patient_dir):
                raise FileNotFoundError(f"Patient directory not found: {patient_dir}")
            patient_dirs.append(patient_dir)

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
            f"SeriesInstanceUID_index={SeriesInstanceUID_index}, "
            f"SeriesNumber={SeriesNumber}, InstanceNumber={InstanceNumber} under {base_dir}"
        )

    # Deterministic ordering
    matches.sort()

    # Summarize patient IDs present in the matches.
    base_path = Path(base_dir).resolve()
    patient_ids_matched: set[str] = set()
    study_dirs: set[str] = set()
    series_dirs: set[str] = set()
    for p in matches:
        p_path = Path(p).resolve()
        try:
            rel = p_path.relative_to(base_path)
        except Exception:
            rel = Path(os.path.relpath(str(p_path), str(base_path)))
        parts = rel.parts
        if parts:
            patient_ids_matched.add(parts[0])
        if len(parts) >= 3:
            study_dirs.add(parts[1])
        if len(parts) >= 4:
            series_dirs.add(parts[2])

    # Filenames summary
    print(f"Filenames found: {len(matches)}")
    if matches:
        first_fname = matches[0]
        last_fname = matches[-1]
        total = len(matches)
        print(f"Filename (1/{total}): {first_fname}")
        print(f"Filename ({total}/{total}): {last_fname}")
    else:
        print("Filename (1/0): UNKNOWN")
        print("Filename (0/0): UNKNOWN")

    patient_ids_sorted = sorted(patient_ids_matched)
    print(f"PatientIDs found: {len(patient_ids_sorted)}")
    if patient_ids_sorted:
        total = len(patient_ids_sorted)
        print(f"PatientID (1/{total}): {patient_ids_sorted[0]}")
        print(f"PatientID ({total}/{total}): {patient_ids_sorted[-1]}")
    else:
        print("PatientID (1/0): UNKNOWN")
        print("PatientID (0/0): UNKNOWN")

    # Study folders summary
    study_sorted = sorted(study_dirs)
    print(f"Study folders found: {len(study_sorted)}")
    if study_sorted:
        total = len(study_sorted)
        print(f"Study folder (1/{total}): {study_sorted[0]}")
        print(f"Study folder ({total}/{total}): {study_sorted[-1]}")
    else:
        print("Study folder (1/0): UNKNOWN")
        print("Study folder (0/0): UNKNOWN")

    # Series folders summary
    series_sorted = sorted(series_dirs)
    print(f"Series folders found: {len(series_sorted)}")
    if series_sorted:
        total = len(series_sorted)
        print(f"Series folder (1/{total}): {series_sorted[0]}")
        print(f"Series folder ({total}/{total}): {series_sorted[-1]}")
    else:
        print("Series folder (1/0): UNKNOWN")
        print("Series folder (0/0): UNKNOWN")

    # DICOMs summary (paths list mirrors filenames count)
    print(f"DICOMs found: {len(matches)}")
    if matches:
        first_dcm = os.path.basename(matches[0])
        last_dcm = os.path.basename(matches[-1])
        total = len(matches)
        print(f"DICOM (1/{total}): {first_dcm}")
        print(f"DICOM ({total}/{total}): {last_dcm}")
    else:
        print("DICOM (1/0): UNKNOWN")
        print("DICOM (0/0): UNKNOWN")

    if len(matches) == 1:
        return matches[0]
    return matches


if __name__ == "__main__":
    dicom_paths = get_dicom(
        PatientID=["LUNG1-001","LUNG1-002"],
        StudyInstanceUID_index=1,
        SeriesInstanceUID_index=None,
        SeriesNumber=None,
        InstanceNumber=None,
    )