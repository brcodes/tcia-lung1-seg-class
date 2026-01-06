import os
import sys

def series_length_val(root_dir):
    """
    Iterate through NSCLC-Radiomics filetree and validate series counts.
    Prints True if all patients have only 1 series with >2 DICOMs.
    Prints False otherwise, and lists offending patients + series info.
    """

    all_good = True
    offenders = {}

    for patient in sorted(os.listdir(root_dir)):
        patient_path = os.path.join(root_dir, patient)
        if not os.path.isdir(patient_path):
            continue

        for study in os.listdir(patient_path):
            study_path = os.path.join(patient_path, study)
            if not os.path.isdir(study_path):
                continue

            series_counts = []
            for idx, series in enumerate(sorted(os.listdir(study_path)), start=1):
                series_path = os.path.join(study_path, series)
                if not os.path.isdir(series_path):
                    continue

                dcm_files = [f for f in os.listdir(series_path) if f.lower().endswith(".dcm")]
                num_dcms = len(dcm_files)
                if num_dcms > 2:
                    series_counts.append((idx, num_dcms))

            if len(series_counts) > 1:
                all_good = False
                offenders[patient] = series_counts

    if all_good:
        print("True")
        print("No Patients with >1 series containing >2 DICOMs")
    else:
        print("False")
        print("Patients with >1 series containing >2 DICOMs:")
        for patient, series_info in offenders.items():
            print(f"  {patient}:")
            for series_idx, num_dcms in series_info:
                print(f"    Series {series_idx} -> {num_dcms} DICOMs")


if __name__ == "__main__":

    root_dir = '../../data/raw/NSCLC-Radiomics/'
    series_length_val(root_dir)