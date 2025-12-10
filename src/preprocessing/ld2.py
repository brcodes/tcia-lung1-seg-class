import os

def choose_best_series(study_path):
    """
    Given a study folder containing multiple series,
    return the path to the series with the most DICOM files.
    """
    best_series = None
    aux_series_length = 2

    for series in os.listdir(study_path):
        series_path = os.path.join(study_path, series)
        if not os.path.isdir(series_path):
            print(f"No series path == {series_path}.")

        # Count DICOM files
        dcm_files = [f for f in os.listdir(series_path) if f.lower().endswith(".dcm")]
        num_dcms = len(dcm_files)

        if num_dcms > aux_series_length:
            best_series = series_path

    return best_series

import os
import SimpleITK as sitk

raw_root = "../../data/raw/NSCLC-Radiomics"
output_root = "nnUNet_raw_data/DatasetXXX_NSCLC/imagesTr"

os.makedirs(output_root, exist_ok=True)

patient_id = 0
for patient in sorted(os.listdir(raw_root)):
    patient_path = os.path.join(raw_root, patient)
    for study in os.listdir(patient_path):
        study_path = os.path.join(patient_path, study)
        for series in os.listdir(study_path):
            series_path = os.path.join(study_path, series)
            dcm_files = [os.path.join(series_path, f) for f in os.listdir(series_path) if f.endswith(".dcm")]
            if not dcm_files:
                print(f"No DICOM files in {series_path}.")

            # Read DICOM series
            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(series_path)
            reader.SetFileNames(dicom_names)
            image = reader.Execute()

            # Save as NIfTI
            out_name = f"NSCLC_{patient_id:04d}.nii.gz"
            sitk.WriteImage(image, os.path.join(output_root, out_name))
            print(f"Converted {series_path} -> {out_name}")
            patient_id += 1