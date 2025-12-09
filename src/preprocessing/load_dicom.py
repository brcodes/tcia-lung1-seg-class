import glob
import pydicom
import SimpleITK as sitk
import nibabel as nib
import numpy as np

import glob
import random
from typing import List

def get_dicom_files(base_dir: str = "data/raw/NSCLC-Radiomics-1", 
                    subset_fraction: float = None, 
                    seed: int = 42) -> List[str]:
    """
    Load DICOM file paths from dataset.
    
    Parameters
    ----------
    base_dir : str
        Root directory containing DICOM files.
    subset_fraction : float, optional
        Fraction of dataset to sample (e.g., 0.05 for 5%).
        If None, returns full dataset.
    seed : int
        Random seed for reproducibility.
    
    Returns
    -------
    List[str]
        List of DICOM file paths.
    """
    # Collect all DICOM files
    dicom_files = glob.glob(f"{base_dir}/**/**/*.dcm", recursive=True)
    
    if subset_fraction is not None:
        random.seed(seed)
        k = int(len(dicom_files) * subset_fraction)
        dicom_files = random.sample(dicom_files, k)
    
    return dicom_files


if __name__ == "__main__":
    # Example usage
    all_files = get_dicom_files()
    print(f"Loaded {len(all_files)} DICOM files (full dataset).")

    subset_files = get_dicom_files(subset_fraction=0.05)
    print(f"Loaded {len(subset_files)} DICOM files (5% subset).")


# Step 1: Metadata inspection
dicom_files = glob.glob("data/raw/NSCLC-Radiomics-1/**/**/*.dcm", recursive=True)
sample = pydicom.dcmread(dicom_files[0])
print("Patient ID:", sample.PatientID)
print("Slice Thickness:", sample.SliceThickness)
print("Pixel Spacing:", sample.PixelSpacing)

# Step 2: Convert DICOM series to NIfTI with resampling
reader = sitk.ImageSeriesReader()
series_IDs = reader.GetGDCMSeriesIDs("data/raw/NSCLC-Radiomics-1/")
series = series_IDs[0]  # take first series for demo
dicom_names = reader.GetGDCMSeriesFileNames("data/raw/NSCLC-Radiomics-1/", series)
reader.SetFileNames(dicom_names)
image = reader.Execute()

# Resample to 1mm isotropic spacing
resample = sitk.ResampleImageFilter()
resample.SetOutputSpacing([1.0, 1.0, 1.0])
new_size = [int(round(sz * spc)) for sz, spc in zip(image.GetSize(), image.GetSpacing())]
resample.SetSize(new_size)
resampled = resample.Execute(image)
sitk.WriteImage(resampled, "data/interim/nifti/sample.nii.gz")

# Step 3: Explore and normalize with nibabel
nii = nib.load("data/interim/nifti/sample.nii.gz")
data = nii.get_fdata()
print("Shape:", data.shape)
print("Voxel dimensions:", nii.header.get_zooms())

# Normalize HU range (lung window)
data = np.clip(data, -1000, 400)
data = (data - np.mean(data)) / np.std(data)
nib.save(nib.Nifti1Image(data, nii.affine), "data/processed/sample_norm.nii.gz")
print("Affine diagonal (voxel sizes):", np.diag(nii.affine)[:3])
