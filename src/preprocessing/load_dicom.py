import glob
import pydicom
import SimpleITK as sitk
import nibabel as nib
import numpy as np

import glob
import random
from typing import List
import argparse

def get_dicom_files(base_dir: str = "data/raw/NSCLC-Radiomics", 
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


# src/preprocessing/load_dicom.py
import glob, random
import pydicom, SimpleITK as sitk, nibabel as nib
import numpy as np

def get_dicom_files(base_dir="data/raw/NSCLC-Radiomics-1", subset_fraction=None, seed=42):
    files = glob.glob(f"{base_dir}/**/**/*.dcm", recursive=True)
    if subset_fraction:
        random.seed(seed)
        k = int(len(files) * subset_fraction)
        files = random.sample(files, k)
    return files

def inspect_metadata(dicom_file):
    ds = pydicom.dcmread(dicom_file)
    print(f"PatientID={ds.PatientID}, SliceThickness={ds.SliceThickness}, PixelSpacing={ds.PixelSpacing}")

def convert_series_to_nifti(series_dir, out_path):
    reader = sitk.ImageSeriesReader()
    series = reader.GetGDCMSeriesIDs(series_dir)[0]
    reader.SetFileNames(reader.GetGDCMSeriesFileNames(series_dir, series))
    img = reader.Execute()
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing([1,1,1])
    new_size = [int(round(sz*spc)) for sz, spc in zip(img.GetSize(), img.GetSpacing())]
    resample.SetSize(new_size)
    sitk.WriteImage(resample.Execute(img), out_path)
    print(f"NIfTI saved to {out_path}")

def explore_and_normalize(nifti_path, out_path):
    nii = nib.load(nifti_path)
    data = nii.get_fdata()
    print(f"Shape={data.shape}, VoxelDims={nii.header.get_zooms()}, AffineDiag={np.diag(nii.affine)[:3]}")
    data = np.clip(data, -1000, 400)
    data = (data - np.mean(data)) / np.std(data)
    nib.save(nib.Nifti1Image(data, nii.affine), out_path)
    print(f"Normalized NIfTI saved to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=float, default=None,
                        help="Fraction of dataset to sample (e.g., 0.05 for 5%)")
    args = parser.parse_args()

    files = get_dicom_files(subset_fraction=args.subset)
    print(f"Loaded {len(files)} files")
    inspect_metadata(files[0])
    nifti_path = convert_series_to_nifti("data/raw/NSCLC-Radiomics-1/", "data/interim/nifti/sample.nii.gz")
    explore_and_normalize(nifti_path, "data/processed/sample_norm.nii.gz")
