import os
import numpy as np
import nibabel as nib
import pydicom

def test_metadata_fields():
    ds = pydicom.dcmread("data/raw/NSCLC-Radiomics-/LUNG-.dcm")
    assert hasattr(ds, "SliceThickness")
    assert hasattr(ds, "PixelSpacing")
    assert float(ds.SliceThickness) > 0
    assert all(float(x) > 0 for x in ds.PixelSpacing)

def test_nifti_conversion():
    assert os.path.exists("data/interim/nifti/sample.nii.gz")
    nii = nib.load("data/interim/nifti/sample.nii.gz")
    zooms = nii.header.get_zooms()
    assert all(abs(z - 1.0) < 1e-3 for z in zooms[:3])  # isotropic spacing

def test_preprocessing_clipping():
    nii = nib.load("data/preprocessed/sample_lungwindow.nii.gz")
    data = nii.get_fdata()
    assert np.min(data) >= -1000
    assert np.max(data) <= 400
    mean, std = np.mean(data), np.std(data)
    assert abs(mean) < 0.05
    assert 0.95 < std < 1.05
