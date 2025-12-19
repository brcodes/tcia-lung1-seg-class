"""
Preprocessing pipeline for CT NIfTI volumes + tumor masks (stub) for:
- segmentation (tumor)
- classification (stage, external labels not handled here)

Input:
    samples: List[Tuple[str, Union[str, sitk.Image], Union[str, sitk.Image]]]
        where each tuple is (patient_id, ct_nifti, tumor_mask_nifti)

Outputs (per patient, written to disk):
    {out_dir}/{patient_id}/
        ct_hu_resampled_ras.nii.gz          # optional audit artifact
        ct_roi_hu.nii.gz                    # optional audit artifact (cropped HU)
        ct_roi_norm.nii.gz                  # model input
        tumor_roi.nii.gz                    # segmentation target
        qc_overlay_mid.png                  # optional QC image (non-identifying)
        meta.json                           # metadata sidecar (written as JSON here)
        transforms.json                     # transform record (written as JSON here)

Notes / decisions implemented based on your request:
- Orientation