"""
Preprocess CT NIfTI volumes + aligned tumor masks for segmentation + downstream classification.

Input:
    samples: List[Tuple[str, ct, tumor_mask]]
        ct and tumor_mask can be file paths (str) or SimpleITK Images.
        Tumor masks are assumed to be spatially aligned with the CT NIfTI.

Pipeline (with optional extras enabled by default):
  Pass 1 (stats-only):
    - Load CT + mask
    - Reorient both to RAS (consistent for training)
    - Resample to target spacing (default 1x1x3mm; linear for CT, NN for mask)
    - Compute tumor bbox and padded bbox sizes -> derive dataset-informed MIN_BOX_MM / MAX_BOX_MM
    - Compute per-case intensity stats after windowing on non-air voxels -> aggregate GLOBAL_MU / GLOBAL_SIGMA

  Pass 2 (write artifacts):
    - Same standardization + resampling as pass 1
    - Crop around tumor bbox + padding, enforce min/max crop size
    - Window HU then GLOBAL z-score normalize
    - Write:
        ct_roi_norm.nii.gz   (model input)
        tumor_roi.nii.gz     (target)
      Optional audit artifacts:
        ct_hu_resampled_ras.nii.gz
        ct_roi_hu.nii.gz
      Optional QC:
        qc_overlay_mid.png
      Metadata:
        meta.json, transforms.json, dataset_pass1_stats.json, failures.json

Notes:
- This code assumes CT values are already in HU (as is typical if you converted from DICOM with slope/intercept).
- Orientation: outputs are RAS-oriented NIfTIs, which is a common choice in ML stacks and works well with MONAI.

Dependencies:
  pip install SimpleITK numpy
"""

import os
import json
import math
from dataclasses import dataclass, asdict
from typing import List, Tuple, Union, Dict, Any, Optional

import numpy as np
import SimpleITK as sitk


PathOrImage = Union[str, sitk.Image]
Sample = Tuple[str, PathOrImage, PathOrImage]


# ----------------------------
# IO / orientation
# ----------------------------

def _read_image(x: PathOrImage) -> sitk.Image:
    if isinstance(x, sitk.Image):
        return x
    return sitk.ReadImage(x)


def _write_image(img: sitk.Image, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sitk.WriteImage(img, path, useCompression=True)


def _to_float32(img: sitk.Image) -> sitk.Image:
    return sitk.Cast(img, sitk.sitkFloat32)


def _to_uint8(img: sitk.Image) -> sitk.Image:
    return sitk.Cast(img, sitk.sitkUInt8)


def _orient_to_ras(img: sitk.Image) -> sitk.Image:
    f = sitk.DICOMOrientImageFilter()
    f.SetDesiredCoordinateOrientation("RAS")
    return f.Execute(img)


# ----------------------------
# Resampling
# ----------------------------

def _resample_to_spacing(
    img: sitk.Image,
    target_spacing: Tuple[float, float, float],
    interpolator: int,
    default_value: float,
) -> sitk.Image:
    original_spacing = img.GetSpacing()
    original_size = img.GetSize()

    # new_size[i] = round(old_size[i] * old_spacing[i] / new_spacing[i])
    new_size = [
        int(math.floor(original_size[i] * (original_spacing[i] / target_spacing[i]) + 0.5))
        for i in range(3)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(default_value)
    return resampler.Execute(img)


# ----------------------------
# BBox / cropping
# ----------------------------

def _mask_bbox_ijk(mask_u8: sitk.Image) -> Optional[Tuple[List[int], List[int]]]:
    """
    Returns bbox as (start_index_ijk, size_ijk). If empty, returns None.
    Assumes foreground label is >0; internally treats any >0 as label 1.
    """
    # Ensure binary-ish: >0 -> 1
    bin_mask = sitk.BinaryThreshold(mask_u8, lowerThreshold=1, upperThreshold=255, insideValue=1, outsideValue=0)

    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(bin_mask)
    if not stats.HasLabel(1):
        return None
    bb = stats.GetBoundingBox(1)  # (x, y, z, sizeX, sizeY, sizeZ)
    start = [int(bb[0]), int(bb[1]), int(bb[2])]
    size = [int(bb[3]), int(bb[4]), int(bb[5])]
    return start, size


def _pad_bbox_mm(
    start: List[int],
    size: List[int],
    spacing: Tuple[float, float, float],
    pad_mm: Tuple[float, float, float],
) -> Tuple[List[int], List[int], List[int]]:
    pad_vox = [int(math.ceil(pad_mm[i] / spacing[i])) for i in range(3)]
    new_start = [start[i] - pad_vox[i] for i in range(3)]
    new_size = [size[i] + 2 * pad_vox[i] for i in range(3)]
    return new_start, new_size, pad_vox


def _enforce_min_max_size_mm(
    start: List[int],
    size: List[int],
    spacing: Tuple[float, float, float],
    min_mm: Tuple[float, float, float],
    max_mm: Tuple[float, float, float],
) -> Tuple[List[int], List[int], Dict[str, Any]]:
    """
    Expand/shrink bbox around center so size_mm lies within [min_mm, max_mm].
    """
    info = {"expanded_to_min": [False, False, False], "clamped_to_max": [False, False, False]}

    center = [start[i] + size[i] / 2.0 for i in range(3)]
    min_vox = [int(math.ceil(min_mm[i] / spacing[i])) for i in range(3)]
    max_vox = [int(math.floor(max_mm[i] / spacing[i])) for i in range(3)]

    new_size = size[:]
    for i in range(3):
        if new_size[i] < min_vox[i]:
            new_size[i] = min_vox[i]
            info["expanded_to_min"][i] = True
        if new_size[i] > max_vox[i]:
            new_size[i] = max_vox[i]
            info["clamped_to_max"][i] = True

    new_start = [int(round(center[i] - new_size[i] / 2.0)) for i in range(3)]
    return new_start, new_size, info


def _crop_with_pad(
    img: sitk.Image,
    start: List[int],
    size: List[int],
    fill_value: float,
) -> Tuple[sitk.Image, Dict[str, Any]]:
    """
    Crop a region even if it extends beyond the image bounds by padding first.
    """
    img_size = list(img.GetSize())
    end = [start[i] + size[i] for i in range(3)]

    pad_lower = [max(0, -start[i]) for i in range(3)]
    pad_upper = [max(0, end[i] - img_size[i]) for i in range(3)]

    if any(p > 0 for p in pad_lower + pad_upper):
        img_p = sitk.ConstantPad(img, pad_lower, pad_upper, fill_value)
    else:
        img_p = img

    start_p = [start[i] + pad_lower[i] for i in range(3)]
    roi = sitk.RegionOfInterest(img_p, size=size, index=start_p)

    record = {
        "requested_start_ijk": start,
        "requested_size_ijk": size,
        "pad_lower_ijk": pad_lower,
        "pad_upper_ijk": pad_upper,
        "start_after_pad_ijk": start_p,
        "orig_size_ijk": img_size,
        "padded_size_ijk": list(img_p.GetSize()),
    }
    return roi, record


# ----------------------------
# Intensity ops / global stats
# ----------------------------

def _window_hu(img: sitk.Image, hu_min: float, hu_max: float) -> sitk.Image:
    return sitk.Clamp(img, lowerBound=hu_min, upperBound=hu_max)


def _compute_case_mu_sigma(ct_win: sitk.Image, norm_threshold_hu: float) -> Tuple[float, float, Dict[str, Any]]:
    arr = sitk.GetArrayViewFromImage(ct_win)  # z,y,x
    mask = arr > norm_threshold_hu
    n = int(mask.sum())

    if n < 1000:
        flat = arr.reshape(-1).astype(np.float64)
        mu = float(flat.mean())
        sigma = float(flat.std() + 1e-8)
        return mu, sigma, {"mask_rule": f"fallback_all_voxels (mask_count={n})", "mask_count": n}

    vals = arr[mask].astype(np.float64)
    mu = float(vals.mean())
    sigma = float(vals.std() + 1e-8)
    return mu, sigma, {"mask_rule": f"ct_win > {norm_threshold_hu}", "mask_count": n}


def _robust_global_mu_sigma(case_stats: List[Tuple[float, float, int]]) -> Tuple[float, float]:
    """
    Weighted aggregation:
      global_mu = sum(w_i * mu_i)
      global_var = sum(w_i * sigma_i^2)
    where w_i proportional to mask_count_i.
    """
    if not case_stats:
        raise ValueError("No case stats to aggregate for global normalization.")

    weights = np.array([max(1, n) for _, _, n in case_stats], dtype=np.float64)
    mus = np.array([mu for mu, _, _ in case_stats], dtype=np.float64)
    sigmas = np.array([s for _, s, _ in case_stats], dtype=np.float64)

    w = weights / weights.sum()
    mu_g = float((w * mus).sum())
    var_g = float((w * (sigmas ** 2)).sum())
    sigma_g = float(math.sqrt(max(var_g, 1e-8)))
    return mu_g, sigma_g


def _normalize_global(ct_win: sitk.Image, global_mu: float, global_sigma: float) -> sitk.Image:
    return sitk.ShiftScale(ct_win, shift=-global_mu, scale=1.0 / (global_sigma + 1e-8))


# ----------------------------
# QC overlay
# ----------------------------

def _save_qc_overlay_mid_slice(ct_win: sitk.Image, tumor_mask: sitk.Image, out_png: str) -> None:
    ct_arr = sitk.GetArrayFromImage(ct_win)  # z,y,x
    m_arr = sitk.GetArrayFromImage(_to_uint8(tumor_mask))  # z,y,x
    z = ct_arr.shape[0] // 2

    ct2 = ct_arr[z].astype(np.float32)
    m2 = (m_arr[z] > 0)

    # scale to 0..255
    ct2 = ct2 - ct2.min()
    denom = (ct2.max() - ct2.min() + 1e-8)
    gray = (ct2 / denom * 255.0).astype(np.uint8)

    rgb = np.stack([gray, gray, gray], axis=-1)
    rgb[m2, 0] = 255
    rgb[m2, 1] = (rgb[m2, 1] * 0.3).astype(np.uint8)
    rgb[m2, 2] = (rgb[m2, 2] * 0.3).astype(np.uint8)

    img2d = sitk.GetImageFromArray(rgb)  # y,x,channels
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    sitk.WriteImage(img2d, out_png)


# ----------------------------
# Config
# ----------------------------

@dataclass
class PreprocessConfig:
    # Given Lung1 slice thickness is 3mm, keep z native to avoid volume explosion:
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 3.0)

    # Initial padding guess (tightened indirectly by pass1 min/max crop tuning):
    pad_mm: Tuple[float, float, float] = (60.0, 60.0, 45.0)

    # Window
    window_hu: Tuple[float, float] = (-900.0, 100.0)

    # Normalization mask rule threshold (exclude air)
    norm_threshold_hu: float = -850.0

    # These get overwritten after pass1
    min_box_mm: Tuple[float, float, float] = (160.0, 160.0, 96.0)
    max_box_mm: Tuple[float, float, float] = (256.0, 256.0, 192.0)

    # Global normalization params (computed in pass1)
    global_mu: Optional[float] = None
    global_sigma: Optional[float] = None

    # Optional artifacts
    save_audit_volumes: bool = True
    save_qc_png: bool = True

    # Pass1 tuning knobs
    bbox_percentile: float = 98.0  # cover ~98% of padded tumor bboxes
    hard_min_box_mm: Tuple[float, float, float] = (160.0, 160.0, 96.0)
    hard_max_box_mm: Tuple[float, float, float] = (256.0, 256.0, 192.0)


# ----------------------------
# Pass 1
# ----------------------------

def _pass1_collect_stats(samples: List[Sample], cfg: PreprocessConfig) -> Dict[str, Any]:
    bbox_mm_sizes_padded = []
    case_stats = []  # (mu, sigma, mask_count)
    empty_masks = []
    per_case_records = []

    for patient_id, ct_in, mask_in in samples:
        ct = _orient_to_ras(_to_float32(_read_image(ct_in)))
        mask = _orient_to_ras(_to_uint8(_read_image(mask_in)))

        ct_r = _resample_to_spacing(ct, cfg.target_spacing, sitk.sitkLinear, default_value=-1000.0)
        mask_r = _resample_to_spacing(mask, cfg.target_spacing, sitk.sitkNearestNeighbor, default_value=0.0)

        bbox = _mask_bbox_ijk(mask_r)
        if bbox is None:
            empty_masks.append(patient_id)
            continue

        start, size = bbox
        spacing = ct_r.GetSpacing()

        start_p, size_p, pad_vox = _pad_bbox_mm(start, size, spacing, cfg.pad_mm)
        size_p_mm = [size_p[i] * spacing[i] for i in range(3)]
        bbox_mm_sizes_padded.append(size_p_mm)

        ct_win = _window_hu(ct_r, cfg.window_hu[0], cfg.window_hu[1])
        mu, sigma, info = _compute_case_mu_sigma(ct_win, norm_threshold_hu=cfg.norm_threshold_hu)
        mask_count = int(info.get("mask_count", 0))
        case_stats.append((mu, sigma, mask_count))

        per_case_records.append({
            "patient_id": patient_id,
            "pad_mm": cfg.pad_mm,
            "pad_vox": pad_vox,
            "bbox_start_ijk": start,
            "bbox_size_ijk": size,
            "bbox_padded_size_mm": size_p_mm,
            "case_mu": mu,
            "case_sigma": sigma,
            "case_norm_info": info,
        })

    if not bbox_mm_sizes_padded:
        raise RuntimeError("Pass1 found no usable cases (all masks empty?).")

    arr = np.array(bbox_mm_sizes_padded, dtype=np.float64)  # N x 3
    p = float(cfg.bbox_percentile)
    p_size = np.percentile(arr, p, axis=0)

    hard_min = np.array(cfg.hard_min_box_mm, dtype=np.float64)
    hard_max = np.array(cfg.hard_max_box_mm, dtype=np.float64)

    # Derived MIN: big enough to cover most tumors+context, but capped for cost
    derived_min = np.minimum(np.maximum(p_size, hard_min), hard_max)

    # Derived MAX: keep fixed hard max for compute control
    derived_max = hard_max

    mu_g, sigma_g = _robust_global_mu_sigma(case_stats)

    return {
        "empty_masks": empty_masks,
        "derived_min_box_mm": tuple(float(x) for x in derived_min.tolist()),
        "derived_max_box_mm": tuple(float(x) for x in derived_max.tolist()),
        "global_mu": mu_g,
        "global_sigma": sigma_g,
        "bbox_percentile": p,
        "per_case_records": per_case_records,
    }


# ----------------------------
# Pass 2 (per-case)
# ----------------------------

def _preprocess_one(
    patient_id: str,
    ct_in: PathOrImage,
    mask_in: PathOrImage,
    out_patient_dir: str,
    cfg: PreprocessConfig,
) -> Dict[str, Any]:
    ct = _orient_to_ras(_to_float32(_read_image(ct_in)))
    mask = _orient_to_ras(_to_uint8(_read_image(mask_in)))

    orig_geo = {
        "spacing": ct.GetSpacing(),
        "size_ijk": ct.GetSize(),
        "origin": ct.GetOrigin(),
        "direction": ct.GetDirection(),
        "orientation_policy": "RAS",
    }

    ct_r = _resample_to_spacing(ct, cfg.target_spacing, sitk.sitkLinear, default_value=-1000.0)
    mask_r = _resample_to_spacing(mask, cfg.target_spacing, sitk.sitkNearestNeighbor, default_value=0.0)

    res_geo = {
        "spacing": ct_r.GetSpacing(),
        "size_ijk": ct_r.GetSize(),
        "origin": ct_r.GetOrigin(),
        "direction": ct_r.GetDirection(),
    }

    bbox = _mask_bbox_ijk(mask_r)
    if bbox is None:
        raise RuntimeError(f"Empty tumor mask after resampling for patient_id={patient_id}")

    start, size = bbox
    spacing = ct_r.GetSpacing()

    start_p, size_p, pad_vox = _pad_bbox_mm(start, size, spacing, cfg.pad_mm)
    start_f, size_f, clamp_info = _enforce_min_max_size_mm(start_p, size_p, spacing, cfg.min_box_mm, cfg.max_box_mm)

    ct_roi_hu, crop_ct = _crop_with_pad(ct_r, start_f, size_f, fill_value=-1000.0)
    tumor_roi, crop_m = _crop_with_pad(mask_r, start_f, size_f, fill_value=0.0)

    ct_roi_win = _window_hu(ct_roi_hu, cfg.window_hu[0], cfg.window_hu[1])

    # Track per-case stats for QC (even though we normalize globally)
    case_mu, case_sigma, case_norm_info = _compute_case_mu_sigma(ct_roi_win, norm_threshold_hu=cfg.norm_threshold_hu)

    if cfg.global_mu is None or cfg.global_sigma is None:
        raise RuntimeError("Global normalization params missing. Run pass1 first.")

    ct_roi_norm = _normalize_global(ct_roi_win, cfg.global_mu, cfg.global_sigma)

    os.makedirs(out_patient_dir, exist_ok=True)
    paths = {}

    # Optional audit artifacts
    if cfg.save_audit_volumes:
        p_res = os.path.join(out_patient_dir, "ct_hu_resampled_ras.nii.gz")
        p_hu = os.path.join(out_patient_dir, "ct_roi_hu.nii.gz")
        _write_image(ct_r, p_res)
        _write_image(ct_roi_hu, p_hu)
        paths["ct_hu_resampled_ras"] = p_res
        paths["ct_roi_hu"] = p_hu

    # Model-ready artifacts
    p_norm = os.path.join(out_patient_dir, "ct_roi_norm.nii.gz")
    p_mask = os.path.join(out_patient_dir, "tumor_roi.nii.gz")
    _write_image(ct_roi_norm, p_norm)
    _write_image(tumor_roi, p_mask)
    paths["ct_roi_norm"] = p_norm
    paths["tumor_roi"] = p_mask

    # QC overlay
    if cfg.save_qc_png:
        p_png = os.path.join(out_patient_dir, "qc_overlay_mid.png")
        _save_qc_overlay_mid_slice(ct_roi_win, tumor_roi, p_png)
        paths["qc_overlay_mid"] = p_png

    transforms = {
        "target_spacing": cfg.target_spacing,
        "pad_mm": cfg.pad_mm,
        "pad_vox": pad_vox,
        "min_box_mm": cfg.min_box_mm,
        "max_box_mm": cfg.max_box_mm,
        "window_hu": cfg.window_hu,
        "norm_threshold_hu": cfg.norm_threshold_hu,
        "global_mu": cfg.global_mu,
        "global_sigma": cfg.global_sigma,
        "bbox_from_mask_start_ijk": start,
        "bbox_from_mask_size_ijk": size,
        "bbox_padded_start_ijk": start_p,
        "bbox_padded_size_ijk": size_p,
        "bbox_final_start_ijk": start_f,
        "bbox_final_size_ijk": size_f,
        "clamp_info": clamp_info,
        "crop_record_ct": crop_ct,
        "crop_record_mask": crop_m,
    }

    meta = {
        "patient_id": patient_id,
        "original_ct_geometry": orig_geo,
        "resampled_ct_geometry": res_geo,
        "case_mu": case_mu,
        "case_sigma": case_sigma,
        "case_norm_info": case_norm_info,
        "notes": "Outputs are RAS oriented. CT linear resample. Mask NN resample. Global z-score normalization after windowing.",
    }

    meta_path = os.path.join(out_patient_dir, "meta.json")
    transforms_path = os.path.join(out_patient_dir, "transforms.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    with open(transforms_path, "w") as f:
        json.dump(transforms, f, indent=2)
    paths["meta"] = meta_path
    paths["transforms"] = transforms_path

    return {"patient_id": patient_id, "paths": paths, "meta": meta}


# ----------------------------
# Public API
# ----------------------------

def preprocessing(
    samples: List[Sample],
    out_dir: str,
    cfg: Optional[PreprocessConfig] = None,
) -> Dict[str, Any]:
    """
    Full preprocessing with optional extras enabled by default:
      - RAS reorientation
      - resample to target spacing
      - tumor-mask bbox crop with padding
      - pass1-derived min/max crop sizes
      - windowing
      - GLOBAL normalization (computed in pass1)
      - audit volumes + QC overlay + metadata sidecars

    Returns a manifest containing config, stats paths, and per-patient output records.
    """
    if cfg is None:
        cfg = PreprocessConfig()

    os.makedirs(out_dir, exist_ok=True)

    # Pass 1
    pass1 = _pass1_collect_stats(samples, cfg)
    cfg.min_box_mm = pass1["derived_min_box_mm"]
    cfg.max_box_mm = pass1["derived_max_box_mm"]
    cfg.global_mu = pass1["global_mu"]
    cfg.global_sigma = pass1["global_sigma"]

    dataset_stats_path = os.path.join(out_dir, "dataset_pass1_stats.json")
    with open(dataset_stats_path, "w") as f:
        json.dump(
            {
                "config_after_pass1": asdict(cfg),
                "pass1_summary": {
                    "num_samples_in": len(samples),
                    "num_empty_masks": len(pass1["empty_masks"]),
                    "empty_masks": pass1["empty_masks"],
                    "bbox_percentile": pass1["bbox_percentile"],
                    "derived_min_box_mm": pass1["derived_min_box_mm"],
                    "derived_max_box_mm": pass1["derived_max_box_mm"],
                    "global_mu": pass1["global_mu"],
                    "global_sigma": pass1["global_sigma"],
                },
                "per_case_records": pass1["per_case_records"],
            },
            f,
            indent=2,
        )

    # Pass 2
    outputs = []
    failures = []

    for patient_id, ct_in, mask_in in samples:
        out_patient_dir = os.path.join(out_dir, patient_id)
        try:
            outputs.append(_preprocess_one(patient_id, ct_in, mask_in, out_patient_dir, cfg))
        except Exception as e:
            failures.append({"patient_id": patient_id, "error": str(e)})

    failures_path = os.path.join(out_dir, "failures.json")
    with open(failures_path, "w") as f:
        json.dump(failures, f, indent=2)

    manifest = {
        "config": asdict(cfg),
        "dataset_pass1_stats_path": dataset_stats_path,
        "failures_path": failures_path,
        "num_failures": len(failures),
        "outputs": outputs,
    }

    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest