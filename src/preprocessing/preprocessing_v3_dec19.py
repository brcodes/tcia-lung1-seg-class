"""
Preprocess CT NIfTI volumes + aligned tumor masks for segmentation + downstream classification.

Updated to align ROI sizing with your expected MONAI patch sizes:
  - Primary patch: 192x192x64
  - Alternate patch: 160x160x64

Key update:
- Pass1 now derives dataset-informed ROI box sizes (mm), then *snaps* them to
  "patch-friendly" voxel sizes (multiples of 32) close to your target patch options.
- Pass2 enforces that final ROI crop sizes in voxels are compatible (multiples of 32),
  which tends to reduce padding/waste during training and avoids shape issues.

Orientation policy:
- Standardize everything to RAS (outputs saved as RAS NIfTI).

Normalization policy:
- Window then GLOBAL z-score (mu/sigma computed in Pass1 across cases).
- Per-case mu/sigma still computed and stored for QC.

Assumptions:
- CT is already in HU.
- Tumor mask is aligned with CT in physical space.
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


def _snap_size_to_patch_friendly(
    size_vox: int,
    preferred: List[int],
    multiple_of: int = 32,
    min_vox: Optional[int] = None,
    max_vox: Optional[int] = None,
) -> int:
    """
    Snap a voxel size to something patch-friendly:
    - Must be multiple_of (default 32)
    - Prefer closeness to one of 'preferred' sizes (e.g., [160, 192, 224, 256])
    - Still respect min_vox/max_vox if provided
    """
    if min_vox is not None:
        size_vox = max(size_vox, min_vox)
    if max_vox is not None:
        size_vox = min(size_vox, max_vox)

    # candidates: preferred sizes filtered by min/max and multiple_of,
    # plus nearest multiples around size_vox if preferred doesn't cover it.
    candidates = []
    for p in preferred:
        if p % multiple_of != 0:
            continue
        if min_vox is not None and p < min_vox:
            continue
        if max_vox is not None and p > max_vox:
            continue
        candidates.append(p)

    # Ensure we always have candidates
    if not candidates:
        # generate a few multiples around size_vox
        base = int(round(size_vox / multiple_of) * multiple_of)
        candidates = sorted(set([base - multiple_of, base, base + multiple_of, base + 2 * multiple_of]))
        candidates = [c for c in candidates if c > 0]
        if min_vox is not None:
            candidates = [c for c in candidates if c >= min_vox]
        if max_vox is not None:
            candidates = [c for c in candidates if c <= max_vox]
        if not candidates:
            return max(multiple_of, base)

    # choose candidate closest to size_vox (tie-break: smaller to save compute)
    candidates = sorted(candidates, key=lambda c: (abs(c - size_vox), c))
    return int(candidates[0])


def _enforce_min_max_size_mm_and_snap(
    start: List[int],
    size: List[int],
    spacing: Tuple[float, float, float],
    min_mm: Tuple[float, float, float],
    max_mm: Tuple[float, float, float],
    preferred_patch_sizes_xyz: Tuple[List[int], List[int], List[int]],
    multiple_of: int = 32,
) -> Tuple[List[int], List[int], Dict[str, Any]]:
    """
    Enforce min/max size in mm, then snap voxel sizes to patch-friendly sizes.
    """
    info = {
        "expanded_to_min": [False, False, False],
        "clamped_to_max": [False, False, False],
        "snapped_to_preferred": [False, False, False],
        "final_size_vox": None,
    }

    center = [start[i] + size[i] / 2.0 for i in range(3)]
    min_vox = [int(math.ceil(min_mm[i] / spacing[i])) for i in range(3)]
    max_vox = [int(math.floor(max_mm[i] / spacing[i])) for i in range(3)]

    # enforce min/max
    s = size[:]
    for i in range(3):
        if s[i] < min_vox[i]:
            s[i] = min_vox[i]
            info["expanded_to_min"][i] = True
        if s[i] > max_vox[i]:
            s[i] = max_vox[i]
            info["clamped_to_max"][i] = True

    # snap each dim to preferred patch sizes
    preferred_x, preferred_y, preferred_z = preferred_patch_sizes_xyz
    preferred = [preferred_x, preferred_y, preferred_z]
    snapped = s[:]
    for i in range(3):
        snapped_i = _snap_size_to_patch_friendly(
            s[i],
            preferred=preferred[i],
            multiple_of=multiple_of,
            min_vox=min_vox[i],
            max_vox=max_vox[i],
        )
        if snapped_i != s[i]:
            info["snapped_to_preferred"][i] = True
        snapped[i] = snapped_i

    new_start = [int(round(center[i] - snapped[i] / 2.0)) for i in range(3)]
    info["final_size_vox"] = snapped
    return new_start, snapped, info


def _crop_with_pad(
    img: sitk.Image,
    start: List[int],
    size: List[int],
    fill_value: float,
) -> Tuple[sitk.Image, Dict[str, Any]]:
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

    ct2 = ct2 - ct2.min()
    denom = (ct2.max() - ct2.min() + 1e-8)
    gray = (ct2 / denom * 255.0).astype(np.uint8)

    rgb = np.stack([gray, gray, gray], axis=-1)
    rgb[m2, 0] = 255
    rgb[m2, 1] = (rgb[m2, 1] * 0.3).astype(np.uint8)
    rgb[m2, 2] = (rgb[m2, 2] * 0.3).astype(np.uint8)

    img2d = sitk.GetImageFromArray(rgb)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    sitk.WriteImage(img2d, out_png)


# ----------------------------
# Config
# ----------------------------

@dataclass
class PreprocessConfig:
    # Lung1: 3mm slices, keep z native by default
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 3.0)

    # initial padding around tumor bbox (mm)
    pad_mm: Tuple[float, float, float] = (60.0, 60.0, 45.0)

    # window
    window_hu: Tuple[float, float] = (-900.0, 100.0)
    norm_threshold_hu: float = -850.0

    # Derived/overwritten after pass1
    min_box_mm: Tuple[float, float, float] = (160.0, 160.0, 96.0)
    max_box_mm: Tuple[float, float, float] = (256.0, 256.0, 192.0)

    # Global normalization after pass1
    global_mu: Optional[float] = None
    global_sigma: Optional[float] = None

    # Optional artifacts
    save_audit_volumes: bool = True
    save_qc_png: bool = True

    # Pass1 tuning
    bbox_percentile: float = 98.0

    # Hard caps aligned with patch sizes you want to try
    # NOTE: hard_max matches 192x192x64 physical extent at 1x1x3mm => (192mm,192mm,192mm)
    hard_min_box_mm: Tuple[float, float, float] = (160.0, 160.0, 192.0)  # prefer z=64 slices coverage
    hard_max_box_mm: Tuple[float, float, float] = (192.0, 192.0, 192.0)  # 192x192x64 @ 1x1x3mm

    # Patch-friendly voxel sizes we want to snap to (multiples of 32)
    # For (x,y): 160 or 192 are your primary candidates; allow 224/256 as escape if needed.
    preferred_sizes_xy: List[int] = None
    preferred_sizes_z: List[int] = None

    # Make ROI sizes divisible by 32 (safe for SwinUNETR)
    snap_multiple_of: int = 32

    def __post_init__(self):
        if self.preferred_sizes_xy is None:
            self.preferred_sizes_xy = [160, 192, 224, 256]
        if self.preferred_sizes_z is None:
            self.preferred_sizes_z = [64, 96]  # 64 preferred; 96 as fallback if tumor is very tall


# ----------------------------
# Pass 1
# ----------------------------

def _pass1_collect_stats(samples: List[Sample], cfg: PreprocessConfig) -> Dict[str, Any]:
    bbox_mm_sizes_padded = []
    case_stats = []
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
            "bbox_padded_size_mm": size_p_mm,
            "case_mu": mu,
            "case_sigma": sigma,
            "case_norm_info": info,
        })

    if not bbox_mm_sizes_padded:
        raise RuntimeError("Pass1 found no usable cases (all masks empty?).")

    arr = np.array(bbox_mm_sizes_padded, dtype=np.float64)
    p = float(cfg.bbox_percentile)
    p_size = np.percentile(arr, p, axis=0)

    hard_min = np.array(cfg.hard_min_box_mm, dtype=np.float64)
    hard_max = np.array(cfg.hard_max_box_mm, dtype=np.float64)

    derived_min = np.minimum(np.maximum(p_size, hard_min), hard_max)
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
# Pass 2
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

    ct_r = _resample_to_spacing(ct, cfg.target_spacing, sitk.sitkLinear, default_value=-1000.0)
    mask_r = _resample_to_spacing(mask, cfg.target_spacing, sitk.sitkNearestNeighbor, default_value=0.0)

    bbox = _mask_bbox_ijk(mask_r)
    if bbox is None:
        raise RuntimeError(f"Empty tumor mask after resampling for patient_id={patient_id}")

    start, size = bbox
    spacing = ct_r.GetSpacing()

    start_p, size_p, pad_vox = _pad_bbox_mm(start, size, spacing, cfg.pad_mm)

    start_f, size_f, size_info = _enforce_min_max_size_mm_and_snap(
        start=start_p,
        size=size_p,
        spacing=spacing,
        min_mm=cfg.min_box_mm,
        max_mm=cfg.max_box_mm,
        preferred_patch_sizes_xyz=(cfg.preferred_sizes_xy, cfg.preferred_sizes_xy, cfg.preferred_sizes_z),
        multiple_of=cfg.snap_multiple_of,
    )

    ct_roi_hu, crop_ct = _crop_with_pad(ct_r, start_f, size_f, fill_value=-1000.0)
    tumor_roi, crop_m = _crop_with_pad(mask_r, start_f, size_f, fill_value=0.0)

    ct_roi_win = _window_hu(ct_roi_hu, cfg.window_hu[0], cfg.window_hu[1])

    # QC per-case stats (even though global norm is used)
    case_mu, case_sigma, case_norm_info = _compute_case_mu_sigma(ct_roi_win, norm_threshold_hu=cfg.norm_threshold_hu)

    ct_roi_norm = _normalize_global(ct_roi_win, cfg.global_mu, cfg.global_sigma)

    os.makedirs(out_patient_dir, exist_ok=True)
    paths = {}

    if cfg.save_audit_volumes:
        p_res = os.path.join(out_patient_dir, "ct_hu_resampled_ras.nii.gz")
        p_hu = os.path.join(out_patient_dir, "ct_roi_hu.nii.gz")
        _write_image(ct_r, p_res)
        _write_image(ct_roi_hu, p_hu)
        paths["ct_hu_resampled_ras"] = p_res
        paths["ct_roi_hu"] = p_hu

    p_norm = os.path.join(out_patient_dir, "ct_roi_norm.nii.gz")
    p_mask = os.path.join(out_patient_dir, "tumor_roi.nii.gz")
    _write_image(ct_roi_norm, p_norm)
    _write_image(tumor_roi, p_mask)
    paths["ct_roi_norm"] = p_norm
    paths["tumor_roi"] = p_mask

    if cfg.save_qc_png:
        p_png = os.path.join(out_patient_dir, "qc_overlay_mid.png")
        _save_qc_overlay_mid_slice(ct_roi_win, tumor_roi, p_png)
        paths["qc_overlay_mid"] = p_png

    meta = {
        "patient_id": patient_id,
        "target_spacing": cfg.target_spacing,
        "pad_mm": cfg.pad_mm,
        "pad_vox": pad_vox,
        "final_roi_size_vox": size_f,
        "final_roi_size_mm": [size_f[i] * spacing[i] for i in range(3)],
        "window_hu": cfg.window_hu,
        "global_mu": cfg.global_mu,
        "global_sigma": cfg.global_sigma,
        "case_mu": case_mu,
        "case_sigma": case_sigma,
        "case_norm_info": case_norm_info,
        "bbox_start_ijk": start,
        "bbox_size_ijk": size,
        "bbox_padded_start_ijk": start_p,
        "bbox_padded_size_ijk": size_p,
        "bbox_final_start_ijk": start_f,
        "bbox_final_size_ijk": size_f,
        "size_info": size_info,
        "crop_record_ct": crop_ct,
        "crop_record_mask": crop_m,
        "orientation_policy": "RAS",
        "notes": "ROI voxel sizes snapped to patch-friendly multiples of 32; x/y prefer 160 or 192; z prefers 64.",
    }

    meta_path = os.path.join(out_patient_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    paths["meta"] = meta_path

    return {"patient_id": patient_id, "paths": paths, "meta": meta}


# ----------------------------
# Public API
# ----------------------------

def preprocessing(
    samples: List[Sample],
    out_dir: str,
    cfg: Optional[PreprocessConfig] = None,
) -> Dict[str, Any]:
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