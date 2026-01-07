# Copilot Chat Conversation Export: Swin-UNETR preprocessing workflow

**User:** @brcodes
**Thread URL:** https://github.com/copilot/c/05217738-5060-44db-8311-23e2ffbf5b32

## @brcodes

this is an incomplete but informative json metadata file (output) for a preprocessing pipeline. I want to preprocess Lung1 Dicoms into Nifti , saving important prepro and base image metadata. The context is creating a clinical app in the Epic App Orchard enviornment (FHIR standards). This particular part doesn't have any PHI. I've been told this workflow is good for what I want- prepro CT lung images for a fine tuning of swin-UNETR with a dual-task cost function (segmentation AND classification of cancer stage), Lung1. Steps are in order of what has been recommended to me. Can you think of anything else? Looking for moderate fine-tune compute here (so not giant 3d volumes- necessary ones only). 80/20 train test split. Don't write json just respond with a pseudocode plan.

"voxel_spacing_mm": image.GetSpacing(),
    "preprocessing_pipeline": [
        {
            "step": "resampling",
            "target": "isotropic_1mm"
        },
        {
            "step": "cropping",
            "method": "bounding_box_lung_mask_with_padding",
            "params": {
                "mask_source": "swin-unetr",
                "padding_mm": { "x": 40, "y": 40, "z": 40 },
                "min_box_mm": [192, 192, 192],
                "max_box_mm": [320, 320, 320]
            }
        },
        {
            "step": "windowing",
            "center": -400,
            "width": 1000,
            "range_hu": [-900, 100]
        },
        {
            "step": "intensity_normalization",
            "method": "z-score",
            "applied_after": "windowing"
        }
    ],

## Copilot

## Pseudocode plan: Lung1 DICOM → NIfTI with preprocessing + metadata (no PHI), moderate compute

### 0) Define goals / constants
- Target model: Swin-UNETR fine-tuning
- Dual-task labels: segmentation (lung / tumor depending on available GT) + classification (cancer stage)
- Compute constraint: avoid full-resolution full-body volumes; keep only necessary cropped ROI volumes
- Split: patient-level 80/20 train/test (optionally stratified by stage)

---

### 1) Ingest + validate DICOM series (per patient / study / series)
```
for each patient_folder in Lung1_dataset:
  series_list = discover_dicom_series(patient_folder)

  select series where:
    modality == "CT"
    and axial chest CT (best effort: SeriesDescription/body part)
    and slice_count adequate
    and consistent ImageOrientationPatient
    and not localizer/scout

  if multiple candidate CT series:
    prefer:
      - thin slices (smaller SliceThickness)
      - complete lung coverage
      - kernel appropriate (if known)
```

**Add checks you’ll be glad you had:**
- Detect mixed slice spacing, gantry tilt, missing slices
- Detect inconsistent rescale slope/intercept across slices
- Confirm HU conversion works (sanity: air around -1000, soft tissue ~0)

---

### 2) Convert DICOM → canonical 3D volume (HU) + canonical orientation
```
img = load_series_as_volume(series)         # keep geometry
img = to_hounsfield_units(img)              # apply RescaleSlope/Intercept
img = reorient_to_RAS(img)                  # canonical orientation for all cases
```

**Metadata to save (non-PHI):**
- voxel spacing, direction cosines, origin
- slice thickness, manufacturer/model (optional), convolution kernel (optional)
- HU conversion params (min/max HU observed)
- series UID hash (not raw UID), study date offset or removed (avoid PHI leakage)

---

### 3) Optional: quality control gates (skip / flag failures)
```
qc = {}
qc["hu_air_check"] = (percent_voxels_in_range(img, [-1100,-700]) > threshold)
qc["body_coverage_check"] = (z_extent_mm(img) > min_required)
qc["artifact_score"] = estimate_artifact(img)   # simple heuristic ok

if qc fails hard:
  mark_case_excluded_with_reason()
  continue
```

---

### 4) Resampling (your Step 1) — but consider anisotropy-aware choice
You were told isotropic 1mm; that’s good but can be expensive in z if original slices are thick.

**Moderate compute option:**
- If original z spacing is coarse (e.g., 2.5–5mm), consider resampling to **1×1×2mm** or **1×1×1.5mm** to avoid hallucinating detail and exploding volume size.
```
target_spacing = choose_spacing(img.spacing,
                               default=(1,1,1),
                               if z_spacing > 2.0 -> (1,1,2))
img_r = resample(img, spacing=target_spacing, interpolator="linear")
```
Save: interpolation type, target spacing, and pre/post shapes.

---

### 5) Lung mask generation (needed for your Step 2 cropping)
You listed mask_source = swin-unetr. That’s fine, but add a fallback.

```
if pretrained_lung_segmenter_available:
  lung_mask = predict_lung_mask(img_r)          # fast model or classical
else:
  lung_mask = classical_lung_segmentation(img_r) # threshold + morphology + CC

lung_mask = postprocess_lung_mask(lung_mask):
  - keep largest 2 connected components
  - fill holes
  - optional airway cleanup
```

Save: mask method, model version/hash, and simple mask stats (volume in mL, bbox extents).

---

### 6) Cropping to lung bounding box + padding (your Step 2) with safeguards
```
bbox = bounding_box(lung_mask)
bbox = pad_bbox_mm(bbox, padding_mm=(40,40,40), spacing=img_r.spacing)

# enforce min/max box size constraints (in mm)
bbox = clamp_bbox_size_mm(bbox, min=(192,192,192), max=(320,320,320))

roi_img = crop(img_r, bbox)
roi_mask = crop(lung_mask, bbox)

# if bbox goes out of image bounds: pad with air HU (-1000)
roi_img = pad_to_bbox(roi_img, fill_value=-1000)
```

**Extra idea (helps classification):**
- Save *two* ROIs optionally:
  1) lung ROI (for segmentation)
  2) slightly larger mediastinum-aware ROI or “thorax ROI” if stage classification benefits from nodes/mediastinum
  (This can be optional/configurable to keep compute moderate.)

---

### 7) Windowing (your Step 3) — store both “model input” and “raw HU” if feasible
For training, windowing is good. For future clinical auditability, raw HU can be useful.

```
roi_hu = roi_img                                # keep HU as float
roi_win = clip(roi_hu, min=-900, max=100)       # per your range
```

Save window params.

---

### 8) Intensity normalization (your Step 4) — prefer robust normalization within ROI
Z-score after windowing is common, but ensure stability:
- Compute mean/std over **non-air** voxels, or within lung mask, to avoid padding dominating.
```
mask_for_norm = roi_mask OR (roi_win > -850)    # heuristic non-air
mu = mean(roi_win[mask_for_norm])
sigma = std(roi_win[mask_for_norm]) + eps
roi_norm = (roi_win - mu) / sigma
```

Save mu/sigma and mask rule.

---

### 9) Label alignment (segmentation + stage classification)
```
seg_label = load_segmentation_if_available(patient_id)     # tumor or lung GT
seg_label = reorient/resample/crop_using_same_transforms(seg_label)

stage_label = load_stage_label(patient_id)                 # e.g., I/II/III/IV
encode stage_label as integer or ordinal
```

Important: apply *identical spatial transforms* (orientation, resampling, crop) to any segmentation labels using nearest-neighbor interpolation.

---

### 10) Write outputs (NIfTI + sidecar metadata, no PHI)
Per case, write:
- `ct_roi_norm.nii.gz` (model input)
- optionally `ct_roi_hu.nii.gz` (auditable HU)
- `lung_mask_roi.nii.gz`
- `seg_label_roi.nii.gz` if present
- metadata sidecar (your JSON concept, but you said don’t write JSON here)

Also save a per-case transform record:
- original spacing/orientation → reoriented → resampled → crop bbox indices/mm
This is critical if you ever need to map predictions back to original DICOM geometry.

---

### 11) Dataset split (patient-level, stratified)
```
patients = unique_patient_ids()
train_ids, test_ids = split(patients, ratio=0.8, stratify_by=stage_label, seed=fixed)
write split manifest
```
Avoid leakage: never split at slice/series level.

---

### 12) Optional but strongly recommended additions (lightweight, high value)
1) **Anonymization guardrails (even if you believe no PHI):**
   - DICOM can contain PHI in headers; ensure you never persist raw headers.
   - If you store any identifiers, store salted hashes only.

2) **Spacing-aware patching for training rather than bigger crops:**
   - Keep ROI moderate, then train with fixed-size patches (e.g., 128³ or 160³) sampled within lung/tumor to control memory.

3) **Simple intensity harmonization flags:**
   - Record kernel / manufacturer if available and consider using it later for domain shift analysis.

4) **QC thumbnails (non-identifying):**
   - Save a few mid-slices PNGs of windowed ROI + mask overlay for quick pipeline validation (no burned-in text).

5) **Class imbalance handling plan:**
   - Stage distribution may be skewed; record counts per split and plan weighted loss/sampling.

---

If you tell me (a) typical original slice thickness in Lung1, (b) whether you have tumor masks or only lung masks, and (c) how many cases per stage, I can tighten the resampling choice (1mm vs 1×1×2mm), ROI sizing, and patch strategy to stay in “moderate compute” while preserving the signal needed for staging.

## @brcodes

(a) the ONLY slice thickness in Lung1 (legitimate scan series only, not calibration scans) is 3 mm, (b) tumor masks  only, and thats all i want, and (c) how many cases per stage, this is my class balance Stage (multi-class) I: 93 II: 40 IIIa: 112 IIIb: 176 

## Copilot

## Updated pseudocode plan (given: 3mm slices, tumor masks only, stage counts)

### Key choices for “moderate compute”
- Keep **in-plane resampling to 1mm**, but **do not force z to 1mm** from 3mm.
  - Use **1×1×3mm** (native z) or **1×1×2mm** (mild upsample) depending on memory.
  - Rationale: going 3→1mm triples slices (and volume) with limited new information.
- Crop tightly around **tumor mask** (not lung mask), with a safety margin to capture context.
- Train with fixed-size patches to avoid huge volumes.

---

## 0) Constants / config
```
TARGET_SPACING = (1.0, 1.0, 3.0)        # recommended default for 3mm source
# optional alternative: (1.0, 1.0, 2.0) if you want a bit more z resolution

TUMOR_PAD_MM = (50, 50, 30)             # more in x/y than z since z is coarse
MIN_BOX_MM   = (160, 160, 96)           # keep compute moderate
MAX_BOX_MM   = (256, 256, 192)

WINDOW_HU = (-900, 100)                # as you had
NORM_MASK_RULE = "non-air or within tumor-expanded ROI"

STAGES = ["I","II","IIIa","IIIb"]       # 4-class
STAGE_COUNTS = {I:93, II:40, IIIa:112, IIIb:176}
SPLIT = 0.8
SEED = fixed
```

---

## 1) Ingest DICOM CT + convert to HU + canonical orientation
```
for case in dataset:
  series = select_legit_ct_series(case)           # exclude scouts/calibration
  img = load_series(series)
  img = to_hu(img)                                # slope/intercept
  img = reorient_to_RAS(img)
```

Save non-PHI base metadata: spacing/origin/direction, slice thickness=3mm, kernel/manufacturer if present, pre-HU min/max.

---

## 2) Load tumor mask (GT) and align to CT before any resampling
Tumor mask likely comes in its own format/space. Ensure it matches the CT geometry first.
```
tumor_mask = load_tumor_mask(case)
tumor_mask = reorient_to_match(tumor_mask, img)

assert same_geometry_or_known_transform(tumor_mask, img)
```

---

## 3) Resample CT and mask (anisotropy-aware)
```
img_r = resample(img, spacing=TARGET_SPACING, interp="linear")

tumor_r = resample(tumor_mask, spacing=TARGET_SPACING, interp="nearest")
```

Save: target spacing, interpolation methods, pre/post shapes.

---

## 4) Crop around tumor bounding box + padding (primary ROI)
This replaces your lung-mask crop since you only want tumor masks.
```
bbox = bounding_box(tumor_r)
bbox = pad_bbox_mm(bbox, padding_mm=TUMOR_PAD_MM, spacing=img_r.spacing)

bbox = clamp_bbox_size_mm(bbox, min=MIN_BOX_MM, max=MAX_BOX_MM)

ct_roi = crop_with_padding(img_r, bbox, fill_value=-1000)     # air fill
tumor_roi = crop_with_padding(tumor_r, bbox, fill_value=0)
```

**Safeguards:**
- If tumor mask is empty/missing → exclude case or route to “unlabeled” bucket (don’t silently proceed).
- Record bbox in both voxel indices and physical mm so you can map predictions back later.

---

## 5) Windowing + robust normalization (avoid padding dominating)
```
ct_win = clip(ct_roi, WINDOW_HU.min, WINDOW_HU.max)

# build normalization mask: focus on body/tissue region
norm_mask = (ct_win > -850)   # crude non-air
# optionally: dilate(tumor_roi) OR combine with non-air

mu = mean(ct_win[norm_mask])
sigma = std(ct_win[norm_mask]) + eps
ct_norm = (ct_win - mu) / sigma
```

Save: window params, mu/sigma, norm_mask rule.

---

## 6) Write NIfTI outputs + transform record (no PHI)
Per case:
- `ct_norm_roi.nii.gz`
- `tumor_roi.nii.gz`
- sidecar metadata (conceptually)
- transform record: orientation change, resample spacing, crop bbox

---

## 7) Patient-level 80/20 split with stratification (given imbalance)
You want stratified split across 4 stages, but Stage II is small (40), so keep it stable.

```
patients = list_unique_patients()

labels = stage_label_per_patient  # one stage per patient
train_ids, test_ids = stratified_split(patients, labels, train_frac=0.8, seed=SEED)

# sanity check counts per stage in each split; if Stage II too small in test:
# enforce minimum test samples per class (e.g., >=6-8) with a constrained split.
write_split_manifest(train_ids, test_ids)
```

---

## 8) Training-time sampling strategy (keeps compute moderate and helps dual-task)
Even with cropped ROIs, use patches so Swin-UNETR doesn’t need full volumes.

```
for each training iteration:
  case = sample_case(weighted_by_stage_and_tumor_presence)

  # segmentation: oversample patches that contain tumor voxels
  patch = sample_patch(ct_norm, center_near=tumor_roi, patch_size=(160,160,64))

  # classification: either
  # (A) use same patch with an auxiliary head, OR
  # (B) use the full ROI pooled features (global pooling) + stage head
```

**Class imbalance handling (recommended):**
- Use weighted cross-entropy or focal loss for stage head:
  - weights ~ inverse frequency (or “effective number of samples”)
- Keep segmentation loss (Dice + CE) separate and combine:
```
loss = λ_seg * (Dice + CE) + λ_cls * (WeightedCE_or_Focal)
```

---

## 9) Extra checks specific to tumor-only workflow (things people forget)
1) **Tumor mask topology QC**
```
if connected_components(tumor_roi) is huge_count or tiny specks:
  flag as suspicious (annotation noise)
```

2) **Tumor context sufficiency**
- Ensure padding captures enough peritumoral context; if tumor touches crop boundary after padding/clamp, log it.

3) **Stage label validity**
- Ensure only {I, II, IIIa, IIIb}; handle unknowns explicitly.

---

If you want, I can suggest concrete values for:
- `TUMOR_PAD_MM`, `MIN_BOX_MM`, `MAX_BOX_MM`
- patch size for 1×1×3mm spacing that fits typical 16–24GB GPUs
based on what GPU memory you’re targeting and whether you want to keep z at 3mm or upsample to 2mm.

## @brcodes

yes for both, given that i want to run this in azure ml studio, and not spend a ton of money

## Copilot

### Recommended settings (cheap-ish Azure ML, 3mm source, Swin-UNETR fine-tune)

#### Resampling choice
Use **TARGET_SPACING = (1.0, 1.0, 3.0)** as your default.
- It keeps z native (no 3× slice explosion).
- It’s usually the best cost/benefit when all scans are 3mm.

Only switch to **(1.0, 1.0, 2.0)** if you confirm it materially helps segmentation/classification *and* you can afford ~1.5× more voxels in z.

---

## Concrete ROI crop parameters (tumor-mask driven)

### Padding (physical mm)
Good starting point:
```
TUMOR_PAD_MM = (60, 60, 45)   # x,y,z
```
Why:
- More context in x/y for pleura/chest wall relationships.
- In z, 45mm at 3mm slices ≈ 15 slices of context, which is usually adequate without blowing up volume size.

### Min/Max ROI box (physical mm)
For **1×1×3mm** spacing:
```
MIN_BOX_MM = (160, 160, 96)   # -> ~160×160×32 vox
MAX_BOX_MM = (256, 256, 192)  # -> ~256×256×64 vox
```
These bounds keep ROI sizes in a regime that’s manageable and still large enough for most tumors + context.

**Implementation detail**
```
bbox = bbox_from_tumor(tumor_r)
bbox = pad_bbox_mm(bbox, TUMOR_PAD_MM)

bbox = enforce_min_size_mm(bbox, MIN_BOX_MM)
bbox = enforce_max_size_mm(bbox, MAX_BOX_MM, strategy="center_on_tumor")
```
If tumor would be clipped by MAX_BOX, log it as a QC flag (don’t silently cut off tumor).

---

## Patch sizes that fit “not expensive” Azure GPUs

### If you keep z at 3mm (recommended)
Start with:
```
PATCH_SIZE_VOX = (160, 160, 48)   # voxels at 1×1×3mm
```
- Physical size ≈ 160mm × 160mm × 144mm.
- Usually fits fine-tuning on common Azure options without going huge.

If memory is tighter:
```
PATCH_SIZE_VOX = (128, 128, 48)
```

If memory is comfortable and you want more context:
```
PATCH_SIZE_VOX = (192, 192, 48)   # first thing to try before increasing z
```

### If you upsample to z=2mm (more expensive)
Start with:
```
PATCH_SIZE_VOX = (160, 160, 64)   # 64 slices at 2mm = 128mm coverage
```
If OOM:
```
PATCH_SIZE_VOX = (128, 128, 64)
```

---

## Sampling strategy (keeps compute low and performance decent)
For each training step:
```
with probability 0.7:
  sample patch centered near tumor (or within dilated tumor mask)
with probability 0.3:
  sample patch from non-tumor lung/nearby tissue (hard negatives)
```
This reduces wasted batches on pure background while still teaching discrimination.

---

## Azure ML “don’t spend a ton” operational tips (still pseudocode-level)
```
choose_single_gpu_vm(low_mid_cost)   # e.g., 1 GPU
enable_mixed_precision(True)         # fp16
use_gradient_accumulation(if needed)
limit_num_workers_and_cache_smartly  # don’t over-provision CPU
store preprocessed NIfTIs in blob storage once; reuse across runs
early_stop_on_val_plateau()
```

---

## Stage imbalance (you have a small Stage II)
Given counts I:93 II:40 IIIa:112 IIIb:176, do:
- **Stratified patient-level split**
- **Weighted loss** for the classification head:
```
class_weights = inverse_frequency_or_effective_number(STAGE_COUNTS)
loss_cls = weighted_cross_entropy(stage_logits, stage_label, class_weights)
```
Keep segmentation loss standard (Dice+CE), and combine:
```
loss = 1.0 * loss_seg + 0.3 * loss_cls   # starting point
```
(Then tune λ based on whether classification underfits.)

---

## Final recommended defaults (copy into your config)
- **Spacing:** `(1.0, 1.0, 3.0)`
- **Tumor padding (mm):** `(60, 60, 45)`
- **ROI min/max (mm):** `min=(160,160,96)`, `max=(256,256,192)`
- **Patch size (vox):** `(160,160,48)` (fallback `(128,128,48)`)

If you tell me what GPU class you’re targeting in Azure ML (e.g., ~16GB vs ~24GB), I can tighten the patch sizes and whether you should bother trying z=2mm at all.

## @brcodes

will swin-unetr take non cubic volumes?

## Copilot

Yes. Swin-UNETR can take **non-cubic 3D volumes/patches** (e.g., 160×160×48) as long as the spatial sizes are compatible with the network’s downsampling / windowing constraints.

What you need to watch:

1) **Divisibility by the total downsampling factor**
- Like other UNet-style encoders/decoders, Swin-UNETR downsamples multiple times.
- Practical rule: make each dimension divisible by \(2^L\) where \(L\) is the number of downsampling stages (commonly 4 → **16**).
- Safe patch examples:  
  - 160×160×48 (all divisible by 16)  
  - 128×128×64 (all divisible by 16)

2) **Swin “window size” constraints**
- Swin blocks use local attention windows (often window size like 7). Some implementations pad internally; others require the feature-map sizes to be compatible with the window partitioning.
- In MONAI’s SwinUNETR, this is usually handled via padding, but to avoid surprises, prefer sizes that are “nice” (divisible by 16, and not tiny in z).

3) **Batching/memory**
- Non-cubic is often *better* for anisotropic medical data (like your 1×1×3mm), because you can spend voxels where resolution is real (x/y) without wasting memory on z.

If you tell me which Swin-UNETR implementation you’re using (MONAI is most common) and the intended `patch_size`/`window_size`, I can confirm the exact shape constraints for that config.

## @brcodes

MONAI

## Copilot

Yes—MONAI’s `monai.networks.nets.SwinUNETR` supports **non-cubic** 3D inputs (non-equal H/W/D). That’s a common setup for anisotropic CT.

### What MONAI requires in practice

1) **Each spatial dim must be divisible by the model’s total downsampling factor**
SwinUNETR has a hierarchical encoder; with the default stage structure, a safe rule is:

- Make `(H, W, D)` divisible by **32** (most conservative), and **at least divisible by 16** in many configs.

If you pick patch sizes divisible by 32, you almost never hit shape issues.

Good non-cubic patch examples (work well and are “safe”):
- `160 × 160 × 64`  (160 divisible by 32? yes (5×32), 64 divisible by 32 yes)
- `192 × 192 × 64`
- `256 × 256 × 64` (bigger, more expensive)

Also typically OK (often divisible by 16 but not 32 in z, depending on config):
- `160 × 160 × 48` (48 divisible by 16 yes, by 32 no)

If you want “cheap and safe”, I’d start with **160×160×64** at 1×1×3mm.

2) **Swin window partitioning**
Swin uses a `window_size` (commonly `7`). MONAI handles padding internally for window partitioning, so you do *not* need cubic inputs. Just avoid very small dimensions where the window becomes awkward after downsampling.

### Recommended for your case (1×1×3mm)
- Prefer **non-cubic** patches like:
  - `160×160×64` (safe divisibility; moderate cost)
  - If you need cheaper: `128×128×64`
- Keep ROI crops moderate, then sample these patches during training.

### Quick sanity check you can run (pseudocode)
```
model = SwinUNETR(img_size=(160,160,64), in_channels=1, out_channels=2, ...)
x = torch.zeros(1, 1, 160, 160, 64)
y = model(x)   # should run without shape errors
```

If you share your intended `feature_size` and whether you’re using the default `patch_size`/`window_size`, I can tell you whether 160×160×48 is safe in *your exact* MONAI configuration or if you should stick to z=64.

## @brcodes

planning this in code now: assum i have a list of all necessary dicom filepaths for each patient. base_path/Patient1/.../.../i dicom
base_path/Patient1/.../.../n dicom
...
base_path/Patient2/.../.../ i dicom
base_path/Patient2/.../.../ n dicom
...
etc.
Where n can be variable.

write code that uses os to iterate through all patients and turns each collection of dcms per patient into a unified nifti volume. Save to interim_path/PatientI.nii.gz. ... PatienN.nii.gz. I have pydicom, nibabel, and simpleITK available. Whatever is most efficient and clinical

## Copilot

```python name=dicom_to_nifti_per_patient.py
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


def dicom_series_to_nifti(patient_dir: str, out_path: str) -> None:
    reader = sitk.ImageSeriesReader()
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()

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


def main(base_path: str, interim_path: str):
    patient_dirs = find_patient_dirs(base_path)
    if not patient_dirs:
        raise SystemExit(f"No patient subdirectories found under base_path={base_path}")

    failures = []
    for pdir in patient_dirs:
        patient_id = os.path.basename(os.path.normpath(pdir))
        out_path = os.path.join(interim_path, f"{patient_id}.nii.gz")

        try:
            dicom_series_to_nifti(pdir, out_path)
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
    # Example usage:
    #   python dicom_to_nifti_per_patient.py /data/Lung1_DICOM /data/interim_nifti
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python dicom_to_nifti_per_patient.py <base_path> <interim_path>")

    base_path = sys.argv[1]
    interim_path = sys.argv[2]
    main(base_path, interim_path)
```

Notes (clinical/efficiency-relevant, but short):
- This uses `GetGDCMSeriesIDs(patient_dir)` and chooses the series with the most slices. If your patient folders can contain multiple CT series (e.g., contrast/non-contrast), tell me your selection rule and I’ll adjust (SeriesDescription, Kernel, etc.).
- Output orientation will be **LPS** (SimpleITK default). MONAI can handle this; just stay consistent across images and masks. If you need RAS NIfTI explicitly, that can be added as a post-step with nibabel.

## @brcodes

please write code now for "preprocessing( )", which takes a list of tuples in form (patient_id, nii volume), and runs through the preprocessing pipeline you outlined. List out your steps in pseudocode first. Do not include 5foldcv, and I just want to take raw niftis and prepare them for my segmentation and classification tasks. If there are any forked path decisions I need to make, ask first.

## Copilot

### Pseudocode for `preprocessing()` (raw NIfTI → model-ready ROI volumes + metadata)

For each `(patient_id, nii_volume)`:

1) **Load / standardize**
- Read NIfTI (SimpleITK)
- Ensure scalar 3D
- Cast to float32 for processing
- (Optional but recommended) reorient to a canonical orientation (e.g., RAS) or keep consistent LPS and record it

2) **(Optional) Convert to HU**
- If the NIfTI is already HU, keep as-is
- If not HU, apply stored DICOM rescale params (usually not available once you’re at raw NIfTI)

3) **Resample to target spacing (anisotropy-aware)**
- Choose target spacing: (1,1,3) recommended for 3mm source
- Resample with linear interpolation
- Record original spacing/size and target spacing/size

4) **Load / align tumor mask**
- Load tumor mask for patient_id (NIfTI)
- Reorient to match CT if needed
- Resample with nearest-neighbor to the same target spacing as CT
- Validate overlap and non-empty tumor

5) **Compute tumor bounding box + padding**
- Get bbox from tumor mask
- Pad bbox by mm
- Enforce min/max bbox size in mm (clamp)
- Crop CT and mask to bbox; pad outside image with air HU (-1000)

6) **Windowing**
- Clip CT to HU window range (e.g., [-900, 100])

7) **Normalization**
- Compute mean/std on a normalization mask (e.g., ct_win > -850) to avoid air padding dominance
- Z-score normalize
- Record mean/std and norm-mask rule

8) **Write outputs**
- Save `ct_norm_roi.nii.gz`
- Save `tumor_roi.nii.gz`
- Save sidecar metadata needed to reproduce transforms and map predictions back (spacing/origin/direction, bbox indices, padding, mu/sigma, etc.)

9) **Return a list of prepared samples**
- For training: paths (or in-memory arrays) + labels (stage) + metadata

---

## Two fork-path decisions I need from you before I lock the code

1) **Where do tumor masks come from and how are they named?**  
   Options:
   - A) `mask_base_path/{patient_id}.nii.gz`
   - B) `mask_base_path/{patient_id}_mask.nii.gz`
   - C) Provided as a dict `{patient_id: mask_path}`
   - D) Provided already in-memory alongside volumes

2) **What orientation policy do you want for outputs?**
   - A) Keep **SimpleITK default (LPS)** everywhere, just be consistent (simplest).
   - B) Reorient everything to **RAS** before saving (common in research stacks).
   - (MONAI can work with either, but mixing is bad—pick one.)

Reply with:
- mask source option (A/B/C/D) + path/pattern if A/B
- orientation policy (A or B)

Once you answer, here is the `preprocessing()` code skeleton I’ll implement (uses SimpleITK; returns output manifest).

## @brcodes

Sorry, check: the tuples all need to be preprocessed  in the same way- is this done sequentially, or is it grouped into 'meta volume' data ( all niftis) before operations?

## Copilot

Sequential per-volume is the right default.

Most of your steps are **image-specific** transforms (resample, crop to that patient’s tumor bbox, window, z-score using that patient’s intensity distribution). Grouping all NIfTIs into a “meta volume” and transforming jointly isn’t meaningful because:

- **Resampling** depends on each image’s original spacing/origin/direction.
- **Cropping** depends on each patient’s tumor mask bbox (different location/size).
- **Normalization** (z-score) is usually per-volume (or per-ROI) to handle scanner / reconstruction variability.

What *is* “group-level” (computed once, then applied consistently) is just the **configuration**:
- chosen target spacing, padding mm, min/max crop size, HU window range, normalization rule, etc.

There are only two common exceptions where you might want some dataset-level computation:

1) **Global intensity normalization** (instead of per-volume):  
   Compute dataset-wide mean/std (or percentile stats) on training set, then apply to all.  
   Pros: consistent intensity scale. Cons: can be less robust across scanners.

2) **Choosing crop size heuristics**:  
   You can scan all tumor bbox sizes first to pick MIN/MAX box sizes that cover (say) 95% of cases without truncation—then do the real preprocessing per-case.

If you want the simplest, clinically sane pipeline for moderate compute:
- **Do preprocessing sequentially per patient**, using fixed config.
- Optionally do a **quick first pass** that only gathers bbox size stats (no writing) to validate your MIN/MAX settings.

If you answer these two, I’ll implement `preprocessing()` accordingly:
1) Per-volume normalization: **per-case z-score** or **global train-set mean/std**?
2) Do you want a **2-pass mode** (pass1: collect bbox stats; pass2: preprocess + write), or just **1-pass**?

## @brcodes

Do everything including the optional things.

A few clarifiers:

assume list of all nifti tuples is actually in form (patiend id, nifti, tumor mask). mask is stub.
I want to use the orientation policy that is most likely to generate good segmentation data. Probably RAS.


## Copilot

```python name=preprocessing_pipeline.py
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

## @brcodes

forget that last prompt. it was incomplete. 
it should have been:

Do everything including the optional things.

A few clarifiers:

assume list of all nifti tuples is actually in form (patiend id, nifti, tumor mask). mask is stub.
I want to use the orientation policy that is most likely to generate good segmentation data. Probably RAS.
do the quick first pass if it could tighten up the segmentation data for training.
Finally, do what could benefit each preprocessed set the most for segmentation in the norm case- global seems better, but that is just a hunch.

## Copilot

### Pseudocode (with your choices: RAS, optional extras, and a quick first pass)

**Inputs:** list of `(patient_id, ct_nifti, tumor_mask_nifti)` where ct/mask can be paths or in-memory images.

**Global config (fixed across dataset):**
- `target_spacing = (1.0, 1.0, 3.0)` (default for 3mm source)
- tumor crop padding in mm (start guess): `(60,60,45)`
- window HU: `[-900, 100]`
- norm mask rule: `ct_win > -850`
- output orientation: `RAS`
- compute + save QC overlays + metadata

**Pass 1 (stats only; improves crop choices + global norm):**
For each patient:
1. Load CT + mask
2. Reorient both to RAS (consistent)
3. Resample CT (linear) + mask (nearest) to target spacing
4. Compute tumor bbox and “padded bbox” with initial padding
5. Record bbox sizes in mm (pre- and post-padding), and flag if would exceed a proposed max
6. Collect intensity stats for *global normalization*:
   - apply windowing to resampled CT
   - compute per-case mean/std on norm mask
   - aggregate into dataset-level mean/std (robustly)

After pass 1:
7. Set `MIN_BOX_MM` and `MAX_BOX_MM` from bbox distribution (e.g., cover 95–98% of cases without truncation, with caps to control compute)
8. Set `GLOBAL_MU` and `GLOBAL_SIGMA` (robust aggregate), plus keep per-case mu/sigma for logging/QC

**Pass 2 (write artifacts):**
For each patient:
1. Load CT + mask
2. Reorient to RAS
3. Resample to target spacing
4. Crop around tumor bbox + padding, then enforce min/max box (from pass 1)
5. Window
6. Normalize using **global** mean/std (and optionally store per-case mean/std too)
7. Write:
   - `ct_roi_norm.nii.gz` (model input)
   - `tumor_roi.nii.gz`
   - optional audit volumes: resampled HU + cropped HU
   - QC PNG overlay mid-slice
   - metadata + transform record (so you can map predictions back)

Return: manifest list with output paths + metadata per patient.

---

## Code: `preprocess(samples, out_dir, ...)` (SimpleITK-first, RAS outputs)

```python
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
# Helpers: IO / orientation
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


def _orient_to_RAS(img: sitk.Image) -> sitk.Image:
    """
    SimpleITK's DICOM convention is typically LPS. For ML, a consistent
    orientation is what matters. We standardize to RAS here.
    """
    f = sitk.DICOMOrientImageFilter()
    f.SetDesiredCoordinateOrientation("RAS")
    return f.Execute(img)


# ----------------------------
# Helpers: resampling
# ----------------------------

def _resample_to_spacing(
    img: sitk.Image,
    target_spacing: Tuple[float, float, float],
    interpolator: int,
    default_value: float,
) -> sitk.Image:
    original_spacing = img.GetSpacing()
    original_size = img.GetSize()

    # Compute new size = round(old_size * old_spacing / new_spacing)
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
# Helpers: bbox / cropping
# ----------------------------

def _mask_bbox_ijk(mask_u8: sitk.Image) -> Optional[Tuple[List[int], List[int]]]:
    """
    Returns bbox as (start_index_ijk, size_ijk) in voxel coordinates.
    If mask is empty, returns None.
    """
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(mask_u8)
    if not stats.HasLabel(1):
        return None
    # BoundingBox returns: (x, y, z, sizeX, sizeY, sizeZ)
    bb = stats.GetBoundingBox(1)
    start = [int(bb[0]), int(bb[1]), int(bb[2])]
    size = [int(bb[3]), int(bb[4]), int(bb[5])]
    return start, size


def _pad_bbox_mm(
    start: List[int],
    size: List[int],
    spacing: Tuple[float, float, float],
    pad_mm: Tuple[float, float, float],
) -> Tuple[List[int], List[int]]:
    pad_vox = [int(math.ceil(pad_mm[i] / spacing[i])) for i in range(3)]
    new_start = [start[i] - pad_vox[i] for i in range(3)]
    new_size = [size[i] + 2 * pad_vox[i] for i in range(3)]
    return new_start, new_size


def _enforce_min_max_size_mm(
    start: List[int],
    size: List[int],
    spacing: Tuple[float, float, float],
    min_mm: Tuple[float, float, float],
    max_mm: Tuple[float, float, float],
) -> Tuple[List[int], List[int], Dict[str, Any]]:
    """
    Adjust bbox size by expanding/shrinking around its center to satisfy min/max in mm.
    Returns updated (start, size) and info flags.
    """
    info = {"clamped_to_max": [False, False, False], "expanded_to_min": [False, False, False]}
    center = [start[i] + size[i] / 2.0 for i in range(3)]

    # Convert min/max in mm to vox
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
    Crops a region potentially outside image bounds by padding first.
    Returns cropped image and a record of padding/crop indices.
    """
    img_size = list(img.GetSize())
    # Desired region [start, start+size)
    end = [start[i] + size[i] for i in range(3)]

    # Compute padding needed on each side
    pad_lower = [max(0, -start[i]) for i in range(3)]
    pad_upper = [max(0, end[i] - img_size[i]) for i in range(3)]

    # Pad if needed
    if any(p > 0 for p in pad_lower + pad_upper):
        img_p = sitk.ConstantPad(img, pad_lower, pad_upper, fill_value)
    else:
        img_p = img

    # After padding, adjust start
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
# Helpers: intensity ops + stats
# ----------------------------

def _window_hu(img: sitk.Image, hu_min: float, hu_max: float) -> sitk.Image:
    return sitk.Clamp(img, lowerBound=hu_min, upperBound=hu_max)


def _compute_case_mu_sigma(ct_win: sitk.Image, norm_threshold_hu: float = -850.0) -> Tuple[float, float, Dict[str, Any]]:
    """
    Compute mean/std on voxels above norm_threshold_hu to avoid air padding.
    """
    arr = sitk.GetArrayViewFromImage(ct_win)  # z,y,x view
    mask = arr > norm_threshold_hu
    n = int(mask.sum())
    if n < 1000:  # too few voxels; fall back to all voxels
        flat = arr.reshape(-1).astype(np.float64)
        mu = float(flat.mean())
        sigma = float(flat.std() + 1e-8)
        info = {"mask_rule": f"fallback_all_voxels (mask_count={n})"}
        return mu, sigma, info

    vals = arr[mask].astype(np.float64)
    mu = float(vals.mean())
    sigma = float(vals.std() + 1e-8)
    info = {"mask_rule": f"ct_win > {norm_threshold_hu}", "mask_count": n}
    return mu, sigma, info


def _normalize_global(ct_win: sitk.Image, global_mu: float, global_sigma: float) -> sitk.Image:
    # (x - mu) / sigma
    return sitk.ShiftScale(ct_win, shift=-global_mu, scale=1.0 / (global_sigma + 1e-8))


def _robust_global_mu_sigma(case_stats: List[Tuple[float, float, int]]) -> Tuple[float, float]:
    """
    Robust-ish aggregation:
    - take weighted average of mu (weights=mask_count)
    - take weighted average of variance (sigma^2), then sqrt
    """
    if not case_stats:
        raise ValueError("No case stats to aggregate.")
    weights = np.array([max(1, n) for _, _, n in case_stats], dtype=np.float64)
    mus = np.array([mu for mu, _, _ in case_stats], dtype=np.float64)
    sigmas = np.array([s for _, s, _ in case_stats], dtype=np.float64)

    w = weights / weights.sum()
    mu_g = float((w * mus).sum())
    var_g = float((w * (sigmas ** 2)).sum())
    sigma_g = float(math.sqrt(max(var_g, 1e-8)))
    return mu_g, sigma_g


# ----------------------------
# Helpers: QC overlay
# ----------------------------

def _save_qc_overlay_mid_slice(ct_win: sitk.Image, tumor_mask: sitk.Image, out_png: str) -> None:
    """
    Writes a simple mid-z overlay PNG:
    - grayscale CT windowed
    - red contour-ish overlay from tumor mask
    Requires only numpy; uses SimpleITK to write 2D PNG.
    """
    ct_arr = sitk.GetArrayFromImage(ct_win)      # z,y,x
    m_arr = sitk.GetArrayFromImage(_to_uint8(tumor_mask))  # z,y,x

    z = ct_arr.shape[0] // 2
    ct2 = ct_arr[z].astype(np.float32)
    m2 = (m_arr[z] > 0)

    # normalize grayscale to 0..255 for viewing
    ct2 = ct2 - ct2.min()
    denom = (ct2.max() - ct2.min() + 1e-8)
    gray = (ct2 / denom * 255.0).astype(np.uint8)

    rgb = np.stack([gray, gray, gray], axis=-1)
    # overlay mask in red (filled)
    rgb[m2, 0] = 255
    rgb[m2, 1] = (rgb[m2, 1] * 0.3).astype(np.uint8)
    rgb[m2, 2] = (rgb[m2, 2] * 0.3).astype(np.uint8)

    img2d = sitk.GetImageFromArray(rgb)  # y,x,channels
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    sitk.WriteImage(img2d, out_png)


# ----------------------------
# Data structures
# ----------------------------

@dataclass
class PreprocessConfig:
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 3.0)
    pad_mm: Tuple[float, float, float] = (60.0, 60.0, 45.0)
    window_hu: Tuple[float, float] = (-900.0, 100.0)
    norm_threshold_hu: float = -850.0

    # These are set/overridden after pass1 based on bbox stats:
    min_box_mm: Tuple[float, float, float] = (160.0, 160.0, 96.0)
    max_box_mm: Tuple[float, float, float] = (256.0, 256.0, 192.0)

    # Global normalization (set after pass1):
    global_mu: Optional[float] = None
    global_sigma: Optional[float] = None

    save_audit_volumes: bool = True
    save_qc_png: bool = True


# ----------------------------
# Pass 1: stats
# ----------------------------

def _pass1_collect_stats(samples: List[Sample], cfg: PreprocessConfig) -> Dict[str, Any]:
    bbox_mm_sizes = []
    bbox_mm_sizes_padded = []
    case_stats = []  # (mu, sigma, mask_count)
    empty_masks = []
    per_case_records = []

    for patient_id, ct_in, mask_in in samples:
        ct = _orient_to_RAS(_to_float32(_read_image(ct_in)))
        mask = _orient_to_RAS(_to_uint8(_read_image(mask_in)))

        # Resample
        ct_r = _resample_to_spacing(ct, cfg.target_spacing, sitk.sitkLinear, default_value=-1000.0)
        mask_r = _resample_to_spacing(mask, cfg.target_spacing, sitk.sitkNearestNeighbor, default_value=0.0)

        bbox = _mask_bbox_ijk(mask_r)
        if bbox is None:
            empty_masks.append(patient_id)
            continue

        start, size = bbox
        spacing = ct_r.GetSpacing()

        size_mm = [size[i] * spacing[i] for i in range(3)]
        bbox_mm_sizes.append(size_mm)

        start_p, size_p = _pad_bbox_mm(start, size, spacing, cfg.pad_mm)
        size_p_mm = [size_p[i] * spacing[i] for i in range(3)]
        bbox_mm_sizes_padded.append(size_p_mm)

        # intensity stats for global normalization (window first)
        ct_win = _window_hu(ct_r, cfg.window_hu[0], cfg.window_hu[1])
        mu, sigma, info = _compute_case_mu_sigma(ct_win, norm_threshold_hu=cfg.norm_threshold_hu)
        # store mask_count if available
        mask_count = int(info.get("mask_count", 0))
        case_stats.append((mu, sigma, mask_count))

        per_case_records.append({
            "patient_id": patient_id,
            "bbox_mm": size_mm,
            "bbox_padded_mm": size_p_mm,
            "case_mu": mu,
            "case_sigma": sigma,
            "case_norm_info": info,
        })

    if not bbox_mm_sizes_padded:
        raise RuntimeError("No usable cases with non-empty masks found in pass1.")

    # Derive min/max boxes from distribution of padded bbox sizes (mm)
    arr = np.array(bbox_mm_sizes_padded, dtype=np.float64)  # shape (N,3)

    # Choose percentiles to cover most cases, but cap to control compute.
    # You can tune these percentiles; 98% is a good start.
    p98 = np.percentile(arr, 98, axis=0)

    # Hard caps (still in line with earlier moderate-compute defaults)
    hard_min = np.array([160.0, 160.0, 96.0])
    hard_max = np.array([256.0, 256.0, 192.0])

    # min_box: at least hard_min, but not bigger than hard_max
    min_box_mm = np.minimum(np.maximum(p98, hard_min), hard_max)

    # max_box: keep to hard_max (you can also set it to p98 capped)
    max_box_mm = hard_max

    # Global mu/sigma
    mu_g, sigma_g = _robust_global_mu_sigma(case_stats)

    return {
        "empty_masks": empty_masks,
        "bbox_mm_sizes": bbox_mm_sizes,
        "bbox_mm_sizes_padded": bbox_mm_sizes_padded,
        "derived_min_box_mm": tuple(float(x) for x in min_box_mm.tolist()),
        "derived_max_box_mm": tuple(float(x) for x in max_box_mm.tolist()),
        "global_mu": mu_g,
        "global_sigma": sigma_g,
        "per_case_records": per_case_records,
    }


# ----------------------------
# Pass 2: actual preprocessing + writing
# ----------------------------

def _preprocess_one(
    patient_id: str,
    ct_in: PathOrImage,
    mask_in: PathOrImage,
    out_patient_dir: str,
    cfg: PreprocessConfig,
) -> Dict[str, Any]:
    ct = _orient_to_RAS(_to_float32(_read_image(ct_in)))
    mask = _orient_to_RAS(_to_uint8(_read_image(mask_in)))

    # Record original geometry
    orig_meta = {
        "spacing": ct.GetSpacing(),
        "size_ijk": ct.GetSize(),
        "origin": ct.GetOrigin(),
        "direction": ct.GetDirection(),
        "orientation_policy": "RAS",
    }

    # Resample
    ct_r = _resample_to_spacing(ct, cfg.target_spacing, sitk.sitkLinear, default_value=-1000.0)
    mask_r = _resample_to_spacing(mask, cfg.target_spacing, sitk.sitkNearestNeighbor, default_value=0.0)

    resampled_meta = {
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

    # Pad bbox
    start_p, size_p = _pad_bbox_mm(start, size, spacing, cfg.pad_mm)

    # Enforce min/max sizes
    start_f, size_f, clamp_info = _enforce_min_max_size_mm(
        start_p, size_p, spacing, cfg.min_box_mm, cfg.max_box_mm
    )

    # Crop with padding
    ct_roi_hu, crop_record_ct = _crop_with_pad(ct_r, start_f, size_f, fill_value=-1000.0)
    tumor_roi, crop_record_m = _crop_with_pad(mask_r, start_f, size_f, fill_value=0.0)

    # Window + normalization
    ct_roi_win = _window_hu(ct_roi_hu, cfg.window_hu[0], cfg.window_hu[1])

    case_mu, case_sigma, case_norm_info = _compute_case_mu_sigma(ct_roi_win, norm_threshold_hu=cfg.norm_threshold_hu)

    if cfg.global_mu is None or cfg.global_sigma is None:
        raise RuntimeError("Global mu/sigma not set. Run pass1 first or set cfg.global_mu/global_sigma.")

    ct_roi_norm = _normalize_global(ct_roi_win, cfg.global_mu, cfg.global_sigma)

    # Write outputs
    os.makedirs(out_patient_dir, exist_ok=True)

    out = {
        "patient_id": patient_id,
        "paths": {},
        "meta": {},
    }

    if cfg.save_audit_volumes:
        out_ct_resampled = os.path.join(out_patient_dir, "ct_hu_resampled_ras.nii.gz")
        out_ct_roi_hu = os.path.join(out_patient_dir, "ct_roi_hu.nii.gz")
        _write_image(ct_r, out_ct_resampled)
        _write_image(ct_roi_hu, out_ct_roi_hu)
        out["paths"]["ct_hu_resampled_ras"] = out_ct_resampled
        out["paths"]["ct_roi_hu"] = out_ct_roi_hu

    out_ct_roi_norm = os.path.join(out_patient_dir, "ct_roi_norm.nii.gz")
    out_tumor_roi = os.path.join(out_patient_dir, "tumor_roi.nii.gz")
    _write_image(ct_roi_norm, out_ct_roi_norm)
    _write_image(tumor_roi, out_tumor_roi)
    out["paths"]["ct_roi_norm"] = out_ct_roi_norm
    out["paths"]["tumor_roi"] = out_tumor_roi

    if cfg.save_qc_png:
        out_png = os.path.join(out_patient_dir, "qc_overlay_mid.png")
        _save_qc_overlay_mid_slice(ct_roi_win, tumor_roi, out_png)
        out["paths"]["qc_overlay_mid"] = out_png

    # Metadata + transform record (kept minimal but sufficient to reproduce)
    transforms = {
        "target_spacing": cfg.target_spacing,
        "pad_mm": cfg.pad_mm,
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
        "crop_record_ct": crop_record_ct,
        "crop_record_mask": crop_record_m,
    }

    meta = {
        "original_ct_geometry": orig_meta,
        "resampled_ct_geometry": resampled_meta,
        "case_mu": case_mu,
        "case_sigma": case_sigma,
        "case_norm_info": case_norm_info,
        "notes": "Outputs are RAS-oriented. Mask resampled with nearest-neighbor. CT resampled with linear.",
    }

    # Write json sidecars
    meta_path = os.path.join(out_patient_dir, "meta.json")
    transforms_path = os.path.join(out_patient_dir, "transforms.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    with open(transforms_path, "w") as f:
        json.dump(transforms, f, indent=2)

    out["paths"]["meta"] = meta_path
    out["paths"]["transforms"] = transforms_path
    out["meta"] = meta
    return out


# ----------------------------
# Public API
# ----------------------------

def preprocessing(
    samples: List[Sample],
    out_dir: str,
    cfg: Optional[PreprocessConfig] = None,
) -> Dict[str, Any]:
    """
    Runs:
    - pass1: bbox + intensity stats (to tighten min_box_mm and compute global mu/sigma)
    - pass2: resample, tumor-based crop, window, global normalize, write outputs

    Returns a dict with:
      - config used
      - pass1 stats
      - per-patient output manifests
    """
    if cfg is None:
        cfg = PreprocessConfig()

    os.makedirs(out_dir, exist_ok=True)

    # ---- Pass 1 ----
    pass1 = _pass1_collect_stats(samples, cfg)
    cfg.min_box_mm = pass1["derived_min_box_mm"]
    cfg.max_box_mm = pass1["derived_max_box_mm"]
    cfg.global_mu = pass1["global_mu"]
    cfg.global_sigma = pass1["global_sigma"]

    # Save dataset-level stats/config for audit/repro
    dataset_meta_path = os.path.join(out_dir, "dataset_pass1_stats.json")
    with open(dataset_meta_path, "w") as f:
        json.dump(
            {
                "config_after_pass1": asdict(cfg),
                "pass1_summary": {
                    "num_samples_in": len(samples),
                    "num_empty_masks": len(pass1["empty_masks"]),
                    "empty_masks": pass1["empty_masks"],
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

    # ---- Pass 2 ----
    outputs = []
    failures = []

    for patient_id, ct_in, mask_in in samples:
        out_patient_dir = os.path.join(out_dir, patient_id)
        try:
            out = _preprocess_one(patient_id, ct_in, mask_in, out_patient_dir, cfg)
            outputs.append(out)
        except Exception as e:
            failures.append({"patient_id": patient_id, "error": str(e)})

    # Save failures
    failures_path = os.path.join(out_dir, "failures.json")
    with open(failures_path, "w") as f:
        json.dump(failures, f, indent=2)

    return {
        "config": asdict(cfg),
        "dataset_pass1_stats_path": dataset_meta_path,
        "failures_path": failures_path,
        "num_failures": len(failures),
        "outputs": outputs,
    }
```

### Notes / things to be aware of
- This assumes your **tumor mask is already aligned to the CT** (same physical space). If not, we need the registration/transform step (not included).
- I implemented **global normalization** (computed in pass 1 from windowed, non-air voxels) and still compute per-case mu/sigma for QC.
- Output sidecars are written as JSON here because the pipeline needs somewhere to persist metadata. If you truly don’t want any JSON files written, say so and I’ll switch to CSV + a Python pickle, or return metadata only in-memory.

If you answer one thing: are your tumor masks guaranteed to have the same geometry/space as the CT NIfTI? If not, what format/space are they in?

## @brcodes

they are. 

## Copilot

```python name=preprocessing_pipeline.py
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
```

If you want this to be even cheaper on Azure ML:
- set `save_audit_volumes=False` (it avoids writing two extra NIfTIs per patient)
- keep `save_qc_png=True` (very small, useful sanity check)

If you tell me your expected MONAI patch size (e.g., 160×160×64), I can adjust the hard max box so the ROI cropping tends to align with your patch strategy (less padding/waste during training).

## @brcodes

expected is 192 × 192 × 64 but want to be able to try 160,160,64

## Copilot

```python name=preprocessing_pipeline.py
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
```

How this matches your patch-size goals:

- With `target_spacing=(1,1,3)`, a z-size of **64 vox** corresponds to **192mm** coverage.
- The hard max box is set to **(192mm,192mm,192mm)**, which naturally yields ROI sizes that align with **192×192×64**.
- The snap-to-preferred step biases x/y to **160 or 192**, so you can train with either patch size with minimal waste.

If you later find some tumors routinely need more context than 192mm in-plane, the cheapest adjustment is to expand `preferred_sizes_xy` to include 224 and increase `hard_max_box_mm` to 224mm (still fairly manageable).

## @brcodes

the z would still be 64 in that case? brief response, please.

## Copilot

Yes—**by default** with this config, the ROI **z-size will snap to 64 slices** (because `preferred_sizes_z = [64, 96]` and `hard_max_box_mm` is set to 192mm, which at 3mm spacing equals 64 slices).

It will only go above 64 if you **raise `hard_max_box_mm` in z** (and/or make 96 allowable by that max).

## @brcodes

would such a skewed xy to z make a difference in my segmentation goasl?

## Copilot

Probably not in a harmful way for Lung1 CT, and it’s a common/appropriate compromise—because your data are inherently anisotropic (1×1×3mm after resample).

What matters:

- **You’re not “skewing” the anatomy**, you’re matching the acquisition: z simply contains less information per slice. Using fewer z slices (relative to x/y) usually saves compute without losing much detail you never had.
- For tumor segmentation, most boundary detail is in-plane; z-boundaries are already coarse at 3mm, so making z physically smaller than x/y often has minimal downside.

When it *can* hurt:
- **Very tall tumors** (large cranio-caudal extent) or cases where context along z matters; then 64 slices (≈192mm) may clip useful context. Your pass1 stats + the option to allow **z=96** are there to cover this.
- If you later resample to **1×1×2mm**, you should revisit z (64 then is only 128mm coverage).

Practical recommendation: keep **z=64** as default, but track a QC flag like “tumor touches z-boundary of ROI”; if that happens often, allow more z (e.g., 96) or increase z padding/max.
