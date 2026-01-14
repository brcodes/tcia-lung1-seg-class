import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd


def iso_now() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()

class UIDMapCollector:
    def __init__(self):
        self._rows: List[Dict[str, Any]] = []

    def add_uid(
        self,
        old_uid: str,
        new_uid: str,
        uid_type: str,
        parent_uid: Optional[str] = None,
        sop_class_uid: Optional[str] = None,
    ) -> None:
        self._rows.append(
            {
                "old_uid": old_uid,
                "new_uid": new_uid,
                "uid_type": uid_type,
                "parent_uid": parent_uid,
                "sop_class_uid": sop_class_uid,
            }
        )

    def to_parquet(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(self._rows)
        df.to_parquet(path, index=False)



class DeidAuditWriter:
    def __init__(self, output_path: Path, script_version: str, ps3_15_profile_version: str):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.script_version = script_version
        self.ps3_15_profile_version = ps3_15_profile_version
        self._fh = open(self.output_path, "w", encoding="utf-8")

    def log_instance(
        self,
        sop_uid: str,
        input_path_token: str,
        output_path_token: str,
        modality: Optional[str],
        tags_removed: List[str],
        tags_cleaned: List[str],
        private_tags_stripped: bool,
        burned_in_text_scan: str,
        checksum_sha256: str,
        tags_removed_eq_max_rem_srch_matches: bool,
        tags_cleaned_eq_max_cln_srch_matches: bool,
        tags_stripped_eq_max_str_srch_matches: Optional[bool],
        deid_criteria_version: str,
        deid: bool,
    ) -> None:
        record = {
            "sop_uid": sop_uid,
            "input_path_token": input_path_token,
            "output_path_token": output_path_token,
            "modality": modality,
            "tags_removed": tags_removed,
            "tags_removed_eq_max_rem_srch_matches": tags_removed_eq_max_rem_srch_matches,
            "tags_cleaned": tags_cleaned,
            "tags_cleaned_eq_max_cln_srch_matches": tags_cleaned_eq_max_cln_srch_matches,
            "private_tags_stripped": private_tags_stripped,
            "tags_stripped_eq_max_str_srch_matches": tags_stripped_eq_max_str_srch_matches,
            "burned_in_text_scan": burned_in_text_scan,
            "checksum_sha256": checksum_sha256,
            "deid": deid,
            "deid_criteria_version": deid_criteria_version,
            "path_token_scheme": "hmac_sha256_v1",
            "script_version": self.script_version,
            "ps3_15_profile_version": self.ps3_15_profile_version,
            "timestamp": iso_now(),
        }
        self._fh.write(json.dumps(record) + "\n")

    def close(self) -> None:
        self._fh.close()

class MetadataAuditCollector:
    def __init__(self):
        self._rows: List[Dict[str, Any]] = []

    def add_instance(
        self,
        sop_uid: str,
        series_uid: str,
        study_uid: str,
        modality: str,
        body_part_examined: Optional[str],
        series_description: Optional[str],
        manufacturer: Optional[str],
        manufacturer_model_name: Optional[str],
        slice_thickness: Optional[float],
        pixel_spacing: Optional[List[float]],
        image_orientation_patient: Optional[List[float]],
        image_position_patient: Optional[List[float]],
        rows: Optional[int],
        columns: Optional[int],
        bits_stored: Optional[int],
        bits_allocated: Optional[int],
    ) -> None:
        pixel_spacing_row = pixel_spacing[0] if pixel_spacing else None
        pixel_spacing_col = pixel_spacing[1] if pixel_spacing else None

        self._rows.append(
            {
                "sop_uid": sop_uid,
                "series_uid": series_uid,
                "study_uid": study_uid,
                "modality": modality,
                "body_part_examined": body_part_examined,
                "series_description": series_description,
                "manufacturer": manufacturer,
                "manufacturer_model_name": manufacturer_model_name,
                "slice_thickness": slice_thickness,
                "pixel_spacing_row": pixel_spacing_row,
                "pixel_spacing_col": pixel_spacing_col,
                "image_orientation_patient": image_orientation_patient,
                "image_position_patient": image_position_patient,
                "rows": rows,
                "columns": columns,
                "bits_stored": bits_stored,
                "bits_allocated": bits_allocated,
            }
        )

    def to_parquet(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(self._rows)
        df.to_parquet(path, index=False)

class CTPreproAudit:
    def __init__(self, script_version: str):
        self.script_version = script_version
        self.series_records: List[Dict[str, Any]] = []

    def add_series(
        self,
        series_uid: str,
        study_uid: str,
        num_slices: int,
        slice_thickness: Optional[float],
        pixel_spacing: Optional[List[float]],
        z_spacing_mean: Optional[float],
        z_spacing_std: Optional[float],
        orientation_matrix: Optional[List[float]],
        volume_extent_mm: Optional[List[float]],
        is_uniform_spacing: bool,
        requires_resampling: bool,
        recommended_preprocessing_pipeline: str,
        notes: Optional[str] = None,
    ) -> None:
        self.series_records.append(
            {
                "series_uid": series_uid,
                "study_uid": study_uid,
                "num_slices": num_slices,
                "slice_thickness": slice_thickness,
                "pixel_spacing": pixel_spacing,
                "z_spacing_mean": z_spacing_mean,
                "z_spacing_std": z_spacing_std,
                "orientation_matrix": orientation_matrix,
                "volume_extent_mm": volume_extent_mm,
                "is_uniform_spacing": is_uniform_spacing,
                "requires_resampling": requires_resampling,
                "recommended_preprocessing_pipeline": recommended_preprocessing_pipeline,
                "notes": notes,
            }
        )

    def to_json(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "script_version": self.script_version,
            "timestamp": iso_now(),
            "series": self.series_records,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
