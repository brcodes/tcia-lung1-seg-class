#!/usr/bin/env python3
"""
build_cohort.py (DuckDB edition)

Build a cohort manifest from a raw manifest CSV for a clinical imaging app.

Input (default):
  ../../data/raw/NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.patient-manifest.csv

Cohort definition (FINAL per user clarification):
    Exclude rows where Overall.Stage is an NA-like literal token
    (case-insensitive, after trimming whitespace) OR when the DICOM audit
    preprocessing signals are absent/not-ready. Only patients with
    ct_seg_and_linkage = TRUE and ct_ser_valid_for_downstream_preprocessing = 'OK'
    remain eligible.

  NA-like tokens (config in code via NA_TOKENS):
    - "NA"
    - "N/A"

Outputs (default directory: ./outputs):
    Always written (Parquet):
                - cleaned.parquet             (raw manifest reduced to PatientID + Overall.Stage)
        - eligibility_table.parquet   (all patients + flags + exclusion_reason)
        - cohort.parquet              (eligible only)
        - exclusions.parquet          (ineligible only)
        - cohort_summary.json         (counts, CONSORT-style + run metadata)
    Optionally written (CSV, if --write-csv):
                - cleaned.csv
        - eligibility_table.csv
        - cohort.csv
        - exclusions.csv

Prerequisite audit artifact:
        ../../data/audit_dicoms/dicom_audit.json (and its flattened derivative
        patient_eligibility_flat.parquet) produced by run_unified_dicom_audit.

Why DuckDB:
  - Embedded SQL engine (no server)
  - Professional + reproducible cohort definitions in SQL
  - Excellent CSV/Parquet support

Dependencies:
  pip install duckdb pandas pyarrow

How to run:
  # From the directory containing this script:
  pip install duckdb pandas pyarrow
  python build_cohort.py --output-dir ./outputs --write-csv

  # If your CSV is elsewhere:
  python build_cohort.py --input /path/to/raw.csv --output-dir ./outputs --write-csv

Artifacts:
  - ./outputs/cohort.duckdb is created for inspection/audit (you can open it with DuckDB).
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import duckdb
import pandas as pd


DEFAULT_INPUT = Path("../../data/raw/NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.patient-manifest.csv")
DEFAULT_OUTPUT_DIR = Path("../../data/cohort")
DEFAULT_DUCKDB_PATH = Path("../../data/cohort/cohort.duckdb")
DEFAULT_AUDIT_UNIFIED_JSON = Path("../../data/audit_dicoms/dicom_audit.json")
DEFAULT_AUDIT_ELIGIBILITY_FLAT = Path("../../data/audit_dicoms/patient_eligibility_flat.parquet")

RAW_TABLE = "patient_manifest_raw"
CLEANED_TABLE = "cleaned"
ELIG_TABLE = "eligibility"
COHORT_TABLE = "cohort"
EXCLUSIONS_TABLE = "exclusions"
AUDIT_ELIGIBILITY_TABLE = "audit_preprocessing_eligibility"

# NA-like tokens observed/expected in the CSV; comparison is case-insensitive after trimming.
NA_TOKENS = ("NA", "N/A")

DEFAULT_CLEANING_SQL = Path(__file__).with_suffix("").parent / "cleaning.sql"
DEFAULT_ELIGIBILITY_SQL = Path(__file__).with_suffix("").parent / "eligibility.sql"

AUDIT_ELIGIBILITY_COLUMNS = [
    "patient_id",
    "ct_seg_and_linkage",
    "ct_seg_and_linkage_reasons_json",
    "ct_seg_linked_files_json",
    "ct_ser_valid_for_downstream_preprocessing",
    "ct_ser_issues_json",
    "ct_ser_warnings_json",
    "ct_ser_segments_json",
]


def _sanitize_flat_row(row: Dict[str, Any]) -> Dict[str, str]:
    sanitized: Dict[str, str] = {}
    for field in AUDIT_ELIGIBILITY_COLUMNS:
        value = row.get(field) if isinstance(row, dict) else None
        sanitized[field] = "" if value is None else str(value)
    return sanitized


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True)


def _rows_from_unified_payload(unified: Dict[str, Any]) -> list[Dict[str, str]]:
    rows: list[Dict[str, str]] = []

    existing = unified.get("eligibility_table")
    if isinstance(existing, list):
        for raw in existing:
            if isinstance(raw, dict):
                rows.append(_sanitize_flat_row(raw))
        if rows or existing:
            return rows

    patient_container = unified.get("by_patient_id")
    if not isinstance(patient_container, dict):
        return rows

    patient_map = patient_container.get("patient_id")
    if not isinstance(patient_map, dict):
        return rows

    for patient_id, payload in patient_map.items():
        if not isinstance(payload, dict):
            continue

        eligibility = payload.get("eligibility")
        if not isinstance(eligibility, dict):
            continue

        ct_seg = eligibility.get("ct_seg_and_linkage") or {}
        ct_ser = eligibility.get("ct_ser_valid_for_downstream_preprocessing") or {}

        row = {
            "patient_id": str(patient_id),
            "ct_seg_and_linkage": "TRUE" if bool(ct_seg.get("value")) else "FALSE",
            "ct_seg_and_linkage_reasons_json": _json_dumps(ct_seg.get("reasons") or []),
            "ct_seg_linked_files_json": _json_dumps(ct_seg.get("linked_seg_files") or []),
            "ct_ser_valid_for_downstream_preprocessing": str(ct_ser.get("decision") or ""),
            "ct_ser_issues_json": _json_dumps(ct_ser.get("issues") or []),
            "ct_ser_warnings_json": _json_dumps(ct_ser.get("warnings") or []),
            "ct_ser_segments_json": _json_dumps(ct_ser.get("segments") or []),
        }

        rows.append(_sanitize_flat_row(row))

    return rows


def run_sql_template(*, con: duckdb.DuckDBPyConnection, sql_path: Path, replacements: Dict[str, str] | None = None) -> None:
    if not sql_path.exists():
        raise FileNotFoundError(f"SQL file not found at: {sql_path}")

    sql = sql_path.read_text(encoding="utf-8")
    if replacements:
        for key, value in replacements.items():
            sql = sql.replace(key, value)

    # Execute scripts deterministically statement-by-statement.
    # This avoids relying on multi-statement behavior that can vary across drivers/versions,
    # and prevents stale tables from previous runs from persisting unnoticed.
    non_comment_lines: list[str] = []
    for line in sql.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("--"):
            continue
        non_comment_lines.append(line)
    sql_no_comments = "\n".join(non_comment_lines)

    for statement in sql_no_comments.split(";"):
        statement = statement.strip()
        if statement:
            con.execute(statement)


def build_cleaned(con: duckdb.DuckDBPyConnection) -> None:
    """Creates CLEANED_TABLE from RAW_TABLE via cleaning.sql."""
    run_sql_template(con=con, sql_path=DEFAULT_CLEANING_SQL)


@dataclass(frozen=True)
class RunMetadata:
    run_utc: str
    input_path: str
    input_sha256: str
    duckdb_path: str
    cohort_definition_version: str
    na_tokens: Tuple[str, ...]
    cleaning_sql_path: str
    cleaning_sql_sha256: str
    eligibility_sql_path: str
    eligibility_sql_sha256: str
    audit_unified_json_path: str
    audit_unified_json_sha256: str
    audit_eligibility_flat_path: str | None
    audit_eligibility_flat_sha256: str | None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_file_optional(path: Optional[Path]) -> Optional[str]:
    if path is None or not path.exists():
        return None
    return sha256_file(path)


def read_raw_csv_as_strings(path: Path) -> Tuple[pd.DataFrame, Dict]:
    """
    Clinical-grade choice: read all columns as strings (no inference),
    preserve literal tokens like 'NA'/'N/A' exactly as present in file.
    """
    df = pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,  # don't auto-convert to NaN
        na_filter=False,
        quoting=csv.QUOTE_MINIMAL,
        encoding="utf-8",
    )
    meta = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": list(df.columns),
    }
    return df, meta


def validate_manifest(df: pd.DataFrame) -> None:
    required = ["PatientID", "Overall.Stage"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")

    pid = df["PatientID"].astype(str)
    blank_pid = (pid.str.strip() == "")
    if blank_pid.any():
        raise ValueError(
            f"Found {int(blank_pid.sum())} rows with blank PatientID. "
            "This violates the stated assumption 'one row per patientid'."
        )

    dup = pid.duplicated(keep=False)
    if dup.any():
        dup_ids = sorted(set(pid[dup].tolist()))
        raise ValueError(
            "Found duplicate PatientID values (violates 'one row per patientid'). "
            f"Examples: {dup_ids[:20]}{'...' if len(dup_ids) > 20 else ''}"
        )


def build_tables(con: duckdb.DuckDBPyConnection, *, audit_table: str) -> None:
    """
    Creates ELIG_TABLE, COHORT_TABLE, EXCLUSIONS_TABLE from RAW_TABLE.
    """

    # Quote tokens for SQL IN (...) list; escape single-quotes defensively.
    na_list_sql = ", ".join(["'" + t.replace("'", "''") + "'" for t in NA_TOKENS])

    run_sql_template(
        con=con,
        sql_path=DEFAULT_ELIGIBILITY_SQL,
        replacements={
            "{{NA_LIST_SQL}}": na_list_sql,
            "{{AUDIT_ELIG_TABLE}}": audit_table,
        },
    )


def export_table_parquet(con: duckdb.DuckDBPyConnection, table: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    con.execute(
        "COPY (SELECT * FROM " + table + " ORDER BY PatientID) TO ? (FORMAT 'parquet');",
        [str(out_path)],
    )


def export_table_csv(con: duckdb.DuckDBPyConnection, table: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    con.execute(
        "COPY (SELECT * FROM " + table + " ORDER BY PatientID) TO ? (FORMAT 'csv', HEADER TRUE);",
        [str(out_path)],
    )


def compute_summary(
    con: duckdb.DuckDBPyConnection,
    run_meta: RunMetadata,
    raw_csv_meta: Dict,
    *,
    audit_table: str,
) -> Dict:
    total_rows = con.execute(f"SELECT COUNT(*) FROM {RAW_TABLE};").fetchone()[0]
    total_patients = con.execute(f"SELECT COUNT(DISTINCT PatientID) FROM {RAW_TABLE};").fetchone()[0]
    cleaned_rows = con.execute(f"SELECT COUNT(*) FROM {CLEANED_TABLE};").fetchone()[0]
    cleaned_patients = con.execute(f"SELECT COUNT(DISTINCT PatientID) FROM {CLEANED_TABLE};").fetchone()[0]
    eligible = con.execute(f"SELECT COUNT(*) FROM {COHORT_TABLE};").fetchone()[0]
    excluded = con.execute(f"SELECT COUNT(*) FROM {EXCLUSIONS_TABLE};").fetchone()[0]

    reasons = con.execute(
        f"""
        SELECT exclusion_reason, COUNT(*) AS n
        FROM {ELIG_TABLE}
        WHERE is_eligible = FALSE
        GROUP BY exclusion_reason
        ORDER BY n DESC, exclusion_reason;
        """
    ).fetchall()

    audit_total_rows = con.execute(f"SELECT COUNT(*) FROM {audit_table};").fetchone()[0]
    audit_missing_patients = con.execute(
        f"SELECT COUNT(*) FROM {ELIG_TABLE} WHERE audit_has_preprocessing_row = FALSE;"
    ).fetchone()[0]
    audit_warn_patients = con.execute(
        f"SELECT COUNT(*) FROM {ELIG_TABLE} WHERE flag_ct_ser_decision_warn = TRUE;"
    ).fetchone()[0]
    audit_failure_patients = con.execute(
        f"SELECT COUNT(*) FROM {ELIG_TABLE} WHERE flag_ct_ser_decision_failure = TRUE;"
    ).fetchone()[0]
    audit_ok_patients = con.execute(
        f"SELECT COUNT(*) FROM {ELIG_TABLE} WHERE flag_ct_ser_decision_ok = TRUE;"
    ).fetchone()[0]
    audit_seg_linkage_false = con.execute(
        f"""
        SELECT COUNT(*)
        FROM {ELIG_TABLE}
        WHERE audit_has_preprocessing_row = TRUE
          AND coalesce(flag_ct_seg_and_linkage_true, FALSE) = FALSE;
        """
    ).fetchone()[0]
    audit_decision_missing = con.execute(
        f"SELECT COUNT(*) FROM {ELIG_TABLE} WHERE coalesce(flag_ct_ser_decision_missing, FALSE) = TRUE;"
    ).fetchone()[0]

    return {
        "run_metadata": {
            "run_utc": run_meta.run_utc,
            "input_path": run_meta.input_path,
            "input_sha256": run_meta.input_sha256,
            "duckdb_path": run_meta.duckdb_path,
            "cohort_definition_version": run_meta.cohort_definition_version,
            "cleaning_sql_path": run_meta.cleaning_sql_path,
            "cleaning_sql_sha256": run_meta.cleaning_sql_sha256,
            "eligibility_sql_path": run_meta.eligibility_sql_path,
            "eligibility_sql_sha256": run_meta.eligibility_sql_sha256,
            "audit_unified_json_path": run_meta.audit_unified_json_path,
            "audit_unified_json_sha256": run_meta.audit_unified_json_sha256,
            "audit_eligibility_flat_path": run_meta.audit_eligibility_flat_path,
            "audit_eligibility_flat_sha256": run_meta.audit_eligibility_flat_sha256,
            "cohort_definition": {
                "excluded_if": (
                    "stage token in "
                    f"{list(run_meta.na_tokens)} OR missing DICOM audit row OR "
                    "ct_seg_and_linkage != TRUE OR ct_ser_valid_for_downstream_preprocessing != 'OK'"
                ),
                "na_tokens": list(run_meta.na_tokens),
                "exclusion_reason_codes": [
                    "stage_overall_is_na",
                    "dicom_audit_missing",
                    "ct_seg_and_linkage_not_true",
                    "ct_ser_preproc_failure",
                    "ct_ser_preproc_warn",
                    "ct_ser_preproc_missing",
                ],
            },
        },
        "raw_csv_characteristics": raw_csv_meta,
        "counts": {
            "rows_in_raw_table": int(total_rows),
            "unique_patients_in_raw_table": int(total_patients),
            "rows_in_cleaned_table": int(cleaned_rows),
            "unique_patients_in_cleaned_table": int(cleaned_patients),
            "eligible": int(eligible),
            "excluded": int(excluded),
        },
        "preprocessing_audit": {
            "rows_in_audit": int(audit_total_rows),
            "patients_missing_in_audit": int(audit_missing_patients),
            "patients_ok": int(audit_ok_patients),
            "patients_warn": int(audit_warn_patients),
            "patients_failure": int(audit_failure_patients),
            "patients_seg_linkage_false": int(audit_seg_linkage_false),
            "patients_preproc_decision_missing": int(audit_decision_missing),
        },
        "exclusion_breakdown": [{"exclusion_reason": r[0], "n": int(r[1])} for r in reasons],
        "consort": {
            "assessed_for_eligibility": int(total_patients),
            "excluded": int(excluded),
            "included_in_cohort": int(eligible),
        },
    }


def build_cohort(
    *,
    input_path: Path = DEFAULT_INPUT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    duckdb_path: Path = DEFAULT_DUCKDB_PATH,
    audit_unified_json: Path = DEFAULT_AUDIT_UNIFIED_JSON,
    audit_eligibility_flat: Optional[Path] = None,
    write_csv: bool = False,
) -> Dict:
    """
    Notebook-friendly entrypoint.

    Runs the cohort build with explicit parameters (no CLI/argparse required).

    Returns the cohort summary dict that is also written to `cohort_summary.json`.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found at: {input_path}")

    audit_unified_json = Path(audit_unified_json)
    if not audit_unified_json.exists():
        raise FileNotFoundError(
            "Unified DICOM audit JSON not found. Run the DICOM audit to generate dicom_audit.json first: "
            f"{audit_unified_json}"
        )

    with audit_unified_json.open("r", encoding="utf-8") as f_json:
        audit_unified_payload = json.load(f_json)

    flat_rows = _rows_from_unified_payload(audit_unified_payload)
    audit_df = pd.DataFrame(flat_rows, columns=AUDIT_ELIGIBILITY_COLUMNS)

    required_audit_columns = set(AUDIT_ELIGIBILITY_COLUMNS)
    missing_audit_columns = required_audit_columns.difference(audit_df.columns)
    if missing_audit_columns:
        raise ValueError(
            "Audit eligibility data is missing required columns: " + ", ".join(sorted(missing_audit_columns))
        )

    audit_df["patient_id"] = audit_df["patient_id"].astype(str)
    blank_audit_ids = audit_df["patient_id"].str.strip() == ""
    if blank_audit_ids.any():
        raise ValueError("Audit eligibility data contains blank patient_id values.")

    duplicate_audit_ids = audit_df["patient_id"].duplicated(keep=False)
    if duplicate_audit_ids.any():
        dup_ids = sorted(set(audit_df.loc[duplicate_audit_ids, "patient_id"]))
        raise ValueError(
            "Audit eligibility data must have one row per patient. Duplicates detected: "
            + ", ".join(dup_ids[:20])
            + ("..." if len(dup_ids) > 20 else "")
        )

    derived_files = audit_unified_payload.get("derived_files")
    derived_flat_path: Optional[Path] = None
    if isinstance(derived_files, dict):
        flat_entry = derived_files.get("patient_eligibility_flat")
        if isinstance(flat_entry, str) and flat_entry.strip():
            derived_flat_path = Path(flat_entry)

    audit_flat_override = Path(audit_eligibility_flat) if audit_eligibility_flat is not None else None
    audit_flat_path = audit_flat_override or derived_flat_path

    df, raw_csv_meta = read_raw_csv_as_strings(input_path)
    validate_manifest(df)

    output_dir.mkdir(parents=True, exist_ok=True)

    audit_flat_path_str = str(audit_flat_path) if audit_flat_path else None
    audit_flat_sha = sha256_file_optional(audit_flat_path)

    run_meta = RunMetadata(
        run_utc=utc_now_iso(),
        input_path=str(input_path),
        input_sha256=sha256_file(input_path),
        duckdb_path=str(duckdb_path),
        cohort_definition_version="v4-duckdb",
        na_tokens=NA_TOKENS,
        cleaning_sql_path=str(DEFAULT_CLEANING_SQL),
        cleaning_sql_sha256=sha256_file(DEFAULT_CLEANING_SQL),
        eligibility_sql_path=str(DEFAULT_ELIGIBILITY_SQL),
        eligibility_sql_sha256=sha256_file(DEFAULT_ELIGIBILITY_SQL),
        audit_unified_json_path=str(audit_unified_json),
        audit_unified_json_sha256=sha256_file(audit_unified_json),
        audit_eligibility_flat_path=audit_flat_path_str,
        audit_eligibility_flat_sha256=audit_flat_sha,
    )

    con = duckdb.connect(str(duckdb_path))

    con.register("raw_df", df)
    con.execute(f"DROP TABLE IF EXISTS {RAW_TABLE};")
    con.execute(f"CREATE TABLE {RAW_TABLE} AS SELECT * FROM raw_df;")

    con.register("audit_df", audit_df)
    con.execute(f"DROP TABLE IF EXISTS {AUDIT_ELIGIBILITY_TABLE};")
    con.execute(
        f"CREATE TABLE {AUDIT_ELIGIBILITY_TABLE} AS SELECT * FROM audit_df;"
    )

    build_cleaned(con)
    build_tables(con, audit_table=AUDIT_ELIGIBILITY_TABLE)

    export_table_parquet(con, CLEANED_TABLE, output_dir / "cleaned.parquet")

    export_table_parquet(con, ELIG_TABLE, output_dir / "eligibility_table.parquet")
    export_table_parquet(con, COHORT_TABLE, output_dir / "cohort.parquet")
    export_table_parquet(con, EXCLUSIONS_TABLE, output_dir / "exclusions.parquet")

    if write_csv:
        export_table_csv(con, CLEANED_TABLE, output_dir / "cleaned.csv")
        export_table_csv(con, ELIG_TABLE, output_dir / "eligibility_table.csv")
        export_table_csv(con, COHORT_TABLE, output_dir / "cohort.csv")
        export_table_csv(con, EXCLUSIONS_TABLE, output_dir / "exclusions.csv")

    summary = compute_summary(con, run_meta, raw_csv_meta, audit_table=AUDIT_ELIGIBILITY_TABLE)
    (output_dir / "cohort_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    con.close()
    return summary


if __name__ == "__main__":
    summary = build_cohort()
    print(f"Done. Outputs written to: {DEFAULT_OUTPUT_DIR.resolve()}")