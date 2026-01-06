#!/usr/bin/env python3
"""
build_cohort.py (DuckDB edition)

Build a cohort manifest from a raw manifest CSV for a clinical imaging app.

Input (default):
  ../../data/raw/NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.patient-manifest.csv

Cohort definition (FINAL per user clarification):
  Exclude rows where Stage.Overall is an NA-like literal token
  (case-insensitive, after trimming whitespace). All other rows are eligible.

  NA-like tokens (config in code via NA_TOKENS):
    - "NA"
    - "N/A"

Outputs (default directory: ./outputs):
  Always written (Parquet):
    - eligibility_table.parquet   (all patients + flags + exclusion_reason)
    - cohort.parquet              (eligible only)
    - exclusions.parquet          (ineligible only)
    - cohort_summary.json         (counts, CONSORT-style + run metadata)
  Optionally written (CSV, if --write-csv):
    - eligibility_table.csv
    - cohort.csv
    - exclusions.csv

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
from typing import Dict, Tuple

import duckdb
import pandas as pd


DEFAULT_INPUT = Path("../../data/raw/NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.patient-manifest.csv")
DEFAULT_OUTPUT_DIR = Path("../../data/cohort")
DEFAULT_DUCKDB_PATH = Path("../../data/cohort/cohort.duckdb")

RAW_TABLE = "patient_manifest_raw"
ELIG_TABLE = "eligibility"
COHORT_TABLE = "cohort"
EXCLUSIONS_TABLE = "exclusions"

# NA-like tokens observed/expected in the CSV; comparison is case-insensitive after trimming.
NA_TOKENS = ("NA", "N/A")


@dataclass(frozen=True)
class RunMetadata:
    run_utc: str
    input_path: str
    input_sha256: str
    duckdb_path: str
    cohort_definition_version: str
    na_tokens: Tuple[str, ...]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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
    required = ["PatientID", "Stage.Overall"]
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


def build_tables(con: duckdb.DuckDBPyConnection) -> None:
    """
    Creates ELIG_TABLE, COHORT_TABLE, EXCLUSIONS_TABLE from RAW_TABLE.
    """
    na_list_sql = ", ".join([f"'{t}'" for t in NA_TOKENS])

    con.execute(f"DROP TABLE IF EXISTS {ELIG_TABLE};")
    con.execute(
        f"""
        CREATE TABLE {ELIG_TABLE} AS
        SELECT
          r.*,
          trim(coalesce("Stage.Overall", '')) AS stage_overall_trimmed,
          upper(trim(coalesce("Stage.Overall", ''))) AS stage_overall_token,
          (upper(trim(coalesce("Stage.Overall", ''))) IN ({na_list_sql})) AS flag_stage_overall_is_na,
          NOT (upper(trim(coalesce("Stage.Overall", ''))) IN ({na_list_sql})) AS is_eligible,
          CASE
            WHEN (upper(trim(coalesce("Stage.Overall", ''))) IN ({na_list_sql})) THEN 'stage_overall_is_na'
            ELSE NULL
          END AS exclusion_reason
        FROM {RAW_TABLE} r;
        """
    )

    con.execute(f"DROP TABLE IF EXISTS {COHORT_TABLE};")
    con.execute(f"CREATE TABLE {COHORT_TABLE} AS SELECT * FROM {ELIG_TABLE} WHERE is_eligible = TRUE;")

    con.execute(f"DROP TABLE IF EXISTS {EXCLUSIONS_TABLE};")
    con.execute(f"CREATE TABLE {EXCLUSIONS_TABLE} AS SELECT * FROM {ELIG_TABLE} WHERE is_eligible = FALSE;")


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


def compute_summary(con: duckdb.DuckDBPyConnection, run_meta: RunMetadata, raw_csv_meta: Dict) -> Dict:
    total_rows = con.execute(f"SELECT COUNT(*) FROM {RAW_TABLE};").fetchone()[0]
    total_patients = con.execute(f"SELECT COUNT(DISTINCT PatientID) FROM {RAW_TABLE};").fetchone()[0]
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

    return {
        "run_metadata": {
            "run_utc": run_meta.run_utc,
            "input_path": run_meta.input_path,
            "input_sha256": run_meta.input_sha256,
            "duckdb_path": run_meta.duckdb_path,
            "cohort_definition_version": run_meta.cohort_definition_version,
            "cohort_definition": {
                "excluded_if": f"upper(trim(Stage.Overall)) IN {list(run_meta.na_tokens)}",
                "na_tokens": list(run_meta.na_tokens),
                "exclusion_reason_codes": ["stage_overall_is_na"],
            },
        },
        "raw_csv_characteristics": raw_csv_meta,
        "counts": {
            "rows_in_raw_table": int(total_rows),
            "unique_patients_in_raw_table": int(total_patients),
            "eligible": int(eligible),
            "excluded": int(excluded),
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
    write_csv: bool = False,
) -> Dict:
    """
    Notebook-friendly entrypoint.

    Runs the cohort build with explicit parameters (no CLI/argparse required).

    Returns the cohort summary dict that is also written to `cohort_summary.json`.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found at: {input_path}")

    df, raw_csv_meta = read_raw_csv_as_strings(input_path)
    validate_manifest(df)

    output_dir.mkdir(parents=True, exist_ok=True)

    run_meta = RunMetadata(
        run_utc=utc_now_iso(),
        input_path=str(input_path),
        input_sha256=sha256_file(input_path),
        duckdb_path=str(duckdb_path),
        cohort_definition_version="v4-duckdb",
        na_tokens=NA_TOKENS,
    )

    con = duckdb.connect(str(duckdb_path))

    con.register("raw_df", df)
    con.execute(f"DROP TABLE IF EXISTS {RAW_TABLE};")
    con.execute(f"CREATE TABLE {RAW_TABLE} AS SELECT * FROM raw_df;")

    build_tables(con)

    export_table_parquet(con, ELIG_TABLE, output_dir / "eligibility_table.parquet")
    export_table_parquet(con, COHORT_TABLE, output_dir / "cohort.parquet")
    export_table_parquet(con, EXCLUSIONS_TABLE, output_dir / "exclusions.parquet")

    if write_csv:
        export_table_csv(con, ELIG_TABLE, output_dir / "eligibility_table.csv")
        export_table_csv(con, COHORT_TABLE, output_dir / "cohort.csv")
        export_table_csv(con, EXCLUSIONS_TABLE, output_dir / "exclusions.csv")

    summary = compute_summary(con, run_meta, raw_csv_meta)
    (output_dir / "cohort_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    con.close()
    return summary


if __name__ == "__main__":
    summary = build_cohort()
    print(f"Done. Outputs written to: {DEFAULT_OUTPUT_DIR.resolve()}")