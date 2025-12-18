# Manifest and Metadata Schema

This project treats **CSVs as manifests** and **JSON sidecars as metadata** for clinical-style integration.

## Overview

- **Manifest (CSV)**: patient cohort table (raw, cleaned, or stratified-split).
- **Metadata (JSON)**: processing and provenance information about a specific manifest.
- PHI status is explicitly recorded to support clinical/production environments.

---

## Cleaning Stage

### Cleaned Manifest CSV

Output of `clean_patient_manifest_csv`:
- A cleaned cohort manifest CSV at `clean_path` (e.g. `...cleaned.csv`).

### Clean Manifest Metadata JSON

Written by `write_clean_manifest_metadata` alongside the cleaned CSV as `*.metadata.json`.

Top-level structure:

- `dataset` (object)
  - `datasetId` (string) – e.g. `"NSCLC-Radiomics-Lung1"`.
  - `description` (string) – description of the cohort/operation.
  - `sourceSystem` (string) – e.g. `"TCIA"`.

- `sourceManifest` (object)
  - `rawManifestPath` (string) – absolute path to the original manifest (or `"dataframe in memory"`).
  - `rawRowCount` (integer) – number of rows before cleaning.
  - `sourceType` (string) – `"DataFrame"` or `"CSV"`.

- `cleanManifest` (object)
  - `cleanManifestPath` (string) – absolute path to cleaned manifest CSV.
  - `rowCount` (integer) – number of rows after cleaning.
  - `idColumns` (array of string) – identifier columns (e.g. `["PatientID"]`).
  - `columnSummary` (object) – keys are column names, values:
    - `dtype` (string) – pandas dtype as string.
    - `nonNullCount` (integer) – count of non-null values.
  - `hash` (string) – SHA256 of the cleaned manifest.

- `processing` (object)
  - `pipelineName` (string) – e.g. `"clean_patient_manifest"`.
  - `pipelineVersion` (string) – code or git version.
  - `executedAt` (string, ISO 8601 UTC) – execution timestamp.
  - `randomState` (integer or null) – random seed used, if any.
  - `pythonVersion` (string).
  - `pandasVersion` (string).
  - `cleaningSummary` (object) – structured summary of cleaning:
    - `columns.original` – `{ count, names }`.
    - `columns.kept` – `{ count, names }`.
    - `columns.removed` – `{ count, names }`.
    - `patients.before_drop` – integer.
    - `patients.after_drop` – integer.
    - `patients.dropped_total` – integer.
    - `patients.dropped_by_column` – mapping of column → count.
    - `patients.dropped_identifiers` – list of dropped patient records.
    - `patients.dropped_identifiers_by_column` – mapping of column → records.

- `phiStatus` (object)
  - `containsPhi` (boolean) – whether PHI is present.
  - `phiLevel` (string) – e.g. `"none"`, `"limited"`, `"full"`.
  - `verificationSource` (string) – e.g. TCIA de-identification policy.
  - `notes` (string) – free text.

---

## Stratified Split Stage

### Stratified Split Manifest CSV

Output of `stratify_split_patient_manifest_csv`:
- A cohort manifest CSV at `output_path` with a `Split` column (`"Train"`, `"Test"`).

### Stratified Split Metadata JSON

Written by `write_stratified_split_metadata` alongside the split CSV as `*.metadata.json`.

Top-level structure:

- `dataset` (object)
  - `datasetId` (string).
  - `description` (string).
  - `sourceSystem` (string).

- `sourceManifest` (object)
  - `cleanManifestPath` (string) – absolute path to input cleaned manifest CSV.
  - `rowCount` (integer) – number of rows in the source manifest.
  - `hash` (string) – SHA256 of the source manifest.

- `splitConfig` (object)
  - `strategy` (string) – e.g. `"stratifiedHoldout"`.
  - `targetColumn` (string) – stratification label column.
  - `fractions` (object) – e.g. `{ "train": 0.8, "test": 0.2 }`.
  - `randomSeed` (integer).
  - `shuffle` (boolean).

- `splits` (array of object)
  - `name` (string) – e.g. `"Train"`, `"Test"`.
  - `manifestPath` (string) – path to the split manifest CSV (currently the same `output_path`).
  - `rowCount` (integer) – number of rows in that split.
  - `labelDistribution` (object):
    - `counts` (object) – label → count.
    - `proportions` (object) – label → proportion.

- `processing` (object)
  - `pipelineName` (string) – e.g. `"stratify_split_patient_manifest"`.
  - `pipelineVersion` (string).
  - `executedAt` (string, ISO 8601 UTC).
  - `pythonVersion` (string).
  - `pandasVersion` (string).
  - `cleaningMetadata` (object or null) – metadata returned from the cleaning stage when applied inline.

- `phiStatus` (object)
  - Same structure as in the cleaning metadata JSON.

This document is the reference for downstream tools (ETL, Epic integrations, notebooks) that consume the manifest metadata JSON files.
