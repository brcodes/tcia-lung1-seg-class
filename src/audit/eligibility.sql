-- Cohort eligibility definition (clinical app)
--
-- This file is intended to be the single source of truth for cohort eligibility.
-- It is executed by src/preprocessing/build_cohort.py to create the eligibility/cohort/exclusions tables.
--
-- Placeholders:
--   {{NA_LIST_SQL}}  -> SQL list of quoted tokens, e.g. 'NA', 'N/A'
--   {{AUDIT_ELIG_TABLE}} -> DuckDB table containing DICOM audit eligibility signals

DROP TABLE IF EXISTS eligibility;

CREATE TABLE eligibility AS
WITH stage_flags AS (
  SELECT
    r."PatientID" AS patient_id,
    r."Overall.Stage" AS overall_stage,
    trim(coalesce(r."Overall.Stage", '')) AS stage_overall_trimmed,
    upper(trim(coalesce(r."Overall.Stage", ''))) AS stage_overall_token,
    (upper(trim(coalesce(r."Overall.Stage", ''))) IN ({{NA_LIST_SQL}})) AS flag_stage_overall_is_na
  FROM cleaned r
),
audit AS (
  SELECT
    patient_id,
    nullif(trim(ct_seg_and_linkage), '') AS ct_seg_and_linkage,
    ct_seg_and_linkage_reasons_json,
    ct_seg_linked_files_json,
    nullif(trim(ct_ser_valid_for_downstream_preprocessing), '') AS ct_ser_valid_for_downstream_preprocessing,
    ct_ser_issues_json,
    ct_ser_warnings_json,
    ct_ser_segments_json
  FROM {{AUDIT_ELIG_TABLE}}
),
joined AS (
  SELECT
    s.patient_id,
    s.overall_stage,
    s.stage_overall_trimmed,
    s.stage_overall_token,
    s.flag_stage_overall_is_na,
    (a.patient_id IS NOT NULL) AS audit_has_preprocessing_row,
    a.ct_seg_and_linkage,
    a.ct_seg_and_linkage_reasons_json,
    a.ct_seg_linked_files_json,
    a.ct_ser_valid_for_downstream_preprocessing,
    a.ct_ser_issues_json,
    a.ct_ser_warnings_json,
    a.ct_ser_segments_json,
    CASE
      WHEN a.patient_id IS NULL THEN NULL
      ELSE upper(coalesce(a.ct_seg_and_linkage, '')) = 'TRUE'
    END AS flag_ct_seg_and_linkage_true,
    CASE
      WHEN a.patient_id IS NULL THEN NULL
      ELSE upper(coalesce(a.ct_ser_valid_for_downstream_preprocessing, '')) = 'OK'
    END AS flag_ct_ser_decision_ok,
    CASE
      WHEN a.patient_id IS NULL THEN NULL
      ELSE upper(coalesce(a.ct_ser_valid_for_downstream_preprocessing, '')) = 'WARN'
    END AS flag_ct_ser_decision_warn,
    CASE
      WHEN a.patient_id IS NULL THEN NULL
      ELSE upper(coalesce(a.ct_ser_valid_for_downstream_preprocessing, '')) = 'FAILURE'
    END AS flag_ct_ser_decision_failure,
    CASE
      WHEN a.patient_id IS NULL THEN NULL
      ELSE a.ct_ser_valid_for_downstream_preprocessing IS NULL
    END AS flag_ct_ser_decision_missing
  FROM stage_flags s
  LEFT JOIN audit a
    ON a.patient_id = s.patient_id
)
SELECT
  patient_id AS "PatientID",
  overall_stage AS "Overall.Stage",
  stage_overall_trimmed,
  stage_overall_token,
  flag_stage_overall_is_na,
  audit_has_preprocessing_row,
  ct_seg_and_linkage,
  flag_ct_seg_and_linkage_true,
  ct_seg_and_linkage_reasons_json,
  ct_seg_linked_files_json,
  ct_ser_valid_for_downstream_preprocessing,
  flag_ct_ser_decision_ok,
  flag_ct_ser_decision_warn,
  flag_ct_ser_decision_failure,
  flag_ct_ser_decision_missing,
  ct_ser_issues_json,
  ct_ser_warnings_json,
  ct_ser_segments_json,
  (
    NOT flag_stage_overall_is_na
    AND audit_has_preprocessing_row
    AND coalesce(flag_ct_seg_and_linkage_true, FALSE)
    AND coalesce(flag_ct_ser_decision_ok, FALSE)
  ) AS is_eligible,
  CASE
    WHEN flag_stage_overall_is_na THEN 'stage_overall_is_na'
    WHEN audit_has_preprocessing_row = FALSE THEN 'dicom_audit_missing'
    WHEN coalesce(flag_ct_seg_and_linkage_true, FALSE) = FALSE THEN 'ct_seg_and_linkage_not_true'
    WHEN coalesce(flag_ct_ser_decision_failure, FALSE) = TRUE THEN 'ct_ser_preproc_failure'
    WHEN coalesce(flag_ct_ser_decision_warn, FALSE) = TRUE THEN 'ct_ser_preproc_warn'
    WHEN coalesce(flag_ct_ser_decision_missing, FALSE) = TRUE THEN 'ct_ser_preproc_missing'
    ELSE NULL
  END AS exclusion_reason
FROM joined;


DROP TABLE IF EXISTS cohort;
CREATE TABLE cohort AS
SELECT *
FROM eligibility
WHERE is_eligible = TRUE;

DROP TABLE IF EXISTS exclusions;
CREATE TABLE exclusions AS
SELECT *
FROM eligibility
WHERE is_eligible = FALSE;
