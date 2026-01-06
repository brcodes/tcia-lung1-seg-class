-- Cohort eligibility definition (clinical app)
--
-- This file is intended to be the single source of truth for cohort eligibility.
-- It is executed by src/preprocessing/build_cohort.py to create the eligibility/cohort/exclusions tables.
--
-- Placeholders:
--   {{NA_LIST_SQL}}  -> SQL list of quoted tokens, e.g. 'NA', 'N/A'

DROP TABLE IF EXISTS eligibility;

CREATE TABLE eligibility AS
SELECT
  r."PatientID",
  r."Overall.Stage",
  trim(coalesce("Overall.Stage", '')) AS stage_overall_trimmed,
  upper(trim(coalesce("Overall.Stage", ''))) AS stage_overall_token,
  (upper(trim(coalesce("Overall.Stage", ''))) IN ({{NA_LIST_SQL}})) AS flag_stage_overall_is_na,
  NOT (upper(trim(coalesce("Overall.Stage", ''))) IN ({{NA_LIST_SQL}})) AS is_eligible,
  CASE
    WHEN (upper(trim(coalesce("Overall.Stage", ''))) IN ({{NA_LIST_SQL}})) THEN 'stage_overall_is_na'
    ELSE NULL
  END AS exclusion_reason
FROM cleaned r;


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
