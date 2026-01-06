-- Cohort cleaning step (clinical app)
--
-- Creates a standardized "cleaned" table used as the input for eligibility rules.
-- Intended to run after the raw CSV is loaded into "patient_manifest_raw",
-- and before eligibility.sql is executed.

DROP TABLE IF EXISTS cleaned;

CREATE TABLE cleaned AS
SELECT
  "PatientID",
  "Overall.Stage"
FROM patient_manifest_raw;
