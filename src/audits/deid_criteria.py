"""
Local de-identification criteria definitions.

This module centralizes the logic and versioning for how the pipeline decides
whether an instance is fully de-identified. The writer should import and use
these helpers to ensure the JSONL audit records include both the criteria name
and version alongside the computed boolean flag.
"""

from typing import Optional

# Bump this when changing the decision logic.
DEID_CRITERIA_VERSION = "1.0.0"

# Human-readable identifier for the rules applied.
DEID_CRITERIA_NAME = "rem_cln_str_all_true_v1"


def evaluate_deid(
    *,
    tags_removed_eq_max_rem_srch_matches: bool,
    tags_cleaned_eq_max_cln_srch_matches: bool,
    tags_stripped_eq_max_str_srch_matches: Optional[bool],
) -> bool:
    """
    Decide whether an instance meets the de-id criteria.

    The policy requires all three completeness flags to be True. If
    tags_stripped_eq_max_str_srch_matches is None (meaning private tags were
    absent to begin with), we treat that as False for the purposes of the
    decision so that the caller must explicitly set it to True when stripping
    happened and private tags are now zero.
    """

    # Only count as de-identified when every check passed.
    return (
        tags_removed_eq_max_rem_srch_matches
        and tags_cleaned_eq_max_cln_srch_matches
        and (tags_stripped_eq_max_str_srch_matches is True)
    )
