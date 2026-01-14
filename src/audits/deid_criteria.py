"""
Local de-identification criteria definitions.

This module centralizes the logic and versioning for how the pipeline decides
whether an instance is fully de-identified. The writer should import and use
these helpers to ensure the JSONL audit records include the criteria version
alongside the computed boolean flag.
"""

from typing import Optional, Dict, Any

# Bump this when changing the decision logic.
DEID_CRITERIA_VERSION = "deid_criteria_1.0.0"


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


def evaluate_deid_from_flags(flags: Dict[str, Any]) -> bool:
    """
    Convenience wrapper to evaluate de-id based on a mapping of flags.
    Expects keys matching the three completeness flags; missing keys default
    to False/None, so callers must populate them intentionally.
    """

    return evaluate_deid(
        tags_removed_eq_max_rem_srch_matches=bool(flags.get("tags_removed_eq_max_rem_srch_matches", False)),
        tags_cleaned_eq_max_cln_srch_matches=bool(flags.get("tags_cleaned_eq_max_cln_srch_matches", False)),
        tags_stripped_eq_max_str_srch_matches=flags.get("tags_stripped_eq_max_str_srch_matches"),
    )
