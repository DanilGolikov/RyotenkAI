"""Constants used by the DatasetValidator stage and its plugins.

Kept separate from the stage module to avoid pulling DatasetValidator's
heavy import chain (data loaders, validation registry) when callers only
need the validation-mode literals or stage-context attribute names.
"""

from __future__ import annotations

# Validation-mode literals echoed in pipeline state and CLI flags.
VALIDATION_MODE_FAST = "fast"
VALIDATION_MODE_FULL = "full"

# Validation status strings — pinned to JSON-visible values; do not rename
# without coordinating with the API + frontend status enum.
VALIDATION_STATUS_SKIPPED = "skipped"
VALIDATION_STATUS_FAILED = "failed"
VALIDATION_STATUS_PASSED = "passed"

# Fast-mode sample cap. The CLI exposes this verbatim in --help; treat as
# product-facing.
VALIDATION_MAX_SAMPLES_FAST = 10000

# Stage-context dict keys (read by orchestrator + ValidationArtifactManager).
VALIDATIONS_ATTR = "validations"
VALIDATION_MODE_ATTR = "mode"
CRITICAL_FAILURES_ATTR = "critical_failures"
VALIDATION_STATUS_KEY = "validation_status"
WARNINGS_KEY = "warnings"

# Dataset split labels used by validation plugin filters.
SPLIT_TRAIN = "train"
SPLIT_EVAL = "eval"

# Mock-mode toggle key (test fixtures use it to short-circuit live data
# loaders).
MOCK_MODE_KEY = "mock_mode"


__all__ = [
    "CRITICAL_FAILURES_ATTR",
    "MOCK_MODE_KEY",
    "SPLIT_EVAL",
    "SPLIT_TRAIN",
    "VALIDATIONS_ATTR",
    "VALIDATION_MAX_SAMPLES_FAST",
    "VALIDATION_MODE_ATTR",
    "VALIDATION_MODE_FAST",
    "VALIDATION_MODE_FULL",
    "VALIDATION_STATUS_FAILED",
    "VALIDATION_STATUS_KEY",
    "VALIDATION_STATUS_PASSED",
    "VALIDATION_STATUS_SKIPPED",
    "WARNINGS_KEY",
]
