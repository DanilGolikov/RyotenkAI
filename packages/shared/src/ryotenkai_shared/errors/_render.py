"""Default titles + trace-id generator for the unified error hierarchy.

``_DEFAULT_TITLES`` maps every :class:`ErrorCode` member to a short
human-readable title. The title SHOULD NOT change between occurrences
(RFC 9457 §3) so dashboards can group on it. ``default_title_for(code)``
falls back to the enum value itself when no entry is registered --
strictly worse for UX but never lies.

Phase A1 seeds entries only for the *new* codes added in this phase
(see plan Layer 2). The pod-runner's pre-existing titles live in
``packages/pod/.../runner/api/errors.py::_DEFAULT_TITLES`` and remain
authoritative for pod codes until Phase B promotes them here.
"""

from __future__ import annotations

import secrets

from ryotenkai_shared.contracts.problem_details import ErrorCode

# Phase A1 catalog -- only the codes introduced in this phase are seeded
# here. Pod-runner codes (JOB_NOT_FOUND, JOB_SPEC_INVALID, ...) keep
# their titles in ``packages/pod/.../runner/api/errors.py`` until
# Phase B unifies the registries.
_DEFAULT_TITLES: dict[ErrorCode, str] = {
    # ----- config ------------------------------------------------------
    ErrorCode.CONFIG_INVALID: "Configuration invalid",
    ErrorCode.CONFIG_DRIFT: "Configuration drift detected",
    ErrorCode.CONFIG_FILE_NOT_FOUND: "Configuration file not found",
    # ----- workspace ---------------------------------------------------
    ErrorCode.PROJECT_NOT_FOUND: "Project not found",
    ErrorCode.PROJECT_ALREADY_EXISTS: "Project already exists",
    ErrorCode.PROVIDER_NOT_FOUND: "Provider not found",
    ErrorCode.INTEGRATION_NOT_FOUND: "Integration not found",
    ErrorCode.WORKSPACE_STORE_FAILED: "Workspace store operation failed",
    # ----- pipeline state / launch ------------------------------------
    ErrorCode.STATE_LOAD_FAILED: "Pipeline state load failed",
    ErrorCode.STATE_LOCKED: "Pipeline state locked",
    ErrorCode.LAUNCH_IN_PROGRESS: "Launch already in progress",
    ErrorCode.LAUNCH_PREPARATION_FAILED: "Launch preparation failed",
    ErrorCode.PIPELINE_STAGE_FAILED: "Pipeline stage failed",
    ErrorCode.RUN_IS_ACTIVE: "Run is currently active",
    # ----- training ----------------------------------------------------
    ErrorCode.TRAINING_FAILED: "Training failed",
    ErrorCode.TRAINING_OOM: "Training ran out of memory",
    # ----- dataset / model / inference --------------------------------
    ErrorCode.DATASET_LOAD_FAILED: "Dataset load failed",
    ErrorCode.DATASET_VALIDATION_FAILED: "Dataset validation failed",
    ErrorCode.MODEL_LOAD_FAILED: "Model load failed",
    ErrorCode.INFERENCE_UNAVAILABLE: "Inference service unavailable",
    # ----- providers ---------------------------------------------------
    ErrorCode.PROVIDER_UNAVAILABLE: "Provider unavailable",
    ErrorCode.PROVIDER_RATE_LIMITED: "Provider rate-limited",
    ErrorCode.PROVIDER_AUTH_FAILED: "Provider authentication failed",
    # ----- SSH ---------------------------------------------------------
    ErrorCode.SSH_CONNECTION_FAILED: "SSH connection failed",
    ErrorCode.SSH_EXEC_FAILED: "SSH command execution failed",
    ErrorCode.SSH_TRANSFER_FAILED: "SSH file transfer failed",
    # ----- HF Hub ------------------------------------------------------
    ErrorCode.HF_AUTH_FAILED: "Hugging Face authentication failed",
    ErrorCode.HF_NOT_FOUND: "Hugging Face resource not found",
    # ----- engines -----------------------------------------------------
    ErrorCode.ENGINE_NOT_REGISTERED: "Engine not registered",
    ErrorCode.ENGINE_CONFIG_INVALID: "Engine configuration invalid",
    # ----- pod-inherited (needed by RyotenkAIError construction) ------
    # Phase B will move the full pod map here. For now we add only what
    # our new exception classes need at construction time.
    ErrorCode.TRANSPORT_UNREACHABLE: "Transport unreachable",
    ErrorCode.INTERNAL_ERROR: "Internal server error",
}


def default_title_for(code: ErrorCode) -> str:
    """Return the registered title for ``code`` or the enum value as fallback.

    Fallback is strictly worse for UX but never lies -- the wire still
    carries a sensible string. Production code should always register
    a title in :data:`_DEFAULT_TITLES`; the sentinel
    ``test_error_code_pinned.py`` enforces this for Phase A1 codes.
    """
    return _DEFAULT_TITLES.get(code, code.value)


def new_trace_id() -> str:
    """Short opaque correlation id (8 hex chars).

    Mirrors the pod runner's ``_new_trace_id``. Phase B unifies both
    sources to this single implementation.
    """
    return secrets.token_hex(4)
