"""Default titles + trace-id generator for the unified error hierarchy.

``_DEFAULT_TITLES`` maps every :class:`ErrorCode` member to a short
human-readable title. The title SHOULD NOT change between occurrences
(RFC 9457 §3) so dashboards can group on it. ``default_title_for(code)``
falls back to the enum value itself when no entry is registered --
strictly worse for UX but never lies.

Phase A1 seeded entries for the codes introduced by the unified
:class:`RyotenkAIError` hierarchy. Phase B (sharded-stargazing-
wigderson, 2026-05-16) merged the pod-runner's parallel
``_DEFAULT_TITLES`` map into this one so there is a **single source
of truth** for code -> title mapping across pod and control APIs.
The pod-runner's old map (``packages/pod/.../runner/api/errors.py``)
is now a re-export shim that reads from here.

The sentinel ``tests/_lint/test_error_code_pinned.py`` enforces that
every :class:`ErrorCode` enum member has a real title here (no enum-
value fallback) -- adding a new code without a title is a CI bug.
"""

from __future__ import annotations

import secrets

from ryotenkai_shared.contracts.problem_details import ErrorCode

# Single source of truth for code -> title. Every ErrorCode member MUST
# appear here; the sentinel ``test_error_code_pinned.py`` blocks
# additions that omit a title.
_DEFAULT_TITLES: dict[ErrorCode, str] = {
    # ----- jobs (pod-runner) -------------------------------------------
    ErrorCode.JOB_NOT_FOUND: "Job not found",
    ErrorCode.JOB_STATE_INVALID: "Invalid job state transition",
    ErrorCode.JOB_SPEC_INVALID: "Job specification invalid",
    ErrorCode.JOB_IN_PROGRESS: "Job already in progress",
    ErrorCode.PLUGIN_UNPACK_FAILED: "Plugin payload unpack failed",
    ErrorCode.SPAWN_FAILED: "Trainer spawn failed",
    # ----- system / lifecycle (pod-runner) -----------------------------
    ErrorCode.RUNNER_NOT_READY: "Runner not ready",
    ErrorCode.RUNNER_BUSY: "Runner busy",
    # ----- diagnostics (pod-runner) ------------------------------------
    ErrorCode.DIAGNOSTIC_FAILED: "Diagnostic collection failed",
    ErrorCode.DIAGNOSTIC_TIMEOUT: "Diagnostic collection timed out",
    ErrorCode.DIAGNOSTIC_INVALID_INCLUDE: "Invalid diagnostics include parameter",
    ErrorCode.DIAGNOSTIC_PERMISSION_DENIED: "Diagnostic permission denied",
    # ----- resources (pod-runner) --------------------------------------
    ErrorCode.RESOURCES_UNAVAILABLE: "Resource snapshot unavailable",
    # ----- logs (pod-runner) -------------------------------------------
    ErrorCode.LOG_NAME_INVALID: "Invalid log name",
    ErrorCode.LOG_NOT_AVAILABLE: "Log file not available",
    ErrorCode.LOG_OFFSET_OUT_OF_RANGE: "Log offset out of range",
    # ----- files (pod-runner) ------------------------------------------
    ErrorCode.FILE_TARGET_INVALID: "Invalid upload target",
    ErrorCode.FILE_TOO_LARGE: "Upload exceeds maximum file size",
    ErrorCode.FILE_WRITE_FAILED: "File write failed",
    ErrorCode.FILE_HASH_MISMATCH: "File checksum mismatch",
    # ----- runtime (pod-runner) ----------------------------------------
    ErrorCode.IMPORT_CHECK_TIMEOUT: "Import check timed out",
    ErrorCode.IMPORT_CHECK_TOO_MANY_MODULES: "Too many modules in import check",
    ErrorCode.IMPORT_CHECK_INVALID_MODULE_NAME: "Invalid module name",
    # ----- internal trainer<->runner enforcement (pod-runner) ----------
    ErrorCode.LOOPBACK_REQUIRED: "Loopback access required",
    ErrorCode.NO_ACTIVE_JOB: "No active job",
    ErrorCode.STOP_NOT_ALLOWED: "Stop request not allowed in current state",
    # ----- config (Phase A1) -------------------------------------------
    ErrorCode.CONFIG_INVALID: "Configuration invalid",
    ErrorCode.CONFIG_DRIFT: "Configuration drift detected",
    ErrorCode.CONFIG_FILE_NOT_FOUND: "Configuration file not found",
    # ----- workspace (Phase A1) ----------------------------------------
    ErrorCode.PROJECT_NOT_FOUND: "Project not found",
    ErrorCode.PROJECT_ALREADY_EXISTS: "Project already exists",
    ErrorCode.PROVIDER_NOT_FOUND: "Provider not found",
    ErrorCode.INTEGRATION_NOT_FOUND: "Integration not found",
    ErrorCode.WORKSPACE_STORE_FAILED: "Workspace store operation failed",
    # ----- pipeline state / launch (Phase A1) --------------------------
    ErrorCode.STATE_LOAD_FAILED: "Pipeline state load failed",
    ErrorCode.STATE_LOCKED: "Pipeline state locked",
    ErrorCode.LAUNCH_IN_PROGRESS: "Launch already in progress",
    ErrorCode.LAUNCH_PREPARATION_FAILED: "Launch preparation failed",
    ErrorCode.PIPELINE_STAGE_FAILED: "Pipeline stage failed",
    ErrorCode.RUN_IS_ACTIVE: "Run is currently active",
    # ----- training (Phase A1) -----------------------------------------
    ErrorCode.TRAINING_FAILED: "Training failed",
    ErrorCode.TRAINING_OOM: "Training ran out of memory",
    ErrorCode.STRATEGY_CHAIN_INVALID: "Strategy chain validation failed",
    # ----- dataset / model / inference (Phase A1) ----------------------
    ErrorCode.DATASET_LOAD_FAILED: "Dataset load failed",
    ErrorCode.DATASET_VALIDATION_FAILED: "Dataset validation failed",
    ErrorCode.MODEL_LOAD_FAILED: "Model load failed",
    ErrorCode.INFERENCE_UNAVAILABLE: "Inference service unavailable",
    # ----- providers (Phase A1) ----------------------------------------
    ErrorCode.PROVIDER_UNAVAILABLE: "Provider unavailable",
    ErrorCode.PROVIDER_RATE_LIMITED: "Provider rate-limited",
    ErrorCode.PROVIDER_AUTH_FAILED: "Provider authentication failed",
    # ----- SSH (Phase A1) ----------------------------------------------
    ErrorCode.SSH_CONNECTION_FAILED: "SSH connection failed",
    ErrorCode.SSH_EXEC_FAILED: "SSH command execution failed",
    ErrorCode.SSH_TRANSFER_FAILED: "SSH file transfer failed",
    # ----- HF Hub (Phase A1) -------------------------------------------
    ErrorCode.HF_AUTH_FAILED: "Hugging Face authentication failed",
    ErrorCode.HF_NOT_FOUND: "Hugging Face resource not found",
    # ----- engines (Phase A1) ------------------------------------------
    ErrorCode.ENGINE_NOT_REGISTERED: "Engine not registered",
    ErrorCode.ENGINE_CONFIG_INVALID: "Engine configuration invalid",
    # ----- post-APIError migration -------------------------------------
    ErrorCode.PERMISSION_DENIED: "Permission denied",
    ErrorCode.RUN_NOT_FOUND: "Run not found",
    ErrorCode.NO_ATTEMPTS: "No attempts recorded",
    ErrorCode.ATTEMPT_NOT_FOUND: "Attempt not found",
    ErrorCode.ATTEMPT_INVALID: "Attempt invalid",
    ErrorCode.JOB_SUBMISSION_MISSING: "Job submission record missing",
    ErrorCode.RUNNER_UNREACHABLE: "Runner unreachable",
    ErrorCode.PROJECT_DIRECTORY_MISSING: "Project directory missing",
    ErrorCode.DATASET_NOT_FOUND: "Dataset not found",
    ErrorCode.PLUGIN_NOT_FOUND: "Plugin not found",
    ErrorCode.PRESET_NOT_FOUND: "Preset not found",
    ErrorCode.INTEGRATION_TYPE_INVALID: "Integration type invalid",
    ErrorCode.LOG_FILE_RANGE_INVALID: "Log file range invalid",
    ErrorCode.REPORT_GENERATION_FAILED: "Report generation failed",
    ErrorCode.HF_LOAD_FAILED: "Hugging Face dataset load failed",
    ErrorCode.DATASETS_LIBRARY_MISSING: "Datasets library missing",
    ErrorCode.PROVIDER_TOKEN_INVALID: "Provider token invalid",
    ErrorCode.CLIENT_DISCONNECT: "Client disconnected mid-upload",
    ErrorCode.TRAINING_TIMEOUT: "Training wall-clock timeout exceeded",
    ErrorCode.METRICS_BUFFER_OVERSIZE: "Metrics buffer exceeds maximum size",
    # ----- transport (Mac client synthesised) --------------------------
    ErrorCode.TRANSPORT_UNREACHABLE: "Transport unreachable",
    # ----- catch-all ---------------------------------------------------
    ErrorCode.INTERNAL_ERROR: "Internal Server Error",
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
    """Short opaque correlation id (16 hex chars, 64 bits).

    Mirrors the pod runner's ``_new_trace_id``. Phase B unifies both
    sources to this single implementation. 64 bits is safe for
    billions of events (birthday paradox at ~4 billion samples) --
    suitable for long-running pipelines.
    """
    return secrets.token_hex(8)
