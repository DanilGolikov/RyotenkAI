"""RFC 9457 problem+json contract — single source of truth.

Lives in the leaf ``ryotenkai_shared`` package so both pod-side
exception handlers and Mac-side parser consume one model.

References
----------
- `RFC 9457: Problem Details for HTTP APIs
  <https://www.rfc-editor.org/rfc/rfc9457.html>`_
- Project plan:
  ``docs/plans/2026-05-04-transport-unification-http-runtime-v2.md``

Project extensions over §3 of the RFC:
- ``code`` (machine-readable :class:`ErrorCode` value, UPPER_SNAKE_CASE)
- ``trace_id`` (correlates server log entry with the response)
- ``request_id`` (FastAPI middleware-set ID for request tracing)
- ``errors`` (list of :class:`FieldError` for validation failures —
  pattern aligns with FastAPI's existing ``RequestValidationError``
  shape, just typed)
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# Mime type fixed by RFC 9457 §3 ("Content-Type"). Both ends should
# import this constant rather than retyping the string.
PROBLEM_JSON_MEDIA_TYPE = "application/problem+json"


class ErrorCode(StrEnum):
    """Machine-readable error identifiers — UPPER_SNAKE_CASE with
    a domain prefix.

    Adding a new code: pick the right domain section and add it.
    Removing a code: NEVER (clients pin on the value); deprecate via
    docstring instead. UPPER_SNAKE_CASE with domain prefix.

    The full registry mirrors the HTTP→code map documented in
    plan §6 (transport-unification-v2). Adding to this enum without
    updating the plan is a doc bug; CI will eventually gate on it
    via PR-3.4 (OpenAPI freshness).

    Phase A1 (sharded-stargazing-wigderson, 2026-05-14) extends the
    registry with the Mac-side / control-plane error catalog (config,
    pipeline state, training, providers, SSH, HF Hub, engines). Prior
    to A1 the enum only carried pod-runner codes; the new entries are
    consumed by ``ryotenkai_shared.errors`` exception subclasses.
    """

    # ----- jobs --------------------------------------------------------
    JOB_NOT_FOUND = "JOB_NOT_FOUND"
    JOB_STATE_INVALID = "JOB_STATE_INVALID"
    JOB_SPEC_INVALID = "JOB_SPEC_INVALID"
    JOB_IN_PROGRESS = "JOB_IN_PROGRESS"
    PLUGIN_UNPACK_FAILED = "PLUGIN_UNPACK_FAILED"
    SPAWN_FAILED = "SPAWN_FAILED"

    # ----- system / lifecycle ------------------------------------------
    RUNNER_NOT_READY = "RUNNER_NOT_READY"
    RUNNER_BUSY = "RUNNER_BUSY"

    # ----- diagnostics (Phase 2 PR-2.1) --------------------------------
    DIAGNOSTIC_FAILED = "DIAGNOSTIC_FAILED"
    DIAGNOSTIC_TIMEOUT = "DIAGNOSTIC_TIMEOUT"
    DIAGNOSTIC_INVALID_INCLUDE = "DIAGNOSTIC_INVALID_INCLUDE"
    DIAGNOSTIC_PERMISSION_DENIED = "DIAGNOSTIC_PERMISSION_DENIED"

    # ----- resources (Phase 2 PR-2.2) ----------------------------------
    RESOURCES_UNAVAILABLE = "RESOURCES_UNAVAILABLE"

    # ----- logs (Phase 2 PR-2.3) ---------------------------------------
    LOG_NAME_INVALID = "LOG_NAME_INVALID"
    LOG_NOT_AVAILABLE = "LOG_NOT_AVAILABLE"
    LOG_OFFSET_OUT_OF_RANGE = "LOG_OFFSET_OUT_OF_RANGE"

    # ----- files (Phase 2 PR-2.4) --------------------------------------
    FILE_TARGET_INVALID = "FILE_TARGET_INVALID"
    FILE_TOO_LARGE = "FILE_TOO_LARGE"
    FILE_WRITE_FAILED = "FILE_WRITE_FAILED"
    FILE_HASH_MISMATCH = "FILE_HASH_MISMATCH"

    # ----- runtime (Phase 2 PR-2.5) ------------------------------------
    IMPORT_CHECK_TIMEOUT = "IMPORT_CHECK_TIMEOUT"
    IMPORT_CHECK_TOO_MANY_MODULES = "IMPORT_CHECK_TOO_MANY_MODULES"
    IMPORT_CHECK_INVALID_MODULE_NAME = "IMPORT_CHECK_INVALID_MODULE_NAME"

    # ----- transport (Mac-side only — never sent over the wire by the
    # runner; raised by the Mac client when the tunnel itself is the
    # failure mode, e.g. ssh -L died) ----------------------------------
    TRANSPORT_UNREACHABLE = "TRANSPORT_UNREACHABLE"

    # ----- internal trainer↔runner enforcement (loopback) --------------
    # Surfaces only over 127.0.0.1; never seen by the Mac client. Kept
    # in the registry so the universal HTTPException adapter has a
    # specific code to map legacy raise sites onto, instead of the
    # noisy INTERNAL_ERROR fallback.
    LOOPBACK_REQUIRED = "LOOPBACK_REQUIRED"
    NO_ACTIVE_JOB = "NO_ACTIVE_JOB"

    # ----- job FSM -----------------------------------------------------
    # Specific 409 for "stop requested while not in a stoppable state".
    # Kept distinct from JOB_STATE_INVALID (which covers any FSM
    # transition denial) because the postmortem in monitor distinguishes
    # them — Phase 3 PR-3.3 will collapse if they prove redundant.
    STOP_NOT_ALLOWED = "STOP_NOT_ALLOWED"

    # ----- config (Phase A1) -------------------------------------------
    CONFIG_INVALID = "CONFIG_INVALID"
    CONFIG_DRIFT = "CONFIG_DRIFT"
    CONFIG_FILE_NOT_FOUND = "CONFIG_FILE_NOT_FOUND"

    # ----- workspace (Phase A1) ----------------------------------------
    PROJECT_NOT_FOUND = "PROJECT_NOT_FOUND"
    PROJECT_ALREADY_EXISTS = "PROJECT_ALREADY_EXISTS"
    PROVIDER_NOT_FOUND = "PROVIDER_NOT_FOUND"
    INTEGRATION_NOT_FOUND = "INTEGRATION_NOT_FOUND"
    WORKSPACE_STORE_FAILED = "WORKSPACE_STORE_FAILED"

    # ----- pipeline state / launch (Phase A1) --------------------------
    STATE_LOAD_FAILED = "STATE_LOAD_FAILED"
    STATE_LOCKED = "STATE_LOCKED"
    LAUNCH_IN_PROGRESS = "LAUNCH_IN_PROGRESS"
    LAUNCH_PREPARATION_FAILED = "LAUNCH_PREPARATION_FAILED"
    PIPELINE_STAGE_FAILED = "PIPELINE_STAGE_FAILED"
    RUN_IS_ACTIVE = "RUN_IS_ACTIVE"

    # ----- training (Phase A1) -----------------------------------------
    TRAINING_FAILED = "TRAINING_FAILED"
    TRAINING_OOM = "TRAINING_OOM"
    STRATEGY_CHAIN_INVALID = "STRATEGY_CHAIN_INVALID"

    # ----- dataset / model / inference (Phase A1) ----------------------
    DATASET_LOAD_FAILED = "DATASET_LOAD_FAILED"
    DATASET_VALIDATION_FAILED = "DATASET_VALIDATION_FAILED"
    MODEL_LOAD_FAILED = "MODEL_LOAD_FAILED"
    INFERENCE_UNAVAILABLE = "INFERENCE_UNAVAILABLE"

    # ----- providers (Phase A1) ----------------------------------------
    PROVIDER_UNAVAILABLE = "PROVIDER_UNAVAILABLE"
    PROVIDER_RATE_LIMITED = "PROVIDER_RATE_LIMITED"
    PROVIDER_AUTH_FAILED = "PROVIDER_AUTH_FAILED"

    # ----- SSH (Phase A1) ----------------------------------------------
    SSH_CONNECTION_FAILED = "SSH_CONNECTION_FAILED"
    SSH_EXEC_FAILED = "SSH_EXEC_FAILED"
    SSH_TRANSFER_FAILED = "SSH_TRANSFER_FAILED"

    # ----- HF Hub (Phase A1) -------------------------------------------
    HF_AUTH_FAILED = "HF_AUTH_FAILED"
    HF_NOT_FOUND = "HF_NOT_FOUND"

    # ----- engines (Phase A1) ------------------------------------------
    ENGINE_NOT_REGISTERED = "ENGINE_NOT_REGISTERED"
    ENGINE_CONFIG_INVALID = "ENGINE_CONFIG_INVALID"

    # ----- catch-all (server-side bug) ---------------------------------
    INTERNAL_ERROR = "INTERNAL_ERROR"


class FieldError(BaseModel):
    """Field-level validation error.

    Used inside :attr:`ProblemDetails.errors` when ``code ==
    JOB_SPEC_INVALID`` (or any other 422 with field-level detail).
    Aligns with FastAPI's ``RequestValidationError`` shape, just
    typed:

        ``{"loc": ["body", "command"], "type": "missing",
           "msg": "Field required"}``
    """

    model_config = ConfigDict(extra="forbid")

    loc: list[str | int] = Field(
        description="Location of the offending field (mirrors FastAPI).",
    )
    type: str = Field(description="Pydantic validator name (``missing``, ``int_parsing``, …).")
    msg: str = Field(description="Human-readable diagnostic.")
    input: Any = Field(
        default=None,
        description="Offending input value (omitted via null-strip if absent).",
    )


class ProblemDetails(BaseModel):
    """RFC 9457 §3 base + project extensions.

    Serialisation rule (RFC 9457 §3.1 recommendation): null fields
    are stripped on the wire. Pydantic does not strip by default;
    callers serialise via ``model_dump(exclude_none=True)``.
    """

    model_config = ConfigDict(extra="forbid")

    # ----- RFC 9457 §3 fields ------------------------------------------
    type: str = Field(
        default="about:blank",
        description=(
            "URI reference identifying the problem type. Defaults "
            "to ``about:blank`` per RFC 9457 §3 when no specific "
            "type is appropriate (the HTTP status is then "
            "self-explanatory)."
        ),
    )
    title: str = Field(
        description=(
            "Short, human-readable summary. SHOULD NOT change between "
            "occurrences (RFC 9457 §3) so dashboards can group on it."
        ),
    )
    status: int = Field(
        ge=100,
        le=599,
        description="HTTP status code mirrored in the body for clients that lose it.",
    )
    detail: str | None = Field(
        default=None,
        description="Human-readable explanation specific to this occurrence.",
    )
    instance: str | None = Field(
        default=None,
        description="URI reference identifying this specific occurrence (typically request path).",
    )

    # ----- project extensions ------------------------------------------
    code: ErrorCode = Field(
        description=(
            "Machine-readable error identifier. Clients switch on this "
            "rather than ``status`` because one status maps to several "
            "domain semantics (e.g. 422 → JOB_SPEC_INVALID vs "
            "PLUGIN_UNPACK_FAILED)."
        ),
    )
    trace_id: str | None = Field(
        default=None,
        description="Correlation ID linking this response to the server log entry.",
    )
    request_id: str | None = Field(
        default=None,
        description="Per-request ID (set by middleware) for request tracing.",
    )
    errors: list[FieldError] | None = Field(
        default=None,
        description=(
            "Field-level errors when ``code`` indicates a validation "
            "failure (typically ``JOB_SPEC_INVALID``)."
        ),
    )


__all__ = [
    "PROBLEM_JSON_MEDIA_TYPE",
    "ErrorCode",
    "FieldError",
    "ProblemDetails",
]
