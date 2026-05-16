"""4xx domain errors -- caller-fixable. See :mod:`.base` for shape.

Each concrete subclass pins ``code`` and ``status`` as ClassVars so the
boundary protocol (RFC 9457 problem+json) and the sentinel
``test_error_code_pinned.py`` can rely on class-level introspection.

Status-code policy (per plan Layer 2 table):

* 400 -- request body/payload semantically invalid (general).
* 401 -- authentication failed.
* 404 -- resource not found.
* 409 -- conflict / state-incompatible operation.
* 422 -- validation failed (Pydantic-style or domain-specific).
"""

from __future__ import annotations

from typing import ClassVar

from ryotenkai_shared.contracts.problem_details import ErrorCode
from ryotenkai_shared.errors.base import DomainError


# ---------------------------------------------------------------------------
# Config (CONFIG_*)
# ---------------------------------------------------------------------------


class ConfigInvalidError(DomainError):
    """Pipeline config failed schema or semantic validation."""

    code: ClassVar[ErrorCode] = ErrorCode.CONFIG_INVALID
    status: ClassVar[int] = 400


class ConfigDriftError(DomainError):
    """Config differs from the version saved for the active run."""

    code: ClassVar[ErrorCode] = ErrorCode.CONFIG_DRIFT
    status: ClassVar[int] = 409


class ConfigFileNotFoundError(DomainError):
    """Config file path does not exist on disk."""

    code: ClassVar[ErrorCode] = ErrorCode.CONFIG_FILE_NOT_FOUND
    status: ClassVar[int] = 404


# ---------------------------------------------------------------------------
# Jobs (JOB_*)
# ---------------------------------------------------------------------------


class JobSpecInvalidError(DomainError):
    """Job specification failed validation."""

    code: ClassVar[ErrorCode] = ErrorCode.JOB_SPEC_INVALID
    status: ClassVar[int] = 422


class JobStateInvalidError(DomainError):
    """Job is not in a state that admits the requested transition."""

    code: ClassVar[ErrorCode] = ErrorCode.JOB_STATE_INVALID
    status: ClassVar[int] = 409


# ---------------------------------------------------------------------------
# Workspace (PROJECT_*, PROVIDER_*, INTEGRATION_*)
# ---------------------------------------------------------------------------


class ProjectNotFoundError(DomainError):
    """Project with the requested id does not exist."""

    code: ClassVar[ErrorCode] = ErrorCode.PROJECT_NOT_FOUND
    status: ClassVar[int] = 404


class ProjectAlreadyExistsError(DomainError):
    """Project id collides with an existing entry."""

    code: ClassVar[ErrorCode] = ErrorCode.PROJECT_ALREADY_EXISTS
    status: ClassVar[int] = 409


class ProviderNotFoundError(DomainError):
    """Provider with the requested id does not exist."""

    code: ClassVar[ErrorCode] = ErrorCode.PROVIDER_NOT_FOUND
    status: ClassVar[int] = 404


class IntegrationNotFoundError(DomainError):
    """Integration with the requested id does not exist."""

    code: ClassVar[ErrorCode] = ErrorCode.INTEGRATION_NOT_FOUND
    status: ClassVar[int] = 404


# ---------------------------------------------------------------------------
# Pipeline state (STATE_*, LAUNCH_*, RUN_*)
# ---------------------------------------------------------------------------


class StateLoadFailedError(DomainError):
    """Pipeline state file missing or unreadable for the given run id."""

    code: ClassVar[ErrorCode] = ErrorCode.STATE_LOAD_FAILED
    status: ClassVar[int] = 404


class StateLockedError(DomainError):
    """Pipeline state is locked by another concurrent operation."""

    code: ClassVar[ErrorCode] = ErrorCode.STATE_LOCKED
    status: ClassVar[int] = 409


class LaunchInProgressError(DomainError):
    """A launch is already running for this pipeline / run."""

    code: ClassVar[ErrorCode] = ErrorCode.LAUNCH_IN_PROGRESS
    status: ClassVar[int] = 409


class RunIsActiveError(DomainError):
    """Run is currently active and cannot be deleted / modified."""

    code: ClassVar[ErrorCode] = ErrorCode.RUN_IS_ACTIVE
    status: ClassVar[int] = 409


# ---------------------------------------------------------------------------
# Hugging Face Hub (HF_*)
# ---------------------------------------------------------------------------


class HFAuthFailedError(DomainError):
    """Hugging Face Hub authentication rejected (bad token, expired)."""

    code: ClassVar[ErrorCode] = ErrorCode.HF_AUTH_FAILED
    status: ClassVar[int] = 401


class HFNotFoundError(DomainError):
    """Hugging Face Hub resource (repo, file, revision) not found."""

    code: ClassVar[ErrorCode] = ErrorCode.HF_NOT_FOUND
    status: ClassVar[int] = 404


# ---------------------------------------------------------------------------
# Providers (PROVIDER_AUTH_FAILED is domain; transient/unavailable is infra)
# ---------------------------------------------------------------------------


class ProviderAuthFailedError(DomainError):
    """Provider rejected credentials (bad API key, expired token)."""

    code: ClassVar[ErrorCode] = ErrorCode.PROVIDER_AUTH_FAILED
    status: ClassVar[int] = 401


# ---------------------------------------------------------------------------
# Dataset (DATASET_*)
# ---------------------------------------------------------------------------


class DatasetLoadFailedError(DomainError):
    """Dataset payload could not be loaded (bad path, format, schema)."""

    code: ClassVar[ErrorCode] = ErrorCode.DATASET_LOAD_FAILED
    status: ClassVar[int] = 422


class DatasetValidationFailedError(DomainError):
    """Dataset content failed semantic validation."""

    code: ClassVar[ErrorCode] = ErrorCode.DATASET_VALIDATION_FAILED
    status: ClassVar[int] = 422


# ---------------------------------------------------------------------------
# Training / strategy (STRATEGY_*)
# ---------------------------------------------------------------------------


class StrategyChainInvalidError(DomainError):
    """Strategy chain transitions or composition are semantically invalid."""

    code: ClassVar[ErrorCode] = ErrorCode.STRATEGY_CHAIN_INVALID
    status: ClassVar[int] = 422


# ---------------------------------------------------------------------------
# Engines (ENGINE_*)
# ---------------------------------------------------------------------------


class EngineNotRegisteredError(DomainError):
    """Requested engine id is not registered in the engine registry."""

    code: ClassVar[ErrorCode] = ErrorCode.ENGINE_NOT_REGISTERED
    status: ClassVar[int] = 404


class EngineConfigInvalidError(DomainError):
    """Engine configuration failed schema or semantic validation."""

    code: ClassVar[ErrorCode] = ErrorCode.ENGINE_CONFIG_INVALID
    status: ClassVar[int] = 422


# ---------------------------------------------------------------------------
# Pod runner endpoints (formerly raised via APIError)
# ---------------------------------------------------------------------------


class JobNotFoundError(DomainError):
    """Active job_id mismatch or no active FSM job for the requested id."""

    code: ClassVar[ErrorCode] = ErrorCode.JOB_NOT_FOUND
    status: ClassVar[int] = 404


class JobInProgressError(DomainError):
    """Submit refused — another (non-terminal) job already runs."""

    code: ClassVar[ErrorCode] = ErrorCode.JOB_IN_PROGRESS
    status: ClassVar[int] = 409


class StopNotAllowedError(DomainError):
    """``POST /jobs/{id}/stop`` called from a non-running state."""

    code: ClassVar[ErrorCode] = ErrorCode.STOP_NOT_ALLOWED
    status: ClassVar[int] = 409


class NoActiveJobError(DomainError):
    """Trainer pushed loopback event but FSM has no active job."""

    code: ClassVar[ErrorCode] = ErrorCode.NO_ACTIVE_JOB
    status: ClassVar[int] = 409


class LoopbackRequiredError(DomainError):
    """Internal endpoint reached from a non-loopback peer."""

    code: ClassVar[ErrorCode] = ErrorCode.LOOPBACK_REQUIRED
    status: ClassVar[int] = 403


class PermissionDeniedError(DomainError):
    """Generic 403 — request rejected by authn/authz layer."""

    code: ClassVar[ErrorCode] = ErrorCode.PERMISSION_DENIED
    status: ClassVar[int] = 403


class PluginUnpackFailedError(DomainError):
    """Multipart ``plugins_payload`` could not be unpacked."""

    code: ClassVar[ErrorCode] = ErrorCode.PLUGIN_UNPACK_FAILED
    status: ClassVar[int] = 422


class FileTargetInvalidError(DomainError):
    """Upload target enum / sanitised filename is invalid."""

    code: ClassVar[ErrorCode] = ErrorCode.FILE_TARGET_INVALID
    status: ClassVar[int] = 422


class FileTooLargeError(DomainError):
    """Upload exceeded ``MAX_FILE_SIZE``."""

    code: ClassVar[ErrorCode] = ErrorCode.FILE_TOO_LARGE
    status: ClassVar[int] = 413


class LogNameInvalidError(DomainError):
    """``LogName`` enum value not mapped to a path (programming bug)."""

    code: ClassVar[ErrorCode] = ErrorCode.LOG_NAME_INVALID
    status: ClassVar[int] = 422


class LogNotAvailableError(DomainError):
    """Requested log file does not exist on disk yet."""

    code: ClassVar[ErrorCode] = ErrorCode.LOG_NOT_AVAILABLE
    status: ClassVar[int] = 404


class LogOffsetOutOfRangeError(DomainError):
    """``offset > total_size`` for a log read; client must reset cursor."""

    code: ClassVar[ErrorCode] = ErrorCode.LOG_OFFSET_OUT_OF_RANGE
    status: ClassVar[int] = 416


class DiagnosticInvalidIncludeError(DomainError):
    """``?include=`` query parameter contains an unknown value."""

    code: ClassVar[ErrorCode] = ErrorCode.DIAGNOSTIC_INVALID_INCLUDE
    status: ClassVar[int] = 422


class ImportCheckTooManyModulesError(DomainError):
    """``POST /runtime/import-check`` exceeded module-count cap."""

    code: ClassVar[ErrorCode] = ErrorCode.IMPORT_CHECK_TOO_MANY_MODULES
    status: ClassVar[int] = 422


class ImportCheckInvalidModuleNameError(DomainError):
    """Module name failed the dotted-identifier regex."""

    code: ClassVar[ErrorCode] = ErrorCode.IMPORT_CHECK_INVALID_MODULE_NAME
    status: ClassVar[int] = 422


# ---------------------------------------------------------------------------
# Control API endpoints (formerly raised via raw HTTPException)
# ---------------------------------------------------------------------------


class RunNotFoundError(DomainError):
    """``run_id`` path component does not resolve to a known run dir."""

    code: ClassVar[ErrorCode] = ErrorCode.RUN_NOT_FOUND
    status: ClassVar[int] = 404


class NoAttemptsError(DomainError):
    """Run directory has no ``attempts/`` subdirectories."""

    code: ClassVar[ErrorCode] = ErrorCode.NO_ATTEMPTS
    status: ClassVar[int] = 404


class AttemptNotFoundError(DomainError):
    """Requested attempt subdirectory does not exist."""

    code: ClassVar[ErrorCode] = ErrorCode.ATTEMPT_NOT_FOUND
    status: ClassVar[int] = 404


class AttemptInvalidError(DomainError):
    """Attempt path / value is structurally invalid."""

    code: ClassVar[ErrorCode] = ErrorCode.ATTEMPT_INVALID
    status: ClassVar[int] = 400


class JobSubmissionMissingError(DomainError):
    """Attempt directory is missing its ``job_submission.json``."""

    code: ClassVar[ErrorCode] = ErrorCode.JOB_SUBMISSION_MISSING
    status: ClassVar[int] = 404


class ProjectDirectoryMissingError(DomainError):
    """Project is registered but its on-disk workspace is gone."""

    code: ClassVar[ErrorCode] = ErrorCode.PROJECT_DIRECTORY_MISSING
    status: ClassVar[int] = 404


class DatasetNotFoundError(DomainError):
    """Pipeline config lacks the requested dataset key."""

    code: ClassVar[ErrorCode] = ErrorCode.DATASET_NOT_FOUND
    status: ClassVar[int] = 404


class PluginNotFoundError(DomainError):
    """Plugin kind/id is not in the community catalog."""

    code: ClassVar[ErrorCode] = ErrorCode.PLUGIN_NOT_FOUND
    status: ClassVar[int] = 404


class PresetNotFoundError(DomainError):
    """``preset_id`` not present in the community catalog."""

    code: ClassVar[ErrorCode] = ErrorCode.PRESET_NOT_FOUND
    status: ClassVar[int] = 404


class IntegrationTypeInvalidError(DomainError):
    """Integration type is structurally unknown / unsupported."""

    code: ClassVar[ErrorCode] = ErrorCode.INTEGRATION_TYPE_INVALID
    status: ClassVar[int] = 400


class LogFileRangeInvalidError(DomainError):
    """Log-file range request rejected (bad offset/size)."""

    code: ClassVar[ErrorCode] = ErrorCode.LOG_FILE_RANGE_INVALID
    status: ClassVar[int] = 400


class ProviderTokenInvalidError(DomainError):
    """Provider token rejected by the provider (canary check)."""

    code: ClassVar[ErrorCode] = ErrorCode.PROVIDER_TOKEN_INVALID
    status: ClassVar[int] = 400


class ClientDisconnectError(DomainError):
    """Multipart upload aborted by the client before completion (499)."""

    code: ClassVar[ErrorCode] = ErrorCode.CLIENT_DISCONNECT
    status: ClassVar[int] = 499


__all__ = [
    "AttemptInvalidError",
    "AttemptNotFoundError",
    "ClientDisconnectError",
    "ConfigDriftError",
    "ConfigFileNotFoundError",
    "ConfigInvalidError",
    "DatasetLoadFailedError",
    "DatasetNotFoundError",
    "DatasetValidationFailedError",
    "DiagnosticInvalidIncludeError",
    "EngineConfigInvalidError",
    "EngineNotRegisteredError",
    "FileTargetInvalidError",
    "FileTooLargeError",
    "HFAuthFailedError",
    "HFNotFoundError",
    "ImportCheckInvalidModuleNameError",
    "ImportCheckTooManyModulesError",
    "IntegrationNotFoundError",
    "IntegrationTypeInvalidError",
    "JobInProgressError",
    "JobNotFoundError",
    "JobSpecInvalidError",
    "JobStateInvalidError",
    "JobSubmissionMissingError",
    "LaunchInProgressError",
    "LogFileRangeInvalidError",
    "LogNameInvalidError",
    "LogNotAvailableError",
    "LogOffsetOutOfRangeError",
    "LoopbackRequiredError",
    "NoActiveJobError",
    "NoAttemptsError",
    "PermissionDeniedError",
    "PluginNotFoundError",
    "PluginUnpackFailedError",
    "PresetNotFoundError",
    "ProjectAlreadyExistsError",
    "ProjectDirectoryMissingError",
    "ProjectNotFoundError",
    "ProviderAuthFailedError",
    "ProviderNotFoundError",
    "ProviderTokenInvalidError",
    "RunIsActiveError",
    "RunNotFoundError",
    "StateLoadFailedError",
    "StateLockedError",
    "StopNotAllowedError",
    "StrategyChainInvalidError",
]
