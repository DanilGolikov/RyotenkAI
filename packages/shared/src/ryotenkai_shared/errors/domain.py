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


__all__ = [
    "ConfigDriftError",
    "ConfigFileNotFoundError",
    "ConfigInvalidError",
    "DatasetLoadFailedError",
    "DatasetValidationFailedError",
    "EngineConfigInvalidError",
    "EngineNotRegisteredError",
    "HFAuthFailedError",
    "HFNotFoundError",
    "IntegrationNotFoundError",
    "JobSpecInvalidError",
    "JobStateInvalidError",
    "LaunchInProgressError",
    "ProjectAlreadyExistsError",
    "ProjectNotFoundError",
    "ProviderAuthFailedError",
    "ProviderNotFoundError",
    "RunIsActiveError",
    "StateLoadFailedError",
    "StateLockedError",
]
