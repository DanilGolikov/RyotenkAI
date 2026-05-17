"""5xx infrastructure errors -- external/transient/bug. See :mod:`.base`.

Each concrete subclass pins ``code`` and ``status`` as ClassVars so the
boundary protocol (RFC 9457 problem+json) and the sentinel
``test_error_code_pinned.py`` can rely on class-level introspection.

Status-code policy (per plan Layer 2 table):

* 429 -- upstream rate-limited (still ``InfrastructureError`` flavour --
  client retries with backoff; not the *user's* fault).
* 500 -- internal error / unexpected failure.
* 502 -- bad gateway (SSH tunnel, upstream proxy failed).
* 503 -- upstream service unavailable / transient outage.
* 599 -- TransportError (Mac-side synthesised, see :mod:`.base`).
"""

from __future__ import annotations

from typing import ClassVar

from ryotenkai_shared.contracts.problem_details import ErrorCode
from ryotenkai_shared.errors.base import InfrastructureError


# ---------------------------------------------------------------------------
# Providers (PROVIDER_*)
# ---------------------------------------------------------------------------


class ProviderUnavailableError(InfrastructureError):
    """Provider API returned a transient/5xx error; retry recommended."""

    code: ClassVar[ErrorCode] = ErrorCode.PROVIDER_UNAVAILABLE
    status: ClassVar[int] = 503


class ProviderRateLimitedError(InfrastructureError):
    """Provider API rate-limited the request (HTTP 429)."""

    code: ClassVar[ErrorCode] = ErrorCode.PROVIDER_RATE_LIMITED
    status: ClassVar[int] = 429


# ---------------------------------------------------------------------------
# SSH (SSH_*)
# ---------------------------------------------------------------------------


class SSHConnectionFailedError(InfrastructureError):
    """SSH connection (handshake / auth at transport level) failed."""

    code: ClassVar[ErrorCode] = ErrorCode.SSH_CONNECTION_FAILED
    status: ClassVar[int] = 502


class SSHExecFailedError(InfrastructureError):
    """SSH remote command exited non-zero or could not be launched."""

    code: ClassVar[ErrorCode] = ErrorCode.SSH_EXEC_FAILED
    status: ClassVar[int] = 502


class SSHTransferFailedError(InfrastructureError):
    """SCP/SFTP file transfer over SSH failed."""

    code: ClassVar[ErrorCode] = ErrorCode.SSH_TRANSFER_FAILED
    status: ClassVar[int] = 502


# ---------------------------------------------------------------------------
# Trainer / training (TRAINING_*)
# ---------------------------------------------------------------------------


class TrainingFailedError(InfrastructureError):
    """Trainer subprocess exited non-zero with a non-OOM failure."""

    code: ClassVar[ErrorCode] = ErrorCode.TRAINING_FAILED
    status: ClassVar[int] = 500


class TrainingOOMError(InfrastructureError):
    """Trainer ran out of memory (OOM-killer or torch OOM)."""

    code: ClassVar[ErrorCode] = ErrorCode.TRAINING_OOM
    status: ClassVar[int] = 500


# ---------------------------------------------------------------------------
# Pipeline (PIPELINE_*, LAUNCH_*)
# ---------------------------------------------------------------------------


class PipelineStageFailedError(InfrastructureError):
    """A pipeline stage raised an unexpected/uncategorised error."""

    code: ClassVar[ErrorCode] = ErrorCode.PIPELINE_STAGE_FAILED
    status: ClassVar[int] = 500


class LaunchPreparationError(InfrastructureError):
    """Pre-launch preparation step failed (rendering, validation, IO)."""

    code: ClassVar[ErrorCode] = ErrorCode.LAUNCH_PREPARATION_FAILED
    status: ClassVar[int] = 500


# ---------------------------------------------------------------------------
# Workspace (WORKSPACE_*) -- 500 because filesystem/store IO failed
# ---------------------------------------------------------------------------


class WorkspaceStoreFailedError(InfrastructureError):
    """Workspace store IO (read/write/lock) failed."""

    code: ClassVar[ErrorCode] = ErrorCode.WORKSPACE_STORE_FAILED
    status: ClassVar[int] = 500


# ---------------------------------------------------------------------------
# Model / inference (MODEL_*, INFERENCE_*)
# ---------------------------------------------------------------------------


class ModelLoadFailedError(InfrastructureError):
    """Model artifacts could not be loaded (filesystem, weights, format)."""

    code: ClassVar[ErrorCode] = ErrorCode.MODEL_LOAD_FAILED
    status: ClassVar[int] = 500


class InferenceUnavailableError(InfrastructureError):
    """Inference service is not reachable / not ready."""

    code: ClassVar[ErrorCode] = ErrorCode.INFERENCE_UNAVAILABLE
    status: ClassVar[int] = 503


# ---------------------------------------------------------------------------
# Pod runner system / lifecycle (formerly raised via APIError)
# ---------------------------------------------------------------------------


class RunnerNotReadyError(InfrastructureError):
    """Runner accessed before its app.state finished initialising."""

    code: ClassVar[ErrorCode] = ErrorCode.RUNNER_NOT_READY
    status: ClassVar[int] = 503


class RunnerBusyError(InfrastructureError):
    """Runner is busy serving another request (transient back-off)."""

    code: ClassVar[ErrorCode] = ErrorCode.RUNNER_BUSY
    status: ClassVar[int] = 503


class SpawnFailedError(InfrastructureError):
    """Trainer subprocess could not be exec()'d."""

    code: ClassVar[ErrorCode] = ErrorCode.SPAWN_FAILED
    status: ClassVar[int] = 422


class FileWriteFailedError(InfrastructureError):
    """Atomic file write failed (disk full, permission denied, rename)."""

    code: ClassVar[ErrorCode] = ErrorCode.FILE_WRITE_FAILED
    status: ClassVar[int] = 502


class ResourcesUnavailableError(InfrastructureError):
    """``GET /resources`` snapshot provider raised — GPU/RAM probe failed."""

    code: ClassVar[ErrorCode] = ErrorCode.RESOURCES_UNAVAILABLE
    status: ClassVar[int] = 502


# ---------------------------------------------------------------------------
# Control API infra failures (formerly raised via raw HTTPException)
# ---------------------------------------------------------------------------


class RunnerUnreachableError(InfrastructureError):
    """SSH tunnel or runner HTTP call failed (transient 502)."""

    code: ClassVar[ErrorCode] = ErrorCode.RUNNER_UNREACHABLE
    status: ClassVar[int] = 502


class ReportGenerationFailedError(InfrastructureError):
    """Report compose / MLflow probe failed transiently."""

    code: ClassVar[ErrorCode] = ErrorCode.REPORT_GENERATION_FAILED
    status: ClassVar[int] = 503


class HFLoadFailedError(InfrastructureError):
    """HF Hub dataset preview/load failed (HTTP 502 upstream)."""

    code: ClassVar[ErrorCode] = ErrorCode.HF_LOAD_FAILED
    status: ClassVar[int] = 502


class DatasetsLibraryMissingError(InfrastructureError):
    """The optional ``datasets`` library is not installed at runtime."""

    code: ClassVar[ErrorCode] = ErrorCode.DATASETS_LIBRARY_MISSING
    status: ClassVar[int] = 500


class TrainingTimeoutError(InfrastructureError):
    """Trainer subprocess exceeded the supervisor's wall-clock watchdog."""

    code: ClassVar[ErrorCode] = ErrorCode.TRAINING_TIMEOUT
    status: ClassVar[int] = 500


# ---------------------------------------------------------------------------
# Metrics buffer (post-Phase-10 honest-failure mode)
# ---------------------------------------------------------------------------


class MetricsBufferTooLargeError(InfrastructureError):
    """Remote MetricsBuffer file exceeded the ultra-large hard cap.

    Raised by :class:`MetricsBufferRetriever` after the env-tunable
    threshold has already been bumped (e.g. via
    ``RYOTENKAI_METRICS_BUFFER_MAX_MB=500``) and the on-pod file is
    *still* larger than the hard cap (default ~1 GiB). Raising rather
    than silently skipping ensures the operator sees the failure and
    can either download the buffer manually or investigate the
    callback that produced it.
    """

    code: ClassVar[ErrorCode] = ErrorCode.METRICS_BUFFER_OVERSIZE
    status: ClassVar[int] = 500


__all__ = [
    "DatasetsLibraryMissingError",
    "FileWriteFailedError",
    "HFLoadFailedError",
    "InferenceUnavailableError",
    "LaunchPreparationError",
    "MetricsBufferTooLargeError",
    "ModelLoadFailedError",
    "PipelineStageFailedError",
    "ProviderRateLimitedError",
    "ProviderUnavailableError",
    "ReportGenerationFailedError",
    "ResourcesUnavailableError",
    "RunnerBusyError",
    "RunnerNotReadyError",
    "RunnerUnreachableError",
    "SSHConnectionFailedError",
    "SSHExecFailedError",
    "SSHTransferFailedError",
    "SpawnFailedError",
    "TrainingFailedError",
    "TrainingOOMError",
    "TrainingTimeoutError",
    "WorkspaceStoreFailedError",
]
