"""Typed exception hierarchy for the RyotenkAI monorepo.

Single root: :class:`RyotenkAIError`. Two abstract markers:
:class:`DomainError` (4xx semantics) and :class:`InfrastructureError`
(5xx semantics). Plus :class:`InternalError` catch-all and
:class:`TransportError` for Mac-side synthesised tunnel failures.

Every concrete subclass pins ``code: ClassVar[ErrorCode]`` and
``status: ClassVar[int]`` so the boundary protocol (RFC 9457
problem+json) can be produced via :meth:`RyotenkAIError.as_problem`
without ad-hoc construction at raise sites.

See ``docs/plans/sharded-stargazing-wigderson.md`` for the design.
"""

from ryotenkai_shared.errors.base import (
    DomainError,
    InfrastructureError,
    InternalError,
    RyotenkAIError,
    TransportError,
)
from ryotenkai_shared.errors.domain import (
    ConfigDriftError,
    ConfigFileNotFoundError,
    ConfigInvalidError,
    DatasetLoadFailedError,
    DatasetValidationFailedError,
    EngineConfigInvalidError,
    EngineNotRegisteredError,
    HFAuthFailedError,
    HFNotFoundError,
    IntegrationNotFoundError,
    JobSpecInvalidError,
    JobStateInvalidError,
    LaunchInProgressError,
    ProjectAlreadyExistsError,
    ProjectNotFoundError,
    ProviderAuthFailedError,
    ProviderNotFoundError,
    RunIsActiveError,
    StateLoadFailedError,
    StateLockedError,
    StrategyChainInvalidError,
)
from ryotenkai_shared.errors.infra import (
    InferenceUnavailableError,
    LaunchPreparationError,
    ModelLoadFailedError,
    PipelineStageFailedError,
    ProviderRateLimitedError,
    ProviderUnavailableError,
    SSHConnectionFailedError,
    SSHExecFailedError,
    SSHTransferFailedError,
    TrainingFailedError,
    TrainingOOMError,
    WorkspaceStoreFailedError,
)

__all__ = [
    # base
    "RyotenkAIError",
    "DomainError",
    "InfrastructureError",
    "InternalError",
    "TransportError",
    # domain (4xx)
    "ConfigInvalidError",
    "ConfigDriftError",
    "ConfigFileNotFoundError",
    "JobSpecInvalidError",
    "JobStateInvalidError",
    "ProjectNotFoundError",
    "ProjectAlreadyExistsError",
    "ProviderNotFoundError",
    "IntegrationNotFoundError",
    "StateLoadFailedError",
    "StateLockedError",
    "StrategyChainInvalidError",
    "LaunchInProgressError",
    "RunIsActiveError",
    "HFAuthFailedError",
    "HFNotFoundError",
    "ProviderAuthFailedError",
    "DatasetLoadFailedError",
    "DatasetValidationFailedError",
    "EngineNotRegisteredError",
    "EngineConfigInvalidError",
    # infra (5xx)
    "ProviderUnavailableError",
    "ProviderRateLimitedError",
    "SSHConnectionFailedError",
    "SSHExecFailedError",
    "SSHTransferFailedError",
    "TrainingFailedError",
    "TrainingOOMError",
    "PipelineStageFailedError",
    "LaunchPreparationError",
    "WorkspaceStoreFailedError",
    "ModelLoadFailedError",
    "InferenceUnavailableError",
]
