"""Phase A1 -- per-class smoke for :mod:`ryotenkai_shared.errors.infra`.

Each concrete subclass gets one construction smoke ensuring ``code``,
``status``, ``title`` and the ``InfrastructureError`` marker are all
wired together. The base hierarchy is covered exhaustively in
``test_base.py``.
"""

from __future__ import annotations

import pytest

from ryotenkai_shared.contracts.problem_details import ErrorCode
from ryotenkai_shared.errors import (
    InferenceUnavailableError,
    InfrastructureError,
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
from ryotenkai_shared.errors.base import RyotenkAIError

pytestmark = pytest.mark.unit


# (cls, expected_code, expected_status)
_INFRA_CATALOG: list[tuple[type[RyotenkAIError], ErrorCode, int]] = [
    (ProviderUnavailableError, ErrorCode.PROVIDER_UNAVAILABLE, 503),
    (ProviderRateLimitedError, ErrorCode.PROVIDER_RATE_LIMITED, 429),
    (SSHConnectionFailedError, ErrorCode.SSH_CONNECTION_FAILED, 502),
    (SSHExecFailedError, ErrorCode.SSH_EXEC_FAILED, 502),
    (SSHTransferFailedError, ErrorCode.SSH_TRANSFER_FAILED, 502),
    (TrainingFailedError, ErrorCode.TRAINING_FAILED, 500),
    (TrainingOOMError, ErrorCode.TRAINING_OOM, 500),
    (PipelineStageFailedError, ErrorCode.PIPELINE_STAGE_FAILED, 500),
    (LaunchPreparationError, ErrorCode.LAUNCH_PREPARATION_FAILED, 500),
    (WorkspaceStoreFailedError, ErrorCode.WORKSPACE_STORE_FAILED, 500),
    (ModelLoadFailedError, ErrorCode.MODEL_LOAD_FAILED, 500),
    (InferenceUnavailableError, ErrorCode.INFERENCE_UNAVAILABLE, 503),
]


class TestPositive:
    """One construction smoke per concrete infra subclass."""

    @pytest.mark.parametrize(("cls", "expected_code", "expected_status"), _INFRA_CATALOG)
    def test_class_constants_pinned(
        self,
        cls: type[RyotenkAIError],
        expected_code: ErrorCode,
        expected_status: int,
    ) -> None:
        assert cls.code is expected_code
        assert cls.status == expected_status

    @pytest.mark.parametrize(("cls", "expected_code", "_status"), _INFRA_CATALOG)
    def test_construction_and_marker(
        self,
        cls: type[RyotenkAIError],
        expected_code: ErrorCode,
        _status: int,
    ) -> None:
        exc = cls("a contextual detail")
        assert exc.code is expected_code
        assert exc.title  # non-empty
        assert isinstance(exc, InfrastructureError)
        assert isinstance(exc, RyotenkAIError)

    def test_catalog_size_matches_plan(self) -> None:
        """Sanity: we shipped exactly 12 infra.py subclasses in Phase A1."""
        assert len(_INFRA_CATALOG) == 12
