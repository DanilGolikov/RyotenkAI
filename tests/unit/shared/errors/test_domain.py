"""Phase A1 -- per-class smoke for :mod:`ryotenkai_shared.errors.domain`.

Each concrete subclass gets one construction smoke ensuring ``code``,
``status``, ``title`` and the ``DomainError`` marker are all wired
together at class level. The base hierarchy (round-trip, invariants,
boundary, regressions, logic-specific) is exhaustively covered in
``test_base.py`` -- this file just guards the per-class pin.
"""

from __future__ import annotations

import pytest

from ryotenkai_shared.contracts.problem_details import ErrorCode
from ryotenkai_shared.errors import (
    ConfigDriftError,
    ConfigFileNotFoundError,
    ConfigInvalidError,
    DatasetLoadFailedError,
    DatasetValidationFailedError,
    DomainError,
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
)
from ryotenkai_shared.errors.base import RyotenkAIError

pytestmark = pytest.mark.unit


# (cls, expected_code, expected_status)
_DOMAIN_CATALOG: list[tuple[type[RyotenkAIError], ErrorCode, int]] = [
    (ConfigInvalidError, ErrorCode.CONFIG_INVALID, 400),
    (ConfigDriftError, ErrorCode.CONFIG_DRIFT, 409),
    (ConfigFileNotFoundError, ErrorCode.CONFIG_FILE_NOT_FOUND, 404),
    (JobSpecInvalidError, ErrorCode.JOB_SPEC_INVALID, 422),
    (JobStateInvalidError, ErrorCode.JOB_STATE_INVALID, 409),
    (ProjectNotFoundError, ErrorCode.PROJECT_NOT_FOUND, 404),
    (ProjectAlreadyExistsError, ErrorCode.PROJECT_ALREADY_EXISTS, 409),
    (ProviderNotFoundError, ErrorCode.PROVIDER_NOT_FOUND, 404),
    (IntegrationNotFoundError, ErrorCode.INTEGRATION_NOT_FOUND, 404),
    (StateLoadFailedError, ErrorCode.STATE_LOAD_FAILED, 404),
    (StateLockedError, ErrorCode.STATE_LOCKED, 409),
    (LaunchInProgressError, ErrorCode.LAUNCH_IN_PROGRESS, 409),
    (RunIsActiveError, ErrorCode.RUN_IS_ACTIVE, 409),
    (HFAuthFailedError, ErrorCode.HF_AUTH_FAILED, 401),
    (HFNotFoundError, ErrorCode.HF_NOT_FOUND, 404),
    (ProviderAuthFailedError, ErrorCode.PROVIDER_AUTH_FAILED, 401),
    (DatasetLoadFailedError, ErrorCode.DATASET_LOAD_FAILED, 422),
    (DatasetValidationFailedError, ErrorCode.DATASET_VALIDATION_FAILED, 422),
    (EngineNotRegisteredError, ErrorCode.ENGINE_NOT_REGISTERED, 404),
    (EngineConfigInvalidError, ErrorCode.ENGINE_CONFIG_INVALID, 422),
]


class TestPositive:
    """One construction smoke per concrete domain subclass."""

    @pytest.mark.parametrize(("cls", "expected_code", "expected_status"), _DOMAIN_CATALOG)
    def test_class_constants_pinned(
        self,
        cls: type[RyotenkAIError],
        expected_code: ErrorCode,
        expected_status: int,
    ) -> None:
        assert cls.code is expected_code
        assert cls.status == expected_status

    @pytest.mark.parametrize(("cls", "expected_code", "_status"), _DOMAIN_CATALOG)
    def test_construction_and_marker(
        self,
        cls: type[RyotenkAIError],
        expected_code: ErrorCode,
        _status: int,
    ) -> None:
        exc = cls("a contextual detail")
        assert exc.code is expected_code
        assert exc.title  # non-empty
        assert isinstance(exc, DomainError)
        assert isinstance(exc, RyotenkAIError)

    def test_catalog_size_matches_plan(self) -> None:
        """Sanity: we shipped exactly 20 domain.py subclasses in Phase A1."""
        assert len(_DOMAIN_CATALOG) == 20
