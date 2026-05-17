"""Unit tests: control-inference event types."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ryotenkai_shared.events import from_jsonl, to_jsonl
from ryotenkai_shared.events.types.control_inference import (
    InferenceDeactivatedEvent,
    InferenceDeactivatedPayload,
    InferenceDeployedEvent,
    InferenceDeployedPayload,
    InferenceDeploymentFailedEvent,
    InferenceDeploymentFailedPayload,
    InferenceDeploymentStartedEvent,
    InferenceDeploymentStartedPayload,
    InferenceHealthCheckCompletedEvent,
    InferenceHealthCheckCompletedPayload,
    InferenceHealthCheckStartedEvent,
    InferenceHealthCheckStartedPayload,
)


def _started() -> InferenceDeploymentStartedEvent:
    return InferenceDeploymentStartedEvent(
        source="control://orchestrator/inference_deployer",
        run_id="r",
        offset=0,
        payload=InferenceDeploymentStartedPayload(target="vllm", model_path="/m"),
    )


def _health() -> InferenceHealthCheckCompletedEvent:
    return InferenceHealthCheckCompletedEvent(
        source="control://orchestrator/inference_deployer",
        run_id="r",
        offset=1,
        payload=InferenceHealthCheckCompletedPayload(
            endpoint="http://i", latency_ms=12.0, model_loaded=True,
        ),
    )


def _deployed() -> InferenceDeployedEvent:
    return InferenceDeployedEvent(
        source="control://orchestrator/inference_deployer",
        run_id="r",
        offset=2,
        payload=InferenceDeployedPayload(
            endpoint="http://i", model_id="m-1",
        ),
    )


def _deactivated() -> InferenceDeactivatedEvent:
    return InferenceDeactivatedEvent(
        source="control://orchestrator/deployment_manager",
        run_id="r",
        offset=3,
        payload=InferenceDeactivatedPayload(endpoint="http://i", reason="evict"),
    )


def _health_started() -> InferenceHealthCheckStartedEvent:
    return InferenceHealthCheckStartedEvent(
        source="control://orchestrator/inference_deployer",
        run_id="r",
        offset=4,
        payload=InferenceHealthCheckStartedPayload(
            endpoint="http://i", timeout_s=600.0,
        ),
    )


def _failed() -> InferenceDeploymentFailedEvent:
    return InferenceDeploymentFailedEvent(
        source="control://orchestrator/inference_deployer",
        run_id="r",
        offset=5,
        payload=InferenceDeploymentFailedPayload(
            target="vllm",
            reason="no GPU capacity",
            error_type="InferenceUnavailableError",
        ),
    )


_ALL = [_started, _health, _deployed, _deactivated]
_ALL_PHASE5 = [_health_started, _failed]


class TestPositive:
    @pytest.mark.parametrize("factory", _ALL + _ALL_PHASE5, ids=lambda f: f.__name__)
    def test_round_trip(self, factory) -> None:
        original = factory()
        restored = from_jsonl(to_jsonl(original), strict=True)
        assert restored == original


class TestNegative:
    def test_started_rejects_unknown_target(self) -> None:
        with pytest.raises(ValidationError):
            InferenceDeploymentStartedPayload(  # type: ignore[arg-type]
                target="ollama",  # not in Literal
                model_path="/m",
            )

    def test_deployed_api_key_ref_optional(self) -> None:
        payload = InferenceDeployedPayload(endpoint="x", model_id="y")
        assert payload.api_key_ref is None

    def test_failed_rejects_unknown_target(self) -> None:
        with pytest.raises(ValidationError):
            InferenceDeploymentFailedPayload(  # type: ignore[arg-type]
                target="ollama",  # not in Literal
                reason="r",
                error_type="E",
            )


class TestInvariants:
    def test_all_inference_events_severity_info(self) -> None:
        for ev in _ALL:
            assert ev().severity == "info"

    def test_health_check_started_severity_debug(self) -> None:
        # Phase 5: started envelope is diagnostic / correlation-only —
        # the terminal "completed" envelope is what reports surface.
        assert _health_started().severity == "debug"

    def test_deployment_failed_severity_error(self) -> None:
        # Phase 5: failures must surface at error severity so reports /
        # alerts can match without parsing message strings.
        assert _failed().severity == "error"

    def test_health_check_started_kind_pinned(self) -> None:
        assert (
            _health_started().kind
            == "ryotenkai.control.inference.health_check_started"
        )

    def test_deployment_failed_kind_pinned(self) -> None:
        assert (
            _failed().kind
            == "ryotenkai.control.inference.deployment_failed"
        )
