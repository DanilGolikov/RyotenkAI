"""Compliance tests for :class:`IMLflowManager`.

Parametrized over ``[fake, real]``. The ``real`` variant requires
``RYOTENKAI_LIVE=1`` and a reachable MLflow tracking URI; otherwise it
``pytest.skip``s. The ``fake`` variant runs on every PR.

Markers:

* ``compliance`` — protocol-compliance suite
* ``exercises_protocol("IMLflowManager")``
* ``uses_fake("FakeMLflowManager")``
"""

from __future__ import annotations

import os
from typing import Any

import pytest

from ryotenkai_shared.infrastructure.mlflow.protocol import IMLflowManager
from tests._fakes.mlflow import (
    FakeMLflowManager,
    MLflowUnavailableError,
    TransientMLflowError,
)

pytestmark = [
    pytest.mark.contract,
    pytest.mark.compliance,
    pytest.mark.exercises_protocol("IMLflowManager"),
    pytest.mark.uses_fake("FakeMLflowManager"),
]


# WHY parametrize: the suite is structured so adding a real impl is a
# one-line ``"real"`` extension to the params. Until then, ``real``
# is a ``pytest.skip``.
@pytest.fixture(params=["fake", pytest.param("real", marks=pytest.mark.live)])
def mlflow_manager(request: pytest.FixtureRequest, manual_clock: Any) -> IMLflowManager:
    if request.param == "real":
        if os.environ.get("RYOTENKAI_LIVE") != "1":
            pytest.skip("real IMLflowManager requires RYOTENKAI_LIVE=1")
        pytest.skip("real MLflowManager not yet wired into compliance suite")
    fake = FakeMLflowManager(clock=manual_clock)
    fake.setup()
    return fake


def _as_fake(manager: IMLflowManager) -> FakeMLflowManager:
    assert isinstance(manager, FakeMLflowManager)
    return manager


class TestMLflowManagerCompliance:
    """Per-Protocol contract — must pass against fake AND real impls."""

    def test_isinstance_protocol(self, mlflow_manager: IMLflowManager) -> None:
        assert isinstance(mlflow_manager, IMLflowManager)

    def test_setup_marks_active(self, mlflow_manager: IMLflowManager) -> None:
        assert mlflow_manager.is_active is True
        assert mlflow_manager.tracking_uri is not None

    def test_start_run_creates_run_record(self, mlflow_manager: IMLflowManager) -> None:
        with mlflow_manager.start_run("exp-positive") as run:
            assert run.info.run_id
        fake = _as_fake(mlflow_manager)
        assert len(fake.runs_for("exp-positive")) == 1

    def test_log_params_round_trips(self, mlflow_manager: IMLflowManager) -> None:
        with mlflow_manager.start_run("exp-params") as run:
            mlflow_manager.log_params({"lr": 0.01, "batch_size": 32})
            run_id = run.info.run_id
        fake = _as_fake(mlflow_manager)
        record = fake.get_run(run_id)
        assert record.params == {"lr": 0.01, "batch_size": 32}

    def test_log_metrics_appends_history_in_order(
        self, mlflow_manager: IMLflowManager, manual_clock: Any,
    ) -> None:
        with mlflow_manager.start_run("exp-history") as run:
            mlflow_manager.log_metrics({"loss": 1.0}, step=0)
            manual_clock.advance(1)
            mlflow_manager.log_metrics({"loss": 0.5}, step=1)
            manual_clock.advance(1)
            mlflow_manager.log_metrics({"loss": 0.25}, step=2)
            run_id = run.info.run_id
        fake = _as_fake(mlflow_manager)
        history = fake.get_metric_history(run_id, "loss")
        assert [m.value for m in history] == [1.0, 0.5, 0.25]
        assert [m.step for m in history] == [0, 1, 2]

    def test_end_run_is_idempotent(self, mlflow_manager: IMLflowManager) -> None:
        # WHY: real mlflow's ``end_run`` is idempotent — second call no-ops.
        # Compliance tests assert this explicitly because callers depend on
        # it for finally-block cleanup.
        with mlflow_manager.start_run("exp-idempotent"):
            mlflow_manager.log_metrics({"x": 1.0})
        mlflow_manager.end_run()  # extra call — must be a no-op
        mlflow_manager.end_run()
        # No exception means we comply.

    def test_nested_run_pops_to_parent(self, mlflow_manager: IMLflowManager) -> None:
        with mlflow_manager.start_run("exp-nested") as parent:
            parent_id = parent.info.run_id
            with mlflow_manager.start_nested_run("child-A") as child:
                child_id = child.info.run_id
                assert child_id != parent_id
            # After child exits we should be back to parent.
            mlflow_manager.log_params({"after_child": True})
        fake = _as_fake(mlflow_manager)
        assert "after_child" in fake.get_run(parent_id).params

    def test_log_event_info_returns_dict_with_message(
        self, mlflow_manager: IMLflowManager,
    ) -> None:
        with mlflow_manager.start_run("exp-event"):
            event = mlflow_manager.log_event_info("starting deploy", category="deployment")
            assert event["kind"] == "info"
            assert event["message"] == "starting deploy"
            assert event["category"] == "deployment"

    def test_log_provider_info_persists_tags(self, mlflow_manager: IMLflowManager) -> None:
        with mlflow_manager.start_run("exp-provider") as run:
            mlflow_manager.log_provider_info(
                provider_name="runpod",
                provider_type="cloud",
                gpu_type="A100",
                resource_id="abc123",
            )
            run_id = run.info.run_id
        fake = _as_fake(mlflow_manager)
        tags = fake.get_run(run_id).tags
        assert tags["provider.name"] == "runpod"
        assert tags["provider.type"] == "cloud"
        assert tags["provider.gpu_type"] == "A100"
        assert tags["provider.resource_id"] == "abc123"

    # -- chaos surface --------------------------------------------------

    def test_chaos_fail_next_n_calls_exhausts_after_count(
        self, mlflow_manager: IMLflowManager,
    ) -> None:
        # Compliance: every chaos method on the fake must be exercised by
        # at least one compliance test (per spec).
        fake = _as_fake(mlflow_manager)
        fake.fail_next_n_calls(2, kind=TransientMLflowError)
        with pytest.raises(TransientMLflowError):
            mlflow_manager.start_run("exp-chaos")
        with pytest.raises(TransientMLflowError):
            mlflow_manager.start_run("exp-chaos")
        # Third call recovers.
        with mlflow_manager.start_run("exp-chaos"):
            pass
        # Pod is now consistent: success path completed.
        assert len(fake.runs_for("exp-chaos")) == 1

    def test_chaos_set_unavailable_blocks_calls(
        self, mlflow_manager: IMLflowManager,
    ) -> None:
        fake = _as_fake(mlflow_manager)
        fake.set_unavailable(True)
        with pytest.raises(MLflowUnavailableError):
            mlflow_manager.start_run("exp-down")
        fake.set_unavailable(False)
        with mlflow_manager.start_run("exp-down"):
            pass

    def test_chaos_inject_latency_records_in_snapshot(
        self, mlflow_manager: IMLflowManager,
    ) -> None:
        # Latency injection on a sync Protocol surface is best-effort
        # visible only via snapshot — but the compliance test still pins
        # it so dead-code regressions are caught.
        fake = _as_fake(mlflow_manager)
        fake.inject_latency_ms(50)
        snap = fake.snapshot()
        assert snap["chaos"]["latency_seconds"] == pytest.approx(0.05)
        fake.reset_chaos()
        assert fake.snapshot()["chaos"]["latency_seconds"] == 0.0

    def test_snapshot_is_json_serializable(self, mlflow_manager: IMLflowManager) -> None:
        import json
        with mlflow_manager.start_run("exp-snap"):
            mlflow_manager.log_metrics({"loss": 0.1}, step=0)
        fake = _as_fake(mlflow_manager)
        # Round-trip through JSON catches non-serializable internals.
        encoded = json.dumps(fake.snapshot())
        round_tripped = json.loads(encoded)
        assert round_tripped["chaos"]["latency_seconds"] == 0.0


__all__: list[str] = []
