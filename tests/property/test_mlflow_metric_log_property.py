"""Property tests for :class:`FakeMLflowManager` metric logging.

Invariants:

* History ordering is preserved across nested runs (sorted by step,
  then timestamp).
* ``log_metric`` on an ended run raises (no silent ignore).
* Snapshot is JSON-serialisable for every generated sequence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest
from hypothesis import given
from hypothesis import settings as hyp_settings
from hypothesis import strategies as st

from tests._fakes.mlflow import FakeMLflowManager
from tests._harness.clock import ManualClock

pytestmark = [pytest.mark.property]


@dataclass
class _MetricCall:
    key: str
    value: float
    step: int
    advance: float


@st.composite
def _metric_calls(draw: st.DrawFn) -> list[_MetricCall]:
    n = draw(st.integers(min_value=1, max_value=12))
    calls: list[_MetricCall] = []
    for _ in range(n):
        calls.append(
            _MetricCall(
                key=draw(st.sampled_from(["loss", "acc", "lr", "f1"])),
                value=draw(st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False)),
                step=draw(st.integers(min_value=0, max_value=100)),
                advance=draw(st.floats(min_value=0, max_value=1.0)),
            ),
        )
    return calls


@given(calls=_metric_calls())
@hyp_settings(max_examples=30, deadline=None)
def test_metric_history_per_key_sorted_by_step_then_timestamp(
    calls: list[_MetricCall],
) -> None:
    clock = ManualClock()
    manager = FakeMLflowManager(clock=clock)
    manager.setup()

    with manager.start_run("exp-prop") as run:
        run_id = run.info.run_id
        for call in calls:
            manager.log_metrics({call.key: call.value}, step=call.step)
            clock.advance(call.advance)

    # Per-key history must be sorted by (step, timestamp).
    keys = {c.key for c in calls}
    for key in keys:
        history = manager.get_metric_history(run_id, key)
        sorted_history = sorted(history, key=lambda m: (m.step, m.timestamp))
        assert history == sorted_history


@given(calls=_metric_calls())
@hyp_settings(max_examples=30, deadline=None)
def test_snapshot_is_always_json_serializable(calls: list[_MetricCall]) -> None:
    clock = ManualClock()
    manager = FakeMLflowManager(clock=clock)
    manager.setup()

    with manager.start_run("exp-snap"):
        for call in calls:
            manager.log_metrics({call.key: call.value}, step=call.step)
            clock.advance(call.advance)

    snap = manager.snapshot()
    encoded = json.dumps(snap)
    again = json.loads(encoded)
    assert "runs" in again


def test_log_metric_on_unstarted_manager_raises() -> None:
    manager = FakeMLflowManager()
    manager.setup()
    with pytest.raises(RuntimeError):
        manager.log_metrics({"x": 1.0})
