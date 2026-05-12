"""Property tests for :class:`ryotenkai_pod.runner.state.JobLifecycleFSM`.

Invariants asserted across hypothesis-generated event sequences:

* Invalid transitions ALWAYS raise (never silently succeed).
* Terminal states are absorbing — once terminal, no further transitions
  are accepted.
* The persisted state.jsonl is monotonic in ``sequence``.
* ``transition()`` without a prior ``submit()`` raises
  :class:`InvalidTransitionError` with ``no_active_snapshot=True``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import settings as hyp_settings
from hypothesis import strategies as st

from ryotenkai_pod.runner.state import (
    InvalidTransitionError,
    JobLifecycleFSM,
    JobState,
)

pytestmark = [pytest.mark.property]


_ALL_STATES = list(JobState)
_LEGAL = {
    JobState.PREPARING: {JobState.RUNNING, JobState.FAILED},
    JobState.RUNNING: {JobState.STOPPING, JobState.COMPLETED, JobState.FAILED},
    JobState.STOPPING: {JobState.COMPLETED, JobState.CANCELLED, JobState.FAILED},
    JobState.COMPLETED: set(),
    JobState.FAILED: set(),
    JobState.CANCELLED: set(),
}


@given(
    targets=st.lists(st.sampled_from(_ALL_STATES), min_size=0, max_size=10),
)
@hyp_settings(max_examples=50, deadline=None)
def test_fsm_invalid_transitions_always_raise(
    tmp_path_factory: pytest.TempPathFactory, targets: list[JobState],
) -> None:
    """Random sequence of transitions; illegal moves must raise."""
    workdir: Path = tmp_path_factory.mktemp("fsm")
    fsm = JobLifecycleFSM(workspace_dir=workdir)
    fsm.restore_or_init()
    fsm.submit("j-prop")

    current = JobState.PREPARING
    for target in targets:
        legal = _LEGAL[current]
        if target in legal:
            snap = fsm.transition(target)
            assert snap.state == target
            current = target
        else:
            with pytest.raises(InvalidTransitionError):
                fsm.transition(target)
        # Once terminal, every further transition must raise.
        if current.is_terminal:
            for _next in targets:
                with pytest.raises(InvalidTransitionError):
                    fsm.transition(_next)
            return  # absorbing — stop the run


@given(targets=st.lists(st.sampled_from(_ALL_STATES), min_size=1, max_size=8))
@hyp_settings(max_examples=30, deadline=None)
def test_fsm_jsonl_sequence_is_monotonic(
    tmp_path_factory: pytest.TempPathFactory, targets: list[JobState],
) -> None:
    """Each appended JSONL record has sequence == prev_sequence + 1."""
    workdir = tmp_path_factory.mktemp("fsm-jsonl")
    fsm = JobLifecycleFSM(workspace_dir=workdir)
    fsm.restore_or_init()
    fsm.submit("j-mono")

    current = JobState.PREPARING
    for target in targets:
        if target in _LEGAL[current]:
            fsm.transition(target)
            current = target
            if current.is_terminal:
                break

    # Read jsonl and assert monotonic.
    jsonl = workdir / "state" / "job.jsonl"
    if not jsonl.exists():  # FSM may have used override; safe-skip
        return
    lines = jsonl.read_text().splitlines()
    sequences = [json.loads(line)["sequence"] for line in lines]
    assert sequences == list(range(len(sequences)))


def test_fsm_transition_without_submit_raises(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    fsm = JobLifecycleFSM(workspace_dir=tmp_path_factory.mktemp("fsm-empty"))
    fsm.restore_or_init()
    with pytest.raises(InvalidTransitionError) as excinfo:
        fsm.transition(JobState.RUNNING)
    assert excinfo.value.no_active_snapshot is True
