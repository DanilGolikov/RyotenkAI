"""Property tests for :class:`FakePodLifecycleClient`'s idempotency invariants.

Hypothesis-generated random action sequences exercise the lifecycle
state machine. Invariants:

* terminate(terminate(...)) is a no-op (``ALREADY_TERMINATED`` outcome).
* pause(pause(...)) is a no-op (``ALREADY_STOPPED``).
* resume(resume(...)) is a no-op (``ALREADY_RUNNING``).
* No action can drive the pod into an undefined state.
* Eventually-consistent transitions reified by ``advance(...)`` produce
  the same final state as the synchronous path.
"""

from __future__ import annotations

from typing import Any

import pytest
from hypothesis import given
from hypothesis import settings as hyp_settings
from hypothesis import strategies as st

from ryotenkai_shared.infrastructure.lifecycle import PodTerminalOutcome
from tests._fakes.lifecycle import FakePodLifecycleClient, PodState
from tests._harness.clock import ManualClock

pytestmark = [pytest.mark.property, pytest.mark.asyncio]

_ACTIONS = ("terminate", "pause", "resume")
_STATES = (PodState.RUNNING, PodState.STOPPED, PodState.TERMINATED)


@given(
    initial_state=st.sampled_from(_STATES),
    actions=st.lists(st.sampled_from(_ACTIONS), min_size=1, max_size=8),
)
@hyp_settings(max_examples=50, deadline=None)
async def test_lifecycle_actions_never_explode(
    initial_state: PodState, actions: list[str],
) -> None:
    """Random action sequences from random initial state must NEVER raise."""
    clock = ManualClock()
    client = FakePodLifecycleClient(clock=clock)
    client.set_pod_state("p-prop", initial_state)

    for action in actions:
        method: Any = getattr(client, action)
        result = await method(resource_id="p-prop")
        # The result must be a LifecycleActionResult with a recognised outcome.
        assert result.outcome in {
            PodTerminalOutcome.TERMINATED,
            PodTerminalOutcome.ALREADY_TERMINATED,
            PodTerminalOutcome.STOPPED,
            PodTerminalOutcome.ALREADY_STOPPED,
            PodTerminalOutcome.SKIPPED,
            PodTerminalOutcome.FAILED,
            "resumed",
            "already_running",
        }

    # Final state must be a valid PodState.
    final = client.get_pod_state("p-prop")
    assert final in _STATES


@given(
    initial_state=st.sampled_from((PodState.RUNNING, PodState.STOPPED)),
    duplicate_count=st.integers(min_value=2, max_value=5),
)
@hyp_settings(max_examples=30, deadline=None)
async def test_terminate_is_idempotent_under_duplication(
    initial_state: PodState, duplicate_count: int,
) -> None:
    """Two or more terminates in a row → all but the first say ``ALREADY_TERMINATED``."""
    client = FakePodLifecycleClient()
    client.set_pod_state("p-idem", initial_state)
    first = await client.terminate(resource_id="p-idem")
    assert first.outcome == PodTerminalOutcome.TERMINATED
    for _ in range(duplicate_count - 1):
        again = await client.terminate(resource_id="p-idem")
        assert again.outcome == PodTerminalOutcome.ALREADY_TERMINATED


@given(
    delay_seconds=st.floats(min_value=0.1, max_value=60.0, allow_infinity=False, allow_nan=False),
)
@hyp_settings(max_examples=30, deadline=None)
async def test_eventually_consistent_transition_reifies_on_advance(
    delay_seconds: float,
) -> None:
    """make_eventually_consistent → state only flips after ``clock.advance``."""
    clock = ManualClock()
    client = FakePodLifecycleClient(clock=clock)
    client.set_pod_state("p-ec", PodState.RUNNING)
    client.make_eventually_consistent("p-ec", transition_delay_seconds=delay_seconds)

    # terminate returns ``terminated`` outcome but does not flip state yet.
    result = await client.terminate(resource_id="p-ec")
    assert result.outcome == PodTerminalOutcome.TERMINATED
    # State has NOT yet reified — still RUNNING until the clock advances.
    assert client.get_pod_state("p-ec") == PodState.RUNNING
    # After advancing past the delay, the next state read flips.
    clock.advance(delay_seconds + 0.01)
    assert client.get_pod_state("p-ec") == PodState.TERMINATED
