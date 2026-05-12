"""Cancellation chain ordering invariant.

Production cancellation telemetry emits four kinds in a strict partial
order::

    cancellation_started < cancellation_finalized < trainer_exited
                          < cancellation_completed

This test generates random monotonic timestamp sequences, assigns them
to events in any permutation, and asserts the validator we'd want to
ship in :mod:`ryotenkai_shared.observability` (or a callsite) accepts
the canonical order and rejects every permutation that violates it.

We intentionally implement the validator inline as a tiny pure
function rather than importing one from production — there is none
yet. When production grows one, swap the import here; the property
test remains the contract.
"""

from __future__ import annotations

from typing import NamedTuple

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

pytestmark = [pytest.mark.contract, pytest.mark.property]


_CANONICAL_ORDER: tuple[str, ...] = (
    "cancellation_started",
    "cancellation_finalized",
    "trainer_exited",
    "cancellation_completed",
)


class TimedEvent(NamedTuple):
    kind: str
    ts: int


def validate_cancellation_chain(events: list[TimedEvent]) -> bool:
    """Return True iff the input respects the canonical partial order.

    Definition (per Phase 3 D8 spec):
      ts(cancellation_started) < ts(cancellation_finalized)
      ts(cancellation_finalized) < ts(trainer_exited)
      ts(trainer_exited) < ts(cancellation_completed)

    Each kind may appear at most once.
    """
    by_kind: dict[str, int] = {}
    for ev in events:
        if ev.kind in by_kind:
            return False  # duplicate
        if ev.kind not in _CANONICAL_ORDER:
            continue  # unrelated event — we don't constrain it
        by_kind[ev.kind] = ev.ts

    if set(by_kind) != set(_CANONICAL_ORDER):
        # If the chain is incomplete, we treat it as valid: the partial
        # order only constrains *present* events. A "started but not
        # finalized" run still fails downstream — but this is not the
        # ordering invariant.
        present_indices = [_CANONICAL_ORDER.index(k) for k in by_kind]
        # Among the present kinds, every adjacent pair must be ordered.
        present_indices.sort()
        sorted_kinds = [_CANONICAL_ORDER[i] for i in present_indices]
        for i in range(len(sorted_kinds) - 1):
            if by_kind[sorted_kinds[i]] >= by_kind[sorted_kinds[i + 1]]:
                return False
        return True

    return (
        by_kind["cancellation_started"]
        < by_kind["cancellation_finalized"]
        < by_kind["trainer_exited"]
        < by_kind["cancellation_completed"]
    )


def test_canonical_chain_validates() -> None:
    chain = [TimedEvent(k, ts=i) for i, k in enumerate(_CANONICAL_ORDER)]
    assert validate_cancellation_chain(chain)


@pytest.mark.parametrize(
    "permutation",
    [
        # finalized before started
        [(_CANONICAL_ORDER[1], 0), (_CANONICAL_ORDER[0], 1), (_CANONICAL_ORDER[2], 2), (_CANONICAL_ORDER[3], 3)],
        # trainer_exited before finalized
        [(_CANONICAL_ORDER[0], 0), (_CANONICAL_ORDER[2], 1), (_CANONICAL_ORDER[1], 2), (_CANONICAL_ORDER[3], 3)],
        # completed first
        [(_CANONICAL_ORDER[3], 0), (_CANONICAL_ORDER[0], 1), (_CANONICAL_ORDER[1], 2), (_CANONICAL_ORDER[2], 3)],
    ],
)
def test_violated_orders_are_rejected(permutation: list[tuple[str, int]]) -> None:
    events = [TimedEvent(k, ts) for k, ts in permutation]
    assert not validate_cancellation_chain(events)


@given(
    starts=st.lists(
        st.tuples(
            st.sampled_from(_CANONICAL_ORDER),
            st.integers(min_value=0, max_value=10_000),
        ),
        min_size=1,
        max_size=4,
        unique_by=lambda pair: pair[0],
    ),
)
@settings(max_examples=80, suppress_health_check=[HealthCheck.too_slow])
def test_property_partial_orders_self_consistent(
    starts: list[tuple[str, int]],
) -> None:
    """If the input is sorted by canonical order with strictly
    monotonic ts, validation accepts; if any adjacent pair has
    ts[i] >= ts[i+1] in canonical order, it rejects."""
    events = [TimedEvent(k, ts) for k, ts in starts]
    by_kind = {k: ts for k, ts in starts}
    canonical_present = [k for k in _CANONICAL_ORDER if k in by_kind]
    times = [by_kind[k] for k in canonical_present]

    expected_valid = all(times[i] < times[i + 1] for i in range(len(times) - 1))
    assert validate_cancellation_chain(events) == expected_valid
