"""Contract test: training provider's recreate-on-error filter MUST cover
every error code the waiter can emit for transient platform issues.

This test pins the relationship so the next person who adds a new
waiter error code sees a failing test pointing them at the provider
filter — instead of finding out at 02:00 from a paged operator.
"""

from __future__ import annotations

import pytest

from src.providers.runpod.training.provider import _RECREATABLE_ERRORS

pytestmark = pytest.mark.unit


# Codes the waiter currently emits for *transient platform issues* —
# i.e. the kind of failure that recreating a fresh pod is likely to
# resolve (RunPod allocator gave us a stuck node, capacity hiccup,
# community-cloud machine without exposed TCP, etc.).
#
# NOT included on purpose:
# * ``RUNPOD_POD_DATA_MISSING`` — the pod literally doesn't exist on
#   RunPod's side (operator deleted it from the console, or registry
#   pointed to a stale id). Recreating won't help; this needs operator
#   intervention.
# * ``RUNPOD_NO_PORTS_ALLOCATED`` — was a short-lived early-bailout
#   that fired at 180s into a 300s window. Removed: RunPod sometimes
#   takes the full window to allocate ports, and the early cutoff
#   forced retries on what would otherwise have been successful boots.
#   The "stuck with port_count==0" symptom now surfaces here as
#   ``RUNPOD_POD_TIMEOUT`` after the full window.
_TRANSIENT_WAITER_CODES: frozenset[str] = frozenset(
    {
        "RUNPOD_POD_TIMEOUT",
        "RUNPOD_POD_FAILED",
        "RUNPOD_NO_EXPOSED_TCP",
    }
)


def test_training_provider_recreates_on_every_transient_waiter_code() -> None:
    """``_RECREATABLE_ERRORS`` must be a superset of every transient
    error code ``PodSshWaiter`` is documented to emit.

    Failure mode if this drifts: the connect path aborts the whole
    pipeline run on a transient platform symptom that should have been
    handled by recreating a fresh pod.
    """
    missing = _TRANSIENT_WAITER_CODES - set(_RECREATABLE_ERRORS)
    assert not missing, (
        f"Training provider's _RECREATABLE_ERRORS is missing waiter codes: {sorted(missing)}. "
        f"Either add them to the tuple in src/providers/runpod/training/provider.py "
        f"or — if the new code is genuinely terminal (operator-only fix) — update "
        f"this test's _TRANSIENT_WAITER_CODES set with a comment explaining why."
    )


def test_recreatable_errors_does_not_include_pod_data_missing() -> None:
    """Sanity guard: ``RUNPOD_POD_DATA_MISSING`` must NEVER be in
    ``_RECREATABLE_ERRORS`` — the pod is gone, recreating in a loop
    won't recover it and would just burn budget."""
    assert "RUNPOD_POD_DATA_MISSING" not in _RECREATABLE_ERRORS, (
        "RUNPOD_POD_DATA_MISSING means the pod doesn't exist on RunPod's side. "
        "Auto-recreating in this case papers over registry/state-sync bugs "
        "that need operator visibility."
    )
