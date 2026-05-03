"""Pod terminal-state outcome vocabulary.

Lived in :mod:`ryotenkai_pod.runner.pod_terminator` until ADR row 7
(Phase B/C audit) — provider adapters needed it but were forbidden
from importing pod-side code. The strings (decision + action) are pure
data; moving them into shared lets every party agree without crossing
the pod boundary.

The class deliberately uses string constants instead of an :mod:`enum`
so payloads round-trip through JSON cleanly and operator alerts can
grep for outcome names directly.
"""

from __future__ import annotations


class PodTerminalOutcome:
    """Outcomes published on the bus for operator dashboards.

    Decision-stage outcomes (what we *intended* to do):
    """

    TERMINATED_USER_STOP = "terminated_user_stop"
    """User clicked Stop → terminate (irreversible)."""

    TERMINATED_SAFETY = "terminated_safety"
    """Failed run on a network volume → terminate (no resume path).

    Pre PR-C this outcome also covered ``failed + mac_alive`` on
    persistent volumes — but that left Mac zero grace to SCP the final
    diagnostics, racing the post-mortem with pod teardown. Persistent
    failures on alive Mac now route to
    :data:`TERMINATED_AFTER_DIAGNOSTIC_GRACE` instead.
    """

    TERMINATED_AFTER_DIAGNOSTIC_GRACE = "terminated_after_diagnostic_grace"
    """PR-C — failed + Mac alive (persistent volume): wait briefly so
    Mac's post-mortem SCP completes, then terminate. Bounded by
    ``PodTerminator.DIAGNOSTIC_GRACE_SECONDS`` and aborted early
    if the heartbeat dies (Mac went away → grace pointless)."""

    STOPPED_FOR_RESUME = "stopped_for_resume"
    """Mac asleep → pause immediately (artifacts wait for resume)."""

    STOPPED_FOR_RESUME_SHORT_GRACE = "stopped_for_resume_short_grace"
    """Mac alive → wait for retriever, then pause."""

    KEPT_ALIVE_FOR_DEBUG = "kept_alive_for_debug"
    """``RUNPOD_KEEP_ON_ERROR=true`` on failed → no-op (SSH-forensics)."""

    DISABLED = "disabled"
    """Reserved — currently unused. Phase 11 removes ``AUTO_STOP``,
    so 'disabled' as a decision is gone. Kept in the enum for
    forward-compat (e.g. future ``RUNPOD_TERMINAL_OFF`` toggle)."""

    SKIPPED = "skipped"
    """Provider has nothing to act on (single-node NoOp client)."""

    #: Action-stage outcomes (what actually happened on the provider call).
    #: Reported alongside the decision outcome so operators can see
    #: "we wanted to terminate, the call returned already-gone".

    TERMINATED = "terminated"
    """Provider terminate returned success."""

    ALREADY_TERMINATED = "already_terminated"
    """Provider terminate returned an already-gone marker; idempotent."""

    STOPPED = "stopped"
    """Provider pause returned success."""

    ALREADY_STOPPED = "already_stopped"
    """Provider pause returned an already-stopped/gone marker; idempotent."""

    FAILED = "failed"
    """All retry attempts exhausted on transient errors."""


__all__ = ["PodTerminalOutcome"]
