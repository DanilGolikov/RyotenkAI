"""Phase 14.B — runner-side pod-lifecycle Protocol.

Distinct from the Mac-side
:class:`~src.providers.training.interfaces.ITerminalActionProvider`
because it runs **inside** an already-active uvicorn event loop and
needs:

* **Async transport** — :class:`~httpx.AsyncClient` is the canonical
  RunPod GraphQL client. Re-using the sync Mac-side Protocol would
  force ``asyncio.run`` from inside a running loop (RuntimeError) or
  threadpool offload (cancellation-leak class of bugs Phase 11.B
  specifically avoided).
* **Idempotent-marker detection** — "already terminated" / "not
  running" / "does not exist" are *successful* outcomes on the
  runner side ("intent satisfied"), not errors. A ``Result``-shaped
  return would lose that nuance.
* **Retry-aware return shape** — ``attempts_made`` is surfaced in
  telemetry so dashboards see "we wanted to terminate, took 3 tries".

Phase 14.B § 1.1 documents the locked decision: two distinct
Protocols (Mac-side sync, runner-side async) coexist with a
CI-checked invariant linking them — neither inherits from the other.

Phase 14.B § 1.2 documents the choice of :class:`LifecycleActionResult`
over :data:`~src.utils.result.Result` — the action-stage outcome
vocabulary
(:class:`~src.runner.pod_terminator.PodTerminalOutcome` strings) is a
closed enum that already exists, telemetry consumers grep for those
strings, and idempotent already-gone is a successful outcome here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

# Phase 14.B — keep ``raw_response_excerpt`` truncated so a flaky
# upstream that returns a 100MB error page doesn't push a 100MB event
# into the bus. 300 is the same length used by ``_call_mutation``'s
# in-flight error-message construction in the pre-14.B
# :mod:`src.runner.pod_terminator` so this preserves the wire shape.
_RAW_RESPONSE_EXCERPT_MAX_CHARS = 300


@dataclass(frozen=True)
class LifecycleActionResult:
    """Outcome of a single ``terminate`` / ``pause`` / ``resume`` call.

    Attributes:
        outcome:
            One of the strings on
            :class:`~src.runner.pod_terminator.PodTerminalOutcome`
            (``terminated`` / ``already_terminated`` / ``stopped`` /
            ``already_stopped`` / ``failed`` / ``skipped``) plus
            two NEW strings only used by
            :meth:`IPodLifecycleClient.resume` —
            ``"resumed"`` and ``"already_running"``. These two are
            **not** added to :class:`PodTerminalOutcome` in Phase 14.B
            (we'd be broadening the public vocabulary unnecessarily —
            no caller inside the pod uses ``resume`` today). Phase
            14.E or beyond may promote them.
        attempts_made:
            ``1..max_attempts`` for transports that retry,
            ``0`` for the no-op single-node case (no transport to
            attempt). Surfaced in the ``pod_stop_attempt`` event so
            operators see retry pressure on their fleet.
        last_error:
            ``repr(exc)`` of the final exception, or a snippet of the
            HTTP response body when the upstream returned a
            non-success that didn't match any idempotency marker.
            ``None`` on success or no-op outcomes.
        raw_response_excerpt:
            First ``_RAW_RESPONSE_EXCERPT_MAX_CHARS`` chars of the
            response body for forensics. Truncated, not raise — a
            response longer than the cap is silently shortened.

    Phase 14.B keeps :attr:`outcome` typed as ``str`` rather than
    ``Literal[*PodTerminalOutcome strings]`` because ``resume``'s
    extra outcomes aren't on the enum yet (see § 1.2 of the plan).
    Tests pin the vocabulary so a typo in a provider impl is caught
    at lint time without forcing the type system to enumerate every
    string.
    """

    outcome: str
    attempts_made: int
    last_error: str | None = None
    raw_response_excerpt: str | None = None

    def __post_init__(self) -> None:
        # Truncation is silent (no error). The cap protects the bus
        # from huge upstream payloads; operators don't lose semantic
        # information because the outcome field already encodes
        # "what happened" — the excerpt is only for forensics.
        if (
            self.raw_response_excerpt is not None
            and len(self.raw_response_excerpt) > _RAW_RESPONSE_EXCERPT_MAX_CHARS
        ):
            object.__setattr__(
                self,
                "raw_response_excerpt",
                self.raw_response_excerpt[:_RAW_RESPONSE_EXCERPT_MAX_CHARS],
            )


@runtime_checkable
class IPodLifecycleClient(Protocol):
    """Runner-side equivalent of
    :class:`~src.providers.training.interfaces.ITerminalActionProvider`.

    Implemented by:

    * :class:`~src.providers.runpod.runtime.lifecycle_client.RunPodPodLifecycleClient`
      — RunPod GraphQL transport.
    * :class:`~src.providers.single_node.runtime.lifecycle_client.NoOpPodLifecycleClient`
      — single-node host (no cloud lifecycle to act on).

    Cross-reference:
    :class:`~src.providers.training.interfaces.ITerminalActionProvider`
    is the Mac-side equivalent. They share method names + verbs +
    :class:`~src.providers.training.interfaces.VolumeKind` but neither
    inherits from the other. See Phase 14.B § 1.1 for rationale.

    Note:
        Method signatures take only ``resource_id`` — no ``reason``
        parameter. The runner-side caller (the supervisor's terminal
        hook) infers reason from ``terminal_state`` and adds it to the
        bus event payload. Mac-side
        :class:`ITerminalActionProvider.terminate` keeps ``reason`` in
        its sig because Mac callers use it for audit logs.
    """

    @property
    def provider_name(self) -> str:
        """Registry key — must equal one of
        :data:`~src.constants.PROVIDER_RUNPOD` /
        :data:`~src.constants.PROVIDER_SINGLE_NODE` /
        future provider names. Used in
        :func:`~src.runner.runtime.provider_registry.resolve_lifecycle_client_from_env`
        to validate the registry-vs-name parity invariant."""
        ...

    async def terminate(self, *, resource_id: str) -> LifecycleActionResult:
        """Permanently destroy the resource. Idempotent: already-gone
        ⇒ ``LifecycleActionResult(outcome="already_terminated", ...)``."""
        ...

    async def pause(self, *, resource_id: str) -> LifecycleActionResult:
        """Halt the resource preserving disk state for resume.
        Idempotent: already-stopped ⇒
        ``LifecycleActionResult(outcome="already_stopped", ...)``."""
        ...

    async def resume(self, *, resource_id: str) -> LifecycleActionResult:
        """Resume a previously paused resource. Symmetric with
        :meth:`pause` — present on the Protocol for completeness even
        though the runner doesn't self-resume today (Mac wakes the
        pod, not the other way around). Outcomes ``"resumed"`` /
        ``"already_running"`` (NOT on
        :class:`~src.runner.pod_terminator.PodTerminalOutcome` in
        Phase 14.B — see :class:`LifecycleActionResult` docstring)."""
        ...
