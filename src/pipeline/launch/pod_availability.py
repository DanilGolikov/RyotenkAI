"""Phase 11.C — pod availability probe + resume-with-retry.

Used by ``ryotenkai run resume`` (CLI) and ``POST /runs/{run_id}/resume``
(Web UI) to:

1. Read the persisted ``pod_metadata`` from the latest attempt's
   ``PipelineState`` (Phase 11.C state schema extension).
2. Query the provider for current pod status.
3. Map the provider's status to one of five availability states:

   * **RUNNING** — pod is up and reachable; no resume needed,
     just continue with the pipeline.
   * **SLEEPING_RESUMABLE** — pod is in the EXITED / stopped state
     (Phase 11.B's ``podStop`` outcome). Call
     :func:`resume_pod_with_retry` to wake it.
   * **SLEEPING_RESUME_FAILED** — pod was sleeping but
     :func:`resume_pod_with_retry` exhausted its budget (typically
     "no GPU available in this datacenter"). Surface to user; let
     them decide whether to wait or recreate from checkpoint.
   * **GONE** — pod is fully terminated (Phase 11.B's
     ``podTerminate`` outcome, or operator manually deleted via
     RunPod console). Resume in-place is impossible; user must
     ``run restart`` from a checkpoint.
   * **PROBE_FAILED** — RunPod GraphQL outage / auth error. We can't
     determine state. Return PROBE_FAILED so the CLI / UI can show
     a useful error rather than silently treat as GONE.

Why a separate module?
----------------------

* Provider-agnostic surface (``PodAvailability`` is a pure enum;
  the probe takes a callable transport so single_node / future
  providers can plug in).
* Sits on the launch path next to :func:`validate_resume_run`
  (``restart_options.py``) — same import-graph layer.
* Tests can mock the transport without setting up RunPod SDK.

Retry semantics for resume_pod_with_retry
-----------------------------------------

RunPod sometimes can't immediately fulfil a ``resume_pod`` request
in the original datacenter (capacity exhausted, GPU type sold out,
etc.). We retry with exponential backoff:

* Backoff sequence: 10s → 30s → 60s → 120s (4 attempts, total 220s
  of waiting + 4 fast probes ≈ 5 minutes total budget).
* Capacity-error detection reuses
  :data:`src.providers.runpod.sdk_adapter._CAPACITY_MARKERS` —
  same set of substrings that govern retry on pod creation.
* Non-capacity errors (auth, malformed request, pod already gone)
  fail fast — no retry, surface immediately.

Out-of-scope (Phase 11.C):
* Auto-fallback to a different datacenter on capacity exhaustion.
* Auto-create new pod from latest checkpoint on GONE. (User must
  explicitly ``run restart`` per § 11.5 — we don't silently re-burn
  budget on their behalf.)
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.pipeline.state.models import PodMetadata


__all__ = [
    "PodAvailability",
    "PodAvailabilityProbe",
    "ProbeResult",
    "RESUME_RETRY_BUDGET_SECONDS",
    "RESUME_BACKOFFS",
    "ResumeResult",
    "load_pod_metadata_for_run",
    "resume_pod_with_retry",
]


# ---------------------------------------------------------------------------
# Retry budget constants
# ---------------------------------------------------------------------------


#: Total wall-clock budget (seconds) for ``resume_pod_with_retry``.
#: After this expires across attempts, surface a clear "capacity
#: unavailable" error rather than retry forever. Long enough to
#: cover transient RunPod queue overflows; short enough that the
#: CLI / UI doesn't appear hung.
RESUME_RETRY_BUDGET_SECONDS: float = 300.0  # 5 minutes

#: Exponential-ish backoffs between attempts. Matches the plan's
#: § 7.2 sequence. The sum (10+30+60+120 = 220s) plus probe latency
#: caps at ~5 min, in line with :data:`RESUME_RETRY_BUDGET_SECONDS`.
RESUME_BACKOFFS: tuple[float, ...] = (10.0, 30.0, 60.0, 120.0)


# ---------------------------------------------------------------------------
# Enum + result dataclass
# ---------------------------------------------------------------------------


class PodAvailability(str, Enum):
    """Coarse availability states the probe maps RunPod statuses into.

    The values are stable strings — operator dashboards and Web UI
    badges grep on them, so renaming is a contract change.
    """

    RUNNING = "running"
    SLEEPING_RESUMABLE = "sleeping_resumable"
    SLEEPING_RESUME_FAILED = "sleeping_resume_failed"
    GONE = "gone"
    PROBE_FAILED = "probe_failed"

    @property
    def is_resume_needed(self) -> bool:
        """True iff the caller should call :func:`resume_pod_with_retry`."""
        return self == PodAvailability.SLEEPING_RESUMABLE

    @property
    def is_recoverable(self) -> bool:
        """True iff the caller can usefully proceed with the pipeline.

        RUNNING ⇒ no action needed; ModelRetriever can SSH right away.
        SLEEPING_RESUMABLE ⇒ resume first, then proceed.
        Other states ⇒ user intervention required.
        """
        return self in (
            PodAvailability.RUNNING,
            PodAvailability.SLEEPING_RESUMABLE,
        )


@dataclass(frozen=True)
class ProbeResult:
    """Carries the probe verdict + any operator-visible context.

    Always returned (never an exception) so callers can render a
    consistent UX. ``message`` is human-readable for CLI / UI surface.
    """
    availability: PodAvailability
    pod_id: str
    runpod_status: str | None = None
    message: str = ""

    @property
    def is_recoverable(self) -> bool:
        """Convenience proxy to :attr:`PodAvailability.is_recoverable`.

        Lets callers write ``probe.probe(meta).is_recoverable`` instead
        of ``probe.probe(meta).availability.is_recoverable``.
        """
        return self.availability.is_recoverable

    @property
    def is_resume_needed(self) -> bool:
        """Convenience proxy to :attr:`PodAvailability.is_resume_needed`."""
        return self.availability.is_resume_needed


# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------


#: Mapping from RunPod's pod ``desiredStatus`` field to internal availability.
#: RunPod uses ``RUNNING`` / ``EXITED`` / ``STOPPED`` / ``TERMINATED``;
#: we map all "stopped-ish" states to SLEEPING_RESUMABLE.
_RUNPOD_STATUS_MAP: dict[str, PodAvailability] = {
    "RUNNING": PodAvailability.RUNNING,
    "EXITED": PodAvailability.SLEEPING_RESUMABLE,
    "STOPPED": PodAvailability.SLEEPING_RESUMABLE,
    "PAUSED": PodAvailability.SLEEPING_RESUMABLE,
    "TERMINATED": PodAvailability.GONE,
    "DEAD": PodAvailability.GONE,
}


class PodAvailabilityProbe:
    """Queries the provider for pod status; maps to :class:`PodAvailability`.

    The transport is injected via constructor — tests pass a mock
    callable returning ``(status_str, error)`` tuples; production
    wires :class:`RunPodAPIClient.query_pod` adapted to that shape.

    Provider-agnostic by design: when single_node grows resume
    semantics, we plug a parallel transport without touching the
    decision logic.
    """

    def __init__(
        self,
        *,
        query_pod: "Callable[[str], dict[str, Any]] | None" = None,
    ) -> None:
        """Build a probe.

        Args:
            query_pod: Callable ``(pod_id) -> dict``. Returns the
                provider's pod-status dict (RunPod shape: keys
                ``desiredStatus``, ``runtime``, etc.). Raise on
                error; the probe catches and maps to PROBE_FAILED.
                ``None`` ⇒ probe always returns PROBE_FAILED (used
                only in tests; production passes a real transport).
        """
        self._query_pod = query_pod

    def probe(self, pod_metadata: "PodMetadata | None") -> ProbeResult:
        """Probe pod availability.

        Args:
            pod_metadata: Persisted attempt metadata. ``None`` ⇒
                legacy attempt without metadata, return RUNNING
                (assume the pod is up; if it isn't, the pipeline's
                own SSH connect step will surface the real error).

        Returns:
            :class:`ProbeResult` with mapped availability +
            human-readable message for CLI / UI surface.
        """
        if pod_metadata is None:
            return ProbeResult(
                availability=PodAvailability.RUNNING,
                pod_id="<no-metadata>",
                message=(
                    "Legacy attempt without pod metadata; "
                    "assuming pod is reachable. SSH connect will "
                    "surface real status if it is not."
                ),
            )

        if self._query_pod is None:
            return ProbeResult(
                availability=PodAvailability.PROBE_FAILED,
                pod_id=pod_metadata.pod_id,
                message="No transport configured for probe",
            )

        try:
            pod_data = self._query_pod(pod_metadata.pod_id)
        except Exception as exc:  # noqa: BLE001 — best-effort
            return ProbeResult(
                availability=PodAvailability.PROBE_FAILED,
                pod_id=pod_metadata.pod_id,
                message=f"Probe failed: {exc!r}",
            )

        if not isinstance(pod_data, dict):
            return ProbeResult(
                availability=PodAvailability.PROBE_FAILED,
                pod_id=pod_metadata.pod_id,
                message="Probe returned non-dict payload",
            )

        runpod_status = self._extract_status(pod_data)
        if runpod_status is None:
            # Pod data shape doesn't include ``desiredStatus`` —
            # could be a stale SDK response or "pod not found"
            # marker. Treat as GONE if explicit "not found" signal,
            # else PROBE_FAILED.
            if self._is_gone_marker(pod_data):
                return ProbeResult(
                    availability=PodAvailability.GONE,
                    pod_id=pod_metadata.pod_id,
                    message="Pod has been terminated and is no longer queryable",
                )
            return ProbeResult(
                availability=PodAvailability.PROBE_FAILED,
                pod_id=pod_metadata.pod_id,
                message="Pod data missing desiredStatus field",
            )

        availability = _RUNPOD_STATUS_MAP.get(
            runpod_status.upper(), PodAvailability.PROBE_FAILED,
        )
        return ProbeResult(
            availability=availability,
            pod_id=pod_metadata.pod_id,
            runpod_status=runpod_status,
            message=self._human_message(availability, pod_metadata.pod_id),
        )

    @staticmethod
    def _extract_status(pod_data: dict[str, Any]) -> str | None:
        """Pull RunPod's pod status; tolerate camelCase + snake_case."""
        for key in ("desiredStatus", "desired_status", "status"):
            value = pod_data.get(key)
            if isinstance(value, str) and value:
                return value
        return None

    @staticmethod
    def _is_gone_marker(pod_data: dict[str, Any]) -> bool:
        """Detect "pod terminated / not found" markers in error fields.

        RunPod's get_pod can return ``{}`` or ``{"errors": [...]}``
        when the pod has been terminated. Distinguishing GONE from
        PROBE_FAILED matters for CLI UX: GONE means "show user the
        ``run restart`` option", PROBE_FAILED means "RunPod is
        flapping, retry later".
        """
        errors = pod_data.get("errors")
        if isinstance(errors, list):
            joined = " ".join(str(e).lower() for e in errors)
            if any(m in joined for m in (
                "not found", "does not exist", "no such pod",
                "terminated", "no pod with",
            )):
                return True
        return False

    @staticmethod
    def _human_message(availability: PodAvailability, pod_id: str) -> str:
        """Render a user-facing message for the verdict."""
        if availability == PodAvailability.RUNNING:
            return f"Pod {pod_id} is running"
        if availability == PodAvailability.SLEEPING_RESUMABLE:
            return (
                f"Pod {pod_id} is sleeping (Phase 11.B podStop outcome); "
                "call resume to wake it"
            )
        if availability == PodAvailability.GONE:
            return (
                f"Pod {pod_id} has been terminated; resume in-place is "
                "not possible. Use 'ryotenkai run restart' to recreate "
                "from a checkpoint."
            )
        return f"Pod {pod_id} availability is unknown"


# ---------------------------------------------------------------------------
# Resume with retry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResumeResult:
    """Outcome of ``resume_pod_with_retry``.

    ``ok=True`` ⇒ pod accepted resume request and (per
    ``wait_for_running=True``) is now in RUNNING state.

    ``ok=False`` ⇒ either capacity exhausted (retries done) or
    fatal error (auth, pod gone). ``error_message`` and
    ``capacity_exhausted`` distinguish the two for CLI / UI.
    """
    ok: bool
    pod_id: str
    attempts: int
    elapsed_seconds: float
    capacity_exhausted: bool = False
    error_message: str = ""


async def resume_pod_with_retry(
    pod_id: str,
    *,
    resume_call: Callable[[str], Awaitable[bool] | bool],
    is_capacity_error: Callable[[str], bool] | None = None,
    backoffs: tuple[float, ...] = RESUME_BACKOFFS,
    budget_seconds: float = RESUME_RETRY_BUDGET_SECONDS,
    sleep: "Callable[[float], Awaitable[None]] | None" = None,
    clock: Callable[[], float] | None = None,
) -> ResumeResult:
    """Call ``resume_call``; retry on capacity errors with exp backoff.

    The function is provider-agnostic — pass any callable that knows
    how to issue the resume request. Production wires
    ``RunPodAPIClient.resume_pod`` (sync) wrapped in an executor;
    tests pass a mock.

    Args:
        pod_id: Pod identifier.
        resume_call: ``(pod_id) -> bool|Awaitable[bool]``. Returns
            True on accepted, False on rejected. Raises on transport
            failure. The probe doesn't care which form (sync or
            async) — we await both shapes.
        is_capacity_error: ``(message) -> bool``. Determines whether
            an exception's message indicates capacity exhaustion
            (retry) vs. fatal error (fail fast). ``None`` ⇒ assume
            all errors are fatal (don't retry).
        backoffs: Sleep durations between attempts. Length is the
            max retries. Default :data:`RESUME_BACKOFFS` ⇒ 4 retries.
        budget_seconds: Total wall-clock budget. After expiry, return
            with capacity_exhausted=True regardless of remaining
            backoffs.
        sleep: Async sleep function. Default ``asyncio.sleep``.
            Tests inject a no-op.
        clock: Wall-clock callable; default ``time.monotonic``.

    Returns:
        :class:`ResumeResult` with the outcome.
    """
    sleep_fn = sleep or asyncio.sleep
    clock_fn = clock or time.monotonic

    started = clock_fn()
    last_error: str = ""
    capacity_exhausted = False

    # Initial attempt + len(backoffs) retries = len(backoffs)+1 total.
    for attempt in range(1, len(backoffs) + 2):
        elapsed = clock_fn() - started
        if elapsed >= budget_seconds:
            return ResumeResult(
                ok=False,
                pod_id=pod_id,
                attempts=attempt - 1,
                elapsed_seconds=elapsed,
                capacity_exhausted=capacity_exhausted,
                error_message=(
                    "Resume budget exhausted "
                    f"({budget_seconds:.0f}s); "
                    f"last error: {last_error or 'capacity unavailable'}"
                ),
            )

        try:
            result = resume_call(pod_id)
            if asyncio.iscoroutine(result):
                ok = await result
            else:
                ok = bool(result)
        except Exception as exc:  # noqa: BLE001 — categorise
            msg = str(exc)
            last_error = msg
            if is_capacity_error and is_capacity_error(msg):
                capacity_exhausted = True
                # fall through to retry logic
            else:
                # Fatal — fail fast.
                return ResumeResult(
                    ok=False,
                    pod_id=pod_id,
                    attempts=attempt,
                    elapsed_seconds=clock_fn() - started,
                    capacity_exhausted=False,
                    error_message=msg,
                )
        else:
            if ok:
                return ResumeResult(
                    ok=True,
                    pod_id=pod_id,
                    attempts=attempt,
                    elapsed_seconds=clock_fn() - started,
                )
            # ``resume_call`` returned False without raising —
            # treat as fatal (the call rejected the request without
            # giving us a retry hint).
            last_error = "resume_call returned False"
            return ResumeResult(
                ok=False,
                pod_id=pod_id,
                attempts=attempt,
                elapsed_seconds=clock_fn() - started,
                capacity_exhausted=False,
                error_message=last_error,
            )

        # We're here only if a capacity error was detected.
        # Apply backoff (if there's another attempt left).
        if attempt - 1 < len(backoffs):
            backoff = backoffs[attempt - 1]
            # Don't oversleep past the budget — clamp.
            remaining = budget_seconds - (clock_fn() - started)
            await sleep_fn(min(backoff, max(0.0, remaining)))

    # Loop exhausted without success.
    return ResumeResult(
        ok=False,
        pod_id=pod_id,
        attempts=len(backoffs) + 1,
        elapsed_seconds=clock_fn() - started,
        capacity_exhausted=True,
        error_message=(
            f"Resume capacity unavailable after {len(backoffs) + 1} "
            f"attempts; last error: {last_error}"
        ),
    )


# ---------------------------------------------------------------------------
# Helper — read pod_metadata from a persisted run directory
# ---------------------------------------------------------------------------


def load_pod_metadata_for_run(run_dir: "Path | str") -> "PodMetadata | None":
    """Read the latest attempt's ``pod_metadata`` from ``run_dir``.

    Convenience helper for CLI / Web UI / launch_service callers
    that need the metadata before running :class:`PodAvailabilityProbe`.

    Returns ``None`` when:
    * ``run_dir`` doesn't exist or doesn't contain a state file.
    * The state has no attempts yet.
    * The latest attempt is legacy (no ``pod_metadata`` field).
    * State file is unreadable / corrupt — we treat it as legacy
      rather than crash the resume flow; the caller's downstream
      step will surface the real error.

    Best-effort by design — the probe + resume flow are already
    defensive, so a stale read here just means we'll do an
    optimistic SSH connect (matches legacy behaviour).
    """
    from src.pipeline.state.store import PipelineStateStore

    try:
        store = PipelineStateStore(Path(run_dir).expanduser().resolve())
        state = store.load()
    except Exception:  # noqa: BLE001 — best-effort
        return None

    if not state.attempts:
        return None
    latest = state.attempts[-1]
    return latest.pod_metadata
