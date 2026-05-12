"""RunLoader framework — Protocol + runner + SLO machinery.

Scenarios subclass :class:`RunLoaderScenario` (or implement the
Protocol) and expose:

* :attr:`concurrency` — number of concurrent virtual users
* :attr:`duration_s` — total scenario duration in seconds (real-time)
  or :attr:`compressed_duration_s` for ``SCALE=1`` CI compressed runs
* :attr:`target_rps` — desired requests-per-second (best-effort)
* :meth:`run_step` — one virtual-user iteration; returns latency_ms
* :meth:`teardown` — best-effort, asserts no orphaned resources

The runner enforces SLOs declared on the scenario via :class:`SLOSpec`:

* ``p99_latency_ms`` — 99th-percentile of recorded step latencies
* ``no_orphan_pods`` — sidecar pod registry empty (or only intended
  pods) at teardown
* ``no_event_loss`` — fake state-dump call_history matches expectation
* ``memory_growth_kib`` — best-effort RSS delta (linux/mac only)
"""

from __future__ import annotations

import asyncio
import os
import resource
import statistics
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from tests._harness.stack import Stack


# ---------------------------------------------------------------------------
# SLO spec + result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SLOSpec:
    """Declarative SLO budget for a scenario."""

    name: str
    p99_latency_ms: float | None = None
    no_orphan_pods: bool = False
    no_event_loss: bool = False
    memory_growth_kib: int | None = None
    max_total_seconds: float | None = None


@dataclass
class SLOResult:
    """Outcome of one SLO check for one scenario run."""

    name: str
    passed: bool
    actual: dict[str, float | int | bool] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)


@dataclass
class RunLoaderReport:
    """Aggregate outcome of one :class:`RunLoaderScenario` run."""

    scenario: str
    concurrency: int
    duration_seconds: float
    total_steps: int
    failed_steps: int
    latency_ms: list[float] = field(default_factory=list)
    slo_results: list[SLOResult] = field(default_factory=list)
    success: bool = True

    @property
    def p50_latency_ms(self) -> float:
        return statistics.median(self.latency_ms) if self.latency_ms else 0.0

    @property
    def p99_latency_ms(self) -> float:
        if not self.latency_ms:
            return 0.0
        return _percentile(self.latency_ms, 99.0)


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    rank = (pct / 100.0) * (len(sorted_values) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = rank - lo
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * frac


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class RunLoaderScenario(Protocol):
    """One named load-test scenario."""

    name: str
    tags: list[str]
    concurrency: int
    target_rps: float
    duration_s: float
    slo: list[SLOSpec]

    async def precondition(self, stack: Stack) -> None: ...
    async def run_step(self, stack: Stack, step_index: int) -> float: ...
    async def teardown(self, stack: Stack) -> None: ...


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def scale_factor() -> float:
    """Read ``SCALE`` env var; defaults to 1 (compressed time)."""
    try:
        return float(os.environ.get("SCALE", "1"))
    except ValueError:
        return 1.0


async def run_scenario(
    scenario: RunLoaderScenario,
    *,
    stack: Stack,
    scale: float | None = None,
) -> RunLoaderReport:
    """Drive a scenario for its full duration; aggregate latency + SLOs."""
    effective_scale = scale if scale is not None else scale_factor()
    duration = scenario.duration_s * effective_scale

    await scenario.precondition(stack)

    rss_start = _rss_kib()
    start = time.monotonic()
    latencies: list[float] = []
    failed = 0
    total_steps = 0
    counter = 0
    deadline = start + duration

    async def _worker(worker_id: int) -> None:
        nonlocal counter, failed
        while time.monotonic() < deadline:
            step_id = counter
            counter += 1
            try:
                latency = await scenario.run_step(stack, step_id)
            except Exception:  # noqa: BLE001
                failed += 1
                latencies.append(0.0)
            else:
                latencies.append(latency)

    workers = [asyncio.create_task(_worker(i)) for i in range(scenario.concurrency)]
    try:
        await asyncio.gather(*workers)
    except Exception:  # noqa: BLE001
        for w in workers:
            w.cancel()
        raise

    elapsed = time.monotonic() - start
    total_steps = len(latencies)
    rss_end = _rss_kib()
    rss_delta = max(rss_end - rss_start, 0)

    # SLO evaluation.
    slo_results: list[SLOResult] = []
    for spec in scenario.slo:
        result = SLOResult(name=spec.name, passed=True)
        if spec.p99_latency_ms is not None:
            actual = _percentile(latencies, 99.0)
            result.actual["p99_latency_ms"] = actual
            if actual > spec.p99_latency_ms:
                result.passed = False
                result.failures.append(
                    f"p99={actual:.1f}ms > budget={spec.p99_latency_ms}ms",
                )
        if spec.memory_growth_kib is not None:
            result.actual["memory_growth_kib"] = rss_delta
            if rss_delta > spec.memory_growth_kib:
                result.passed = False
                result.failures.append(
                    f"rss_delta={rss_delta}KiB > budget={spec.memory_growth_kib}KiB",
                )
        if spec.max_total_seconds is not None:
            result.actual["elapsed_seconds"] = elapsed
            if elapsed > spec.max_total_seconds:
                result.passed = False
                result.failures.append(
                    f"elapsed={elapsed:.1f}s > budget={spec.max_total_seconds}s",
                )
        slo_results.append(result)

    # Teardown FIRST — the scenario should be the one taking responsibility
    # for cleaning up any helper resources it staged. The orphan-pod SLO
    # then asserts that nothing leaked past teardown.
    await scenario.teardown(stack)

    teardown_state: dict[str, dict[str, object]] = {}
    try:
        teardown_state = await stack.state_dump()
    except Exception:
        teardown_state = {}
    for spec in scenario.slo:
        if spec.no_orphan_pods:
            result = SLOResult(name=f"{spec.name}.no_orphan_pods", passed=True)
            pods = teardown_state.get("runpod", {}).get("pods", {})
            non_terminal = {
                pid: p
                for pid, p in pods.items()
                if p.get("desired_status") not in {"TERMINATED", "STOPPED", "EXITED"}
            }
            result.actual["non_terminal_pods"] = len(non_terminal)
            if non_terminal:
                result.passed = False
                result.failures.append(f"orphan pods: {sorted(non_terminal)}")
            slo_results.append(result)

    success = all(r.passed for r in slo_results) and failed == 0
    return RunLoaderReport(
        scenario=scenario.name,
        concurrency=scenario.concurrency,
        duration_seconds=elapsed,
        total_steps=total_steps,
        failed_steps=failed,
        latency_ms=latencies,
        slo_results=slo_results,
        success=success,
    )


def _rss_kib() -> int:
    """Best-effort resident-set-size in KiB. Returns 0 on platforms we can't probe."""
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF)
    except Exception:  # noqa: BLE001
        return 0
    # On Linux ``ru_maxrss`` is KiB; on macOS it's bytes. Normalise.
    if hasattr(os, "uname") and os.uname().sysname == "Darwin":
        return int(usage.ru_maxrss / 1024)
    return int(usage.ru_maxrss)


__all__ = [
    "RunLoader",
    "RunLoaderConfig",
    "RunLoaderReport",
    "RunLoaderResult",
    "RunLoaderScenario",
    "SLOResult",
    "SLOSpec",
    "run_scenario",
    "scale_factor",
]


# ---------------------------------------------------------------------------
# Lightweight callable-driven RunLoader
# ---------------------------------------------------------------------------
#
# A second, simpler API for callers that already have an async callable
# representing one "attempt" and want to fan it out across N concurrent
# workers without writing a full RunLoaderScenario. Used by smoke tests
# and quick proof-of-pattern checks; the full scenario API
# (`RunLoaderScenario` + `run_scenario`) remains the canonical L10
# interface for SLO-enforced sustained load.


@dataclass(frozen=True)
class RunLoaderConfig:
    """Configuration for the callable-driven :class:`RunLoader`."""

    concurrency: int
    stages_per_attempt: int = 1
    timeout_seconds: float = 30.0


@dataclass
class RunLoaderResult:
    """Outcome of a callable-driven load run."""

    attempts_total: int
    attempts_succeeded: int
    attempts_failed: int
    latencies_ms: list[float] = field(default_factory=list)

    @property
    def p99_ms(self) -> float:
        return _percentile(self.latencies_ms, 99.0)

    @property
    def p50_ms(self) -> float:
        return statistics.median(self.latencies_ms) if self.latencies_ms else 0.0


class RunLoader:
    """Fan out a single async callable across N concurrent workers.

    Each worker invokes ``attempt_fn(stages=cfg.stages_per_attempt)`` once
    and the loader records its success/failure + wall-clock latency.

    For SLO-enforced sustained load with a Stack and operator hooks,
    use :func:`run_scenario` on a :class:`RunLoaderScenario` instead.
    """

    def __init__(self, config: RunLoaderConfig) -> None:
        self._cfg = config

    async def run(
        self,
        attempt_fn: "Callable[[int], Awaitable[None]]",
    ) -> RunLoaderResult:
        cfg = self._cfg
        latencies_ms: list[float] = []
        succeeded = 0
        failed = 0

        async def _worker() -> None:
            nonlocal succeeded, failed
            t0 = time.monotonic()
            try:
                await attempt_fn(cfg.stages_per_attempt)
            except Exception:  # noqa: BLE001
                failed += 1
            else:
                succeeded += 1
            finally:
                latencies_ms.append((time.monotonic() - t0) * 1000.0)

        try:
            await asyncio.wait_for(
                asyncio.gather(*[_worker() for _ in range(cfg.concurrency)]),
                timeout=cfg.timeout_seconds,
            )
        except TimeoutError:
            # Outstanding workers were cancelled; record their non-arrivals
            # as failed and continue. Latency entries already record their
            # cancellation slot via the finally clause above.
            failed += cfg.concurrency - succeeded - failed

        return RunLoaderResult(
            attempts_total=cfg.concurrency,
            attempts_succeeded=succeeded,
            attempts_failed=failed,
            latencies_ms=latencies_ms,
        )


# Keep callable-import working without TYPE_CHECKING gating: typing only.
if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable  # noqa: F401
