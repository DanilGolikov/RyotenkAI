"""ChaosScenario framework — declarative fault-injection harness.

A scenario is a small object with four async lifecycle methods:

* :meth:`ChaosScenario.precondition` — set up world state (register
  pods, inject baseline state, etc.).
* :meth:`ChaosScenario.inject` — inject the fault.
* :meth:`ChaosScenario.steady_state` — assert the system recovers
  within :attr:`ChaosScenario.recovery_window`. Assertions inside
  ``steady_state`` should use
  :func:`tests._harness.wait.Eventually` /
  :func:`tests._harness.wait.Consistently` rather than ``time.sleep``.
* :meth:`ChaosScenario.cleanup` — best-effort cleanup; never raises.

Registration: scenarios decorate themselves with
:func:`register_scenario`. The catalog test
(``tests/chaos/test_catalog.py``) discovers every registered scenario
and exercises ``precondition + inject + steady_state + cleanup``
against an isolated :class:`Stack`. Individual per-scenario tests
live under ``tests/chaos/scenarios/test_<name>.py`` and import the
scenario class to drive it explicitly with custom assertions.

The framework deliberately does NOT couple to pytest — runs can be
driven from a plain script as well (see ``ScenarioRunner.run``).
Recording integrates with :class:`DebugRecorder` so on-failure
debug bundles can include the chaos timeline.
"""

from __future__ import annotations

import asyncio
import json
import os
import traceback
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from tests._harness.chaos_recorder import DebugRecorder
from tests._harness.clock import Clock, ManualClock, RealClock

if TYPE_CHECKING:
    from tests._harness.stack.orchestrator import Stack


# ---------------------------------------------------------------------------
# Context + Report + Protocol
# ---------------------------------------------------------------------------


@dataclass
class ScenarioContext:
    """State threaded through every step of a single scenario run.

    Attributes:
        stack: live :class:`Stack` instance booted for this scenario.
        clock: in-process clock; mirrors the sidecar clock kind via
            :func:`Stack.advance_clock`.
        seed: deterministic RNG seed for any random choice the
            scenario makes; default ``0``.
        debug_recorder: append-only timeline of events.
        extras: free-form scratch space, e.g. per-scenario pod ids,
            request handles, etc. Not interpreted by the framework.
    """

    stack: Stack
    clock: Clock
    seed: int = 0
    debug_recorder: DebugRecorder = field(default_factory=DebugRecorder)
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScenarioReport:
    """Outcome of a single :meth:`ScenarioRunner.run` call."""

    name: str
    success: bool
    seed: int
    duration_seconds: float
    error: str | None = None
    traceback_lines: list[str] = field(default_factory=list)
    pre_state: dict[str, Any] = field(default_factory=dict)
    post_state: dict[str, Any] = field(default_factory=dict)
    timeline: list[dict[str, Any]] = field(default_factory=list)
    skipped: bool = False
    skip_reason: str | None = None


@runtime_checkable
class ChaosScenario(Protocol):
    """One declarative fault-injection recipe.

    Concrete scenarios live in ``tests/chaos/scenarios/<name>.py`` and
    register themselves via :func:`register_scenario`. They must
    expose:

    * :attr:`name` — short, kebab-or-snake-case, unique across catalog
    * :attr:`tags` — list of tag strings used for filtering
      (``"transient"``, ``"network"``, ``"lifecycle"``, ...)
    * :attr:`recovery_window` — max time the system gets to recover
      after :meth:`inject`; assertions in :meth:`steady_state` should
      not exceed this budget.
    """

    name: str
    tags: list[str]
    recovery_window: timedelta

    async def precondition(self, ctx: ScenarioContext) -> None: ...
    async def inject(self, ctx: ScenarioContext) -> None: ...
    async def steady_state(self, ctx: ScenarioContext) -> None: ...
    async def cleanup(self, ctx: ScenarioContext) -> None: ...


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


_REGISTRY: dict[str, type[ChaosScenario]] = {}


def register_scenario(cls: type[ChaosScenario]) -> type[ChaosScenario]:
    """Decorator. Add a scenario class to the global registry.

    The scenario must expose a unique :attr:`name` class attribute.
    Duplicate registrations fail loudly — keeping the catalog honest.
    """
    name = getattr(cls, "name", None)
    if not isinstance(name, str) or not name:
        raise TypeError(
            f"ChaosScenario subclass {cls.__qualname__} must declare a non-empty "
            f"'name' class attribute",
        )
    if name in _REGISTRY:
        existing = _REGISTRY[name]
        raise RuntimeError(
            f"duplicate ChaosScenario name {name!r}: already registered as "
            f"{existing.__qualname__}; new class {cls.__qualname__}",
        )
    _REGISTRY[name] = cls
    return cls


def all_scenarios() -> list[type[ChaosScenario]]:
    """Return the registry sorted by name (deterministic test parametrize order)."""
    return [_REGISTRY[name] for name in sorted(_REGISTRY)]


def get_scenario(name: str) -> type[ChaosScenario]:
    return _REGISTRY[name]


def filter_by_tag(tag: str) -> list[type[ChaosScenario]]:
    return [cls for cls in all_scenarios() if tag in getattr(cls, "tags", [])]


def _discover_scenarios() -> None:
    """Import every module under ``tests.chaos.scenarios`` to populate the registry.

    Called lazily by :func:`load_catalog` so test-collection time only
    pays the import cost once.
    """
    import importlib
    import pkgutil

    import tests.chaos.scenarios as scenarios_pkg

    for module_info in pkgutil.iter_modules(scenarios_pkg.__path__):
        if module_info.name.startswith("_"):
            continue
        importlib.import_module(f"tests.chaos.scenarios.{module_info.name}")


def load_catalog() -> list[type[ChaosScenario]]:
    """Force-import every scenario module then return the registry."""
    _discover_scenarios()
    return all_scenarios()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class ScenarioRunner:
    """Drive one :class:`ChaosScenario` end-to-end against a live :class:`Stack`.

    The runner:

    * snapshots sidecar state before and after the run
    * captures step-by-step DebugRecorder timeline
    * on failure, writes a JSON debug bundle to
      ``tests/.debug_bundles/<scenario>-<ts>.chaos.json``
    * always invokes :meth:`ChaosScenario.cleanup` regardless of
      success / failure
    """

    def __init__(
        self,
        *,
        bundle_dir: Path | None = None,
        clock: Clock | None = None,
    ) -> None:
        self._bundle_dir = bundle_dir or (
            Path(__file__).resolve().parents[1] / ".debug_bundles"
        )
        self._clock: Clock = clock if clock is not None else RealClock()

    async def run(
        self,
        scenario: ChaosScenario,
        *,
        stack: Stack,
        seed: int = 0,
        clock: Clock | None = None,
    ) -> ScenarioReport:
        """Execute the scenario; never raises, returns the report instead."""
        effective_clock = clock or self._clock
        ctx = ScenarioContext(
            stack=stack,
            clock=effective_clock,
            seed=seed,
            debug_recorder=DebugRecorder(clock=effective_clock),
        )
        start = effective_clock.now()
        pre_state: dict[str, Any] = {}
        post_state: dict[str, Any] = {}
        error: str | None = None
        traceback_lines: list[str] = []
        success = True
        skipped = False
        skip_reason: str | None = None

        # Pre-state snapshot — best-effort.
        try:
            pre_state = await stack.state_dump()
        except Exception as exc:
            pre_state = {"error": repr(exc)}

        try:
            ctx.debug_recorder.record("precondition", "begin", scenario=scenario.name)
            await scenario.precondition(ctx)
            ctx.debug_recorder.record("precondition", "end")

            ctx.debug_recorder.record("inject", "begin")
            await scenario.inject(ctx)
            ctx.debug_recorder.record("inject", "end")

            ctx.debug_recorder.record("steady_state", "begin")
            await scenario.steady_state(ctx)
            ctx.debug_recorder.record("steady_state", "end")
        except BaseException as exc:
            # ``pytest.skip`` raises ``_pytest.outcomes.Skipped`` which
            # subclasses BaseException (NOT Exception). Distinguish it
            # from real failures so partial scenarios don't fail.
            exc_name = type(exc).__name__
            if exc_name == "Skipped":
                skipped = True
                skip_reason = str(exc) or repr(exc)
                ctx.debug_recorder.record(
                    "steady_state", "skipped", reason=skip_reason,
                )
            else:
                success = False
                error = repr(exc)
                traceback_lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
                ctx.debug_recorder.record(
                    "steady_state",
                    "failed",
                    error=error,
                )

        # Post-state snapshot — captured BEFORE cleanup so meta-tests
        # can introspect what the scenario actually observed at the end
        # of ``steady_state``. Cleanup typically wipes sidecar state.
        try:
            post_state = await stack.state_dump()
        except Exception as exc:
            post_state = {"error": repr(exc)}

        try:
            ctx.debug_recorder.record("cleanup", "begin")
            await scenario.cleanup(ctx)
            ctx.debug_recorder.record("cleanup", "end")
        except Exception as exc:
            ctx.debug_recorder.record(
                "cleanup",
                "failed",
                error=repr(exc),
            )

        duration = max(effective_clock.now() - start, 0.0)
        report = ScenarioReport(
            name=scenario.name,
            success=success,
            seed=seed,
            duration_seconds=duration,
            error=error,
            traceback_lines=traceback_lines,
            pre_state=pre_state,
            post_state=post_state,
            timeline=ctx.debug_recorder.to_list(),
            skipped=skipped,
            skip_reason=skip_reason,
        )

        if not success:
            await self._dump_bundle(report)

        return report

    async def _dump_bundle(self, report: ScenarioReport) -> None:
        try:
            self._bundle_dir.mkdir(parents=True, exist_ok=True)
            ts = int(self._clock.now() * 1000)
            path = self._bundle_dir / f"{report.name}-{ts}.chaos.json"
            payload = {
                "name": report.name,
                "seed": report.seed,
                "success": report.success,
                "duration_seconds": report.duration_seconds,
                "error": report.error,
                "traceback": report.traceback_lines,
                "pre_state": report.pre_state,
                "post_state": report.post_state,
                "timeline": report.timeline,
            }
            path.write_text(json.dumps(payload, indent=2, default=str))
        except Exception:
            # Best-effort — never let bundle writes shadow the original failure.
            return


# ---------------------------------------------------------------------------
# Stack helpers (chaos integration)
# ---------------------------------------------------------------------------


_FAULT_TARGETS = ("runpod", "mlflow", "vllm", "hf_hub")
_CHAOS_TARGET_ENDPOINTS = {
    "inject_429": "/control/inject_429",
    "inject_5xx": "/control/inject_5xx",
    "inject_partial_response": "/control/inject_partial_response",
    "set_pod_state": "/control/set_pod_state",
    "register_pod": "/control/register_pod",
    "set_unavailable": "/control/set_unavailable",
    "advance_clock": "/control/advance_clock",
    "inject_latency": "/control/inject_latency",
    "reset": "/control/reset",
}


async def fault_inject(
    stack: Stack,
    target: str,
    fault: dict[str, Any],
) -> dict[str, Any]:
    """Generic dispatcher that POSTs a fault to the right sidecar control endpoint.

    ``target`` is one of ``runpod`` / ``mlflow`` / ``vllm`` / ``hf_hub``.
    ``fault`` has a single key (``"inject_429"``, ``"set_unavailable"``,
    ...) and a value passed as query params. Unknown endpoints raise.

    This is intentionally a thin wrapper — scenarios that need
    fine-grained control should call the per-sidecar
    ``stack._post(...)`` directly. The generic dispatcher exists so
    ad-hoc tests can fault-inject without poking at the orchestrator
    internals.
    """
    if target not in _FAULT_TARGETS:
        raise ValueError(f"unknown fault target {target!r}; expected {_FAULT_TARGETS}")
    if len(fault) != 1:
        raise ValueError(
            f"fault dict must have exactly one key (endpoint name); got {list(fault.keys())}",
        )
    endpoint_name, params = next(iter(fault.items()))
    endpoint = _CHAOS_TARGET_ENDPOINTS.get(endpoint_name)
    if endpoint is None:
        raise ValueError(
            f"unknown fault endpoint {endpoint_name!r}; expected one of "
            f"{sorted(_CHAOS_TARGET_ENDPOINTS)}",
        )
    if not isinstance(params, dict):
        params = {"value": params}
    base = stack.sidecars[target].base_url
    response = await stack._post(base + endpoint, params=params)  # type: ignore[arg-type]
    try:
        return response.json()  # type: ignore[no-any-return]
    except Exception:
        return {"status_code": response.status_code, "text": response.text}


async def run_chaos_scenario(
    stack: Stack,
    scenario: ChaosScenario,
    *,
    seed: int = 0,
    runner: ScenarioRunner | None = None,
) -> ScenarioReport:
    """Drive a scenario end-to-end on the given stack and return its report."""
    effective_runner = runner or ScenarioRunner()
    return await effective_runner.run(scenario, stack=stack, seed=seed)



# Convenience: clock advance helper that also updates an in-process
# ManualClock if it happens to be the test's own. Sidecar manual
# clocks are advanced via the Stack broadcast.
async def advance_clock_everywhere(
    stack: Stack,
    seconds: float,
    *,
    in_process: Clock | None = None,
) -> None:
    """Advance both sidecar clocks (via broadcast) and an optional in-process clock."""
    await stack.advance_clock(seconds)
    if isinstance(in_process, ManualClock):
        in_process.advance(seconds)
    # Yield once so any awaiting sleeper picks up the advance.
    await asyncio.sleep(0)


# ---------------------------------------------------------------------------
# Defaults that scenarios share
# ---------------------------------------------------------------------------


def default_seed() -> int:
    """Return the seed honoured by every scenario unless overridden.

    Hooked from ``RYOTENKAI_TEST_SEED`` so the catalog respects the
    project-wide RNG seed convention. Default is ``0``.
    """
    return int(os.environ.get("RYOTENKAI_TEST_SEED", "0"))


__all__ = [
    "ChaosScenario",
    "DebugRecorder",
    "ScenarioContext",
    "ScenarioReport",
    "ScenarioRunner",
    "advance_clock_everywhere",
    "all_scenarios",
    "default_seed",
    "fault_inject",
    "filter_by_tag",
    "get_scenario",
    "load_catalog",
    "register_scenario",
    "run_chaos_scenario",
]
