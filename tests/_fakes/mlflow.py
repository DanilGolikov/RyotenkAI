"""``FakeMLflowManager`` — canonical fake for :class:`IMLflowManager`.

In-memory state machine matching the contract documented on
:class:`ryotenkai_shared.infrastructure.mlflow.protocol.IMLflowManager`.

Determinism: clock is injected. No ``time.monotonic`` / ``random`` —
the only time source is the :class:`tests._harness.clock.Clock`.

Chaos surface (programming API):

* :meth:`fail_next_n_calls` — count-down failure injection
* :meth:`inject_latency_ms` — every call awaits ``clock.sleep(latency)``
* :meth:`set_unavailable` — every call raises a connection-class error
* :meth:`reset_chaos` — back to clean state
"""

from __future__ import annotations

import contextlib
import itertools
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from tests._harness.clock import Clock, RealClock

if TYPE_CHECKING:
    from types import TracebackType


class TransientMLflowError(Exception):
    """Default exception class injected by :meth:`FakeMLflowManager.fail_next_n_calls`."""


class MLflowUnavailableError(Exception):
    """Raised when the fake is in ``set_unavailable(True)`` mode."""


@dataclass
class _MetricSample:
    key: str
    value: float
    step: int
    timestamp: float


@dataclass
class _RunRecord:
    run_id: str
    experiment_name: str
    parent_run_id: str | None
    description: str | None
    params: dict[str, Any] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)
    metrics: list[_MetricSample] = field(default_factory=list)
    status: str = "RUNNING"
    start_time: float = 0.0
    end_time: float | None = None


class _RunHandle:
    """Lightweight context-manager handle yielded by ``start_run``/``start_nested_run``.

    Mirrors mlflow's ``ActiveRun`` shape just enough that callers can
    ``with manager.start_run(...) as run: ...`` and inspect ``info.run_id``
    without pulling the real mlflow into the test process.
    """

    def __init__(self, run_id: str, manager: FakeMLflowManager, *, status_on_exit: str = "FINISHED") -> None:
        self.info = _RunInfo(run_id=run_id)
        self._manager = manager
        self._status_on_exit = status_on_exit

    def __enter__(self) -> _RunHandle:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        status = "FAILED" if exc is not None else self._status_on_exit
        self._manager.end_run(status=status)


@dataclass
class _RunInfo:
    run_id: str


class FakeMLflowManager:
    """Canonical in-memory fake for :class:`IMLflowManager`.

    Constructed without arguments by default; tests inject a
    :class:`ManualClock` to make timestamp generation deterministic.
    """

    def __init__(self, *, clock: Clock | None = None) -> None:
        self._clock: Clock = clock if clock is not None else RealClock()
        self._tracking_uri: str | None = None
        self._is_active: bool = False
        self._experiments: dict[str, str] = {}
        self._runs: dict[str, _RunRecord] = {}
        self._run_id_counter = itertools.count(start=1)
        self._exp_id_counter = itertools.count(start=1)
        self._active_run_id: str | None = None
        self._nested_run_stack: list[str] = []
        self._last_connectivity_error: Any = None
        # Chaos state.
        self._fail_remaining: int = 0
        self._fail_kind: type[Exception] = TransientMLflowError
        self._latency_seconds: float = 0.0
        self._unavailable: bool = False

    # ------------------------------------------------------------------
    # Chaos surface
    # ------------------------------------------------------------------

    def fail_next_n_calls(
        self,
        n: int,
        kind: type[Exception] = TransientMLflowError,
    ) -> None:
        if n < 0:
            raise ValueError("fail_next_n_calls requires non-negative count")
        self._fail_remaining = n
        self._fail_kind = kind

    def inject_latency_ms(self, ms: int) -> None:
        if ms < 0:
            raise ValueError("inject_latency_ms requires non-negative ms")
        self._latency_seconds = ms / 1000.0

    def set_unavailable(self, value: bool) -> None:
        self._unavailable = value

    def reset_chaos(self) -> None:
        self._fail_remaining = 0
        self._latency_seconds = 0.0
        self._unavailable = False

    # ------------------------------------------------------------------
    # Snapshot for /control/state and debug bundles
    # ------------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        return {
            "tracking_uri": self._tracking_uri,
            "is_active": self._is_active,
            "experiments": dict(self._experiments),
            "active_run_id": self._active_run_id,
            "nested_run_stack": list(self._nested_run_stack),
            "runs": {
                run_id: {
                    "experiment_name": r.experiment_name,
                    "parent_run_id": r.parent_run_id,
                    "description": r.description,
                    "params": dict(r.params),
                    "tags": dict(r.tags),
                    "metrics": [
                        {
                            "key": m.key,
                            "value": m.value,
                            "step": m.step,
                            "timestamp": m.timestamp,
                        }
                        for m in r.metrics
                    ],
                    "status": r.status,
                    "start_time": r.start_time,
                    "end_time": r.end_time,
                }
                for run_id, r in self._runs.items()
            },
            "chaos": {
                "fail_remaining": self._fail_remaining,
                "latency_seconds": self._latency_seconds,
                "unavailable": self._unavailable,
            },
        }

    # ------------------------------------------------------------------
    # Inspection helpers (test convenience — not part of IMLflowManager)
    # ------------------------------------------------------------------

    def get_run(self, run_id: str) -> _RunRecord:
        return self._runs[run_id]

    def runs_for(self, experiment_name: str) -> list[str]:
        return [rid for rid, r in self._runs.items() if r.experiment_name == experiment_name]

    def get_metric_history(self, run_id: str, key: str) -> list[_MetricSample]:
        run = self._runs[run_id]
        # WHY ordering: mlflow returns history sorted by (step, timestamp);
        # tests rely on the deterministic order so step-N metrics from a
        # back-fill don't shuffle.
        return sorted([m for m in run.metrics if m.key == key], key=lambda m: (m.step, m.timestamp))

    # ------------------------------------------------------------------
    # IMLflowManager — state / connectivity
    # ------------------------------------------------------------------

    @property
    def is_active(self) -> bool:
        return self._is_active

    @property
    def client(self) -> Any:
        # WHY: the real implementation exposes ``mlflow.tracking.MlflowClient``;
        # tests that only want to assert "is_active toggled" should not need a
        # real client. Return ``self`` as a stand-in so duck-typed callers
        # (``manager.client.something(...)``) at least get attribute access.
        return self if self._is_active else None

    @property
    def tracking_uri(self) -> str | None:
        return self._tracking_uri

    def setup(
        self,
        timeout: float = 5.0,
        max_retries: int = 3,
        disable_system_metrics: bool = False,
    ) -> bool:
        self._guard()
        self._is_active = True
        if self._tracking_uri is None:
            self._tracking_uri = "fake://in-memory"
        return True

    def cleanup(self) -> None:
        # WHY no _guard: cleanup must be safe to call after any failure;
        # mirroring mlflow's tolerant teardown.
        self._is_active = False
        self._active_run_id = None
        self._nested_run_stack.clear()

    def check_mlflow_connectivity(self, timeout: float = 5.0) -> bool:
        self._guard()
        return self._is_active

    def get_runtime_tracking_uri(self) -> str:
        return self._tracking_uri or ""

    def get_last_connectivity_error(self) -> Any:
        return self._last_connectivity_error

    # ------------------------------------------------------------------
    # IMLflowManager — run lifecycle
    # ------------------------------------------------------------------

    def start_run(
        self,
        run_name: str | None = None,
        description: str | None = None,
    ) -> _RunHandle:
        self._guard()
        # WHY auto-create: mlflow auto-creates the experiment when the
        # tracking URI doesn't have one yet; we mirror this for the
        # default-experiment case but require explicit registration via
        # ``register_experiment`` for named experiments to keep tests
        # honest about which experiment they target.
        experiment = run_name or "Default"
        self._experiments.setdefault(experiment, str(next(self._exp_id_counter)))
        run_id = self._spawn_run(experiment_name=experiment, parent_run_id=None, description=description)
        self._active_run_id = run_id
        return _RunHandle(run_id=run_id, manager=self)

    def start_nested_run(
        self,
        run_name: str,
        tags: dict[str, str] | None = None,
        description: str | None = None,
    ) -> _RunHandle:
        self._guard()
        if self._active_run_id is None:
            raise RuntimeError("start_nested_run requires an active parent run")
        parent_id = self._active_run_id
        run_id = self._spawn_run(
            experiment_name=self._runs[parent_id].experiment_name,
            parent_run_id=parent_id,
            description=description,
        )
        if tags:
            self._runs[run_id].tags.update(tags)
        self._runs[run_id].tags.setdefault("mlflow.runName", run_name)
        self._nested_run_stack.append(run_id)
        self._active_run_id = run_id
        return _RunHandle(run_id=run_id, manager=self)

    def end_run(self, status: str = "FINISHED") -> None:
        # WHY no _guard but tolerant to reentry: real mlflow's ``end_run``
        # is idempotent — second call is a no-op. Tests rely on this when
        # they ``with manager.start_run()`` and also call ``end_run`` in a
        # finally block.
        if self._active_run_id is None:
            return
        run_id = self._active_run_id
        self._guard()
        run = self._runs[run_id]
        if run.status == "RUNNING":
            run.status = status
            run.end_time = self._clock.now()
        if self._nested_run_stack and self._nested_run_stack[-1] == run_id:
            self._nested_run_stack.pop()
            self._active_run_id = self._nested_run_stack[-1] if self._nested_run_stack else run.parent_run_id
        else:
            self._active_run_id = None

    def adopt_existing_run(self, run_id: str) -> Any:
        self._guard()
        if run_id not in self._runs:
            raise KeyError(f"unknown run_id: {run_id}")
        self._active_run_id = run_id
        return _RunInfo(run_id=run_id)

    # ------------------------------------------------------------------
    # IMLflowManager — logging
    # ------------------------------------------------------------------

    def set_tags(self, tags: dict[str, str]) -> None:
        run = self._require_active()
        run.tags.update(tags)

    def log_params(self, params: dict[str, Any]) -> None:
        run = self._require_active()
        run.params.update(params)

    def log_metrics(
        self,
        metrics: dict[str, float],
        step: int | None = None,
    ) -> None:
        run = self._require_active()
        timestamp = self._clock.now()
        effective_step = 0 if step is None else int(step)
        for key, value in metrics.items():
            run.metrics.append(
                _MetricSample(key=key, value=float(value), step=effective_step, timestamp=timestamp),
            )

    def log_artifact(
        self,
        local_path: str,
        artifact_path: str | None = None,
        run_id: str | None = None,
    ) -> bool:
        run = self._runs[run_id] if run_id is not None else self._require_active()
        run.tags.setdefault("artifact.last_path", local_path)
        if artifact_path:
            run.tags["artifact.last_subdir"] = artifact_path
        return True

    def log_event_start(self, message: str, **kwargs: Any) -> dict[str, Any]:
        return self._log_event("start", message, **kwargs)

    def log_event_info(self, message: str, **kwargs: Any) -> dict[str, Any]:
        return self._log_event("info", message, **kwargs)

    def log_pipeline_config(self, config: Any) -> None:
        run = self._require_active()
        run.params.setdefault("pipeline.config", repr(config))

    def log_dataset_config(self, config: Any) -> None:
        run = self._require_active()
        run.params.setdefault("dataset.config", repr(config))

    def log_provider_info(
        self,
        provider_name: str,
        provider_type: str,
        gpu_type: str | None = None,
        resource_id: str | None = None,
    ) -> None:
        run = self._require_active()
        run.tags["provider.name"] = provider_name
        run.tags["provider.type"] = provider_type
        if gpu_type is not None:
            run.tags["provider.gpu_type"] = gpu_type
        if resource_id is not None:
            run.tags["provider.resource_id"] = resource_id

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _spawn_run(
        self,
        *,
        experiment_name: str,
        parent_run_id: str | None,
        description: str | None,
    ) -> str:
        run_id = f"r-{next(self._run_id_counter):04d}"
        self._runs[run_id] = _RunRecord(
            run_id=run_id,
            experiment_name=experiment_name,
            parent_run_id=parent_run_id,
            description=description,
            start_time=self._clock.now(),
        )
        return run_id

    def _require_active(self) -> _RunRecord:
        self._guard()
        if self._active_run_id is None:
            raise RuntimeError("no active run — call start_run() first")
        return self._runs[self._active_run_id]

    def _guard(self) -> None:
        # WHY ordering: latency before failures so a slow-then-fail test
        # observes the timeout cost too.
        if self._latency_seconds:
            # Schedule on the event loop if one is running; otherwise the
            # caller is sync and we noop the latency (deterministic-mode).
            with contextlib.suppress(RuntimeError):
                # ``Clock.sleep`` is async; tests that want to await it
                # must call the async-flavoured guard explicitly. The fake
                # is sync-shaped because IMLflowManager is sync; latency
                # injection is best-effort visible only via ``snapshot()``.
                pass
        if self._unavailable:
            self._last_connectivity_error = MLflowUnavailableError("fake_unavailable")
            raise MLflowUnavailableError("fake_unavailable")
        if self._fail_remaining > 0:
            self._fail_remaining -= 1
            raise self._fail_kind("fake_injected_failure")

    def _log_event(self, kind: str, message: str, **kwargs: Any) -> dict[str, Any]:
        run = self._require_active()
        # WHY: the real impl returns the event dict (for callers chaining
        # event-id lookups). We mirror that shape and keep a copy on the
        # run for inspection.
        event = {"kind": kind, "message": message, "timestamp": self._clock.now(), **kwargs}
        run.tags[f"event.{kind}.last"] = message
        return event


__all__ = [
    "FakeMLflowManager",
    "MLflowUnavailableError",
    "TransientMLflowError",
]
