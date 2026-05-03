"""FastAPI entry point for the in-pod job server.

Used by ``uvicorn src.runner.main:app`` (the entrypoint.sh inside
the docker image runs exactly this command).

Phase 1 owns:
- ``GET /healthz`` — liveness probe.
- ``GET /readyz`` — readiness probe (true once the FSM has
  ``restore_or_init``-ed).
- ``GET /version`` — pinned image tag for client compatibility checks.
- ``POST /api/v1/jobs`` — multipart submit.
- ``GET  /api/v1/jobs/{job_id}`` — current snapshot.
- ``POST /api/v1/jobs/{job_id}/stop`` — graceful stop request.
- ``WS   /api/v1/jobs/{job_id}/events`` — replay + live event stream.
- ``POST /api/v1/internal/events`` — trainer-side event push (loopback).

Singletons:
The FSM and the EventBus live on ``app.state`` and are constructed
in :func:`_lifespan` so each :class:`fastapi.testclient.TestClient`
spins up its own pair (no cross-test contamination).

The app binds ``127.0.0.1:8080`` per :file:`docker/training/entrypoint.sh`
— never 0.0.0.0. Mac control plane reaches it through ``ssh -L``.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from fastapi import FastAPI

from ryotenkai_shared.constants import RUNTIME_IMAGE
from ryotenkai_pod.runner.api import control as control_api
from ryotenkai_pod.runner.api import events as events_api
from ryotenkai_pod.runner.api import internal as internal_api
from ryotenkai_pod.runner.api import jobs as jobs_api
from ryotenkai_pod.runner.cancellation_telemetry import EVENTS_DISK_PRESSURE
from ryotenkai_pod.runner.event_bus import EventBus
from ryotenkai_pod.runner.event_journal import (
    DEFAULT_FILE_SIZE_CAP,
    DEFAULT_MAX_FILES,
    EventJournal,
)
from ryotenkai_pod.runner.health_reporter import (
    DEFAULT_HEALTH_INTERVAL,
    HealthReporter,
)
from ryotenkai_pod.runner.mlflow_relay import (
    MLflowRelay,
    make_mlflow_forward_fn,
)
from ryotenkai_pod.runner.plugin_unpacker import PluginUnpacker
from ryotenkai_pod.runner.heartbeat import MacHeartbeat
from ryotenkai_pod.runner.pod_terminator import PodTerminator, run_terminal_hook
from ryotenkai_pod.runner.runtime.provider_registry import (
    resolve_keep_on_error_from_env,
    resolve_lifecycle_client_from_env,
    resolve_resource_id_from_env,
    resolve_volume_kind_from_env,
)
from ryotenkai_pod.runner.state import JobLifecycleFSM
from ryotenkai_pod.runner.supervisor import Supervisor, TerminalHook
from ryotenkai_shared.utils.pod_layout import PodLayout

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

API_V1_PREFIX = "/api/v1"


class _SupervisorFactory(Protocol):
    """Constructor signature shared by :class:`Supervisor` and any
    test double. Keep narrow — only the boot-time call signature.

    ``terminal_hook`` is optional: production passes the RunPod
    auto-stop callback; tests omit it (the :class:`MockSupervisor`
    fixture takes a 2-arg call shape).

    ``stdio_log_path`` is optional too: production passes
    ``pod_layout.trainer_stdio_log`` so trainer stdout/stderr lands
    on the per-run pod-side ground-truth file.
    """

    def __call__(
        self,
        fsm: JobLifecycleFSM,
        bus: EventBus,
        *,
        terminal_hook: TerminalHook | None = ...,
        stdio_log_path: Path | None = ...,
    ) -> Supervisor: ...


def _resolve_workspace() -> Path:
    """Per-run workspace directory.

    The runner is launched with ``cwd=<workspace>`` by Mac-side
    ``runner_launcher.launch_runner`` where workspace =
    ``<provider-base>/runs/<run_id>``. Inside the runner we read the
    workspace via env var ``RYOTENKAI_WORKSPACE`` (set by the launcher);
    fallback to ``os.getcwd()`` for safety. The legacy default
    ``/workspace`` (pre-PodLayout / global) is no longer used —
    sequential runs on the same pod each get their own per-run
    directory under ``/workspace/runs/<run_id>``.
    """
    raw = os.environ.get("RYOTENKAI_WORKSPACE")
    if raw:
        return Path(raw)
    return Path(os.getcwd())


def _make_lifespan(supervisor_factory: _SupervisorFactory):  # type: ignore[no-untyped-def]
    """Build a lifespan that uses ``supervisor_factory`` to build the
    Supervisor.

    Tests pass a mock factory through :func:`create_app`; production
    uses :class:`Supervisor` directly. Wrapping in a closure keeps
    the lifespan ``@asynccontextmanager`` shape FastAPI expects
    while still letting us swap implementations.
    """

    @asynccontextmanager
    async def _lifespan(app: FastAPI) -> "AsyncIterator[None]":
        """Wire up FSM + EventBus + Supervisor on ``app.state``.

        Order:
        1. Build FSM at the workspace dir; call ``restore_or_init()``
           so a container restart resumes the previous state (or
           transitions an unsafe state to ``failed`` — see
           :meth:`JobLifecycleFSM.restore_or_init`).
        2. Build EventBus with the env-driven default capacity.
        3. Build :class:`MacHeartbeat` (Phase 11.B) and :class:`PodTerminator`
           (Phase 11.B). The heartbeat tracks Mac control-plane liveness
           via WS yields + REST hits; the terminator picks between
           ``podStop`` (sleep, default for natural completion) and
           ``podTerminate`` (delete, default for user-stop / failed /
           network-volume) per the decision matrix in
           :mod:`src.runner.pod_terminator`. ``RUNPOD_AUTO_STOP`` env
           is removed in Phase 11.B (no toggle — terminate-on-terminal
           is always-on).
        4. Build Supervisor (or test double) bound to (fsm, bus, hook).
        5. Yield — endpoints serve traffic.
        6. On shutdown, stop the trainer first so its SIGTERM-driven
           save can write through, *then* close the bus so the final
           ``trainer_exited`` event reaches subscribers.
        """
        workspace = _resolve_workspace()

        # PodLayout — single source of truth for every pod-side path
        # the runner owns (logs/, events/, state/, output/, ...).
        # Provider-agnostic: workspace is the per-run root supplied
        # by the Mac-side ``runner_launcher`` cwd. PurePosixPath
        # because pod is always Linux.
        from pathlib import PurePosixPath
        pod_layout = PodLayout.from_root(PurePosixPath(str(workspace)))

        # Idempotent directory bootstrap. Creates every subdirectory
        # in the layout (logs/, events/, state/, ...). Best-effort —
        # failures fall through; downstream components (FSM, journal,
        # supervisor) will surface their own errors if directories
        # are still missing after this.
        import subprocess
        with contextlib.suppress(Exception):
            subprocess.run(
                ["sh", "-c", pod_layout.ensure_dirs_command()],
                check=False,
                timeout=10,
            )

        fsm = JobLifecycleFSM(
            workspace_dir=workspace,
            state_dir_override=Path(str(pod_layout.state_dir)),
        )
        fsm.restore_or_init()

        # Phase 12.B — durable event journal under
        # ``<workspace>/events/``. Layout owned by PodLayout so the
        # path stays in sync with the rest of the per-run tree. The
        # bus reconciles its starting offset from the journal's newest
        # persisted record so a runner restart resumes the offset
        # sequence without collisions. If construction fails (read-only
        # fs etc.), we log + fall back to the journal-less behaviour
        # so the runner boots.
        #
        # Phase 14.E (V1) — deferred binding via
        # :meth:`EventBus.attach_journal_rotation_listener`. Replaces
        # the pre-14.E circular-binding-closure pattern (mutable
        # dict cell holding a future ``bus.publish`` reference). The
        # ordering is: build journal → build bus → bus attaches as
        # the journal's rotation observer. No rotations can fire
        # between bus construction and the explicit attach call
        # (rotations are append-driven; bus init does no appends).
        journal: EventJournal | None
        try:
            journal = EventJournal(root_dir=Path(str(pod_layout.events_dir)))
        except Exception as exc:  # noqa: BLE001 — defensive
            import logging
            logging.getLogger(__name__).warning(
                "[LIFESPAN] EventJournal init failed (%s); "
                "falling back to ring-only behaviour", exc,
            )
            journal = None

        bus = EventBus(journal=journal)
        bus.attach_journal_rotation_listener()

        # The plugin unpacker is stateless beyond its workspace path,
        # so we build it once at boot and reuse across requests.
        plugin_unpacker = PluginUnpacker(workspace_dir=workspace)

        # Phase 11.B — heartbeat ledger lives at app.state level so
        # both WS handler (post-yield mark_active) and REST handler
        # (post-response mark_active) share the same instance the
        # PodTerminator reads on terminal hooks.
        heartbeat = MacHeartbeat()

        # Phase 14.B — env-driven lifecycle client resolution. This is
        # the single env-reading seam; the terminator never touches
        # ``os.environ`` again. ``BootstrapConfigError`` propagates
        # so uvicorn exits non-zero on misconfigured env (Phase 14.B
        # § 1.7 — no graceful degradation).
        lifecycle_client = resolve_lifecycle_client_from_env(os.environ)
        pod_terminator = PodTerminator(
            client=lifecycle_client,
            resource_id=resolve_resource_id_from_env(os.environ),
            volume_kind=resolve_volume_kind_from_env(os.environ),
            keep_on_error=resolve_keep_on_error_from_env(os.environ),
        )

        async def _terminal_hook(terminal_state: str) -> None:
            await run_terminal_hook(
                pod_terminator,
                terminal_state=terminal_state,
                heartbeat=heartbeat,
                bus_publish=bus.publish,
            )

        supervisor = supervisor_factory(
            fsm,
            bus,
            terminal_hook=_terminal_hook,
            stdio_log_path=Path(str(pod_layout.trainer_stdio_log)),
        )

        # Optional MLflow relay (Phase 4.3). Activated only when the
        # operator opts in via ``RYOTENKAI_RUNNER_MLFLOW_RELAY=1`` AND
        # an upstream URI is configured. In the default flow trainer
        # talks to MLflow directly through ``ResilientMLflowTransport``;
        # the relay adds a process-independent buffer + circuit
        # breaker for deployments where the trainer cannot reach
        # MLflow directly.
        mlflow_relay = _build_mlflow_relay()
        await mlflow_relay.start()

        # Periodic resource snapshot publisher — emits ``health_snapshot``
        # every :data:`DEFAULT_HEALTH_INTERVAL` seconds so the Mac
        # control plane can render `[MONITOR] ALIVE | …` status lines
        # and feed the live load chart. Without this the bus has no
        # GPU/CPU/RAM source, and the Training Monitor stage logs no
        # status updates between trainer events.
        health_reporter = HealthReporter(
            bus=bus,
            interval=DEFAULT_HEALTH_INTERVAL,
        )
        health_reporter.start()

        app.state.fsm = fsm
        app.state.bus = bus
        app.state.journal = journal  # Phase 12.B (None if init failed)
        app.state.heartbeat = heartbeat  # Phase 11.B
        app.state.pod_terminator = pod_terminator  # Phase 11.B (renamed from pod_stopper)
        app.state.plugin_unpacker = plugin_unpacker
        app.state.supervisor = supervisor
        app.state.mlflow_relay = mlflow_relay
        app.state.health_reporter = health_reporter

        # Phase 12.C — periodic journal health check. Emits
        # ``events_disk_pressure`` when the journal footprint passes
        # 90% of (file_size_cap × max_files). Operator dashboards use
        # the signal to alert on sustained disk-pressure rather than
        # on a single failed write. Running as a background task tied
        # to lifespan; cancelled on shutdown.
        health_task: asyncio.Task[None] | None = None
        if journal is not None:
            health_task = asyncio.create_task(
                _periodic_journal_health_check(bus=bus, journal=journal)
            )

        try:
            yield
        finally:
            if health_task is not None:
                health_task.cancel()
                try:
                    await health_task
                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass
            await health_reporter.stop()
            await mlflow_relay.stop()
            await supervisor.shutdown()
            bus.close()

    return _lifespan


_JOURNAL_HEALTH_INTERVAL_S = 60.0
_JOURNAL_DISK_PRESSURE_THRESHOLD_FRACTION = 0.9


async def _periodic_journal_health_check(
    *,
    bus: EventBus,
    journal: EventJournal,
    interval_s: float = _JOURNAL_HEALTH_INTERVAL_S,
    threshold_fraction: float = _JOURNAL_DISK_PRESSURE_THRESHOLD_FRACTION,
) -> None:
    """Phase 12.C — background task that monitors journal footprint.

    Every ``interval_s`` seconds, computes ``total_bytes / cap``
    where cap = ``file_size_cap × max_files``. When the ratio
    crosses ``threshold_fraction`` (default 90%), publishes an
    ``events_disk_pressure`` event. Each crossing fires once;
    while we stay over the threshold no further events are emitted
    until the bus' rate-limit interval (1 min from
    :meth:`EventBus._signal_disk_pressure`) elapses — but that
    rate limit is for the WRITE-failure path; this health check
    is fire-and-forget.

    Cancellation: this is run as ``asyncio.create_task`` from the
    lifespan and cancelled on shutdown. Catches CancelledError to
    exit cleanly without leaking warnings.
    """
    cap_bytes = DEFAULT_FILE_SIZE_CAP * DEFAULT_MAX_FILES
    threshold_bytes = int(cap_bytes * threshold_fraction)
    last_alerted = False
    while True:
        try:
            await asyncio.sleep(interval_s)
        except asyncio.CancelledError:
            return
        try:
            total = journal.total_bytes()
            files = journal.file_count()
        except Exception:  # noqa: BLE001 — defensive
            continue
        if total > threshold_bytes:
            if not last_alerted:
                try:
                    bus.publish(EVENTS_DISK_PRESSURE, {
                        "total_bytes": total,
                        "file_count": files,
                        "threshold_bytes": threshold_bytes,
                        "cap_bytes": cap_bytes,
                    })
                except Exception:  # noqa: BLE001 — defensive
                    pass
                last_alerted = True
        else:
            last_alerted = False


def _build_mlflow_relay() -> MLflowRelay:
    """Build the MLflow relay based on env-driven configuration.

    Activation contract:

    - ``RYOTENKAI_RUNNER_MLFLOW_RELAY``  (1/true/on) — opt-in toggle.
      Defaults to *off*, so existing deployments keep talking to
      MLflow directly through the trainer's own resilient transport.
    - ``MLFLOW_TRACKING_URI`` — required when relay is enabled;
      missing/empty disables the relay even if the toggle is on.

    A disabled relay still exists as a no-op object so endpoints
    that depend on ``app.state.mlflow_relay`` don't have to gate
    every call. ``submit()`` returns ``False`` and ``start()`` /
    ``stop()`` are no-ops.
    """
    enabled = os.environ.get(
        "RYOTENKAI_RUNNER_MLFLOW_RELAY", "",
    ).strip().lower() in {"1", "true", "on", "yes"}
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "").strip()

    if not enabled or not tracking_uri:
        return MLflowRelay(forward_fn=None)

    forward_fn = make_mlflow_forward_fn(tracking_uri)
    return MLflowRelay(forward_fn=forward_fn)


def create_app(
    supervisor_factory: _SupervisorFactory | None = None,
) -> FastAPI:
    """Build a fresh app instance.

    ``supervisor_factory`` is a callable ``(fsm, bus) -> Supervisor``
    used to construct the supervisor in the lifespan. Defaults to
    the real :class:`Supervisor`. Tests pass :class:`MockSupervisor`
    (or any other test double matching the protocol) for unit
    coverage of the API layer without paying for real subprocess
    semantics.
    """
    factory: _SupervisorFactory = supervisor_factory or Supervisor
    app = FastAPI(
        title="RyotenkAI Runner",
        version=RUNTIME_IMAGE,
        description=(
            "In-pod control plane for remote training. "
            "Loopback-only (127.0.0.1:8080); reached from Mac via "
            "ssh -L tunnel."
        ),
        lifespan=_make_lifespan(factory),
    )

    @app.get("/healthz", tags=["meta"])
    def healthz() -> dict[str, str]:
        """Liveness — always 200 while the process is up."""
        return {"status": "ok"}

    @app.get("/readyz", tags=["meta"])
    def readyz() -> dict[str, object]:
        """Readiness — FSM has restored, bus is open, supervisor wired."""
        bus_open = not app.state.bus.is_closed
        return {
            "status": "ready" if bus_open else "draining",
            "bus_open": bus_open,
            "fsm_restored": True,  # restore_or_init() ran in lifespan
            "supervisor_running": app.state.supervisor.is_running,
        }

    @app.get("/version", tags=["meta"])
    def version() -> dict[str, str]:
        """Pinned image identity for client compatibility checks."""
        return {"image": RUNTIME_IMAGE}

    app.include_router(jobs_api.router, prefix=API_V1_PREFIX)
    app.include_router(internal_api.router, prefix=API_V1_PREFIX)
    app.include_router(events_api.router, prefix=API_V1_PREFIX)
    # Phase 11.E — control-plane heartbeat surface
    app.include_router(control_api.router, prefix=API_V1_PREFIX)

    return app


app = create_app()


__all__ = ["API_V1_PREFIX", "app", "create_app"]
