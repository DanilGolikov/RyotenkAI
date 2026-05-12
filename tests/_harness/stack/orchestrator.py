"""Hermetic stack orchestrator (the "Gauntlet").

Boots fake-* sidecars as Python subprocesses bound to ephemeral ports,
broadcasts ``/control/advance_clock`` so per-sidecar :class:`ManualClock`
instances stay in lock-step, and (via context managers) wires the real
control plane / runner subprocesses to point at the sidecar URLs.

Sidecars are processes (not Docker containers) so the harness has no
docker dependency. Real network is still real — every call goes via
``127.0.0.1`` HTTP. A separate :file:`docker-compose.yml` exists for the
``make start-stack`` dev workflow; both paths share the same sidecar
Python entrypoints.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import httpx

from tests._harness.stack._context import current_stack
from tests._harness.stack.ports import allocate_port_block
from tests._harness.stack.process import (
    ManagedProcess,
    python_executable,
    spawn,
    wait_for_health,
)

_DEFAULT_SERVICES: tuple[str, ...] = ("runpod", "mlflow", "vllm", "hf_hub")
_LOG_ROOT = Path(__file__).resolve().parents[2] / ".stack_logs"
_REPO_ROOT = Path(__file__).resolve().parents[3]


def _python_path_for_subprocess() -> str:
    """Build PYTHONPATH so subprocess imports pick up worktree-local packages.

    The conftest hacks sys.path at test-collect time to prefer the
    worktree's ``packages/*/src`` over the editable install (which
    resolves to the parent worktree). Subprocesses inherit none of that —
    we explicitly prepend the same src dirs to PYTHONPATH.
    """
    parts = [str(_REPO_ROOT)]
    parts.extend(str(p) for p in sorted((_REPO_ROOT / "packages").glob("*/src")))
    existing = os.environ.get("PYTHONPATH")
    if existing:
        parts.append(existing)
    return os.pathsep.join(parts)


@dataclass
class _SidecarHandle:
    name: str
    port: int
    process: ManagedProcess
    base_url: str


@dataclass
class Stack:
    """Per-test instance of the hermetic sidecar fleet."""

    sidecars: dict[str, _SidecarHandle] = field(default_factory=dict)
    log_dir: Path = field(default_factory=lambda: _LOG_ROOT / "default")
    clock_kind: Literal["real", "manual"] = "manual"
    _booted: bool = False
    _shutdown: bool = False
    _client: httpx.AsyncClient | None = None
    _stack_token: Any = None

    # ------------------------------------------------------------------
    # Boot / shutdown
    # ------------------------------------------------------------------

    @classmethod
    async def boot(
        cls,
        *,
        clock: Literal["real", "manual"] = "manual",
        services: list[str] | None = None,
        log_dir: Path | None = None,
        env: dict[str, str] | None = None,
    ) -> Stack:
        names = tuple(services) if services else _DEFAULT_SERVICES
        log_dir = log_dir or (_LOG_ROOT / f"session-{os.getpid()}")
        log_dir.mkdir(parents=True, exist_ok=True)
        ports = allocate_port_block(len(names))

        stack = cls(log_dir=log_dir, clock_kind=clock)

        spawn_tasks: list[asyncio.Task[ManagedProcess]] = []
        for name, port in zip(names, ports, strict=True):
            spawn_tasks.append(
                asyncio.create_task(stack._spawn_sidecar(name=name, port=port, clock=clock, env=env)),
            )

        try:
            processes = await asyncio.gather(*spawn_tasks)
        except BaseException:
            # Best-effort: kill anything that did boot successfully so we
            # don't leak ports/processes into the next test.
            for task in spawn_tasks:
                if task.done() and not task.exception():
                    proc = task.result()
                    with contextlib.suppress(Exception):
                        await proc.shutdown()
            raise

        # Wait for /health on each in parallel.
        health_tasks = [
            asyncio.create_task(
                wait_for_health(
                    f"http://127.0.0.1:{port}/health",
                    timeout=20.0,
                    process=proc,
                ),
            )
            for proc, port in zip(processes, ports, strict=True)
        ]
        try:
            await asyncio.gather(*health_tasks)
        except BaseException:
            for proc in processes:
                with contextlib.suppress(Exception):
                    await proc.shutdown()
            raise

        for name, port, proc in zip(names, ports, processes, strict=True):
            stack.sidecars[name] = _SidecarHandle(
                name=name,
                port=port,
                process=proc,
                base_url=f"http://127.0.0.1:{port}",
            )

        stack._client = httpx.AsyncClient(timeout=10.0)
        stack._booted = True
        stack._stack_token = current_stack.set(stack)
        return stack

    async def _spawn_sidecar(
        self,
        *,
        name: str,
        port: int,
        clock: Literal["real", "manual"],
        env: dict[str, str] | None,
    ) -> ManagedProcess:
        module = f"tests._harness.stack.sidecars.{name}_server"
        log_path = self.log_dir / f"{name}.log"
        cmd = [python_executable(), "-m", module, "--port", str(port), "--clock", clock]
        sidecar_env = {
            "PYTHONPATH": _python_path_for_subprocess(),
            "RYOTENKAI_TEST_CLOCK": clock,
        }
        if env:
            sidecar_env.update(env)
        return await spawn(name=name, cmd=cmd, log_path=log_path, env=sidecar_env, cwd=_REPO_ROOT)

    async def shutdown(self) -> None:
        # WHY idempotent: tests put this in a finally block which may run
        # after partial boot failures; double-shutdown must be safe.
        if self._shutdown:
            return
        self._shutdown = True

        if self._stack_token is not None:
            with contextlib.suppress(Exception):
                current_stack.reset(self._stack_token)
            self._stack_token = None

        if self._client is not None:
            with contextlib.suppress(Exception):
                await self._client.aclose()
            self._client = None

        await asyncio.gather(
            *[handle.process.shutdown(grace_seconds=3.0) for handle in self.sidecars.values()],
            return_exceptions=True,
        )

    # ------------------------------------------------------------------
    # Broadcast helpers
    # ------------------------------------------------------------------

    async def advance_clock(self, seconds: float) -> None:
        if self.clock_kind != "manual":
            raise RuntimeError(
                "advance_clock requires Stack.boot(clock='manual')",
            )
        await asyncio.gather(
            *[
                self._post(handle.base_url + "/control/advance_clock", params={"seconds": seconds})
                for handle in self.sidecars.values()
            ],
        )

    async def state_dump(self) -> dict[str, dict[str, Any]]:
        results: dict[str, dict[str, Any]] = {}
        async with httpx.AsyncClient(timeout=5.0) as client:
            tasks = [
                asyncio.create_task(client.get(handle.base_url + "/control/state"))
                for handle in self.sidecars.values()
            ]
            responses = await asyncio.gather(*tasks, return_exceptions=True)
        for handle, response in zip(self.sidecars.values(), responses, strict=True):
            if isinstance(response, BaseException):
                results[handle.name] = {"error": repr(response)}
                continue
            results[handle.name] = response.json()
        return results

    async def reset(self) -> None:
        await asyncio.gather(
            *[self._post(handle.base_url + "/control/reset") for handle in self.sidecars.values()],
            return_exceptions=True,
        )

    # ------------------------------------------------------------------
    # Chaos helpers (Phase 5)
    # ------------------------------------------------------------------

    async def fault_inject(self, target: str, fault: dict[str, Any]) -> dict[str, Any]:
        """Dispatch a fault to a sidecar's ``/control/*`` endpoint.

        Thin wrapper over
        :func:`tests._harness.chaos.fault_inject` so scenarios can
        write ``await stack.fault_inject("runpod", {"inject_429": 5})``
        instead of poking at the orchestrator internals.
        """
        # Imported lazily to avoid an import cycle: chaos.py imports
        # the orchestrator type for type-checking.
        from tests._harness.chaos import fault_inject as _fault_inject

        return await _fault_inject(self, target, fault)

    async def run_chaos_scenario(
        self,
        scenario: Any,
        *,
        seed: int = 0,
    ) -> Any:
        """Drive a ChaosScenario end-to-end against this stack.

        Returns a :class:`tests._harness.chaos.ScenarioReport`.
        """
        from tests._harness.chaos import run_chaos_scenario as _run

        return await _run(self, scenario, seed=seed)

    async def _post(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
    ) -> httpx.Response:
        if self._client is None:
            raise RuntimeError("Stack is not booted (or already shut down)")
        return await self._client.post(url, params=params, json=json)

    # ------------------------------------------------------------------
    # URL accessors
    # ------------------------------------------------------------------

    @property
    def runpod_url(self) -> str:
        return self.sidecars["runpod"].base_url

    @property
    def mlflow_url(self) -> str:
        return self.sidecars["mlflow"].base_url

    @property
    def vllm_url(self) -> str:
        return self.sidecars["vllm"].base_url

    @property
    def hf_hub_url(self) -> str:
        return self.sidecars["hf_hub"].base_url

    # ------------------------------------------------------------------
    # Real subprocess wrappers
    # ------------------------------------------------------------------

    @asynccontextmanager
    async def control_plane(
        self,
        *,
        env: dict[str, str] | None = None,
        runs_dir: Path | None = None,
    ) -> AsyncIterator[str]:
        """Boot the real FastAPI control plane subprocess pointed at the sidecars."""
        if not self._booted:
            raise RuntimeError("Stack must be booted before control_plane()")

        port = allocate_port_block(1)[0]
        runs_dir = runs_dir or (self.log_dir / "control-runs")
        runs_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            python_executable(),
            "-m",
            "uvicorn",
            "ryotenkai_control.api.main:create_app",
            "--factory",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ]
        log_path = self.log_dir / "control_plane.log"
        full_env: dict[str, str] = {
            "PYTHONPATH": _python_path_for_subprocess(),
            "RYOTENKAI_API_HOST": "127.0.0.1",
            "RYOTENKAI_API_PORT": str(port),
            "RYOTENKAI_API_RUNS_DIR": str(runs_dir),
            # Stack-aware env so additive code paths can pick the sidecars
            # up — control plane today doesn't read these but we publish
            # them so Phase 1+ code can opt in incrementally.
            "RYOTENKAI_RUNPOD_API_BASE_URL": self.runpod_url,
            "MLFLOW_TRACKING_URI": self.mlflow_url,
            "RYOTENKAI_VLLM_BASE_URL": self.vllm_url,
            "HF_ENDPOINT": self.hf_hub_url,
        }
        if env:
            full_env.update(env)

        process = await spawn(
            name="control_plane",
            cmd=cmd,
            log_path=log_path,
            env=full_env,
            cwd=_REPO_ROOT,
        )
        try:
            await wait_for_health(
                f"http://127.0.0.1:{port}/api/v1/health",
                timeout=30.0,
                process=process,
            )
            yield f"http://127.0.0.1:{port}"
        finally:
            await process.shutdown(grace_seconds=5.0)

    @asynccontextmanager
    async def runner(
        self,
        *,
        env: dict[str, str] | None = None,
    ) -> AsyncIterator[str]:
        """Boot the real runner FastAPI subprocess pointed at the sidecars.

        The runner expects RUNPOD_API_KEY etc; we stub deterministic
        secrets and point the lifecycle/MLflow URLs at the sidecars.
        """
        if not self._booted:
            raise RuntimeError("Stack must be booted before runner()")

        port = allocate_port_block(1)[0]
        cmd = [
            python_executable(),
            "-m",
            "uvicorn",
            "ryotenkai_pod.runner.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ]
        log_path = self.log_dir / "runner.log"
        full_env: dict[str, str] = {
            "PYTHONPATH": _python_path_for_subprocess(),
            "RUNPOD_API_KEY": "fake-runpod-key",
            "RUNPOD_POD_ID": "fake-pod-1",
            "MLFLOW_TRACKING_URI": self.mlflow_url,
            "RYOTENKAI_RUNPOD_API_BASE_URL": self.runpod_url,
            "HF_ENDPOINT": self.hf_hub_url,
        }
        if env:
            full_env.update(env)

        process = await spawn(
            name="runner",
            cmd=cmd,
            log_path=log_path,
            env=full_env,
            cwd=_REPO_ROOT,
        )
        try:
            await wait_for_health(
                f"http://127.0.0.1:{port}/healthz",
                timeout=30.0,
                process=process,
            )
            yield f"http://127.0.0.1:{port}"
        finally:
            await process.shutdown(grace_seconds=3.0)


# ---------------------------------------------------------------------------
# CLI entry point — `python -m tests._harness.stack.orchestrator ...`
# ---------------------------------------------------------------------------


_DEV_PIDFILE = _REPO_ROOT / "tests" / ".stack_logs" / "dev.pid"
_DEV_STATE_FILE = _REPO_ROOT / "tests" / ".stack_logs" / "dev.json"
_DEV_PORTS = {
    "runpod": 18091,
    "mlflow": 18092,
    "vllm": 18093,
    "hf_hub": 18094,
}


async def _start_dev() -> int:
    import json

    log_dir = _LOG_ROOT / "dev"
    log_dir.mkdir(parents=True, exist_ok=True)
    procs: dict[str, ManagedProcess] = {}
    state: dict[str, Any] = {"pid": os.getpid(), "ports": dict(_DEV_PORTS)}
    for name, port in _DEV_PORTS.items():
        module = f"tests._harness.stack.sidecars.{name}_server"
        cmd = [python_executable(), "-m", module, "--port", str(port), "--clock", "real"]
        proc = await spawn(
            name=name,
            cmd=cmd,
            log_path=log_dir / f"{name}.log",
            env={"PYTHONPATH": _python_path_for_subprocess()},
            cwd=_REPO_ROOT,
        )
        procs[name] = proc

    try:
        await asyncio.gather(
            *[
                wait_for_health(f"http://127.0.0.1:{port}/health", timeout=20.0, process=procs[name])
                for name, port in _DEV_PORTS.items()
            ],
        )
    except Exception:
        for proc in procs.values():
            await proc.shutdown()
        raise

    _DEV_PIDFILE.parent.mkdir(parents=True, exist_ok=True)
    _DEV_PIDFILE.write_text(json.dumps({name: proc.pid for name, proc in procs.items()}))
    _DEV_STATE_FILE.write_text(json.dumps(state, indent=2))
    print(f"dev stack ready; ports={_DEV_PORTS}")
    print(f"logs: {log_dir}")
    print("press Ctrl-C to stop")
    try:
        await asyncio.gather(*[proc.process.wait() for proc in procs.values()])
    except asyncio.CancelledError:
        pass
    finally:
        await asyncio.gather(*[proc.shutdown() for proc in procs.values()])
        _DEV_PIDFILE.unlink(missing_ok=True)
    return 0


async def _stop_dev() -> int:
    import json
    import signal

    if not _DEV_PIDFILE.exists():
        print("dev stack pidfile not found; nothing to stop")
        return 0
    pids = json.loads(_DEV_PIDFILE.read_text())
    for name, pid in pids.items():
        try:
            os.kill(pid, signal.SIGTERM)
            print(f"stopped {name} (pid={pid})")
        except ProcessLookupError:
            print(f"{name} pid={pid} not running")
    _DEV_PIDFILE.unlink(missing_ok=True)
    return 0


async def _dump_dev() -> int:
    import json

    out: dict[str, Any] = {}
    async with httpx.AsyncClient(timeout=3.0) as client:
        for name, port in _DEV_PORTS.items():
            try:
                response = await client.get(f"http://127.0.0.1:{port}/control/state")
                out[name] = response.json()
            except Exception as exc:
                out[name] = {"error": repr(exc)}
    print(json.dumps(out, indent=2))
    return 0


def _main() -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="orchestrator")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("start").add_argument("--profile", default="dev")
    sub.add_parser("stop").add_argument("--profile", default="dev")
    sub.add_parser("dump").add_argument("--profile", default="dev")
    args = parser.parse_args()

    if args.cmd == "start":
        return asyncio.run(_start_dev())
    if args.cmd == "stop":
        return asyncio.run(_stop_dev())
    if args.cmd == "dump":
        return asyncio.run(_dump_dev())
    parser.error(f"unknown command {args.cmd!r}")
    return 2


if __name__ == "__main__":
    sys.exit(_main())


__all__ = ["Stack"]
