"""Subprocess spawn / health-poll / shutdown helpers.

Process model:

* Spawn the sidecar (or the real control plane / runner) under
  ``asyncio.create_subprocess_exec`` so the orchestrator can boot many in
  parallel without blocking.
* Capture stdout+stderr to a per-process log file under
  ``tests/.stack_logs/<run_id>/<name>.log`` — never to the test runner's
  console.
* Poll a health URL with :class:`httpx.AsyncClient` until it returns 200.
* Shut down with SIGTERM, then SIGKILL after a grace period; idempotent.
"""

from __future__ import annotations

import asyncio
import os
import signal
import sys
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

import httpx


@dataclass
class ManagedProcess:
    name: str
    process: asyncio.subprocess.Process
    log_path: Path
    log_file: IO[bytes]
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def pid(self) -> int:
        return self.process.pid

    def is_alive(self) -> bool:
        return self.process.returncode is None

    async def shutdown(self, *, grace_seconds: float = 5.0) -> None:
        if self.process.returncode is not None:
            self._close_log()
            return

        with suppress(ProcessLookupError):
            self.process.terminate()

        try:
            await asyncio.wait_for(self.process.wait(), timeout=grace_seconds)
        except TimeoutError:
            with suppress(ProcessLookupError):
                self.process.kill()
            with suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self.process.wait(), timeout=grace_seconds)

        self._close_log()

    def _close_log(self) -> None:
        with suppress(Exception):
            self.log_file.flush()
        with suppress(Exception):
            self.log_file.close()


async def spawn(
    *,
    name: str,
    cmd: list[str],
    log_path: Path,
    env: dict[str, str] | None = None,
    cwd: Path | None = None,
) -> ManagedProcess:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("wb")
    full_env = os.environ.copy()
    if env:
        full_env.update(env)

    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=log_file,
        stderr=asyncio.subprocess.STDOUT,
        env=full_env,
        cwd=str(cwd) if cwd else None,
        # Detach into a new process group so SIGTERM on the parent
        # doesn't blast the kids before we get a chance to clean up.
        start_new_session=True,
    )
    return ManagedProcess(name=name, process=process, log_path=log_path, log_file=log_file)


async def wait_for_health(
    url: str,
    *,
    timeout: float = 30.0,
    poll: float = 0.1,
    process: ManagedProcess | None = None,
) -> None:
    """Poll ``url`` until 200 or timeout. If ``process`` exits early, raise."""
    deadline = asyncio.get_event_loop().time() + timeout
    last_error: BaseException | None = None
    async with httpx.AsyncClient(timeout=2.0) as client:
        while asyncio.get_event_loop().time() < deadline:
            if process is not None and process.process.returncode is not None:
                raise RuntimeError(
                    f"process {process.name} exited (code={process.process.returncode}) "
                    f"before health probe succeeded; logs at {process.log_path}",
                )
            try:
                response = await client.get(url)
                if response.status_code == 200:
                    return
                last_error = RuntimeError(f"health probe returned {response.status_code}")
            except (httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError) as exc:
                last_error = exc
            await asyncio.sleep(poll)

    err_msg = f"health probe at {url} timed out after {timeout}s"
    if last_error is not None:
        err_msg = f"{err_msg} (last error: {last_error!r})"
    raise TimeoutError(err_msg)


def python_executable() -> str:
    return sys.executable or "python"


def shutdown_signal() -> int:
    return signal.SIGTERM


__all__ = [
    "ManagedProcess",
    "python_executable",
    "shutdown_signal",
    "spawn",
    "wait_for_health",
]
