"""``FakeSSHClient`` — canonical fake for :class:`ISSHClient`.

In-memory virtual filesystem + programmable command handler. No real
sockets. State machine modelled around three phases:

* connected (true after construction; ``close`` flips to false)
* command history (every ``exec`` is recorded)
* virtual fs (``upload_file`` writes a key; ``download_file`` reads a
  key; ``file_exists`` membership-tests)

Chaos surface:

* :meth:`inject_connect_timeout` — every method raises
  :class:`SSHConnectionError`
* :meth:`inject_command_failure` — next N execs return non-zero
* :meth:`inject_disconnect_after_n_commands` — close the connection
  after N more execs
* :meth:`inject_transfer_failure` — next N transfers raise
* :meth:`set_command_response` — register a canned ``(exit_code,
  stdout, stderr)`` triplet for a substring match on commands
* :meth:`reset_chaos`
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from ryotenkai_shared.infrastructure.ssh import (
    ISSHClient,
    SSHCommandResult,
    SSHConnectionError,
    SSHTransferError,
)
from tests._harness.clock import Clock, RealClock


@dataclass
class _ChaosState:
    connect_timeout: bool = False
    command_failures_remaining: int = 0
    transfer_failures_remaining: int = 0
    disconnect_after: int | None = None


@dataclass
class _CannedResponse:
    pattern: str
    exit_code: int
    stdout: str
    stderr: str


class FakeSSHClient:
    """Deterministic in-memory fake for :class:`ISSHClient`."""

    def __init__(
        self,
        *,
        host: str = "fake-host",
        clock: Clock | None = None,
    ) -> None:
        self._host = host
        self._clock: Clock = clock if clock is not None else RealClock()
        self._connected = True
        self._fs: dict[str, bytes] = {}
        self._command_history: list[dict[str, Any]] = []
        self._chaos = _ChaosState()
        self._canned: list[_CannedResponse] = []
        self._exec_counter = 0

    # ------------------------------------------------------------------
    # Chaos surface
    # ------------------------------------------------------------------

    def inject_connect_timeout(self, value: bool = True) -> None:
        self._chaos.connect_timeout = value

    def inject_command_failure(self, count: int = 1) -> None:
        if count < 0:
            raise ValueError("count must be non-negative")
        self._chaos.command_failures_remaining = count

    def inject_transfer_failure(self, count: int = 1) -> None:
        if count < 0:
            raise ValueError("count must be non-negative")
        self._chaos.transfer_failures_remaining = count

    def inject_disconnect_after_n_commands(self, n: int) -> None:
        if n < 0:
            raise ValueError("n must be non-negative")
        self._chaos.disconnect_after = n

    def set_command_response(
        self,
        pattern: str,
        *,
        exit_code: int = 0,
        stdout: str = "",
        stderr: str = "",
    ) -> None:
        """Register a canned response for any exec whose command matches ``pattern``.

        ``pattern`` is a regex (compiled with :func:`re.search`).
        """
        self._canned.append(
            _CannedResponse(
                pattern=pattern, exit_code=exit_code, stdout=stdout, stderr=stderr,
            ),
        )

    def reset_chaos(self) -> None:
        self._chaos = _ChaosState()
        self._canned.clear()

    # ------------------------------------------------------------------
    # Inspection helpers (test-only)
    # ------------------------------------------------------------------

    def write_remote_file(self, path: str, content: bytes) -> None:
        self._fs[path] = content

    def read_remote_file(self, path: str) -> bytes:
        return self._fs[path]

    def list_remote_files(self) -> list[str]:
        return sorted(self._fs.keys())

    def command_history(self) -> list[dict[str, Any]]:
        return list(self._command_history)

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        return {
            "host": self._host,
            "connected": self._connected,
            "fs_keys": sorted(self._fs.keys()),
            "fs_byte_sizes": {k: len(v) for k, v in self._fs.items()},
            "command_history": list(self._command_history),
            "chaos": {
                "connect_timeout": self._chaos.connect_timeout,
                "command_failures_remaining": self._chaos.command_failures_remaining,
                "transfer_failures_remaining": self._chaos.transfer_failures_remaining,
                "disconnect_after": self._chaos.disconnect_after,
            },
            "canned": [
                {
                    "pattern": c.pattern,
                    "exit_code": c.exit_code,
                    "stdout_size": len(c.stdout),
                    "stderr_size": len(c.stderr),
                }
                for c in self._canned
            ],
        }

    # ------------------------------------------------------------------
    # ISSHClient surface
    # ------------------------------------------------------------------

    @property
    def host(self) -> str:
        return self._host

    async def exec(
        self,
        command: str,
        *,
        timeout: float | None = None,
    ) -> SSHCommandResult:
        # WHY pre-guard for disconnect-after: the chaos contract is
        # "Nth exec succeeds, (N+1)th raises". The decrement happens
        # BEFORE the call so callers see the disconnection on call
        # N+1, not after-the-fact on call N+2.
        self._consume_disconnect_after_budget()
        self._guard()
        self._exec_counter += 1
        # Track BEFORE running canned responses / failures so the
        # history reflects "the call was made" even if it failed.
        record: dict[str, Any] = {"command": command, "exec_index": self._exec_counter}
        self._command_history.append(record)

        if self._chaos.command_failures_remaining > 0:
            self._chaos.command_failures_remaining -= 1
            result = SSHCommandResult(
                exit_code=1, stdout="", stderr="fake injected command failure",
            )
            record["exit_code"] = 1
            return result

        # Canned responses — first match wins (registration order).
        for canned in self._canned:
            if re.search(canned.pattern, command):
                result = SSHCommandResult(
                    exit_code=canned.exit_code, stdout=canned.stdout, stderr=canned.stderr,
                )
                record["exit_code"] = canned.exit_code
                return result

        # Default — empty stdout, success.
        result = SSHCommandResult(exit_code=0, stdout="", stderr="")
        record["exit_code"] = 0
        return result

    async def upload_file(
        self,
        local_path: str,
        remote_path: str,
        *,
        timeout: float | None = None,
    ) -> None:
        self._guard()
        self._fire_transfer_chaos()
        # Best-effort: the fake's "fs" is purely keyed by ``remote_path``;
        # the ``local_path`` is stored too for debug purposes via history.
        self._command_history.append(
            {"command": "upload_file", "local": local_path, "remote": remote_path},
        )
        # The fake doesn't actually read ``local_path`` from disk —
        # tests that want byte content can call ``write_remote_file``
        # explicitly. Default: leave a marker.
        if remote_path not in self._fs:
            self._fs[remote_path] = b""

    async def download_file(
        self,
        remote_path: str,
        local_path: str,
        *,
        timeout: float | None = None,
    ) -> None:
        self._guard()
        self._fire_transfer_chaos()
        self._command_history.append(
            {"command": "download_file", "remote": remote_path, "local": local_path},
        )
        if remote_path not in self._fs:
            raise SSHTransferError(f"remote file does not exist: {remote_path!r}")

    async def file_exists(self, remote_path: str) -> bool:
        self._guard()
        # ``file_exists`` doesn't count as a real "command" but we
        # record it for observability.
        self._command_history.append({"command": "file_exists", "remote": remote_path})
        return remote_path in self._fs

    async def close(self) -> None:
        self._connected = False

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _guard(self) -> None:
        if self._chaos.connect_timeout:
            raise SSHConnectionError("fake injected connect timeout")
        if not self._connected:
            raise SSHConnectionError(f"connection to {self._host!r} is closed")

    def _fire_transfer_chaos(self) -> None:
        if self._chaos.transfer_failures_remaining > 0:
            self._chaos.transfer_failures_remaining -= 1
            raise SSHTransferError("fake injected transfer failure")

    def _consume_disconnect_after_budget(self) -> None:
        # Contract: ``inject_disconnect_after_n_commands(N)`` means the
        # next N execs succeed, the (N+1)th raises ``SSHConnectionError``.
        # Each call decrements the remaining budget; once it would go
        # negative, flip ``_connected`` to False so the subsequent
        # ``_guard`` raises.
        if self._chaos.disconnect_after is None:
            return
        if self._chaos.disconnect_after == 0:
            self._connected = False
            self._chaos.disconnect_after = None
            return
        self._chaos.disconnect_after -= 1


# Static guarantee.
_runtime_check: ISSHClient = FakeSSHClient()


__all__ = [
    "FakeSSHClient",
]
