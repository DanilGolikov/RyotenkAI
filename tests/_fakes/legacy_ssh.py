"""``FakeLegacySSHClient`` — canonical fake for ``utils.ssh_client.SSHClient``.

The legacy (synchronous) SSH client API is still used by the
:mod:`ryotenkai_providers.single_node` provider tree and a handful of
helper modules. It differs from the modern
:class:`ryotenkai_shared.infrastructure.ssh.ISSHClient` Protocol:

* Synchronous (``exec_command`` returns a tuple, not an awaitable).
* Return shape is ``(success: bool, stdout: str, stderr: str)``.
* Additional helpers (``test_connection``, ``directory_exists``,
  ``create_directory``, ``upload_file``) that the modern Protocol
  does not expose.

Until the single_node tree migrates to ``ISSHClient`` this fake
provides the canonical test seam so individual test files do not roll
their own one-off stubs.

Capabilities:

* :meth:`set_command_response` — register a canned tuple per regex.
* :meth:`set_default_response` — global fallback when no canned
  pattern matches (default is ``(True, "", "")``).
* :meth:`inject_command_failure` — next N execs return ``(False, "", err)``.
* :meth:`inject_exception` — next N execs raise the supplied exception.
* :meth:`commands_log` — chronological history of executed commands +
  their ``timeout``/``silent`` kwargs (for invariant assertions).

The fake explicitly does NOT mimic the legacy class's hidden behaviour
(spawning real ``ssh`` subprocesses, building remote shell commands,
masking secrets). Production code under test must depend only on the
documented surface; if a test wants to verify the underlying shell
command the production code generates, it inspects ``commands_log``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class _CannedResponse:
    pattern: str
    success: bool
    stdout: str
    stderr: str


@dataclass
class _ExecCall:
    """One recorded ``exec_command`` invocation."""

    command: str
    timeout: int
    silent: bool
    background: bool


class FakeLegacySSHClient:
    """Deterministic in-memory fake for ``utils.ssh_client.SSHClient``."""

    def __init__(
        self,
        *,
        host: str = "fake-host",
        port: int = 22,
        username: str | None = "fake-user",
        key_path: str = "/fake/key",
        connect_timeout: int = 10,
        connection_ok: bool = True,
    ) -> None:
        self.host = host
        self.port = port
        self.username = username
        self.key_path = key_path
        self.connect_timeout = connect_timeout
        self._connection_ok = connection_ok

        self._canned: list[_CannedResponse] = []
        self._default = _CannedResponse(pattern="", success=True, stdout="", stderr="")
        self._fail_remaining: int = 0
        self._raise_remaining: int = 0
        self._raise_exception: Exception | None = None
        self._commands: list[_ExecCall] = []
        self._dirs: set[str] = set()
        self._uploads: list[tuple[str, str]] = []

    # ------------------------------------------------------------------
    # Programming surface (test-side)
    # ------------------------------------------------------------------

    def set_command_response(
        self,
        pattern: str,
        *,
        success: bool = True,
        stdout: str = "",
        stderr: str = "",
    ) -> None:
        """Register a canned response. ``pattern`` is a ``re.search`` regex."""
        self._canned.append(
            _CannedResponse(pattern=pattern, success=success, stdout=stdout, stderr=stderr),
        )

    def set_default_response(
        self,
        *,
        success: bool = True,
        stdout: str = "",
        stderr: str = "",
    ) -> None:
        """Set the response used when no canned pattern matches."""
        self._default = _CannedResponse(pattern="", success=success, stdout=stdout, stderr=stderr)

    def inject_command_failure(self, count: int = 1) -> None:
        if count < 0:
            raise ValueError("count must be non-negative")
        self._fail_remaining = count

    def inject_exception(self, exc: Exception, count: int = 1) -> None:
        """Next ``count`` ``exec_command`` calls raise ``exc``."""
        if count < 0:
            raise ValueError("count must be non-negative")
        self._raise_remaining = count
        self._raise_exception = exc

    def set_connection_ok(self, ok: bool) -> None:
        self._connection_ok = ok

    def register_directory(self, path: str) -> None:
        """Mark ``path`` as already existing on the remote host."""
        self._dirs.add(path)

    # ------------------------------------------------------------------
    # Inspection surface (test-side)
    # ------------------------------------------------------------------

    @property
    def commands_log(self) -> list[_ExecCall]:
        """Chronological history of every ``exec_command`` invocation."""
        return list(self._commands)

    @property
    def commands(self) -> list[str]:
        """Just the command strings (legacy compatibility — many tests
        read ``.commands`` directly on inline stubs)."""
        return [c.command for c in self._commands]

    @property
    def uploads(self) -> list[tuple[str, str]]:
        return list(self._uploads)

    # ------------------------------------------------------------------
    # SSHClient public surface — keep in sync with utils.ssh_client.SSHClient
    # ------------------------------------------------------------------

    def exec_command(
        self,
        command: str,
        background: bool = False,
        timeout: int = 30,
        silent: bool = False,
    ) -> tuple[bool, str, str]:
        self._commands.append(
            _ExecCall(command=command, timeout=timeout, silent=silent, background=background),
        )

        if self._raise_remaining > 0:
            self._raise_remaining -= 1
            assert self._raise_exception is not None
            raise self._raise_exception

        if self._fail_remaining > 0:
            self._fail_remaining -= 1
            return False, "", "fake injected command failure"

        for canned in self._canned:
            if re.search(canned.pattern, command):
                return canned.success, canned.stdout, canned.stderr

        return self._default.success, self._default.stdout, self._default.stderr

    def test_connection(self, max_retries: int = 3, retry_delay: int = 5) -> tuple[bool, str]:
        if self._connection_ok:
            return True, ""
        return False, "fake injected connection failure"

    def directory_exists(self, remote_path: str) -> bool:
        return remote_path in self._dirs

    def create_directory(self, remote_path: str) -> tuple[bool, str]:
        self._dirs.add(remote_path)
        return True, ""

    def upload_file(self, local_path: str, remote_path: str) -> tuple[bool, str]:
        self._uploads.append((local_path, remote_path))
        return True, ""

    def snapshot(self) -> dict[str, Any]:
        return {
            "host": self.host,
            "port": self.port,
            "username": self.username,
            "dirs": sorted(self._dirs),
            "commands": [c.command for c in self._commands],
            "uploads": list(self._uploads),
        }


__all__ = ["FakeLegacySSHClient"]
