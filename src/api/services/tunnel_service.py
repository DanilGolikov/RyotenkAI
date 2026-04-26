"""SSH ``-L`` tunnel manager — Phase 5.

The runner serves loopback-only HTTP on ``127.0.0.1:8080`` inside the
pod (see :file:`docker/training/entrypoint.sh`). The Mac control
plane reaches it through an ``ssh -L`` tunnel — typically the same
SSH endpoint that :class:`src.utils.ssh_client.SSHClient` already
uses for rsync / exec, but with **a separate ControlMaster socket**.

Why a separate socket: if the tunnel piggy-backed on the rsync
client's master, then closing the rsync client at the end of a
deployment stage would tear down the tunnel mid-job. They have
different lifetimes — rsync is per-stage, the tunnel is per-job — so
they get isolated control sockets.

This module is sync-with-async-shell — :class:`SSHTunnelManager` exposes
async ``open()``/``close()`` to fit the rest of the Phase 5 client
plumbing, but the heavy lifting is plain :mod:`subprocess` calls.

Tunnel lifetime:

::

    open()                                          close()
      │                                                ▲
      ▼                                                │
    ┌──────────────────────────────────────────────────┴────┐
    │ ssh -fN -L <local>:127.0.0.1:8080 -o ControlPath=...  │
    │ host                                                  │
    └───────────────────────────────────────────────────────┘
                          │
                          ▼ (loopback inside pod)
                   uvicorn 127.0.0.1:8080
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import shutil
import socket
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "DEFAULT_LOCAL_PORT_RANGE",
    "DEFAULT_REMOTE_PORT",
    "SSHTunnelEndpoint",
    "SSHTunnelError",
    "SSHTunnelManager",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


# Pod-side runner port. Hard-coded to match
# :file:`docker/training/entrypoint.sh` — changing it requires a
# coordinated bump in both places. Phase 5 doesn't try to discover
# this; the contract IS that the runner binds 8080.
DEFAULT_REMOTE_PORT = 8080

# Local-port allocation window. 18080-18099 gives 20 simultaneous
# tunnels per Mac, plenty for one user. We pick a free port in this
# range rather than 0 so the URL is stable across restarts (helpful
# when a user has the URL pinned in their browser to a Web UI tab).
DEFAULT_LOCAL_PORT_RANGE = range(18080, 18100)

# How long to wait for the SSH tunnel to be reachable on the local
# end after ``ssh -fN`` returns. ``ssh -f`` only waits for auth, not
# for the forward to settle, so a tight probe loop is the only way
# to know we're actually open.
_TUNNEL_READY_TIMEOUT_SECONDS = 10.0
_TUNNEL_PROBE_INTERVAL_SECONDS = 0.1

# Where we stash the dedicated ControlMaster socket. Separate dir
# from :class:`SSHClient`'s ``~/.ssh/sockets/`` so a stray
# ``close_master()`` from rsync code can't kill our tunnel.
_DEFAULT_SOCKET_DIR = Path.home() / ".ssh" / "control_sockets" / "ryotenkai_runner"
_SSH_SOCKET_DIR_MODE = 0o700


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SSHTunnelEndpoint:
    """Where the tunnel SSHes *to* — hand-built so callers can pass
    raw host/port without having to construct an
    :class:`src.utils.ssh_client.SSHClient` first.

    ``key_path`` is optional: if ``None``, the tunnel relies on
    ``~/.ssh/config`` resolving the host (alias mode). This matches
    the existing :class:`SSHClient` semantics.
    """

    host: str
    port: int = 22
    username: str = "root"
    key_path: str | None = None

    @property
    def target(self) -> str:
        """``user@host`` (alias mode just returns the host alone)."""
        return f"{self.username}@{self.host}" if self.username else self.host


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class SSHTunnelError(RuntimeError):
    """Tunnel could not be opened, or closed unexpectedly."""


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class SSHTunnelManager:
    """Open and tear down a single ``ssh -L`` tunnel.

    Construct one per (endpoint × remote_port) pairing. Use as
    ``async with`` for automatic cleanup, or call ``open()``/``close()``
    explicitly when the lifetime spans multiple awaits / functions.

    Args:
        endpoint: SSH target (host/port/user/key).
        remote_port: pod-side port the tunnel forwards to (loopback
            inside the pod, not the SSH port). Defaults to
            :data:`DEFAULT_REMOTE_PORT` — change only if the runner
            ever moves off 8080.
        local_port: pin a specific local port. ``None`` = auto-pick
            the lowest free port in :data:`DEFAULT_LOCAL_PORT_RANGE`.
        socket_dir: directory for the ControlMaster socket. Override
            in tests; production uses :data:`_DEFAULT_SOCKET_DIR`.
        ssh_executable: path to ``ssh``. Falls back to ``shutil.which``
            so the tests can stub a fake binary.
        runner: injectable subprocess runner — tests pass a mock to
            assert on the argv without spawning a real ssh.
        port_probe: injectable TCP probe — tests pass a stub that
            returns immediately so they don't have to wait on a
            real socket connect.

    The manager is **NOT** thread-safe. Don't share an instance across
    asyncio tasks — build one per task / job.
    """

    def __init__(
        self,
        endpoint: SSHTunnelEndpoint,
        *,
        remote_port: int = DEFAULT_REMOTE_PORT,
        local_port: int | None = None,
        socket_dir: Path | None = None,
        ssh_executable: str | None = None,
        runner: _SubprocessRunner | None = None,
        port_probe: _PortProbe | None = None,
    ) -> None:
        self._endpoint = endpoint
        self._remote_port = remote_port
        self._requested_local_port = local_port
        self._socket_dir = socket_dir or _DEFAULT_SOCKET_DIR
        self._ssh_executable = ssh_executable or shutil.which("ssh") or "ssh"
        self._runner = runner or _default_runner
        self._port_probe = port_probe or _default_port_probe

        # Set on open(); cleared on close().
        self._control_path: str | None = None
        self._local_port: int | None = None
        self._is_open: bool = False

    # --- read-only state --------------------------------------------------

    @property
    def is_open(self) -> bool:
        return self._is_open

    @property
    def local_port(self) -> int:
        """Local port the tunnel is bound to. Raises if not yet open."""
        if self._local_port is None:
            raise SSHTunnelError("tunnel is not open")
        return self._local_port

    @property
    def base_url(self) -> str:
        """``http://127.0.0.1:<local_port>`` — convenient handle to
        feed straight to :class:`src.api.clients.JobClient`. Raises
        if the tunnel hasn't been opened yet."""
        return f"http://127.0.0.1:{self.local_port}"

    @property
    def control_path(self) -> str | None:
        return self._control_path

    # --- lifecycle --------------------------------------------------------

    async def __aenter__(self) -> SSHTunnelManager:
        await self.open()
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.close()

    async def open(self) -> None:
        """Open the tunnel — idempotent.

        Steps:
        1. Ensure the ControlMaster socket dir exists with 0o700
           perms (otherwise SSH refuses to use it).
        2. Pick a free local port from
           :data:`DEFAULT_LOCAL_PORT_RANGE`.
        3. Spawn ``ssh -fN -L <local>:127.0.0.1:<remote>`` with our
           dedicated ControlMaster.
        4. Probe the local port until it accepts (up to
           :data:`_TUNNEL_READY_TIMEOUT_SECONDS`).

        Raises:
            SSHTunnelError: socket dir creation failed, no free
                port in range, ssh exited non-zero, or the local
                port never accepted connections within the timeout.
        """
        if self._is_open:
            return

        try:
            self._socket_dir.mkdir(parents=True, exist_ok=True, mode=_SSH_SOCKET_DIR_MODE)
            self._socket_dir.chmod(_SSH_SOCKET_DIR_MODE)
        except OSError as exc:
            raise SSHTunnelError(
                f"failed to prepare control socket dir {self._socket_dir}: {exc}",
            ) from exc

        # ``%C`` expands to a hash of (host, port, user) — uniquely
        # tying the socket to its connection so two pods can have
        # tunnels open simultaneously without colliding on a path.
        self._control_path = str(self._socket_dir / "tunnel_%C")

        if self._requested_local_port is not None:
            # Caller pinned a port — use it as-is, but verify it's
            # actually free so we fail fast instead of getting a
            # cryptic "address already in use" out of ssh.
            if not self._port_probe(self._requested_local_port, mode="is_free"):
                raise SSHTunnelError(
                    f"requested local port {self._requested_local_port} "
                    f"is already in use",
                )
            self._local_port = self._requested_local_port
        else:
            self._local_port = _pick_free_port(
                DEFAULT_LOCAL_PORT_RANGE, probe=self._port_probe,
            )

        argv = self._build_open_argv()
        try:
            result = await asyncio.to_thread(self._runner, argv, _SSH_OPEN_TIMEOUT)
        except subprocess.TimeoutExpired as exc:
            self._reset_after_failed_open()
            raise SSHTunnelError(
                f"ssh -fN timed out after {_SSH_OPEN_TIMEOUT}s opening tunnel",
            ) from exc

        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            stdout = (result.stdout or "").strip()
            self._reset_after_failed_open()
            raise SSHTunnelError(
                f"ssh exited {result.returncode} opening tunnel: "
                f"{stderr or stdout!r}",
            )

        # ``ssh -fN`` returns once auth is complete, but the
        # forwarding setup may still be settling. Probe until the
        # local port accepts.
        try:
            await self._wait_until_reachable()
        except SSHTunnelError:
            # Tunnel never came up — mark closed before bubbling up
            # so the caller sees a clean ``is_open == False`` state.
            self._reset_after_failed_open()
            raise

        self._is_open = True
        logger.info(
            "ssh tunnel open: 127.0.0.1:%s → %s:%s "
            "(control socket %s)",
            self._local_port, self._endpoint.target, self._remote_port,
            self._control_path,
        )

    def _reset_after_failed_open(self) -> None:
        """Roll partial state back to "never opened" — used when
        ``open()`` raises mid-flight so the manager doesn't leak
        a half-allocated port or a control-path that points at no
        live ssh process."""
        self._is_open = False
        self._local_port = None
        self._control_path = None

    async def close(self) -> None:
        """Close the tunnel via ``ssh -O exit`` against our
        ControlMaster socket. Idempotent and best-effort — any
        failure is logged, never raised, since teardown happens at
        end-of-scope where exceptions would mask the real problem.
        """
        if not self._is_open:
            return

        if self._control_path:
            argv = [
                self._ssh_executable,
                "-o", f"ControlPath={self._control_path}",
                "-O", "exit",
                self._endpoint.target,
            ]
            with contextlib.suppress(Exception):
                await asyncio.to_thread(self._runner, argv, _SSH_CLOSE_TIMEOUT)

        self._is_open = False
        self._control_path = None
        self._local_port = None

    # --- internals --------------------------------------------------------

    def _build_open_argv(self) -> list[str]:
        """Construct the ``ssh -fN -L ...`` argv.

        ``-f`` (background after auth) + ``-N`` (no remote command)
        is the canonical "tunnel only" combo. We pin a few hardening
        options that match :class:`SSHClient`:
        - ``BatchMode=yes`` so we never block on a password prompt.
        - ``StrictHostKeyChecking=no`` + ``UserKnownHostsFile=/dev/null``
          because the pod's host key changes every provision.
        - ``ServerAliveInterval=30`` so an idle Mac sleep doesn't
          leave a zombie tunnel.
        """
        endpoint = self._endpoint
        opts: list[str] = [
            self._ssh_executable,
            "-f", "-N",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "BatchMode=yes",
            "-o", "PasswordAuthentication=no",
            "-o", "ServerAliveInterval=30",
            "-o", "ServerAliveCountMax=3",
            "-o", "ExitOnForwardFailure=yes",  # fail fast if the -L can't bind
            "-o", "LogLevel=ERROR",
        ]
        if self._control_path:
            opts.extend([
                "-o", "ControlMaster=auto",
                "-o", f"ControlPath={self._control_path}",
                "-o", f"ControlPersist={_SSH_CONTROL_PERSIST_SECONDS}",
            ])
        if endpoint.port and endpoint.port != 22:
            opts.extend(["-p", str(endpoint.port)])
        if endpoint.key_path:
            opts.extend(["-i", endpoint.key_path])

        opts.extend([
            "-L", f"{self._local_port}:127.0.0.1:{self._remote_port}",
            endpoint.target,
        ])
        return opts

    async def _wait_until_reachable(self) -> None:
        """Probe the local port until a TCP connect succeeds.

        ``ssh -fN`` returns the moment authentication finishes, but
        the local-side socket may not be accepting yet. We poll
        with a short interval until either the probe wins or we
        hit :data:`_TUNNEL_READY_TIMEOUT_SECONDS`.
        """
        loop = asyncio.get_event_loop()
        deadline = loop.time() + _TUNNEL_READY_TIMEOUT_SECONDS
        port = self._local_port  # captured for type narrowing
        assert port is not None  # set before this is called

        while loop.time() < deadline:
            if self._port_probe(port, mode="is_open"):
                return
            await asyncio.sleep(_TUNNEL_PROBE_INTERVAL_SECONDS)
        raise SSHTunnelError(
            f"ssh tunnel did not become reachable on 127.0.0.1:{port} "
            f"within {_TUNNEL_READY_TIMEOUT_SECONDS}s",
        )


# ---------------------------------------------------------------------------
# Subprocess + port helpers (extracted so tests can swap them)
# ---------------------------------------------------------------------------


# How long ssh has to authenticate + fork. RunPod auth typically
# completes in 2-3 s; 30 s gives plenty of headroom for slow links.
_SSH_OPEN_TIMEOUT = 30.0
# Closing via ``-O exit`` is fast — usually under 100 ms. 5 s is
# defensive.
_SSH_CLOSE_TIMEOUT = 5.0
# How long the ControlMaster keeps the master connection alive
# after the last forwarding session exits. ``yes`` = forever. We
# want forever because the tunnel is the only forwarding session
# and the lifecycle is owned by ``close()``.
_SSH_CONTROL_PERSIST_SECONDS = "yes"


# Test seam protocols. Production wires :func:`_default_runner` /
# :func:`_default_port_probe`; tests inject mocks.
_SubprocessRunner = Callable[[list[str], float], "subprocess.CompletedProcess[str]"]
_PortProbe = Callable[..., bool]


def _default_runner(argv: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
    """Production subprocess invocation. Captures stdout / stderr,
    forces a clean stdin, applies ``timeout`` so a hung ssh can never
    wedge the lifespan."""
    return subprocess.run(
        argv,
        capture_output=True,
        stdin=subprocess.DEVNULL,
        text=True,
        timeout=timeout,
        check=False,
    )


def _default_port_probe(port: int, *, mode: str) -> bool:
    """Default TCP probe — used by :meth:`open` to verify the
    local port is reachable, and by :func:`_pick_free_port` to find
    a free slot.

    ``mode`` is ``"is_free"`` or ``"is_open"``:

    - ``is_free``: bind 127.0.0.1:port, immediately close. Returns
      ``True`` if the bind succeeded.
    - ``is_open``: connect to 127.0.0.1:port. Returns ``True`` if
      the connect succeeded.

    Two modes share the same callable so tests can mock with one
    function and observe both call sites.
    """
    if mode == "is_free":
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(("127.0.0.1", port))
                return True
        except OSError:
            return False
    if mode == "is_open":
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.25):
                return True
        except OSError:
            return False
    raise ValueError(f"unknown probe mode: {mode!r}")


def _pick_free_port(port_range: range, *, probe) -> int:
    """Walk ``port_range`` in order; return the first free port.

    Sequential rather than random because predictable port choice
    helps debugging — a stray ``netstat -anp | grep 18080`` always
    shows the first tunnel.
    """
    for port in port_range:
        if probe(port, mode="is_free"):
            return port
    raise SSHTunnelError(
        f"no free local port in range {port_range} for ssh tunnel",
    )


