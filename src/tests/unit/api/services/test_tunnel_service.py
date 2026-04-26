"""Phase 5 — :class:`src.api.services.tunnel_service.SSHTunnelManager`.

Verifies the tunnel-management contract without spawning real ``ssh``
or binding real sockets:

- TestArgvBuild        ssh argv has ``-fN -L`` + ControlMaster + correct opts
- TestPortAllocation   auto-pick walks the range; pinned port is honoured
- TestOpenLifecycle    successful open → reachable; non-zero ssh → error
- TestReadinessProbe   probe loop times out cleanly when port never opens
- TestCloseIsBestEffort  close failures are swallowed; idempotent
- TestSocketDirIsolation  uses a dedicated control_sockets dir, not
                          SSHClient's shared one

We deliberately avoid importing :mod:`src.api.services` (heavy package
init pulls in colorlog) and load tunnel_service via importlib instead.
The CI image has all deps; this is a dev-environment workaround.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


def _load_tunnel_service():
    """Import :mod:`tunnel_service` directly, skipping the
    :mod:`src.api.services` package init (which pulls heavy deps)."""
    if "ryotenkai_tunnel_service" in sys.modules:
        return sys.modules["ryotenkai_tunnel_service"]
    repo_root = Path(__file__).resolve().parents[5]
    src_path = repo_root / "src" / "api" / "services" / "tunnel_service.py"
    spec = importlib.util.spec_from_file_location(
        "ryotenkai_tunnel_service", str(src_path),
    )
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ryotenkai_tunnel_service"] = mod
    spec.loader.exec_module(mod)
    return mod


_tunnel_mod = _load_tunnel_service()
SSHTunnelManager = _tunnel_mod.SSHTunnelManager
SSHTunnelEndpoint = _tunnel_mod.SSHTunnelEndpoint
SSHTunnelError = _tunnel_mod.SSHTunnelError
DEFAULT_REMOTE_PORT = _tunnel_mod.DEFAULT_REMOTE_PORT


pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ok_runner(*, calls: list[list[str]] | None = None):
    """Pretend ``ssh`` always succeeds.

    If ``calls`` is provided, append every argv to it (useful for
    asserting on the constructed command). Returns a callable that
    matches the tunnel manager's ``runner`` injection point.
    """
    def _run(argv: list[str], _timeout: float) -> subprocess.CompletedProcess:
        if calls is not None:
            calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
    return _run


def _failing_runner(rc: int = 255, stderr: str = "auth failed"):
    """Pretend ``ssh`` always exits non-zero — used to verify the
    open() error path."""
    def _run(argv: list[str], _timeout: float) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(argv, rc, stdout="", stderr=stderr)
    return _run


def _always_open_probe(_port: int, *, mode: str) -> bool:
    """Probe stub that says everything is in the right state — port
    18080 is free for binding AND the tunnel is reachable. Lets the
    tests run without touching real sockets."""
    return True


def _probe_with_taken_ports(taken: set[int]):
    """Probe stub that pretends ``taken`` are already bound.

    - is_free → False if port in ``taken``, else True
    - is_open → True if port in ``taken`` (something is listening),
      else False
    """
    def _probe(port: int, *, mode: str) -> bool:
        if mode == "is_free":
            return port not in taken
        if mode == "is_open":
            return port in taken
        raise ValueError(mode)
    return _probe


def _endpoint() -> SSHTunnelEndpoint:
    return SSHTunnelEndpoint(
        host="1.2.3.4", port=22022, username="root", key_path="/k/id_rsa",
    )


# ---------------------------------------------------------------------------
# argv construction
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestArgvBuild:
    async def test_argv_contains_fn_and_forward(
        self, tmp_path: Path,
    ) -> None:
        calls: list[list[str]] = []
        mgr = SSHTunnelManager(
            _endpoint(),
            socket_dir=tmp_path / "sockets",
            ssh_executable="ssh",
            runner=_ok_runner(calls=calls),
            port_probe=_always_open_probe,
        )
        await mgr.open()
        try:
            argv = calls[0]
            assert "-f" in argv
            assert "-N" in argv
            # ``-L <local>:127.0.0.1:8080`` must appear
            forward_idx = argv.index("-L")
            mapping = argv[forward_idx + 1]
            assert mapping.startswith("18080:127.0.0.1:")
            assert mapping.endswith(str(DEFAULT_REMOTE_PORT))
            # endpoint comes last
            assert argv[-1] == "root@1.2.3.4"
            # SSH port is non-default → -p must appear
            assert "-p" in argv
            assert argv[argv.index("-p") + 1] == "22022"
            # key path is propagated
            assert "-i" in argv
            assert argv[argv.index("-i") + 1] == "/k/id_rsa"
            # ExitOnForwardFailure prevents silent half-open tunnels
            assert "ExitOnForwardFailure=yes" in argv
            # ControlMaster + ControlPath wired
            opt_indices = [i for i, v in enumerate(argv) if v == "-o"]
            opts_after = [argv[i + 1] for i in opt_indices]
            assert any(o.startswith("ControlPath=") for o in opts_after)
            assert "ControlMaster=auto" in opts_after
        finally:
            await mgr.close()


# ---------------------------------------------------------------------------
# port allocation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestPortAllocation:
    async def test_picks_first_free_port(self, tmp_path: Path) -> None:
        # Pretend 18080 is already taken — manager should pick 18081.
        probe = _probe_with_taken_ports({18080})
        # is_open must always return True for the chosen port so the
        # readiness probe completes — wrap the probe to flip that
        # bit on whichever port we end up choosing.
        def _probe(port: int, *, mode: str) -> bool:
            if mode == "is_open":
                return port == 18081
            return probe(port, mode=mode)

        mgr = SSHTunnelManager(
            _endpoint(),
            socket_dir=tmp_path / "sockets",
            runner=_ok_runner(),
            port_probe=_probe,
        )
        await mgr.open()
        try:
            assert mgr.local_port == 18081
            assert mgr.base_url == "http://127.0.0.1:18081"
        finally:
            await mgr.close()

    async def test_pinned_port_is_used(self, tmp_path: Path) -> None:
        mgr = SSHTunnelManager(
            _endpoint(),
            local_port=18099,
            socket_dir=tmp_path / "sockets",
            runner=_ok_runner(),
            port_probe=_always_open_probe,
        )
        await mgr.open()
        try:
            assert mgr.local_port == 18099
        finally:
            await mgr.close()

    async def test_pinned_port_already_taken_raises(self, tmp_path: Path) -> None:
        probe = _probe_with_taken_ports({18099})
        mgr = SSHTunnelManager(
            _endpoint(),
            local_port=18099,
            socket_dir=tmp_path / "sockets",
            runner=_ok_runner(),
            port_probe=probe,
        )
        with pytest.raises(SSHTunnelError, match="18099"):
            await mgr.open()
        assert not mgr.is_open

    async def test_no_free_port_raises(self, tmp_path: Path) -> None:
        # Every port in the range is taken — open should refuse.
        every_port = set(range(18080, 18100))

        def _probe(port: int, *, mode: str) -> bool:
            if mode == "is_free":
                return port not in every_port
            return False

        mgr = SSHTunnelManager(
            _endpoint(),
            socket_dir=tmp_path / "sockets",
            runner=_ok_runner(),
            port_probe=_probe,
        )
        with pytest.raises(SSHTunnelError, match="no free local port"):
            await mgr.open()


# ---------------------------------------------------------------------------
# open lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestOpenLifecycle:
    async def test_idempotent_open(self, tmp_path: Path) -> None:
        calls: list[list[str]] = []
        mgr = SSHTunnelManager(
            _endpoint(),
            socket_dir=tmp_path / "sockets",
            runner=_ok_runner(calls=calls),
            port_probe=_always_open_probe,
        )
        await mgr.open()
        await mgr.open()  # second call no-ops
        try:
            # ssh was invoked exactly once even though open() ran twice
            assert len([c for c in calls if "-f" in c]) == 1
        finally:
            await mgr.close()

    async def test_ssh_failure_raises_and_marks_closed(self, tmp_path: Path) -> None:
        mgr = SSHTunnelManager(
            _endpoint(),
            socket_dir=tmp_path / "sockets",
            runner=_failing_runner(rc=255, stderr="permission denied"),
            port_probe=_always_open_probe,
        )
        with pytest.raises(SSHTunnelError, match="255"):
            await mgr.open()
        assert not mgr.is_open
        # Reading local_port without an open tunnel must raise — we
        # treat ``mgr`` as unusable until a successful open().
        with pytest.raises(SSHTunnelError):
            _ = mgr.local_port

    async def test_async_context_manager(self, tmp_path: Path) -> None:
        async with SSHTunnelManager(
            _endpoint(),
            socket_dir=tmp_path / "sockets",
            runner=_ok_runner(),
            port_probe=_always_open_probe,
        ) as mgr:
            assert mgr.is_open
            assert mgr.base_url.startswith("http://127.0.0.1:")
        # Exiting context closes the tunnel
        assert not mgr.is_open


# ---------------------------------------------------------------------------
# readiness probe
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestReadinessProbe:
    async def test_unreachable_local_port_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Probe says is_free=True (so allocation succeeds) but never
        # reports is_open=True — the readiness loop must time out
        # cleanly. Shrink the timeout so the test stays fast.
        monkeypatch.setattr(
            _tunnel_mod, "_TUNNEL_READY_TIMEOUT_SECONDS", 0.05,
        )
        monkeypatch.setattr(
            _tunnel_mod, "_TUNNEL_PROBE_INTERVAL_SECONDS", 0.01,
        )

        def _probe(port: int, *, mode: str) -> bool:
            if mode == "is_free":
                return True
            if mode == "is_open":
                return False  # never accepts
            raise ValueError(mode)

        mgr = SSHTunnelManager(
            _endpoint(),
            socket_dir=tmp_path / "sockets",
            runner=_ok_runner(),
            port_probe=_probe,
        )
        with pytest.raises(SSHTunnelError, match="not become reachable"):
            await mgr.open()


# ---------------------------------------------------------------------------
# close()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestCloseIsBestEffort:
    async def test_close_calls_ssh_o_exit(self, tmp_path: Path) -> None:
        calls: list[list[str]] = []
        mgr = SSHTunnelManager(
            _endpoint(),
            socket_dir=tmp_path / "sockets",
            runner=_ok_runner(calls=calls),
            port_probe=_always_open_probe,
        )
        await mgr.open()
        await mgr.close()
        # The second invocation is the close — argv must include
        # ``-O exit``.
        close_argv = calls[-1]
        assert "-O" in close_argv
        assert close_argv[close_argv.index("-O") + 1] == "exit"
        assert not mgr.is_open

    async def test_close_when_never_opened_is_noop(self, tmp_path: Path) -> None:
        mgr = SSHTunnelManager(
            _endpoint(),
            socket_dir=tmp_path / "sockets",
            runner=_ok_runner(),
            port_probe=_always_open_probe,
        )
        await mgr.close()  # must not raise

    async def test_close_swallows_runner_exceptions(self, tmp_path: Path) -> None:
        # First call (open) succeeds; second call (close) raises.
        # close() must swallow it.
        invocations = {"count": 0}

        def _runner(argv: list[str], _t: float) -> subprocess.CompletedProcess:
            invocations["count"] += 1
            if invocations["count"] == 1:
                return subprocess.CompletedProcess(argv, 0)
            raise OSError("ssh exited weird")

        mgr = SSHTunnelManager(
            _endpoint(),
            socket_dir=tmp_path / "sockets",
            runner=_runner,
            port_probe=_always_open_probe,
        )
        await mgr.open()
        await mgr.close()  # must not propagate the OSError
        assert not mgr.is_open


# ---------------------------------------------------------------------------
# socket dir isolation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestSocketDirIsolation:
    async def test_uses_supplied_socket_dir(self, tmp_path: Path) -> None:
        # The control socket must live under the directory we passed,
        # NOT under SSHClient's shared ``~/.ssh/sockets``.
        sock_dir = tmp_path / "isolated_sockets"
        mgr = SSHTunnelManager(
            _endpoint(),
            socket_dir=sock_dir,
            runner=_ok_runner(),
            port_probe=_always_open_probe,
        )
        await mgr.open()
        try:
            assert sock_dir.exists()
            # Permission should be 0o700 — SSH refuses sockets in a
            # world-readable dir.
            mode = sock_dir.stat().st_mode & 0o777
            assert mode == 0o700, f"socket dir mode is {oct(mode)}, expected 0o700"
            assert mgr.control_path is not None
            assert mgr.control_path.startswith(str(sock_dir))
        finally:
            await mgr.close()
