"""
runner_launcher — SSH-exec uvicorn + readiness probe.

Categories:
* Positive — happy path, idempotent re-launch (already-running)
* Negative — SSH command fails, runner doesn't bind, runner crashes
* Boundary — empty stdout, empty stderr, both empty
* Dependency errors — missing /usr/local/bin/python3, curl unavailable
* Regression — exec_command called with the documented timeout
* Invariants — Result-only return type, no exceptions leaked
* Logic-specific — command structure (idempotency check, redirect,
  readiness probe loop)
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from src.pipeline.stages.managers.deployment.runner_launcher import (
    RUNNER_LOG_PATH,
    RUNNER_PORT,
    RUNNER_READY_TIMEOUT_SECONDS,
    _build_launch_command,
    launch_runner,
)
from src.utils.result import ProviderError


@dataclass
class _SSHStub:
    """Minimal SSH client stub — exec_command returns a fixed tuple."""
    success: bool = True
    stdout: str = "runner ready"
    stderr: str = ""
    last_command: str = ""
    last_timeout: float | None = None
    raise_on_exec: BaseException | None = None

    def exec_command(self, *, command: str, silent: bool = True, timeout: float | None = None, **_: object):
        self.last_command = command
        self.last_timeout = timeout
        if self.raise_on_exec is not None:
            raise self.raise_on_exec
        return self.success, self.stdout, self.stderr


# ---------------------------------------------------------------------------
# Positive
# ---------------------------------------------------------------------------


def test_launch_runner_success_returns_ok() -> None:
    """SSH command succeeded and 'runner ready' in stdout → Ok(None)."""
    ssh = _SSHStub(success=True, stdout="runner ready", stderr="")
    result = launch_runner(ssh)  # type: ignore[arg-type]
    assert result.is_ok(), f"expected Ok, got {result!r}"
    assert result.unwrap() is None


def test_launch_runner_idempotent_when_already_running() -> None:
    """If the curl-probe finds the runner already healthy on its
    port, the script exits 0 with stdout=='runner already running'
    and we treat that as Ok — no duplicate launch."""
    ssh = _SSHStub(
        success=True,
        stdout="runner already running",
        stderr="",
    )
    result = launch_runner(ssh)  # type: ignore[arg-type]
    assert result.is_ok(), f"expected Ok, got {result!r}"


# ---------------------------------------------------------------------------
# Negative
# ---------------------------------------------------------------------------


def test_launch_runner_ssh_command_fails_returns_err() -> None:
    """exec_command success=False → Err with launcher code."""
    ssh = _SSHStub(success=False, stdout="", stderr="ssh: connection refused")
    result = launch_runner(ssh)  # type: ignore[arg-type]
    assert result.is_err(), f"expected Err, got {result!r}"
    err = result.unwrap_err()
    assert isinstance(err, ProviderError)
    assert err.code == "RUNNER_LAUNCH_FAILED"
    assert "ssh: connection refused" in str(err.details)


def test_launch_runner_healthz_timeout_returns_err_with_log_tail() -> None:
    """Runner failed to bind /healthz; script tail's runner.log to stderr.
    The tail must surface in the ProviderError details for diagnosis."""
    ssh = _SSHStub(
        success=False,
        stdout="",
        stderr=(
            "runner did not become ready within 30s\n"
            "--- tail of /workspace/runner.log ---\n"
            "ModuleNotFoundError: No module named 'src.utils'\n"
        ),
    )
    result = launch_runner(ssh)  # type: ignore[arg-type]
    assert result.is_err(), f"expected Err, got {result!r}"
    err = result.unwrap_err()
    assert "ModuleNotFoundError" in str(err.details)


def test_launch_runner_success_but_no_ready_marker_returns_err() -> None:
    """Defensive: if exec_command reports success but stdout doesn't
    contain 'runner ready' (e.g. a future protocol change broke
    parsing), we treat it as failure rather than silent-pass."""
    ssh = _SSHStub(success=True, stdout="some unrelated output", stderr="")
    result = launch_runner(ssh)  # type: ignore[arg-type]
    assert result.is_err(), f"expected Err, got {result!r}"


# ---------------------------------------------------------------------------
# Boundary
# ---------------------------------------------------------------------------


def test_launch_runner_empty_outputs_returns_err_with_placeholder() -> None:
    """Both stdout and stderr empty → Err with 'no diagnostic output'."""
    ssh = _SSHStub(success=False, stdout="", stderr="")
    result = launch_runner(ssh)  # type: ignore[arg-type]
    assert result.is_err(), f"expected Err, got {result!r}"
    err = result.unwrap_err()
    assert "no diagnostic output" in str(err.details)


def test_launch_runner_whitespace_only_outputs_returns_err() -> None:
    """Stdout/stderr full of just whitespace must still be treated as
    no diagnostic info — protects against false-positive stderr_tail
    in the error message."""
    ssh = _SSHStub(success=False, stdout="   \n\t", stderr="\n\n")
    result = launch_runner(ssh)  # type: ignore[arg-type]
    assert result.is_err(), f"expected Err, got {result!r}"


# ---------------------------------------------------------------------------
# Dependency errors / Invariants
# ---------------------------------------------------------------------------


def test_exec_command_uses_documented_timeout() -> None:
    """The launcher must allow exec_command longer than the readiness
    loop, so even when the loop times out we still get the tail-log
    output back. Default: timeout = readiness + 15s.
    """
    ssh = _SSHStub(success=True, stdout="runner ready")
    launch_runner(ssh)  # type: ignore[arg-type]
    assert ssh.last_timeout is not None
    assert ssh.last_timeout >= RUNNER_READY_TIMEOUT_SECONDS + 10


def test_launch_runner_does_not_swallow_exec_exceptions_silently() -> None:
    """If exec_command raises (network glitch, key issue), the
    exception propagates — caller wraps it in their own Result.
    The launcher itself does not catch arbitrary exceptions because
    that would mask SSH-level bugs as launcher-level errors."""
    ssh = _SSHStub(raise_on_exec=RuntimeError("ssh glitch"))
    with pytest.raises(RuntimeError, match="ssh glitch"):
        launch_runner(ssh)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Logic-specific — command structure
# ---------------------------------------------------------------------------


def test_command_idempotency_uses_healthz_not_pgrep() -> None:
    """Idempotency check must be a curl probe of /healthz — NOT
    ``pgrep -f``.

    Why: ``pgrep -f PATTERN`` matches on the full command line of
    every process. The very bash subshell that's evaluating this
    launch script has the pattern as one of its arguments, so
    pgrep self-matches, the ``if`` branch fires "runner already
    running", uvicorn never gets launched, and the readiness
    probe times out 30 s later with a confusing "file does not
    exist" diagnostic.

    A curl probe to /healthz can't false-positive: if no process
    is bound to port 8080, the probe returns non-zero and we
    proceed to launch.
    """
    cmd = _build_launch_command()
    # Active assertion: idempotency check is a curl probe.
    assert f"curl -sf http://127.0.0.1:{RUNNER_PORT}/healthz" in cmd
    # Negative regression: pgrep was the bug.
    assert "pgrep" not in cmd, \
        "pgrep idempotency check is forbidden — it self-matches the launch script"


def test_command_redirects_to_runner_log() -> None:
    """uvicorn stdout/stderr must be APPENDED (``>>``, not ``>``) to
    the canonical /workspace/runner.log so a runner-crash retry
    doesn't truncate the previous log — accumulating successive
    boots is forensically useful and also flushes the last line of
    a fast crash by the time the readiness probe times out."""
    cmd = _build_launch_command()
    assert RUNNER_LOG_PATH in cmd
    assert f">> {RUNNER_LOG_PATH} 2>&1" in cmd


def test_command_uses_nohup_disown_for_detachment() -> None:
    """The launched uvicorn must outlive the SSH session that started
    it. nohup + & + disown is the canonical pattern."""
    cmd = _build_launch_command()
    assert "nohup" in cmd
    assert "disown" in cmd


def test_command_polls_healthz_until_ready() -> None:
    """A readiness loop with curl must be present so
    SSH-exec returns only after uvicorn is actually serving.

    Without this we'd return Ok before the runner bound port 8080,
    and the next stage's tunnel /healthz probe would race."""
    cmd = _build_launch_command()
    assert f"http://127.0.0.1:{RUNNER_PORT}/healthz" in cmd
    assert "curl" in cmd
    assert f"seq 1 {RUNNER_READY_TIMEOUT_SECONDS}" in cmd


def test_command_dumps_log_tail_on_failure() -> None:
    """On readiness timeout, the script must dump diagnostics:
    ``ls -la`` of the log file (so we can tell a missing redirect
    from an empty traceback) AND a ``tail -100`` of the contents.
    stderr from those commands must NOT be silenced — masking
    "file not found" was a real bug we just fixed."""
    cmd = _build_launch_command()
    assert "ls -la " + RUNNER_LOG_PATH in cmd
    assert "tail -100 " + RUNNER_LOG_PATH in cmd
    # Critical: tail's stderr must not be swallowed.
    assert f"tail -100 {RUNNER_LOG_PATH} >&2 2>/dev/null" not in cmd, \
        "tail's stderr must be visible — masking it hides 'file not found'"


def test_command_uses_pythonpath_with_image_baseline() -> None:
    """uvicorn import path must include /opt/ryotenkai (where the image
    bakes its src/ baseline). Without it, ``src.runner.main`` won't
    import and the runner crashes immediately."""
    cmd = _build_launch_command()
    assert "PYTHONPATH=/opt/ryotenkai" in cmd
    assert "src.runner.main:app" in cmd


# ---------------------------------------------------------------------------
# Env-var injection (provider's required_runtime_env_vars)
# ---------------------------------------------------------------------------


def test_command_injects_provider_env_vars() -> None:
    """env=dict is forwarded as ``env KEY=VALUE ...`` between nohup
    and stdbuf — that's what makes RYOTENKAI_RUNTIME_PROVIDER
    reach the runner's lifespan hook."""
    cmd = _build_launch_command(
        env={"RYOTENKAI_RUNTIME_PROVIDER": "runpod", "RUNPOD_POD_ID": "abc123"},
    )
    assert "RYOTENKAI_RUNTIME_PROVIDER=runpod" in cmd
    assert "RUNPOD_POD_ID=abc123" in cmd


def test_command_shell_escapes_special_chars_in_env() -> None:
    """API keys / secrets may contain spaces, quotes, ``$``, backticks.
    ``shlex.quote`` must wrap them so the shell doesn't reinterpret —
    otherwise the launch command syntax-errors and uvicorn never starts.
    """
    import shlex as _shlex
    payload = "key with spaces and 'quotes' and $(evil)"
    cmd = _build_launch_command(env={"RUNPOD_API_KEY": payload})
    # Authoritative check: the substring matches shlex.quote's output
    # exactly. If shlex.quote ever stops being used, this trips first.
    expected_token = f"RUNPOD_API_KEY={_shlex.quote(payload)}"
    assert expected_token in cmd, (
        f"expected shlex-quoted env token {expected_token!r} in command"
    )


def test_command_with_no_env_omits_assignments() -> None:
    """When env is None or empty, there should be no spurious
    KEY=VALUE between nohup and stdbuf — keeps the command minimal
    for the test/dev case where no provider env is needed."""
    cmd_none = _build_launch_command(env=None)
    cmd_empty = _build_launch_command(env={})
    # Both must still launch uvicorn; just no extra env tokens.
    for cmd in (cmd_none, cmd_empty):
        assert "src.runner.main:app" in cmd
        # Sanity: no leftover empty-value patterns from a buggy join.
        assert "= " not in cmd
        assert "=  " not in cmd


def test_launch_runner_passes_env_through_to_command() -> None:
    """The public API must wire env into the exec_command call,
    not silently drop it."""
    ssh = _SSHStub(success=True, stdout="runner ready")
    launch_runner(ssh, env={"RYOTENKAI_RUNTIME_PROVIDER": "single_node"})  # type: ignore[arg-type]
    assert "RYOTENKAI_RUNTIME_PROVIDER=single_node" in ssh.last_command
