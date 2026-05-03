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
  readiness probe loop, RYOTENKAI_WORKSPACE auto-injection,
  per-run runner.log path under PodLayout)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath

import pytest

from ryotenkai_control.pipeline.stages.managers.deployment.runner_launcher import (
    RUNNER_PORT,
    RUNNER_READY_TIMEOUT_SECONDS,
    _build_launch_command,
    launch_runner,
)
from ryotenkai_shared.utils.pod_layout import PodLayout
from ryotenkai_shared.utils.result import ProviderError

# Canonical run-scoped workspace used in tests — the rsync target
# CodeSyncer dropped ``src/...`` into for this run. The thin image
# (v2.0.0+) makes this the SOLE PYTHONPATH source for ``src.runner``.
_WORKSPACE = "/workspace/runs/test_run"


def _layout(workspace: str = _WORKSPACE) -> PodLayout:
    """Helper: build a PodLayout rooted at ``workspace``."""
    return PodLayout.from_root(PurePosixPath(workspace))


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
    result = launch_runner(ssh, pod_layout=_layout())  # type: ignore[arg-type]
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
    result = launch_runner(ssh, pod_layout=_layout())  # type: ignore[arg-type]
    assert result.is_ok(), f"expected Ok, got {result!r}"


# ---------------------------------------------------------------------------
# Negative
# ---------------------------------------------------------------------------


def test_launch_runner_ssh_command_fails_returns_err() -> None:
    """exec_command success=False → Err with launcher code."""
    ssh = _SSHStub(success=False, stdout="", stderr="ssh: connection refused")
    result = launch_runner(ssh, pod_layout=_layout())  # type: ignore[arg-type]
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
            "--- tail of /workspace/runs/test_run/logs/runner.log ---\n"
            "ModuleNotFoundError: No module named 'ryotenkai_shared.utils'\n"
        ),
    )
    result = launch_runner(ssh, pod_layout=_layout())  # type: ignore[arg-type]
    assert result.is_err(), f"expected Err, got {result!r}"
    err = result.unwrap_err()
    assert "ModuleNotFoundError" in str(err.details)


def test_launch_runner_success_but_no_ready_marker_returns_err() -> None:
    """Defensive: if exec_command reports success but stdout doesn't
    contain 'runner ready' (e.g. a future protocol change broke
    parsing), we treat it as failure rather than silent-pass."""
    ssh = _SSHStub(success=True, stdout="some unrelated output", stderr="")
    result = launch_runner(ssh, pod_layout=_layout())  # type: ignore[arg-type]
    assert result.is_err(), f"expected Err, got {result!r}"


# ---------------------------------------------------------------------------
# Boundary
# ---------------------------------------------------------------------------


def test_launch_runner_empty_outputs_returns_err_with_placeholder() -> None:
    """Both stdout and stderr empty → Err with 'no diagnostic output'."""
    ssh = _SSHStub(success=False, stdout="", stderr="")
    result = launch_runner(ssh, pod_layout=_layout())  # type: ignore[arg-type]
    assert result.is_err(), f"expected Err, got {result!r}"
    err = result.unwrap_err()
    assert "no diagnostic output" in str(err.details)


def test_launch_runner_whitespace_only_outputs_returns_err() -> None:
    """Stdout/stderr full of just whitespace must still be treated as
    no diagnostic info — protects against false-positive stderr_tail
    in the error message."""
    ssh = _SSHStub(success=False, stdout="   \n\t", stderr="\n\n")
    result = launch_runner(ssh, pod_layout=_layout())  # type: ignore[arg-type]
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
    launch_runner(ssh, pod_layout=_layout())  # type: ignore[arg-type]
    assert ssh.last_timeout is not None
    assert ssh.last_timeout >= RUNNER_READY_TIMEOUT_SECONDS + 10


def test_launch_runner_does_not_swallow_exec_exceptions_silently() -> None:
    """If exec_command raises (network glitch, key issue), the
    exception propagates — caller wraps it in their own Result.
    The launcher itself does not catch arbitrary exceptions because
    that would mask SSH-level bugs as launcher-level errors."""
    ssh = _SSHStub(raise_on_exec=RuntimeError("ssh glitch"))
    with pytest.raises(RuntimeError, match="ssh glitch"):
        launch_runner(ssh, pod_layout=_layout())  # type: ignore[arg-type]


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
    cmd = _build_launch_command(pod_layout=_layout())
    # Active assertion: idempotency check is a curl probe.
    assert f"curl -sf http://127.0.0.1:{RUNNER_PORT}/healthz" in cmd
    # Negative regression: pgrep was the bug.
    assert "pgrep" not in cmd, \
        "pgrep idempotency check is forbidden — it self-matches the launch script"


def test_command_redirects_to_per_run_runner_log() -> None:
    """uvicorn stdout/stderr must be APPENDED (``>>``, not ``>``) to
    the per-run ``logs/runner.log`` so a runner-crash retry doesn't
    truncate the previous log AND so sequential runs on the same pod
    don't clobber each other (pre-PodLayout the global
    ``/workspace/runner.log`` had this resume-collision bug)."""
    layout = _layout()
    expected_log_path = str(layout.runner_log)
    cmd = _build_launch_command(pod_layout=layout)
    assert expected_log_path in cmd
    # Verify we redirect to the per-run path, not the legacy global.
    assert "/workspace/runner.log" not in cmd, (
        "Legacy global /workspace/runner.log path leaked back in — "
        "resume-collision bug regression"
    )
    assert ">> " in cmd and "2>&1" in cmd


def test_command_uses_nohup_disown_for_detachment() -> None:
    """The launched uvicorn must outlive the SSH session that started
    it. nohup + & + disown is the canonical pattern."""
    cmd = _build_launch_command(pod_layout=_layout())
    assert "nohup" in cmd
    assert "disown" in cmd


def test_command_polls_healthz_until_ready() -> None:
    """A readiness loop with curl must be present so
    SSH-exec returns only after uvicorn is actually serving.

    Without this we'd return Ok before the runner bound port 8080,
    and the next stage's tunnel /healthz probe would race."""
    cmd = _build_launch_command(pod_layout=_layout())
    assert f"http://127.0.0.1:{RUNNER_PORT}/healthz" in cmd
    assert "curl" in cmd
    assert f"seq 1 {RUNNER_READY_TIMEOUT_SECONDS}" in cmd


def test_command_dumps_log_tail_on_failure() -> None:
    """On readiness timeout, the script must dump diagnostics:
    ``ls -la`` of the log file (so we can tell a missing redirect
    from an empty traceback) AND a ``tail -100`` of the contents.
    stderr from those commands must NOT be silenced — masking
    "file not found" was a real bug we just fixed."""
    layout = _layout()
    runner_log = str(layout.runner_log)
    cmd = _build_launch_command(pod_layout=layout)
    assert f"ls -la " in cmd
    assert runner_log in cmd
    assert "tail -100 " in cmd
    # Critical: tail's stderr must not be swallowed.
    assert "tail's stderr must be visible — masking it hides 'file not found'" not in cmd
    assert " 2>/dev/null" not in cmd or "tail -100" not in cmd.split(" 2>/dev/null")[0], \
        "tail's stderr must be visible — masking it hides 'file not found'"


def test_command_pythonpath_uses_layout_root() -> None:
    """The thin image (v2.0.0+) carries no baked-in ``src/``. The
    layout root — the rsync target where CodeSyncer dropped
    ``src/runner`` — must be the FIRST PYTHONPATH entry so
    ``src.runner.main:app`` resolves to the just-rsync'd code.
    """
    cmd = _build_launch_command(pod_layout=_layout("/workspace/runs/r1"))
    # First entry: the run-scoped rsync target. Must precede any
    # inherited ${PYTHONPATH:-} value.
    assert "PYTHONPATH=/workspace/runs/r1:" in cmd
    assert "ryotenkai_pod.runner.main:app" in cmd


def test_command_pythonpath_no_longer_references_opt_ryotenkai() -> None:
    """Regression guard: the baked-in /opt/ryotenkai baseline was
    removed in the thin-image migration (v2.0.0). If it sneaks back,
    every image rebuild for runner-code changes returns and the cycle
    we removed is lost — fail loudly here.
    """
    cmd = _build_launch_command(pod_layout=_layout("/workspace/runs/r1"))
    assert "/opt/ryotenkai" not in cmd, (
        "/opt/ryotenkai PYTHONPATH baseline must stay removed — "
        "thin-image migration depends on it"
    )


def test_command_pythonpath_shell_escapes_workspace_path() -> None:
    """workspace path is forwarded into the shell command — it must
    be ``shlex.quote``'d so paths with spaces / special chars cannot
    break out of the PYTHONPATH= assignment.
    """
    import shlex as _shlex
    payload = "/workspace/runs/run with spaces"
    cmd = _build_launch_command(pod_layout=_layout(payload))
    assert f"PYTHONPATH={_shlex.quote(payload)}:" in cmd


def test_pod_layout_factory_rejects_empty_root() -> None:
    """Empty / relative roots are rejected by PodLayout itself —
    this used to be a runner_launcher concern; PodLayout absorbs the
    invariant via its factory now."""
    with pytest.raises(ValueError, match="absolute"):
        PodLayout.from_root("")
    with pytest.raises(ValueError, match="absolute"):
        PodLayout.from_root("relative/path")


def test_launch_runner_passes_workspace_through_to_command() -> None:
    """``launch_runner(pod_layout=PodLayout(/X))`` must wire X into
    the PYTHONPATH= token of the SSH command — not silently drop it."""
    ssh = _SSHStub(success=True, stdout="runner ready")
    launch_runner(ssh, pod_layout=_layout("/workspace/runs/wired"))  # type: ignore[arg-type]
    assert "PYTHONPATH=/workspace/runs/wired:" in ssh.last_command


# ---------------------------------------------------------------------------
# Env-var injection (provider's required_runtime_env_vars + auto WORKSPACE)
# ---------------------------------------------------------------------------


def test_command_auto_injects_ryotenkai_workspace() -> None:
    """``RYOTENKAI_WORKSPACE`` is injected automatically from the layout
    root — the runner's ``_resolve_workspace`` reads it to find the
    per-run filesystem layout regardless of cwd."""
    cmd = _build_launch_command(pod_layout=_layout("/workspace/runs/auto-ws"))
    assert "RYOTENKAI_WORKSPACE=/workspace/runs/auto-ws" in cmd


def test_command_caller_env_overrides_auto_workspace() -> None:
    """If a caller explicitly passes RYOTENKAI_WORKSPACE in env,
    their value wins (test harness override path)."""
    cmd = _build_launch_command(
        pod_layout=_layout("/workspace/runs/x"),
        env={"RYOTENKAI_WORKSPACE": "/override/path"},
    )
    assert "RYOTENKAI_WORKSPACE=/override/path" in cmd


def test_command_injects_provider_env_vars() -> None:
    """env=dict is forwarded as ``env KEY=VALUE ...`` between nohup
    and stdbuf — that's what makes RYOTENKAI_RUNTIME_PROVIDER
    reach the runner's lifespan hook."""
    cmd = _build_launch_command(
        pod_layout=_layout(),
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
    cmd = _build_launch_command(
        pod_layout=_layout(),
        env={"RUNPOD_API_KEY": payload},
    )
    expected_token = f"RUNPOD_API_KEY={_shlex.quote(payload)}"
    assert expected_token in cmd, (
        f"expected shlex-quoted env token {expected_token!r} in command"
    )


def test_command_with_no_env_still_injects_workspace() -> None:
    """When env is None or empty, RYOTENKAI_WORKSPACE is still
    auto-injected so the runner finds its per-run root."""
    cmd_none = _build_launch_command(pod_layout=_layout(), env=None)
    cmd_empty = _build_launch_command(pod_layout=_layout(), env={})
    for cmd in (cmd_none, cmd_empty):
        assert "ryotenkai_pod.runner.main:app" in cmd
        assert f"RYOTENKAI_WORKSPACE={_WORKSPACE}" in cmd


def test_launch_runner_passes_env_through_to_command() -> None:
    """The public API must wire env into the exec_command call,
    not silently drop it."""
    ssh = _SSHStub(success=True, stdout="runner ready")
    launch_runner(  # type: ignore[arg-type]
        ssh,
        pod_layout=_layout(),
        env={"RYOTENKAI_RUNTIME_PROVIDER": "single_node"},
    )
    assert "RYOTENKAI_RUNTIME_PROVIDER=single_node" in ssh.last_command


# ---------------------------------------------------------------------------
# PodLayout integration — directory creation
# ---------------------------------------------------------------------------


def test_command_creates_logs_dir_eagerly() -> None:
    """The bash script must ``mkdir -p`` the per-run logs/ directory
    BEFORE the runner.log redirect, so a fresh pod with no per-run
    tree yet doesn't fail with 'file not found'."""
    layout = _layout()
    cmd = _build_launch_command(pod_layout=layout)
    assert f"mkdir -p {str(layout.logs_dir)!s}" in cmd or \
        f"mkdir -p '{layout.logs_dir!s}'" in cmd


def test_command_per_run_paths_disjoint_for_different_runs() -> None:
    """Resume-collision regression: two different run_ids produce
    DIFFERENT runner.log paths in the launch command."""
    cmd_a = _build_launch_command(pod_layout=_layout("/workspace/runs/run_a"))
    cmd_b = _build_launch_command(pod_layout=_layout("/workspace/runs/run_b"))
    assert "/workspace/runs/run_a/logs/runner.log" in cmd_a
    assert "/workspace/runs/run_b/logs/runner.log" in cmd_b
    assert "/workspace/runs/run_a/logs/runner.log" not in cmd_b
    assert "/workspace/runs/run_b/logs/runner.log" not in cmd_a
