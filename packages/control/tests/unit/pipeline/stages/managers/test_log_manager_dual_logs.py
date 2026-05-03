"""
LogManager — dual remote-log support (training.log + runner.log).

Covers the keyword-only ``local_path`` argument added to support
pulling the in-pod uvicorn ``runner.log`` alongside the existing
``training.log``. The default-arg path MUST stay byte-identical to
the previous behavior so existing call-sites keep working.

Categories:
* Positive — explicit local_path override; defaults unchanged
* Negative — remote file missing returns False without crash
* Boundary — empty local_path/None handled; two managers are
  independent and don't trample each other's _last_size
* Invariants — file lands at the path the caller asked for, not the
  layout default, when override is given
* Regression — keyword-only enforcement on local_path; positional use
  is rejected at runtime
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

import src.pipeline.stages.managers.log_manager as lm
from src.pipeline.stages.managers.log_manager import LogManager
from src.utils.logs_layout import LogLayout


def _layout(attempt_dir: Path) -> LogLayout:
    layout = LogLayout(attempt_dir)
    layout.ensure_logs_dir()
    return layout


@dataclass
class _SSH:
    """Map exec_command -> (success, stdout, stderr)."""
    responses: dict[str, tuple[bool, str, str]] = field(default_factory=dict)

    def exec_command(self, *, command: str, silent: bool = True, **kwargs):
        return self.responses.get(command, (False, "", "not found"))


# ---------------------------------------------------------------------------
# Positive
# ---------------------------------------------------------------------------


def test_default_local_path_falls_back_to_layout_training_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No local_path → manager writes to <attempt>/logs/training.log."""
    monkeypatch.setattr(lm, "get_run_log_layout", lambda: _layout(tmp_path))
    ssh = _SSH()
    mgr = LogManager(ssh)

    expected = tmp_path / "logs" / "training.log"
    assert mgr.local_path == expected


def test_explicit_local_path_overrides_layout_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """local_path keyword arg replaces the layout-derived default."""
    monkeypatch.setattr(lm, "get_run_log_layout", lambda: _layout(tmp_path))
    ssh = _SSH()
    custom = tmp_path / "logs" / "runner.log"
    mgr = LogManager(ssh, local_path=custom)

    assert mgr.local_path == custom


def test_explicit_local_path_writes_full_content_to_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """download() writes to local_path, not to layout default."""
    monkeypatch.setattr(lm, "get_run_log_layout", lambda: _layout(tmp_path))
    custom = tmp_path / "logs" / "runner.log"
    payload = "uvicorn boot line 1\nuvicorn boot line 2\n"
    ssh = _SSH(responses={
        "cat /workspace/runner.log 2>/dev/null || echo 'LOG_NOT_FOUND'": (True, payload, ""),
    })
    mgr = LogManager(ssh, remote_path="/workspace/runner.log", local_path=custom)

    assert mgr.download() is True
    assert custom.read_text() == payload
    # And the default-layout file was NOT created by this manager.
    default = tmp_path / "logs" / "training.log"
    assert not default.exists()


# ---------------------------------------------------------------------------
# Boundary — two managers, two files, no cross-talk
# ---------------------------------------------------------------------------


def test_two_managers_with_different_local_paths_are_independent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A LogManager for runner.log doesn't perturb the training.log instance."""
    monkeypatch.setattr(lm, "get_run_log_layout", lambda: _layout(tmp_path))
    training_payload = "epoch 0 step 0 loss 1.23\n"
    runner_payload = "INFO:uvicorn:Started server process\n"
    ssh = _SSH(responses={
        "cat /workspace/training.log 2>/dev/null || echo 'LOG_NOT_FOUND'": (True, training_payload, ""),
        "cat /workspace/runner.log 2>/dev/null || echo 'LOG_NOT_FOUND'": (True, runner_payload, ""),
    })

    training_mgr = LogManager(ssh)  # default
    runner_mgr = LogManager(
        ssh,
        remote_path="/workspace/runner.log",
        local_path=tmp_path / "logs" / "runner.log",
    )

    assert training_mgr.download() is True
    assert runner_mgr.download() is True

    assert (tmp_path / "logs" / "training.log").read_text() == training_payload
    assert (tmp_path / "logs" / "runner.log").read_text() == runner_payload


# ---------------------------------------------------------------------------
# Negative — missing remote file
# ---------------------------------------------------------------------------


def test_runner_log_missing_remote_returns_false_no_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If /workspace/runner.log doesn't exist on the pod, return False quietly."""
    monkeypatch.setattr(lm, "get_run_log_layout", lambda: _layout(tmp_path))
    ssh = _SSH(responses={
        "cat /workspace/runner.log 2>/dev/null || echo 'LOG_NOT_FOUND'": (True, "LOG_NOT_FOUND\n", ""),
    })
    mgr = LogManager(
        ssh,
        remote_path="/workspace/runner.log",
        local_path=tmp_path / "logs" / "runner.log",
    )

    assert mgr.download() is False
    assert not (tmp_path / "logs" / "runner.log").exists()


# ---------------------------------------------------------------------------
# Regression — local_path is keyword-only
# ---------------------------------------------------------------------------


def test_local_path_is_keyword_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Positional invocation of local_path must raise TypeError.

    Prevents accidental call-site coupling to argument order — adding
    a positional arg would silently break existing two-arg callers.
    """
    monkeypatch.setattr(lm, "get_run_log_layout", lambda: _layout(tmp_path))
    ssh = _SSH()
    custom = tmp_path / "logs" / "runner.log"

    with pytest.raises(TypeError):
        # Three positional args: ssh, remote_path, local_path → must reject.
        LogManager(ssh, "/workspace/runner.log", custom)  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Invariant — last_size tracking is per-instance
# ---------------------------------------------------------------------------


def test_last_size_isolated_per_manager_instance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two managers writing to different files keep _last_size independently."""
    monkeypatch.setattr(lm, "get_run_log_layout", lambda: _layout(tmp_path))
    ssh = _SSH(responses={
        "cat /workspace/training.log 2>/dev/null || echo 'LOG_NOT_FOUND'": (True, "x" * 50, ""),
        "cat /workspace/runner.log 2>/dev/null || echo 'LOG_NOT_FOUND'": (True, "y" * 100, ""),
    })

    training_mgr = LogManager(ssh)
    runner_mgr = LogManager(
        ssh,
        remote_path="/workspace/runner.log",
        local_path=tmp_path / "logs" / "runner.log",
    )

    training_mgr.download()
    runner_mgr.download()

    # Each manager's last_size reflects its OWN local file size.
    assert training_mgr._last_size == 50
    assert runner_mgr._last_size == 100
