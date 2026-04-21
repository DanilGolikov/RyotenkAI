"""
Additional coverage tests for LogManager.

Targets missed lines:
- 70-71  : _get_local_size_bytes — OSError path
- 86     : _get_remote_size_bytes — first stat command not-success → continue
- 90-93  : _get_remote_size_bytes — wc -c fallback, ValueError continue, return None
- 141-142: download — remote_size is None
- 146-157: download — remote_size < local_size (log rotation/truncation)
- 161-162: download — remote_size == local_size (no new bytes)
- 173-177: download — tail fails → full download fallback
- 186    : download — silent=False after appending delta
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


# ---------------------------------------------------------------------------
# Minimal SSH stub (command-map based)
# ---------------------------------------------------------------------------

@dataclass
class _SSH:
    responses: dict[str, tuple[bool, str, str]] = field(default_factory=dict)

    def exec_command(self, *, command: str, silent: bool = True, **kwargs):
        return self.responses.get(command, (False, "", "not found"))


# ---------------------------------------------------------------------------
# _get_local_size_bytes — OSError path (lines 70-71)
# ---------------------------------------------------------------------------

def test_get_local_size_bytes_oserror(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lm, "get_run_log_layout", lambda: _layout(tmp_path))
    ssh = _SSH()
    mgr = LogManager(ssh)

    # Patch stat to raise a generic OSError (not FileNotFoundError)
    original_stat = Path.stat

    def raising_stat(self, *args, **kwargs):
        if self == mgr._local_path:
            raise OSError("permission denied")
        return original_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", raising_stat)
    result = mgr._get_local_size_bytes()
    assert result == 0


# ---------------------------------------------------------------------------
# _get_remote_size_bytes — various paths
# ---------------------------------------------------------------------------

def test_get_remote_size_bytes_stat_fails_wc_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """stat returns LOG_NOT_FOUND → fall through to wc -c which works."""
    monkeypatch.setattr(lm, "get_run_log_layout", lambda: _layout(tmp_path))
    ssh = _SSH(responses={
        "stat -c%s /workspace/training.log 2>/dev/null || echo 'LOG_NOT_FOUND'": (True, "LOG_NOT_FOUND\n", ""),
        "wc -c < /workspace/training.log 2>/dev/null || echo 'LOG_NOT_FOUND'": (True, "42\n", ""),
    })
    mgr = LogManager(ssh)
    assert mgr._get_remote_size_bytes() == 42


def test_get_remote_size_bytes_stat_not_success_wc_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """stat SSH call fails (not success) → continue; wc -c returns valid size."""
    monkeypatch.setattr(lm, "get_run_log_layout", lambda: _layout(tmp_path))
    ssh = _SSH(responses={
        "stat -c%s /workspace/training.log 2>/dev/null || echo 'LOG_NOT_FOUND'": (False, "", ""),
        "wc -c < /workspace/training.log 2>/dev/null || echo 'LOG_NOT_FOUND'": (True, "100\n", ""),
    })
    mgr = LogManager(ssh)
    assert mgr._get_remote_size_bytes() == 100


def test_get_remote_size_bytes_both_fail_returns_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both stat and wc return LOG_NOT_FOUND → returns None."""
    monkeypatch.setattr(lm, "get_run_log_layout", lambda: _layout(tmp_path))
    ssh = _SSH(responses={
        "stat -c%s /workspace/training.log 2>/dev/null || echo 'LOG_NOT_FOUND'": (True, "LOG_NOT_FOUND\n", ""),
        "wc -c < /workspace/training.log 2>/dev/null || echo 'LOG_NOT_FOUND'": (True, "LOG_NOT_FOUND\n", ""),
    })
    mgr = LogManager(ssh)
    assert mgr._get_remote_size_bytes() is None


def test_get_remote_size_bytes_invalid_output_returns_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both commands return non-numeric output → ValueError continue → None."""
    monkeypatch.setattr(lm, "get_run_log_layout", lambda: _layout(tmp_path))
    ssh = _SSH(responses={
        "stat -c%s /workspace/training.log 2>/dev/null || echo 'LOG_NOT_FOUND'": (True, "not-a-number\n", ""),
        "wc -c < /workspace/training.log 2>/dev/null || echo 'LOG_NOT_FOUND'": (True, "also-bad\n", ""),
    })
    mgr = LogManager(ssh)
    assert mgr._get_remote_size_bytes() is None


# ---------------------------------------------------------------------------
# download() — remote_size is None (lines 141-142)
# ---------------------------------------------------------------------------

def test_download_returns_false_when_remote_size_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Incremental path: local file exists but remote size lookup fails."""
    monkeypatch.setattr(lm, "get_run_log_layout", lambda: _layout(tmp_path))

    # Create a local file so local_size > 0
    local_file = tmp_path / "logs" / LogManager.LOCAL_LOG_NAME
    local_file.parent.mkdir(parents=True, exist_ok=True)
    local_file.write_text("existing content\n", encoding="utf-8")

    ssh = _SSH(responses={
        # Both size commands fail
        "stat -c%s /workspace/training.log 2>/dev/null || echo 'LOG_NOT_FOUND'": (True, "LOG_NOT_FOUND\n", ""),
        "wc -c < /workspace/training.log 2>/dev/null || echo 'LOG_NOT_FOUND'": (True, "LOG_NOT_FOUND\n", ""),
    })
    mgr = LogManager(ssh)
    assert mgr.download() is False


# ---------------------------------------------------------------------------
# download() — remote_size < local_size (log rotation, lines 146-157)
# ---------------------------------------------------------------------------

def test_download_handles_log_rotation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Remote log was rotated (smaller) → re-download full."""
    monkeypatch.setattr(lm, "get_run_log_layout", lambda: _layout(tmp_path))

    old_content = "line1\nline2\nline3\n"
    new_content = "fresh start\n"

    local_file = tmp_path / "logs" / LogManager.LOCAL_LOG_NAME
    local_file.parent.mkdir(parents=True, exist_ok=True)
    local_file.write_text(old_content, encoding="utf-8")

    remote_size = len(new_content.encode("utf-8"))

    ssh = _SSH(responses={
        "stat -c%s /workspace/training.log 2>/dev/null || echo 'LOG_NOT_FOUND'": (True, f"{remote_size}\n", ""),
        "cat /workspace/training.log 2>/dev/null || echo 'LOG_NOT_FOUND'": (True, new_content, ""),
    })
    mgr = LogManager(ssh)
    assert mgr.download() is True
    assert local_file.read_text(encoding="utf-8") == new_content


def test_download_handles_log_rotation_silent_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Log rotation path with silent=False logs info message."""
    monkeypatch.setattr(lm, "get_run_log_layout", lambda: _layout(tmp_path))

    old_content = "aaaa\n" * 10
    new_content = "new\n"

    local_file = tmp_path / "logs" / LogManager.LOCAL_LOG_NAME
    local_file.parent.mkdir(parents=True, exist_ok=True)
    local_file.write_text(old_content, encoding="utf-8")

    remote_size = len(new_content.encode("utf-8"))

    ssh = _SSH(responses={
        "stat -c%s /workspace/training.log 2>/dev/null || echo 'LOG_NOT_FOUND'": (True, f"{remote_size}\n", ""),
        "cat /workspace/training.log 2>/dev/null || echo 'LOG_NOT_FOUND'": (True, new_content, ""),
    })
    mgr = LogManager(ssh)
    assert mgr.download(silent=False) is True
    assert local_file.read_text(encoding="utf-8") == new_content


def test_download_log_rotation_full_download_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Log rotation: remote smaller, but full re-download also fails."""
    monkeypatch.setattr(lm, "get_run_log_layout", lambda: _layout(tmp_path))

    local_file = tmp_path / "logs" / LogManager.LOCAL_LOG_NAME
    local_file.parent.mkdir(parents=True, exist_ok=True)
    local_file.write_text("old long content\n" * 5, encoding="utf-8")

    ssh = _SSH(responses={
        "stat -c%s /workspace/training.log 2>/dev/null || echo 'LOG_NOT_FOUND'": (True, "1\n", ""),
        "cat /workspace/training.log 2>/dev/null || echo 'LOG_NOT_FOUND'": (True, "LOG_NOT_FOUND\n", ""),
    })
    mgr = LogManager(ssh)
    assert mgr.download() is False


# ---------------------------------------------------------------------------
# download() — remote_size == local_size, no new content (lines 161-162)
# ---------------------------------------------------------------------------

def test_download_no_new_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Remote size equals local size → return True without fetching."""
    monkeypatch.setattr(lm, "get_run_log_layout", lambda: _layout(tmp_path))

    content = "same content\n"
    local_file = tmp_path / "logs" / LogManager.LOCAL_LOG_NAME
    local_file.parent.mkdir(parents=True, exist_ok=True)
    local_file.write_text(content, encoding="utf-8")

    size = local_file.stat().st_size

    ssh = _SSH(responses={
        "stat -c%s /workspace/training.log 2>/dev/null || echo 'LOG_NOT_FOUND'": (True, f"{size}\n", ""),
    })
    mgr = LogManager(ssh)
    result = mgr.download()
    assert result is True
    # File must be unchanged
    assert local_file.read_text(encoding="utf-8") == content


# ---------------------------------------------------------------------------
# download() — tail fails → fallback to full download (lines 173-177)
# ---------------------------------------------------------------------------

def test_download_tail_fails_uses_full_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Delta path: tail fails → full re-download via cat."""
    monkeypatch.setattr(lm, "get_run_log_layout", lambda: _layout(tmp_path))

    first_content = "line1\n"
    full_content = "line1\nline2\n"
    delta_bytes = len(full_content.encode("utf-8")) - len(first_content.encode("utf-8"))

    local_file = tmp_path / "logs" / LogManager.LOCAL_LOG_NAME
    local_file.parent.mkdir(parents=True, exist_ok=True)
    local_file.write_text(first_content, encoding="utf-8")

    ssh = _SSH(responses={
        "stat -c%s /workspace/training.log 2>/dev/null || echo 'LOG_NOT_FOUND'": (
            True,
            f"{len(full_content.encode('utf-8'))}\n",
            "",
        ),
        f"tail -c {delta_bytes} /workspace/training.log 2>/dev/null || echo 'LOG_NOT_FOUND'": (
            True,
            "LOG_NOT_FOUND\n",
            "",
        ),
        "cat /workspace/training.log 2>/dev/null || echo 'LOG_NOT_FOUND'": (True, full_content, ""),
    })
    mgr = LogManager(ssh)
    assert mgr.download() is True
    assert local_file.read_text(encoding="utf-8") == full_content


def test_download_tail_not_success_uses_full_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Delta path: tail SSH call fails → full re-download."""
    monkeypatch.setattr(lm, "get_run_log_layout", lambda: _layout(tmp_path))

    first_content = "aa\n"
    full_content = "aa\nbb\n"
    delta_bytes = len(full_content.encode("utf-8")) - len(first_content.encode("utf-8"))

    local_file = tmp_path / "logs" / LogManager.LOCAL_LOG_NAME
    local_file.parent.mkdir(parents=True, exist_ok=True)
    local_file.write_text(first_content, encoding="utf-8")

    ssh = _SSH(responses={
        "stat -c%s /workspace/training.log 2>/dev/null || echo 'LOG_NOT_FOUND'": (
            True,
            f"{len(full_content.encode('utf-8'))}\n",
            "",
        ),
        f"tail -c {delta_bytes} /workspace/training.log 2>/dev/null || echo 'LOG_NOT_FOUND'": (
            False,
            "",
            "ssh error",
        ),
        "cat /workspace/training.log 2>/dev/null || echo 'LOG_NOT_FOUND'": (True, full_content, ""),
    })
    mgr = LogManager(ssh)
    assert mgr.download() is True
    assert local_file.read_text(encoding="utf-8") == full_content


def test_download_tail_fallback_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Delta path: both tail and cat fail → returns False."""
    monkeypatch.setattr(lm, "get_run_log_layout", lambda: _layout(tmp_path))

    first_content = "aa\n"
    full_size = len(first_content.encode("utf-8")) + 3
    delta_bytes = 3

    local_file = tmp_path / "logs" / LogManager.LOCAL_LOG_NAME
    local_file.parent.mkdir(parents=True, exist_ok=True)
    local_file.write_text(first_content, encoding="utf-8")

    ssh = _SSH(responses={
        "stat -c%s /workspace/training.log 2>/dev/null || echo 'LOG_NOT_FOUND'": (True, f"{full_size}\n", ""),
        f"tail -c {delta_bytes} /workspace/training.log 2>/dev/null || echo 'LOG_NOT_FOUND'": (
            True, "LOG_NOT_FOUND\n", ""
        ),
        "cat /workspace/training.log 2>/dev/null || echo 'LOG_NOT_FOUND'": (True, "LOG_NOT_FOUND\n", ""),
    })
    mgr = LogManager(ssh)
    assert mgr.download() is False


# ---------------------------------------------------------------------------
# download() — silent=False after successful delta append (line 186)
# ---------------------------------------------------------------------------

def test_download_silent_false_after_delta_append(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Delta path succeeds with silent=False → info log is emitted."""
    monkeypatch.setattr(lm, "get_run_log_layout", lambda: _layout(tmp_path))

    first_content = "line1\n"
    delta_text = "line2\n"
    full_content = first_content + delta_text
    delta_bytes = len(delta_text.encode("utf-8"))

    local_file = tmp_path / "logs" / LogManager.LOCAL_LOG_NAME
    local_file.parent.mkdir(parents=True, exist_ok=True)
    local_file.write_text(first_content, encoding="utf-8")

    ssh = _SSH(responses={
        "stat -c%s /workspace/training.log 2>/dev/null || echo 'LOG_NOT_FOUND'": (
            True,
            f"{len(full_content.encode('utf-8'))}\n",
            "",
        ),
        f"tail -c {delta_bytes} /workspace/training.log 2>/dev/null || echo 'LOG_NOT_FOUND'": (
            True,
            delta_text,
            "",
        ),
    })
    mgr = LogManager(ssh)
    assert mgr.download(silent=False) is True
    assert local_file.read_text(encoding="utf-8") == full_content


# ---------------------------------------------------------------------------
# Backward compatibility alias
# ---------------------------------------------------------------------------

def test_runpod_log_manager_alias(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.pipeline.stages.managers.log_manager import RunPodLogManager

    monkeypatch.setattr(lm, "get_run_log_layout", lambda: _layout(tmp_path))
    ssh = _SSH()
    mgr = RunPodLogManager(ssh)
    assert isinstance(mgr, LogManager)
