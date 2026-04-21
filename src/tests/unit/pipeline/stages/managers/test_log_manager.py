from __future__ import annotations

from dataclasses import dataclass
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
    responses: dict[str, tuple[bool, str, str]]

    def exec_command(self, *, command: str, silent: bool = True, **kwargs):
        return self.responses.get(command, (False, "", "unknown"))


def test_download_returns_false_when_log_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lm, "get_run_log_layout", lambda: _layout(tmp_path))
    ssh = _SSH(
        responses={
            "cat /workspace/training.log 2>/dev/null || echo 'LOG_NOT_FOUND'": (True, "LOG_NOT_FOUND\n", ""),
        }
    )
    mgr = LogManager(ssh)
    assert mgr.download() is False
    assert not mgr.local_path.exists()


def test_download_writes_file_and_tracks_size(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lm, "get_run_log_layout", lambda: _layout(tmp_path))
    content = "a\nb\n"
    ssh = _SSH(
        responses={
            "cat /workspace/training.log 2>/dev/null || echo 'LOG_NOT_FOUND'": (True, content, ""),
        }
    )
    mgr = LogManager(ssh)
    assert mgr.download(silent=False) is True
    assert mgr.local_path.read_text(encoding="utf-8") == content
    # last_size updated
    assert mgr._last_size == mgr.local_path.stat().st_size


def test_download_appends_only_new_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lm, "get_run_log_layout", lambda: _layout(tmp_path))

    first = "a\n"
    second = "b\n"
    combined = first + second

    # First call: full download via cat.
    # Second call: incremental download via wc -c + tail -c <delta>.
    ssh = _SSH(
        responses={
            "cat /workspace/training.log 2>/dev/null || echo 'LOG_NOT_FOUND'": (True, first, ""),
            "stat -c%s /workspace/training.log 2>/dev/null || echo 'LOG_NOT_FOUND'": (
                True,
                f"{len(combined.encode('utf-8'))}\n",
                "",
            ),
            "tail -c 2 /workspace/training.log 2>/dev/null || echo 'LOG_NOT_FOUND'": (True, second, ""),
        }
    )

    mgr = LogManager(ssh)

    assert mgr.download(silent=True) is True
    assert mgr.local_path.read_text(encoding="utf-8") == first

    assert mgr.download(silent=True) is True
    assert mgr.local_path.read_text(encoding="utf-8") == combined


def test_get_last_lines_returns_empty_on_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lm, "get_run_log_layout", lambda: _layout(tmp_path))
    ssh = _SSH(responses={"tail -n 5 /workspace/training.log 2>/dev/null || echo ''": (False, "", "err")})
    mgr = LogManager(ssh)
    assert mgr.get_last_lines(5) == []


def test_get_last_lines_splits_lines(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lm, "get_run_log_layout", lambda: _layout(tmp_path))
    ssh = _SSH(responses={"tail -n 2 /workspace/training.log 2>/dev/null || echo ''": (True, "x\ny\n", "")})
    mgr = LogManager(ssh)
    assert mgr.get_last_lines(2) == ["x", "y"]


def test_download_on_error_calls_download(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lm, "get_run_log_layout", lambda: _layout(tmp_path))
    ssh = _SSH(responses={"cat /workspace/training.log 2>/dev/null || echo 'LOG_NOT_FOUND'": (True, "ok\n", "")})
    mgr = LogManager(ssh)
    mgr.download_on_error("boom")
    assert mgr.local_path.exists()
