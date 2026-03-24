from __future__ import annotations

from pathlib import Path

from src.tui.live_logs import LiveLogTail


def test_live_log_tail_loads_full_file_and_then_appends(tmp_path: Path) -> None:
    log_path = tmp_path / "pipeline.log"
    log_path.write_text("line 1\nline 2\n", encoding="utf-8")
    tail = LiveLogTail()

    assert tail.load_full(log_path) == ["line 1", "line 2"]

    log_path.write_text("line 1\nline 2\nline 3\n", encoding="utf-8")

    assert tail.read_new_lines() == ["line 3"]


def test_live_log_tail_rewinds_when_file_is_truncated(tmp_path: Path) -> None:
    log_path = tmp_path / "pipeline.log"
    log_path.write_text("line 1\nline 2\nline 3\n", encoding="utf-8")
    tail = LiveLogTail()
    tail.load_full(log_path)

    log_path.write_text("fresh line\n", encoding="utf-8")

    assert tail.read_new_lines() == ["fresh line"]


def test_live_log_tail_returns_empty_for_missing_file(tmp_path: Path) -> None:
    tail = LiveLogTail(path=tmp_path / "missing.log")

    assert tail.read_new_lines() == []
