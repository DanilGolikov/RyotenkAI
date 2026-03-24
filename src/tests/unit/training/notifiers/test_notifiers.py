from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.training.notifiers.log import LogNotifier
from src.training.notifiers.marker_file import MarkerFileNotifier


def test_log_notifier_paths_and_branches_do_not_crash() -> None:
    n = LogNotifier()
    n.notify_complete({"output_path": "/tmp/x", "model_name": "m", "strategies": ["sft", "dpo"], "total_phases": 2})
    n.notify_failed("boom", {"error_type": "X", "model_name": "m", "phase": 1})


def test_marker_file_notifier_complete_and_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("time.time", lambda: 123.0)
    n = MarkerFileNotifier(base_path=str(tmp_path))
    n.notify_complete({"output_path": "out"})

    p = tmp_path / MarkerFileNotifier.COMPLETE_MARKER
    assert p.exists()
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["status"] == "complete"
    assert data["timestamp"] == 123.0
    assert n.get_status() == "complete"


def test_marker_file_notifier_failed_cleanup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("time.time", lambda: 123.0)
    n = MarkerFileNotifier(base_path=str(tmp_path))
    n.notify_failed("err", {"error_type": "E"})

    p = tmp_path / MarkerFileNotifier.FAILED_MARKER
    assert p.exists()
    assert n.get_status() == "failed"

    n.cleanup()
    assert not p.exists()
    assert n.get_status() is None
