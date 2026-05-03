"""Tests for the ``resolve_run_dir`` path-traversal guard."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import HTTPException

from src.api.dependencies import resolve_run_dir


def _make_run(runs_dir: Path, name: str) -> Path:
    run = runs_dir / name
    run.mkdir(parents=True, exist_ok=True)
    return run


def test_happy_path_returns_resolved_dir(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    _make_run(runs_dir, "ok")

    resolved = resolve_run_dir("ok", runs_dir=runs_dir)
    assert resolved == (runs_dir / "ok").resolve()


def test_dotdot_segment_is_rejected(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True)

    with pytest.raises(HTTPException) as exc_info:
        resolve_run_dir("../etc", runs_dir=runs_dir)
    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "invalid_run_id"


def test_empty_run_id_is_rejected(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True)

    with pytest.raises(HTTPException) as exc_info:
        resolve_run_dir("", runs_dir=runs_dir)
    assert exc_info.value.status_code == 400


def test_symlink_escape_is_rejected(tmp_path: Path) -> None:
    """A symlink pointing outside ``runs_dir`` must not be dereferenced silently.

    Historically ``relative_to`` was called against a possibly-unresolved
    ``runs_dir``. When both sides are fully resolved the escape is caught
    cleanly via ``is_relative_to``.
    """
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    outside = tmp_path / "outside_secret"
    outside.mkdir()

    # runs_dir/escape → ../outside_secret
    (runs_dir / "escape").symlink_to(outside)

    with pytest.raises(HTTPException) as exc_info:
        resolve_run_dir("escape", runs_dir=runs_dir)
    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "run_id_outside_runs_dir"


def test_unresolved_runs_dir_still_safe(tmp_path: Path) -> None:
    """Callers that forget to pre-resolve ``runs_dir`` must still be protected.

    The dep itself resolves both sides; we verify this by passing a symlink
    for ``runs_dir`` and expecting the happy path to work (inside) and the
    escape path to fail (outside).
    """
    real_runs = tmp_path / "real_runs"
    real_runs.mkdir()
    (real_runs / "inside").mkdir()

    link = tmp_path / "runs_link"
    link.symlink_to(real_runs)

    resolved = resolve_run_dir("inside", runs_dir=link)
    assert resolved == (real_runs / "inside").resolve()

    outside = tmp_path / "outside"
    outside.mkdir()
    (real_runs / "escape").symlink_to(outside)

    with pytest.raises(HTTPException):
        resolve_run_dir("escape", runs_dir=link)


def test_nonexistent_returns_404(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True)

    with pytest.raises(HTTPException) as exc_info:
        resolve_run_dir("missing", runs_dir=runs_dir)
    assert exc_info.value.status_code == 404
