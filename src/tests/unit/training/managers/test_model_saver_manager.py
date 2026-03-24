from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock

from src.training.managers.model_saver import ModelSaverEventCallbacks, ModelSaverManager


def test_save_model_saves_model_and_tokenizer_and_calls_callback(tmp_path: Path) -> None:
    called: list[str] = []
    cb = ModelSaverEventCallbacks(on_model_saved=lambda p: called.append(p))
    mgr = ModelSaverManager(create_dirs=True, callbacks=cb)

    model = MagicMock()
    tok = MagicMock()
    out_dir = tmp_path / "out"

    res = mgr.save_model(model, tok, str(out_dir), save_tokenizer=True)
    assert res.is_success()
    assert out_dir.exists()
    model.save_pretrained.assert_called_once()
    tok.save_pretrained.assert_called_once()
    assert called == [str(out_dir)]


def test_save_model_handles_exception(tmp_path: Path) -> None:
    mgr = ModelSaverManager(create_dirs=True)
    model = MagicMock()
    model.save_pretrained.side_effect = RuntimeError("boom")
    tok = MagicMock()

    res = mgr.save_model(model, tok, str(tmp_path / "out"))
    assert res.is_failure()
    assert "boom" in str(res.unwrap_err())


def test_save_checkpoint_naming_priority_step_over_epoch(tmp_path: Path) -> None:
    mgr = ModelSaverManager(create_dirs=True)
    model = MagicMock()

    res = mgr.save_checkpoint(model, str(tmp_path), step=10, epoch=2)
    assert res.is_success()
    assert res.unwrap().endswith("checkpoint-step-10")


def test_cleanup_checkpoints_keeps_last_n_and_calls_callback(tmp_path: Path) -> None:
    called: list[int] = []
    cb = ModelSaverEventCallbacks(on_cleanup_completed=lambda n: called.append(n))
    mgr = ModelSaverManager(create_dirs=True, callbacks=cb)

    base = tmp_path / "ckpts"
    base.mkdir()
    # Create 4 checkpoints with increasing mtime (newest last)
    paths = []
    for i in range(4):
        p = base / f"checkpoint-{i}"
        p.mkdir()
        # ensure mtime ordering
        os.utime(p, (100 + i, 100 + i))
        paths.append(p)

    res = mgr.cleanup_checkpoints(str(base), keep_last_n=2)
    assert res.is_success()
    assert res.unwrap() == 2
    # only 2 newest remain
    remaining = {p.name for p in base.iterdir() if p.is_dir()}
    assert remaining == {"checkpoint-3", "checkpoint-2"}
    assert called == [2]


def test_cleanup_checkpoints_missing_dir_is_ok(tmp_path: Path) -> None:
    mgr = ModelSaverManager(create_dirs=True)
    res = mgr.cleanup_checkpoints(str(tmp_path / "missing"))
    assert res.is_success()
    assert res.unwrap() == 0
