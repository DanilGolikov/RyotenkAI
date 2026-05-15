from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ryotenkai_pod.trainer.managers.model_saver import ModelSaverEventCallbacks, ModelSaverManager
from ryotenkai_shared.contracts.problem_details import ErrorCode
from ryotenkai_shared.errors import ModelLoadFailedError

# ---------------------------------------------------------------------------
# save_model — happy path + error
# ---------------------------------------------------------------------------


def test_save_model_returns_path_and_calls_callback(tmp_path: Path) -> None:
    """Happy path: returns str path directly, fires on_model_saved with the same path."""
    called: list[str] = []
    cb = ModelSaverEventCallbacks(on_model_saved=lambda p: called.append(p))
    mgr = ModelSaverManager(create_dirs=True, callbacks=cb)

    model = MagicMock()
    tok = MagicMock()
    out_dir = tmp_path / "out"

    result = mgr.save_model(model, tok, str(out_dir), save_tokenizer=True)

    assert result == str(out_dir)
    assert out_dir.exists()
    model.save_pretrained.assert_called_once()
    tok.save_pretrained.assert_called_once()
    assert called == [str(out_dir)]


def test_save_model_save_tokenizer_false_skips_tokenizer(tmp_path: Path) -> None:
    """save_tokenizer=False skips tokenizer.save_pretrained but still saves model."""
    mgr = ModelSaverManager(create_dirs=True)
    model = MagicMock()
    tok = MagicMock()
    out_dir = tmp_path / "out_no_tok"

    result = mgr.save_model(model, tok, str(out_dir), save_tokenizer=False)

    assert result == str(out_dir)
    model.save_pretrained.assert_called_once()
    tok.save_pretrained.assert_not_called()


def test_save_model_raises_typed_on_exception(tmp_path: Path) -> None:
    """Any underlying exception is translated to ModelLoadFailedError with MODEL_SAVE_FAILED legacy code."""
    mgr = ModelSaverManager(create_dirs=True)
    model = MagicMock()
    model.save_pretrained.side_effect = RuntimeError("boom")
    tok = MagicMock()
    out_dir = tmp_path / "out_fail"

    with pytest.raises(ModelLoadFailedError) as excinfo:
        mgr.save_model(model, tok, str(out_dir))

    assert excinfo.value.code == ErrorCode.MODEL_LOAD_FAILED
    assert "boom" in (excinfo.value.detail or "")
    assert excinfo.value.context.get("legacy_code") == "MODEL_SAVE_FAILED"
    assert excinfo.value.context.get("output_dir") == str(out_dir)
    assert isinstance(excinfo.value.__cause__, RuntimeError)


def test_save_model_no_callback_does_not_crash(tmp_path: Path) -> None:
    """Default callbacks (None) — save still works."""
    mgr = ModelSaverManager(create_dirs=True)
    model = MagicMock()
    tok = MagicMock()
    out_dir = tmp_path / "out_no_cb"

    result = mgr.save_model(model, tok, str(out_dir))

    assert result == str(out_dir)


# ---------------------------------------------------------------------------
# save_checkpoint — naming + error
# ---------------------------------------------------------------------------


def test_save_checkpoint_naming_priority_step_over_epoch(tmp_path: Path) -> None:
    """step takes precedence over epoch in the checkpoint dir name."""
    mgr = ModelSaverManager(create_dirs=True)
    model = MagicMock()

    result = mgr.save_checkpoint(model, str(tmp_path), step=10, epoch=2)
    assert result.endswith("checkpoint-step-10")


def test_save_checkpoint_naming_epoch_when_no_step(tmp_path: Path) -> None:
    """Without step, epoch becomes the suffix."""
    mgr = ModelSaverManager(create_dirs=True)
    model = MagicMock()

    result = mgr.save_checkpoint(model, str(tmp_path), epoch=7)
    assert result.endswith("checkpoint-epoch-7")


def test_save_checkpoint_naming_default(tmp_path: Path) -> None:
    """Without step or epoch, name is just 'checkpoint'."""
    mgr = ModelSaverManager(create_dirs=True)
    model = MagicMock()

    result = mgr.save_checkpoint(model, str(tmp_path))
    assert result.endswith("checkpoint")
    assert not result.endswith("-step-None")


def test_save_checkpoint_raises_typed_on_exception(tmp_path: Path) -> None:
    """save_pretrained failure → ModelLoadFailedError(CHECKPOINT_SAVE_FAILED)."""
    mgr = ModelSaverManager(create_dirs=True)
    model = MagicMock()
    model.save_pretrained.side_effect = OSError("disk full")

    with pytest.raises(ModelLoadFailedError) as excinfo:
        mgr.save_checkpoint(model, str(tmp_path), step=5)

    assert excinfo.value.context.get("legacy_code") == "CHECKPOINT_SAVE_FAILED"
    assert excinfo.value.context.get("step") == 5
    assert excinfo.value.context.get("checkpoint_dir") == str(tmp_path)
    assert isinstance(excinfo.value.__cause__, OSError)


def test_save_checkpoint_fires_callback(tmp_path: Path) -> None:
    """on_checkpoint_saved fires with (path, step, epoch)."""
    captured: list[tuple[str, int | None, int | None]] = []
    cb = ModelSaverEventCallbacks(on_checkpoint_saved=lambda p, s, e: captured.append((p, s, e)))
    mgr = ModelSaverManager(create_dirs=True, callbacks=cb)
    model = MagicMock()

    path = mgr.save_checkpoint(model, str(tmp_path), step=3, epoch=1)
    assert captured == [(path, 3, 1)]


# ---------------------------------------------------------------------------
# cleanup_checkpoints
# ---------------------------------------------------------------------------


def test_cleanup_checkpoints_keeps_last_n_and_calls_callback(tmp_path: Path) -> None:
    """Keep the N newest by mtime; older are deleted."""
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
        os.utime(p, (100 + i, 100 + i))
        paths.append(p)

    deleted = mgr.cleanup_checkpoints(str(base), keep_last_n=2)
    assert deleted == 2
    remaining = {p.name for p in base.iterdir() if p.is_dir()}
    assert remaining == {"checkpoint-3", "checkpoint-2"}
    assert called == [2]


def test_cleanup_checkpoints_missing_dir_returns_zero(tmp_path: Path) -> None:
    """Missing checkpoint dir is not an error — returns 0."""
    mgr = ModelSaverManager(create_dirs=True)
    deleted = mgr.cleanup_checkpoints(str(tmp_path / "missing"))
    assert deleted == 0


def test_cleanup_checkpoints_no_callback_when_nothing_deleted(tmp_path: Path) -> None:
    """When deleted == 0, on_cleanup_completed is NOT fired."""
    called: list[int] = []
    cb = ModelSaverEventCallbacks(on_cleanup_completed=lambda n: called.append(n))
    mgr = ModelSaverManager(create_dirs=True, callbacks=cb)

    base = tmp_path / "ckpts_keep_all"
    base.mkdir()
    # Only 1 checkpoint, keep_last_n=3 → nothing to delete
    (base / "checkpoint-0").mkdir()

    deleted = mgr.cleanup_checkpoints(str(base), keep_last_n=3)
    assert deleted == 0
    assert called == []


def test_cleanup_checkpoints_iterdir_failure_raises_typed(tmp_path: Path) -> None:
    """iterdir failure on the base dir → ModelLoadFailedError(CHECKPOINT_CLEANUP_FAILED)."""
    mgr = ModelSaverManager(create_dirs=True)

    base = tmp_path / "ckpts_fail"
    base.mkdir()
    # Make stat raise so the sort key blows up — simulates underlying IO failure.
    # We pre-create one checkpoint then delete it before sorting reaches stat.
    (base / "checkpoint-x").mkdir()

    # Patch Path.iterdir on this specific path to raise.
    from unittest.mock import patch

    with (
        patch("pathlib.Path.iterdir", side_effect=PermissionError("no access")),
        pytest.raises(ModelLoadFailedError) as excinfo,
    ):
        mgr.cleanup_checkpoints(str(base), keep_last_n=2)

    assert excinfo.value.context.get("legacy_code") == "CHECKPOINT_CLEANUP_FAILED"
    assert excinfo.value.context.get("checkpoint_dir") == str(base)
    assert isinstance(excinfo.value.__cause__, PermissionError)
