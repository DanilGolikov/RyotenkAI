from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from datasets import Dataset

import src.training.managers.data_loader as dl
from src.training.managers.data_loader import DataLoaderEventCallbacks, DataLoaderManager


def _ds(n: int) -> Dataset:
    return Dataset.from_dict({"text": [f"t{i}" for i in range(n)]})


def test_load_datasets_local_requires_source_local() -> None:
    cfg = MagicMock()
    dataset_cfg = MagicMock()
    dataset_cfg.get_source_type.return_value = "local"
    dataset_cfg.source_local = None
    cfg.get_primary_dataset.return_value = dataset_cfg

    mgr = DataLoaderManager(cfg)
    res = mgr.load_datasets()
    assert res.is_failure()
    assert "requires source_local" in str(res.unwrap_err())


def test_load_datasets_huggingface_requires_source_hf() -> None:
    cfg = MagicMock()
    dataset_cfg = MagicMock()
    dataset_cfg.get_source_type.return_value = "huggingface"
    dataset_cfg.source_hf = None
    cfg.get_primary_dataset.return_value = dataset_cfg

    mgr = DataLoaderManager(cfg)
    res = mgr.load_datasets()
    assert res.is_failure()
    assert "requires source_hf" in str(res.unwrap_err())


def test_load_datasets_local_happy_path_with_max_samples_and_callbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = MagicMock()

    dataset_cfg = MagicMock()
    dataset_cfg.get_source_type.return_value = "local"
    dataset_cfg.source_local = SimpleNamespace(
        local_paths=SimpleNamespace(train="train.jsonl", eval="eval.jsonl"),
    )
    dataset_cfg.max_samples = 10
    cfg.get_primary_dataset.return_value = dataset_cfg

    # Patch load_dataset to return deterministic Datasets
    def fake_load_dataset(kind, data_files=None, split=None, trust_remote_code=None):
        if data_files == "data/sft/train.jsonl":
            return _ds(20)
        if data_files == "data/sft/eval.jsonl":
            return _ds(5)
        raise AssertionError(f"unexpected data_files={data_files}")

    monkeypatch.setattr(dl, "load_dataset", fake_load_dataset)

    calls: list[tuple[int, int | None]] = []
    cb = DataLoaderEventCallbacks(on_dataset_loaded=lambda tr, ev: calls.append((tr, ev)))

    mgr = DataLoaderManager(cfg, callbacks=cb)
    res = mgr.load_datasets()

    assert res.is_success()
    train_ds, eval_ds = res.unwrap()
    assert len(train_ds) == 10
    assert eval_ds is not None
    # eval limited to max_samples//10 = 1
    assert len(eval_ds) == 1
    assert calls == [(10, 1)]


def test_validate_datasets_empty_train_is_error() -> None:
    cfg = MagicMock()
    mgr = DataLoaderManager(cfg)

    empty = Dataset.from_dict({"text": []})
    res = mgr.validate_datasets(empty, None)
    assert res.is_failure()
    assert "empty" in str(res.unwrap_err()).lower()


def test_validate_datasets_success_calls_callback() -> None:
    cfg = MagicMock()
    called = {"ok": False}
    cb = DataLoaderEventCallbacks(on_dataset_validated=lambda ok: called.__setitem__("ok", ok))
    mgr = DataLoaderManager(cfg, callbacks=cb)

    train = _ds(1)
    res = mgr.validate_datasets(train, None)
    assert res.is_success()
    assert called["ok"] is True
