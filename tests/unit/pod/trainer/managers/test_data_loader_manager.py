from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from datasets import Dataset

import ryotenkai_pod.trainer.managers.data_loader as dl
from ryotenkai_pod.trainer.managers.data_loader import DataLoaderEventCallbacks, DataLoaderManager
from ryotenkai_shared.contracts.problem_details import ErrorCode
from ryotenkai_shared.errors import (
    DatasetLoadFailedError,
    DatasetValidationFailedError,
)


def _ds(n: int) -> Dataset:
    return Dataset.from_dict({"text": [f"t{i}" for i in range(n)]})


# ---------------------------------------------------------------------------
# load_datasets — error paths
# ---------------------------------------------------------------------------


def test_load_datasets_unknown_source_raises_typed() -> None:
    """Source neither DatasetSourceLocal nor DatasetSourceHF → DatasetLoadFailedError."""
    cfg = MagicMock()
    dataset_cfg = SimpleNamespace(source=MagicMock(kind="alien"))
    cfg.get_primary_dataset.return_value = dataset_cfg

    mgr = DataLoaderManager(cfg)
    with pytest.raises(DatasetLoadFailedError) as excinfo:
        mgr.load_datasets()

    assert excinfo.value.code == ErrorCode.DATASET_LOAD_FAILED
    assert "Unknown dataset source kind" in (excinfo.value.detail or "")
    assert excinfo.value.context.get("legacy_code") == "DATA_LOADER_UNKNOWN_SOURCE_KIND"
    assert excinfo.value.context.get("source_kind") == "alien"


def test_load_datasets_unexpected_exception_wraps(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unexpected exception is wrapped as DatasetLoadFailedError(DATA_LOADER_LOAD_FAILED)."""
    from ryotenkai_shared.config import DatasetSourceLocal
    from ryotenkai_shared.config.datasets.sources import DatasetLocalPaths

    cfg = MagicMock()
    dataset_cfg = SimpleNamespace(
        source=DatasetSourceLocal(local_paths=DatasetLocalPaths(train="x.jsonl")),
        max_samples=None,
    )
    cfg.get_primary_dataset.return_value = dataset_cfg

    def boom(*a, **k):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(dl, "load_dataset", boom)

    mgr = DataLoaderManager(cfg)
    with pytest.raises(DatasetLoadFailedError) as excinfo:
        mgr.load_datasets()

    assert excinfo.value.context.get("legacy_code") == "DATA_LOADER_LOAD_FAILED"
    assert "Dataset loading failed" in (excinfo.value.detail or "")
    assert isinstance(excinfo.value.__cause__, RuntimeError)


# ---------------------------------------------------------------------------
# load_datasets — happy paths (local + HF)
# ---------------------------------------------------------------------------


def test_load_datasets_local_happy_path_with_max_samples_and_callbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Local source returns (train, eval) tuple directly; callback fires once with exact counts."""
    from ryotenkai_shared.config import DatasetSourceLocal
    from ryotenkai_shared.config.datasets.sources import DatasetLocalPaths

    cfg = MagicMock()

    dataset_cfg = SimpleNamespace(
        source=DatasetSourceLocal(
            local_paths=DatasetLocalPaths(train="train.jsonl", eval="eval.jsonl"),
        ),
        max_samples=10,
    )
    cfg.get_primary_dataset.return_value = dataset_cfg

    def fake_load_dataset(kind, data_files=None, split=None, trust_remote_code=None):
        if data_files == "data/train.jsonl":
            return _ds(20)
        if data_files == "data/eval.jsonl":
            return _ds(5)
        raise AssertionError(f"unexpected data_files={data_files}")

    monkeypatch.setattr(dl, "load_dataset", fake_load_dataset)

    calls: list[tuple[int, int | None]] = []
    cb = DataLoaderEventCallbacks(on_dataset_loaded=lambda tr, ev: calls.append((tr, ev)))

    mgr = DataLoaderManager(cfg, callbacks=cb)
    train_ds, eval_ds = mgr.load_datasets()

    assert len(train_ds) == 10  # limited from 20
    assert eval_ds is not None
    # eval limited to max_samples // 10 = 1
    assert len(eval_ds) == 1
    assert calls == [(10, 1)]


def test_load_datasets_local_no_eval(monkeypatch: pytest.MonkeyPatch) -> None:
    """Local source without eval — returns (train, None)."""
    from ryotenkai_shared.config import DatasetSourceLocal
    from ryotenkai_shared.config.datasets.sources import DatasetLocalPaths

    cfg = MagicMock()
    dataset_cfg = SimpleNamespace(
        source=DatasetSourceLocal(local_paths=DatasetLocalPaths(train="t.jsonl")),
        max_samples=None,
    )
    cfg.get_primary_dataset.return_value = dataset_cfg

    monkeypatch.setattr(dl, "load_dataset", lambda *a, **k: _ds(4))  # noqa: ARG005

    mgr = DataLoaderManager(cfg)
    train_ds, eval_ds = mgr.load_datasets()

    assert len(train_ds) == 4
    assert eval_ds is None


def test_load_datasets_hf_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """HF source returns (train, eval) — both loaded via load_dataset."""
    from ryotenkai_shared.config import DatasetSourceHF

    cfg = MagicMock()
    dataset_cfg = SimpleNamespace(
        source=DatasetSourceHF(train_id="org/train", eval_id="org/eval"),
        max_samples=None,
    )
    cfg.get_primary_dataset.return_value = dataset_cfg

    seen: list[str] = []

    def fake_load_dataset(repo_id, split=None, trust_remote_code=None):
        seen.append(repo_id)
        return _ds(3)

    monkeypatch.setattr(dl, "load_dataset", fake_load_dataset)

    mgr = DataLoaderManager(cfg)
    train_ds, eval_ds = mgr.load_datasets()

    assert len(train_ds) == 3
    assert eval_ds is not None
    assert seen == ["org/train", "org/eval"]


def test_load_datasets_hf_train_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """HF source without eval_id — only train is loaded."""
    from ryotenkai_shared.config import DatasetSourceHF

    cfg = MagicMock()
    dataset_cfg = SimpleNamespace(
        source=DatasetSourceHF(train_id="org/train"),
        max_samples=None,
    )
    cfg.get_primary_dataset.return_value = dataset_cfg

    seen: list[str] = []

    def fake_load_dataset(repo_id, split=None, trust_remote_code=None):
        seen.append(repo_id)
        return _ds(2)

    monkeypatch.setattr(dl, "load_dataset", fake_load_dataset)

    mgr = DataLoaderManager(cfg)
    train_ds, eval_ds = mgr.load_datasets()

    assert len(train_ds) == 2
    assert eval_ds is None
    assert seen == ["org/train"]


# ---------------------------------------------------------------------------
# validate_datasets — error paths
# ---------------------------------------------------------------------------


def test_validate_datasets_empty_train_raises_typed() -> None:
    """Empty train → DatasetValidationFailedError(EMPTY_TRAIN)."""
    cfg = SimpleNamespace()
    mgr = DataLoaderManager(cfg)

    empty = Dataset.from_dict({"text": []})
    with pytest.raises(DatasetValidationFailedError) as excinfo:
        mgr.validate_datasets(empty, None)

    assert excinfo.value.context.get("legacy_code") == "DATA_LOADER_EMPTY_TRAIN"
    assert "empty" in (excinfo.value.detail or "").lower()


def test_validate_datasets_empty_eval_raises_typed() -> None:
    """Non-empty train + empty eval → DatasetValidationFailedError(EMPTY_EVAL)."""
    cfg = SimpleNamespace()
    mgr = DataLoaderManager(cfg)

    train = _ds(2)
    empty_eval = Dataset.from_dict({"text": []})
    with pytest.raises(DatasetValidationFailedError) as excinfo:
        mgr.validate_datasets(train, empty_eval)

    assert excinfo.value.context.get("legacy_code") == "DATA_LOADER_EMPTY_EVAL"


# ---------------------------------------------------------------------------
# validate_datasets — happy path
# ---------------------------------------------------------------------------


def test_validate_datasets_success_returns_true_and_calls_callback() -> None:
    """Success returns True directly + fires on_dataset_validated callback."""
    cfg = SimpleNamespace()
    called = {"ok": False}
    cb = DataLoaderEventCallbacks(on_dataset_validated=lambda ok: called.__setitem__("ok", ok))
    mgr = DataLoaderManager(cfg, callbacks=cb)

    train = _ds(1)
    result = mgr.validate_datasets(train, None)
    assert result is True
    assert called["ok"] is True


def test_validate_datasets_success_with_eval() -> None:
    """Success path with eval — eval present + non-empty → True."""
    cfg = SimpleNamespace()
    mgr = DataLoaderManager(cfg)

    train = _ds(3)
    eval_ds = _ds(1)
    assert mgr.validate_datasets(train, eval_ds) is True
