"""Unit tests for src.pipeline.stages.dataset_validator.split_loader."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.pipeline.stages.dataset_validator.split_loader import DatasetSplitLoader

pytestmark = pytest.mark.unit


@pytest.fixture
def loader_factory() -> MagicMock:
    return MagicMock()


@pytest.fixture
def split_loader(loader_factory: MagicMock) -> DatasetSplitLoader:
    return DatasetSplitLoader(loader_factory=loader_factory)


# ============================================================================
# load_train (HF branch)
# ============================================================================


def test_load_train_hf_no_train_id_returns_none(split_loader, loader_factory):
    ds_cfg = MagicMock()
    ds_cfg.get_source_type.return_value = "huggingface"
    ds_cfg.validations = MagicMock(mode="fast")
    ds_cfg.max_samples = None
    ds_cfg.source_hf = MagicMock(train_id=None, eval_id=None)
    loader_factory.create_for_dataset.return_value = MagicMock()

    assert split_loader.load_train(ds_cfg) is None


def test_load_train_hf_non_iterable_return_returns_none(split_loader, loader_factory, monkeypatch):
    ds_cfg = MagicMock()
    ds_cfg.get_source_type.return_value = "huggingface"
    ds_cfg.validations = MagicMock(mode="fast")
    ds_cfg.max_samples = None
    ds_cfg.source_hf = MagicMock(train_id="org/ds", eval_id=None)
    loader_factory.create_for_dataset.return_value = MagicMock()

    monkeypatch.setattr("datasets.load_dataset", lambda *_a, **_k: object())

    class _Base:
        pass

    monkeypatch.setattr("datasets.IterableDataset", _Base)
    assert split_loader.load_train(ds_cfg) is None


def test_load_train_hf_exception_returns_none(split_loader, loader_factory, monkeypatch):
    ds_cfg = MagicMock()
    ds_cfg.get_source_type.return_value = "huggingface"
    ds_cfg.validations = MagicMock(mode="fast")
    ds_cfg.max_samples = None
    ds_cfg.source_hf = MagicMock(train_id="org/ds", eval_id=None)
    loader_factory.create_for_dataset.return_value = MagicMock()

    monkeypatch.setattr("datasets.load_dataset", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom")))
    assert split_loader.load_train(ds_cfg) is None


# ============================================================================
# load_train (local branch)
# ============================================================================


def test_load_train_local_source_local_none_returns_none(split_loader, loader_factory):
    ds_cfg = MagicMock()
    ds_cfg.get_source_type.return_value = "local"
    ds_cfg.validations = MagicMock(mode="fast")
    ds_cfg.source_local = None
    loader_factory.create_for_dataset.return_value = MagicMock()
    assert split_loader.load_train(ds_cfg) is None


def test_load_train_local_paths_missing_returns_none(split_loader, loader_factory):
    ds_cfg = MagicMock()
    ds_cfg.get_source_type.return_value = "local"
    ds_cfg.validations = MagicMock(mode="fast")
    ds_cfg.source_local = MagicMock(local_paths=MagicMock(train=None, eval=None))
    loader_factory.create_for_dataset.return_value = MagicMock()
    assert split_loader.load_train(ds_cfg) is None


def test_load_train_local_file_not_exists_returns_none(split_loader, loader_factory, tmp_path):
    ds_cfg = MagicMock()
    ds_cfg.get_source_type.return_value = "local"
    ds_cfg.validations = MagicMock(mode="fast")
    ds_cfg.source_local = MagicMock(local_paths=MagicMock(train=str(tmp_path / "missing.jsonl"), eval=None))
    loader_factory.create_for_dataset.return_value = MagicMock()
    assert split_loader.load_train(ds_cfg) is None


def test_load_train_local_fast_mode_samples_when_over_threshold(split_loader, loader_factory, tmp_path):
    file_path = tmp_path / "train.jsonl"
    file_path.write_text("{}", encoding="utf-8")

    ds_cfg = MagicMock()
    ds_cfg.get_source_type.return_value = "local"
    ds_cfg.validations = MagicMock(mode="fast")
    ds_cfg.source_local = MagicMock(local_paths=MagicMock(train=str(file_path), eval=None))

    big_ds = MagicMock()
    big_ds.__len__.return_value = 20000
    sampled = MagicMock()
    big_ds.select.return_value = sampled

    loader = MagicMock()
    loader.load.return_value = big_ds
    loader_factory.create_for_dataset.return_value = loader

    out = split_loader.load_train(ds_cfg)
    assert out is sampled
    big_ds.select.assert_called_once()


# ============================================================================
# get_train_ref / try_load_eval
# ============================================================================


def test_get_train_ref_returns_unknown_on_error(split_loader):
    bad = MagicMock()
    bad.get_source_type.side_effect = Exception("boom")
    assert split_loader.get_train_ref(bad) == "unknown"


def test_get_train_ref_returns_hf_train_id(split_loader):
    hf = MagicMock()
    hf.get_source_type.return_value = "huggingface"
    hf.source_hf = MagicMock(train_id="org/ds-train")
    assert split_loader.get_train_ref(hf) == "org/ds-train"


def test_get_train_ref_returns_local_path(split_loader):
    local = MagicMock()
    local.get_source_type.return_value = "local"
    local.source_hf = None
    local.source_local = MagicMock(local_paths=MagicMock(train="/abs/data/train.jsonl"))
    assert split_loader.get_train_ref(local) == "/abs/data/train.jsonl"


def test_try_load_eval_hf_branch(split_loader, loader_factory, monkeypatch):
    hf = MagicMock()
    hf.get_source_type.return_value = "huggingface"
    hf.source_hf = MagicMock(eval_id="org/ds", train_id="org/ds-train")
    hf.validations = MagicMock(mode="fast")
    hf.max_samples = None
    loader_factory.create_for_dataset.return_value = MagicMock()

    # bypass internal _load by stubbing it on the instance
    monkeypatch.setattr(DatasetSplitLoader, "_load", staticmethod(lambda *_a, **_k: object()))

    ds, ref = split_loader.try_load_eval(hf)
    assert ds is not None
    assert ref == "org/ds"


def test_try_load_eval_hf_no_eval_id_returns_none(split_loader, loader_factory):
    hf = MagicMock()
    hf.get_source_type.return_value = "huggingface"
    hf.source_hf = MagicMock(eval_id=None)
    loader_factory.create_for_dataset.return_value = MagicMock()
    assert split_loader.try_load_eval(hf) == (None, None)


def test_try_load_eval_exception_returns_none(split_loader, loader_factory):
    bad = MagicMock()
    bad.get_source_type.side_effect = Exception("boom")
    loader_factory.create_for_dataset.return_value = MagicMock()
    assert split_loader.try_load_eval(bad) == (None, None)


# ============================================================================
# get_size
# ============================================================================


def test_get_size_iterable_returns_minus_one():
    from datasets import IterableDataset

    iterable = MagicMock(spec=IterableDataset)
    assert DatasetSplitLoader.get_size(iterable) == -1


def test_get_size_dataset_returns_len():
    ds = MagicMock()
    ds.__len__.return_value = 42
    # not IterableDataset
    assert DatasetSplitLoader.get_size(ds) == 42
