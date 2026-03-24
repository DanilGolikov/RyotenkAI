"""
Unit tests for src/training/orchestrator/dataset_loader.py (NEW schema).

Focus:
- local runtime loads from auto-generated data/{strategy_type}/{basename(local_paths.*)}
- errors when auto-generated train file is missing
- optional eval loading
- HF loads use source_hf.train_id/eval_id (mocked)
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.training.orchestrator.dataset_loader import DatasetLoader


def _mk_config_for_local(local_train: str, local_eval: str | None = None) -> MagicMock:
    cfg = MagicMock()
    ds = MagicMock()
    ds.get_source_type.return_value = "local"
    ds.max_samples = None
    ds.source_local = MagicMock()
    ds.source_local.local_paths = MagicMock()
    ds.source_local.local_paths.train = local_train
    ds.source_local.local_paths.eval = local_eval
    cfg.get_dataset_for_strategy.return_value = ds
    return cfg


def _mk_config_for_hf(train_id: str, eval_id: str | None = None) -> MagicMock:
    cfg = MagicMock()
    ds = MagicMock()
    ds.get_source_type.return_value = "huggingface"
    ds.max_samples = None
    ds.source_hf = MagicMock()
    ds.source_hf.train_id = train_id
    ds.source_hf.eval_id = eval_id
    cfg.get_dataset_for_strategy.return_value = ds
    return cfg


def test_local_missing_train_file_returns_err() -> None:
    cfg = _mk_config_for_local(local_train="/any/train.jsonl")
    loader = DatasetLoader(cfg)
    phase = MagicMock()
    phase.strategy_type = "sft"
    res = loader.load_for_phase(phase)
    assert res.is_failure()
    err = str(res.unwrap_err())
    assert "Dataset file not found" in err
    assert "data/sft/train.jsonl" in err


@patch("src.training.orchestrator.dataset_loader.load_dataset")
def test_local_loads_train_from_auto_generated_path(
    mock_load_dataset: MagicMock,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "data" / "sft").mkdir(parents=True)
    (tmp_path / "data" / "sft" / "train.jsonl").write_text('{"text":"x"}\n', encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    cfg = _mk_config_for_local(local_train="/any/train.jsonl")
    loader = DatasetLoader(cfg)
    phase = MagicMock()
    phase.strategy_type = "sft"

    fake_ds = MagicMock()
    fake_ds.__len__.return_value = 1
    mock_load_dataset.return_value = fake_ds

    res = loader.load_for_phase(phase)
    assert res.is_success()
    train, ev = res.unwrap()
    assert train == fake_ds
    assert ev is None


@patch("src.training.orchestrator.dataset_loader.load_dataset")
def test_local_loads_eval_when_present(
    mock_load_dataset: MagicMock,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "data" / "sft").mkdir(parents=True)
    (tmp_path / "data" / "sft" / "train.jsonl").write_text('{"text":"x"}\n', encoding="utf-8")
    (tmp_path / "data" / "sft" / "eval.jsonl").write_text('{"text":"y"}\n', encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    cfg = _mk_config_for_local(local_train="/any/train.jsonl", local_eval="/any/eval.jsonl")
    loader = DatasetLoader(cfg)
    phase = MagicMock()
    phase.strategy_type = "sft"

    train_ds = MagicMock()
    train_ds.__len__.return_value = 1
    eval_ds = MagicMock()
    eval_ds.__len__.return_value = 1

    # First call for train, second for eval
    mock_load_dataset.side_effect = [train_ds, eval_ds]

    res = loader.load_for_phase(phase)
    assert res.is_success()
    train, ev = res.unwrap()
    assert train == train_ds
    assert ev == eval_ds


@patch("src.training.orchestrator.dataset_loader.load_dataset")
def test_hf_loads_train_and_optional_eval(mock_load_dataset: MagicMock) -> None:
    cfg = _mk_config_for_hf("org/train", eval_id="org/eval")
    loader = DatasetLoader(cfg)
    phase = MagicMock()

    train_ds = MagicMock()
    train_ds.__len__.return_value = 10
    eval_ds = MagicMock()
    eval_ds.__len__.return_value = 5
    mock_load_dataset.side_effect = [train_ds, eval_ds]

    res = loader.load_for_phase(phase)
    assert res.is_success()
    train, ev = res.unwrap()
    assert train == train_ds
    assert ev == eval_ds


