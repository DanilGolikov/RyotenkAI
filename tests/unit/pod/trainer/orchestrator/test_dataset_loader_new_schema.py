"""
Unit tests for packages/pod/.../orchestrator/dataset_loader.py (NEW schema).

Focus:
- local runtime loads from data/{basename(local_paths.*)} (flat layout —
  matches the FileUploader → POST /api/v1/files/upload contract)
- missing-train-file raises DatasetLoadFailedError (Batch 14: raise-based)
- optional eval loading
- HF loads use source_hf.train_id/eval_id (mocked)

Path-resolution contract (post 2026-05-07 bugfix):
  Pod-side dataset files live at ``<run_dir>/data/<basename>`` — flat
  layout, no strategy-type prefix. The previous ``data/{strategy}/...``
  layout never matched the actual upload location.
"""

from __future__ import annotations

from types import SimpleNamespace

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ryotenkai_pod.trainer.orchestrator.dataset_loader import DatasetLoader
from ryotenkai_shared.config import DatasetSourceHF, DatasetSourceLocal
from ryotenkai_shared.config.datasets.sources import DatasetLocalPaths
from ryotenkai_shared.errors import DatasetLoadFailedError


def _mk_config_for_local(local_train: str, local_eval: str | None = None) -> MagicMock:
    cfg = MagicMock()
    ds = SimpleNamespace(max_samples=None, source=DatasetSourceLocal(
        local_paths=DatasetLocalPaths(train=local_train, eval=local_eval),
    ))
    cfg.get_dataset_for_strategy.return_value = ds
    return cfg


def _mk_config_for_hf(train_id: str, eval_id: str | None = None) -> MagicMock:
    cfg = MagicMock()
    ds = SimpleNamespace(max_samples=None, source=DatasetSourceHF(train_id=train_id, eval_id=eval_id))
    cfg.get_dataset_for_strategy.return_value = ds
    return cfg


def test_local_missing_train_file_raises() -> None:
    cfg = _mk_config_for_local(local_train="/any/train.jsonl")
    loader = DatasetLoader(cfg)
    phase = SimpleNamespace(strategy_type="sft")
    with pytest.raises(DatasetLoadFailedError) as exc_info:
        loader.load_for_phase(phase)
    err = exc_info.value.detail or ""
    assert "Dataset file not found" in err
    # Flat layout: data/<basename>, NOT data/{strategy}/<basename>.
    assert "data/train.jsonl" in err
    assert exc_info.value.context["legacy_code"] == "DATA_LOADER_FILE_NOT_FOUND"


@patch("ryotenkai_pod.trainer.orchestrator.dataset_loader.load_dataset")
def test_local_loads_train_from_flat_path(
    mock_load_dataset: MagicMock,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Flat layout — NO strategy subdir.
    (tmp_path / "data").mkdir(parents=True)
    (tmp_path / "data" / "train.jsonl").write_text('{"text":"x"}\n', encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    cfg = _mk_config_for_local(local_train="/any/train.jsonl")
    loader = DatasetLoader(cfg)
    phase = SimpleNamespace(strategy_type="sft")

    fake_ds = MagicMock()
    fake_ds.__len__.return_value = 1
    mock_load_dataset.return_value = fake_ds

    train, ev = loader.load_for_phase(phase)
    assert train == fake_ds
    assert ev is None


@patch("ryotenkai_pod.trainer.orchestrator.dataset_loader.load_dataset")
def test_local_loads_eval_when_present(
    mock_load_dataset: MagicMock,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Flat layout — NO strategy subdir.
    (tmp_path / "data").mkdir(parents=True)
    (tmp_path / "data" / "train.jsonl").write_text('{"text":"x"}\n', encoding="utf-8")
    (tmp_path / "data" / "eval.jsonl").write_text('{"text":"y"}\n', encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    cfg = _mk_config_for_local(local_train="/any/train.jsonl", local_eval="/any/eval.jsonl")
    loader = DatasetLoader(cfg)
    phase = SimpleNamespace(strategy_type="sft")

    train_ds = MagicMock()
    train_ds.__len__.return_value = 1
    eval_ds = MagicMock()
    eval_ds.__len__.return_value = 1

    # First call for train, second for eval
    mock_load_dataset.side_effect = [train_ds, eval_ds]

    train, ev = loader.load_for_phase(phase)
    assert train == train_ds
    assert ev == eval_ds


@patch("ryotenkai_pod.trainer.orchestrator.dataset_loader.load_dataset")
def test_hf_loads_train_and_optional_eval(mock_load_dataset: MagicMock) -> None:
    cfg = _mk_config_for_hf("org/train", eval_id="org/eval")
    loader = DatasetLoader(cfg)
    phase = SimpleNamespace()

    train_ds = MagicMock()
    train_ds.__len__.return_value = 10
    eval_ds = MagicMock()
    eval_ds.__len__.return_value = 5
    mock_load_dataset.side_effect = [train_ds, eval_ds]

    train, ev = loader.load_for_phase(phase)
    assert train == train_ds
    assert ev == eval_ds


class TestLoadForPhaseRaiseContract:
    """7-class coverage for the new raise-based DatasetLoader.load_for_phase."""

    @patch("ryotenkai_pod.trainer.orchestrator.dataset_loader.load_dataset")
    def test_positive_returns_tuple_train_and_optional_eval(
        self, mock_load_dataset: MagicMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        (tmp_path / "data").mkdir()
        (tmp_path / "data" / "t.jsonl").write_text('{"text":"x"}\n')
        monkeypatch.chdir(tmp_path)
        cfg = _mk_config_for_local("/any/t.jsonl")
        fake_ds = MagicMock()
        fake_ds.__len__.return_value = 1
        mock_load_dataset.return_value = fake_ds
        out = DatasetLoader(cfg).load_for_phase(SimpleNamespace(strategy_type="sft"))
        # Tuple shape (train, eval) — not Result-wrapped.
        assert isinstance(out, tuple) and len(out) == 2

    def test_negative_unknown_source_kind_raises(self) -> None:
        cfg = MagicMock()
        # Source with an unsupported kind that is neither HF nor Local.
        cfg.get_dataset_for_strategy.return_value = SimpleNamespace(
            max_samples=None, source=SimpleNamespace(kind="weird"),
        )
        with pytest.raises(DatasetLoadFailedError) as exc:
            DatasetLoader(cfg).load_for_phase(SimpleNamespace(strategy_type="sft"))
        assert exc.value.context["legacy_code"] == "DATA_LOADER_UNKNOWN_SOURCE"

    def test_boundary_key_error_during_get_dataset_raises_typed(self) -> None:
        cfg = MagicMock()
        cfg.get_dataset_for_strategy.side_effect = KeyError("phase-x")
        with pytest.raises(DatasetLoadFailedError) as exc:
            DatasetLoader(cfg).load_for_phase(SimpleNamespace(strategy_type="sft", dataset="missing"))
        assert exc.value.context["legacy_code"] == "DATA_LOADER_DATASET_NOT_FOUND"
        assert exc.value.context["phase_dataset"] == "missing"

    def test_invariant_no_result_wrapper_on_success(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        (tmp_path / "data").mkdir()
        (tmp_path / "data" / "t.jsonl").write_text('{"text":"x"}\n')
        monkeypatch.chdir(tmp_path)
        cfg = _mk_config_for_local("/any/t.jsonl")
        with patch("ryotenkai_pod.trainer.orchestrator.dataset_loader.load_dataset") as mock_load:
            mock_load.return_value = MagicMock(__len__=lambda self: 1)
            out = DatasetLoader(cfg).load_for_phase(SimpleNamespace(strategy_type="sft"))
        assert not hasattr(out, "is_failure")
        assert not hasattr(out, "unwrap")

    def test_dependency_error_load_dataset_failure_wraps_in_typed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        (tmp_path / "data").mkdir()
        (tmp_path / "data" / "t.jsonl").write_text('{"text":"x"}\n')
        monkeypatch.chdir(tmp_path)
        cfg = _mk_config_for_local("/any/t.jsonl")
        with patch(
            "ryotenkai_pod.trainer.orchestrator.dataset_loader.load_dataset",
            side_effect=RuntimeError("hf-down"),
        ):
            with pytest.raises(DatasetLoadFailedError) as exc:
                DatasetLoader(cfg).load_for_phase(SimpleNamespace(strategy_type="sft"))
        assert exc.value.context["legacy_code"] == "DATA_LOADER_LOAD_FAILED"
        assert isinstance(exc.value.__cause__, RuntimeError)

    def test_regression_file_not_found_legacy_code_preserved(self) -> None:
        cfg = _mk_config_for_local(local_train="/any/x.jsonl")
        with pytest.raises(DatasetLoadFailedError) as exc:
            DatasetLoader(cfg).load_for_phase(SimpleNamespace(strategy_type="sft"))
        # The old DATA_LOADER_FILE_NOT_FOUND code must remain on context so
        # downstream listeners that pattern-match on it stay green.
        assert exc.value.context["legacy_code"] == "DATA_LOADER_FILE_NOT_FOUND"

    def test_combinatorial_max_samples_applied_after_load(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When ``max_samples`` is set and load succeeds, ``select`` is
        invoked on the train (and eval) datasets — no exception path."""
        (tmp_path / "data").mkdir()
        (tmp_path / "data" / "t.jsonl").write_text('{"text":"x"}\n')
        monkeypatch.chdir(tmp_path)
        cfg = MagicMock()
        ds_cfg = SimpleNamespace(
            max_samples=5,
            source=DatasetSourceLocal(local_paths=DatasetLocalPaths(train="/any/t.jsonl")),
        )
        cfg.get_dataset_for_strategy.return_value = ds_cfg

        train_ds = MagicMock()
        train_ds.__len__.return_value = 100
        sliced = MagicMock()
        sliced.__len__.return_value = 5
        train_ds.select.return_value = sliced

        with patch(
            "ryotenkai_pod.trainer.orchestrator.dataset_loader.load_dataset",
            return_value=train_ds,
        ):
            t, e = DatasetLoader(cfg).load_for_phase(SimpleNamespace(strategy_type="sft"))
        train_ds.select.assert_called_once()
        assert t is sliced
        assert e is None
