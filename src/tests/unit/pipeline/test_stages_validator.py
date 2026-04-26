"""
Unit tests for DatasetValidator stage (NEW dataset schema).

We focus on:
- plugin loading (defaults vs configured)
- critical_failures semantics (critical vs advisory)
- HF streaming loader call shape (train_id)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.data.validation.base import ValidationResult
from src.pipeline.stages.dataset_validator import DatasetValidator, DatasetValidatorEventCallbacks
from src.utils.config import DatasetConfig, PipelineConfig
from src.utils.result import DatasetError, Err, Ok


def _mk_primary_only_config(ds: DatasetConfig) -> MagicMock:
    cfg = MagicMock(spec=PipelineConfig)
    cfg.get_primary_dataset.return_value = ds
    cfg.training = MagicMock()
    cfg.training.strategies = []
    return cfg


def _local_ds(path: str, *, plugins: list[dict] | None = None, critical_failures: int = 1) -> DatasetConfig:
    return DatasetConfig(
        source_type="local",
        source_local={"local_paths": {"train": path, "eval": None}},
        validations={"plugins": plugins or [], "mode": "fast", "critical_failures": critical_failures},
    )


def _hf_ds(train_id: str, *, plugins: list[dict] | None = None, critical_failures: int = 1) -> DatasetConfig:
    return DatasetConfig(
        source_type="huggingface",
        source_hf={"train_id": train_id, "eval_id": None},
        validations={"plugins": plugins or [], "mode": "fast", "critical_failures": critical_failures},
    )


def test_execute_returns_err_on_critical_failure(tmp_path) -> None:
    # Create tiny dataset (will fail min_samples threshold)
    dataset_file = tmp_path / "train.jsonl"
    dataset_file.write_text('{"text": "x"}\n', encoding="utf-8")

    ds = _local_ds(
        str(dataset_file),
        plugins=[{"id": "min_samples_main", "plugin": "min_samples", "params": {"threshold": 100}, "apply_to": ["train"]}],
        critical_failures=1,
    )
    cfg = _mk_primary_only_config(ds)

    from src.community.catalog import catalog

    catalog.reload()

    validator = DatasetValidator(cfg)
    res = validator.execute({})
    assert res.is_failure()


def test_execute_returns_ok_with_failed_status_in_advisory_mode(tmp_path) -> None:
    dataset_file = tmp_path / "train.jsonl"
    dataset_file.write_text('{"text": "x"}\n', encoding="utf-8")

    ds = _local_ds(
        str(dataset_file),
        plugins=[{"id": "min_samples_main", "plugin": "min_samples", "params": {"threshold": 100}, "apply_to": ["train"]}],
        critical_failures=0,
    )
    cfg = _mk_primary_only_config(ds)

    from src.community.catalog import catalog

    catalog.reload()

    validator = DatasetValidator(cfg)
    res = validator.execute({})
    assert res.is_success()
    ctx = res.unwrap()
    assert ctx["validation_status"] == "failed"


def test_execute_returns_ok_when_failed_plugins_below_critical_threshold(tmp_path) -> None:
    dataset_file = tmp_path / "train.jsonl"
    dataset_file.write_text('{"text": "x"}\n', encoding="utf-8")

    ds = _local_ds(
        str(dataset_file),
        plugins=[{"id": "min_samples_main", "plugin": "min_samples", "params": {"threshold": 100}, "apply_to": ["train"]}],
        critical_failures=2,
    )
    cfg = _mk_primary_only_config(ds)

    from src.community.catalog import catalog

    catalog.reload()

    validator = DatasetValidator(cfg)
    res = validator.execute({})
    assert res.is_success()
    ctx = res.unwrap()
    assert ctx["validation_status"] == "failed"


@patch("datasets.load_dataset")
def test_hf_streaming_fast_uses_train_id(mock_load_dataset: MagicMock) -> None:
    # Mock HF iterable dataset object (IterableDataset type check happens)
    class _FakeIterable:  # minimal duck-typing
        def take(self, n):
            return self

    mock_load_dataset.return_value = _FakeIterable()

    ds = _hf_ds("test/dataset", plugins=[], critical_failures=1)
    cfg = _mk_primary_only_config(ds)

    # Mock loader with token
    loader_factory = MagicMock()
    loader = MagicMock()
    loader.token = "test_token"
    loader_factory.create_for_dataset.return_value = loader

    with patch("src.pipeline.stages.dataset_validator.stage.DatasetLoaderFactory", return_value=loader_factory):
        validator = DatasetValidator(cfg)
        _ = validator.execute({})

    # Ensure load_dataset called with train_id and streaming=True
    args, kwargs = mock_load_dataset.call_args
    assert args[0] == "test/dataset"
    assert kwargs["streaming"] is True
    assert kwargs["token"] == "test_token"


class TestDatasetValidatorAdditionalCoverage:
    def test_execute_skips_when_no_datasets_and_tqdm_init_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cfg = _mk_primary_only_config(_local_ds("data/train.jsonl", plugins=[], critical_failures=0))
        v = DatasetValidator(cfg)

        # Force tqdm init to fail -> should be swallowed
        import tqdm

        monkeypatch.setattr(tqdm, "tqdm", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
        monkeypatch.setattr(v, "_get_datasets_to_validate", lambda: {})

        res = v.execute({})
        assert res.is_success()
        assert res.unwrap()["validation_status"] == "skipped"

    def test_execute_fail_fast_cancels_remaining_futures(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ds1 = MagicMock()
        ds1.validations = MagicMock(mode="fast", critical_failures=1)
        ds2 = MagicMock()
        ds2.validations = MagicMock(mode="fast", critical_failures=1)

        cfg = _mk_primary_only_config(_local_ds("data/train.jsonl", plugins=[], critical_failures=1))
        callbacks = DatasetValidatorEventCallbacks(on_dataset_scheduled=MagicMock())
        v = DatasetValidator(cfg, callbacks=callbacks)
        monkeypatch.setattr(v, "_get_datasets_to_validate", lambda: {"d1": (ds1, []), "d2": (ds2, [])})
        monkeypatch.setattr(v._split_loader, "get_train_ref", lambda c: "ref")  # noqa: ARG005

        class _F:
            def __init__(self, *, value=None, exc: Exception | None = None, done: bool = True):
                self._value = value
                self._exc = exc
                self._done = done
                self.cancelled = False

            def result(self):
                if self._exc:
                    raise self._exc
                return self._value

            def done(self) -> bool:
                return self._done

            def cancel(self) -> bool:
                self.cancelled = True
                self._done = True
                return True

        f1 = _F(value=Err(DatasetError(message="boom", code="DATASET_VALIDATION_CRITICAL_FAILURE")), done=True)
        f2 = _F(value=Ok({"ok": True}), done=False)
        futures_queue = [f1, f2]

        class _Exec:
            def __init__(self, futures):
                self._futures = list(futures)

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def submit(self, fn, *args, **kwargs):
                _ = fn, args, kwargs
                return self._futures.pop(0)

        monkeypatch.setattr("concurrent.futures.ThreadPoolExecutor", lambda max_workers: _Exec(futures_queue))
        monkeypatch.setattr("concurrent.futures.as_completed", lambda fut_map: list(fut_map.keys()))

        res = v.execute({})
        assert res.is_failure()
        assert f2.cancelled is True
        assert callbacks.on_dataset_scheduled.call_count == 2

    def test_execute_crash_is_treated_as_critical_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ds1 = MagicMock()
        ds1.validations = MagicMock(mode="fast", critical_failures=1)
        ds2 = MagicMock()
        ds2.validations = MagicMock(mode="fast", critical_failures=1)

        cfg = _mk_primary_only_config(_local_ds("data/train.jsonl", plugins=[], critical_failures=1))
        v = DatasetValidator(cfg)
        monkeypatch.setattr(v, "_get_datasets_to_validate", lambda: {"d1": (ds1, []), "d2": (ds2, [])})
        monkeypatch.setattr(v._split_loader, "get_train_ref", lambda c: "ref")  # noqa: ARG005

        class _F:
            def __init__(self, *, value=None, exc: Exception | None = None, done: bool = True):
                self._value = value
                self._exc = exc
                self._done = done
                self.cancelled = False

            def result(self):
                if self._exc:
                    raise self._exc
                return self._value

            def done(self) -> bool:
                return self._done

            def cancel(self) -> bool:
                self.cancelled = True
                self._done = True
                return True

        f1 = _F(exc=RuntimeError("crash"), done=True)
        f2 = _F(value=Ok({"ok": True}), done=False)
        futures_queue = [f1, f2]

        class _Exec:
            def __init__(self, futures):
                self._futures = list(futures)

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def submit(self, fn, *args, **kwargs):
                _ = fn, args, kwargs
                return self._futures.pop(0)

        monkeypatch.setattr("concurrent.futures.ThreadPoolExecutor", lambda max_workers: _Exec(futures_queue))
        monkeypatch.setattr("concurrent.futures.as_completed", lambda fut_map: list(fut_map.keys()))

        res = v.execute({})
        assert res.is_failure()
        assert f2.cancelled is True

    @patch("src.pipeline.stages.dataset_validator.stage.DatasetLoaderFactory")
    def test_get_datasets_to_validate_falls_back_when_strategy_dataset_missing(self, _mock_loader_factory) -> None:
        cfg = MagicMock(spec=PipelineConfig)
        primary = MagicMock()
        cfg.get_primary_dataset.return_value = primary

        cfg.training = MagicMock()
        cfg.training.strategies = [MagicMock(dataset="missing")]
        cfg.get_dataset_for_strategy.side_effect = KeyError("missing")

        v = DatasetValidator(cfg)
        out = v._get_datasets_to_validate()
        dataset_config, strategy_phases = out["primary"]
        assert dataset_config is primary
        assert strategy_phases == []

    def test_validate_single_dataset_runs_eval_and_merges_errors(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Facade-orchestration smoke: train OK + eval FAIL → overall Err with 'eval:' prefix."""
        cfg = _mk_primary_only_config(_local_ds("data/train.jsonl", plugins=[], critical_failures=0))
        v = DatasetValidator(cfg)

        dataset_config = MagicMock()
        dataset_config.validations = MagicMock(critical_failures=0)

        # Component patches: split_loader, format_checker, plugin_loader.
        monkeypatch.setattr(v._split_loader, "load_train", lambda *a, **k: object())
        monkeypatch.setattr(v._split_loader, "get_train_ref", lambda *a, **k: "train_ref")
        monkeypatch.setattr(v._split_loader, "get_size", lambda *a, **k: 0)
        monkeypatch.setattr(v._split_loader, "try_load_eval", lambda *a, **k: (object(), "eval_ref"))
        monkeypatch.setattr(v._format_checker, "check", lambda *a, **k: Ok(None))
        monkeypatch.setattr(v._plugin_loader, "load_for_dataset", lambda *a, **k: [])

        # train OK, eval FAIL -> overall Err with eval prefix
        monkeypatch.setattr(
            v._plugin_runner, "run", lambda *a, **k: Ok({"m": 1}) if k.get("split_name") == "train" else Err("bad")
        )
        res = v._validate_single_dataset("d", dataset_config, [], context={})
        assert res.is_failure()
        assert "eval:" in str(res.unwrap_err())


