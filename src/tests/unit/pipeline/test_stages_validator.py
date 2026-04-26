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


@patch("src.pipeline.stages.dataset_validator.stage.DatasetLoaderFactory")
@patch("src.pipeline.stages.dataset_validator.stage.validation_registry")
def test_loads_default_plugins_when_no_plugins_configured(mock_registry, mock_loader_factory) -> None:
    ds = _local_ds("data/train.jsonl", plugins=[], critical_failures=1)
    cfg = _mk_primary_only_config(ds)

    # Default plugin load calls registry.instantiate(...) 4 times
    mock_registry.instantiate.return_value = MagicMock(name="x")

    _ = DatasetValidator(cfg)
    assert mock_registry.instantiate.call_count == 4


@patch("src.pipeline.stages.dataset_validator.stage.DatasetLoaderFactory")
@patch("src.pipeline.stages.dataset_validator.stage.validation_registry")
def test_loads_configured_plugins_only(mock_registry, mock_loader_factory) -> None:
    ds = _local_ds(
        "data/train.jsonl",
        plugins=[
            {"id": "custom_plugin_main", "plugin": "custom_plugin", "params": {"threshold": 100}, "apply_to": ["train"]},
            {"id": "another_plugin_main", "plugin": "another_plugin", "params": {}, "apply_to": ["train"]},
        ],
        critical_failures=1,
    )
    cfg = _mk_primary_only_config(ds)

    mock_registry.instantiate.return_value = MagicMock(name="x")
    _ = DatasetValidator(cfg)
    assert mock_registry.instantiate.call_count == 2


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
        monkeypatch.setattr(v, "_get_dataset_train_ref", lambda c: "ref")  # noqa: ARG005

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
        monkeypatch.setattr(v, "_get_dataset_train_ref", lambda c: "ref")  # noqa: ARG005

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
        cfg = _mk_primary_only_config(_local_ds("data/train.jsonl", plugins=[], critical_failures=0))
        v = DatasetValidator(cfg)

        dataset_config = MagicMock()
        dataset_config.validations = MagicMock(critical_failures=0)
        v._loader_factory = MagicMock()
        v._loader_factory.create_for_dataset.return_value = MagicMock()

        monkeypatch.setattr(v, "_load_dataset_for_validation", lambda *a, **k: object())
        monkeypatch.setattr(v, "_get_dataset_train_ref", lambda *a, **k: "train_ref")
        monkeypatch.setattr(v, "_load_plugins_for_dataset", lambda *a, **k: [])
        monkeypatch.setattr(v, "_try_load_eval_dataset_for_validation", lambda *a, **k: (object(), "eval_ref"))

        # train OK, eval FAIL -> overall Err with eval prefix
        monkeypatch.setattr(
            v, "_run_plugin_validations", lambda *a, **k: Ok({"m": 1}) if k.get("split_name") == "train" else Err("bad")
        )
        monkeypatch.setattr(v, "_check_dataset_format", lambda *a, **k: Ok(None))
        res = v._validate_single_dataset("d", dataset_config, [], context={})
        assert res.is_failure()
        assert "eval:" in str(res.unwrap_err())

    def test_load_dataset_for_validation_hf_branches(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cfg = _mk_primary_only_config(_local_ds("data/train.jsonl", plugins=[], critical_failures=0))
        v = DatasetValidator(cfg)

        ds_cfg = MagicMock()
        ds_cfg.get_source_type.return_value = "huggingface"
        ds_cfg.validations = MagicMock(mode="fast")
        ds_cfg.max_samples = None
        ds_cfg.source_hf = MagicMock(train_id=None, eval_id=None)

        loader = MagicMock()
        assert v._load_dataset_for_validation(ds_cfg, loader, split_name="train") is None

        # Non-iterable return -> None
        ds_cfg.source_hf.train_id = "org/ds"
        monkeypatch.setattr("datasets.load_dataset", lambda *a, **k: object())

        # Make isinstance check deterministic by patching datasets.IterableDataset base
        class _Base:
            pass

        monkeypatch.setattr("datasets.IterableDataset", _Base)
        assert v._load_dataset_for_validation(ds_cfg, loader, split_name="train") is None

        # Exception -> None
        monkeypatch.setattr("datasets.load_dataset", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
        assert v._load_dataset_for_validation(ds_cfg, loader, split_name="train") is None

    def test_load_dataset_for_validation_local_branches(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        cfg = _mk_primary_only_config(_local_ds("data/train.jsonl", plugins=[], critical_failures=0))
        v = DatasetValidator(cfg)

        # source_local None
        ds_cfg = MagicMock()
        ds_cfg.get_source_type.return_value = "local"
        ds_cfg.validations = MagicMock(mode="fast")
        ds_cfg.source_local = None
        assert v._load_dataset_for_validation(ds_cfg, loader=MagicMock(), split_name="train") is None

        # local_paths missing
        ds_cfg2 = MagicMock()
        ds_cfg2.get_source_type.return_value = "local"
        ds_cfg2.validations = MagicMock(mode="fast")
        ds_cfg2.source_local = MagicMock(local_paths=MagicMock(train=None, eval=None))
        assert v._load_dataset_for_validation(ds_cfg2, loader=MagicMock(), split_name="train") is None

        # file not exists
        ds_cfg3 = MagicMock()
        ds_cfg3.get_source_type.return_value = "local"
        ds_cfg3.validations = MagicMock(mode="fast")
        ds_cfg3.source_local = MagicMock(local_paths=MagicMock(train=str(tmp_path / "missing.jsonl"), eval=None))
        assert v._load_dataset_for_validation(ds_cfg3, loader=MagicMock(), split_name="train") is None

        # fast mode sampling > 10k
        file_path = tmp_path / "train.jsonl"
        file_path.write_text("{}", encoding="utf-8")
        ds_cfg4 = MagicMock()
        ds_cfg4.get_source_type.return_value = "local"
        ds_cfg4.validations = MagicMock(mode="fast")
        ds_cfg4.source_local = MagicMock(local_paths=MagicMock(train=str(file_path), eval=None))

        big_ds = MagicMock()
        big_ds.__len__.return_value = 20000
        sampled = MagicMock()
        big_ds.select.return_value = sampled

        loader = MagicMock()
        loader.load.return_value = big_ds
        out = v._load_dataset_for_validation(ds_cfg4, loader=loader, split_name="train")
        assert out is sampled
        big_ds.select.assert_called_once()

    def test_run_plugin_validations_success_and_failure_callbacks(self) -> None:
        cfg = _mk_primary_only_config(_local_ds("data/train.jsonl", plugins=[], critical_failures=0))
        cb = DatasetValidatorEventCallbacks(
            on_plugin_start=MagicMock(),
            on_plugin_complete=MagicMock(),
            on_plugin_failed=MagicMock(),
            on_validation_completed=MagicMock(),
            on_validation_failed=MagicMock(),
        )
        v = DatasetValidator(cfg, callbacks=cb)

        dataset_config = MagicMock()
        dataset_config.validations = MagicMock(critical_failures=0)

        class _P:
            name = "p"
            params = {"x": 1}
            thresholds = {"threshold": 1}

            def get_description(self):
                return "d"

            def validate(self, dataset):
                _ = dataset
                return ValidationResult(
                    plugin_name="p",
                    passed=True,
                    params={"x": 1},
                    thresholds={"threshold": 1},
                    metrics={"m": 1.0},
                    warnings=["w"],
                    errors=[],
                    execution_time_ms=1.0,
                )

            def get_recommendations(self, result):
                _ = result
                return []

        res = v._run_plugin_validations(
            "d",
            "ref",
            dataset=object(),
            _context={},
            dataset_config=dataset_config,
            plugins=[("p_main", "p", _P(), {"train"})],
            split_name="train",
        )
        assert res.is_success()
        cb.on_plugin_complete.assert_called_once()
        cb.on_validation_completed.assert_called_once()

        # Failure path + critical threshold
        dataset_config2 = MagicMock()
        dataset_config2.validations = MagicMock(critical_failures=1)

        class _PF:
            name = "p"
            params = {"x": 1}
            thresholds = {"threshold": 1}

            def get_description(self):
                return "d"

            def validate(self, dataset):
                _ = dataset
                return ValidationResult(
                    plugin_name="p",
                    passed=False,
                    params={"x": 1},
                    thresholds={"threshold": 1},
                    metrics={"m": 0.0},
                    warnings=[],
                    errors=["e"],
                    execution_time_ms=1.0,
                )

            def get_recommendations(self, result):
                _ = result
                return ["r"]

        res2 = v._run_plugin_validations(
            "d",
            "ref",
            dataset=object(),
            _context={},
            dataset_config=dataset_config2,
            plugins=[("p_main", "p", _PF(), {"train"})],
            split_name="train",
        )
        assert res2.is_failure()
        cb.on_plugin_failed.assert_called()
        failed_call = cb.on_plugin_failed.call_args
        assert failed_call is not None
        assert failed_call.args[7] == pytest.approx(1.0)
        cb.on_validation_failed.assert_called()

    def test_run_plugin_validations_plugin_crash_calls_failed_callback(self) -> None:
        cfg = _mk_primary_only_config(_local_ds("data/train.jsonl", plugins=[], critical_failures=0))
        cb = DatasetValidatorEventCallbacks(on_plugin_failed=MagicMock(), on_validation_failed=MagicMock())
        v = DatasetValidator(cfg, callbacks=cb)

        dataset_config = MagicMock()
        dataset_config.validations = MagicMock(critical_failures=1)

        class _P:
            name = "p"
            params = {"x": 1}
            thresholds = {"threshold": 1}

            def get_description(self):
                return "d"

            def validate(self, dataset):
                raise RuntimeError("boom")

            def get_recommendations(self, result):
                _ = result
                return []

        res = v._run_plugin_validations(
            "d",
            "ref",
            dataset=object(),
            _context={},
            dataset_config=dataset_config,
            plugins=[("p_main", "p", _P(), {"train"})],
            split_name="train",
        )
        assert res.is_failure()
        cb.on_plugin_failed.assert_called_once()
        failed_call = cb.on_plugin_failed.call_args
        assert failed_call is not None
        assert failed_call.args[7] >= 0.0

    def test_get_dataset_train_ref_and_try_load_eval_dataset_branches(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cfg = _mk_primary_only_config(_local_ds("data/train.jsonl", plugins=[], critical_failures=0))
        v = DatasetValidator(cfg)

        bad = MagicMock()
        bad.get_source_type.side_effect = Exception("boom")
        assert v._get_dataset_train_ref(bad) == "unknown"

        # huggingface eval branch
        hf = MagicMock()
        hf.get_source_type.return_value = "huggingface"
        hf.source_hf = MagicMock(eval_id="org/ds", train_id="org/ds-train")
        monkeypatch.setattr(v, "_load_dataset_for_validation", lambda *a, **k: object())
        ds, ref = v._try_load_eval_dataset_for_validation(hf, loader=MagicMock())
        assert ds is not None
        assert ref == "org/ds"

        # exception branch -> None, None
        hf.get_source_type.side_effect = Exception("boom")
        assert v._try_load_eval_dataset_for_validation(hf, loader=MagicMock()) == (None, None)
