"""Integration tests for DatasetValidator with plugin system (NEW schema)."""

from __future__ import annotations

import json
from unittest.mock import Mock

from src.pipeline.stages.dataset_validator import DatasetValidator, DatasetValidatorEventCallbacks
from src.utils.config import DatasetConfig, PipelineConfig


def _mk_primary_only_config(ds: DatasetConfig) -> Mock:
    cfg = Mock(spec=PipelineConfig)
    cfg.get_primary_dataset.return_value = ds
    cfg.training = Mock()
    cfg.training.strategies = []
    return cfg


def _mk_local_ds(train_path: str, *, plugins: list[dict] | None = None, critical_failures: int = 1) -> DatasetConfig:
    return DatasetConfig(
        source_type="local",
        source_local={"local_paths": {"train": train_path, "eval": None}},
        validations={
            "critical_failures": critical_failures,
            "mode": "fast",
            "plugins": plugins or [],
        },
    )


class TestDatasetValidatorIntegration:
    def test_plugin_mode_success(self, tmp_path) -> None:
        dataset_file = tmp_path / "train.jsonl"
        dataset_file.write_text('{"text": "sample text long enough"}\n' * 10, encoding="utf-8")

        cfg = _mk_primary_only_config(
            _mk_local_ds(
                str(dataset_file),
                plugins=[
                    {"id": "min_samples_main", "plugin": "min_samples", "thresholds": {"threshold": 5}, "apply_to": ["train"]},
                    {"id": "avg_length_main", "plugin": "avg_length", "thresholds": {"min": 5, "max": 100}, "apply_to": ["train"]},
                ],
                critical_failures=1,
            )
        )

        from src.community.catalog import catalog

        catalog.reload()
        validator = DatasetValidator(cfg)
        result = validator.execute({})
        assert result.is_success()
        ctx = result.unwrap()
        assert ctx["validation_status"] == "passed"
        assert "primary.train.min_samples_main.sample_count" in ctx

    def test_plugin_mode_failure_is_err_when_critical_enabled(self, tmp_path) -> None:
        dataset_file = tmp_path / "train.jsonl"
        dataset_file.write_text('{"text": "test"}\n', encoding="utf-8")  # 1 sample

        cfg = _mk_primary_only_config(
            _mk_local_ds(
                str(dataset_file),
                plugins=[{"id": "min_samples_main", "plugin": "min_samples", "thresholds": {"threshold": 100}, "apply_to": ["train"]}],
                critical_failures=1,
            )
        )

        from src.community.catalog import catalog

        catalog.reload()
        validator = DatasetValidator(cfg)
        result = validator.execute({})
        assert result.is_failure()
        assert "validation failed" in str(result.unwrap_err()).lower() or "critical" in str(result.unwrap_err()).lower()

    def test_empty_plugins_uses_defaults(self, tmp_path) -> None:
        dataset_file = tmp_path / "train.jsonl"
        # 150 samples with enough length/diversity to satisfy defaults
        with dataset_file.open("w", encoding="utf-8") as f:
            for i in range(150):
                text = (
                    f"Sample {i}: machine learning topic {i % 10}, data science algorithms, neural networks, "
                    "training models, testing validation, evaluation metrics, optimization techniques."
                )
                json.dump({"text": text}, f)
                f.write("\n")

        cfg = _mk_primary_only_config(_mk_local_ds(str(dataset_file), plugins=[], critical_failures=1))

        from src.community.catalog import catalog

        catalog.reload()
        validator = DatasetValidator(cfg)
        result = validator.execute({})
        assert result.is_success()
        ctx = result.unwrap()
        assert ctx["validation_status"] == "passed"
        assert any(k.startswith("primary.train.") for k in ctx.keys())

    def test_callbacks_called(self, tmp_path) -> None:
        dataset_file = tmp_path / "train.jsonl"
        dataset_file.write_text('{"text": "sample text long enough"}\n' * 10, encoding="utf-8")

        cfg = _mk_primary_only_config(
            _mk_local_ds(
                str(dataset_file),
                plugins=[{"id": "min_samples_main", "plugin": "min_samples", "thresholds": {"threshold": 5}, "apply_to": ["train"]}],
                critical_failures=1,
            )
        )

        from src.community.catalog import catalog

        catalog.reload()
        callbacks = DatasetValidatorEventCallbacks(
            on_dataset_loaded=Mock(),
            on_validation_completed=Mock(),
            on_plugin_start=Mock(),
            on_plugin_complete=Mock(),
        )

        validator = DatasetValidator(cfg, callbacks=callbacks)
        result = validator.execute({})
        assert result.is_success()

        assert callbacks.on_dataset_loaded.called
        assert callbacks.on_validation_completed.called
        assert callbacks.on_plugin_start.called
        assert callbacks.on_plugin_complete.called


