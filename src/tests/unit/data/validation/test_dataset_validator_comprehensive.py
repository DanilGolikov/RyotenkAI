"""
Comprehensive tests for DatasetValidator stage (NEW dataset schema).

Key changes vs legacy:
- DatasetConfig uses source_local/source_hf blocks
- validations is an object: validations.{mode,critical_failures,plugins[]}
- Metrics are split-prefixed: "{dataset}.{train|eval}.{plugin}.{metric}"
"""

from __future__ import annotations

import json
from unittest.mock import Mock

import pytest

from src.pipeline.stages.dataset_validator import DatasetValidator
from src.utils.config import DatasetConfig, PipelineConfig


def _mk_local_dataset_config(
    train_path: str,
    *,
    plugins: list[dict] | None = None,
    critical_failures: int = 1,
) -> DatasetConfig:
    return DatasetConfig(
        source_type="local",
        source_local={"local_paths": {"train": train_path, "eval": None}},
        validations={
            "critical_failures": critical_failures,
            "mode": "fast",
            "plugins": plugins or [],
        },
    )


def _mk_config_primary_only(dataset_config: DatasetConfig) -> Mock:
    cfg = Mock(spec=PipelineConfig)
    cfg.get_primary_dataset.return_value = dataset_config
    # Force DatasetValidator to use primary dataset path fallback (avoid strategy mocking noise)
    cfg.training = Mock()
    cfg.training.strategies = []
    return cfg


class TestDatasetValidatorBoundary:
    def test_exact_thresholds_pass(self, tmp_path) -> None:
        dataset_file = tmp_path / "exact.jsonl"
        # 10 samples, avg_length == 50
        with dataset_file.open("w", encoding="utf-8") as f:
            for _ in range(10):
                json.dump({"text": "x" * 50}, f)
                f.write("\n")

        cfg = _mk_config_primary_only(
            _mk_local_dataset_config(
                str(dataset_file),
                plugins=[
                    {"id": "min_samples_main", "plugin": "min_samples", "thresholds": {"threshold": 10}, "apply_to": ["train"]},
                    {"id": "avg_length_main", "plugin": "avg_length", "thresholds": {"min": 50, "max": 50}, "apply_to": ["train"]},
                ],
            )
        )

        from src.community.catalog import catalog

        catalog.reload()
        validator = DatasetValidator(cfg)
        result = validator.execute({})
        assert result.is_success()
        ctx = result.unwrap()
        assert ctx["validation_status"] == "passed"
        assert ctx["primary.train.min_samples_main.sample_count"] == 10

    def test_empty_dataset_load_error_is_reported_as_failed_validation(self, tmp_path) -> None:
        dataset_file = tmp_path / "empty.jsonl"
        dataset_file.write_text("", encoding="utf-8")

        cfg = _mk_config_primary_only(
            _mk_local_dataset_config(
                str(dataset_file),
                plugins=[{"id": "min_samples_main", "plugin": "min_samples", "thresholds": {"threshold": 1}, "apply_to": ["train"]}],
                critical_failures=1,
            )
        )

        from src.community.catalog import catalog

        catalog.reload()
        validator = DatasetValidator(cfg)
        result = validator.execute({})
        assert result.is_success()
        ctx = result.unwrap()
        assert ctx["validation_status"] == "failed"
        assert any("DATASET_LOAD_ERROR" in warning for warning in ctx.get("warnings", []))

    def test_empty_dataset_is_non_critical_when_critical_failures_zero(self, tmp_path) -> None:
        dataset_file = tmp_path / "empty.jsonl"
        dataset_file.write_text("", encoding="utf-8")

        cfg = _mk_config_primary_only(
            _mk_local_dataset_config(
                str(dataset_file),
                plugins=[{"id": "min_samples_main", "plugin": "min_samples", "thresholds": {"threshold": 1}, "apply_to": ["train"]}],
                critical_failures=0,
            )
        )

        from src.community.catalog import catalog

        catalog.reload()
        validator = DatasetValidator(cfg)
        result = validator.execute({})
        assert result.is_success()
        ctx = result.unwrap()
        assert ctx["validation_status"] == "failed"
        assert any("ERROR:" in w for w in ctx.get("warnings", []))


class TestDatasetValidatorNegative:
    def test_invalid_plugin_name_raises_on_init(self, tmp_path) -> None:
        dataset_file = tmp_path / "train.jsonl"
        dataset_file.write_text('{"text": "sample"}\n' * 10, encoding="utf-8")

        cfg = _mk_config_primary_only(
            _mk_local_dataset_config(
                str(dataset_file),
                plugins=[{"id": "missing_main", "plugin": "nonexistent_plugin", "params": {}, "apply_to": ["train"]}],
                critical_failures=1,
            )
        )

        # Should raise during plugin load (constructor)
        with pytest.raises(KeyError):
            _ = DatasetValidator(cfg)


