"""Integration tests for DatasetValidator with plugin system (NEW schema).

Phase A2 Batch 8 — raise-based migration. ``execute()`` returns ``dict``
on success / advisory failure, raises
:class:`DatasetValidationFailedError` on critical failure.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from ryotenkai_control.pipeline.stages.dataset_validator import DatasetValidator, DatasetValidatorEventCallbacks
from ryotenkai_shared.config import DatasetConfig, PipelineConfig
from ryotenkai_shared.errors import DatasetValidationFailedError


def _mk_primary_only_config(ds: DatasetConfig) -> Mock:
    cfg = Mock(spec=PipelineConfig)
    cfg.get_primary_dataset.return_value = ds
    cfg.training = Mock()
    cfg.training.strategies = []
    return cfg


def _mk_local_ds(train_path: str, *, plugins: list[dict] | None = None, critical_failures: int = 1) -> DatasetConfig:
    return DatasetConfig(
        source={"kind": "local", "local_paths": {"train": train_path, "eval": None}},
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

        from ryotenkai_community.catalog import catalog

        catalog.reload()
        validator = DatasetValidator(cfg)
        result = validator.execute({})
        assert isinstance(result, dict)
        assert result["validation_status"] == "passed"
        assert "primary.train.min_samples_main.sample_count" in result

    def test_plugin_mode_failure_raises_when_critical_enabled(self, tmp_path) -> None:
        dataset_file = tmp_path / "train.jsonl"
        dataset_file.write_text('{"text": "test"}\n', encoding="utf-8")  # 1 sample

        cfg = _mk_primary_only_config(
            _mk_local_ds(
                str(dataset_file),
                plugins=[{"id": "min_samples_main", "plugin": "min_samples", "thresholds": {"threshold": 100}, "apply_to": ["train"]}],
                critical_failures=1,
            )
        )

        from ryotenkai_community.catalog import catalog

        catalog.reload()
        validator = DatasetValidator(cfg)
        with pytest.raises(DatasetValidationFailedError) as excinfo:
            validator.execute({})
        message = str(excinfo.value).lower() + " " + (excinfo.value.detail or "").lower()
        assert "validation failed" in message or "critical" in message

    def test_empty_plugins_skips_plugin_checks_no_hidden_defaults(self, tmp_path) -> None:
        """Empty plugins config → no plugins run, validation passes with no metrics.

        Validation must be EXPLICIT — empty plugins means user opted out of
        plugin checks (format check still runs separately).
        """
        dataset_file = tmp_path / "train.jsonl"
        # Tiny dataset: would fail any default min_samples threshold if defaults
        # were silently injected. Test asserts they are NOT.
        dataset_file.write_text('{"text": "x"}\n' * 5, encoding="utf-8")

        cfg = _mk_primary_only_config(_mk_local_ds(str(dataset_file), plugins=[], critical_failures=1))

        from ryotenkai_community.catalog import catalog

        catalog.reload()
        validator = DatasetValidator(cfg)
        result = validator.execute({})
        assert isinstance(result, dict)
        assert result["validation_status"] == "passed"
        # No plugin metrics — nothing was checked.
        assert not any(k.startswith("primary.train.") for k in result)

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

        from ryotenkai_community.catalog import catalog

        catalog.reload()
        callbacks = DatasetValidatorEventCallbacks(
            on_dataset_loaded=Mock(),
            on_validation_completed=Mock(),
            on_plugin_start=Mock(),
            on_plugin_complete=Mock(),
        )

        validator = DatasetValidator(cfg, callbacks=callbacks)
        result = validator.execute({})
        assert isinstance(result, dict)

        assert callbacks.on_dataset_loaded.called
        assert callbacks.on_validation_completed.called
        assert callbacks.on_plugin_start.called
        assert callbacks.on_plugin_complete.called
