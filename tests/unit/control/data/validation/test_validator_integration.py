"""Integration tests for DatasetValidator with plugin system (NEW schema).

Phase A2 Batch 8 — raise-based migration. ``execute()`` returns ``dict``
on success / advisory failure, raises
:class:`DatasetValidationFailedError` on critical failure.

Phase 4 (event-system unification, 2026-05-16) — the legacy
``DatasetValidatorEventCallbacks`` dataclass was removed. Stages now
take an optional :class:`IEventEmitter` for typed event emission. This
test asserts the typed ``ryotenkai.control.dataset.validation_*`` events
fire in place of the removed callback methods.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from ryotenkai_control.pipeline.stages.dataset_validator import DatasetValidator
from ryotenkai_shared.config import DatasetConfig, PipelineConfig
from ryotenkai_shared.errors import DatasetValidationFailedError
from ryotenkai_shared.events.types.control_dataset import (
    DatasetValidationCompletedEvent,
    DatasetValidationStartedEvent,
)

from tests._fakes.event_emitter import FakeEventEmitter


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

    def test_emitter_receives_started_and_completed_events(self, tmp_path) -> None:
        """Phase 4 — assert typed events fire in place of the removed
        ``DatasetValidatorEventCallbacks.on_*`` callbacks. A successful run
        emits exactly one ``DatasetValidationStartedEvent`` (aggregate
        per-stage) and one ``DatasetValidationCompletedEvent``."""
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

        emitter = FakeEventEmitter()
        validator = DatasetValidator(cfg, emitter=emitter)
        result = validator.execute({})
        assert isinstance(result, dict)
        assert result["validation_status"] == "passed"

        started = [e for e in emitter.emitted if isinstance(e, DatasetValidationStartedEvent)]
        completed = [e for e in emitter.emitted if isinstance(e, DatasetValidationCompletedEvent)]
        assert len(started) == 1
        assert len(completed) == 1
        # ``dataset_path`` payload carries the (comma-joined) scheduled paths.
        assert str(dataset_file) in started[0].payload.dataset_path
        # Completed event surfaces checks_passed (per-dataset keys).
        assert completed[0].payload.checks_passed
