"""
ValidationArtifactManager — accumulates dataset validation results and flushes artifacts.

Extracted from PipelineOrchestrator to reduce orchestrator size and isolate
validation-specific callback handling into a self-contained unit.

Responsibilities:
  - Accumulate per-dataset validation results via DatasetValidator callbacks
  - Track plugin descriptions across start/complete/failed lifecycle
  - Flush the final dataset_validator_results.json artifact
  - Build validation-specific state outputs for pipeline state persistence
"""

from __future__ import annotations

from typing import Any

from src.pipeline.artifacts import (
    StageArtifactCollector,
    ValidationArtifactData,
    ValidationDatasetData,
    ValidationPluginData,
)
from src.pipeline.stages import StageNames

_STATUS_FAILED = "failed"
_STATUS_PASSED = "passed"
_VALIDATION_STATUS_FAILED = "failed"
_VALIDATION_STATUS_PASSED = "passed"
_VALIDATION_ARTIFACT_REF = "dataset_validator_results.json"


class ValidationArtifactManager:
    """Accumulates dataset validation results and flushes the final artifact.

    Parameters
    ----------
    collectors:
        Stage artifact collectors keyed by stage name.
        Only the ``StageNames.DATASET_VALIDATOR`` entry is used.
    context:
        Shared pipeline context dict (read-only access for flush).
    """

    def __init__(
        self,
        collectors: dict[str, StageArtifactCollector],
        context: dict[str, Any],
    ) -> None:
        self._collectors = collectors
        self._context = context
        # key: dataset_path → accumulated validation data
        self._validation_accumulator: dict[str, ValidationDatasetData] = {}
        # key: (dataset_path, plugin_id) → description string
        self._validation_plugin_descriptions: dict[tuple[str, str], str] = {}

    # ------------------------------------------------------------------
    # DatasetValidator callbacks
    # ------------------------------------------------------------------

    def on_dataset_scheduled(self, dataset_name: str, dataset_path: str, _validation_mode: str) -> None:
        """Callback: dataset scheduled for validation — initialize accumulator entry."""
        acc: ValidationDatasetData = {
            "name": dataset_name,  # noqa: WPS226
            "path": dataset_path,
            "sample_count": None,
            "status": "scheduled",  # noqa: WPS226
            "critical_failures": 0,
            "plugins": [],
        }
        self._validation_accumulator[dataset_path] = acc

    def on_dataset_loaded(
        self, _dataset_name: str, dataset_path: str, sample_count: int, critical_failures: int
    ) -> None:
        """Callback: dataset loaded — update accumulator with sample count."""
        entry = self._validation_accumulator.get(dataset_path)
        if entry:
            entry["sample_count"] = sample_count
            entry["critical_failures"] = critical_failures

    def on_validation_completed(
        self, _dataset_name: str, dataset_path: str, _metrics: dict, _warnings: list[str]
    ) -> None:
        """Callback: validation completed successfully — mark dataset as passed."""
        entry = self._validation_accumulator.get(dataset_path)
        if entry:
            entry["status"] = _STATUS_PASSED

    def on_validation_failed(self, _dataset_name: str, dataset_path: str, _errors: list[str]) -> None:
        """Callback: validation failed — mark dataset as failed."""
        entry = self._validation_accumulator.get(dataset_path)
        if entry:
            entry["status"] = _STATUS_FAILED

    def on_plugin_start(
        self,
        _dataset_name: str,
        dataset_path: str,
        plugin_id: str,
        _plugin_name: str,
        description: str,
    ) -> None:
        """Callback: validation plugin started — cache plugin description."""
        self._validation_plugin_descriptions[(dataset_path, plugin_id)] = description

    def on_plugin_complete(
        self,
        _dataset_name: str,
        dataset_path: str,
        plugin_id: str,
        plugin_name: str,
        params: dict,
        thresholds: dict,
        metrics: dict,
        duration_ms: float,
    ) -> None:
        """Callback: validation plugin completed — append to accumulator."""
        description = self._validation_plugin_descriptions.pop((dataset_path, plugin_id), "")
        plugin_data: ValidationPluginData = {
            "id": plugin_id,  # noqa: WPS226
            "plugin_name": plugin_name,  # noqa: WPS226
            "passed": True,
            "duration_ms": duration_ms,
            "description": description,
            "metrics": metrics,  # noqa: WPS226
            "params": params,
            "thresholds": thresholds,
            "errors": [],
            "recommendations": [],
        }
        entry = self._validation_accumulator.get(dataset_path)
        if entry:
            entry["plugins"].append(plugin_data)

    def on_plugin_failed(
        self,
        _dataset_name: str,
        dataset_path: str,
        plugin_id: str,
        plugin_name: str,
        params: dict,
        thresholds: dict,
        metrics: dict,
        duration_ms: float,
        errors: list[str],
        recommendations: list[str],
    ) -> None:
        """Callback: validation plugin failed — append to accumulator."""
        description = self._validation_plugin_descriptions.pop((dataset_path, plugin_id), "")
        plugin_data: ValidationPluginData = {
            "id": plugin_id,  # noqa: WPS226
            "plugin_name": plugin_name,  # noqa: WPS226
            "passed": False,
            "duration_ms": duration_ms,
            "description": description,
            "metrics": metrics,  # noqa: WPS226
            "params": params,
            "thresholds": thresholds,
            "errors": errors,
            "recommendations": recommendations,
        }
        entry = self._validation_accumulator.get(dataset_path)
        if entry:
            entry["plugins"].append(plugin_data)

    # ------------------------------------------------------------------
    # Artifact flushing
    # ------------------------------------------------------------------

    def flush_validation_artifact(self, started_at: str, duration_seconds: float) -> None:
        """Write dataset_validator_results.json from accumulated data.

        Called after DatasetValidator stage finishes (success or failure).
        """
        collector = self._collectors.get(StageNames.DATASET_VALIDATOR)
        if not collector or collector.is_flushed:
            return

        datasets: list[ValidationDatasetData] = list(self._validation_accumulator.values())
        artifact_data: ValidationArtifactData = {"datasets": datasets}
        collector.put(**artifact_data)

        all_passed = all(d.get("status") in {_STATUS_PASSED, "scheduled"} for d in datasets)  # noqa: WPS226
        if all_passed:
            collector.flush_ok(
                started_at=started_at,
                duration_seconds=duration_seconds,
                context=self._context,
            )
        else:
            failed = [d["name"] for d in datasets if d.get("status") == _STATUS_FAILED]  # noqa: WPS226
            collector.flush_error(
                error=f"Dataset validation failed: {', '.join(failed)}",
                started_at=started_at,
                duration_seconds=duration_seconds,
                context=self._context,
            )

    def build_dataset_validation_state_outputs(
        self,
        *,
        stage_ctx: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> dict[str, Any]:
        """Build outputs dict for pipeline state persistence."""
        datasets = list(self._validation_accumulator.values())
        failed_datasets = [dataset["name"] for dataset in datasets if dataset.get("status") == _STATUS_FAILED]
        passed_count = sum(1 for dataset in datasets if dataset.get("status") == _STATUS_PASSED)
        outputs: dict[str, Any] = {
            "validation_artifact_ref": _VALIDATION_ARTIFACT_REF,
        }

        if datasets:
            outputs.update(
                {
                    "datasets_validated": len(datasets),
                    "datasets_passed": passed_count,
                    "datasets_failed": len(failed_datasets),
                }
            )
            if failed_datasets:
                outputs["failed_datasets"] = failed_datasets

        if stage_ctx is not None:
            validation_status = stage_ctx.get("validation_status")
            if isinstance(validation_status, str) and validation_status:
                outputs["validation_status"] = validation_status
            warnings = stage_ctx.get("warnings")
            if isinstance(warnings, list):
                outputs["validation_warning_count"] = len(warnings)
            message = stage_ctx.get("message")
            if message:
                outputs["validation_message"] = str(message)
        elif error is not None:
            outputs["validation_status"] = _VALIDATION_STATUS_FAILED
            outputs["validation_message"] = error

        if "validation_status" not in outputs:
            outputs["validation_status"] = _VALIDATION_STATUS_FAILED if failed_datasets else _VALIDATION_STATUS_PASSED
        return outputs


__all__ = ["ValidationArtifactManager"]
