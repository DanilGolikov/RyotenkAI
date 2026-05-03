"""Run validation plugins against one dataset split, fire per-plugin events.

Owns the per-plugin loop:

* fire ``on_plugin_start`` before each plugin
* call ``plugin.validate(dataset)`` (catch + report crashes)
* fire ``on_plugin_complete`` for passed plugins
* fire ``on_plugin_failed`` (with recommendations) for failed plugins
  AND for crashes — same callback shape, so artifact accumulator
  treats both uniformly
* track ``failed_plugins`` count against the dataset's
  ``critical_failures`` threshold; early-break the loop when reached
* fire ``on_validation_completed`` / ``on_validation_failed`` at the end
* return ``Ok(metrics)`` on success or ``Err(DatasetError)`` with code
  ``DATASET_VALIDATION_CRITICAL_FAILURE`` / ``DATASET_VALIDATION_ERROR``
"""

from __future__ import annotations

from time import perf_counter
from typing import TYPE_CHECKING, Any, Literal

from src.data.validation.base import ValidationPlugin
from src.pipeline.stages.dataset_validator.constants import (
    VALIDATION_STATUS_KEY,
    VALIDATION_STATUS_PASSED,
    VALIDATIONS_ATTR,
)
from src.utils.logger import logger
from src.utils.result import AppError, DatasetError, Err, Ok, Result

if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset

    from src.pipeline.stages.dataset_validator.plugin_loader import PluginTuple
    from src.pipeline.stages.dataset_validator.stage import DatasetValidatorEventCallbacks


class PluginRunner:
    """Run a list of validation plugins against one dataset split."""

    def __init__(self, callbacks: DatasetValidatorEventCallbacks) -> None:
        self._callbacks = callbacks

    def run(
        self,
        dataset_name: str,
        dataset_path: str,
        dataset: Dataset | IterableDataset,
        dataset_config: Any,
        plugins: list[PluginTuple],
        *,
        split_name: Literal["train", "eval"],
    ) -> Result[dict[str, Any], AppError]:
        """Run plugins; fire callbacks; honour critical_failures threshold."""
        logger.info(f"[{dataset_name}] Running {len(plugins)} validation plugins on {split_name}")

        all_metrics: dict[str, Any] = {}
        all_errors: list[str] = []
        all_warnings: list[str] = []
        all_recommendations: list[str] = []
        failed_plugins: list[str] = []
        critical_threshold_reached = False

        for plugin_id, plugin_name, plugin, _apply_to in plugins:
            logger.info(f"  [{dataset_name}] Running plugin: {plugin_id} ({plugin_name}, {split_name})")
            plugin_started_at = perf_counter()

            if self._callbacks.on_plugin_start:
                self._callbacks.on_plugin_start(
                    dataset_name,
                    dataset_path,
                    plugin_id,
                    plugin_name,
                    plugin.get_description(),
                )

            try:
                result = plugin.validate(dataset)

                for key, value in result.metrics.items():
                    all_metrics[f"{split_name}.{plugin_id}.{key}"] = value

                all_warnings.extend(result.warnings)

                if result.passed:
                    logger.info(
                        f"    [{dataset_name}] ✓ {plugin_id} ({plugin_name}) passed "
                        f"({result.execution_time_ms:.1f}ms)"
                    )

                    if self._callbacks.on_plugin_complete:
                        self._callbacks.on_plugin_complete(
                            dataset_name,
                            dataset_path,
                            plugin_id,
                            plugin.name,
                            result.params,
                            result.thresholds,
                            result.metrics,
                            result.execution_time_ms,
                        )
                else:
                    logger.error(f"    [{dataset_name}] ✗ {plugin_id} ({plugin_name}) failed")
                    display_errors = list(result.errors)
                    display_errors.extend(ValidationPlugin.render_error_groups(result.error_groups))
                    all_errors.extend(display_errors)
                    failed_plugins.append(plugin_id)

                    recommendations = plugin.get_recommendations(result)
                    all_recommendations.extend(recommendations)

                    if self._callbacks.on_plugin_failed:
                        self._callbacks.on_plugin_failed(
                            dataset_name,
                            dataset_path,
                            plugin_id,
                            plugin.name,
                            result.params,
                            result.thresholds,
                            result.metrics,
                            result.execution_time_ms,
                            display_errors,
                            recommendations,
                        )

                    if self._is_critical_threshold_reached(dataset_config, failed_plugins):
                        critical_threshold_reached = True
                        self._log_critical_threshold_reached(dataset_name, dataset_config, failed_plugins)
                        break
            except Exception as e:
                crashed_duration_ms = (perf_counter() - plugin_started_at) * 1000.0
                error_msg = f"Plugin '{plugin_id}' ({plugin_name}) crashed: {e}"
                logger.error(f"    [{dataset_name}] ✗ {error_msg}")
                all_errors.append(error_msg)
                failed_plugins.append(plugin_id)

                # Crashes fire the same on_plugin_failed callback so the
                # artifact accumulator treats them uniformly with regular
                # failures.
                if self._callbacks.on_plugin_failed:
                    self._callbacks.on_plugin_failed(
                        dataset_name,
                        dataset_path,
                        plugin_id,
                        plugin.name,
                        dict(plugin.params),
                        dict(plugin.thresholds),
                        all_metrics,
                        crashed_duration_ms,
                        [error_msg],
                        [],
                    )

                if self._is_critical_threshold_reached(dataset_config, failed_plugins):
                    critical_threshold_reached = True
                    self._log_critical_threshold_reached(dataset_name, dataset_config, failed_plugins)
                    break

        return self._build_result(
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            all_metrics=all_metrics,
            all_warnings=all_warnings,
            all_errors=all_errors,
            all_recommendations=all_recommendations,
            critical_threshold_reached=critical_threshold_reached,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _is_critical_threshold_reached(dataset_config: Any, failed_plugins: list[str]) -> bool:
        critical_threshold = getattr(getattr(dataset_config, VALIDATIONS_ATTR, None), "critical_failures", 0)
        return 0 < critical_threshold <= len(failed_plugins)

    @staticmethod
    def _log_critical_threshold_reached(
        dataset_name: str, dataset_config: Any, failed_plugins: list[str]
    ) -> None:
        critical_threshold = getattr(getattr(dataset_config, VALIDATIONS_ATTR, None), "critical_failures", 0)
        logger.error(
            f"    [{dataset_name}] Critical failure threshold reached: "
            f"{len(failed_plugins)}/{critical_threshold} plugins failed"
        )
        logger.error(f"    [{dataset_name}] Stopping validation for this dataset")

    def _build_result(
        self,
        *,
        dataset_name: str,
        dataset_path: str,
        all_metrics: dict[str, Any],
        all_warnings: list[str],
        all_errors: list[str],
        all_recommendations: list[str],
        critical_threshold_reached: bool,
    ) -> Result[dict[str, Any], AppError]:
        logger.info(f"[{dataset_name}] Dataset Validation Metrics:")
        for key, value in all_metrics.items():
            logger.info(f"  - {key}: {value}")

        if all_warnings:
            logger.warning(f"[{dataset_name}] Validation Warnings:")
            for warning in all_warnings:
                logger.warning(f"  - {warning}")

        if all_errors:
            logger.error(f"[{dataset_name}] Dataset Validation Failed:")
            for error in all_errors:
                logger.error(f"  - {error}")

            if all_recommendations:
                logger.info("")
                logger.info(f"[{dataset_name}] 💡 Recommendations:")
                for rec in all_recommendations:
                    logger.info(f"  - {rec}")

            if self._callbacks.on_validation_failed:
                self._callbacks.on_validation_failed(dataset_name, dataset_path, all_errors)

            error_summary = f"{len(all_errors)} validation errors"
            error_code = (
                "DATASET_VALIDATION_CRITICAL_FAILURE" if critical_threshold_reached else "DATASET_VALIDATION_ERROR"
            )
            return Err(DatasetError(message=error_summary, code=error_code, details={"errors": all_errors}))

        logger.info(f"[{dataset_name}] ✅ All validation checks passed!")

        if self._callbacks.on_validation_completed:
            self._callbacks.on_validation_completed(dataset_name, dataset_path, all_metrics, all_warnings)

        result_data = {
            VALIDATION_STATUS_KEY: VALIDATION_STATUS_PASSED,
            "warnings": all_warnings,
            **all_metrics,
        }
        return Ok(result_data)


__all__ = ["PluginRunner"]
