"""Run validation plugins against one dataset split, record per-plugin events.

Owns the per-plugin loop:

* notify ``artifact_recorder.on_plugin_start`` before each plugin
* call ``plugin.validate(dataset)`` (catch + report crashes)
* notify ``on_plugin_complete`` for passed plugins
* notify ``on_plugin_failed`` (with recommendations) for failed plugins
  AND for crashes — same recorder surface, so the artifact accumulator
  treats both uniformly
* track ``failed_plugins`` count against the dataset's
  ``critical_failures`` threshold; early-break the loop when reached
* notify ``on_validation_completed`` / ``on_validation_failed`` at the end
* return ``dict[str, Any]`` on success or raise
  :class:`DatasetValidationFailedError` whose ``context["critical"]`` flag
  distinguishes the legacy ``DATASET_VALIDATION_CRITICAL_FAILURE`` /
  ``DATASET_VALIDATION_ERROR`` codes

Phase 4 (event-system unification): the runner now takes the
:class:`ValidationArtifactManager` directly (replaces the previous
``DatasetValidatorEventCallbacks`` dataclass). The artifact recorder is
optional — when ``None`` no per-plugin rows are accumulated and the
runner still returns / raises the same shape so existing test fixtures
that exercise the pure plugin-loop semantics work without rewiring.

Post-Phase-10 visibility gap close (2026-05-17): the runner also takes
an optional :class:`IEventEmitter` and a ``run_id`` so each plugin run
surfaces typed
``ryotenkai.control.dataset.validation_plugin_{started,completed,failed}``
envelopes on the unified timeline — parity with the evaluation runner's
per-plugin events. When the emitter is ``None`` the runner silently
skips emission (legacy tests / standalone invocations stay unaffected).
"""

from __future__ import annotations

import traceback
from time import perf_counter
from typing import TYPE_CHECKING, Any, Literal

from ryotenkai_control.data.validation.base import ValidationPlugin
from ryotenkai_control.pipeline.stages.dataset_validator.constants import (
    VALIDATION_STATUS_KEY,
    VALIDATION_STATUS_PASSED,
    VALIDATIONS_ATTR,
)
from ryotenkai_shared.errors import DatasetValidationFailedError
from ryotenkai_shared.events import UNKNOWN_OFFSET
from ryotenkai_shared.events.types.control_dataset import (
    DatasetValidationPluginCompletedEvent,
    DatasetValidationPluginCompletedPayload,
    DatasetValidationPluginFailedEvent,
    DatasetValidationPluginFailedPayload,
    DatasetValidationPluginStartedEvent,
    DatasetValidationPluginStartedPayload,
)
from ryotenkai_shared.utils.logger import logger

if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset

    from ryotenkai_control.pipeline.stages.dataset_validator.artifact_manager import (
        ValidationArtifactManager,
    )
    from ryotenkai_control.pipeline.stages.dataset_validator.plugin_loader import PluginTuple
    from ryotenkai_shared.events import IEventEmitter


# Source URI for envelopes the per-plugin emitter publishes. Matches
# the stage-level source so consumers can correlate per-plugin events
# with the outer dataset_validator ``validation_started`` /
# ``validation_completed`` span by ``(run_id, source)``.
_RUNNER_EVENT_SOURCE = "control://orchestrator/dataset_validator"

# Hard ceiling on the per-event traceback excerpt. The journal row
# stays bounded; full tracebacks belong in logs / artifact JSON, not
# the SSE timeline.
_TRACEBACK_EXCERPT_LIMIT = 2048


def _truncate_traceback(exc: BaseException, *, limit: int = _TRACEBACK_EXCERPT_LIMIT) -> str:
    """Render an exception's traceback and clip to ``limit`` characters.

    Trailing newlines are stripped so the truncated suffix reads
    cleanly when the journal row is rendered. ``limit`` is enforced
    at character (not byte) count — close enough to the documented
    2KB ceiling for ASCII-dominated tracebacks.
    """
    formatted = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    formatted = formatted.rstrip()
    if len(formatted) <= limit:
        return formatted
    return formatted[: limit - len("…[truncated]")] + "…[truncated]"


class PluginRunner:
    """Run a list of validation plugins against one dataset split.

    Parameters:
        artifact_recorder: optional accumulator that builds the
            ``dataset_validator_results.json`` artifact rows.
        emitter: optional :class:`IEventEmitter`. When supplied, each
            plugin run emits a typed
            ``ryotenkai.control.dataset.validation_plugin_*`` envelope
            trio (started + completed | failed). Legacy / standalone
            test fixtures that omit this argument keep the pre-event
            behaviour unchanged.
        run_id: required when ``emitter`` is supplied (envelopes carry
            ``run_id`` as an identity field). Ignored otherwise; the
            default ``""`` is sufficient when no emitter is wired.
    """

    def __init__(
        self,
        *,
        artifact_recorder: ValidationArtifactManager | None = None,
        emitter: IEventEmitter | None = None,
        run_id: str = "",
    ) -> None:
        self._artifact_recorder = artifact_recorder
        self._emitter = emitter
        self._run_id = run_id

    def run(
        self,
        dataset_name: str,
        dataset_path: str,
        dataset: Dataset | IterableDataset,
        dataset_config: Any,
        plugins: list[PluginTuple],
        *,
        split_name: Literal["train", "eval"],
    ) -> dict[str, Any]:
        """Run plugins; notify the recorder; honour critical_failures threshold.

        Returns the metrics dict on success. Raises
        :class:`DatasetValidationFailedError` on failure with
        ``context["critical"]`` set when the critical_failures threshold
        was reached (legacy ``DATASET_VALIDATION_CRITICAL_FAILURE``).
        """
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

            self._record_plugin_start(
                dataset_name, dataset_path, plugin_id, plugin_name, plugin.get_description(),
            )
            # Typed event emission — mirrors the evaluation runner's
            # per-plugin pattern so the timeline shows uniform
            # "plugin span" UI primitives across both stages.
            self._emit_plugin_started(plugin, dataset_path)

            try:
                result = plugin.validate(dataset)

                for key, value in result.metrics.items():
                    all_metrics[f"{split_name}.{plugin_id}.{key}"] = value

                all_warnings.extend(result.warnings)

                if result.passed:
                    logger.info(
                        f"    [{dataset_name}] OK {plugin_id} ({plugin_name}) passed "
                        f"({result.execution_time_ms:.1f}ms)"
                    )
                    self._record_plugin_complete(
                        dataset_name,
                        dataset_path,
                        plugin_id,
                        plugin.name,
                        result.params,
                        result.thresholds,
                        result.metrics,
                        result.execution_time_ms,
                    )
                    self._emit_plugin_completed(
                        plugin,
                        result,
                        duration_s=perf_counter() - plugin_started_at,
                    )
                else:
                    logger.error(f"    [{dataset_name}] FAIL {plugin_id} ({plugin_name}) failed")
                    display_errors = list(result.errors)
                    display_errors.extend(ValidationPlugin.render_error_groups(result.error_groups))
                    all_errors.extend(display_errors)
                    failed_plugins.append(plugin_id)

                    recommendations = plugin.get_recommendations(result)
                    all_recommendations.extend(recommendations)

                    self._record_plugin_failed(
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
                    # Plugin returned ``passed=False`` is still a
                    # "completed" run from the timeline's POV — the
                    # plugin invoked successfully, it just reported
                    # failures. The ``num_failed`` / ``num_passed``
                    # split on the payload carries the verdict.
                    # ``validation_plugin_failed`` is reserved for the
                    # crash path below (plugin threw an exception).
                    self._emit_plugin_completed(
                        plugin,
                        result,
                        duration_s=perf_counter() - plugin_started_at,
                    )

                    if self._is_critical_threshold_reached(dataset_config, failed_plugins):
                        critical_threshold_reached = True
                        self._log_critical_threshold_reached(dataset_name, dataset_config, failed_plugins)
                        break
            except Exception as e:
                crashed_duration_ms = (perf_counter() - plugin_started_at) * 1000.0
                error_msg = f"Plugin '{plugin_id}' ({plugin_name}) crashed: {e}"
                logger.error(f"    [{dataset_name}] FAIL {error_msg}")
                all_errors.append(error_msg)
                failed_plugins.append(plugin_id)

                # Crashes notify the same on_plugin_failed recorder so
                # the artifact accumulator treats them uniformly with
                # regular failures.
                self._record_plugin_failed(
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
                # Crash → emit the typed ``plugin_failed`` envelope
                # with truncated traceback for the timeline.
                self._emit_plugin_failed(plugin, e)

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

    # Typed event emission helpers — no-op when ``self._emitter`` is None
    # so the standalone tests / legacy fixtures don't need to wire one.
    # The ``plugin`` argument is :class:`ValidationPlugin`; we read
    # ``plugin.name`` (manifest identity) and ``plugin.version``
    # (declared in manifest.toml; defaults to ``""`` if unset).
    #
    # ``num_checked`` / ``num_passed`` / ``num_failed`` are derived from
    # the :class:`ValidationResult` triplet at this seam because the
    # dataclass itself does not surface them — different plugins
    # measure "checked" against different denominators (samples vs.
    # ngrams vs. characters). We pick the most defensible
    # interpretation per result:
    #
    #   * ``num_checked = num_passed + num_failed`` when error_groups
    #     carry counts (the runner has visibility on both passed and
    #     failed sample counts).
    #   * ``num_failed = sum(group.total_count for group in error_groups)``.
    #     This is the most actionable signal for operators triaging
    #     "how many samples did this plugin flag?".
    #   * ``num_passed`` falls back to ``0`` when only failed counts
    #     are available — better than fabricating numbers from
    #     ``len(dataset)`` that the plugin may not have inspected.

    def _emit_plugin_started(self, plugin: ValidationPlugin, dataset_path: str) -> None:
        if self._emitter is None:
            return
        self._emitter.emit(
            DatasetValidationPluginStartedEvent(
                source=_RUNNER_EVENT_SOURCE,
                run_id=self._run_id,
                offset=UNKNOWN_OFFSET,
                payload=DatasetValidationPluginStartedPayload(
                    plugin_name=plugin.name,
                    plugin_version=getattr(plugin, "version", "") or "",
                    dataset_path=dataset_path,
                ),
            ),
        )

    def _emit_plugin_completed(
        self,
        plugin: ValidationPlugin,
        result: Any,
        *,
        duration_s: float,
    ) -> None:
        if self._emitter is None:
            return
        # Compute num_failed from error_groups (machine-readable
        # source of truth). When error_groups is empty + passed=True we
        # know the plugin found nothing wrong; when groups exist they
        # carry total_count per error type.
        error_groups = getattr(result, "error_groups", []) or []
        num_failed = sum(int(getattr(g, "total_count", 0)) for g in error_groups)
        # When the plugin passed, we report 0 failures regardless of
        # whether error_groups is empty — defensive against plugins
        # that emit warnings via error_groups but still pass.
        if getattr(result, "passed", False):
            num_failed = 0
        # num_passed is the complement we *can* defend: when the
        # plugin surfaced a "sample_count" or similar metric the
        # runner can subtract; otherwise zero. We keep this
        # conservative — the artifact JSON is the authoritative
        # source for detailed per-sample breakdown.
        num_passed = 0
        metrics = getattr(result, "metrics", {}) or {}
        for key in ("num_passed", "sample_count", "num_checked"):
            if key in metrics:
                try:
                    num_passed = max(0, int(metrics[key]) - num_failed)
                except (TypeError, ValueError):
                    num_passed = 0
                break
        num_checked = num_passed + num_failed
        self._emitter.emit(
            DatasetValidationPluginCompletedEvent(
                source=_RUNNER_EVENT_SOURCE,
                run_id=self._run_id,
                offset=UNKNOWN_OFFSET,
                payload=DatasetValidationPluginCompletedPayload(
                    plugin_name=plugin.name,
                    num_checked=num_checked,
                    num_passed=num_passed,
                    num_failed=num_failed,
                    duration_s=duration_s,
                ),
            ),
        )

    def _emit_plugin_failed(self, plugin: ValidationPlugin, exc: BaseException) -> None:
        if self._emitter is None:
            return
        self._emitter.emit(
            DatasetValidationPluginFailedEvent(
                source=_RUNNER_EVENT_SOURCE,
                run_id=self._run_id,
                offset=UNKNOWN_OFFSET,
                payload=DatasetValidationPluginFailedPayload(
                    plugin_name=plugin.name,
                    error_type=type(exc).__name__,
                    message=str(exc),
                    traceback_excerpt=_truncate_traceback(exc),
                ),
            ),
        )

    def _record_plugin_start(
        self,
        dataset_name: str,
        dataset_path: str,
        plugin_id: str,
        plugin_name: str,
        description: str,
    ) -> None:
        if self._artifact_recorder is None:
            return
        self._artifact_recorder.on_plugin_start(
            dataset_name, dataset_path, plugin_id, plugin_name, description,
        )

    def _record_plugin_complete(
        self,
        dataset_name: str,
        dataset_path: str,
        plugin_id: str,
        plugin_name: str,
        params: dict,
        thresholds: dict,
        metrics: dict,
        duration_ms: float,
    ) -> None:
        if self._artifact_recorder is None:
            return
        self._artifact_recorder.on_plugin_complete(
            dataset_name,
            dataset_path,
            plugin_id,
            plugin_name,
            params,
            thresholds,
            metrics,
            duration_ms,
        )

    def _record_plugin_failed(
        self,
        dataset_name: str,
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
        if self._artifact_recorder is None:
            return
        self._artifact_recorder.on_plugin_failed(
            dataset_name,
            dataset_path,
            plugin_id,
            plugin_name,
            params,
            thresholds,
            metrics,
            duration_ms,
            errors,
            recommendations,
        )

    def _record_validation_completed(
        self,
        dataset_name: str,
        dataset_path: str,
        metrics: dict,
        warnings: list[str],
    ) -> None:
        if self._artifact_recorder is None:
            return
        self._artifact_recorder.on_validation_completed(
            dataset_name, dataset_path, metrics, warnings,
        )

    def _record_validation_failed(
        self,
        dataset_name: str,
        dataset_path: str,
        errors: list[str],
    ) -> None:
        if self._artifact_recorder is None:
            return
        self._artifact_recorder.on_validation_failed(
            dataset_name, dataset_path, errors,
        )

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
    ) -> dict[str, Any]:
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
                logger.info(f"[{dataset_name}] Recommendations:")
                for rec in all_recommendations:
                    logger.info(f"  - {rec}")

            self._record_validation_failed(dataset_name, dataset_path, all_errors)

            error_summary = f"{len(all_errors)} validation errors"
            raise DatasetValidationFailedError(
                detail=error_summary,
                context={
                    "errors": all_errors,
                    "critical": critical_threshold_reached,
                    "dataset_name": dataset_name,
                },
            )

        logger.info(f"[{dataset_name}] All validation checks passed!")

        self._record_validation_completed(
            dataset_name, dataset_path, all_metrics, all_warnings,
        )

        return {
            VALIDATION_STATUS_KEY: VALIDATION_STATUS_PASSED,
            "warnings": all_warnings,
            **all_metrics,
        }


__all__ = ["PluginRunner"]
