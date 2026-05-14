"""
Stage 0: Dataset Validator
Validates dataset quality before training to prevent wasted resources.

Now supports plugin system for extensible validation.

Phase A2 Batch 8 — raise-based migration. ``execute()`` returns
``dict[str, Any]`` and raises typed errors
(:class:`DatasetValidationFailedError` / :class:`DatasetLoadFailedError`).
Internal helpers no longer return ``Result``; they raise and the parent
catches per-dataset to accumulate advisory mode reports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ryotenkai_control.pipeline.stages.base import PipelineStage
from ryotenkai_control.pipeline.stages.constants import StageNames
from ryotenkai_control.pipeline.stages.dataset_validator.constants import (
    CRITICAL_FAILURES_ATTR,
    SPLIT_EVAL,
    SPLIT_TRAIN,
    VALIDATION_MODE_ATTR,
    VALIDATION_MODE_FAST,
    VALIDATION_STATUS_FAILED,
    VALIDATION_STATUS_KEY,
    VALIDATION_STATUS_PASSED,
    VALIDATION_STATUS_SKIPPED,
    VALIDATIONS_ATTR,
    WARNINGS_KEY,
)
from ryotenkai_control.pipeline.stages.dataset_validator.format_checker import FormatChecker
from ryotenkai_control.pipeline.stages.dataset_validator.plugin_loader import PluginLoader
from ryotenkai_control.pipeline.stages.dataset_validator.plugin_runner import PluginRunner
from ryotenkai_control.pipeline.stages.dataset_validator.split_loader import DatasetSplitLoader
from ryotenkai_pod.trainer.data_loaders.factory import DatasetLoaderFactory
from ryotenkai_shared.errors import (
    DatasetLoadFailedError,
    DatasetValidationFailedError,
)
from ryotenkai_shared.utils.logger import logger

if TYPE_CHECKING:
    from collections.abc import Callable

    from ryotenkai_shared.config import PipelineConfig
    from ryotenkai_shared.config.secrets.model import Secrets


# =============================================================================
# EVENT CALLBACKS
# =============================================================================


@dataclass
class DatasetValidatorEventCallbacks:
    """
    Callbacks for DatasetValidator events (SOLID-compliant event collection).

    Used to integrate DatasetValidator with MLflow or other logging systems.

    All callbacks now include dataset context (dataset_name, dataset_path)
    to support multi-dataset validation and proper event grouping.
    """

    # Dataset scheduled for validation (NEW)
    on_dataset_scheduled: Callable[[str, str, str], None] | None = None
    # Args: dataset_name, dataset_path, validation_mode

    # Dataset loaded event
    on_dataset_loaded: Callable[[str, str, int, int], None] | None = None
    # Args: dataset_name, dataset_path, sample_count, validation_critical_failures

    # Validation completed event
    on_validation_completed: Callable[[str, str, dict, list[str]], None] | None = None
    # Args: dataset_name, dataset_path, metrics, warnings

    # Validation failed event
    on_validation_failed: Callable[[str, str, list[str]], None] | None = None
    # Args: dataset_name, dataset_path, errors

    # Plugin-specific events (with dataset context)
    on_plugin_start: Callable[[str, str, str, str, str], None] | None = None
    # Args: dataset_name, dataset_path, plugin_id, plugin_name, description

    on_plugin_complete: Callable[[str, str, str, str, dict, dict, dict, float], None] | None = None
    # Args: dataset_name, dataset_path, plugin_id, plugin_name, params, thresholds, metrics, duration_ms

    on_plugin_failed: Callable[[str, str, str, str, dict, dict, dict, float, list[str], list[str]], None] | None = None
    # Args: dataset_name, dataset_path, plugin_id, plugin_name, params, thresholds, metrics, duration_ms, errors, recommendations


def _is_critical_validation_failure(exc: BaseException) -> bool:
    """True iff this exception represents a critical-threshold violation.

    The plugin runner tags exceptions with ``context["critical"]`` when
    the dataset's ``validations.critical_failures`` threshold is reached.
    Other typed errors (e.g. :class:`DatasetLoadFailedError`) are NOT
    automatically critical — the parent loop decides per
    ``dataset_config.validations.critical_failures`` whether to fail fast.
    Generic non-typed exceptions ARE treated as critical because we
    cannot know whether the dataset is partially validated.
    """
    if isinstance(exc, DatasetValidationFailedError):
        return bool(getattr(exc, "context", {}).get("critical", False))
    # Load failures are not automatically critical; generic crashes are
    # (conservative — we cannot know whether the dataset is partially
    # validated).
    return not isinstance(exc, DatasetLoadFailedError)


class DatasetValidator(PipelineStage):
    """
    Validates dataset quality using plugin-based system.

    Features:
    - Supports local and HuggingFace datasets (via DatasetLoaderFactory)
    - Streaming support for large HF datasets
    - Extensible plugin system for custom validations
    - DTST_* secret injection for plugins that require external credentials

    Configuration:
    - dataset.validations - explicit plugin configuration (preferred)
    - If not specified, uses sensible defaults
    """

    def __init__(
        self,
        config: PipelineConfig,
        secrets: Secrets | None = None,
        callbacks: DatasetValidatorEventCallbacks | None = None,
    ):
        super().__init__(config, StageNames.DATASET_VALIDATOR)
        self._config = config
        self._callbacks = callbacks or DatasetValidatorEventCallbacks()
        self._secrets = secrets

        self._loader_factory = DatasetLoaderFactory(config)
        self._format_checker = FormatChecker(config)
        self._plugin_loader = PluginLoader(config, secrets)
        self._split_loader = DatasetSplitLoader(self._loader_factory)
        self._plugin_runner = PluginRunner(self._callbacks)

        # Eager validation of the primary dataset's plugin config: instantiate
        # plugins now so a typo in `validations.plugins[*].plugin` fails the
        # stage at construction time (before GPU spin-up) rather than mid-run.
        # If the primary dataset has no validations.plugins block, this is a
        # no-op — no hidden defaults are injected.
        self._plugin_loader.load_for_dataset(config.get_primary_dataset())

    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        """
        Validate all datasets used in training strategies.

        Supports multiple datasets with parallel validation using ThreadPoolExecutor.
        Each dataset is validated independently with its own plugins and settings.

        Args:
            context: Pipeline context

        Returns:
            Aggregated validation metrics dict.

        Raises:
            DatasetValidationFailedError: when a dataset's critical-failure
                threshold is reached (or a worker thread crashes
                unexpectedly), aborting the pipeline.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from contextvars import copy_context

        # FIX: Prevent tqdm race condition in multi-threading
        # See: https://github.com/huggingface/datasets/issues/7660
        # Problem: When multiple threads call load_dataset() simultaneously, tqdm._lock
        # initialization causes AttributeError. This is a known HuggingFace datasets issue.
        #
        # Solution: Initialize tqdm in main thread BEFORE ThreadPoolExecutor starts.
        # This ensures tqdm._lock exists before any threads try to use it.
        try:
            import tqdm

            # Warm up tqdm by creating a dummy instance to initialize _lock
            _ = tqdm.tqdm([], desc="Initializing tqdm", disable=True)
            logger.debug("[VALIDATOR] Initialized tqdm for thread-safe dataset loading")
        except Exception as e:
            logger.warning(f"[VALIDATOR] Failed to initialize tqdm: {e}")

        # Get all unique datasets from training strategies
        datasets_to_validate = self._get_datasets_to_validate()

        if not datasets_to_validate:
            logger.warning("[VALIDATOR] No datasets found to validate")
            return {VALIDATION_STATUS_KEY: VALIDATION_STATUS_SKIPPED, "message": "No datasets configured"}

        logger.info(f"[VALIDATOR] Validating {len(datasets_to_validate)} dataset(s)")

        # Log all datasets scheduled for validation (for complete reporting)
        for dataset_name, (dataset_config, _strategy_phases) in datasets_to_validate.items():
            dataset_path = self._split_loader.get_train_ref(dataset_config)
            validation_mode = getattr(
                getattr(dataset_config, VALIDATIONS_ATTR, None), VALIDATION_MODE_ATTR, VALIDATION_MODE_FAST
            )
            if self._callbacks.on_dataset_scheduled:
                self._callbacks.on_dataset_scheduled(
                    dataset_name,
                    dataset_path,
                    validation_mode,
                )
            logger.debug(f"[VALIDATOR] Scheduled: {dataset_name} ({dataset_path})")

        # Validate datasets in parallel. We collect per-dataset success
        # metrics or failure messages; critical failures bubble up to a
        # pipeline-aborting raise after the pool drains.
        all_metrics: dict[str, dict[str, Any]] = {}
        all_errors: list[str] = []
        critical_failure = False

        # Propagate ContextVars (e.g. per-stage logging context) into worker
        # threads — ThreadPoolExecutor does not do this automatically.
        parent_ctx = copy_context()
        with ThreadPoolExecutor(max_workers=len(datasets_to_validate)) as executor:
            futures = {
                executor.submit(
                    parent_ctx.run,
                    self._validate_single_dataset,
                    dataset_name,
                    dataset_config,
                    strategy_phases,
                ): (
                    dataset_name,
                    dataset_config,
                )
                for dataset_name, (dataset_config, strategy_phases) in datasets_to_validate.items()
            }

            def stop_on_critical_failure(reason: str, dataset_cfg: Any) -> None:
                nonlocal critical_failure
                critical_threshold = getattr(getattr(dataset_cfg, VALIDATIONS_ATTR, None), CRITICAL_FAILURES_ATTR, 0)
                if critical_threshold <= 0:
                    return
                logger.error(reason)
                logger.error("[VALIDATOR] Stopping validation pipeline (fail-fast mode)")
                critical_failure = True
                # Cancel remaining futures
                for remaining_future in futures:
                    if not remaining_future.done():
                        remaining_future.cancel()

            # Collect results with fail-fast support
            for future in as_completed(futures):
                dataset_name, dataset_config = futures[future]
                try:
                    metrics = future.result()
                    all_metrics[dataset_name] = metrics
                except DatasetValidationFailedError as exc:
                    error_msg = f"{dataset_name}: {exc.detail or exc}"
                    all_errors.append(error_msg)
                    if _is_critical_validation_failure(exc):
                        stop_on_critical_failure(
                            reason=f"[VALIDATOR] Critical failure detected for {dataset_name}",
                            dataset_cfg=dataset_config,
                        )
                        if critical_failure:
                            break
                except DatasetLoadFailedError as exc:
                    # Load failures are not automatically critical; they
                    # follow the legacy ``DATASET_LOAD_ERROR`` path which
                    # falls through to :meth:`_aggregate_results` so the
                    # caller sees a ``validation_status == "failed"`` dict
                    # (unless other datasets succeed, in which case this
                    # one becomes a warning). We preserve the legacy
                    # ``[DATASET_LOAD_ERROR] <detail>`` format so warnings
                    # remain greppable by callers.
                    legacy_code = exc.context.get("legacy_code", "DATASET_LOAD_ERROR")
                    error_msg = f"{dataset_name}: [{legacy_code}] {exc.detail or exc}"
                    all_errors.append(error_msg)
                except Exception as exc:
                    error_msg = f"{dataset_name}: validation crashed - {exc}"
                    logger.error(f"[VALIDATOR] {error_msg}")
                    all_errors.append(error_msg)

                    stop_on_critical_failure(
                        reason=f"[VALIDATOR] Critical failure (crash) detected for {dataset_name}",
                        dataset_cfg=dataset_config,
                    )
                    if critical_failure:
                        break

        # If critical failure detected, fail immediately
        if critical_failure:
            error_summary = (
                f"Dataset validation failed critically. "
                f"Validated {len(all_metrics)}/{len(datasets_to_validate)} datasets. "
                f"Errors: {'; '.join(all_errors)}"
            )
            raise DatasetValidationFailedError(
                detail=error_summary,
                context={
                    "legacy_code": "VALIDATION_CRITICAL_FAILURE",
                    "errors": all_errors,
                    "datasets_validated": len(all_metrics),
                    "datasets_total": len(datasets_to_validate),
                },
            )

        # Aggregate results
        return self._aggregate_results(all_metrics, all_errors, len(datasets_to_validate))

    def _get_datasets_to_validate(self) -> dict[str, tuple[Any, list[Any]]]:
        """
        Extract all unique datasets from training strategies.

        Returns dict: {dataset_name: (DatasetConfig, [StrategyPhaseConfig, ...])}
        Each dataset maps to its config and all strategy phases that consume it.
        """
        datasets: dict[str, tuple[Any, list[Any]]] = {}

        # Get datasets from training strategies
        if hasattr(self._config, "training") and hasattr(self._config.training, "strategies"):
            for strategy in self._config.training.strategies:
                dataset_name = strategy.dataset
                if dataset_name:
                    try:
                        dataset_config = self._config.get_dataset_for_strategy(strategy)
                        if dataset_name not in datasets:
                            datasets[dataset_name] = (dataset_config, [])
                        datasets[dataset_name][1].append(strategy)
                    except KeyError:
                        logger.warning(f"[VALIDATOR] Dataset '{dataset_name}' not found in config")

        # Fallback: use primary dataset if no strategies defined
        if not datasets:
            logger.debug("[VALIDATOR] No strategies found, using primary dataset")
            primary = self._config.get_primary_dataset()
            datasets["primary"] = (primary, [])

        return datasets

    def _validate_single_dataset(
        self,
        dataset_name: str,
        dataset_config: Any,
        strategy_phases: list[Any],
    ) -> dict[str, Any]:
        """
        Validate a single dataset with its specific configuration.

        Args:
            dataset_name: Unique dataset identifier
            dataset_config: Dataset configuration
            strategy_phases: Strategy phase configs that consume this dataset

        Returns:
            Validation metrics dict (merged from train + optional eval).

        Raises:
            DatasetLoadFailedError: if the train split could not be loaded.
            DatasetValidationFailedError: on format check or plugin failure.
                ``context["critical"]`` flags critical-threshold violations.
        """
        logger.info(f"[VALIDATOR] Starting validation for dataset: {dataset_name}")

        # Load train split
        dataset = self._split_loader.load_train(dataset_config)

        if dataset is None:
            raise DatasetLoadFailedError(
                detail=f"Failed to load dataset: {dataset_name}",
                context={"dataset_name": dataset_name, "legacy_code": "DATASET_LOAD_ERROR"},
            )

        # FORMAT CHECK FIRST — fail fast before quality plugins (before GPU)
        # Propagates DatasetValidationFailedError verbatim.
        self._format_checker.check(dataset, dataset_name, strategy_phases)

        # Get dataset train ref for event tracking
        dataset_path = self._split_loader.get_train_ref(dataset_config)

        # Fire callback: train dataset loaded
        if self._callbacks.on_dataset_loaded:
            sample_count = self._split_loader.get_size(dataset)
            critical_threshold = getattr(getattr(dataset_config, VALIDATIONS_ATTR, None), "critical_failures", 0)
            self._callbacks.on_dataset_loaded(dataset_name, dataset_path, sample_count, critical_threshold)

        # Load plugins for this dataset (each item: (plugin_id, plugin_name, plugin_instance, apply_to_set))
        plugins = self._plugin_loader.load_for_dataset(dataset_config)

        # Run validations: train always
        train_plugins = [
            (plugin_id, plugin_name, p, apply_to)
            for (plugin_id, plugin_name, p, apply_to) in plugins
            if SPLIT_TRAIN in apply_to
        ]
        merged: dict[str, Any] = {}
        errors: list[str] = []
        critical_errors: list[str] = []

        try:
            train_metrics = self._plugin_runner.run(
                dataset_name,
                dataset_path,
                dataset,
                dataset_config,
                train_plugins,
                split_name=SPLIT_TRAIN,  # type: ignore[arg-type]
            )
            merged.update(train_metrics)
        except DatasetValidationFailedError as train_exc:
            errors.append(f"train: {train_exc.detail or train_exc}")
            if _is_critical_validation_failure(train_exc):
                critical_errors.append(f"train: {train_exc.detail or train_exc}")

        # Optional eval
        eval_dataset, eval_ref = self._split_loader.try_load_eval(dataset_config)
        if eval_dataset is not None and eval_ref is not None:
            # FORMAT CHECK for eval dataset too
            self._format_checker.check(eval_dataset, dataset_name, strategy_phases)

            eval_plugins = [
                (plugin_id, plugin_name, p, apply_to)
                for (plugin_id, plugin_name, p, apply_to) in plugins
                if SPLIT_EVAL in apply_to
            ]
            try:
                eval_metrics = self._plugin_runner.run(
                    dataset_name,
                    eval_ref,
                    eval_dataset,
                    dataset_config,
                    eval_plugins,
                    split_name=SPLIT_EVAL,  # type: ignore[arg-type]
                )
                merged.update(eval_metrics)
            except DatasetValidationFailedError as eval_exc:
                errors.append(f"eval: {eval_exc.detail or eval_exc}")
                if _is_critical_validation_failure(eval_exc):
                    critical_errors.append(f"eval: {eval_exc.detail or eval_exc}")

        if critical_errors:
            raise DatasetValidationFailedError(
                detail="; ".join(critical_errors),
                context={
                    "critical": True,
                    "dataset_name": dataset_name,
                    "legacy_code": "DATASET_VALIDATION_CRITICAL_FAILURE",
                },
            )

        if errors:
            raise DatasetValidationFailedError(
                detail="; ".join(errors),
                context={
                    "critical": False,
                    "dataset_name": dataset_name,
                    "legacy_code": "DATASET_VALIDATION_ERROR",
                },
            )

        return merged

    @staticmethod
    def _aggregate_results(
        all_metrics: dict[str, dict[str, Any]],
        all_errors: list[str],
        total_datasets: int,
    ) -> dict[str, Any]:
        """
        Aggregate validation results from all datasets.

        Continue strategy: collect all errors, return aggregated metrics.
        ``all_metrics`` carries successful datasets (post-raise migration:
        a dataset is either fully successful and present here, or its
        error message is in ``all_errors``).
        """
        aggregated_metrics: dict[str, Any] = {}
        aggregated_warnings: list[str] = []

        for dataset_name, metrics in all_metrics.items():
            # Prefix metrics with dataset name
            for key, value in metrics.items():
                aggregated_metrics[f"{dataset_name}.{key}"] = value

            # Collect warnings
            warnings = metrics.get(WARNINGS_KEY, [])
            for warning in warnings:
                aggregated_warnings.append(f"[{dataset_name}] {warning}")

        datasets_passed = len(all_metrics)

        # If all datasets failed, return failed-status dict (advisory mode).
        # Critical failures are handled earlier (fail-fast raise).
        if all_errors and not aggregated_metrics:
            error_summary = "; ".join(all_errors)
            return {
                VALIDATION_STATUS_KEY: VALIDATION_STATUS_FAILED,
                "datasets_validated": total_datasets,
                "datasets_passed": 0,
                "datasets_failed": len(all_errors),
                WARNINGS_KEY: [f"ERROR: {e}" for e in all_errors],
                "message": f"All datasets failed validation (non-critical): {error_summary}",
            }

        # If some failed, add warnings
        if all_errors:
            for error in all_errors:
                aggregated_warnings.append(f"ERROR: {error}")

        # Build final result with all aggregated data
        final_result_data: dict[str, Any] = {
            VALIDATION_STATUS_KEY: VALIDATION_STATUS_FAILED if all_errors else VALIDATION_STATUS_PASSED,
            "datasets_validated": total_datasets,
            "datasets_passed": datasets_passed,
            "datasets_failed": len(all_errors),
            **aggregated_metrics,
        }

        if aggregated_warnings:
            final_result_data[WARNINGS_KEY] = aggregated_warnings

        return final_result_data
