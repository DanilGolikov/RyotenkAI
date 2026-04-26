"""
Stage 0: Dataset Validator
Validates dataset quality before training to prevent wasted resources.

Now supports plugin system for extensible validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any, Literal

from src.config.datasets.constants import SOURCE_TYPE_HUGGINGFACE
from src.data.loaders.factory import DatasetLoaderFactory
from src.data.validation.base import ValidationPlugin
from src.pipeline.stages.base import PipelineStage
from src.pipeline.stages.constants import StageNames
from src.pipeline.stages.dataset_validator.constants import (
    CRITICAL_FAILURES_ATTR,
    SPLIT_EVAL,
    SPLIT_TRAIN,
    VALIDATION_MAX_SAMPLES_FAST,
    VALIDATION_MODE_ATTR,
    VALIDATION_MODE_FAST,
    VALIDATION_STATUS_FAILED,
    VALIDATION_STATUS_KEY,
    VALIDATION_STATUS_PASSED,
    VALIDATION_STATUS_SKIPPED,
    VALIDATIONS_ATTR,
    WARNINGS_KEY,
)
from src.pipeline.stages.dataset_validator.format_checker import FormatChecker
from src.pipeline.stages.dataset_validator.plugin_loader import PluginLoader
from src.utils.logger import logger
from src.utils.result import AppError, DatasetError, Err, Ok, Result

if TYPE_CHECKING:
    from collections.abc import Callable

    from datasets import Dataset, IterableDataset

    from src.config.secrets.model import Secrets
    from src.utils.config import PipelineConfig


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

        # Eagerly resolve the default plugin set so a misconfigured registry
        # fails fast at construction time (mirrors original behaviour, was
        # the only purpose of self._plugins which is otherwise unused).
        self._plugin_loader.load_for_default_dataset()

    def execute(self, context: dict[str, Any]) -> Result[dict[str, Any], AppError]:
        """
        Validate all datasets used in training strategies.

        Supports multiple datasets with parallel validation using ThreadPoolExecutor.
        Each dataset is validated independently with its own plugins and settings.

        Args:
            context: Pipeline context

        Returns:
            Result with aggregated validation metrics or DatasetError
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
            return Ok({VALIDATION_STATUS_KEY: VALIDATION_STATUS_SKIPPED, "message": "No datasets configured"})

        logger.info(f"[VALIDATOR] Validating {len(datasets_to_validate)} dataset(s)")

        # Log all datasets scheduled for validation (for complete reporting)
        for dataset_name, (dataset_config, _strategy_phases) in datasets_to_validate.items():
            dataset_path = self._get_dataset_train_ref(dataset_config)
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

        # Validate datasets in parallel
        all_results = {}
        all_errors = []
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
                    context,
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
                    result = future.result()
                    all_results[dataset_name] = result

                    if result.is_err():
                        error_msg = f"{dataset_name}: {result.unwrap_err()}"
                        all_errors.append(error_msg)

                        err = result.unwrap_err()
                        if isinstance(err, AppError) and err.code == "DATASET_VALIDATION_CRITICAL_FAILURE":
                            stop_on_critical_failure(
                                reason=f"[VALIDATOR] Critical failure detected for {dataset_name}",
                                dataset_cfg=dataset_config,
                            )
                            if critical_failure:
                                break

                except Exception as e:
                    error_msg = f"{dataset_name}: validation crashed - {e}"
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
                f"Validated {len(all_results)}/{len(datasets_to_validate)} datasets. "
                f"Errors: {'; '.join(all_errors)}"
            )
            return Err(DatasetError(message=error_summary, code="VALIDATION_CRITICAL_FAILURE"))

        # Aggregate results
        return self._aggregate_results(all_results, all_errors)

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
        context: dict[str, Any],
    ) -> Result[dict[str, Any], AppError]:
        """
        Validate a single dataset with its specific configuration.

        Args:
            dataset_name: Unique dataset identifier
            dataset_config: Dataset configuration
            strategy_phases: Strategy phase configs that consume this dataset
            context: Pipeline context

        Returns:
            Result with validation metrics or DatasetError
        """
        logger.info(f"[VALIDATOR] Starting validation for dataset: {dataset_name}")

        # Load dataset
        loader = self._loader_factory.create_for_dataset(dataset_config)
        dataset = self._load_dataset_for_validation(dataset_config, loader)

        if dataset is None:
            return Err(DatasetError(message=f"Failed to load dataset: {dataset_name}", code="DATASET_LOAD_ERROR"))

        # FORMAT CHECK FIRST — fail fast before quality plugins (before GPU)
        fmt_result = self._check_dataset_format(dataset, dataset_name, strategy_phases)
        if fmt_result.is_err():
            return fmt_result

        # Get dataset train ref for event tracking
        dataset_path = self._get_dataset_train_ref(dataset_config)

        # Fire callback: train dataset loaded
        if self._callbacks.on_dataset_loaded:
            sample_count = self._get_dataset_size(dataset)
            critical_threshold = getattr(getattr(dataset_config, VALIDATIONS_ATTR, None), "critical_failures", 0)
            self._callbacks.on_dataset_loaded(dataset_name, dataset_path, sample_count, critical_threshold)

        # Load plugins for this dataset (each item: (plugin_id, plugin_name, plugin_instance, apply_to_set))
        plugins = self._load_plugins_for_dataset(dataset_config)

        # Run validations: train always
        train_plugins = [
            (plugin_id, plugin_name, p, apply_to)
            for (plugin_id, plugin_name, p, apply_to) in plugins
            if SPLIT_TRAIN in apply_to
        ]
        train_result = self._run_plugin_validations(
            dataset_name,
            dataset_path,
            dataset,
            context,
            dataset_config,
            train_plugins,
            split_name=SPLIT_TRAIN,  # type: ignore[arg-type]
        )

        # Optional eval
        eval_result: Result[dict[str, Any], AppError] | None = None
        eval_dataset, eval_ref = self._try_load_eval_dataset_for_validation(dataset_config, loader)
        if eval_dataset is not None and eval_ref is not None:
            # FORMAT CHECK for eval dataset too
            fmt_eval_result = self._check_dataset_format(eval_dataset, dataset_name, strategy_phases)
            if fmt_eval_result.is_err():
                return fmt_eval_result

            eval_plugins = [
                (plugin_id, plugin_name, p, apply_to)
                for (plugin_id, plugin_name, p, apply_to) in plugins
                if SPLIT_EVAL in apply_to
            ]
            eval_result = self._run_plugin_validations(
                dataset_name,
                eval_ref,
                eval_dataset,
                context,
                dataset_config,
                eval_plugins,
                split_name=SPLIT_EVAL,  # type: ignore[arg-type]
            )

        # Merge results
        merged: dict[str, Any] = {}
        errors: list[str] = []
        critical_errors: list[str] = []

        if train_result.is_ok():
            merged.update(train_result.unwrap())
        else:
            train_error = train_result.unwrap_err()
            errors.append(f"train: {train_error}")
            if isinstance(train_error, AppError) and train_error.code == "DATASET_VALIDATION_CRITICAL_FAILURE":
                critical_errors.append(f"train: {train_error}")

        if eval_result is not None:
            if eval_result.is_ok():
                merged.update(eval_result.unwrap())
            else:
                eval_error = eval_result.unwrap_err()
                errors.append(f"eval: {eval_error}")
                if isinstance(eval_error, AppError) and eval_error.code == "DATASET_VALIDATION_CRITICAL_FAILURE":
                    critical_errors.append(f"eval: {eval_error}")

        if critical_errors:
            return Err(DatasetError(message="; ".join(critical_errors), code="DATASET_VALIDATION_CRITICAL_FAILURE"))

        if errors:
            return Err(DatasetError(message="; ".join(errors), code="DATASET_VALIDATION_ERROR"))

        return Ok(merged)

    def _check_dataset_format(
        self,
        dataset: Dataset | IterableDataset,
        dataset_name: str,
        strategy_phases: list[Any],
    ) -> Result[None, AppError]:
        """Proxy to :meth:`FormatChecker.check` — kept until callers migrate."""
        return self._format_checker.check(dataset, dataset_name, strategy_phases)

    def _load_plugins_for_dataset(self, dataset_config: Any) -> list:
        """Proxy to :meth:`PluginLoader.load_for_dataset` — kept until callers migrate."""
        return self._plugin_loader.load_for_dataset(dataset_config)

    @staticmethod
    def _aggregate_results(
        all_results: dict[str, Result[dict[str, Any], AppError]],
        all_errors: list[str],
    ) -> Result[dict[str, Any], AppError]:
        """
        Aggregate validation results from all datasets.

        Continue strategy: collect all errors, return aggregated metrics.
        """
        aggregated_metrics: dict[str, Any] = {}
        aggregated_warnings: list[str] = []

        for dataset_name, result in all_results.items():
            if result.is_ok():
                metrics = result.unwrap()
                # Prefix metrics with dataset name
                for key, value in metrics.items():
                    aggregated_metrics[f"{dataset_name}.{key}"] = value

                # Collect warnings
                warnings = metrics.get(WARNINGS_KEY, [])
                for warning in warnings:
                    aggregated_warnings.append(f"[{dataset_name}] {warning}")

        # If all datasets failed, return Ok with warnings (advisory mode).
        # Critical failures are handled earlier (fail-fast).
        if all_errors and not aggregated_metrics:
            error_summary = "; ".join(all_errors)
            result_data: dict[str, Any] = {
                VALIDATION_STATUS_KEY: VALIDATION_STATUS_FAILED,
                "datasets_validated": len(all_results),
                "datasets_passed": 0,
                "datasets_failed": len(all_errors),
                WARNINGS_KEY: [f"ERROR: {e}" for e in all_errors],
                "message": f"All datasets failed validation (non-critical): {error_summary}",
            }
            return Ok(result_data)

        # If some failed, add warnings
        if all_errors:
            for error in all_errors:
                aggregated_warnings.append(f"ERROR: {error}")

        # Build final result with all aggregated data
        final_result_data: dict[str, Any] = {
            VALIDATION_STATUS_KEY: VALIDATION_STATUS_FAILED if all_errors else VALIDATION_STATUS_PASSED,
            "datasets_validated": len(all_results),
            "datasets_passed": sum(1 for r in all_results.values() if r.is_ok()),
            "datasets_failed": len(all_errors),
            **aggregated_metrics,
        }

        if aggregated_warnings:
            final_result_data[WARNINGS_KEY] = aggregated_warnings

        return Ok(final_result_data)

    @staticmethod
    def _load_dataset_for_validation(
        dataset_config,
        loader,
        *,
        split_name: Literal["train", "eval"] = SPLIT_TRAIN,  # type: ignore[assignment]
    ) -> Dataset | IterableDataset | None:
        """
        Load dataset for validation with source type awareness.

        For HuggingFace: uses streaming mode to avoid loading entire dataset.
        For local files: loads normally via loader.

        Respects validation_mode from dataset_config:
        - fast: limits to 10K samples
        - full: loads entire dataset
        """
        from datasets import IterableDataset as HFIterableDataset

        source_type = dataset_config.get_source_type()
        validation_mode = getattr(
            getattr(dataset_config, VALIDATIONS_ATTR, None), VALIDATION_MODE_ATTR, VALIDATION_MODE_FAST
        )

        if source_type == SOURCE_TYPE_HUGGINGFACE:
            # HuggingFace: streaming mode + limit
            try:
                from datasets import load_dataset

                # Get token if configured
                token = None
                if hasattr(loader, "token"):
                    token = loader.token

                src = (
                    dataset_config.source_hf.train_id if split_name == SPLIT_TRAIN else dataset_config.source_hf.eval_id
                )
                if not src:
                    return None

                dataset = load_dataset(
                    src,
                    split="train",
                    streaming=True,  # KEY: Don't download full dataset!
                    token=token,
                    trust_remote_code=True,
                )

                # Check if result is iterable dataset
                if not isinstance(dataset, HFIterableDataset):
                    logger.warning(f"Expected IterableDataset, got {type(dataset)}")
                    return None

                # Apply validation mode
                if validation_mode == VALIDATION_MODE_FAST:
                    max_samples = dataset_config.max_samples or VALIDATION_MAX_SAMPLES_FAST
                    dataset = dataset.take(max_samples)
                    logger.info(f"Loaded HF dataset (streaming, fast mode): {src}")
                    logger.info(f"  Validation sample size: {max_samples}")
                else:
                    # full mode: no limit (will iterate through entire stream)
                    logger.info(f"Loaded HF dataset (streaming, full mode): {src}")
                    logger.warning("  Full validation mode - this may take a while for large datasets")

                return dataset

            except Exception as e:
                logger.error(f"Failed to load HF dataset: {e}")
                return None

        else:
            # Local files: normal load via loader
            source_local = dataset_config.source_local
            if source_local is None:
                return None
            local_path_str = (
                source_local.local_paths.train if split_name == SPLIT_TRAIN else source_local.local_paths.eval
            )
            if not local_path_str:
                return None

            local_path = Path(local_path_str)
            if not local_path.exists():
                logger.error(f"Dataset not found: {local_path}")
                return None

            try:
                dataset = loader.load(str(local_path), split="train")
                total_samples = len(dataset)

                # Apply validation mode for local datasets
                if validation_mode == VALIDATION_MODE_FAST and total_samples > VALIDATION_MAX_SAMPLES_FAST:
                    # Sample first 10K for fast mode
                    dataset = dataset.select(range(VALIDATION_MAX_SAMPLES_FAST))
                    logger.info(f"Loaded local dataset (fast mode): {local_path}")
                    logger.info(f"  Validation sample: {VALIDATION_MAX_SAMPLES_FAST} / {total_samples}")
                else:
                    logger.info(f"Loaded local dataset (full mode): {local_path}")
                    logger.info(f"  Total samples: {total_samples}")

                return dataset

            except Exception as e:
                logger.error(f"Failed to load local dataset: {e}")
                return None

    @staticmethod
    def _get_dataset_size(dataset: Dataset | IterableDataset) -> int:
        """Get dataset size (works with both types)."""
        from datasets import IterableDataset

        if isinstance(dataset, IterableDataset):
            return -1  # Unknown for streaming
        else:
            return len(dataset)

    def _run_plugin_validations(
        self,
        dataset_name: str,
        dataset_path: str,
        dataset: Dataset | IterableDataset,
        _context: dict[str, Any],
        dataset_config,
        plugins: list[tuple[str, str, Any, set[str]]],
        *,
        split_name: Literal["train", "eval"],
    ) -> Result[dict[str, Any], AppError]:
        """
        Run plugin-based validations for a single dataset.

        Args:
            dataset_name: Unique dataset identifier
            dataset_path: Dataset path/URI for event tracking
            dataset: Loaded dataset
            _context: Pipeline context (unused)
            dataset_config: Dataset configuration
            plugins: List of validation plugins to run
            split_name: Dataset split name ("train" or "eval")

        Returns:
            Result with validation metrics or DatasetError
        """
        logger.info(f"[{dataset_name}] Running {len(plugins)} validation plugins on {split_name}")

        all_metrics = {}
        all_errors = []
        all_warnings = []
        all_recommendations = []
        failed_plugins = []
        critical_threshold_reached = False

        for plugin_id, plugin_name, plugin, _apply_to in plugins:
            logger.info(f"  [{dataset_name}] Running plugin: {plugin_id} ({plugin_name}, {split_name})")
            plugin_started_at = perf_counter()

            # Fire callback: plugin start
            if self._callbacks.on_plugin_start:
                self._callbacks.on_plugin_start(
                    dataset_name,
                    dataset_path,
                    plugin_id,
                    plugin_name,
                    plugin.get_description(),
                )

            try:
                # Run validation
                result = plugin.validate(dataset)

                # Collect metrics
                for key, value in result.metrics.items():
                    all_metrics[f"{split_name}.{plugin_id}.{key}"] = value

                all_warnings.extend(result.warnings)

                if result.passed:
                    logger.info(
                        f"    [{dataset_name}] ✓ {plugin_id} ({plugin_name}) passed "
                        f"({result.execution_time_ms:.1f}ms)"
                    )

                    # Fire callback: plugin complete
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

                    # Get recommendations
                    recommendations = plugin.get_recommendations(result)
                    all_recommendations.extend(recommendations)

                    # Fire callback: plugin failed (include config, metrics, and recommendations)
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

                    # Check critical failure threshold
                    critical_threshold = getattr(
                        getattr(dataset_config, VALIDATIONS_ATTR, None), "critical_failures", 0
                    )
                    if 0 < critical_threshold <= len(failed_plugins):
                        critical_threshold_reached = True
                        logger.error(
                            f"    [{dataset_name}] Critical failure threshold reached: "
                            f"{len(failed_plugins)}/{critical_threshold} plugins failed"
                        )
                        logger.error(f"    [{dataset_name}] Stopping validation for this dataset")
                        break  # Stop validating this dataset
            # noinspection PyBroadException
            except Exception as e:
                crashed_duration_ms = (perf_counter() - plugin_started_at) * 1000.0
                error_msg = f"Plugin '{plugin_id}' ({plugin_name}) crashed: {e}"
                logger.error(f"    [{dataset_name}] ✗ {error_msg}")
                all_errors.append(error_msg)
                failed_plugins.append(plugin_id)

                # Fire callback for crashes too (consistency with regular failures)
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

                # Check critical failure threshold for crashes too
                critical_threshold = getattr(getattr(dataset_config, VALIDATIONS_ATTR, None), "critical_failures", 0)
                if 0 < critical_threshold <= len(failed_plugins):
                    critical_threshold_reached = True
                    logger.error(
                        f"    [{dataset_name}] Critical failure threshold reached: "
                        f"{len(failed_plugins)}/{critical_threshold} plugins failed"
                    )
                    logger.error(f"    [{dataset_name}] Stopping validation for this dataset")
                    break  # Stop validating this dataset

        # Log results
        logger.info(f"[{dataset_name}] Dataset Validation Metrics:")
        for key, value in all_metrics.items():
            logger.info(f"  - {key}: {value}")

        if all_warnings:
            logger.warning(f"[{dataset_name}] Validation Warnings:")
            for warning in all_warnings:
                logger.warning(f"  - {warning}")

        # If there are errors, fail this dataset's validation
        if all_errors:
            logger.error(f"[{dataset_name}] Dataset Validation Failed:")
            for error in all_errors:
                logger.error(f"  - {error}")

            # Show recommendations
            if all_recommendations:
                logger.info("")
                logger.info(f"[{dataset_name}] 💡 Recommendations:")
                for rec in all_recommendations:
                    logger.info(f"  - {rec}")

            # Fire callback: validation failed
            if self._callbacks.on_validation_failed:
                self._callbacks.on_validation_failed(dataset_name, dataset_path, all_errors)

            error_summary = f"{len(all_errors)} validation errors"
            error_code = (
                "DATASET_VALIDATION_CRITICAL_FAILURE" if critical_threshold_reached else "DATASET_VALIDATION_ERROR"
            )
            return Err(DatasetError(message=error_summary, code=error_code, details={"errors": all_errors}))

        # Success
        logger.info(f"[{dataset_name}] ✅ All validation checks passed!")

        # Fire callback: validation completed
        if self._callbacks.on_validation_completed:
            self._callbacks.on_validation_completed(dataset_name, dataset_path, all_metrics, all_warnings)

        # Return result data
        result_data = {
            VALIDATION_STATUS_KEY: VALIDATION_STATUS_PASSED,
            "warnings": all_warnings,
            **all_metrics,
        }

        return Ok(result_data)

    @staticmethod
    def _get_dataset_train_ref(dataset_config: Any) -> str:
        """Get a stable train reference string for logging/events."""
        try:
            if dataset_config.get_source_type() == SOURCE_TYPE_HUGGINGFACE and dataset_config.source_hf is not None:
                return dataset_config.source_hf.train_id
            if dataset_config.source_local is not None:
                return dataset_config.source_local.local_paths.train
        except Exception:
            pass
        return "unknown"

    def _try_load_eval_dataset_for_validation(
        self,
        dataset_config: Any,
        loader: Any,
    ) -> tuple[Dataset | IterableDataset | None, str | None]:
        """Load eval dataset if configured; returns (dataset, ref)."""
        try:
            if dataset_config.get_source_type() == SOURCE_TYPE_HUGGINGFACE:
                if dataset_config.source_hf is None or not dataset_config.source_hf.eval_id:
                    return None, None
                ds = self._load_dataset_for_validation(dataset_config, loader, split_name="eval")
                return ds, dataset_config.source_hf.eval_id

            # local
            if dataset_config.source_local is None or not dataset_config.source_local.local_paths.eval:
                return None, None
            ds = self._load_dataset_for_validation(dataset_config, loader, split_name="eval")
            return ds, dataset_config.source_local.local_paths.eval

        except Exception:
            return None, None
