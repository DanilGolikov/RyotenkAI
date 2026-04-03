"""
MLflowManager — facade for MLflow experiment tracking.

Owns only:
  - Run lifecycle: setup(), start_run(), end_run(), start_nested_run(), cleanup()
  - Logging primitives: log_params(), log_metrics(), log_artifact(), log_dict(),
    set_tags(), set_tag(), log_text(), log_summary_artifact()
  - State: _mlflow module, _run, _run_id, _parent_run_id, _nested_run_stack

All other responsibilities delegated to subcomponents:
  - MLflowEventLog   — event log (in-memory)
  - MLflowAutologManager — autolog and tracing
  - MLflowModelRegistry  — model registration and aliases
  - MLflowDatasetLogger  — dataset logging
  - MLflowDomainLogger   — domain-specific log_* helpers
  - MLflowRunAnalytics   — run search, comparison, summary

Usage:
    manager = MLflowManager(config)
    manager.setup()

    with manager.start_run(run_name="sft_v1"):
        manager.log_params({"lr": 2e-4})
        manager.log_stage_start("Training", 0, 3)
        ...
        manager.log_metrics({"loss": 0.42})
"""

from __future__ import annotations

import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from src.infrastructure.mlflow.gateway import IMLflowGateway, MLflowGateway, NullMLflowGateway
from src.training.constants import (
    MLFLOW_EXPERIMENT_DEFAULT_ID,
    MLFLOW_TRUNCATE_FEEDBACK,
    MLFLOW_TRUNCATE_PROMPT,
    MLFLOW_TRUNCATE_RESPONSE,
)
from src.training.mlflow.autolog import MLflowAutologManager
from src.training.mlflow.dataset_logger import MLflowDatasetLogger
from src.training.mlflow.domain_logger import MLflowDomainLogger
from src.training.mlflow.event_log import MLflowEventLog
from src.training.mlflow.model_registry import MLflowModelRegistry
from src.training.mlflow.run_analytics import MLflowRunAnalytics
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import Generator

    from src.utils.config import PipelineConfig

logger = get_logger(__name__)


class MLflowManager:
    """
    MLflow experiment tracking — facade coordinating all subcomponents.

    Implements IMLflowManager protocol (defined in src/utils/container.py).
    All subcomponents are created in setup() once the MLflow module is available.
    """

    # Severity map exposed as ClassVar for backwards-compat (tests may read it).
    _SEVERITY_MAP: ClassVar[dict[str, tuple[str, int]]] = {
        "start": ("INFO", 9),  # noqa: WPS226
        "complete": ("INFO", 9),  # noqa: WPS226
        "info": ("INFO", 9),  # noqa: WPS226
        "checkpoint": ("INFO", 9),  # noqa: WPS226
        "warning": ("WARN", 13),
        "error": ("ERROR", 17),
    }

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self._mlflow_config = config.experiment_tracking.mlflow
        self._mlflow: Any = None
        self._run: Any = None
        self._run_id: str | None = None
        self._parent_run_id: str | None = None
        self._nested_run_stack: list[str] = []

        # Gateway: real instance created in setup()
        self._gateway: IMLflowGateway = NullMLflowGateway()

        # Event log: created immediately (events may be logged before setup)
        self._event_log = MLflowEventLog()

        # DomainLogger created immediately (no mlflow dependency)
        self._domain_logger: MLflowDomainLogger = MLflowDomainLogger(
            self,  # type: ignore[arg-type]  # MLflowManager satisfies IMLflowPrimitives
            self._event_log,
        )

        # DatasetLogger: needs mlflow module, created immediately with None
        self._dataset_logger: MLflowDatasetLogger = MLflowDatasetLogger(
            mlflow_module=None,
            primitives=self,  # type: ignore[arg-type]
            has_active_run=lambda: self._run is not None,
        )

        # AutologManager: needs mlflow module, created immediately with None
        self._autolog: MLflowAutologManager = MLflowAutologManager(
            mlflow_module=None,
            tracking_uri=None,
        )

        # Registry: needs gateway + mlflow module — created in setup()
        self._registry: MLflowModelRegistry | None = None
        self._analytics: MLflowRunAnalytics = MLflowRunAnalytics(
            self._gateway,
            None,  # mlflow module — assigned in setup()
            experiment_name=None,
            event_log=self._event_log,
        )

    # =========================================================================
    # PROPERTIES
    # =========================================================================

    @property
    def is_active(self) -> bool:
        """True only if setup() succeeded and mlflow module is loaded."""
        return self._mlflow is not None

    @property
    def client(self) -> Any:
        if self._mlflow is None:
            return None
        return self._gateway.get_client()

    @property
    def run_id(self) -> str | None:
        return self._run_id

    @property
    def parent_run_id(self) -> str | None:
        return self._parent_run_id

    @property
    def is_nested(self) -> bool:
        return bool(self._nested_run_stack)

    def _get_active_run_id(self, run_id: str | None = None) -> str | None:
        return run_id or self._run_id

    # =========================================================================
    # SETUP
    # =========================================================================

    def setup(
        self,
        timeout: float = 5.0,
        max_retries: int = 3,
        disable_system_metrics: bool = False,
    ) -> bool:
        """
        Initialize MLflow connection with retries.

        Returns:
            True if setup successful
        """
        try:
            import mlflow

            self._mlflow = mlflow

            self._gateway = MLflowGateway(self._mlflow_config.tracking_uri)
            tracking_uri = self._gateway.uri
            mlflow.set_tracking_uri(tracking_uri)
            logger.info(f"MLflow tracking URI: {tracking_uri}")

            if tracking_uri and tracking_uri.startswith("http"):
                connected = False
                for attempt in range(1, max_retries + 1):
                    if self._gateway.check_connectivity(timeout):
                        connected = True
                        break
                    logger.warning(f"MLflow connection attempt {attempt}/{max_retries} failed")

                if not connected:
                    error_msg = f"MLflow server not reachable at {tracking_uri} after {max_retries} attempts"
                    logger.error(f"[MLFLOW] {error_msg}")
                    self._event_log.log_event_error(
                        error_msg,
                        category="system",
                        source="MLflowManager",
                        error_type="ConnectionError",
                        severity="ERROR",
                    )
                    self._mlflow = None
                    self._gateway = NullMLflowGateway()
                    return False

            # Restore deleted experiments before setting active one
            try:
                client = self._gateway.get_client()
                try:
                    exp0 = client.get_experiment(MLFLOW_EXPERIMENT_DEFAULT_ID)
                    if getattr(exp0, "lifecycle_stage", None) == "deleted":
                        client.restore_experiment(MLFLOW_EXPERIMENT_DEFAULT_ID)
                        logger.warning("Restored deleted MLflow experiment: id=0 (Default)")
                except Exception:
                    pass
                try:
                    exp = client.get_experiment_by_name(self._mlflow_config.experiment_name)
                    if exp is not None and getattr(exp, "lifecycle_stage", None) == "deleted":
                        client.restore_experiment(exp.experiment_id)
                        logger.warning(f"Restored deleted MLflow experiment: {self._mlflow_config.experiment_name}")
                except Exception:
                    pass
            except Exception as e:
                logger.debug(f"MLflow experiment restore skipped: {e}")

            mlflow.set_experiment(self._mlflow_config.experiment_name)
            logger.info(f"MLflow experiment: {self._mlflow_config.experiment_name}")

            should_enable_metrics = not disable_system_metrics and self._mlflow_config.system_metrics_callback_enabled
            if should_enable_metrics:
                try:
                    mlflow.enable_system_metrics_logging()
                    interval = self._mlflow_config.system_metrics_sampling_interval
                    samples = self._mlflow_config.system_metrics_samples_before_logging
                    mlflow.set_system_metrics_sampling_interval(interval)
                    mlflow.set_system_metrics_samples_before_logging(samples)
                    logger.info(f"MLflow system metrics enabled (interval={interval}s, samples={samples})")
                except Exception as e:
                    logger.debug(f"System metrics logging not available: {e}")
            else:
                logger.debug("MLflow system metrics logging disabled for this process")

            # Build subcomponents now that mlflow module is available
            self._build_subcomponents()
            return True

        except ImportError:
            logger.warning("MLflow not installed.")
            self._mlflow = None
            return False
        except Exception as e:
            logger.warning(f"MLflow setup failed: {e}")
            self._mlflow = None
            return False

    def check_mlflow_connectivity(self, timeout: float = 5.0) -> bool:
        """Check whether the configured MLflow tracking backend is reachable."""
        tracking_uri = self._mlflow_config.tracking_uri
        if not tracking_uri.startswith("http"):
            return True
        return MLflowGateway(tracking_uri).check_connectivity(timeout)

    def _build_subcomponents(self) -> None:
        """Update mlflow-dependent subcomponents after successful setup."""
        assert self._mlflow_config is not None
        log_model = getattr(self._mlflow_config, "log_model", True)
        experiment_name = getattr(self._mlflow_config, "experiment_name", None)
        tracking_uri = self._gateway.uri

        # Update eagerly-initialized components with real mlflow module
        self._autolog._mlflow = self._mlflow
        self._autolog._tracking_uri = tracking_uri
        self._analytics._mlflow = self._mlflow
        self._analytics._experiment_name = experiment_name
        self._analytics._gateway = self._gateway

        # Registry always requires real mlflow — create fresh
        self._registry = MLflowModelRegistry(self._gateway, self._mlflow, log_model)
        # Update dataset_logger's mlflow module (was None before setup)
        self._dataset_logger = MLflowDatasetLogger(
            self._mlflow,
            self,  # type: ignore[arg-type]
            has_active_run=lambda: self._run is not None,
        )
        # DomainLogger stays the same (no mlflow dependency), just recreate for consistency
        self._domain_logger = MLflowDomainLogger(
            self,  # type: ignore[arg-type]
            self._event_log,
        )

    # =========================================================================
    # RUN LIFECYCLE
    # =========================================================================

    def _load_description_file(self) -> str | None:
        """Load run description from file (custom or default template)."""
        default_template = Path(__file__).parent.parent / "templates" / "experiment_description.md"
        custom_path = None
        if self._mlflow_config and self._mlflow_config.run_description_file:
            custom_path = Path(self._mlflow_config.run_description_file)

        if custom_path and custom_path.exists():
            logger.debug(f"[MLFLOW:DESC] Loaded from {custom_path}")
            return custom_path.read_text(encoding="utf-8")

        if default_template.exists():
            return default_template.read_text(encoding="utf-8")

        return None

    @contextmanager
    def start_run(self, run_name: str | None = None, description: str | None = None) -> Generator[Any, None, None]:
        """Start a parent MLflow run (context manager)."""
        if self._mlflow is None:
            yield None
            return

        if description is None:
            description = self._load_description_file()

        try:
            with self._mlflow.start_run(run_name=run_name, description=description, log_system_metrics=True) as run:
                self._run = run
                self._run_id = run.info.run_id
                self._parent_run_id = run.info.run_id
                logger.info(f"[MLFLOW] Run started: {run.info.run_id}")
                yield run
                logger.info(f"[MLFLOW] Run completed: {run.info.run_id}")
        except Exception as e:
            logger.warning(f"[MLFLOW] Run error: {e}")
            yield None
        finally:
            self._run = None
            self._parent_run_id = None
            self._nested_run_stack.clear()

    def end_run(self, status: str = "FINISHED") -> None:
        """Explicitly end current run with status."""
        if self._mlflow is None:
            return
        try:
            self._mlflow.end_run(status=status)
            logger.info(f"[MLFLOW] Run ended: {status}")
        except Exception as e:
            logger.warning(f"[MLFLOW] Failed to end run: {e}")

    @contextmanager
    def start_nested_run(
        self,
        run_name: str,
        tags: dict[str, str] | None = None,
        description: str | None = None,
    ) -> Generator[Any, None, None]:
        """Start a nested (child) run within current parent run."""
        if self._mlflow is None:
            yield None
            return

        if self._parent_run_id is None:
            logger.warning("[MLFLOW] start_nested_run called without parent run, starting as regular run")
            with self.start_run(run_name=run_name, description=description) as run:
                if tags:
                    self.set_tags(tags)
                yield run
            return

        try:
            with self._mlflow.start_run(
                run_name=run_name,
                nested=True,
                description=description,
                log_system_metrics=True,
            ) as nested_run:
                self._run = nested_run
                self._run_id = nested_run.info.run_id
                self._nested_run_stack.append(nested_run.info.run_id)

                if tags:
                    self._mlflow.set_tags(tags)

                self._mlflow.set_tags(
                    {
                        "mlflow.parentRunId": self._parent_run_id,
                        "nested_run_depth": str(len(self._nested_run_stack)),
                    }
                )

                logger.info(f"[MLFLOW] Nested run started: {run_name} (parent: {self._parent_run_id[:8]}...)")
                yield nested_run
                logger.info(f"[MLFLOW] Nested run completed: {run_name}")

        except Exception as e:
            logger.warning(f"[MLFLOW] Nested run error: {e}")
            yield None
        finally:
            if self._nested_run_stack:
                self._nested_run_stack.pop()
            self._run_id = self._nested_run_stack[-1] if self._nested_run_stack else self._parent_run_id

    # =========================================================================
    # PRIMITIVES (log_params, log_metrics, log_artifact, log_dict, set_tags)
    # =========================================================================

    def log_params(self, params: dict[str, Any]) -> None:
        """Log parameters to current run."""
        if self._mlflow is None or self._run is None:
            return
        try:
            clean_params = {k: str(v) if v is not None else "None" for k, v in params.items()}
            self._mlflow.log_params(clean_params)
            logger.debug(f"[MLFLOW:PARAMS] {len(clean_params)} params logged")
        except Exception as e:
            logger.warning(f"[MLFLOW] log_params failed: {e}")

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log metrics to current run."""
        if self._mlflow is None or self._run is None:
            return
        try:
            for key, value in metrics.items():
                if value is not None:
                    self._mlflow.log_metric(key, float(value), step=step)
            logger.debug(f"[MLFLOW:METRICS] {metrics}")
        except Exception as e:
            logger.warning(f"[MLFLOW] log_metrics failed: {e}")

    def log_artifact(self, local_path: str, artifact_path: str | None = None, run_id: str | None = None) -> bool:
        """Log artifact file to MLflow run via HTTP API."""
        if not self._mlflow_config or not self._mlflow_config.log_artifacts:
            return False

        target_run_id = self._get_active_run_id(run_id)
        if not target_run_id or self.client is None:
            return False

        path = Path(local_path)
        if not path.exists():
            logger.warning(f"[MLFLOW] Artifact not found: {local_path}")
            return False

        try:
            content = path.read_text(encoding="utf-8")
            artifact_name = path.name
            if artifact_path:
                artifact_name = f"{artifact_path}/{artifact_name}"
            self.client.log_text(target_run_id, content, artifact_name)
            logger.debug(f"[MLFLOW:ARTIFACT] {local_path} -> {artifact_name}")
            return True
        except UnicodeDecodeError:
            logger.debug(f"[MLFLOW:ARTIFACT] Skipping binary file: {local_path}")
            return False
        except Exception as e:
            logger.warning(f"[MLFLOW] log_artifact failed: {e}")
            return False

    def log_dict(self, dictionary: dict[str, Any], artifact_file: str, run_id: str | None = None) -> bool:
        """Log dict as JSON artifact to specific run."""
        target_run_id = self._get_active_run_id(run_id)
        if not target_run_id:
            logger.debug(f"[MLFLOW:DICT] Skipped {artifact_file} - no active run")
            return False

        try:
            self.client.log_dict(target_run_id, dictionary, artifact_file)
            logger.debug(f"[MLFLOW:DICT] {artifact_file} -> run_id={target_run_id[:8]}...")
            return True
        except Exception as e:
            logger.error(f"[MLFLOW:DICT] Failed to log {artifact_file}: {e}")
            logger.error(f"[MLFLOW:DICT] Traceback:\n{traceback.format_exc()}")
            return False

    def log_text(self, text: str, artifact_file: str, run_id: str | None = None) -> bool:
        """Log text content as artifact to specific run."""
        target_run_id = self._get_active_run_id(run_id)
        if not target_run_id:
            return False
        try:
            self.client.log_text(target_run_id, text, artifact_file)
            logger.debug(f"[MLFLOW:TEXT] {artifact_file} -> run_id={target_run_id[:8]}...")
            return True
        except Exception as e:
            logger.error(f"[MLFLOW:TEXT] Failed: {e}")
            return False

    def set_tags(self, tags: dict[str, str]) -> None:
        """Set tags on current run."""
        if self._mlflow is None or self._run is None:
            return
        try:
            self._mlflow.set_tags(tags)
        except Exception as e:
            logger.warning(f"[MLFLOW] set_tags failed: {e}")

    def set_tag(self, key: str, value: str) -> None:
        """Set a single tag on current run."""
        self.set_tags({key: value})

    # =========================================================================
    # EVENT LOG — delegate to MLflowEventLog
    # =========================================================================

    def log_event(
        self,
        event_type: str,
        message: str,
        *,
        category: str = "training",  # noqa: WPS226
        source: str = "",
        **metadata: Any,
    ) -> dict[str, Any]:
        return self._event_log.log_event(event_type, message, category=category, source=source, **metadata)

    def log_event_start(self, message: str, **kwargs: Any) -> dict[str, Any]:
        return self._event_log.log_event_start(message, **kwargs)

    def log_event_complete(self, message: str, **kwargs: Any) -> dict[str, Any]:
        return self._event_log.log_event_complete(message, **kwargs)

    def log_event_error(self, message: str, **kwargs: Any) -> dict[str, Any]:
        return self._event_log.log_event_error(message, **kwargs)

    def log_event_warning(self, message: str, **kwargs: Any) -> dict[str, Any]:
        return self._event_log.log_event_warning(message, **kwargs)

    def log_event_info(self, message: str, **kwargs: Any) -> dict[str, Any]:
        return self._event_log.log_event_info(message, **kwargs)

    def log_event_checkpoint(self, message: str, **kwargs: Any) -> dict[str, Any]:
        return self._event_log.log_event_checkpoint(message, **kwargs)

    def get_events(self, category: str | None = None) -> list[dict[str, Any]]:
        return self._event_log.get_events(category)

    def log_events_artifact(self, artifact_name: str = "training_events.json", run_id: str | None = None) -> bool:
        return self._event_log.log_events_artifact(artifact_name, log_dict_fn=self.log_dict, run_id=run_id)

    def log_summary_artifact(
        self,
        events_artifact_name: str = "training_events.json",
        parent_run_id: str | None = None,
    ) -> bool:
        """Generate and log summary as MLflow artifact."""
        target_run_id = self._get_active_run_id(parent_run_id)
        try:
            events_ok = self.log_events_artifact(events_artifact_name, run_id=target_run_id)
            if events_ok:
                logger.info("[MLFLOW:SUMMARY] Events artifact logged")
                return True
            logger.warning("[MLFLOW:SUMMARY] No artifacts were logged")
            return False
        except Exception as e:
            logger.warning(f"[MLFLOW:SUMMARY] Failed to log summary: {e}")
            return False

    # =========================================================================
    # AUTOLOG / TRACING — delegate to MLflowAutologManager
    # =========================================================================

    def enable_autolog(
        self,
        log_models: bool = False,
        log_input_examples: bool = False,
        log_model_signatures: bool = True,
        log_every_n_steps: int | None = None,
        disable_for_unsupported_versions: bool = True,
        silent: bool = False,
    ) -> bool:
        self._autolog._mlflow = self._mlflow
        return self._autolog.enable_autolog(
            log_models=log_models,
            log_input_examples=log_input_examples,
            log_model_signatures=log_model_signatures,
            log_every_n_steps=log_every_n_steps,
            disable_for_unsupported_versions=disable_for_unsupported_versions,
            silent=silent,
        )

    def disable_autolog(self) -> bool:
        self._autolog._mlflow = self._mlflow
        return self._autolog.disable_autolog()

    def enable_pytorch_autolog(
        self,
        log_models: bool = False,
        log_every_n_epoch: int = 1,
        log_every_n_step: int | None = None,
    ) -> bool:
        self._autolog._mlflow = self._mlflow
        return self._autolog.enable_pytorch_autolog(
            log_models=log_models,
            log_every_n_epoch=log_every_n_epoch,
            log_every_n_step=log_every_n_step,
        )

    def enable_tracing(self) -> bool:
        self._autolog._mlflow = self._mlflow
        return self._autolog.enable_tracing()

    def disable_tracing(self) -> bool:
        self._autolog._mlflow = self._mlflow
        return self._autolog.disable_tracing()

    def trace_llm_call(
        self,
        name: str,
        model_name: str | None = None,
        span_type: str = "LLM",
        attributes: dict[str, Any] | None = None,
    ) -> Any:
        self._autolog._mlflow = self._mlflow
        return self._autolog.trace_llm_call(name, model_name=model_name, span_type=span_type, attributes=attributes)

    def log_trace_io(
        self,
        input_data: str | dict[str, Any] | None = None,
        output_data: str | dict[str, Any] | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        latency_ms: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if self._mlflow is None:
            return
        self._autolog._mlflow = self._mlflow
        self._autolog.log_trace_io(
            input_data=input_data,
            output_data=output_data,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            metadata=metadata,
        )

    def create_trace_decorator(self, name: str | None = None, span_type: str = "LLM") -> Any:
        self._autolog._mlflow = self._mlflow
        return self._autolog.create_trace_decorator(name=name, span_type=span_type)

    def get_trace_url(self, trace_id: str | None = None) -> str | None:
        self._autolog._mlflow = self._mlflow
        # Use gateway URI or fall back to config URI (when setup() hasn't run yet)
        tracking_uri = self._gateway.uri or (self._mlflow_config.tracking_uri if self._mlflow_config else None)
        self._autolog._tracking_uri = tracking_uri or None
        return self._autolog.get_trace_url(trace_id=trace_id)

    # =========================================================================
    # MODEL REGISTRY — delegate to MLflowModelRegistry
    # =========================================================================

    def register_model(
        self,
        model_name: str,
        model_uri: str | None = None,
        alias: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> str | None:
        if self._registry is None:
            return None
        return self._registry.register_model(
            model_name, run_id=self._run_id or "", model_uri=model_uri, alias=alias, tags=tags
        )

    def set_model_alias(self, model_name: str, alias: str, version: int | str) -> bool:
        if self._registry is None:
            return False
        return self._registry.set_model_alias(model_name, alias, version)

    def get_model_by_alias(self, model_name: str, alias: str) -> dict[str, Any] | None:
        if self._registry is None:
            return None
        return self._registry.get_model_by_alias(model_name, alias)

    def delete_model_alias(self, model_name: str, alias: str) -> bool:
        if self._registry is None:
            return False
        return self._registry.delete_model_alias(model_name, alias)

    def promote_model(
        self,
        model_name: str,
        from_alias: str = "staging",
        to_alias: str = "champion",
    ) -> bool:
        if self._registry is None:
            return False
        return self._registry.promote_model(model_name, from_alias=from_alias, to_alias=to_alias)

    def get_model_aliases(self, model_name: str) -> dict[str, int]:
        if self._registry is None:
            return {}
        return self._registry.get_model_aliases(model_name)

    def load_model_by_alias(self, model_name: str, alias: str = "champion") -> Any:
        if self._registry is None:
            return None
        return self._registry.load_model_by_alias(model_name, alias)

    # =========================================================================
    # DATASET LOGGING — delegate to MLflowDatasetLogger
    # =========================================================================

    def log_dataset(
        self,
        data: Any,
        name: str,
        source: str | None = None,
        context: str = "training",
        targets: str | None = None,
        predictions: str | None = None,
    ) -> bool:
        self._dataset_logger._mlflow = self._mlflow
        return self._dataset_logger.log_dataset(
            data, name, source=source, context=context, targets=targets, predictions=predictions
        )

    def log_dataset_from_file(self, file_path: str, name: str | None = None, context: str = "training") -> bool:
        self._dataset_logger._mlflow = self._mlflow
        return self._dataset_logger.log_dataset_from_file(file_path, name=name, context=context)

    def log_dataset_info(
        self,
        name: str,
        path: str | None = None,
        source: str | None = None,
        version: str | None = None,
        num_rows: int = 0,
        num_samples: int | None = None,
        num_features: int | None = None,
        context: str = "training",
        extra_info: dict[str, Any] | None = None,
        extra_tags: dict[str, str] | None = None,
    ) -> None:
        self._dataset_logger._mlflow = self._mlflow
        self._dataset_logger.log_dataset_info(
            name,
            path=path,
            source=source,
            version=version,
            num_rows=num_rows,
            num_samples=num_samples,
            num_features=num_features,
            context=context,
            extra_info=extra_info,
            extra_tags=extra_tags,
        )

    def create_mlflow_dataset(self, data: Any, name: str, source: str, targets: str | None = None) -> Any:
        self._dataset_logger._mlflow = self._mlflow
        return self._dataset_logger.create_mlflow_dataset(data, name, source, targets=targets)

    def log_dataset_input(self, dataset: Any, context: str = "training") -> bool:
        self._dataset_logger._mlflow = self._mlflow
        return self._dataset_logger.log_dataset_input(dataset, context=context)

    # =========================================================================
    # DOMAIN LOGGING — delegate to MLflowDomainLogger
    # =========================================================================

    def log_training_config(self, config: PipelineConfig) -> None:
        self._domain_logger.log_training_config(config)

    def log_pipeline_config(self, config: PipelineConfig) -> None:
        self._domain_logger.log_pipeline_config(config)

    def log_dataset_config(self, config: PipelineConfig) -> None:
        self._domain_logger.log_dataset_config(config)

    def log_provider_info(
        self,
        provider_name: str,
        provider_type: str,
        gpu_type: str | None = None,
        resource_id: str | None = None,
    ) -> None:
        self._domain_logger.log_provider_info(provider_name, provider_type, gpu_type=gpu_type, resource_id=resource_id)

    def log_strategy_info(self, strategy_type: str, phase_idx: int, total_phases: int) -> None:
        self._domain_logger.log_strategy_info(strategy_type, phase_idx, total_phases)

    def log_gpu_metrics(
        self,
        gpu_memory_used_gb: float,
        gpu_memory_total_gb: float,
        gpu_utilization: float | None = None,
        step: int | None = None,
    ) -> None:
        self._domain_logger.log_gpu_metrics(
            gpu_memory_used_gb, gpu_memory_total_gb, gpu_utilization=gpu_utilization, step=step
        )

    def log_throughput(self, tokens_per_second: float, samples_per_second: float, step: int | None = None) -> None:
        self._domain_logger.log_throughput(tokens_per_second, samples_per_second, step=step)

    def log_gpu_detection(self, name: str, vram_gb: float, tier: str) -> None:
        self._domain_logger.log_gpu_detection(name, vram_gb, tier)

    def log_memory_warning(
        self,
        utilization_percent: float,
        used_mb: int,
        total_mb: int,
        is_critical: bool,
    ) -> None:
        self._domain_logger.log_memory_warning(utilization_percent, used_mb, total_mb, is_critical)

    def log_oom(self, operation: str, free_mb: int | None = None) -> None:
        self._domain_logger.log_oom(operation, free_mb=free_mb)

    def log_oom_recovery(self, operation: str, attempt: int, max_attempts: int) -> None:
        self._domain_logger.log_oom_recovery(operation, attempt, max_attempts)

    def log_cache_cleared(self, freed_mb: int) -> None:
        self._domain_logger.log_cache_cleared(freed_mb)

    def log_memory_snapshot(
        self,
        phase: str,
        used_mb: int,
        free_mb: int,
        total_mb: int,
        utilization_percent: float,
    ) -> None:
        self._domain_logger.log_memory_snapshot(phase, used_mb, free_mb, total_mb, utilization_percent)

    def log_pipeline_initialized(self, run_id: str, total_phases: int, strategy_chain: list[str]) -> None:
        self._domain_logger.log_pipeline_initialized(run_id, total_phases, strategy_chain)

    def log_state_saved(self, run_id: str, path: str) -> None:
        self._domain_logger.log_state_saved(run_id, path)

    def log_checkpoint_cleanup(self, cleaned_count: int, freed_mb: int) -> None:
        self._domain_logger.log_checkpoint_cleanup(cleaned_count, freed_mb)

    def log_stage_start(self, stage_name: str, stage_idx: int, total_stages: int) -> None:
        self._domain_logger.log_stage_start(stage_name, stage_idx, total_stages)

    def log_stage_complete(self, stage_name: str, stage_idx: int, duration_seconds: float | None = None) -> None:
        self._domain_logger.log_stage_complete(stage_name, stage_idx, duration_seconds=duration_seconds)

    def log_stage_failed(self, stage_name: str, stage_idx: int, error: str) -> None:
        self._domain_logger.log_stage_failed(stage_name, stage_idx, error)

    def log_environment(self, env_snapshot: dict[str, Any] | None = None) -> None:
        self._domain_logger.log_environment(env_snapshot)

    # =========================================================================
    # RUN ANALYTICS — delegate to MLflowRunAnalytics
    # =========================================================================

    def get_child_runs(self, parent_run_id: str | None = None) -> list[dict[str, Any]]:
        parent_id = parent_run_id or self._parent_run_id
        if not parent_id:
            return []
        self._analytics._mlflow = self._mlflow
        self._analytics._gateway = self._gateway
        return self._analytics.get_child_runs(parent_id)

    def get_best_run(
        self,
        metric: str = "eval_loss",
        mode: str = "min",
        experiment_name: str | None = None,
        filter_string: str | None = None,
    ) -> dict[str, Any] | None:
        self._analytics._mlflow = self._mlflow
        return self._analytics.get_best_run(
            metric=metric, mode=mode, experiment_name=experiment_name, filter_string=filter_string
        )

    def compare_runs(
        self,
        run_ids: list[str],
        metrics: list[str] | None = None,
        params: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        self._analytics._mlflow = self._mlflow
        self._analytics._gateway = self._gateway
        return self._analytics.compare_runs(run_ids, metrics=metrics, params=params)

    def search_runs(
        self,
        filter_string: str | None = None,
        experiment_name: str | None = None,
        order_by: list[str] | None = None,
        max_results: int = 100,
    ) -> list[dict[str, Any]]:
        self._analytics._mlflow = self._mlflow
        return self._analytics.search_runs(
            filter_string=filter_string,
            experiment_name=experiment_name,
            order_by=order_by,
            max_results=max_results,
        )

    def get_run_metrics_history(self, run_id: str, metric: str) -> list[dict[str, Any]]:
        self._analytics._mlflow = self._mlflow
        self._analytics._gateway = self._gateway
        return self._analytics.get_run_metrics_history(run_id, metric)

    def get_experiment_summary(self, experiment_name: str | None = None) -> dict[str, Any]:
        self._analytics._mlflow = self._mlflow
        return self._analytics.get_experiment_summary(experiment_name=experiment_name)

    def generate_summary_markdown(self) -> str:
        return self._analytics.generate_summary_markdown(run_id=self._run_id)

    def _get_run_data(self) -> dict[str, Any] | None:
        if self._run_id is None:
            return None
        self._analytics._mlflow = self._mlflow
        self._analytics._gateway = self._gateway
        return self._analytics.get_run_data(self._run_id)

    # =========================================================================
    # LLM EVALUATION
    # =========================================================================

    def log_llm_evaluation(
        self,
        prompt: str,
        response: str,
        expected: str | None = None,
        score: float | None = None,
        feedback: str | None = None,
        evaluator: str = "human",
    ) -> None:
        """Log LLM evaluation result."""
        if self._mlflow is None:
            return

        try:
            evaluation_data: dict[str, Any] = {
                "prompt": prompt[:MLFLOW_TRUNCATE_PROMPT],
                "response": response[:MLFLOW_TRUNCATE_RESPONSE],
                "evaluator": evaluator,
            }
            if expected:
                evaluation_data["expected"] = expected[:MLFLOW_TRUNCATE_RESPONSE]
            if feedback:
                evaluation_data["feedback"] = feedback[:MLFLOW_TRUNCATE_FEEDBACK]

            artifact_name = f"evaluation_{evaluator}_{self._run_id[:8] if self._run_id else 'unknown'}.json"
            self.log_dict(evaluation_data, artifact_name)

            if score is not None:
                self._mlflow.log_metric(f"eval_score_{evaluator}", score)

            logger.debug(f"[MLFLOW:EVAL] evaluator={evaluator}, score={score}")

        except Exception as e:
            logger.debug(f"[MLFLOW:EVAL] Failed to log evaluation: {e}")

    # =========================================================================
    # CLEANUP
    # =========================================================================

    def cleanup(self) -> None:
        """Reset all runtime state."""
        self._run = None
        self._run_id = None
        self._mlflow = None
        self._event_log.clear()
        # Reset mlflow-dependent subcomponents (rebuilt on next setup())
        self._autolog._mlflow = None
        self._autolog._tracking_uri = None
        self._registry = None
        self._analytics._mlflow = None
        self._analytics._experiment_name = None
        # Rebuild dataset_logger without mlflow module (domain_logger is mlflow-free)
        from src.training.mlflow.dataset_logger import MLflowDatasetLogger

        self._dataset_logger = MLflowDatasetLogger(
            mlflow_module=None,
            primitives=self,  # type: ignore[arg-type]
            has_active_run=lambda: self._run is not None,
        )


def get_mlflow_manager(config: PipelineConfig) -> MLflowManager:
    """Factory function for MLflowManager."""
    return MLflowManager(config)


__all__ = [
    "MLflowManager",
    "get_mlflow_manager",
]
