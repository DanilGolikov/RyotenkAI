"""
MLflowManager — facade for MLflow experiment tracking.

Owns only:
  - State: _mlflow, _run, _run_id, _parent_run_id, _nested_run_stack
  - __init__: creates all subcomponents
  - Properties: is_active, client, run_id, parent_run_id, is_nested
  - Cleanup: cleanup()
  - Delegation methods to all subcomponents

Behavior is implemented by mixins:
  - MLflowSetupMixin       (setup.py)     — setup(), connectivity, _build_subcomponents()
  - MLflowRunLifecycleMixin (run_lifecycle.py) — start_run(), end_run(), start_nested_run()
  - MLflowLoggingMixin     (logging_core.py)  — log_params(), log_metrics(), log_artifact(), etc.

Subcomponents (src/training/mlflow/):
  - MLflowAutologManager  — autolog and tracing
  - MLflowModelRegistry   — model registration and aliases
  - MLflowDatasetLogger   — dataset logging
  - MLflowDomainLogger    — domain-specific log_* helpers
  - MLflowRunAnalytics    — run search, comparison, summary

Phase 7 retired the legacy ``MLflowEventLog`` subcomponent and the
matching ``log_event_*`` / ``log_events_artifact`` delegations — the
typed event journal (``events.jsonl``) is the single source of truth.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ryotenkai_pod.trainer.managers.mlflow_manager.logging_core import MLflowLoggingMixin
from ryotenkai_pod.trainer.managers.mlflow_manager.run_lifecycle import MLflowRunLifecycleMixin
from ryotenkai_pod.trainer.managers.mlflow_manager.setup import MLflowSetupMixin
from ryotenkai_pod.trainer.mlflow.autolog import MLflowAutologManager
from ryotenkai_pod.trainer.mlflow.dataset_logger import MLflowDatasetLogger
from ryotenkai_pod.trainer.mlflow.domain_logger import MLflowDomainLogger
from ryotenkai_pod.trainer.mlflow.resilient_transport import ResilientMLflowTransport
from ryotenkai_pod.trainer.mlflow.run_analytics import MLflowRunAnalytics
from ryotenkai_shared.infrastructure.mlflow.gateway import IMLflowGateway, NullMLflowGateway
from ryotenkai_shared.utils.logger import get_logger

if TYPE_CHECKING:
    from ryotenkai_pod.trainer.mlflow.model_registry import MLflowModelRegistry
    from ryotenkai_shared.config import PipelineConfig
    from ryotenkai_shared.infrastructure.mlflow.environment import MLflowEnvironment
    from ryotenkai_shared.infrastructure.mlflow.uri_resolver import MLflowRuntimeRole, ResolvedMLflowUris

logger = get_logger(__name__)


class MLflowManager(MLflowSetupMixin, MLflowRunLifecycleMixin, MLflowLoggingMixin):
    """
    MLflow experiment tracking — facade coordinating all subcomponents.

    Implements IMLflowManager protocol (defined in src/utils/container.py).
    All subcomponents are created in setup() once the MLflow module is available.
    """

    def __init__(self, config: PipelineConfig, *, runtime_role: MLflowRuntimeRole = "control_plane") -> None:
        self.config = config
        self._mlflow_config = config.integrations.mlflow
        self._runtime_role: MLflowRuntimeRole = runtime_role
        # Mixin parent declares these as non-None for the post-setup state;
        # concrete instances start in the pre-setup state, so we widen here.
        self._resolved_uris: ResolvedMLflowUris | None = None  # type: ignore[assignment]
        self._environment: MLflowEnvironment | None = None  # type: ignore[assignment]
        self._mlflow: Any = None
        self._run: Any = None
        self._run_id: str | None = None
        self._parent_run_id: str | None = None
        self._nested_run_stack: list[str] = []

        # Mixin declares ``MLflowGateway`` (concrete class); we use the
        # interface for null-object pattern before setup completes.
        self._gateway: IMLflowGateway = NullMLflowGateway()  # type: ignore[assignment]
        # Phase 7: MLflowEventLog retired; the typed event journal is
        # the SSOT. Domain/run-analytics subcomponents no longer carry
        # an event_log dependency.
        self._domain_logger: MLflowDomainLogger = MLflowDomainLogger(self)  # type: ignore[arg-type]
        self._dataset_logger: MLflowDatasetLogger = self._make_dataset_logger(mlflow_module=None)
        self._autolog: MLflowAutologManager = MLflowAutologManager(
            mlflow_module=None,
            tracking_uri=None,
        )
        self._resilient_transport = ResilientMLflowTransport()
        self._registry: MLflowModelRegistry | None = None  # type: ignore[assignment]
        self._analytics: MLflowRunAnalytics = MLflowRunAnalytics(
            self._gateway,
            None,
            experiment_name=None,
        )

    # =========================================================================
    # FACTORY HELPERS
    # =========================================================================

    def _make_dataset_logger(self, *, mlflow_module: Any = None) -> MLflowDatasetLogger:
        """Create an MLflowDatasetLogger bound to this manager.

        Centralises construction so that __init__, _build_subcomponents, and
        cleanup all use the same parameter shape.  A change to the
        MLflowDatasetLogger constructor signature only needs to be updated here.
        """
        return MLflowDatasetLogger(
            mlflow_module=mlflow_module,
            primitives=self,  # type: ignore[arg-type]
            has_active_run=lambda: self._run is not None,
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

    @property
    def tracking_uri(self) -> str | None:
        """Public accessor for the configured tracking URI.

        Returns ``None`` when setup() has not completed; otherwise the gateway URI.
        Prefer this over reaching into ``_gateway.uri`` directly.
        """
        if self._mlflow is None:
            return None
        return self._gateway.uri

    # =========================================================================
    # PHASE 9.B — explicit drain + terminate-by-status helpers
    # =========================================================================
    #
    # Both helpers proxy to the underlying ``ResilientMLflowTransport`` /
    # MLflow client so the cancellation callback (and any future
    # finalization layer) doesn't have to know about the resilient
    # transport's existence. Keeps the manager's surface narrow.

    def flush_buffer(self) -> int:
        """Drain the resilient transport's buffered metrics now.

        Phase 9.B explicit-drain entry point used by
        :class:`~src.training.callbacks.cancellation_callback.CancellationCallback`
        on ``on_train_end`` (wrapped in a 5-second hard budget — see
        :func:`~src.training._concurrent_helpers.with_timeout`).

        Returns the number of records drained. ``0`` when the manager
        is not active or no records were pending. Best-effort by
        contract — failures in the underlying drain are logged inside
        the transport, not raised.
        """
        if self._mlflow is None:
            return 0
        return self._resilient_transport.flush_buffer()

    def set_run_terminated(
        self,
        run_id: str,
        status: str = "KILLED",
    ) -> bool:
        """Force-set a run's terminal status via the MLflow client.

        Used by Mac-side reconciliation in Phase 9.C: when the runner
        wrote a ``cancelled.marker`` file (its 5-second flush budget
        ran out and the process was SIGKILLed before ``end_run``
        committed) but the MLflow UI still shows ``RunStatus.RUNNING``,
        the orchestrator calls this from Mac to bring the upstream in
        sync.

        Wraps :py:meth:`mlflow.tracking.MlflowClient.set_terminated`,
        which takes a string status from the canonical RunStatus enum
        (``FINISHED``, ``FAILED``, ``KILLED``, ``SCHEDULED``).
        Best-effort: returns ``False`` when the manager isn't active
        or the upstream call raises; failure is logged but never
        propagates so reconciliation can be safely fire-and-forget.

        Args:
            run_id: The MLflow run id to terminate.
            status: One of MLflow's RunStatus strings. Defaults to
                ``"KILLED"`` which is the canonical "stopped by user"
                status (Phase 9.1.C — single source of truth).

        Returns:
            ``True`` when the upstream call returned without error,
            ``False`` otherwise (manager inactive, MLflow unreachable,
            run_id unknown to the tracking server, etc.).
        """
        if self._mlflow is None:
            return False
        client = self.client
        if client is None:
            return False
        try:
            client.set_terminated(run_id=run_id, status=status)
            return True
        except Exception as exc:
            logger.warning(
                "[MLFLOW] set_run_terminated(run_id=%s, status=%s) failed: %s",
                run_id,
                status,
                exc,
            )
            return False

    def adopt_existing_run(self, run_id: str) -> Any:
        """Reopen an existing MLflow run_id as the active root run on this manager.

        Used by the orchestrator to continue writing to a parent run that was
        opened in a previous attempt (e.g. on resume/restart). Replaces the
        former private-attribute mutation in MLflowAttemptManager.

        Returns the mlflow Run object, or ``None`` if mlflow isn't available.
        """
        if self._mlflow is None:
            return None
        run = self._mlflow.start_run(run_id=run_id, nested=False, log_system_metrics=False)
        self._run = run
        self._run_id = run_id
        self._parent_run_id = run_id
        return run

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
        tracking_uri = self._gateway.uri or self.get_runtime_tracking_uri()
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

    def delete_run_tree(self, root_run_id: str) -> list[str]:
        """Soft-delete a root run and all descendants in child-first order."""
        client = self.client
        if client is None:
            raise RuntimeError("MLflow client is unavailable")
        if not root_run_id:
            return []

        root_run = client.get_run(root_run_id)
        experiment_id = root_run.info.experiment_id
        queue: list[tuple[str, int]] = [(root_run_id, 0)]
        visited: set[str] = set()
        run_ids_with_depth: list[tuple[str, int]] = []

        while queue:
            current_id, depth = queue.pop(0)
            if current_id in visited:
                continue
            visited.add(current_id)
            run_ids_with_depth.append((current_id, depth))
            children = client.search_runs(
                experiment_ids=[experiment_id],
                filter_string=f"tags.`mlflow.parentRunId` = '{current_id}'",
            )
            for child in children:
                child_id = child.info.run_id
                if child_id not in visited:
                    queue.append((child_id, depth + 1))

        ordered_run_ids = [
            run_id for run_id, _depth in sorted(run_ids_with_depth, key=lambda item: (item[1], item[0]), reverse=True)
        ]
        for run_id in ordered_run_ids:
            client.delete_run(run_id)
        return ordered_run_ids

    # =========================================================================
    # CLEANUP
    # =========================================================================

    def cleanup(self) -> None:
        """Reset all runtime state and restore process-wide MLflow env."""
        self._resilient_transport.uninstall()
        if self._environment is not None:
            self._environment.deactivate()
            self._environment = None
        self._run = None
        self._run_id = None
        self._mlflow = None
        self._autolog._mlflow = None
        self._autolog._tracking_uri = None
        self._registry = None
        self._analytics._mlflow = None
        self._analytics._experiment_name = None
        self._dataset_logger = self._make_dataset_logger(mlflow_module=None)


def get_mlflow_manager(config: PipelineConfig) -> MLflowManager:
    """Factory function for MLflowManager."""
    return MLflowManager(config)


__all__ = ["MLflowManager", "get_mlflow_manager"]
