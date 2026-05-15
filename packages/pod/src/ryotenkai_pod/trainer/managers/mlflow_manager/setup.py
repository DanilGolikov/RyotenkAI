"""
MLflowSetupMixin — MLflow initialization and connectivity logic.

Responsibilities:
  - setup(): import mlflow, connect, restore experiments, configure system metrics
  - check_mlflow_connectivity()
  - _build_subcomponents(): create/update all subcomponent instances after setup
  - _load_description_file(): read run description template
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ryotenkai_shared.errors import RyotenkAIError
from ryotenkai_shared.infrastructure.mlflow.environment import MLflowEnvironment
from ryotenkai_shared.infrastructure.mlflow.gateway import MLflowGateway, NullMLflowGateway
from ryotenkai_shared.infrastructure.mlflow.uri_resolver import resolve_mlflow_uris
from ryotenkai_pod.trainer.constants import MLFLOW_EXPERIMENT_DEFAULT_ID
from ryotenkai_pod.trainer.mlflow.domain_logger import MLflowDomainLogger
from ryotenkai_pod.trainer.mlflow.model_registry import MLflowModelRegistry
from ryotenkai_shared.utils.logger import get_logger

logger = get_logger(__name__)


class MLflowSetupMixin:
    """
    Mixin: MLflow setup, connectivity, and subcomponent initialization.

    Assumes the following attributes exist on self (set by MLflowManager.__init__):
      _mlflow_config, _mlflow, _gateway, _event_log, _autolog, _registry,
      _analytics, _dataset_logger, _domain_logger, _run
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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

            self._mlflow = mlflow  # type: ignore[attr-defined]

            self._resolved_uris = resolve_mlflow_uris(
                self._mlflow_config,  # type: ignore[attr-defined]
                runtime_role=self._runtime_role,  # type: ignore[attr-defined]
            )
            self._gateway = MLflowGateway(
                self._resolved_uris.runtime_tracking_uri,
                ca_bundle_path=self._mlflow_config.ca_bundle_path,  # type: ignore[attr-defined]
            )
            tracking_uri = self._gateway.uri  # type: ignore[attr-defined]
            self._environment = MLflowEnvironment(  # type: ignore[attr-defined]
                tracking_uri,
                ca_bundle_path=self._mlflow_config.ca_bundle_path,  # type: ignore[attr-defined]
            )
            self._environment.activate()  # type: ignore[attr-defined]
            logger.info(
                "MLflow tracking URI: %s (role=%s, raw=%s, local=%s, remote=%s)",
                tracking_uri,
                self._runtime_role,  # type: ignore[attr-defined]
                self._resolved_uris.tracking_uri,  # type: ignore[attr-defined]
                self._resolved_uris.effective_local_tracking_uri,  # type: ignore[attr-defined]
                self._resolved_uris.effective_remote_tracking_uri,  # type: ignore[attr-defined]
            )

            if tracking_uri and tracking_uri.startswith("http"):
                connected = False
                for attempt in range(1, max_retries + 1):
                    if self._gateway.check_connectivity(timeout):  # type: ignore[attr-defined]
                        connected = True
                        break
                    gateway_error = self._gateway.last_connectivity_error  # type: ignore[attr-defined]
                    if gateway_error is not None:
                        logger.warning(
                            "MLflow connection attempt %s/%s failed: %s",
                            attempt,
                            max_retries,
                            gateway_error,
                        )
                    else:
                        logger.warning(f"MLflow connection attempt {attempt}/{max_retries} failed")

                if not connected:
                    gateway_error = self._gateway.last_connectivity_error  # type: ignore[attr-defined]
                    error_msg = f"MLflow server not reachable at {tracking_uri} after {max_retries} attempts"
                    if gateway_error is not None:
                        error_msg = f"{error_msg}: {gateway_error}"
                    logger.error(f"[MLFLOW] {error_msg}")
                    self._event_log.log_event_error(  # type: ignore[attr-defined]
                        error_msg,
                        category="system",
                        source="MLflowManager",
                        error_type="ConnectionError",
                        severity="ERROR",
                    )
                    self._mlflow = None  # type: ignore[attr-defined]
                    self._gateway = NullMLflowGateway()  # type: ignore[assignment]
                    return False

            self._restore_deleted_experiments()
            mlflow.set_experiment(self._mlflow_config.experiment_name)  # type: ignore[attr-defined]
            logger.info(f"MLflow experiment: {self._mlflow_config.experiment_name}")  # type: ignore[attr-defined]

            # Native MLflow background sampler (``enable_system_metrics_logging``)
            # is intentionally NOT used: it bypasses our
            # ``ResilientMLflowTransport`` and silently drops samples on
            # offline windows. ``SystemMetricsCallback`` (registered by
            # ``TrainerFactory`` when ``system_metrics.callback_enabled``
            # is true) is the single source of truth — its payloads flow
            # through ``mlflow.log_metrics`` → resilient transport →
            # ``MetricsBuffer``.
            #
            # ``disable_system_metrics`` kwarg is retained for backwards
            # compatibility with callers that still pass it; it has no
            # effect on the new path.
            _ = disable_system_metrics

            self._install_resilient_transport_if_needed(mlflow)
            self._build_subcomponents()
            return True

        except ImportError:
            logger.warning("MLflow not installed.")
            self._mlflow = None  # type: ignore[attr-defined]
            return False
        except Exception as e:
            logger.warning(f"MLflow setup failed: {e}")
            self._resilient_transport.uninstall()  # type: ignore[attr-defined]
            self._mlflow = None  # type: ignore[attr-defined]
            return False

    def check_mlflow_connectivity(self, timeout: float = 5.0) -> bool:
        """Check whether the configured MLflow tracking backend is reachable."""
        tracking_uri = self.get_runtime_tracking_uri()
        if not tracking_uri or not tracking_uri.startswith("http"):
            return True
        if isinstance(self._gateway, NullMLflowGateway):  # type: ignore[arg-type]
            self._gateway = MLflowGateway(
                tracking_uri,
                ca_bundle_path=self._mlflow_config.ca_bundle_path,  # type: ignore[attr-defined]
            )  # type: ignore[attr-defined]
        return self._gateway.check_connectivity(timeout)  # type: ignore[attr-defined]

    def get_runtime_tracking_uri(self) -> str:
        if self._resolved_uris is None:  # type: ignore[attr-defined,unreachable]
            self._resolved_uris = resolve_mlflow_uris(  # type: ignore[unreachable]
                self._mlflow_config,  # type: ignore[attr-defined]
                runtime_role=self._runtime_role,  # type: ignore[attr-defined]
            )
        return self._resolved_uris.runtime_tracking_uri  # type: ignore[attr-defined]

    def get_effective_local_tracking_uri(self) -> str:
        if self._resolved_uris is None:  # type: ignore[attr-defined,unreachable]
            self._resolved_uris = resolve_mlflow_uris(  # type: ignore[unreachable]
                self._mlflow_config,  # type: ignore[attr-defined]
                runtime_role=self._runtime_role,  # type: ignore[attr-defined]
            )
        return self._resolved_uris.effective_local_tracking_uri  # type: ignore[attr-defined]

    def get_effective_remote_tracking_uri(self) -> str:
        if self._resolved_uris is None:  # type: ignore[attr-defined,unreachable]
            self._resolved_uris = resolve_mlflow_uris(  # type: ignore[unreachable]
                self._mlflow_config,  # type: ignore[attr-defined]
                runtime_role=self._runtime_role,  # type: ignore[attr-defined]
            )
        return self._resolved_uris.effective_remote_tracking_uri  # type: ignore[attr-defined]

    def get_raw_tracking_uri(self) -> str | None:
        return getattr(self._mlflow_config, "tracking_uri", None)  # type: ignore[attr-defined]

    def get_raw_local_tracking_uri(self) -> str | None:
        return getattr(self._mlflow_config, "local_tracking_uri", None)  # type: ignore[attr-defined]

    def get_last_connectivity_error(self) -> RyotenkAIError | None:
        return self._gateway.last_connectivity_error  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _install_resilient_transport_if_needed(self, mlflow: Any) -> None:
        tracking_uri = self.get_runtime_tracking_uri()  # type: ignore[attr-defined]
        if self._runtime_role != "training":  # type: ignore[attr-defined]
            return
        if not tracking_uri.startswith("http"):
            return
        self._resilient_transport.install(mlflow)  # type: ignore[attr-defined]

        # Attach metrics buffer for offline buffering during circuit-breaker-open periods.
        # Phase 12.A.2 — pass training.metrics_buffer config through if it's
        # reachable on the manager. Defensive lookup so test instances built
        # with partial mocks still work; absent config = lossless default
        # (keep_all=True), which is strictly more permissive than legacy.
        try:
            import os

            from ryotenkai_pod.trainer.mlflow.metrics_buffer import MetricsBuffer

            workspace = os.environ.get("WORKSPACE_PATH", "/workspace")
            buffer_config = self._resolve_metrics_buffer_config()
            buffer = MetricsBuffer(buffer_dir=workspace, config=buffer_config)
            self._resilient_transport.attach_buffer(buffer)  # type: ignore[attr-defined]
        except Exception as e:
            from ryotenkai_shared.utils.logger import get_logger

            get_logger(__name__).debug("Metrics buffer not attached: %s", e)

    def _resolve_metrics_buffer_config(self) -> Any | None:
        """Best-effort lookup of ``training.metrics_buffer`` from the
        manager's pipeline config.

        Returns ``None`` when the config isn't reachable — caller falls
        back to MetricsBuffer's lossless default. Callers who want
        custom decimation set ``training.metrics_buffer.keep_all=false``
        in their YAML; everyone else gets the safer keep-everything
        behaviour.
        """
        # MLflowManager.__init__ stashes the PipelineConfig on `self.config`.
        # ``training`` is a top-level field; ``metrics_buffer`` is the
        # Phase 12.A.2 sub-block.
        pipeline_cfg = getattr(self, "config", None)
        if pipeline_cfg is None:
            return None
        training_cfg = getattr(pipeline_cfg, "training", None)
        if training_cfg is None:
            return None
        return getattr(training_cfg, "metrics_buffer", None)

    def _restore_deleted_experiments(self) -> None:
        """Restore soft-deleted experiments before setting the active one."""
        try:
            client = self._gateway.get_client()  # type: ignore[attr-defined]
            try:
                exp0 = client.get_experiment(MLFLOW_EXPERIMENT_DEFAULT_ID)
                if getattr(exp0, "lifecycle_stage", None) == "deleted":
                    client.restore_experiment(MLFLOW_EXPERIMENT_DEFAULT_ID)
                    logger.warning("Restored deleted MLflow experiment: id=0 (Default)")
            except Exception:
                pass
            try:
                exp = client.get_experiment_by_name(self._mlflow_config.experiment_name)  # type: ignore[attr-defined]
                if exp is not None and getattr(exp, "lifecycle_stage", None) == "deleted":
                    client.restore_experiment(exp.experiment_id)
                    logger.warning(f"Restored deleted MLflow experiment: {self._mlflow_config.experiment_name}")  # type: ignore[attr-defined]
            except Exception:
                pass
        except Exception as e:
            logger.debug(f"MLflow experiment restore skipped: {e}")

    def _build_subcomponents(self) -> None:
        """Update mlflow-dependent subcomponents after successful setup."""
        assert self._mlflow_config is not None  # type: ignore[attr-defined]
        experiment_name = getattr(self._mlflow_config, "experiment_name", None)  # type: ignore[attr-defined]
        tracking_uri = self._gateway.uri  # type: ignore[attr-defined]

        self._autolog._mlflow = self._mlflow  # type: ignore[attr-defined]
        self._autolog._tracking_uri = tracking_uri  # type: ignore[attr-defined]
        self._analytics._mlflow = self._mlflow  # type: ignore[attr-defined]
        self._analytics._experiment_name = experiment_name  # type: ignore[attr-defined]
        self._analytics._gateway = self._gateway  # type: ignore[attr-defined]

        self._registry = MLflowModelRegistry(self._gateway, self._mlflow, log_model_enabled=False)  # type: ignore[attr-defined]
        self._dataset_logger = self._make_dataset_logger(mlflow_module=self._mlflow)  # type: ignore[attr-defined]
        self._domain_logger = MLflowDomainLogger(  # type: ignore[attr-defined]
            self,  # type: ignore[arg-type]
            self._event_log,  # type: ignore[attr-defined]
        )

    def _load_description_file(self) -> str | None:
        """Load run description from file (custom or default template)."""
        default_template = Path(__file__).parent.parent.parent / "templates" / "experiment_description.md"
        custom_path = None
        if self._mlflow_config and self._mlflow_config.run_description_file:  # type: ignore[attr-defined]
            custom_path = Path(self._mlflow_config.run_description_file)  # type: ignore[attr-defined]

        if custom_path and custom_path.exists():
            logger.debug(f"[MLFLOW:DESC] Loaded from {custom_path}")
            return custom_path.read_text(encoding="utf-8")

        if default_template.exists():
            return default_template.read_text(encoding="utf-8")

        return None


__all__ = ["MLflowSetupMixin"]
