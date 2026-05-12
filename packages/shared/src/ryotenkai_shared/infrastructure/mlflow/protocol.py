"""Pipeline-side Protocol for the MLflow manager.

The pipeline (Mac orchestrator) calls into the MLflow manager that
lives in :mod:`src.training.managers.mlflow_manager`. The concrete
``MLflowManager`` class belongs to the *trainer* (pod-side); pipeline
should not import it directly because:

* It pulls trainer-only deps into the pipeline import closure.
* After Phase B (monorepo packagization) the trainer code is in a
  different uv-workspace package (``ryotenkai-pod``) that the
  control-plane package (``ryotenkai-control``) cannot import.

This Protocol captures the surface the pipeline actually uses
(catalogued by ``grep -rn "mlflow_manager\\.\\w" src/pipeline/`` —
plan §A.5 / R41). The trainer's concrete ``MLflowManager`` already
implements all of these methods, so structural typing makes the swap
zero-cost at runtime.

Tests can use any object that implements the listed methods; mocks
no longer have to subclass the heavyweight concrete class.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Generator
    from contextlib import AbstractContextManager


@runtime_checkable
class IMLflowManager(Protocol):
    """Pipeline-facing surface of the trainer's MLflow manager."""

    # ------------------------------------------------------------------
    # State / connectivity
    # ------------------------------------------------------------------

    @property
    def is_active(self) -> bool:
        """``True`` once :meth:`setup` succeeded and an mlflow client is wired."""
        ...

    @property
    def client(self) -> Any:
        """Underlying ``mlflow.tracking.MlflowClient`` (may be ``None``)."""
        ...

    @property
    def tracking_uri(self) -> str | None:
        """Effective tracking URI in use, or ``None`` if not initialised."""
        ...

    def setup(
        self,
        timeout: float = 5.0,
        max_retries: int = 3,
        disable_system_metrics: bool = False,
    ) -> bool: ...

    def cleanup(self) -> None: ...

    def check_mlflow_connectivity(self, timeout: float = 5.0) -> bool: ...

    def get_runtime_tracking_uri(self) -> str: ...

    def get_last_connectivity_error(self) -> Any: ...

    # ------------------------------------------------------------------
    # Run lifecycle
    # ------------------------------------------------------------------

    def start_run(
        self,
        run_name: str | None = None,
        description: str | None = None,
    ) -> AbstractContextManager[Any]: ...

    def start_nested_run(
        self,
        run_name: str,
        tags: dict[str, str] | None = None,
        description: str | None = None,
    ) -> AbstractContextManager[Any]: ...

    def end_run(self, status: str = "FINISHED") -> None: ...

    def adopt_existing_run(self, run_id: str) -> Any: ...

    # ------------------------------------------------------------------
    # Logging — params / metrics / tags / artifacts
    # ------------------------------------------------------------------

    def set_tags(self, tags: dict[str, str]) -> None: ...

    def log_params(self, params: dict[str, Any]) -> None: ...

    def log_metrics(
        self,
        metrics: dict[str, float],
        step: int | None = None,
    ) -> None: ...

    def log_artifact(
        self,
        local_path: str,
        artifact_path: str | None = None,
        run_id: str | None = None,
    ) -> bool: ...

    # ------------------------------------------------------------------
    # Domain logging — events / config / provider
    # ------------------------------------------------------------------

    def log_event_start(self, message: str, **kwargs: Any) -> dict[str, Any]: ...

    def log_event_info(self, message: str, **kwargs: Any) -> dict[str, Any]: ...

    def log_pipeline_config(self, config: Any) -> None: ...

    def log_dataset_config(self, config: Any) -> None: ...

    def log_provider_info(
        self,
        provider_name: str,
        provider_type: str,
        gpu_type: str | None = None,
        resource_id: str | None = None,
    ) -> None: ...


__all__ = ["IMLflowManager"]
