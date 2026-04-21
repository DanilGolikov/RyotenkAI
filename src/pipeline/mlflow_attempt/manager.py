"""MLflow attempt lifecycle: setup, preflight, root/attempt runs, teardown.

Encapsulates the full MLflow integration for one pipeline attempt. Holds the
MLflowManager plus the open root/attempt run objects so the orchestrator only
sees a single collaborator instead of four scattered attributes.

Design choices:

* One class (no bootstrap/attempt split). The six original methods share
  ``_manager`` / ``_run_context`` / ``_attempt_run`` state; separating them
  would require cross-object reference juggling without any gain in testability.
* Setup is wrapped in a ``try`` / partial-cleanup branch to guarantee no
  orphan root/attempt runs if any step after their opening fails
  (mitigation for MLflow double-close risk).
* Preflight returns ``AppError | None`` instead of raising a bespoke exception,
  so the orchestrator keeps control over launch-rejection policy.
* Teardown accepts two optional hooks to let the orchestrator inject
  reporting side effects (training-metrics aggregation, experiment report)
  without this manager knowing about them.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

from src.pipeline.constants import MLFLOW_CATEGORY_PIPELINE, MLFLOW_SOURCE_ORCHESTRATOR
from src.pipeline.stages import PipelineContextKeys
from src.training.managers.mlflow_manager import MLflowManager
from src.utils.logger import logger
from src.utils.result import AppError

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from src.pipeline.state import PipelineAttemptState, PipelineState
    from src.utils.config import PipelineConfig


class MLflowAttemptManager:
    """Owns MLflowManager + root/attempt run lifecycle for one pipeline attempt."""

    def __init__(self, config: PipelineConfig, config_path: Path) -> None:
        self._config = config
        self._config_path = config_path
        self._manager: MLflowManager | None = None
        self._run_context: Any = None
        self._root_run: Any = None
        self._attempt_run: Any = None

    # ---- public accessors ---------------------------------------------------

    @property
    def manager(self) -> MLflowManager | None:
        return self._manager

    @property
    def is_active(self) -> bool:
        return self._manager is not None and self._manager.is_active

    def get_run_id(self) -> str | None:
        """Current MLflow run_id (prefers public property; falls back to legacy attr)."""
        if not self._manager:
            return None
        run_id = getattr(self._manager, "run_id", None)
        if isinstance(run_id, str) and run_id:
            return run_id
        legacy_run_id = getattr(self._manager, "_run_id", None)
        if isinstance(legacy_run_id, str) and legacy_run_id:
            return legacy_run_id
        return None

    # ---- setup --------------------------------------------------------------

    def bootstrap(self) -> MLflowManager | None:
        """Create and configure MLflowManager (control-plane, system metrics off)."""
        try:
            # 1. Force config setting — disable system metrics on the control-plane run
            self._config.experiment_tracking.mlflow.system_metrics_callback_enabled = False

            # 2. Force environment variable (critical for MLflow internals)
            import os

            os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "false"

            manager = MLflowManager(self._config, runtime_role="control_plane")

            # 3. Disable system metrics logging in MLflow client directly (best-effort)
            try:
                import mlflow

                mlflow.disable_system_metrics_logging()
            except Exception:
                # MLflow may be missing or API may differ — manager.setup still handles it.
                pass

            manager.setup(disable_system_metrics=True)
            self._manager = manager
            return manager
        except Exception as e:
            logger.warning(f"MLflow setup failed: {e}")
            self._manager = None
            return None

    def setup_for_attempt(
        self,
        *,
        state: PipelineState,
        attempt: PipelineAttemptState,
        start_stage_idx: int,
        context: dict[str, Any],
        total_stages: int,
        run_directory: Path | None,
        manager: MLflowManager | None = None,
    ) -> None:
        """Open the root + nested attempt MLflow runs and record their IDs on state.

        If any step after opening a run fails, the partially-opened run is
        closed before re-raising so we never leak an active MLflow run.

        Args:
            manager: Pre-bootstrapped MLflowManager. When ``None`` the manager
                is bootstrapped in-place — kept for backward compat with callers
                that don't want to manage bootstrap themselves.
        """
        self._manager = manager if manager is not None else self.bootstrap()
        if not self._manager or not self._manager.is_active:
            return

        runtime_tracking_uri = self._manager.get_runtime_tracking_uri()
        ca_bundle_path = getattr(self._config.experiment_tracking.mlflow, "ca_bundle_path", None)
        state.mlflow_runtime_tracking_uri = (
            runtime_tracking_uri if isinstance(runtime_tracking_uri, str) and runtime_tracking_uri else None
        )
        state.mlflow_ca_bundle_path = ca_bundle_path if isinstance(ca_bundle_path, str) and ca_bundle_path else None

        try:
            import mlflow

            mlflow.disable_system_metrics_logging()
        except Exception:
            pass

        try:
            self._open_root_run(state=state, attempt=attempt)
            self._open_attempt_run(state=state, attempt=attempt)
            self._log_attempt_metadata(
                context=context,
                total_stages=total_stages,
                start_stage_idx=start_stage_idx,
                run_directory=run_directory,
            )
        except Exception:
            # Close anything we may have opened so teardown does not double-close.
            self._cleanup_partial_runs()
            raise

    def _require_manager(self) -> MLflowManager:
        """Return ``self._manager`` or raise a clear error instead of a bare AttributeError.

        Use this anywhere a method assumes bootstrap() has already put a manager
        in place. ``assert`` is avoided — Python -O removes asserts in production.
        """
        if self._manager is None:
            raise MLflowManagerNotInitializedError(
                "MLflowAttemptManager has no active MLflowManager; bootstrap() must be called first"
            )
        return self._manager

    def _open_root_run(self, *, state: PipelineState, attempt: PipelineAttemptState) -> None:
        manager = self._require_manager()
        if state.root_mlflow_run_id:
            self._root_run = self.open_existing_root_run(state.root_mlflow_run_id)
            attempt.root_mlflow_run_id = state.root_mlflow_run_id
        else:
            self._run_context = manager.start_run(run_name=state.logical_run_id)
            self._root_run = self._run_context.__enter__()
            state.root_mlflow_run_id = self.get_run_id()
            attempt.root_mlflow_run_id = state.root_mlflow_run_id

    def _open_attempt_run(self, *, state: PipelineState, attempt: PipelineAttemptState) -> None:
        manager = self._require_manager()
        attempt_name = f"{state.logical_run_id}_attempt_{attempt.attempt_no}"
        attempt_tags = {
            "pipeline.logical_run_id": state.logical_run_id,
            "pipeline.attempt_id": attempt.attempt_id,
            "pipeline.attempt_no": str(attempt.attempt_no),
        }
        self._attempt_run = manager.start_nested_run(run_name=attempt_name, tags=attempt_tags)
        self._attempt_run.__enter__()
        attempt.pipeline_attempt_mlflow_run_id = self.get_run_id()

    def _log_attempt_metadata(
        self,
        *,
        context: dict[str, Any],
        total_stages: int,
        start_stage_idx: int,
        run_directory: Path | None,
    ) -> None:
        manager = self._require_manager()
        context[PipelineContextKeys.MLFLOW_PARENT_RUN_ID] = self.get_run_id()
        context[PipelineContextKeys.MLFLOW_MANAGER] = manager
        manager.log_event_start(
            "Pipeline attempt started",
            category=MLFLOW_CATEGORY_PIPELINE,
            source=MLFLOW_SOURCE_ORCHESTRATOR,
        )
        manager.log_pipeline_config(self._config)
        manager.log_dataset_config(self._config)
        manager.log_params(
            {
                "pipeline.total_stages": total_stages,
                "pipeline.start_stage": start_stage_idx,
                "pipeline.run_directory": str(run_directory),
            }
        )

    def _cleanup_partial_runs(self) -> None:
        """Best-effort close of runs already opened during a failed setup."""
        if self._attempt_run is not None:
            with contextlib.suppress(Exception):
                self._attempt_run.__exit__(None, None, None)
            self._attempt_run = None
        if self._run_context is not None:
            with contextlib.suppress(Exception):
                self._run_context.__exit__(None, None, None)
            self._run_context = None
        self._root_run = None

    # ---- preflight ----------------------------------------------------------

    def ensure_preflight(self) -> AppError | None:
        """Validate that MLflow setup and connectivity are alive.

        Returns an AppError the orchestrator can wrap in a LaunchPreparationError,
        or None when everything is healthy.
        """
        mlflow_cfg = self._config.experiment_tracking.mlflow
        raw_tracking_uri = getattr(mlflow_cfg, "tracking_uri", None)
        raw_local_tracking_uri = getattr(mlflow_cfg, "local_tracking_uri", None)
        tracking_uri = (
            self._manager.get_runtime_tracking_uri()
            if self._manager is not None
            else (raw_local_tracking_uri or raw_tracking_uri)
        )

        if self._manager is None or not self._manager.is_active:
            return AppError(
                code="MLFLOW_PREFLIGHT_SETUP_FAILED",
                message=(
                    "MLflow setup failed "
                    f"(effective_uri={tracking_uri}, raw_tracking_uri={raw_tracking_uri}, "
                    f"raw_local_tracking_uri={raw_local_tracking_uri})"
                ),
                details={
                    "effective_uri": tracking_uri,
                    "raw_tracking_uri": raw_tracking_uri,
                    "raw_local_tracking_uri": raw_local_tracking_uri,
                },
            )
        if not self._manager.check_mlflow_connectivity():
            gateway_error = self._manager.get_last_connectivity_error()
            error_code = gateway_error.code if gateway_error is not None else "MLFLOW_PREFLIGHT_UNREACHABLE"
            error_message = (
                f"MLflow not reachable (effective_uri={tracking_uri}, raw_tracking_uri={raw_tracking_uri}, "
                f"raw_local_tracking_uri={raw_local_tracking_uri})"
            )
            if gateway_error is not None:
                error_message = f"{error_message}: {gateway_error.message}"
            return AppError(
                code=error_code,
                message=error_message,
                details={
                    "effective_uri": tracking_uri,
                    "raw_tracking_uri": raw_tracking_uri,
                    "raw_local_tracking_uri": raw_local_tracking_uri,
                    "gateway_error": gateway_error.to_log_dict() if gateway_error is not None else None,
                },
            )
        return None

    def log_config_artifact(self) -> None:
        """Log the pipeline config file as an MLflow artifact if present."""
        if self._manager is not None and self._config_path.exists():
            self._manager.log_artifact(str(self._config_path))

    # ---- root run -----------------------------------------------------------

    def open_existing_root_run(self, root_run_id: str) -> Any:
        """Reopen an existing root run so nested attempts log under the same parent.

        Delegates to ``MLflowManager.adopt_existing_run`` so we don't touch the
        manager's private attributes.
        """
        manager = self._require_manager()
        return manager.adopt_existing_run(root_run_id)

    # ---- teardown -----------------------------------------------------------

    def teardown_attempt(
        self,
        *,
        pipeline_success: bool,
        attempt_run_id: str | None,
        on_before_end: Callable[[], None] | None = None,
        state_path_supplier: Callable[[], Path | None] | None = None,
        on_after_end: Callable[[str | None], None] | None = None,
    ) -> None:
        """Close the nested attempt run, log final state, close the root run, cleanup.

        Hook order mirrors the original in-orchestrator implementation:
        1. ``on_before_end`` — aggregate metrics while the attempt run is still open.
        2. Close the attempt run.
        3. ``state_path_supplier`` — orchestrator syncs the final state to disk and
           returns its path (``None`` skips the artifact upload).
        4. ``end_run`` on the root run.
        5. ``on_after_end`` — generate the experiment report.
        """
        if self._manager:
            if on_before_end is not None:
                try:
                    on_before_end()
                except Exception as e:
                    logger.warning(f"MLflow teardown before-end hook failed: {e}")

            if self._attempt_run is not None:
                with contextlib.suppress(Exception):
                    if pipeline_success:
                        self._attempt_run.__exit__(None, None, None)
                    else:
                        _exc = RuntimeError("Pipeline attempt failed")
                        self._attempt_run.__exit__(type(_exc), _exc, None)
                self._attempt_run = None

            if state_path_supplier is not None:
                with contextlib.suppress(Exception):
                    state_path = state_path_supplier()
                    if state_path is not None and state_path.exists():
                        self._manager.log_artifact(str(state_path))

            root_status = "FINISHED" if pipeline_success else "FAILED"
            self._manager.end_run(status=root_status)

            if on_after_end is not None:
                try:
                    on_after_end(attempt_run_id)
                except Exception as e:
                    logger.warning(f"MLflow teardown after-end hook failed: {e}")

        if self._run_context is not None:
            with contextlib.suppress(Exception):
                self._run_context.__exit__(None, None, None)
            self._run_context = None

        self._root_run = None

        if self._manager:
            self._manager.cleanup()


class MLflowManagerNotInitializedError(RuntimeError):
    """Raised when an operation needs a ready MLflowManager but none is bootstrapped.

    Prefer this over ``assert`` — asserts are disabled under ``python -O`` and
    would silently degrade to opaque AttributeError in production.
    """


__all__ = [
    "MLflowAttemptManager",
    "MLflowManagerNotInitializedError",
]
