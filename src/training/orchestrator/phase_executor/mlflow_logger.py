"""
MlflowPhaseLogger — MLflow experiment tracking for individual training phases.

Handles:
- Starting/ending nested MLflow runs per phase
- Logging phase params, dataset info, metrics, and errors
- System metrics lifecycle management for nested vs parent runs
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from src.training.constants import (
    TAG_PHASE_IDX,
    TAG_STRATEGY_TYPE,
    TRUNCATE_ERROR_MSG,
)
from src.training.metrics_models import TrainingMetricsSnapshot
from src.utils.logger import logger

if TYPE_CHECKING:
    from src.config import PipelineConfig, StrategyPhaseConfig
    from src.utils.container import IMLflowManager


# Phase 9.B — retry-with-grace constants for ``start_nested_run``.
#
# Real-world flap profile we cover: Mac sleeps for ≤30 seconds at a
# phase boundary (CPT → SFT → DPO), the trainer hits ``start_run`` mid-
# flap, fails on the first try, retries with exponential backoff, and
# the upstream comes back online before the budget expires. 5 attempts
# at 1s + 2s + 4s + 8s + 16s = 31s total wait.
#
# When the budget is exhausted we still return ``None`` (current
# behaviour) — losing the nested run is degraded but acceptable; the
# trainer continues writing into the parent run as a fallback so we
# never crash mid-cycle on a transient outage.
_NESTED_RUN_MAX_ATTEMPTS: int = 5
_NESTED_RUN_INITIAL_BACKOFF_S: float = 1.0
_NESTED_RUN_BACKOFF_MULTIPLIER: float = 2.0


class MlflowPhaseLogger:
    """
    Encapsulates all MLflow interactions for a single training phase.

    Responsibilities:
    - Nested run lifecycle (start / end with correct status)
    - Logging params, datasets, metrics, errors, tags
    - System metrics switch between parent and nested runs
    """

    def __init__(
        self,
        mlflow_manager: IMLflowManager | None,
        config: PipelineConfig,
    ) -> None:
        self._mlflow_manager = mlflow_manager
        self.config = config

    # ------------------------------------------------------------------
    # Nested run lifecycle
    # ------------------------------------------------------------------

    def start_nested_run(
        self,
        phase_idx: int,
        phase: StrategyPhaseConfig,
        *,
        sleep: Any | None = None,
    ) -> Any:
        """
        Start MLflow nested run for phase, with Phase 9.B retry-grace.

        Returns the run object on success, or ``None`` after all retry
        attempts exhausted (or MLflow disabled). When MLflow upstream
        flaps for a short window — e.g. Mac asleep at the phase
        boundary in a multi-strategy chain — the retry loop covers up
        to ~31 seconds before giving up.

        Retry policy: 5 attempts with exponential backoff
        (1s → 2s → 4s → 8s → 16s, total 31s). Picked to cover the
        common "Mac quick nap" cancellation pattern; longer Mac
        sleeps (overnight) still fall through to ``None`` and the
        trainer writes into the parent run as a fallback.

        Args:
            phase_idx: 0-based phase index, used for the run name +
                ``mlflow.phase_idx`` tag.
            phase: phase config (provides ``strategy_type``).
            sleep: injectable sleep function for tests — defaults to
                :func:`time.sleep`. Tests pass a no-op so the backoff
                doesn't actually wait.

        Returns:
            The MLflow run object if any attempt succeeded; ``None``
            otherwise.
        """
        if self._mlflow_manager is None or not self._mlflow_manager.is_active:
            return None

        sleep_fn = sleep if sleep is not None else time.sleep
        run_name = f"phase_{phase_idx}_{phase.strategy_type}"
        last_error: Exception | None = None

        for attempt in range(1, _NESTED_RUN_MAX_ATTEMPTS + 1):
            try:
                import mlflow

                # Native MLflow background sampler is no longer used in this
                # codebase — ``SystemMetricsCallback`` (HF Trainer-aligned,
                # buffered through ``ResilientMLflowTransport``) is the only
                # source of system metrics. The previous toggling of
                # ``mlflow.enable_system_metrics_logging`` /
                # ``disable_system_metrics_logging`` around nested-run
                # boundaries is now a no-op concern.

                run = mlflow.start_run(run_name=run_name, nested=True)

                mlflow.set_tags(
                    {
                        TAG_PHASE_IDX: str(phase_idx),
                        TAG_STRATEGY_TYPE: phase.strategy_type,
                    }
                )

                if attempt > 1:
                    logger.info(
                        "[PE:MLFLOW_NESTED_RUN_STARTED_AFTER_RETRY] "
                        "phase=%d strategy=%s attempt=%d",
                        phase_idx, phase.strategy_type, attempt,
                    )
                else:
                    logger.debug(
                        "[PE:MLFLOW_NESTED_RUN_STARTED] phase_%d_%s",
                        phase_idx, phase.strategy_type,
                    )
                return run

            except Exception as exc:
                last_error = exc

                if attempt >= _NESTED_RUN_MAX_ATTEMPTS:
                    # Budget exhausted — fall through to the warn + return.
                    break

                backoff = _NESTED_RUN_INITIAL_BACKOFF_S * (
                    _NESTED_RUN_BACKOFF_MULTIPLIER ** (attempt - 1)
                )
                logger.warning(
                    "[PE:MLFLOW_NESTED_RUN_START_RETRY] phase=%d strategy=%s "
                    "attempt=%d/%d failed (%s); waiting %.1fs",
                    phase_idx, phase.strategy_type,
                    attempt, _NESTED_RUN_MAX_ATTEMPTS, exc, backoff,
                )
                sleep_fn(backoff)

        # All attempts failed — log warning and return None.
        # Trainer falls back to writing into the parent run.
        logger.warning(
            "[PE:MLFLOW_NESTED_RUN_START_FAILED] phase=%d strategy=%s "
            "after %d attempts: %s — phase will log into parent run",
            phase_idx, phase.strategy_type,
            _NESTED_RUN_MAX_ATTEMPTS, last_error,
        )
        return None

    def end_nested_run(self, run: Any, status: str = "FINISHED") -> None:
        """
        End MLflow nested run with explicit status and restore parent run.

        Args:
            run: Run object from start_nested_run
            status: Run status - "FINISHED" for success, "FAILED" for error
        """
        if run is None:
            return

        try:
            import mlflow

            parent_run_id = self._mlflow_manager.parent_run_id if self._mlflow_manager else None

            mlflow.end_run(status=status)
            logger.debug(f"[PE:MLFLOW_NESTED_RUN_ENDED] status={status}")

            # MLflow automatically restores the parent run after ending a nested run.
            # Only call start_run if the parent is not already the active run.
            if parent_run_id:
                active = mlflow.active_run()
                if active is None or active.info.run_id != parent_run_id:
                    mlflow.start_run(run_id=parent_run_id, nested=False)
                    logger.debug(f"[PE:MLFLOW_PARENT_RUN_RESTORED] run_id={parent_run_id}")
                else:
                    logger.debug(f"[PE:MLFLOW_PARENT_RUN_ALREADY_ACTIVE] run_id={parent_run_id}")

        except Exception as e:
            logger.debug(f"[PE:MLFLOW_NESTED_RUN_END_FAILED] {e}")

    # ------------------------------------------------------------------
    # Phase logging helpers
    # ------------------------------------------------------------------

    def log_phase_start(self, phase_idx: int, phase: StrategyPhaseConfig) -> None:
        """Log phase start params to MLflow nested run."""
        if self._mlflow_manager is None:
            return

        global_hp = self.config.training.hyperparams
        phase_hp = phase.hyperparams

        effective_params = global_hp.model_dump(exclude_none=True)
        if phase_hp:
            effective_params.update(phase_hp.model_dump(exclude_none=True))

        params_to_log = {
            TAG_PHASE_IDX: phase_idx,
            TAG_STRATEGY_TYPE: phase.strategy_type,
            "dataset": phase.dataset or "default",
        }
        for k, v in effective_params.items():
            params_to_log[f"training.hyperparams.actual.{k}"] = v

        self._mlflow_manager.log_params(params_to_log)
        self._mlflow_manager.set_tags(
            {
                TAG_STRATEGY_TYPE: phase.strategy_type,
                TAG_PHASE_IDX: str(phase_idx),
            }
        )

    def log_dataset(
        self,
        phase_idx: int,
        dataset: Any,
        phase: StrategyPhaseConfig,
    ) -> None:
        """
        Log dataset to MLflow with proper experiment → dataset → run linking.
        """
        if self._mlflow_manager is None:
            return

        try:
            num_samples = len(dataset) if hasattr(dataset, "__len__") else 0
            dataset_name = phase.dataset or f"phase_{phase_idx}_dataset"

            dataset_config = self.config.get_dataset_for_strategy(phase)
            source_uri = dataset_config.get_source_uri()

            mlflow_dataset = self._mlflow_manager.create_mlflow_dataset(
                data=dataset,
                name=dataset_name,
                source=source_uri,
            )

            if mlflow_dataset is not None:
                self._mlflow_manager.log_dataset_input(mlflow_dataset, context="training")
                logger.debug(
                    f"[PE:MLFLOW_DATASET_LINKED] name={dataset_name}, "
                    f"source={source_uri}, samples={num_samples}"
                )
            else:
                self._mlflow_manager.log_dataset_info(
                    name=dataset_name,
                    source=dataset_config.get_source_type(),
                    num_rows=num_samples,
                    extra_tags={
                        TAG_PHASE_IDX: str(phase_idx),
                        TAG_STRATEGY_TYPE: phase.strategy_type,
                        "dataset.source_uri": source_uri,
                    },
                )
                logger.debug(f"[PE:MLFLOW_DATASET_INFO] name={dataset_name}, samples={num_samples}")

        except Exception as e:
            logger.debug(f"[PE:MLFLOW_DATASET_LOG_FAILED] {e}")

    def log_completion(
        self,
        phase_idx: int,
        metrics: TrainingMetricsSnapshot,
        checkpoint_path: str,
    ) -> None:
        """Log phase completion to MLflow."""
        if self._mlflow_manager is None:
            return

        float_metrics = {k: float(v) for k, v in metrics.numeric_kwargs().items()}
        if float_metrics:
            self._mlflow_manager.log_metrics(float_metrics)

        self._mlflow_manager.log_params({"checkpoint_path": checkpoint_path})
        self._mlflow_manager.set_tags({TAG_PHASE_IDX: str(phase_idx), "status": "completed"})

    def log_cache_hit(self, phase_idx: int, phase: StrategyPhaseConfig) -> None:
        """Log adapter cache hit to MLflow nested run."""
        if self._mlflow_manager is None:
            return

        self._mlflow_manager.set_tags(
            {
                TAG_PHASE_IDX: str(phase_idx),
                TAG_STRATEGY_TYPE: phase.strategy_type,
                "status": "cache_hit",
                "adapter_cache.hit": "true",
            }
        )
        self._mlflow_manager.log_params({"adapter_cache.hit": True})

    def log_error(self, phase_idx: int, error_type: str, error_msg: str) -> None:
        """Log error to MLflow."""
        if self._mlflow_manager is None:
            return

        self._mlflow_manager.set_tags(
            {
                TAG_PHASE_IDX: str(phase_idx),
                "status": "failed",
                "error_type": error_type,
                "error_msg": error_msg[:TRUNCATE_ERROR_MSG],
            }
        )


__all__ = ["MlflowPhaseLogger"]
