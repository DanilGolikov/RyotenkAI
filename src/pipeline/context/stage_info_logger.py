"""Post-stage MLflow info logging.

Extracted from PipelineOrchestrator as ``_log_stage_specific_info``. After
each stage completes, the orchestrator calls ``StageInfoLogger.log(...)``
which inspects the stage's context entry and emits stage-specific
MLflow events / params / metrics.

The logger is stateless: the MLflow manager and the context dict are
passed explicitly on each call. No-op when ``mlflow_manager`` is ``None``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.pipeline.constants import (
    CTX_PROVIDER_NAME_UNKNOWN,
    CTX_PROVIDER_TYPE_UNKNOWN,
    CTX_RUNTIME_SECONDS,
    CTX_TRAINING_DURATION,
    CTX_TRAINING_INFO,
)
from src.pipeline.stages import StageNames

if TYPE_CHECKING:
    from src.training.managers.mlflow_manager import MLflowManager


_KEY_UPLOAD_DURATION = "upload_duration_seconds"


class StageInfoLogger:
    """Emit per-stage info events/params to MLflow after a stage completes."""

    def log(
        self,
        *,
        mlflow_manager: MLflowManager | None,
        context: dict[str, Any],
        stage_name: str,
    ) -> None:
        if mlflow_manager is None:
            return

        if stage_name == StageNames.GPU_DEPLOYER and StageNames.GPU_DEPLOYER in context:
            self._log_gpu_deployer(mlflow_manager, context[StageNames.GPU_DEPLOYER])
        elif stage_name == StageNames.DATASET_VALIDATOR and StageNames.DATASET_VALIDATOR in context:
            self._log_dataset_validator(mlflow_manager, context[StageNames.DATASET_VALIDATOR])
        elif stage_name == StageNames.TRAINING_MONITOR and StageNames.TRAINING_MONITOR in context:
            self._log_training_monitor(mlflow_manager, context[StageNames.TRAINING_MONITOR])
        elif stage_name == StageNames.MODEL_RETRIEVER and StageNames.MODEL_RETRIEVER in context:
            self._log_model_retriever(mlflow_manager, context[StageNames.MODEL_RETRIEVER])

    # ---- per-stage handlers -------------------------------------------------

    @staticmethod
    def _log_gpu_deployer(mlflow_manager: MLflowManager, deployer_ctx: Any) -> None:
        if not isinstance(deployer_ctx, dict):
            return
        mlflow_manager.log_provider_info(
            provider_name=deployer_ctx.get("provider_name", CTX_PROVIDER_NAME_UNKNOWN),
            provider_type=deployer_ctx.get("provider_type", CTX_PROVIDER_TYPE_UNKNOWN),
            gpu_type=deployer_ctx.get("gpu_type"),
            resource_id=deployer_ctx.get("resource_id"),
        )
        upload_dur = deployer_ctx.get(_KEY_UPLOAD_DURATION)
        deps_dur = deployer_ctx.get("deps_duration_seconds")
        if upload_dur:
            mlflow_manager.log_event_info(
                f"Files uploaded ({upload_dur:.1f}s)",
                category="deployment",
                source=StageNames.GPU_DEPLOYER,
                upload_duration_seconds=upload_dur,
            )
        if deps_dur:
            mlflow_manager.log_event_info(
                f"Dependencies installed ({deps_dur:.1f}s)",
                category="deployment",
                source=StageNames.GPU_DEPLOYER,
                deps_duration_seconds=deps_dur,
            )

    @staticmethod
    def _log_dataset_validator(mlflow_manager: MLflowManager, validator_ctx: Any) -> None:
        if not isinstance(validator_ctx, dict):
            return
        metrics = validator_ctx.get("metrics", {})  # noqa: WPS226
        if not metrics:
            return
        validation_mode = validator_ctx.get("validation_mode", "legacy")
        if validation_mode == "plugin":
            # Plugin-based validation: log every plugin metric, coercing to float when possible.
            params_to_log: dict[str, float | str] = {}
            for key, value in metrics.items():
                try:
                    params_to_log[f"dataset.{key}"] = float(value)
                except (ValueError, TypeError):
                    params_to_log[f"dataset.{key}"] = str(value)
            mlflow_manager.log_params(params_to_log)
        else:
            mlflow_manager.log_params(
                {
                    "dataset.sample_count": validator_ctx.get("sample_count", 0),
                    "dataset.avg_length": metrics.get("avg_length", 0),
                    "dataset.empty_ratio": metrics.get("empty_ratio", 0),
                    "dataset.diversity_score": metrics.get("diversity_score", 0),
                }
            )

    @staticmethod
    def _log_training_monitor(mlflow_manager: MLflowManager, monitor_ctx: Any) -> None:
        if not isinstance(monitor_ctx, dict):
            return
        training_dur = monitor_ctx.get(CTX_TRAINING_DURATION)
        if training_dur:
            mlflow_manager.log_event_info(
                f"Training completed ({training_dur:.1f}s)",
                category="training",
                source=StageNames.TRAINING_MONITOR,
                training_duration_seconds=training_dur,
            )

        training_info = monitor_ctx.get(CTX_TRAINING_INFO, {})
        if not isinstance(training_info, dict) or not training_info:
            return
        metrics_to_log: dict[str, float] = {}
        if training_info.get(CTX_RUNTIME_SECONDS):
            metrics_to_log[f"training.{CTX_RUNTIME_SECONDS}"] = training_info[CTX_RUNTIME_SECONDS]
        if training_info.get("final_loss"):
            metrics_to_log["training.final_loss"] = training_info["final_loss"]
        if training_info.get("final_accuracy"):
            metrics_to_log["training.final_accuracy"] = training_info["final_accuracy"]
        if training_info.get("total_steps"):
            metrics_to_log["training.total_steps"] = float(training_info["total_steps"])
        if metrics_to_log:
            mlflow_manager.log_metrics(metrics_to_log)

    @staticmethod
    def _log_model_retriever(mlflow_manager: MLflowManager, retriever_ctx: Any) -> None:
        if not isinstance(retriever_ctx, dict):
            return
        model_size = retriever_ctx.get("model_size_mb")
        if model_size:
            mlflow_manager.log_event_info(
                f"Model size: {model_size:.1f} MB",
                category="model",
                source=StageNames.MODEL_RETRIEVER,
                model_size_mb=model_size,
            )

        hf_uploaded = retriever_ctx.get("hf_uploaded")
        upload_dur = retriever_ctx.get(_KEY_UPLOAD_DURATION)
        if hf_uploaded and upload_dur:
            hf_repo = retriever_ctx.get("hf_repo_id", CTX_PROVIDER_NAME_UNKNOWN)
            mlflow_manager.log_event_info(
                f"Model uploaded to HF: {hf_repo} ({upload_dur:.1f}s)",
                category="model",
                source=StageNames.MODEL_RETRIEVER,
                hf_repo_id=hf_repo,
                upload_duration_seconds=upload_dur,
            )


__all__ = ["StageInfoLogger"]
