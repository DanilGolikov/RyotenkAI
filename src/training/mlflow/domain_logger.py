"""
MLflowDomainLogger — domain-specific logging for pipeline, training, memory events.

Responsibilities (Single Responsibility):
  - Translate high-level domain events into MLflow log_params/log_metrics/set_tags calls
  - Translate memory/pipeline events into MLflowEventLog.log_event calls
  - Know about domain concepts (GPU tiers, strategy chains, pipeline stages, OOM)

Depends on:
  - IMLflowPrimitives (log_params, log_metrics, set_tags, log_dict)
  - MLflowEventLog (log_event)

Does NOT know about MLflow SDK, gateway, or run lifecycle.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.training.constants import (
    MLFLOW_CATEGORY_MEMORY,
    MLFLOW_SEVERITY_ERROR,
    MLFLOW_SEVERITY_INFO,
    MLFLOW_SEVERITY_WARNING,
    MLFLOW_SOURCE_MEMORY_MANAGER,
)
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.training.mlflow.event_log import MLflowEventLog
    from src.training.mlflow.primitives import IMLflowPrimitives
    from src.utils.config import PipelineConfig

logger = get_logger(__name__)


class MLflowDomainLogger:
    """
    Domain-specific logging facade over MLflow primitives and event log.

    Translates high-level domain events (training config, GPU metrics,
    pipeline stages, OOM events, etc.) into the correct MLflow calls.

    Args:
        primitives: MLflow logging primitives (log_params, log_metrics, set_tags, log_dict)
        event_log: In-memory event log
    """

    def __init__(
        self,
        primitives: IMLflowPrimitives,
        event_log: MLflowEventLog,
    ) -> None:
        self._p = primitives
        self._event_log = event_log

    # =========================================================================
    # TRAINING CONFIG
    # =========================================================================

    def log_training_config(self, config: PipelineConfig) -> None:
        """Log training configuration as parameters and tags."""
        hp = config.training.hyperparams

        params: dict[str, Any] = {
            "model_name": config.model.name,
            "training_type": config.training.type,
            "batch_size": hp.per_device_train_batch_size,
            "gradient_accumulation": hp.gradient_accumulation_steps,
            "bf16": hp.bf16,
            "fp16": hp.fp16,
            "learning_rate_scheduler": hp.lr_scheduler_type,
            "warmup_ratio": hp.warmup_ratio,
            "weight_decay": hp.weight_decay,
        }

        lora = None
        if config.training.type in ("lora", "qlora"):
            with contextlib.suppress(ValueError):
                lora = config.training.get_adapter_config()
        if lora is not None and config.training.type in ("lora", "qlora"):
            params.update(
                {
                    "lora_r": lora.r,
                    "lora_alpha": lora.lora_alpha,
                    "lora_dropout": lora.lora_dropout,
                    "lora_target_modules": str(lora.target_modules),
                }
            )

        params.update(
            {
                "training_type": config.training.type,
                "load_in_4bit": config.training.get_effective_load_in_4bit(),
            }
        )

        self._p.log_params(params)

        strategies = config.training.get_strategy_chain()
        strategy_chain = "→".join(s.strategy_type for s in strategies) if strategies else "none"
        self._p.set_tags(
            {
                "model_base": config.model.name.split("/")[-1],
                "model_full": config.model.name,
                "training_type": config.training.type,
                "strategy_chain": strategy_chain,
                "num_phases": str(len(strategies)) if strategies else "0",
            }
        )

    def log_pipeline_config(self, config: PipelineConfig) -> None:
        """Log full pipeline configuration as parameters and tags."""
        strategies = config.training.get_strategy_chain()
        strategy_chain = "→".join(s.strategy_type for s in strategies) if strategies else "none"

        tags: dict[str, str] = {
            "model.name": config.model.name,
            "model.base": config.model.name.split("/")[-1],
            "training.type": config.training.type,
            "training.strategy_chain": strategy_chain,
            "training.num_phases": str(len(strategies)) if strategies else "0",
        }

        with contextlib.suppress(AttributeError, KeyError, RuntimeError, TypeError, ValueError):
            provider_name = config.get_active_provider_name()
            tags["provider.name"] = provider_name  # noqa: WPS226
            provider_config = config.get_provider_config()
            provider_train_cfg_obj = (
                config.get_provider_training_config()
                if hasattr(config, "get_provider_training_config")
                else provider_config
            )
            if provider_train_cfg_obj is not None:
                if isinstance(provider_train_cfg_obj, dict):
                    tags["provider.gpu_type"] = str(provider_train_cfg_obj.get("gpu_type", "unknown"))
                else:
                    tags["provider.gpu_type"] = str(getattr(provider_train_cfg_obj, "gpu_type", "unknown"))  # type: ignore[unreachable]

        if "provider.name" not in tags:  # noqa: WPS226
            tags["provider.name"] = "unknown"  # noqa: WPS226

        self._p.set_tags(tags)

        params: dict[str, Any] = {
            "config.model.name": config.model.name,
            "config.model.device_map": config.model.device_map or "auto",
            "config.model.flash_attention": config.model.flash_attention,
            "config.training.type": config.training.type,
            "config.training.output_dir": "output",
        }

        hp = config.training.hyperparams
        for field, value in hp.model_dump(exclude_none=True).items():
            params[f"training.hyperparams.{field}"] = value
        params["training.hyperparams.load_in_4bit"] = config.training.get_effective_load_in_4bit()

        lora = None
        if config.training.type in ("lora", "qlora"):
            with contextlib.suppress(ValueError):
                lora = config.training.get_adapter_config()
        if lora is not None and config.training.type in ("lora", "qlora"):
            params["config.lora.r"] = lora.r
            params["config.lora.alpha"] = lora.lora_alpha
            params["config.lora.dropout"] = lora.lora_dropout
            params["config.lora.target_modules"] = str(lora.target_modules)
            if hasattr(lora, "use_dora"):
                params["config.lora.use_dora"] = lora.use_dora
            if hasattr(lora, "use_rslora"):
                params["config.lora.use_rslora"] = lora.use_rslora

        if strategies:
            for i, strategy in enumerate(strategies):
                params[f"config.strategy.{i}.type"] = strategy.strategy_type
                for field, value in strategy.hyperparams.model_dump(exclude_none=True).items():
                    params[f"config.strategy.{i}.hyperparams.{field}"] = value
                if strategy.dataset:
                    params[f"config.strategy.{i}.dataset"] = strategy.dataset

        self._p.log_params(params)

    def log_dataset_config(self, config: PipelineConfig) -> None:
        """Log dataset information for all strategies."""
        try:
            strategies = config.training.get_strategy_chain()
            if not strategies:
                logger.debug("[MLFLOW:DOMAIN] No strategies configured, skipping dataset logging")
                return

            dataset_names: set[str] = set()
            for strategy in strategies:
                ds_name = strategy.dataset or "default"
                if ds_name in config.datasets:
                    dataset_names.add(ds_name)

            if not dataset_names:
                logger.warning("[MLFLOW:DOMAIN] No valid datasets found in strategies")
                return

            params: dict[str, Any] = {}
            for ds_name in sorted(dataset_names):
                ds_cfg = config.datasets[ds_name]

                if ds_cfg.get_source_type() == "huggingface" and ds_cfg.source_hf is not None:
                    display_name = Path(ds_cfg.source_hf.train_id).name
                elif ds_cfg.source_local is not None:
                    display_name = Path(ds_cfg.source_local.local_paths.train).stem
                else:
                    display_name = ds_name

                params[f"dataset.{ds_name}.name"] = display_name
                params[f"dataset.{ds_name}.source_type"] = ds_cfg.get_source_type()
                params[f"dataset.{ds_name}.adapter_type"] = ds_cfg.adapter_type or "auto"

                if ds_cfg.get_source_type() == "huggingface" and ds_cfg.source_hf is not None:
                    params[f"dataset.{ds_name}.hf.train_id"] = ds_cfg.source_hf.train_id
                    if ds_cfg.source_hf.eval_id:
                        params[f"dataset.{ds_name}.hf.eval_id"] = ds_cfg.source_hf.eval_id
                elif ds_cfg.source_local is not None:
                    params[f"dataset.{ds_name}.local.train_path"] = ds_cfg.source_local.local_paths.train
                    if ds_cfg.source_local.local_paths.eval:
                        params[f"dataset.{ds_name}.local.eval_path"] = ds_cfg.source_local.local_paths.eval

                if ds_cfg.max_samples:
                    params[f"dataset.{ds_name}.max_samples"] = str(ds_cfg.max_samples)

            self._p.log_params(params)

            dataset_names_list = sorted(dataset_names)
            self._p.set_tags(
                {
                    "dataset.names": ",".join(dataset_names_list),
                    "dataset.count": str(len(dataset_names_list)),
                }
            )

        except Exception as e:
            logger.warning(f"[MLFLOW:DOMAIN] Failed to log dataset config: {e}")

    def log_provider_info(
        self,
        provider_name: str,
        provider_type: str,
        gpu_type: str | None = None,
        resource_id: str | None = None,
    ) -> None:
        """Log provider information as tags."""
        tags: dict[str, str] = {
            "provider.name": provider_name,  # noqa: WPS226
            "provider.type": provider_type,
        }
        if gpu_type:
            tags["provider.gpu_type"] = gpu_type
        if resource_id:
            tags["provider.resource_id"] = resource_id
        self._p.set_tags(tags)

    def log_strategy_info(self, strategy_type: str, phase_idx: int, total_phases: int) -> None:
        """Log strategy phase information."""
        self._p.set_tags(
            {
                "current_strategy": strategy_type,
                "current_phase": str(phase_idx),
                "total_phases": str(total_phases),
            }
        )

    # =========================================================================
    # CUSTOM METRICS
    # =========================================================================

    def log_gpu_metrics(
        self,
        gpu_memory_used_gb: float,
        gpu_memory_total_gb: float,
        gpu_utilization: float | None = None,
        step: int | None = None,
    ) -> None:
        """Log GPU memory and utilization metrics."""
        metrics: dict[str, float] = {
            "gpu_memory_used_gb": gpu_memory_used_gb,
            "gpu_memory_total_gb": gpu_memory_total_gb,
            "gpu_memory_pct": (gpu_memory_used_gb / gpu_memory_total_gb * 100) if gpu_memory_total_gb > 0 else 0,
        }
        if gpu_utilization is not None:
            metrics["gpu_utilization"] = gpu_utilization
        self._p.log_metrics(metrics, step=step)

    def log_throughput(self, tokens_per_second: float, samples_per_second: float, step: int | None = None) -> None:
        """Log training throughput metrics."""
        self._p.log_metrics(
            {
                "tokens_per_second": tokens_per_second,
                "samples_per_second": samples_per_second,
            },
            step=step,
        )

    # =========================================================================
    # MEMORY EVENTS
    # =========================================================================

    def log_gpu_detection(self, name: str, vram_gb: float, tier: str) -> None:
        """Log GPU detection event and params."""
        self._event_log.log_event(
            MLFLOW_SEVERITY_INFO,
            f"GPU detected: {name} ({vram_gb:.0f}GB)",
            category=MLFLOW_CATEGORY_MEMORY,
            source=MLFLOW_SOURCE_MEMORY_MANAGER,
            gpu_name=name,
            vram_gb=vram_gb,
            tier=tier,
        )
        self._p.log_params(
            {
                "gpu_name": name,
                "gpu_vram_gb": vram_gb,
                "gpu_tier": tier,
            }
        )

    def log_memory_warning(
        self,
        utilization_percent: float,
        used_mb: int,
        total_mb: int,
        is_critical: bool,
    ) -> None:
        """Log memory warning/critical event."""
        level = "critical" if is_critical else "warning"
        label = "CRITICAL" if is_critical else "WARNING"
        self._event_log.log_event(
            MLFLOW_SEVERITY_WARNING,
            f"Memory {label}: {utilization_percent:.0f}% VRAM used",
            category=MLFLOW_CATEGORY_MEMORY,
            source=MLFLOW_SOURCE_MEMORY_MANAGER,
            utilization_percent=utilization_percent,
            used_mb=used_mb,
            total_mb=total_mb,
            level=level,
        )

    def log_oom(self, operation: str, free_mb: int | None = None) -> None:
        """Log OOM error event."""
        msg = f"OOM during '{operation}'"
        if free_mb is not None:
            msg += f" (free: {free_mb}MB)"
        self._event_log.log_event(
            MLFLOW_SEVERITY_ERROR,
            msg,
            category=MLFLOW_CATEGORY_MEMORY,
            source=MLFLOW_SOURCE_MEMORY_MANAGER,
            operation=operation,
            free_mb=free_mb,
        )

    def log_oom_recovery(self, operation: str, attempt: int, max_attempts: int) -> None:
        """Log OOM recovery attempt."""
        self._event_log.log_event(
            MLFLOW_SEVERITY_WARNING,
            f"OOM recovery attempt {attempt}/{max_attempts} for '{operation}'",
            category=MLFLOW_CATEGORY_MEMORY,
            source=MLFLOW_SOURCE_MEMORY_MANAGER,
            operation=operation,
            attempt=attempt,
            max_attempts=max_attempts,
        )

    def log_cache_cleared(self, freed_mb: int) -> None:
        """Log cache cleared event."""
        if freed_mb > 0:
            self._event_log.log_event(
                MLFLOW_SEVERITY_INFO,
                f"Cache cleared: {freed_mb}MB freed",
                category="memory",
                source="MemoryManager",
                freed_mb=freed_mb,
            )

    def log_memory_snapshot(
        self,
        phase: str,
        used_mb: int,
        free_mb: int,
        total_mb: int,
        utilization_percent: float,
    ) -> None:
        """Log memory snapshot at key training phases."""
        self._event_log.log_event(
            MLFLOW_SEVERITY_INFO,
            f"Memory [{phase}]: {used_mb}MB / {total_mb}MB ({utilization_percent:.1f}%)",
            category=MLFLOW_CATEGORY_MEMORY,
            source=MLFLOW_SOURCE_MEMORY_MANAGER,
            phase=phase,
            used_mb=used_mb,
            free_mb=free_mb,
            total_mb=total_mb,
            utilization_percent=utilization_percent,
        )

    # =========================================================================
    # PIPELINE / DATA BUFFER EVENTS
    # =========================================================================

    def log_pipeline_initialized(
        self,
        run_id: str,
        total_phases: int,
        strategy_chain: list[str],
    ) -> None:
        """Log pipeline initialization event."""
        chain_str = " -> ".join(s.upper() for s in strategy_chain)
        self._event_log.log_event(
            "start",
            f"Pipeline initialized: {chain_str}",
            category="training",
            source="DataBuffer",
            run_id=run_id,
            total_phases=total_phases,
            strategy_chain=strategy_chain,
        )

    def log_state_saved(self, run_id: str, path: str) -> None:
        """Log state save event."""
        self._event_log.log_event(
            "checkpoint",
            "Pipeline state saved",
            category="training",
            source="DataBuffer",
            run_id=run_id,
            path=path,
        )

    def log_checkpoint_cleanup(self, cleaned_count: int, freed_mb: int) -> None:
        """Log checkpoint cleanup event."""
        if cleaned_count > 0:
            self._event_log.log_event(
                "info",
                f"Cleaned {cleaned_count} old checkpoints (~{freed_mb}MB freed)",
                category="training",
                source="DataBuffer",
                cleaned_count=cleaned_count,
                freed_mb=freed_mb,
            )

    def log_stage_start(self, stage_name: str, stage_idx: int, total_stages: int) -> None:
        """Log pipeline stage start."""
        self._event_log.log_event(
            "start",
            f"Stage {stage_idx + 1}/{total_stages}: {stage_name} started",
            category="pipeline",
            source=stage_name,
            stage_idx=stage_idx,
            total_stages=total_stages,
        )

    def log_stage_complete(
        self,
        stage_name: str,
        stage_idx: int,
        duration_seconds: float | None = None,
    ) -> None:
        """Log pipeline stage completion."""
        msg = f"Stage {stage_idx + 1}: {stage_name} completed"
        if duration_seconds:
            msg += f" ({duration_seconds:.1f}s)"
        self._event_log.log_event(
            "complete",
            msg,
            category="pipeline",
            source=stage_name,
            stage_idx=stage_idx,
            duration_seconds=duration_seconds,
        )

    def log_stage_failed(self, stage_name: str, stage_idx: int, error: str) -> None:
        """Log pipeline stage failure."""
        self._event_log.log_event(
            "error",
            f"Stage {stage_idx + 1}: {stage_name} failed: {error}",
            category="pipeline",
            source=stage_name,
            stage_idx=stage_idx,
            error=error,
        )

    # =========================================================================
    # ENVIRONMENT
    # =========================================================================

    def log_environment(self, env_snapshot: dict[str, Any] | None = None) -> None:
        """
        Log environment information (library versions, GPU info).

        Args:
            env_snapshot: Pre-collected snapshot. If None, collects automatically.
        """
        if env_snapshot is None:
            try:
                import platform
                import sys

                import torch as torch_module

                torch: Any = torch_module
                env_snapshot = {
                    "python_version": sys.version.split()[0],
                    "platform": platform.platform(),
                    "torch_version": torch.__version__,
                    "cuda_available": torch.cuda.is_available(),
                    "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
                }

                try:
                    import transformers as transformers_module
                except ImportError:
                    transformers_module = None  # type: ignore[assignment]
                if transformers_module is not None:
                    env_snapshot["transformers_version"] = transformers_module.__version__

                try:
                    import peft as peft_module
                except ImportError:
                    peft_module = None  # type: ignore[assignment]
                if peft_module is not None:
                    env_snapshot["peft_version"] = peft_module.__version__

                try:
                    import trl as trl_module
                except ImportError:
                    trl_module = None  # type: ignore[assignment]
                if trl_module is not None:
                    env_snapshot["trl_version"] = trl_module.__version__

            except Exception as e:
                logger.warning(f"[MLFLOW:DOMAIN] Failed to collect environment info: {e}")
                return

        params: dict[str, Any] = {}
        for key, value in env_snapshot.items():
            if value is not None:
                params[f"env.{key}"] = str(value)

        self._p.log_params(params)
        logger.debug(f"[MLFLOW:DOMAIN] Logged environment: {len(params)} params")


__all__ = ["MLflowDomainLogger"]
