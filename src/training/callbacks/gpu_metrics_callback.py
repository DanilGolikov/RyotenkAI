from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING, Any

import psutil as psutil_module
import torch as torch_module
from transformers import TrainerCallback

from src.utils.logger import logger

psutil: Any = psutil_module
torch: Any = torch_module

if TYPE_CHECKING:
    from transformers import TrainerControl, TrainerState, TrainingArguments


class GPUMetricsCallback(TrainerCallback):
    """
    Logs System metrics (GPU, CPU, RAM, Disk) to MLflow during training.
    This ensures phase-specific runs get their own hardware metrics.

    Metrics logged:
    - system/gpu_0_memory_usage_megabytes
    - system/gpu_0_memory_usage_percentage
    - system/gpu_0_utilization_percentage  <-- Added
    - system/cpu_utilization_percentage
    - system/system_memory_usage_megabytes
    - system/system_memory_usage_percentage
    - system/disk_usage_percentage
    """

    def __init__(self, mlflow_manager):
        self.mlflow_manager = mlflow_manager
        logger.debug("[GPU_CB:INIT] Initialized System Metrics (GPU+CPU) Callback")

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> Any:
        _ = args, control, kwargs
        if not self.mlflow_manager or not self.mlflow_manager.is_enabled:
            return

        try:
            metrics = {}

            # 1. GPU Memory (Lightweight check via torch)
            if torch.cuda.is_available():
                info = torch.cuda.mem_get_info()
                free = info[0] / 1024**2
                total = info[1] / 1024**2
                used = total - free
                util_pct = (used / total) * 100

                metrics.update(
                    {
                        "system/gpu_0_memory_usage_megabytes": used,
                        "system/gpu_0_memory_usage_percentage": util_pct,
                    }
                )

                # 1.1 GPU Utilization (via nvidia-smi subprocess)
                # Safest method to avoid pynvml hangs
                try:
                    result = subprocess.check_output(
                        ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                        encoding="utf-8",
                        timeout=0.5,  # 500ms max
                    )
                    # Take first line (first GPU)
                    gpu_util = float(result.strip().split("\n")[0])
                    metrics["system/gpu_0_utilization_percentage"] = gpu_util
                except Exception:
                    # Ignore if nvidia-smi missing or fails (best-effort metric)
                    pass

            # 2. CPU & RAM (Lightweight check via psutil)
            # CPU percent (interval=None is non-blocking but requires previous call,
            # first call returns 0.0, subsequent calls return avg since last call)
            metrics["system/cpu_utilization_percentage"] = psutil.cpu_percent(interval=None)

            vm = psutil.virtual_memory()
            metrics["system/system_memory_usage_megabytes"] = vm.used / 1024**2
            metrics["system/system_memory_usage_percentage"] = vm.percent

            # 3. Disk (Root volume)
            disk = psutil.disk_usage("/")
            metrics["system/disk_usage_percentage"] = disk.percent

            # Log to CURRENT active run (which is the Nested Phase Run)
            self.mlflow_manager.log_metrics(metrics, step=state.global_step)

            # Verbose logging only on specific steps
            if state.global_step % 100 == 0:
                logger.debug(
                    f"[SYSTEM_CB:LOG] step={state.global_step} cpu={metrics['system/cpu_utilization_percentage']}%"
                )

        except Exception as e:
            # Silent fail to not disrupt training, but log error once
            if state.global_step < 5:
                logger.warning(f"[SYSTEM_CB:ERROR] Failed to log system metrics: {e}")
