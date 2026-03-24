"""
System Metrics Callback for HuggingFace Trainer.

Logs GPU, VRAM, CPU metrics to MLflow during training.

Usage:
    from src.training.callbacks.system_metrics_callback import SystemMetricsCallback

    callback = SystemMetricsCallback(log_every_n_steps=10)
    trainer = SFTTrainer(..., callbacks=[callback])
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

from transformers import TrainerCallback

from src.utils.logger import get_logger

if TYPE_CHECKING:
    from transformers import TrainerControl, TrainerState, TrainingArguments

logger = get_logger(__name__)


class SystemMetricsCallback(TrainerCallback):
    """
    Callback to log system metrics (GPU, VRAM, CPU) to MLflow.

    Metrics logged:
    - gpu/utilization: GPU utilization %
    - gpu/memory_used_gb: VRAM used in GB
    - gpu/memory_total_gb: Total VRAM in GB
    - gpu/memory_percent: VRAM usage %
    - gpu/temperature: GPU temperature °C
    - cpu/percent: CPU utilization %
    - ram/used_gb: RAM used in GB
    - ram/percent: RAM usage %
    """

    def __init__(self, log_every_n_steps: int = 10):
        """
        Args:
            log_every_n_steps: Log metrics every N training steps
        """
        self.log_every_n_steps = log_every_n_steps
        self._mlflow: Any | None = None
        self._pynvml: Any | None = None
        self._gpu_handle: Any | None = None
        self._psutil: Any | None = None
        self._pynvml_available = False
        self._psutil_available = False
        self._setup_done = False

    def _setup(self) -> None:
        """Lazy setup of monitoring libraries."""
        if self._setup_done:
            return

        # Try to import MLflow
        try:
            import mlflow as mlflow_module
        except ImportError:
            mlflow_module = None  # type: ignore[assignment]
            logger.warning("MLflow not available, system metrics will not be logged")
        self._mlflow = mlflow_module

        # Try to import pynvml for GPU metrics
        try:
            import pynvml as _pynvml
        except ImportError as e:
            _pynvml = None
            logger.debug(f"pynvml not available: {e}")
            self._pynvml_available = False
        else:
            nvml_error = getattr(_pynvml, "NVMLError", RuntimeError)
            try:
                _pynvml.nvmlInit()
                self._pynvml = _pynvml
                self._pynvml_available = True
                self._gpu_handle = _pynvml.nvmlDeviceGetHandleByIndex(0)
                logger.debug("pynvml initialized for GPU metrics")
            except (OSError, nvml_error) as e:  # type: ignore[misc]
                logger.debug(f"pynvml not available: {e}")
                self._pynvml_available = False

        # Try to import psutil for CPU/RAM metrics
        try:
            import psutil as _psutil
        except ImportError:
            _psutil = None
            logger.debug("psutil not available")
            self._psutil_available = False
        else:
            self._psutil = _psutil
            self._psutil_available = True
            logger.debug("psutil initialized for CPU/RAM metrics")

        self._setup_done = True

    def _get_gpu_metrics(self) -> dict[str, float]:
        """Get GPU metrics using pynvml."""
        if not self._pynvml_available or not self._pynvml:
            return {}

        try:
            # GPU utilization
            util = self._pynvml.nvmlDeviceGetUtilizationRates(self._gpu_handle)  # type: ignore[union-attr]

            # Memory info
            mem_info = self._pynvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)  # type: ignore[union-attr]
            mem_used_gb = float(mem_info.used) / (1024**3)
            mem_total_gb = float(mem_info.total) / (1024**3)
            mem_percent = (float(mem_info.used) / float(mem_info.total)) * 100

            # Temperature
            nvml_error = getattr(self._pynvml, "NVMLError", RuntimeError)
            try:
                temp_raw = self._pynvml.nvmlDeviceGetTemperature(self._gpu_handle, self._pynvml.NVML_TEMPERATURE_GPU)  # type: ignore[union-attr]
                temp = float(temp_raw)
            except (OSError, TypeError, ValueError, nvml_error):  # type: ignore[misc]
                temp = 0.0

            return {
                "gpu/utilization": float(util.gpu),
                "gpu/memory_used_gb": mem_used_gb,
                "gpu/memory_total_gb": mem_total_gb,
                "gpu/memory_percent": mem_percent,
                "gpu/temperature": temp,
            }
        except Exception as e:
            logger.debug(f"Failed to get GPU metrics: {e}")
            return {}

    def _get_cpu_metrics(self) -> dict[str, float]:
        """Get CPU/RAM metrics using psutil."""
        if not self._psutil_available:
            return {}

        try:
            # CPU
            cpu_percent = self._psutil.cpu_percent(interval=None)  # type: ignore[union-attr]

            # RAM
            mem = self._psutil.virtual_memory()  # type: ignore[union-attr]
            ram_used_gb = mem.used / (1024**3)
            ram_percent = mem.percent

            return {
                "cpu/percent": cpu_percent,
                "ram/used_gb": ram_used_gb,
                "ram/percent": ram_percent,
            }
        except Exception as e:
            logger.debug(f"Failed to get CPU metrics: {e}")
            return {}

    def on_step_end(
        self,
        args: TrainingArguments,  # noqa: ARG002
        state: TrainerState,
        control: TrainerControl,  # noqa: ARG002
        **kwargs,  # noqa: ARG002
    ) -> None:
        """Log metrics at end of each step."""
        self._setup()

        # Only log every N steps
        if state.global_step % self.log_every_n_steps != 0:
            return

        # Skip if MLflow not available
        if self._mlflow is None:
            return

        # Collect metrics
        metrics: dict[str, float] = {}  # type: ignore[unreachable]
        metrics.update(self._get_gpu_metrics())
        metrics.update(self._get_cpu_metrics())

        # Log to MLflow
        if metrics:
            try:
                self._mlflow.log_metrics(metrics, step=state.global_step)
            except Exception as e:
                logger.debug(f"Failed to log system metrics: {e}")

    def on_train_end(
        self,
        args: TrainingArguments,  # noqa: ARG002
        state: TrainerState,  # noqa: ARG002
        control: TrainerControl,  # noqa: ARG002
        **kwargs,  # noqa: ARG002
    ) -> None:
        """Cleanup pynvml on training end."""
        if self._pynvml_available:
            with contextlib.suppress(Exception):
                self._pynvml.nvmlShutdown()  # type: ignore[union-attr]


__all__ = ["SystemMetricsCallback"]
