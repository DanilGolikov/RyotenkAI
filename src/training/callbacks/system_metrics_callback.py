"""System Metrics Callback for HuggingFace Trainer.

Logs GPU, VRAM, CPU, RAM metrics to MLflow during training.

Single source of truth for system metrics in this codebase. Writes
through ``mlflow.log_metrics`` and ``mlflow.set_tags`` — both are
monkey-patched by ``ResilientMLflowTransport`` so payloads are buffered
to ``MetricsBuffer`` on offline windows and replayed on recovery.

Behaviour:
- **Per-step** (``on_step_end``): emits ``gpu/{idx}/utilization``,
  ``gpu/{idx}/memory_used_gb``, ``gpu/{idx}/memory_percent``,
  ``gpu/{idx}/temperature``, ``cpu/percent``, ``ram/used_gb``,
  ``ram/percent``. Logs every step (no throttle) — at our typical
  1-3 s/step training that's ~0.3-1 Hz of metric data, well within
  MLflow comfort. Rank-zero only in DDP to avoid N-fold duplicates.
- **Once** (``on_train_begin``): sets static run tags
  ``system.gpu.count``, ``system.gpu.{i}.name``,
  ``system.gpu.{i}.memory_total_gb``, ``system.driver.version``,
  ``system.cpu.count``. Static info doesn't belong on time-series
  graphs.

Multi-GPU: iterates over all NVML devices, not just index 0.
"""

from __future__ import annotations

import contextlib
import math
from typing import TYPE_CHECKING, Any

from transformers import TrainerCallback

from src.utils.logger import get_logger

if TYPE_CHECKING:
    from transformers import TrainerControl, TrainerState, TrainingArguments

logger = get_logger(__name__)


class SystemMetricsCallback(TrainerCallback):
    """Logs system metrics (GPU, VRAM, CPU, RAM) to MLflow each step."""

    def __init__(self) -> None:
        self._mlflow: Any | None = None
        self._pynvml: Any | None = None
        self._gpu_handles: list[Any] = []
        self._gpu_count: int = 0
        self._psutil: Any | None = None
        self._pynvml_available = False
        self._psutil_available = False
        self._setup_done = False
        # First-failure logging guard so silent loops don't hide real
        # issues but we don't spam at WARNING every step on chronic
        # failures (e.g. NVML lost a device mid-training).
        self._warned_gpu = False
        self._warned_cpu = False
        self._warned_log = False

    # ------------------------------------------------------------------
    # Setup / teardown
    # ------------------------------------------------------------------

    def _setup(self) -> None:
        """Lazy-init monitoring libraries on first hook invocation."""
        if self._setup_done:
            return

        # MLflow
        try:
            import mlflow as mlflow_module
        except ImportError:
            mlflow_module = None  # type: ignore[assignment]
            logger.warning("MLflow not available, system metrics will not be logged")
        self._mlflow = mlflow_module

        # NVML — multi-GPU
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
                self._gpu_count = int(_pynvml.nvmlDeviceGetCount())
                self._gpu_handles = [_pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(self._gpu_count)]
                logger.debug(f"pynvml initialized — {self._gpu_count} GPU(s) visible")
            except (OSError, nvml_error) as e:  # type: ignore[misc]
                logger.debug(f"pynvml not available: {e}")
                self._pynvml_available = False

        # psutil — CPU + RAM
        try:
            import psutil as _psutil
        except ImportError:
            _psutil = None
            logger.debug("psutil not available")
            self._psutil_available = False
        else:
            self._psutil = _psutil
            self._psutil_available = True
            # Prime ``cpu_percent``: the first call returns ``0.0``
            # because there's no previous reference window. Discard it
            # here so the first real measurement in ``_get_cpu_metrics``
            # reflects actual usage.
            with contextlib.suppress(Exception):
                self._psutil.cpu_percent(interval=None)
            logger.debug("psutil initialized for CPU/RAM metrics")

        self._setup_done = True

    # ------------------------------------------------------------------
    # Metric collection
    # ------------------------------------------------------------------

    def _get_gpu_metrics(self) -> dict[str, float]:
        """Per-GPU metrics, keyed ``gpu/{idx}/<metric>``.

        Skips keys for failed reads (no misleading 0.0 / NaN injected
        into the payload — MLflow simply doesn't draw points for
        absent keys).
        """
        if not self._pynvml_available or not self._gpu_handles:
            return {}
        # Narrow ``self._pynvml`` from ``Any | None`` — ``_pynvml_available``
        # is set to True only after the import + ``nvmlInit`` both succeeded,
        # which means ``self._pynvml`` is the imported module reference.
        assert self._pynvml is not None

        out: dict[str, float] = {}
        nvml_error = getattr(self._pynvml, "NVMLError", RuntimeError)

        for idx, handle in enumerate(self._gpu_handles):
            prefix = f"gpu/{idx}"

            with contextlib.suppress(OSError, TypeError, ValueError, nvml_error):  # type: ignore[misc]
                util = self._pynvml.nvmlDeviceGetUtilizationRates(handle)
                out[f"{prefix}/utilization"] = float(util.gpu)

            with contextlib.suppress(OSError, TypeError, ValueError, nvml_error):  # type: ignore[misc]
                mem = self._pynvml.nvmlDeviceGetMemoryInfo(handle)
                used_gb = float(mem.used) / (1024**3)
                total_gb = float(mem.total) / (1024**3)
                out[f"{prefix}/memory_used_gb"] = used_gb
                out[f"{prefix}/memory_total_gb"] = total_gb
                if total_gb > 0:
                    out[f"{prefix}/memory_percent"] = (used_gb / total_gb) * 100

            with contextlib.suppress(OSError, TypeError, ValueError, nvml_error):  # type: ignore[misc]
                temp = self._pynvml.nvmlDeviceGetTemperature(handle, self._pynvml.NVML_TEMPERATURE_GPU)
                # Temperature can legitimately be 0 only if the sensor
                # reports it; treat genuine errors as missing instead.
                temp_f = float(temp)
                if not math.isnan(temp_f):
                    out[f"{prefix}/temperature"] = temp_f

        return out

    def _get_cpu_metrics(self) -> dict[str, float]:
        """CPU + RAM metrics, system-level."""
        if not self._psutil_available:
            return {}
        assert self._psutil is not None  # _psutil_available pins this

        out: dict[str, float] = {}
        try:
            out["cpu/percent"] = float(self._psutil.cpu_percent(interval=None))
            mem = self._psutil.virtual_memory()
            out["ram/used_gb"] = mem.used / (1024**3)
            out["ram/percent"] = float(mem.percent)
        except Exception as e:
            if not self._warned_cpu:
                logger.warning(f"psutil read failed (further failures suppressed): {e}")
                self._warned_cpu = True
        return out

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def on_train_begin(
        self,
        args: TrainingArguments,  # noqa: ARG002
        state: TrainerState,
        control: TrainerControl,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        """Set static system tags once per run (rank-0 only)."""
        self._setup()

        if not state.is_world_process_zero:
            return
        if self._mlflow is None:
            return

        tags: dict[str, str] = {}

        if self._pynvml_available and self._gpu_handles:
            assert self._pynvml is not None  # _pynvml_available pins this
            tags["system.gpu.count"] = str(self._gpu_count)
            for i, handle in enumerate(self._gpu_handles):
                with contextlib.suppress(Exception):
                    name = self._pynvml.nvmlDeviceGetName(handle)
                    if isinstance(name, bytes):
                        name = name.decode("utf-8", errors="replace")
                    tags[f"system.gpu.{i}.name"] = str(name)
                with contextlib.suppress(Exception):
                    mem = self._pynvml.nvmlDeviceGetMemoryInfo(handle)
                    tags[f"system.gpu.{i}.memory_total_gb"] = f"{mem.total / (1024**3):.1f}"
            with contextlib.suppress(Exception):
                drv = self._pynvml.nvmlSystemGetDriverVersion()
                if isinstance(drv, bytes):
                    drv = drv.decode("utf-8", errors="replace")
                tags["system.driver.version"] = str(drv)

        if self._psutil_available:
            assert self._psutil is not None  # _psutil_available pins this
            with contextlib.suppress(Exception):
                tags["system.cpu.count"] = str(self._psutil.cpu_count())

        if not tags:
            return

        try:
            self._mlflow.set_tags(tags)
        except Exception as e:
            logger.debug(f"Failed to set system tags: {e}")

    def on_step_end(
        self,
        args: TrainingArguments,  # noqa: ARG002
        state: TrainerState,
        control: TrainerControl,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        """Log system metrics every step (rank-0 only)."""
        self._setup()

        # DDP guard: avoid N-fold duplicates from non-zero ranks.
        if not state.is_world_process_zero:
            return

        if self._mlflow is None:
            return

        metrics: dict[str, float] = {}
        try:
            metrics.update(self._get_gpu_metrics())
        except Exception as e:
            if not self._warned_gpu:
                logger.warning(f"GPU metric collection failed (further failures suppressed): {e}")
                self._warned_gpu = True

        metrics.update(self._get_cpu_metrics())

        if not metrics:
            return

        try:
            self._mlflow.log_metrics(metrics, step=state.global_step)
        except Exception as e:
            if not self._warned_log:
                logger.warning(f"mlflow.log_metrics failed (further failures suppressed): {e}")
                self._warned_log = True

    def on_train_end(
        self,
        args: TrainingArguments,  # noqa: ARG002
        state: TrainerState,  # noqa: ARG002
        control: TrainerControl,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        """Best-effort NVML shutdown."""
        if self._pynvml_available and self._pynvml is not None:
            with contextlib.suppress(Exception):
                self._pynvml.nvmlShutdown()


__all__ = ["SystemMetricsCallback"]
