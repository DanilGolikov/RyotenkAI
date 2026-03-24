from __future__ import annotations

import builtins
import sys
from types import ModuleType, SimpleNamespace

import pytest

from src.training.callbacks.system_metrics_callback import SystemMetricsCallback


def _install_fake_module(monkeypatch: pytest.MonkeyPatch, name: str, module: ModuleType) -> None:
    monkeypatch.setitem(sys.modules, name, module)


def test_on_step_end_logs_metrics_when_deps_available(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[dict[str, float], int]] = []

    # Fake mlflow
    mlflow = ModuleType("mlflow")

    def log_metrics(metrics: dict[str, float], step: int) -> None:
        calls.append((metrics, step))

    mlflow.log_metrics = log_metrics  # type: ignore[attr-defined]
    _install_fake_module(monkeypatch, "mlflow", mlflow)

    # Fake pynvml (GPU)
    shutdown_called = {"value": False}
    pynvml = ModuleType("pynvml")
    pynvml.NVML_TEMPERATURE_GPU = 0  # type: ignore[attr-defined]

    def nvmlInit() -> None:  # noqa: N802
        return None

    def nvmlDeviceGetHandleByIndex(idx: int):  # noqa: N802
        return object()

    def nvmlDeviceGetUtilizationRates(handle):  # noqa: N802
        return SimpleNamespace(gpu=55)

    def nvmlDeviceGetMemoryInfo(handle):  # noqa: N802
        return SimpleNamespace(used=2 * 1024**3, total=8 * 1024**3)

    def nvmlDeviceGetTemperature(handle, sensor):  # noqa: N802
        raise RuntimeError("no temp sensor")

    def nvmlShutdown() -> None:  # noqa: N802
        shutdown_called["value"] = True

    pynvml.nvmlInit = nvmlInit  # type: ignore[attr-defined]
    pynvml.nvmlDeviceGetHandleByIndex = nvmlDeviceGetHandleByIndex  # type: ignore[attr-defined]
    pynvml.nvmlDeviceGetUtilizationRates = nvmlDeviceGetUtilizationRates  # type: ignore[attr-defined]
    pynvml.nvmlDeviceGetMemoryInfo = nvmlDeviceGetMemoryInfo  # type: ignore[attr-defined]
    pynvml.nvmlDeviceGetTemperature = nvmlDeviceGetTemperature  # type: ignore[attr-defined]
    pynvml.nvmlShutdown = nvmlShutdown  # type: ignore[attr-defined]
    _install_fake_module(monkeypatch, "pynvml", pynvml)

    # Fake psutil (CPU/RAM)
    psutil = ModuleType("psutil")

    def cpu_percent(interval=None):
        return 12.5

    def virtual_memory():
        return SimpleNamespace(used=3 * 1024**3, percent=33.3)

    psutil.cpu_percent = cpu_percent  # type: ignore[attr-defined]
    psutil.virtual_memory = virtual_memory  # type: ignore[attr-defined]
    _install_fake_module(monkeypatch, "psutil", psutil)

    cb = SystemMetricsCallback(log_every_n_steps=2)
    state = SimpleNamespace(global_step=2)

    cb.on_step_end(args=object(), state=state, control=object())

    assert calls, "Expected mlflow.log_metrics to be called"
    metrics, step = calls[-1]
    assert step == 2

    # GPU metrics
    assert metrics["gpu/utilization"] == 55.0
    assert metrics["gpu/memory_used_gb"] == pytest.approx(2.0)
    assert metrics["gpu/memory_total_gb"] == pytest.approx(8.0)
    assert metrics["gpu/memory_percent"] == pytest.approx(25.0)
    assert metrics["gpu/temperature"] == 0.0  # temp failure is tolerated

    # CPU/RAM metrics
    assert metrics["cpu/percent"] == 12.5
    assert metrics["ram/used_gb"] == pytest.approx(3.0)
    assert metrics["ram/percent"] == pytest.approx(33.3)

    cb.on_train_end(args=object(), state=state, control=object())
    assert shutdown_called["value"] is True


def test_on_step_end_skips_when_not_on_interval(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[dict[str, float], int]] = []

    mlflow = ModuleType("mlflow")

    def log_metrics(metrics: dict[str, float], step: int) -> None:
        calls.append((metrics, step))

    mlflow.log_metrics = log_metrics  # type: ignore[attr-defined]
    _install_fake_module(monkeypatch, "mlflow", mlflow)

    # Ensure psutil provides something (so metrics would be non-empty if run)
    psutil = ModuleType("psutil")
    psutil.cpu_percent = lambda interval=None: 1.0  # type: ignore[attr-defined, ARG005]
    psutil.virtual_memory = lambda: SimpleNamespace(used=1, percent=1.0)  # type: ignore[attr-defined]
    _install_fake_module(monkeypatch, "psutil", psutil)

    cb = SystemMetricsCallback(log_every_n_steps=10)
    state = SimpleNamespace(global_step=1)

    cb.on_step_end(args=object(), state=state, control=object())
    assert calls == []


def test_setup_tolerates_missing_mlflow(monkeypatch: pytest.MonkeyPatch) -> None:
    # Force ImportError only for mlflow
    real_import = builtins.__import__

    def fake_import(name: str, *args, **kwargs):
        if name == "mlflow":
            raise ImportError("mlflow missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    cb = SystemMetricsCallback(log_every_n_steps=1)
    cb._setup()

    assert cb._mlflow is None
