from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

import src.training.callbacks.gpu_metrics_callback as gm
from src.training.callbacks.gpu_metrics_callback import GPUMetricsCallback


@dataclass
class FakeMLflowManager:
    active: bool = True
    calls: list[tuple[dict[str, float], int]] = field(default_factory=list)

    @property
    def is_active(self) -> bool:
        return self.active

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        assert step is not None
        self.calls.append((metrics, step))


def test_noop_when_mlflow_inactive() -> None:
    mgr = FakeMLflowManager(active=False)
    cb = GPUMetricsCallback(mlflow_manager=mgr)

    state = SimpleNamespace(global_step=1)
    cb.on_step_end(args=object(), state=state, control=object())

    assert mgr.calls == []


def test_logs_cpu_ram_disk_and_gpu_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    mgr = FakeMLflowManager(active=True)
    cb = GPUMetricsCallback(mlflow_manager=mgr)

    # Force "GPU available" path deterministically
    monkeypatch.setattr(gm.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(gm.torch.cuda, "mem_get_info", lambda: (2 * 1024**2, 10 * 1024**2))

    monkeypatch.setattr(gm.subprocess, "check_output", lambda *a, **k: "75\n")

    monkeypatch.setattr(gm.psutil, "cpu_percent", lambda interval=None: 12.5)  # noqa: ARG005
    monkeypatch.setattr(gm.psutil, "virtual_memory", lambda: SimpleNamespace(used=3 * 1024**3, percent=33.3))
    monkeypatch.setattr(gm.psutil, "disk_usage", lambda p: SimpleNamespace(percent=55.5))  # noqa: ARG005

    state = SimpleNamespace(global_step=10)
    cb.on_step_end(args=object(), state=state, control=object())

    assert mgr.calls
    metrics, step = mgr.calls[-1]
    assert step == 10

    # Always logged (CPU/RAM/Disk)
    assert metrics["system/cpu_utilization_percentage"] == 12.5
    assert metrics["system/system_memory_usage_percentage"] == pytest.approx(33.3)
    assert metrics["system/disk_usage_percentage"] == pytest.approx(55.5)

    # GPU memory + util
    assert metrics["system/gpu_0_memory_usage_megabytes"] == pytest.approx(8.0)
    assert metrics["system/gpu_0_memory_usage_percentage"] == pytest.approx(80.0)
    assert metrics["system/gpu_0_utilization_percentage"] == pytest.approx(75.0)


def test_ignores_nvidia_smi_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    mgr = FakeMLflowManager(active=True)
    cb = GPUMetricsCallback(mlflow_manager=mgr)

    monkeypatch.setattr(gm.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(gm.torch.cuda, "mem_get_info", lambda: (5 * 1024**2, 10 * 1024**2))
    monkeypatch.setattr(gm.subprocess, "check_output", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))

    monkeypatch.setattr(gm.psutil, "cpu_percent", lambda interval=None: 1.0)  # noqa: ARG005
    monkeypatch.setattr(gm.psutil, "virtual_memory", lambda: SimpleNamespace(used=1, percent=1.0))
    monkeypatch.setattr(gm.psutil, "disk_usage", lambda p: SimpleNamespace(percent=1.0))  # noqa: ARG005

    state = SimpleNamespace(global_step=1)
    cb.on_step_end(args=object(), state=state, control=object())

    metrics, _ = mgr.calls[-1]
    assert "system/gpu_0_utilization_percentage" not in metrics


def test_cpu_only_path(monkeypatch: pytest.MonkeyPatch) -> None:
    mgr = FakeMLflowManager(active=True)
    cb = GPUMetricsCallback(mlflow_manager=mgr)

    monkeypatch.setattr(gm.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(gm.psutil, "cpu_percent", lambda interval=None: 7.0)  # noqa: ARG005
    monkeypatch.setattr(gm.psutil, "virtual_memory", lambda: SimpleNamespace(used=1, percent=7.0))
    monkeypatch.setattr(gm.psutil, "disk_usage", lambda p: SimpleNamespace(percent=9.0))  # noqa: ARG005

    state = SimpleNamespace(global_step=2)
    cb.on_step_end(args=object(), state=state, control=object())

    metrics, _ = mgr.calls[-1]
    assert "system/gpu_0_memory_usage_megabytes" not in metrics
    assert metrics["system/cpu_utilization_percentage"] == 7.0
