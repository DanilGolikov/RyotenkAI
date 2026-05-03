"""Tests for :class:`SystemMetricsCallback`.

Covers the post-refactor contract: no constructor args, multi-GPU
iteration with ``gpu/{idx}/*`` namespace, rank-0 guard, CPU prime,
missing-metric omission (no fake 0.0), and static tags via
``on_train_begin``.
"""

from __future__ import annotations

import builtins
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from ryotenkai_pod.trainer.callbacks.system_metrics_callback import SystemMetricsCallback

# ---------------------------------------------------------------------------
# Test fixtures: fake mlflow / pynvml / psutil modules
# ---------------------------------------------------------------------------


def _install_fake_module(monkeypatch: pytest.MonkeyPatch, name: str, module: ModuleType) -> None:
    monkeypatch.setitem(sys.modules, name, module)


def _make_fake_mlflow(
    log_metrics_calls: list[tuple[dict[str, float], int]] | None = None,
    set_tags_calls: list[dict[str, str]] | None = None,
) -> ModuleType:
    mlflow = ModuleType("mlflow")

    if log_metrics_calls is not None:
        def log_metrics(metrics: dict[str, float], step: int) -> None:
            log_metrics_calls.append((metrics, step))
        mlflow.log_metrics = log_metrics  # type: ignore[attr-defined]
    else:
        mlflow.log_metrics = lambda *a, **k: None  # type: ignore[attr-defined]  # noqa: ARG005

    if set_tags_calls is not None:
        def set_tags(tags: dict[str, str]) -> None:
            set_tags_calls.append(dict(tags))
        mlflow.set_tags = set_tags  # type: ignore[attr-defined]
    else:
        mlflow.set_tags = lambda *a, **k: None  # type: ignore[attr-defined]  # noqa: ARG005

    return mlflow


def _make_fake_pynvml(
    *,
    gpu_count: int = 1,
    util: float = 55.0,
    used_gb: float = 2.0,
    total_gb: float = 8.0,
    temp: float | None = 67.0,
    name: str | bytes = "Tesla T4",
    driver: str | bytes = "535.104.05",
    raise_temp: bool = False,
    shutdown_flag: dict[str, bool] | None = None,
    cpu_prime_flag: dict[str, int] | None = None,
) -> ModuleType:
    pynvml = ModuleType("pynvml")
    pynvml.NVML_TEMPERATURE_GPU = 0  # type: ignore[attr-defined]

    def nvmlInit() -> None:  # noqa: N802
        return None

    def nvmlDeviceGetCount() -> int:  # noqa: N802
        return gpu_count

    def nvmlDeviceGetHandleByIndex(idx: int) -> object:  # noqa: N802
        # Use the index itself as the handle so callers can disambiguate.
        return idx

    def nvmlDeviceGetUtilizationRates(handle: Any) -> Any:  # noqa: N802
        return SimpleNamespace(gpu=util, memory=10)

    def nvmlDeviceGetMemoryInfo(handle: Any) -> Any:  # noqa: N802
        return SimpleNamespace(used=int(used_gb * 1024**3), total=int(total_gb * 1024**3))

    def nvmlDeviceGetTemperature(handle: Any, sensor: int) -> float:  # noqa: N802
        if raise_temp:
            raise RuntimeError("no temp sensor")
        return float(temp) if temp is not None else 0.0

    def nvmlDeviceGetName(handle: Any) -> str | bytes:  # noqa: N802
        return name

    def nvmlSystemGetDriverVersion() -> str | bytes:  # noqa: N802
        return driver

    def nvmlShutdown() -> None:  # noqa: N802
        if shutdown_flag is not None:
            shutdown_flag["value"] = True

    pynvml.nvmlInit = nvmlInit  # type: ignore[attr-defined]
    pynvml.nvmlDeviceGetCount = nvmlDeviceGetCount  # type: ignore[attr-defined]
    pynvml.nvmlDeviceGetHandleByIndex = nvmlDeviceGetHandleByIndex  # type: ignore[attr-defined]
    pynvml.nvmlDeviceGetUtilizationRates = nvmlDeviceGetUtilizationRates  # type: ignore[attr-defined]
    pynvml.nvmlDeviceGetMemoryInfo = nvmlDeviceGetMemoryInfo  # type: ignore[attr-defined]
    pynvml.nvmlDeviceGetTemperature = nvmlDeviceGetTemperature  # type: ignore[attr-defined]
    pynvml.nvmlDeviceGetName = nvmlDeviceGetName  # type: ignore[attr-defined]
    pynvml.nvmlSystemGetDriverVersion = nvmlSystemGetDriverVersion  # type: ignore[attr-defined]
    pynvml.nvmlShutdown = nvmlShutdown  # type: ignore[attr-defined]
    return pynvml


def _make_fake_psutil(
    *,
    cpu_percent_seq: list[float] | None = None,
    used_gb: float = 3.0,
    percent: float = 33.3,
    cpu_count: int = 16,
    cpu_prime_calls: list[float | None] | None = None,
) -> ModuleType:
    psutil = ModuleType("psutil")

    seq = list(cpu_percent_seq) if cpu_percent_seq is not None else [12.5]
    seq_iter = iter(seq) if seq else iter([12.5])
    cpu_count_value = cpu_count  # capture before defining the inner func that shadows the name

    def cpu_percent(interval: float | None = None) -> float:
        if cpu_prime_calls is not None:
            cpu_prime_calls.append(interval)
        try:
            return next(seq_iter)
        except StopIteration:
            return seq[-1] if seq else 0.0

    def virtual_memory() -> Any:
        return SimpleNamespace(used=int(used_gb * 1024**3), percent=percent)

    def fake_cpu_count() -> int:
        return cpu_count_value

    psutil.cpu_percent = cpu_percent  # type: ignore[attr-defined]
    psutil.virtual_memory = virtual_memory  # type: ignore[attr-defined]
    psutil.cpu_count = fake_cpu_count  # type: ignore[attr-defined]
    return psutil


def _state(global_step: int = 1, *, is_zero: bool = True) -> SimpleNamespace:
    return SimpleNamespace(global_step=global_step, is_world_process_zero=is_zero)


# ---------------------------------------------------------------------------
# 1. Constructor — no args
# ---------------------------------------------------------------------------


class TestConstructor:
    def test_no_args(self) -> None:
        # Pin: post-refactor constructor takes no arguments. Removing
        # ``log_every_n_steps`` here ensures TrainerFactory's bare
        # ``SystemMetricsCallback()`` keeps working.
        cb = SystemMetricsCallback()
        assert cb is not None


# ---------------------------------------------------------------------------
# 2. Per-step metrics
# ---------------------------------------------------------------------------


class TestOnStepEnd:
    def test_logs_full_metrics_on_step_end(self, monkeypatch: pytest.MonkeyPatch) -> None:
        log_calls: list[tuple[dict[str, float], int]] = []
        _install_fake_module(monkeypatch, "mlflow", _make_fake_mlflow(log_calls))
        _install_fake_module(monkeypatch, "pynvml", _make_fake_pynvml())
        _install_fake_module(monkeypatch, "psutil", _make_fake_psutil())

        cb = SystemMetricsCallback()
        cb.on_step_end(args=object(), state=_state(global_step=5), control=object())

        assert log_calls, "Expected mlflow.log_metrics to be called"
        metrics, step = log_calls[-1]
        assert step == 5
        # Multi-GPU namespace: gpu/{idx}/*
        assert metrics["gpu/0/utilization"] == 55.0
        assert metrics["gpu/0/memory_used_gb"] == pytest.approx(2.0)
        assert metrics["gpu/0/memory_total_gb"] == pytest.approx(8.0)
        assert metrics["gpu/0/memory_percent"] == pytest.approx(25.0)
        assert metrics["gpu/0/temperature"] == 67.0
        # CPU/RAM
        assert metrics["cpu/percent"] == 12.5
        assert metrics["ram/used_gb"] == pytest.approx(3.0)
        assert metrics["ram/percent"] == pytest.approx(33.3)

    def test_logs_every_step_no_throttle(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Pin: ``callback_interval`` knob was removed — every
        # ``on_step_end`` invocation results in a log call.
        log_calls: list[tuple[dict[str, float], int]] = []
        _install_fake_module(monkeypatch, "mlflow", _make_fake_mlflow(log_calls))
        _install_fake_module(monkeypatch, "pynvml", _make_fake_pynvml())
        _install_fake_module(monkeypatch, "psutil", _make_fake_psutil(cpu_percent_seq=[1.0, 2.0, 3.0]))

        cb = SystemMetricsCallback()
        for s in (1, 2, 3):
            cb.on_step_end(args=object(), state=_state(global_step=s), control=object())

        assert [step for _, step in log_calls] == [1, 2, 3]


# ---------------------------------------------------------------------------
# 3. Multi-GPU iteration
# ---------------------------------------------------------------------------


class TestMultiGPU:
    def test_iterates_over_all_gpus(self, monkeypatch: pytest.MonkeyPatch) -> None:
        log_calls: list[tuple[dict[str, float], int]] = []
        _install_fake_module(monkeypatch, "mlflow", _make_fake_mlflow(log_calls))
        _install_fake_module(monkeypatch, "pynvml", _make_fake_pynvml(gpu_count=4))
        _install_fake_module(monkeypatch, "psutil", _make_fake_psutil())

        cb = SystemMetricsCallback()
        cb.on_step_end(args=object(), state=_state(), control=object())

        metrics, _ = log_calls[-1]
        # Every GPU index must show up
        for i in range(4):
            assert f"gpu/{i}/utilization" in metrics
            assert f"gpu/{i}/memory_used_gb" in metrics


# ---------------------------------------------------------------------------
# 4. Rank-0 guard (DDP)
# ---------------------------------------------------------------------------


class TestRankGuard:
    def test_non_zero_rank_does_not_log_metrics(self, monkeypatch: pytest.MonkeyPatch) -> None:
        log_calls: list[tuple[dict[str, float], int]] = []
        _install_fake_module(monkeypatch, "mlflow", _make_fake_mlflow(log_calls))
        _install_fake_module(monkeypatch, "pynvml", _make_fake_pynvml())
        _install_fake_module(monkeypatch, "psutil", _make_fake_psutil())

        cb = SystemMetricsCallback()
        cb.on_step_end(args=object(), state=_state(is_zero=False), control=object())

        assert log_calls == []

    def test_non_zero_rank_does_not_set_tags(self, monkeypatch: pytest.MonkeyPatch) -> None:
        tag_calls: list[dict[str, str]] = []
        _install_fake_module(monkeypatch, "mlflow", _make_fake_mlflow(set_tags_calls=tag_calls))
        _install_fake_module(monkeypatch, "pynvml", _make_fake_pynvml())
        _install_fake_module(monkeypatch, "psutil", _make_fake_psutil())

        cb = SystemMetricsCallback()
        cb.on_train_begin(args=object(), state=_state(is_zero=False), control=object())

        assert tag_calls == []


# ---------------------------------------------------------------------------
# 5. CPU prime — first read is not a fake 0.0
# ---------------------------------------------------------------------------


class TestCPUPrime:
    def test_setup_primes_cpu_percent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Pin: ``_setup`` calls ``cpu_percent(interval=None)`` once to
        # discard the first-call 0.0. The next call (in
        # ``_get_cpu_metrics``) returns a real value.
        prime_calls: list[float | None] = []
        _install_fake_module(monkeypatch, "mlflow", _make_fake_mlflow())
        _install_fake_module(monkeypatch, "pynvml", _make_fake_pynvml())
        _install_fake_module(
            monkeypatch,
            "psutil",
            _make_fake_psutil(cpu_percent_seq=[0.0, 42.0], cpu_prime_calls=prime_calls),
        )

        cb = SystemMetricsCallback()
        cb._setup()

        # At least one priming call happened during setup.
        assert len(prime_calls) >= 1


# ---------------------------------------------------------------------------
# 6. Missing-metric handling — failed reads are OMITTED, not 0.0
# ---------------------------------------------------------------------------


class TestMissingMetrics:
    def test_temperature_failure_omits_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        log_calls: list[tuple[dict[str, float], int]] = []
        _install_fake_module(monkeypatch, "mlflow", _make_fake_mlflow(log_calls))
        _install_fake_module(monkeypatch, "pynvml", _make_fake_pynvml(raise_temp=True))
        _install_fake_module(monkeypatch, "psutil", _make_fake_psutil())

        cb = SystemMetricsCallback()
        cb.on_step_end(args=object(), state=_state(), control=object())

        metrics, _ = log_calls[-1]
        # Failed temp read must NOT inject a misleading 0.0 — the
        # key must be absent from the payload entirely.
        assert "gpu/0/temperature" not in metrics
        # Other GPU metrics still present.
        assert "gpu/0/utilization" in metrics


# ---------------------------------------------------------------------------
# 7. Static tags on train_begin
# ---------------------------------------------------------------------------


class TestStaticTags:
    def test_on_train_begin_sets_static_tags(self, monkeypatch: pytest.MonkeyPatch) -> None:
        tag_calls: list[dict[str, str]] = []
        _install_fake_module(monkeypatch, "mlflow", _make_fake_mlflow(set_tags_calls=tag_calls))
        _install_fake_module(monkeypatch, "pynvml", _make_fake_pynvml(gpu_count=2, name="A100", driver="535.1"))
        _install_fake_module(monkeypatch, "psutil", _make_fake_psutil(cpu_count=32))

        cb = SystemMetricsCallback()
        cb.on_train_begin(args=object(), state=_state(), control=object())

        assert len(tag_calls) == 1
        tags = tag_calls[0]
        assert tags["system.gpu.count"] == "2"
        assert tags["system.gpu.0.name"] == "A100"
        assert tags["system.gpu.1.name"] == "A100"
        assert tags["system.gpu.0.memory_total_gb"] == "8.0"
        assert tags["system.driver.version"] == "535.1"
        assert tags["system.cpu.count"] == "32"

    def test_on_train_begin_handles_bytes_returned_by_nvml(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Some NVML bindings return bytes; tags must be UTF-8 decoded
        # to plain strings before set_tags.
        tag_calls: list[dict[str, str]] = []
        _install_fake_module(monkeypatch, "mlflow", _make_fake_mlflow(set_tags_calls=tag_calls))
        _install_fake_module(
            monkeypatch,
            "pynvml",
            _make_fake_pynvml(name=b"H100", driver=b"550.54"),
        )
        _install_fake_module(monkeypatch, "psutil", _make_fake_psutil())

        cb = SystemMetricsCallback()
        cb.on_train_begin(args=object(), state=_state(), control=object())

        tags = tag_calls[0]
        assert tags["system.gpu.0.name"] == "H100"
        assert tags["system.driver.version"] == "550.54"


# ---------------------------------------------------------------------------
# 8. Lifecycle: shutdown
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_on_train_end_shuts_down_nvml(self, monkeypatch: pytest.MonkeyPatch) -> None:
        flag = {"value": False}
        _install_fake_module(monkeypatch, "mlflow", _make_fake_mlflow())
        _install_fake_module(monkeypatch, "pynvml", _make_fake_pynvml(shutdown_flag=flag))
        _install_fake_module(monkeypatch, "psutil", _make_fake_psutil())

        cb = SystemMetricsCallback()
        cb._setup()  # init pynvml
        cb.on_train_end(args=object(), state=_state(), control=object())

        assert flag["value"] is True


# ---------------------------------------------------------------------------
# 9. Graceful degradation — missing deps
# ---------------------------------------------------------------------------


class TestMissingDeps:
    def test_setup_tolerates_missing_mlflow(self, monkeypatch: pytest.MonkeyPatch) -> None:
        real_import = builtins.__import__

        def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "mlflow":
                raise ImportError("mlflow missing")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        cb = SystemMetricsCallback()
        cb._setup()
        assert cb._mlflow is None
