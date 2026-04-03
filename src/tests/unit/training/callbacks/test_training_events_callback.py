from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

import src.training.callbacks.training_events_callback as te
from src.training.callbacks.training_events_callback import TrainingEventsCallback


@dataclass
class FakeMLflowManager:
    active: bool = True
    calls: list[tuple[str, str, dict]] = field(default_factory=list)
    metrics_calls: list[tuple[dict[str, float], int | None]] = field(default_factory=list)

    @property
    def is_active(self) -> bool:
        return self.active

    def log_event_start(self, message: str, **kwargs) -> None:
        self.calls.append(("start", message, kwargs))

    def log_event_complete(self, message: str, **kwargs) -> None:
        self.calls.append(("complete", message, kwargs))

    def log_event_checkpoint(self, message: str, **kwargs) -> None:
        self.calls.append(("checkpoint", message, kwargs))

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        self.metrics_calls.append((metrics, step))


def test_on_train_begin_logs_event_when_active(monkeypatch: pytest.MonkeyPatch) -> None:
    mlflow_mgr = FakeMLflowManager(active=True)
    cb = TrainingEventsCallback(mlflow_manager=mlflow_mgr)

    monkeypatch.setattr(te.time, "time", lambda: 100.0)

    args = SimpleNamespace(num_train_epochs=3)
    state = SimpleNamespace(max_steps=123, global_step=0, epoch=0.0)
    control = object()

    cb.on_train_begin(args=args, state=state, control=control)

    assert mlflow_mgr.calls
    kind, msg, payload = mlflow_mgr.calls[-1]
    assert kind == "start"
    assert msg == "Training loop started"
    assert payload["category"] == "training"
    assert payload["source"] == "TrainingEventsCallback"
    assert payload["total_steps"] == 123
    assert payload["num_epochs"] == 3


def test_on_train_begin_noop_when_inactive(monkeypatch: pytest.MonkeyPatch) -> None:
    mlflow_mgr = FakeMLflowManager(active=False)
    cb = TrainingEventsCallback(mlflow_manager=mlflow_mgr)

    monkeypatch.setattr(te.time, "time", lambda: 100.0)

    args = SimpleNamespace(num_train_epochs=1)
    state = SimpleNamespace(max_steps=10, global_step=0, epoch=0.0)
    control = object()

    cb.on_train_begin(args=args, state=state, control=control)

    assert mlflow_mgr.calls == []


def test_on_epoch_begin_sets_epoch_and_logs(monkeypatch: pytest.MonkeyPatch) -> None:
    mlflow_mgr = FakeMLflowManager(active=True)
    cb = TrainingEventsCallback(mlflow_manager=mlflow_mgr)

    monkeypatch.setattr(te.time, "time", lambda: 200.0)

    args = SimpleNamespace(num_train_epochs=5)
    state = SimpleNamespace(global_step=0, epoch=0.0)
    control = object()

    cb.on_epoch_begin(args=args, state=state, control=control)

    assert cb._current_epoch == 1
    kind, msg, payload = mlflow_mgr.calls[-1]
    assert kind == "start"
    assert msg == "Epoch 1/5 started"
    assert payload["epoch"] == 1
    assert payload["total_epochs"] == 5


def test_on_epoch_end_logs_complete_and_metric(monkeypatch: pytest.MonkeyPatch) -> None:
    mlflow_mgr = FakeMLflowManager(active=True)
    cb = TrainingEventsCallback(mlflow_manager=mlflow_mgr)

    times = iter([300.0, 305.5])
    monkeypatch.setattr(te.time, "time", lambda: next(times))

    args = SimpleNamespace(num_train_epochs=2)
    state = SimpleNamespace(global_step=42, epoch=0.0)
    control = object()

    cb.on_epoch_begin(args=args, state=state, control=control)
    cb.on_epoch_end(args=args, state=state, control=control)

    # epoch complete event
    kind, msg, payload = mlflow_mgr.calls[-1]
    assert kind == "complete"
    assert "Epoch 1 completed" in msg
    assert payload["epoch"] == 1
    assert payload["global_step"] == 42
    assert payload["epoch_duration_seconds"] == pytest.approx(5.5)

    # epoch duration metric
    assert mlflow_mgr.metrics_calls
    metrics, step = mlflow_mgr.metrics_calls[-1]
    assert metrics["epoch_duration_seconds"] == pytest.approx(5.5)
    assert step == 1


def test_on_train_end_logs_total_duration(monkeypatch: pytest.MonkeyPatch) -> None:
    mlflow_mgr = FakeMLflowManager(active=True)
    cb = TrainingEventsCallback(mlflow_manager=mlflow_mgr)

    times = iter([10.0, 25.0])
    monkeypatch.setattr(te.time, "time", lambda: next(times))

    args = SimpleNamespace(num_train_epochs=1)
    state = SimpleNamespace(max_steps=10, global_step=7, epoch=0.25)
    control = object()

    cb.on_train_begin(args=args, state=state, control=control)
    cb.on_train_end(args=args, state=state, control=control)

    kind, msg, payload = mlflow_mgr.calls[-1]
    assert kind == "complete"
    assert "Training loop completed" in msg
    assert payload["final_step"] == 7
    assert payload["final_epoch"] == 0.25
    assert payload["total_duration_seconds"] == pytest.approx(15.0)


def test_on_save_logs_checkpoint() -> None:
    mlflow_mgr = FakeMLflowManager(active=True)
    cb = TrainingEventsCallback(mlflow_manager=mlflow_mgr)

    args = SimpleNamespace(num_train_epochs=1)
    state = SimpleNamespace(global_step=99, epoch=1.0)
    control = object()

    cb.on_save(args=args, state=state, control=control)

    kind, msg, payload = mlflow_mgr.calls[-1]
    assert kind == "checkpoint"
    assert msg == "Checkpoint saved at step 99"
    assert payload["step"] == 99
    assert payload["epoch"] == 1.0


def test_on_train_end_noop_when_never_started() -> None:
    mlflow_mgr = FakeMLflowManager(active=True)
    cb = TrainingEventsCallback(mlflow_manager=mlflow_mgr)

    args = SimpleNamespace(num_train_epochs=1)
    state = SimpleNamespace(global_step=1, epoch=0.0)
    control = object()

    cb.on_train_end(args=args, state=state, control=control)

    assert mlflow_mgr.calls == []
