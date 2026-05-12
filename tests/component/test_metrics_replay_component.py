"""L2 component test для :class:`BufferedMetricsReplay`.

SUT: :class:`BufferedMetricsReplay` из ``model_retriever.metrics_replay``.
Все коллабораторы заменены каноническими fakes — здесь это маленький
адаптер поверх ``FakeMLflowManager``, который выставляет узкий
``log_metric``-Protocol, ожидаемый SUT.

Покрываем 3 пути:

* happy — буфер с двумя метриками реплеится в MLflow целиком.
* negative — MLflow поднимает ошибку на первом log_metric, replay
  продолжается, ошибки записываются.
* boundary — буфер пустой / отсутствует → replay no-op'ит без
  ошибок.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ryotenkai_control.pipeline.stages.model_retriever.metrics_replay import (
    BufferedMetricsReplay,
)
from tests._fakes.mlflow import FakeMLflowManager

pytestmark = pytest.mark.component


# --- адаптер: узкий ``log_metric``-Protocol поверх FakeMLflowManager --------


class _MlflowLogMetricAdapter:
    """Thin wrapper: SUT ожидает ``log_metric(run_id, key, value, timestamp,
    step)`` сигнатуру. ``FakeMLflowManager.log_metrics`` принимает dict —
    адаптируем за один вызов. Все остальные методы прокидываются как есть,
    но SUT их не зовёт."""

    def __init__(self, mgr: FakeMLflowManager, *, fail_on_keys: set[str] | None = None) -> None:
        self._mgr = mgr
        self._fail_on_keys = fail_on_keys or set()
        self.calls: list[dict[str, Any]] = []

    def log_metric(
        self,
        *,
        run_id: str,
        key: str,
        value: float,
        step: int | None = None,
        timestamp: int | None = None,
    ) -> None:
        self.calls.append({"run_id": run_id, "key": key, "value": value, "step": step, "timestamp": timestamp})
        if key in self._fail_on_keys:
            raise RuntimeError(f"injected log_metric failure for {key}")
        # Пишем в FakeMLflowManager, через adopt существующего run'а
        self._mgr.adopt_existing_run(run_id)
        self._mgr.log_metrics({key: value}, step=step)


# --- fixtures ----------------------------------------------------------------


def _seed_buffer(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")


# --- tests -------------------------------------------------------------------


class TestPositive:
    def test_happy_path_replays_all_entries(self, tmp_path: Path) -> None:
        mlflow = FakeMLflowManager()
        mlflow.setup()
        with mlflow.start_run() as run:
            run_id = run.info.run_id
        adapter = _MlflowLogMetricAdapter(mlflow)
        replay = BufferedMetricsReplay(adapter)

        buffer = tmp_path / "metrics_buffer.jsonl"
        _seed_buffer(
            buffer,
            [
                {"key": "loss", "value": 0.5, "step": 1, "timestamp": 1.0},
                {"key": "loss", "value": 0.3, "step": 2, "timestamp": 2.0},
            ],
        )

        result = replay.replay(buffer_path=buffer, run_id=run_id)

        assert result.replayed == 2
        assert result.failed == 0
        assert result.skipped == 0
        assert result.first_step == 1
        assert result.last_step == 2
        # adapter получил два вызова log_metric с правильными значениями
        assert [c["value"] for c in adapter.calls] == [0.5, 0.3]


class TestNegative:
    def test_mlflow_log_metric_raises_replay_continues(self, tmp_path: Path) -> None:
        """SUT гарантирует best-effort: одна неудачная запись не должна
        останавливать остальные."""
        mlflow = FakeMLflowManager()
        mlflow.setup()
        with mlflow.start_run() as run:
            run_id = run.info.run_id
        # На ключе ``bad`` адаптер бросает — SUT должен записать ошибку,
        # но продолжить с ``good``.
        adapter = _MlflowLogMetricAdapter(mlflow, fail_on_keys={"bad"})
        replay = BufferedMetricsReplay(adapter)

        buffer = tmp_path / "metrics_buffer.jsonl"
        _seed_buffer(
            buffer,
            [
                {"key": "bad", "value": 1.0, "step": 1, "timestamp": 1.0},
                {"key": "good", "value": 2.0, "step": 2, "timestamp": 2.0},
            ],
        )

        result = replay.replay(buffer_path=buffer, run_id=run_id)

        assert result.replayed == 1
        assert result.failed == 1
        assert len(result.errors) == 1
        assert "bad" in result.errors[0]


class TestBoundary:
    def test_missing_buffer_is_no_op(self, tmp_path: Path) -> None:
        mlflow = FakeMLflowManager()
        adapter = _MlflowLogMetricAdapter(mlflow)
        replay = BufferedMetricsReplay(adapter)

        result = replay.replay(buffer_path=tmp_path / "nope.jsonl", run_id="r-0001")

        assert result.replayed == 0
        assert result.failed == 0
        assert adapter.calls == []

    def test_empty_buffer_is_no_op(self, tmp_path: Path) -> None:
        mlflow = FakeMLflowManager()
        adapter = _MlflowLogMetricAdapter(mlflow)
        replay = BufferedMetricsReplay(adapter)

        buffer = tmp_path / "empty.jsonl"
        buffer.write_text("")

        result = replay.replay(buffer_path=buffer, run_id="r-0001")

        assert result.replayed == 0
        assert result.failed == 0
        assert adapter.calls == []
