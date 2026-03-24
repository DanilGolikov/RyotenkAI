from __future__ import annotations


def test_callbacks_package_exports() -> None:
    from src.training.callbacks import SystemMetricsCallback, TrainingEventsCallback

    assert SystemMetricsCallback is not None
    assert TrainingEventsCallback is not None
