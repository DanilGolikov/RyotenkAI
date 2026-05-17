from __future__ import annotations


def test_callbacks_package_exports() -> None:
    """Phase 2 removed TrainingEventsCallback (dual-path with the unified
    event system). Only SystemMetricsCallback is re-exported now."""
    from ryotenkai_pod.trainer.callbacks import SystemMetricsCallback

    assert SystemMetricsCallback is not None
