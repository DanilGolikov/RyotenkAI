"""Training callbacks for HuggingFace Trainer.

Phase 2 (ethereal-tumbling-patterson) removed
``TrainingEventsCallback``: it was a dual-path sibling to
:class:`~ryotenkai_pod.trainer.callbacks.runner_event_callback.RunnerEventCallback`
writing the same start/checkpoint/epoch events into MLflow directly,
which caused divergence when one channel failed. The new SSOT routes
through the runner's bus + journal; MLflow attribution will be
populated from the journal by ``MlflowFinalizer`` in Phase 6.
"""

from ryotenkai_pod.trainer.callbacks.system_metrics_callback import SystemMetricsCallback

__all__ = ["SystemMetricsCallback"]
