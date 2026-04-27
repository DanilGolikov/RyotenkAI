"""MLflow system-metrics collection settings (CPU / GPU / RAM).

Phase 14 follow-up — sub-block for MLflow system-metrics knobs by
analogy with :class:`~src.config.training.metrics_buffer.MetricsBufferConfig`.
Pre-refactor four flat fields lived directly on
:class:`~src.config.integrations.mlflow.MLflowConfig` /
:class:`~src.config.integrations.mlflow_integration.MLflowIntegrationConfig`:

    system_metrics_sampling_interval
    system_metrics_samples_before_logging
    system_metrics_callback_enabled
    system_metrics_callback_interval

Now grouped under a single ``system_metrics`` block with hardcoded
defaults — the operator still tunes them via YAML, but a fresh project
gets a working default without copy-pasting four lines.

Schema example (``experiment_tracking.mlflow.system_metrics`` block,
or the same path on the integration side):

.. code-block:: yaml

    experiment_tracking:
      mlflow:
        integration: my_mlflow
        experiment_name: my_run
        system_metrics:
          sampling_interval: 1            # seconds between samples
          samples_before_logging: 1       # batch size before flush
          callback_enabled: false         # HF Trainer SystemMetricsCallback
          callback_interval: 10           # log every N steps when callback on

Old configs with the flat fields fail with a clear migration hint
(see :data:`~src.config.integrations.experiment_tracking._LEGACY_MLFLOW_KEYS`).

Two distinct mechanisms are configured here:

1. **MLflow native sampling** (``sampling_interval`` +
   ``samples_before_logging``) — always active when MLflow is tracking
   system metrics; controls how often the background sampler reads
   ``psutil`` / ``nvidia-smi`` and how many samples it batches before
   pushing to the tracking backend.

2. **HF Trainer SystemMetricsCallback** (``callback_enabled`` +
   ``callback_interval``) — opt-in supplementary callback that may
   hang on some cloud GPU images. Default off because the native
   MLflow sampler covers the same ground without the hang risk.

Default tuning (1 s sampling) trades extra MLflow chatter for
high-fidelity dashboards. For very long production runs consider
raising ``sampling_interval`` to 5-10 s to reduce MLflow upstream
write volume.
"""

from __future__ import annotations

from pydantic import Field

from ..base import StrictBaseModel


class SystemMetricsConfig(StrictBaseModel):
    """Configures MLflow's system-metric (CPU / GPU / RAM) collection.

    All four knobs have sensible hardcoded defaults — operators only
    set the YAML block to override. Defaults match the previous
    inline values that lived on
    :class:`~src.config.integrations.mlflow_integration.MLflowIntegrationConfig`
    pre-refactor, except ``sampling_interval`` (was ``5`` → now ``1``)
    which prefers high-fidelity dashboards out-of-the-box; this is
    cheap because ``samples_before_logging=1`` flushes straight to
    the backend without buffering in-process.
    """

    sampling_interval: int = Field(
        default=1,
        ge=1,
        le=60,
        description=(
            "MLflow native system-metric sampling interval in "
            "seconds. Background sampler reads psutil / nvidia-smi "
            "this often. Default 1 s for dev-grade dashboards; raise "
            "to 5-10 s for long production runs to reduce MLflow "
            "upstream write volume."
        ),
    )
    samples_before_logging: int = Field(
        default=1,
        ge=1,
        le=10,
        description=(
            "Number of samples MLflow batches before flushing to the "
            "tracking backend. Default 1 = flush each sample as it "
            "arrives. Raise (e.g. 5) to batch and reduce upstream "
            "round-trips at the cost of dashboard latency."
        ),
    )
    callback_enabled: bool = Field(
        default=False,
        description=(
            "Enable the HF Trainer ``SystemMetricsCallback``. Off by "
            "default — known to hang on some cloud GPU images. The "
            "native MLflow sampler "
            "(``sampling_interval`` + ``samples_before_logging``) "
            "is sufficient for most runs; turn this on only if you "
            "specifically need step-aligned system metrics tied to "
            "trainer.global_step."
        ),
    )
    callback_interval: int = Field(
        default=10,
        ge=1,
        le=100,
        description=(
            "Log system metrics every N trainer steps when "
            "``callback_enabled=True``. Ignored when the callback "
            "is off. Default 10 mirrors HF's recommendation."
        ),
    )


__all__ = ["SystemMetricsConfig"]
