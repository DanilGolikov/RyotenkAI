"""MLflow system-metrics collection settings (CPU / GPU / RAM).

Single boolean knob. Pre-refactor four flat fields lived directly on
:class:`~src.config.integrations.mlflow.MLflowConfig` /
:class:`~src.config.integrations.mlflow_integration.MLflowIntegrationConfig`:

    system_metrics_sampling_interval     (controlled native MLflow sampler)
    system_metrics_samples_before_logging (controlled native MLflow sampler)
    system_metrics_callback_enabled       (controlled our HF Trainer callback)
    system_metrics_callback_interval      (step-throttle for our callback)

A previous follow-up grouped them into a nested ``system_metrics`` block.
This refactor further collapses the block to a single boolean: the
codebase no longer enables the native MLflow background sampler at all
(it bypassed ``ResilientMLflowTransport`` and dropped samples on offline
windows), and the trainer-side callback now logs every step (the throttle
was over-engineering for our typical 1-3 s/step training).

Schema example (``integrations.mlflow.system_metrics`` block, or
the same path on the integration side):

.. code-block:: yaml

    integrations:
      mlflow:
        integration: my_mlflow
        experiment_name: my_run
        system_metrics:
          callback_enabled: true   # default — HF Trainer SystemMetricsCallback

Old configs with the flat fields, or with the now-removed nested fields
(``sampling_interval``, ``samples_before_logging``, ``callback_interval``),
fail with a clear migration hint — see
:data:`~src.config.integrations.root._LEGACY_MLFLOW_KEYS`.

Mechanism
---------

The HF Trainer ``SystemMetricsCallback`` (registered by ``TrainerFactory``
when ``callback_enabled=True``) emits step-aligned ``gpu/{idx}/*`` /
``cpu/*`` / ``ram/*`` metrics through ``mlflow.log_metrics``. The
``ResilientMLflowTransport`` shim wraps that call and buffers payloads
to ``MetricsBuffer`` on offline windows, so metrics survive Mac-asleep /
network-blip events.
"""

from __future__ import annotations

from pydantic import Field

from ..base import StrictBaseModel


class SystemMetricsConfig(StrictBaseModel):
    """Configures the HF Trainer ``SystemMetricsCallback``.

    Single field. Operators only override via YAML when they need to
    disable the callback entirely (rare — only on cloud GPU images
    where ``pynvml.nvmlInit`` hangs).
    """

    callback_enabled: bool = Field(
        default=True,
        description=(
            "Enable the HF Trainer ``SystemMetricsCallback``. On by "
            "default — emits step-aligned ``gpu/{idx}/*`` / ``cpu/*`` / "
            "``ram/*`` metrics that flow through "
            "``ResilientMLflowTransport`` and survive offline windows "
            "via ``MetricsBuffer``. Set to ``False`` only if "
            "``pynvml.nvmlInit`` hangs on your specific cloud GPU image."
        ),
    )


__all__ = ["SystemMetricsConfig"]
