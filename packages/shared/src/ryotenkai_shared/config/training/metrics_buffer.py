"""
Phase 12.A.2 — config-driven metrics decimation.

Background
----------
:class:`~src.training.mlflow.metrics_buffer.MetricsBuffer` decimates
buffered MLflow metrics to keep the on-disk JSONL bounded during long
training runs where the MLflow upstream is unreachable (e.g. Mac
asleep). Phase 9 used a hard-coded 3-tier policy: keep all in the
first 10 minutes, then every 2nd step for 10–30 min, then every 5th
step after.

Phase 12.A.2 lifts that policy into user config so:

* Default = ``keep_all=true`` — every metric is preserved losslessly.
  The user mandate ("по умолчанию давай сделаем без ограничений,
  пускай каждую метрики сохраняем") makes data fidelity the default;
  decimation becomes opt-in for long runs that genuinely need to
  bound disk / replay overhead.
* Three time-windowed thresholds (first / mid / late) replicate the
  Phase 9 hard-coded behaviour when ``keep_all=false``, but every
  knob is now configurable. Keeps the existing "tighter precision
  early, coarser later" intuition intact.

Schema example (``training.metrics_buffer`` block):

.. code-block:: yaml

    training:
      metrics_buffer:
        keep_all: true                  # Default — keep every metric.
        decimation:                     # Active only when keep_all=false.
          window_first_minutes: 10
          window_first_keep_every: 1
          window_mid_minutes: 30
          window_mid_keep_every: 2
          window_late_keep_every: 5

Old configs without a ``metrics_buffer`` block fall through the
default factory (``keep_all=true``), which is **strictly more
permissive** than the Phase 9 hard-coded policy — no run ever loses
metrics it would have kept before.
"""

from __future__ import annotations

from pydantic import Field

from ..base import StrictBaseModel


class DecimationWindowConfig(StrictBaseModel):
    """Three-window step-decimation thresholds.

    Active only when :attr:`MetricsBufferConfig.keep_all` is ``False``.
    Each window keeps every Nth step within its time band:

    * ``[0, window_first_minutes)`` →   step % ``window_first_keep_every`` == 0
    * ``[window_first_minutes, window_first_minutes + window_mid_minutes)`` →
                                       step % ``window_mid_keep_every`` == 0
    * ``[window_first_minutes + window_mid_minutes, ∞)`` →
                                       step % ``window_late_keep_every`` == 0

    Defaults mirror the Phase 9 hard-coded policy (1 / 2 / 5) so
    operators flipping ``keep_all=false`` get the same behaviour they
    had pre-Phase 12 unless they tune the windows.
    """

    window_first_minutes: int = Field(
        default=10,
        ge=1,
        description=(
            "Width of the first decimation window in minutes. "
            "All steps in [0, this) are subject to "
            "``window_first_keep_every``."
        ),
    )
    window_first_keep_every: int = Field(
        default=1,
        ge=1,
        description=(
            "Within the first window, keep every Nth step "
            "(step %% N == 0). Default 1 = keep every step."
        ),
    )
    window_mid_minutes: int = Field(
        default=30,
        ge=1,
        description=(
            "Width of the mid decimation window in minutes. The "
            "window covers [first_minutes, first_minutes + this)."
        ),
    )
    window_mid_keep_every: int = Field(
        default=2,
        ge=1,
        description=(
            "Within the mid window, keep every Nth step. Default 2 "
            "= keep every other step."
        ),
    )
    window_late_keep_every: int = Field(
        default=5,
        ge=1,
        description=(
            "Beyond the mid window, keep every Nth step. Default 5 "
            "= one in five."
        ),
    )


class MetricsBufferConfig(StrictBaseModel):
    """Controls :class:`MetricsBuffer` decimation policy.

    By default (``keep_all=True``) every metric is preserved
    losslessly — the buffer can grow as large as needed and Phase
    12.A.1 retrieval+replay ships the full history into MLflow on
    Mac wake. Set ``keep_all=False`` to enable adaptive time-windowed
    decimation for very long runs where bounded disk + replay
    overhead matters more than per-step granularity.
    """

    keep_all: bool = Field(
        default=True,
        description=(
            "If True (default), keep every metric — no decimation. "
            "Set False to enable time-windowed decimation governed "
            "by the ``decimation`` block."
        ),
    )
    decimation: DecimationWindowConfig = Field(
        default_factory=DecimationWindowConfig,
        description=(
            "Decimation parameters (active only when keep_all=False). "
            "Three time-windowed thresholds with per-window "
            "keep-every-N step."
        ),
    )


__all__ = [
    "DecimationWindowConfig",
    "MetricsBufferConfig",
]
