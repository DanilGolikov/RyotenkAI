"""``make_run_data`` — factory for real :class:`mlflow.entities.RunData`.

Replaces ``MagicMock(spec=RunData)`` in greenfield report tests. The real
``RunData`` exposes read-only ``metrics``/``params``/``tags`` properties that
can only be populated through typed entity lists (``Metric``, ``Param``,
``RunTag``); the factory accepts plain dicts for ergonomic test setup and
performs the conversion under the hood.

WHY a factory instead of a Fake: ``RunData`` is a pure value object (no
collaborators, no I/O, no lifecycle). A factory is the canonical pattern
for value objects per the Phase 3B plan in
``docs/plans/mock-elimination-architecture.md``.
"""

from __future__ import annotations

from typing import Any

from mlflow.entities import Metric, Param, RunData, RunTag


def _metric_entities(metrics: dict[str, float] | None, *, timestamp: int, step: int) -> list[Metric]:
    if not metrics:
        return []
    # WHY a stable timestamp/step: tests treat ``run_data.metrics`` as
    # a plain ``dict[str, float]``; the timestamp/step are required by
    # mlflow's constructor but aren't observed in greenfield assertions.
    return [Metric(key=k, value=float(v), timestamp=timestamp, step=step) for k, v in metrics.items()]


def _param_entities(params: dict[str, Any] | None) -> list[Param]:
    if not params:
        return []
    # mlflow stores params as strings; mirror that behaviour so the
    # resulting ``.params`` dict matches the real shape.
    return [Param(key=k, value=str(v)) for k, v in params.items()]


def _tag_entities(tags: dict[str, str] | None) -> list[RunTag]:
    if not tags:
        return []
    return [RunTag(key=k, value=v) for k, v in tags.items()]


def make_run_data(
    *,
    metrics: dict[str, float] | None = None,
    params: dict[str, Any] | None = None,
    tags: dict[str, str] | None = None,
    timestamp: int = 1_700_000_000_000,
    step: int = 0,
) -> RunData:
    """Build a real :class:`mlflow.entities.RunData` from plain dicts.

    Args:
        metrics: ``{metric_name: float}`` — populated as scalar values
            on the returned ``.metrics`` dict.
        params: ``{param_name: value}`` — coerced to strings (mlflow
            stores params as strings).
        tags: ``{tag_key: tag_value}`` — populated on the returned
            ``.tags`` dict.
        timestamp: timestamp (ms epoch) applied to every metric;
            tests rarely care, defaults to a fixed sentinel.
        step: training step applied to every metric;
            tests rarely care, defaults to 0.

    Returns:
        A real :class:`RunData` whose ``.metrics``/``.params``/``.tags``
        properties expose the input dicts.
    """
    return RunData(
        metrics=_metric_entities(metrics, timestamp=timestamp, step=step),
        params=_param_entities(params),
        tags=_tag_entities(tags),
    )


__all__ = ["make_run_data"]
