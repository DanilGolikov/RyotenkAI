"""
Standalone helpers for one-off dataset validation.

Reused by:
  - DatasetValidator pipeline stage (delegates format check here so the
    same logic answers both pipeline and HTTP API requests)
  - HTTP API endpoint that runs validation without spinning up the full
    pipeline / GPU (see src/api/routers/datasets.py)

Design notes:
  - No callbacks, no logging side-effects, no pipeline context — pure
    functions that take inputs and return data.
  - Plugin runs are sequential and per-plugin try/except. The pipeline
    stage keeps its own loop with callbacks and threshold-stop logic;
    the standalone variant is a deliberate, simpler peer (single
    responsibility per call site).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import TYPE_CHECKING, Any

from src.data.validation.base import ValidationErrorGroup, ValidationPlugin
from src.utils.result import AppError, DatasetError, Err, Ok, Result

if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset


@dataclass
class FormatCheckResult:
    """Per-strategy outcome of a dataset format compatibility check."""

    strategy_type: str
    ok: bool
    message: str = ""


def check_dataset_format(
    dataset: Dataset | IterableDataset,
    dataset_name: str,
    strategy_phases: list[Any],
    pipeline_config: Any,
) -> Result[list[FormatCheckResult], AppError]:
    """
    Run ``strategy.validate_dataset(dataset)`` for each unique
    ``strategy_type`` referenced in ``strategy_phases`` and aggregate
    the outcomes.

    Returns ``Ok(list)`` even when individual strategies fail — the
    caller decides whether the *first* failure should short-circuit the
    pipeline (DatasetValidator does so) or whether a full report is
    needed (HTTP /validate endpoint).

    The only ``Err`` path is "strategy class missing for declared type"
    — that's a config-level bug, not a data quality issue, so it short-
    circuits unconditionally.
    """
    if not strategy_phases:
        return Ok([])

    # Local import to avoid a circular dependency: factory imports
    # config which transitively imports validation plugins.
    from src.training.strategies.factory import StrategyFactory

    factory = StrategyFactory()
    seen: set[str] = set()
    out: list[FormatCheckResult] = []

    for phase in strategy_phases:
        strategy_type = phase.strategy_type
        if strategy_type in seen:
            continue
        seen.add(strategy_type)

        try:
            strategy = factory.create_from_phase(phase, pipeline_config)
        except ValueError as exc:
            return Err(
                DatasetError(
                    message=(
                        f"[{dataset_name}] Unknown strategy type "
                        f"'{strategy_type}': {exc}"
                    ),
                    code="DATASET_FORMAT_ERROR",
                )
            )

        check = strategy.validate_dataset(dataset)
        if check.is_failure():
            err = check.unwrap_err()
            out.append(
                FormatCheckResult(
                    strategy_type=strategy_type,
                    ok=False,
                    message=err.message,
                )
            )
        else:
            out.append(FormatCheckResult(strategy_type=strategy_type, ok=True))

    return Ok(out)


@dataclass
class StandalonePluginRun:
    """Outcome of running a single validation plugin in standalone mode."""

    plugin_id: str
    plugin_name: str
    passed: bool
    metrics: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    error_groups: list[ValidationErrorGroup] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    duration_ms: float = 0.0
    crashed: bool = False


def run_plugins(
    dataset: Dataset | IterableDataset,
    plugins: list[tuple[str, str, ValidationPlugin]],
) -> list[StandalonePluginRun]:
    """
    Run validation plugins sequentially against ``dataset``. Per-plugin
    try/except guarantees a crashed plugin doesn't poison the others;
    the failure surfaces as a ``StandalonePluginRun`` with
    ``crashed=True`` and the exception message in ``errors``.

    ``plugins`` is a list of ``(plugin_id, plugin_name, plugin)``
    triples — same shape the pipeline-stage uses internally so callers
    can build it from ``validation_registry.instantiate(...)`` calls.
    """
    results: list[StandalonePluginRun] = []
    for plugin_id, plugin_name, plugin in plugins:
        started = perf_counter()
        try:
            result = plugin.validate(dataset)
            duration_ms = (
                result.execution_time_ms
                if result.execution_time_ms
                else (perf_counter() - started) * 1000.0
            )
            display_errors = list(result.errors)
            recs: list[str] = []
            if not result.passed:
                display_errors.extend(
                    ValidationPlugin.render_error_groups(result.error_groups)
                )
                recs = list(plugin.get_recommendations(result))
            results.append(
                StandalonePluginRun(
                    plugin_id=plugin_id,
                    plugin_name=plugin_name,
                    passed=result.passed,
                    metrics=dict(result.metrics),
                    warnings=list(result.warnings),
                    errors=display_errors,
                    error_groups=list(result.error_groups),
                    recommendations=recs,
                    duration_ms=duration_ms,
                )
            )
        except Exception as exc:
            # Plugins are user-supplied code via the catalog discovery
            # path — we MUST contain the blast radius to a single row in
            # the response. The pipeline-stage equivalent does the same.
            duration_ms = (perf_counter() - started) * 1000.0
            results.append(
                StandalonePluginRun(
                    plugin_id=plugin_id,
                    plugin_name=plugin_name,
                    passed=False,
                    duration_ms=duration_ms,
                    errors=[f"Plugin '{plugin_id}' ({plugin_name}) crashed: {exc}"],
                    crashed=True,
                )
            )
    return results


__all__ = [
    "FormatCheckResult",
    "StandalonePluginRun",
    "check_dataset_format",
    "run_plugins",
]
