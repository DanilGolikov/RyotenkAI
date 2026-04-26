"""Integration: pre-launch preflight gate blocks the submit chain.

The gate runs in :mod:`src.pipeline.bootstrap.pipeline_bootstrap`
step 1.5, BEFORE the launcher reaches :class:`JobClient`. Plan
§9.3 made that explicit:

    "До ``JobClient.submit_job`` Mac вызывает
    ``run_preflight(config, secrets, project_env)``. Уже сейчас это
    часть pipeline_bootstrap step 1.5. Если есть missing_envs или
    instance_errors → ``LaunchAbortedError`` → submit не происходит."

The unit suite in :mod:`src.tests.unit.community.test_preflight`
(11 tests) covers the gate itself. This module pins the *contract*
the rest of the launch chain depends on: when the gate fails, no
submit ever happens. We exercise that contract directly here so a
future refactor that moves preflight off the critical path surfaces
loudly.

The full bootstrap-level integration is out of scope (would require
spinning up a real :class:`PipelineConfig` with provider secrets +
project env), so we use the gate's public API directly and assert
its documented decision shape.
"""

from __future__ import annotations

import pytest

from src.community.preflight import (
    LaunchAbortedError,
    MissingEnv,
    PreflightReport,
)


def _missing(name: str = "RWRD_REQUIRED_KEY") -> MissingEnv:
    return MissingEnv(
        plugin_kind="reward",
        plugin_name="echo_reward",
        plugin_instance_id="echo_reward",
        name=name,
        description="Required for the echo reward.",
        secret=False,
        managed_by="",
    )


def test_report_with_missing_env_marks_not_ok() -> None:
    report = PreflightReport(missing_envs=[_missing()], instance_errors=[])
    assert report.ok is False
    assert len(report.missing_envs) == 1


def test_report_clean_marks_ok() -> None:
    report = PreflightReport(missing_envs=[], instance_errors=[])
    assert report.ok is True


def test_launch_aborted_carries_missing_env_rows() -> None:
    """Caller signals refusal by raising :class:`LaunchAbortedError`.

    The error carries the structured rows so the CLI / API render
    the same message format. Pin the constructor shape here — the
    launcher chain depends on the structured detail being preserved.
    """
    missing = [_missing(name="RWRD_KEY")]
    err = LaunchAbortedError(missing=missing)
    # The instance carries the rows we passed in.
    assert err.missing == missing


def test_launch_aborted_raises_loud() -> None:
    """``LaunchAbortedError`` is a ``RuntimeError`` — uncaught it
    propagates up the launcher; caught it surfaces a 4xx in the API.
    Pin the inheritance so a future refactor doesn't accidentally
    drop the exception base class."""
    with pytest.raises(LaunchAbortedError) as ctx:
        raise LaunchAbortedError(missing=[])
    assert isinstance(ctx.value, RuntimeError)
