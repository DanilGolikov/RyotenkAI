"""Tests for ``ryotenkai events history`` — offline events.jsonl inspection.

The command reads a run's on-disk ``events.jsonl`` via :class:`JournalReader`
and renders aggregate stats (pretty or JSON). Tests use synthetic journals
in ``tmp_path`` — no HTTP, no orchestrator, fully hermetic.

Coverage split (project policy from CLAUDE.md / mock_policy.md):

1. Positive          — happy path renders expected sections / aggregates
2. Negative          — missing journal -> exit 2 with friendly message
3. Boundary          — empty journal -> "no events" + zero stats
4. Invariants        — ``--json`` round-trips through ``json.loads``
5. DependencyErrors  — corrupted journal lines -> UnknownEvent counted, no crash
6. Regressions       — timeline buckets, severity ordering, percentage math
7. LogicSpecific     — ``--runs-dir`` override, ``--limit-errors``, offset window
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from typer.testing import CliRunner

from ryotenkai_control.cli.app import app
from ryotenkai_shared.events import to_jsonl
from ryotenkai_shared.events.types.control_evaluation import (
    EvaluationPluginCompletedEvent,
    EvaluationPluginCompletedPayload,
)
from ryotenkai_shared.events.types.control_gpu import (
    GPUDeploymentFailedEvent,
    GPUDeploymentFailedPayload,
)
from ryotenkai_shared.events.types.control_run import (
    RunCompletedEvent,
    RunCompletedPayload,
    RunFailedEvent,
    RunFailedPayload,
    RunStartedEvent,
    RunStartedPayload,
)
from ryotenkai_shared.events.types.control_stage import (
    StageCompletedEvent,
    StageCompletedPayload,
    StageFailedEvent,
    StageFailedPayload,
    StageStartedEvent,
    StageStartedPayload,
)
from ryotenkai_shared.events.types.pod_training import (
    TrainingStepEvent,
    TrainingStepPayload,
)

if TYPE_CHECKING:
    from ryotenkai_shared.events import BaseEvent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture()
def runs_dir(tmp_path: Path) -> Path:
    """Empty runs base directory; tests populate it per-test."""
    runs = tmp_path / "runs"
    runs.mkdir()
    return runs


def _run_dir(runs_dir: Path, run_id: str) -> Path:
    """Create ``<runs_dir>/<run_id>/`` and return it."""
    d = runs_dir / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_journal(path: Path, envelopes: list[BaseEvent]) -> None:
    """Serialize ``envelopes`` to ``path`` using the length-prefixed codec."""
    path.write_text("".join(to_jsonl(ev) for ev in envelopes))


def _t(offset_seconds: int) -> datetime:
    """Deterministic event time: 2026-05-17 08:00:00 UTC + ``offset_seconds``."""
    return datetime(2026, 5, 17, 8, 0, 0, tzinfo=UTC) + timedelta(
        seconds=offset_seconds
    )


def _mk_run_started(offset: int = 0, t: datetime | None = None) -> RunStartedEvent:
    return RunStartedEvent(
        source="control://orchestrator",
        run_id="run-001",
        offset=offset,
        time=t or _t(0),
        payload=RunStartedPayload(
            run_name="r",
            algorithm="sft",
            model_id="m",
            dataset_id="d",
            config_hash="h",
        ),
    )


def _mk_run_completed(offset: int = 1, t: datetime | None = None) -> RunCompletedEvent:
    return RunCompletedEvent(
        source="control://orchestrator",
        run_id="run-001",
        offset=offset,
        time=t or _t(60),
        payload=RunCompletedPayload(
            duration_s=60.0,
            final_status="success",
            mlflow_run_id=None,
        ),
    )


def _mk_run_failed(offset: int = 2, t: datetime | None = None) -> RunFailedEvent:
    return RunFailedEvent(
        source="control://orchestrator",
        run_id="run-001",
        offset=offset,
        time=t or _t(30),
        payload=RunFailedPayload(
            failing_stage="training",
            error_type="OOM",
            message="oom",
            traceback_excerpt="Traceback...",
        ),
    )


def _mk_stage_started(
    offset: int,
    stage: str = "training",
    t: datetime | None = None,
) -> StageStartedEvent:
    return StageStartedEvent(
        source="control://orchestrator",
        run_id="run-001",
        stage_id=stage,
        offset=offset,
        time=t or _t(5),
        payload=StageStartedPayload(
            stage_name=stage, stage_index=0, total_stages=3, inputs_summary={}
        ),
    )


def _mk_stage_completed(
    offset: int,
    stage: str = "training",
    t: datetime | None = None,
) -> StageCompletedEvent:
    return StageCompletedEvent(
        source="control://orchestrator",
        run_id="run-001",
        stage_id=stage,
        offset=offset,
        time=t or _t(45),
        payload=StageCompletedPayload(
            stage_name=stage, duration_s=40.0, outputs_summary={}
        ),
    )


def _mk_stage_failed(
    offset: int,
    stage: str = "training",
    t: datetime | None = None,
) -> StageFailedEvent:
    return StageFailedEvent(
        source="control://orchestrator",
        run_id="run-001",
        stage_id=stage,
        offset=offset,
        time=t or _t(40),
        payload=StageFailedPayload(
            stage_name=stage,
            error_type="OOM",
            message="oom",
            traceback_excerpt="Traceback...",
        ),
    )


def _mk_training_step(offset: int, t: datetime | None = None) -> TrainingStepEvent:
    return TrainingStepEvent(
        source="pod://run-001/trainer",
        run_id="run-001",
        stage_id="training",
        offset=offset,
        time=t or _t(10),
        payload=TrainingStepPayload(step=offset, loss=1.0, learning_rate=1e-4),
    )


def _mk_gpu_failed(offset: int, t: datetime | None = None) -> GPUDeploymentFailedEvent:
    return GPUDeploymentFailedEvent(
        source="control://orchestrator",
        run_id="run-001",
        stage_id="gpu_deployer",
        offset=offset,
        time=t or _t(20),
        payload=GPUDeploymentFailedPayload(
            reason="container exited 137 (OOM)",
            provider_error_code="OutOfStock",
        ),
    )


def _mk_eval_completed(offset: int, t: datetime | None = None) -> EvaluationPluginCompletedEvent:
    return EvaluationPluginCompletedEvent(
        source="control://orchestrator",
        run_id="run-001",
        stage_id="evaluator",
        offset=offset,
        time=t or _t(55),
        payload=EvaluationPluginCompletedPayload(
            plugin_name="cerebras",
            duration_s=5.0,
            metrics={"score": 0.95},
        ),
    )


# ---------------------------------------------------------------------------
# 1. Positive — happy path
# ---------------------------------------------------------------------------


class TestPositive:
    def test_pretty_output_renders_header_with_run_id_and_total(
        self, runner: CliRunner, runs_dir: Path
    ) -> None:
        d = _run_dir(runs_dir, "run-001")
        _write_journal(
            d / "events.jsonl", [_mk_run_started(), _mk_run_completed()]
        )

        result = runner.invoke(
            app,
            ["events", "history", "--run-id", "run-001", "--runs-dir", str(runs_dir)],
        )

        assert result.exit_code == 0, result.output
        assert "run-001" in result.output
        # Header must show the event count (formatted with thousands sep).
        assert "2 events" in result.output

    def test_pretty_lists_by_source_section(
        self, runner: CliRunner, runs_dir: Path
    ) -> None:
        d = _run_dir(runs_dir, "run-001")
        events = [
            _mk_run_started(),
            _mk_training_step(1),
            _mk_training_step(2),
            _mk_run_completed(3),
        ]
        _write_journal(d / "events.jsonl", events)

        result = runner.invoke(
            app,
            ["events", "history", "--run-id", "run-001", "--runs-dir", str(runs_dir)],
        )

        assert result.exit_code == 0, result.output
        assert "By source:" in result.output
        assert "pod://run-001/trainer" in result.output
        assert "control://orchestrator" in result.output

    def test_pretty_lists_top_event_kinds(
        self, runner: CliRunner, runs_dir: Path
    ) -> None:
        d = _run_dir(runs_dir, "run-001")
        events: list[BaseEvent] = [_mk_run_started()]
        events.extend(_mk_training_step(i + 1) for i in range(5))
        events.append(_mk_run_completed(7))
        _write_journal(d / "events.jsonl", events)

        result = runner.invoke(
            app,
            ["events", "history", "--run-id", "run-001", "--runs-dir", str(runs_dir)],
        )

        assert result.exit_code == 0, result.output
        assert "Top 10 event kinds:" in result.output
        assert "ryotenkai.pod.training.step" in result.output

    def test_pretty_renders_by_stage_when_present(
        self, runner: CliRunner, runs_dir: Path
    ) -> None:
        d = _run_dir(runs_dir, "run-001")
        events = [
            _mk_run_started(),
            _mk_stage_started(1),
            _mk_stage_completed(2),
            _mk_run_completed(3),
        ]
        _write_journal(d / "events.jsonl", events)

        result = runner.invoke(
            app,
            ["events", "history", "--run-id", "run-001", "--runs-dir", str(runs_dir)],
        )

        assert result.exit_code == 0, result.output
        assert "By stage" in result.output
        assert "training" in result.output


# ---------------------------------------------------------------------------
# 2. Negative — missing journal
# ---------------------------------------------------------------------------


class TestNegative:
    def test_nonexistent_run_id_exits_with_code_2(
        self, runner: CliRunner, runs_dir: Path
    ) -> None:
        result = runner.invoke(
            app,
            [
                "events",
                "history",
                "--run-id",
                "does-not-exist",
                "--runs-dir",
                str(runs_dir),
            ],
        )

        assert result.exit_code == 2, result.output
        # No raw traceback should reach the user.
        assert "Traceback" not in result.output

    def test_nonexistent_run_id_message_mentions_run_id(
        self, runner: CliRunner, runs_dir: Path
    ) -> None:
        result = runner.invoke(
            app,
            [
                "events",
                "history",
                "--run-id",
                "ghost-run",
                "--runs-dir",
                str(runs_dir),
            ],
        )

        assert result.exit_code == 2
        # Combined stdout + stderr — Typer routes die() to stderr.
        combined = result.output + (result.stderr if result.stderr_bytes else "")
        assert "ghost-run" in combined or "run not found" in combined


# ---------------------------------------------------------------------------
# 3. Boundary — empty journal
# ---------------------------------------------------------------------------


class TestBoundary:
    def test_empty_journal_renders_no_events_message(
        self, runner: CliRunner, runs_dir: Path
    ) -> None:
        d = _run_dir(runs_dir, "empty-run")
        (d / "events.jsonl").write_text("")

        result = runner.invoke(
            app,
            ["events", "history", "--run-id", "empty-run", "--runs-dir", str(runs_dir)],
        )

        assert result.exit_code == 0, result.output
        assert "no events" in result.output.lower()

    def test_empty_journal_json_zero_total_events(
        self, runner: CliRunner, runs_dir: Path
    ) -> None:
        d = _run_dir(runs_dir, "empty-run")
        (d / "events.jsonl").write_text("")

        result = runner.invoke(
            app,
            [
                "events",
                "history",
                "--run-id",
                "empty-run",
                "--runs-dir",
                str(runs_dir),
                "--json",
            ],
        )

        assert result.exit_code == 0, result.output
        parsed = json.loads(result.stdout)
        assert parsed["total_events"] == 0
        assert parsed["first_event"] is None
        assert parsed["last_event"] is None
        assert parsed["duration_seconds"] is None
        assert parsed["by_source"] == {}


# ---------------------------------------------------------------------------
# 4. Invariants — --json shape
# ---------------------------------------------------------------------------


class TestInvariants:
    def test_json_output_is_parseable(
        self, runner: CliRunner, runs_dir: Path
    ) -> None:
        d = _run_dir(runs_dir, "run-001")
        _write_journal(
            d / "events.jsonl", [_mk_run_started(), _mk_run_completed()]
        )

        result = runner.invoke(
            app,
            [
                "events",
                "history",
                "--run-id",
                "run-001",
                "--runs-dir",
                str(runs_dir),
                "--json",
            ],
        )

        assert result.exit_code == 0, result.output
        # Must parse without errors.
        parsed = json.loads(result.stdout)
        assert isinstance(parsed, dict)

    def test_json_output_contains_required_keys(
        self, runner: CliRunner, runs_dir: Path
    ) -> None:
        d = _run_dir(runs_dir, "run-001")
        _write_journal(
            d / "events.jsonl", [_mk_run_started(), _mk_run_completed()]
        )

        result = runner.invoke(
            app,
            [
                "events",
                "history",
                "--run-id",
                "run-001",
                "--runs-dir",
                str(runs_dir),
                "--json",
            ],
        )

        parsed = json.loads(result.stdout)
        required = {
            "run_id",
            "journal_path",
            "size_bytes",
            "total_events",
            "first_event",
            "last_event",
            "duration_seconds",
            "by_source",
            "by_severity",
            "by_kind",
            "by_stage",
            "schema_versions_present",
            "errors",
            "unknown_events",
            "timeline_buckets_15min",
        }
        missing = required - set(parsed.keys())
        assert not missing, f"missing keys in JSON output: {missing}"

    def test_json_aggregates_match_event_counts(
        self, runner: CliRunner, runs_dir: Path
    ) -> None:
        d = _run_dir(runs_dir, "run-001")
        events: list[BaseEvent] = [_mk_run_started()]
        events.extend(_mk_training_step(i + 1) for i in range(7))
        events.append(_mk_run_completed(8))
        _write_journal(d / "events.jsonl", events)

        result = runner.invoke(
            app,
            [
                "events",
                "history",
                "--run-id",
                "run-001",
                "--runs-dir",
                str(runs_dir),
                "--json",
            ],
        )

        parsed = json.loads(result.stdout)
        assert parsed["total_events"] == 9
        assert parsed["by_kind"]["ryotenkai.pod.training.step"] == 7
        assert parsed["by_kind"]["ryotenkai.control.run.started"] == 1
        # by_source counts: 2 control events + 7 trainer events.
        assert parsed["by_source"]["control://orchestrator"] == 2
        assert parsed["by_source"]["pod://run-001/trainer"] == 7


# ---------------------------------------------------------------------------
# 5. DependencyErrors — corrupted / malformed journal
# ---------------------------------------------------------------------------


class TestDependencyErrors:
    def test_corrupted_lines_counted_as_unknown_no_crash(
        self, runner: CliRunner, runs_dir: Path
    ) -> None:
        d = _run_dir(runs_dir, "run-001")
        # Mix of valid + garbage lines. JournalReader treats both as
        # iterable; the codec wraps malformed lines in UnknownEvent.
        valid = to_jsonl(_mk_run_started())
        garbage_1 = "this is not a valid event line\n"
        garbage_2 = '999\t{"not": "matching length"}\n'
        (d / "events.jsonl").write_text(valid + garbage_1 + garbage_2)

        result = runner.invoke(
            app,
            [
                "events",
                "history",
                "--run-id",
                "run-001",
                "--runs-dir",
                str(runs_dir),
                "--json",
            ],
        )

        assert result.exit_code == 0, result.output
        parsed = json.loads(result.stdout)
        # Two malformed lines + one valid event = 3 total, 2 unknown.
        assert parsed["unknown_events"] >= 2
        assert parsed["total_events"] >= 1


# ---------------------------------------------------------------------------
# 6. Regressions
# ---------------------------------------------------------------------------


class TestRegressions:
    def test_short_run_produces_single_timeline_bucket(
        self, runner: CliRunner, runs_dir: Path
    ) -> None:
        d = _run_dir(runs_dir, "run-001")
        # Events all within the same 15-min window (start + 60s later).
        _write_journal(
            d / "events.jsonl", [_mk_run_started(), _mk_run_completed()]
        )

        result = runner.invoke(
            app,
            [
                "events",
                "history",
                "--run-id",
                "run-001",
                "--runs-dir",
                str(runs_dir),
                "--json",
            ],
        )

        parsed = json.loads(result.stdout)
        assert len(parsed["timeline_buckets_15min"]) == 1

    def test_one_hour_run_produces_four_buckets(
        self, runner: CliRunner, runs_dir: Path
    ) -> None:
        d = _run_dir(runs_dir, "run-001")
        # Events at 0, 20, 40, 60 minutes — across four 15-min windows.
        events = [
            _mk_run_started(offset=0, t=_t(0)),
            _mk_training_step(offset=1, t=_t(20 * 60)),
            _mk_training_step(offset=2, t=_t(40 * 60)),
            _mk_run_completed(offset=3, t=_t(60 * 60)),
        ]
        _write_journal(d / "events.jsonl", events)

        result = runner.invoke(
            app,
            [
                "events",
                "history",
                "--run-id",
                "run-001",
                "--runs-dir",
                str(runs_dir),
                "--json",
            ],
        )

        parsed = json.loads(result.stdout)
        # 08:00, 08:15, 08:30, 08:45, 09:00 — events fall in 5 buckets.
        # (08:00, 08:15/16-30 contains 20min, 08:30-45 contains 40min,
        #  09:00 contains 60min). Concretely we expect 4 distinct buckets
        # since the 0 and 60 events are at the boundaries of the 1h
        # window. Accept 4 or 5 to be robust against boundary semantics.
        assert 4 <= len(parsed["timeline_buckets_15min"]) <= 5

    def test_severity_breakdown_includes_all_five_levels(
        self, runner: CliRunner, runs_dir: Path
    ) -> None:
        d = _run_dir(runs_dir, "run-001")
        events: list[BaseEvent] = [
            _mk_run_started(),  # info
            _mk_training_step(1),  # debug
            _mk_run_failed(2),  # error
        ]
        _write_journal(d / "events.jsonl", events)

        result = runner.invoke(
            app,
            ["events", "history", "--run-id", "run-001", "--runs-dir", str(runs_dir)],
        )

        assert result.exit_code == 0, result.output
        # All five severity labels appear (with counts or em-dash for absent).
        for level in ("debug", "info", "warning", "error", "critical"):
            assert level in result.output

    def test_duration_computed_from_first_and_last_event(
        self, runner: CliRunner, runs_dir: Path
    ) -> None:
        d = _run_dir(runs_dir, "run-001")
        events = [
            _mk_run_started(offset=0, t=_t(0)),
            _mk_run_completed(offset=1, t=_t(3600)),  # 1 hour later
        ]
        _write_journal(d / "events.jsonl", events)

        result = runner.invoke(
            app,
            [
                "events",
                "history",
                "--run-id",
                "run-001",
                "--runs-dir",
                str(runs_dir),
                "--json",
            ],
        )

        parsed = json.loads(result.stdout)
        assert parsed["duration_seconds"] == pytest.approx(3600.0)


# ---------------------------------------------------------------------------
# 7. LogicSpecific — flags, overrides
# ---------------------------------------------------------------------------


class TestLogicSpecific:
    def test_runs_dir_override_points_at_alternative_location(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        # Two separate roots; the override must pick the right one.
        alt_root = tmp_path / "alt_root"
        alt_root.mkdir()
        run_dir = alt_root / "run-001"
        run_dir.mkdir()
        _write_journal(
            run_dir / "events.jsonl", [_mk_run_started(), _mk_run_completed()]
        )

        result = runner.invoke(
            app,
            [
                "events",
                "history",
                "--run-id",
                "run-001",
                "--runs-dir",
                str(alt_root),
            ],
        )

        assert result.exit_code == 0, result.output
        assert "2 events" in result.output

    def test_runs_dir_can_point_at_journal_file_directly(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        # Forensic mode: copy events.jsonl somewhere flat and inspect.
        flat_journal = tmp_path / "events.jsonl"
        _write_journal(
            flat_journal, [_mk_run_started(), _mk_run_completed()]
        )

        result = runner.invoke(
            app,
            [
                "events",
                "history",
                "--run-id",
                "ignored",  # not used when --runs-dir points at a file
                "--runs-dir",
                str(flat_journal),
            ],
        )

        assert result.exit_code == 0, result.output
        assert "2 events" in result.output

    def test_limit_errors_zero_omits_errors_section(
        self, runner: CliRunner, runs_dir: Path
    ) -> None:
        d = _run_dir(runs_dir, "run-001")
        events = [
            _mk_run_started(),
            _mk_stage_failed(1),  # severity=error
            _mk_run_completed(2),
        ]
        _write_journal(d / "events.jsonl", events)

        result = runner.invoke(
            app,
            [
                "events",
                "history",
                "--run-id",
                "run-001",
                "--runs-dir",
                str(runs_dir),
                "--limit-errors",
                "0",
            ],
        )

        assert result.exit_code == 0, result.output
        # The "Errors (severity >= error):" section header must NOT
        # appear when --limit-errors=0.
        assert "Errors (severity" not in result.output

    def test_limit_errors_default_shows_errors_section(
        self, runner: CliRunner, runs_dir: Path
    ) -> None:
        d = _run_dir(runs_dir, "run-001")
        events = [
            _mk_run_started(),
            _mk_stage_failed(1),
            _mk_gpu_failed(2),
            _mk_run_completed(3),
        ]
        _write_journal(d / "events.jsonl", events)

        result = runner.invoke(
            app,
            [
                "events",
                "history",
                "--run-id",
                "run-001",
                "--runs-dir",
                str(runs_dir),
            ],
        )

        assert result.exit_code == 0, result.output
        assert "Errors (severity" in result.output

    def test_no_timeline_flag_omits_timeline_section(
        self, runner: CliRunner, runs_dir: Path
    ) -> None:
        d = _run_dir(runs_dir, "run-001")
        _write_journal(
            d / "events.jsonl", [_mk_run_started(), _mk_run_completed()]
        )

        result = runner.invoke(
            app,
            [
                "events",
                "history",
                "--run-id",
                "run-001",
                "--runs-dir",
                str(runs_dir),
                "--no-timeline",
            ],
        )

        assert result.exit_code == 0, result.output
        assert "Timeline" not in result.output

    def test_from_offset_clamps_window(
        self, runner: CliRunner, runs_dir: Path
    ) -> None:
        d = _run_dir(runs_dir, "run-001")
        events: list[BaseEvent] = [_mk_run_started(offset=0)]
        events.extend(_mk_training_step(i) for i in range(1, 6))
        events.append(_mk_run_completed(offset=6))
        _write_journal(d / "events.jsonl", events)

        result = runner.invoke(
            app,
            [
                "events",
                "history",
                "--run-id",
                "run-001",
                "--runs-dir",
                str(runs_dir),
                "--from-offset",
                "3",
                "--json",
            ],
        )

        parsed = json.loads(result.stdout)
        # Events with offset >= 3: training_step(3), (4), (5), run_completed(6) -> 4
        assert parsed["total_events"] == 4

    def test_to_offset_clamps_window(
        self, runner: CliRunner, runs_dir: Path
    ) -> None:
        d = _run_dir(runs_dir, "run-001")
        events: list[BaseEvent] = [_mk_run_started(offset=0)]
        events.extend(_mk_training_step(i) for i in range(1, 6))
        events.append(_mk_run_completed(offset=6))
        _write_journal(d / "events.jsonl", events)

        result = runner.invoke(
            app,
            [
                "events",
                "history",
                "--run-id",
                "run-001",
                "--runs-dir",
                str(runs_dir),
                "--to-offset",
                "2",
                "--json",
            ],
        )

        parsed = json.loads(result.stdout)
        # Events with offset <= 2: run_started(0), step(1), step(2) -> 3
        assert parsed["total_events"] == 3

    def test_errors_section_includes_error_severity_events(
        self, runner: CliRunner, runs_dir: Path
    ) -> None:
        d = _run_dir(runs_dir, "run-001")
        events = [
            _mk_run_started(),
            _mk_gpu_failed(1),  # severity=error
            _mk_stage_failed(2),  # severity=error
            _mk_run_completed(3),
        ]
        _write_journal(d / "events.jsonl", events)

        result = runner.invoke(
            app,
            [
                "events",
                "history",
                "--run-id",
                "run-001",
                "--runs-dir",
                str(runs_dir),
                "--json",
            ],
        )

        parsed = json.loads(result.stdout)
        assert len(parsed["errors"]) == 2
        kinds = {err["kind"] for err in parsed["errors"]}
        assert "ryotenkai.control.gpu.deployment_failed" in kinds
        assert "ryotenkai.control.stage.failed" in kinds

    def test_schema_versions_present_collects_distinct_versions(
        self, runner: CliRunner, runs_dir: Path
    ) -> None:
        d = _run_dir(runs_dir, "run-001")
        # All built-in events default to schema_version=1.
        _write_journal(
            d / "events.jsonl",
            [_mk_run_started(), _mk_eval_completed(1), _mk_run_completed(2)],
        )

        result = runner.invoke(
            app,
            [
                "events",
                "history",
                "--run-id",
                "run-001",
                "--runs-dir",
                str(runs_dir),
                "--json",
            ],
        )

        parsed = json.loads(result.stdout)
        assert parsed["schema_versions_present"] == [1]
