"""Tests for ``ryotenkai events metrics`` — post-Phase-10 TODO #8.

The command is a thin HTTP client over ``GET /api/v1/health/events``;
every test patches ``ryotenkai_control.cli.commands.events.httpx.get``
to keep the test runtime hermetic.

Coverage split (project policy):

1. Positive          — happy path: summary + detail render expected sections
2. Negative          — API unreachable -> friendly die, no traceback
3. Boundary          — no active runs -> dedicated message
4. Invariants        — fetch invoked exactly once per command
5. DependencyErrors  — HTTP error / non-JSON body -> friendly die
6. Regressions       — --json produces JSON that round-trips through parser
7. LogicSpecific     — --run-id filters output to the detail block
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from ryotenkai_control.cli.app import app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


def _payload_two_runs() -> dict[str, Any]:
    """Two healthy runs with non-zero counters."""
    return {
        "status": "healthy",
        "active_runs": ["run-abc12345", "run-def67890"],
        "per_run": {
            "run-abc12345": {
                "emitter_events_emitted_total": 15234,
                "emitter_events_emit_failed_total": {},
                "emitter_events_remote_accepted_total": 892,
                "emitter_events_remote_dropped_total": {},
                "emitter_offset_collisions_detected_total": 0,
                "bus_published_total": 15234,
                "bus_dropped_total": 0,
                "bus_dropped_per_consumer": {},
                "bus_current_depth": 125,
                "bus_capacity": 10000,
                "bus_subscriber_count": 3,
                "journal_appended_total": 15234,
                "journal_fsync_total": 312,
                "journal_fsync_failed_total": 0,
                "journal_total_bytes_written": 18_734_562,
                "journal_write_failed_total": 0,
                "journal_last_fsync_age_seconds": 0.3,
                "dedup_size": 42,
                "dedup_seen_total": 892,
                "dedup_hits_total": 0,
                "dedup_evicted_total": 0,
            },
            "run-def67890": {
                "emitter_events_emitted_total": 800,
                "emitter_events_emit_failed_total": {},
                "emitter_events_remote_accepted_total": 0,
                "emitter_events_remote_dropped_total": {},
                "emitter_offset_collisions_detected_total": 0,
                "bus_published_total": 800,
                "bus_dropped_total": 0,
                "bus_dropped_per_consumer": {},
                "bus_current_depth": 0,
                "bus_capacity": 10000,
                "bus_subscriber_count": 1,
                "journal_appended_total": 800,
                "journal_fsync_total": 16,
                "journal_fsync_failed_total": 0,
                "journal_total_bytes_written": 73_410,
                "journal_write_failed_total": 0,
                "journal_last_fsync_age_seconds": 12.7,
                "dedup_size": 0,
                "dedup_seen_total": 0,
                "dedup_hits_total": 0,
                "dedup_evicted_total": 0,
            },
        },
        "health_indicators": {
            "any_emit_failures": False,
            "any_drops": False,
            "any_fsync_failures": False,
            "any_write_failures": False,
            "any_offset_collisions": False,
        },
    }


def _payload_empty() -> dict[str, Any]:
    return {
        "status": "no_active_runs",
        "active_runs": [],
        "per_run": {},
        "health_indicators": {
            "any_emit_failures": False,
            "any_drops": False,
            "any_fsync_failures": False,
            "any_write_failures": False,
            "any_offset_collisions": False,
        },
    }


def _fake_response(payload: dict[str, Any], status_code: int = 200) -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = payload
    response.text = json.dumps(payload)
    return response


def _patch_httpx_get(response: MagicMock | Exception):
    """Patch the lazy-imported httpx.get used inside the events command."""
    # The events command does ``import httpx`` lazily inside the
    # fetch helper, so we patch at the import-site module.
    return patch("httpx.get", return_value=response if not isinstance(response, Exception) else None,
                 side_effect=response if isinstance(response, Exception) else None)


# ---------------------------------------------------------------------------
# 1. Positive
# ---------------------------------------------------------------------------


class TestPositive:
    def test_summary_lists_each_active_run(self, runner: CliRunner) -> None:
        with _patch_httpx_get(_fake_response(_payload_two_runs())):
            result = runner.invoke(app, ["events", "metrics"])

        assert result.exit_code == 0, result.output
        assert "Active runs: 2" in result.output
        assert "run-abc12345" in result.output
        assert "run-def67890" in result.output
        # Counters should appear formatted with thousands separators.
        assert "15,234" in result.output
        assert "healthy" in result.output

    def test_summary_includes_all_required_fields(self, runner: CliRunner) -> None:
        with _patch_httpx_get(_fake_response(_payload_two_runs())):
            result = runner.invoke(app, ["events", "metrics"])

        # Each labelled field from the plan must appear in the output.
        for label in (
            "Events emitted:",
            "Events dropped:",
            "Journal size:",
            "Last fsync:",
            "Dedup set:",
            "In-memory bus:",
        ):
            assert label in result.output, f"missing label: {label!r}"

    def test_detail_lists_per_collaborator_sections(self, runner: CliRunner) -> None:
        payload = _payload_two_runs()
        with _patch_httpx_get(_fake_response(payload)):
            result = runner.invoke(
                app, ["events", "metrics", "--run-id", "run-abc12345"],
            )

        assert result.exit_code == 0, result.output
        # Each collaborator gets a section heading + key counter labels.
        for heading in ("Emitter:", "Bus:", "Journal:", "Dedup:"):
            assert heading in result.output, f"missing section {heading!r}"
        assert "events_emitted_total" in result.output
        assert "fsyncs_total" in result.output
        assert "size" in result.output


# ---------------------------------------------------------------------------
# 2. Negative
# ---------------------------------------------------------------------------


class TestNegative:
    def test_connect_error_renders_friendly_error(self, runner: CliRunner) -> None:
        import httpx

        with _patch_httpx_get(httpx.ConnectError("connection refused")):
            result = runner.invoke(app, ["events", "metrics"])

        assert result.exit_code != 0
        assert "Traceback" not in result.output
        # The friendly message must mention "API" so the operator knows
        # the failure mode at a glance.
        assert "API" in result.output or "api" in result.output

    def test_timeout_renders_friendly_error(self, runner: CliRunner) -> None:
        import httpx

        with _patch_httpx_get(httpx.ReadTimeout("slow server")):
            result = runner.invoke(app, ["events", "metrics", "--timeout", "0.1"])

        assert result.exit_code != 0
        assert "Traceback" not in result.output


# ---------------------------------------------------------------------------
# 3. Boundary
# ---------------------------------------------------------------------------


class TestBoundary:
    def test_no_active_runs_prints_dedicated_message(self, runner: CliRunner) -> None:
        with _patch_httpx_get(_fake_response(_payload_empty())):
            result = runner.invoke(app, ["events", "metrics"])

        assert result.exit_code == 0, result.output
        assert "No active runs" in result.output
        assert "no active runs" in result.output  # in health summary

    def test_detail_for_missing_run_reports_not_found(self, runner: CliRunner) -> None:
        # API legitimately returns an empty per_run dict when the
        # caller asks for a run that's not registered. The CLI must
        # render that as a clear "not found" instead of crashing on
        # a missing key.
        with _patch_httpx_get(_fake_response(_payload_empty())):
            result = runner.invoke(
                app, ["events", "metrics", "--run-id", "ghost-run"],
            )
        assert result.exit_code == 0
        assert "ghost-run" in result.output


# ---------------------------------------------------------------------------
# 4. Invariants
# ---------------------------------------------------------------------------


class TestInvariants:
    def test_fetch_called_exactly_once(self, runner: CliRunner) -> None:
        response = _fake_response(_payload_two_runs())
        with patch("httpx.get", return_value=response) as mock_get:
            result = runner.invoke(app, ["events", "metrics"])
        assert result.exit_code == 0
        assert mock_get.call_count == 1

    def test_run_id_passed_through_to_query_string(self, runner: CliRunner) -> None:
        response = _fake_response(_payload_two_runs())
        with patch("httpx.get", return_value=response) as mock_get:
            runner.invoke(
                app, ["events", "metrics", "--run-id", "specific-run"],
            )

        call = mock_get.call_args
        params = call.kwargs.get("params", {})
        assert params.get("run_id") == "specific-run"

    def test_run_id_absent_when_summary_requested(self, runner: CliRunner) -> None:
        response = _fake_response(_payload_two_runs())
        with patch("httpx.get", return_value=response) as mock_get:
            runner.invoke(app, ["events", "metrics"])
        call = mock_get.call_args
        params = call.kwargs.get("params", {})
        assert "run_id" not in params


# ---------------------------------------------------------------------------
# 5. Dependency errors
# ---------------------------------------------------------------------------


class TestDependencyErrors:
    def test_http_500_renders_friendly_error(self, runner: CliRunner) -> None:
        response = MagicMock()
        response.status_code = 500
        response.text = "internal server error"
        response.json.side_effect = ValueError("not json")

        with patch("httpx.get", return_value=response):
            result = runner.invoke(app, ["events", "metrics"])

        assert result.exit_code != 0
        assert "Traceback" not in result.output
        assert "500" in result.output

    def test_non_json_body_renders_friendly_error(self, runner: CliRunner) -> None:
        response = MagicMock()
        response.status_code = 200
        response.text = "<html>oops</html>"
        response.json.side_effect = ValueError("Expecting value")

        with patch("httpx.get", return_value=response):
            result = runner.invoke(app, ["events", "metrics"])

        assert result.exit_code != 0
        assert "Traceback" not in result.output


# ---------------------------------------------------------------------------
# 6. Regressions
# ---------------------------------------------------------------------------


class TestRegressions:
    def test_json_flag_emits_valid_round_trippable_json(
        self, runner: CliRunner,
    ) -> None:
        payload = _payload_two_runs()
        with _patch_httpx_get(_fake_response(payload)):
            result = runner.invoke(app, ["events", "metrics", "--json"])

        assert result.exit_code == 0, result.output
        # The buffered JsonRenderer dumps the dict on flush; parsing back
        # must give us the exact payload we fed in.
        parsed = json.loads(result.stdout)
        assert parsed["status"] == "healthy"
        assert "run-abc12345" in parsed["per_run"]

    def test_global_json_output_overrides_text_path(self, runner: CliRunner) -> None:
        payload = _payload_two_runs()
        with _patch_httpx_get(_fake_response(payload)):
            result = runner.invoke(app, ["-o", "json", "events", "metrics"])

        assert result.exit_code == 0, result.output
        parsed = json.loads(result.stdout)
        assert parsed["active_runs"] == ["run-abc12345", "run-def67890"]

    def test_help_renders_without_network(self, runner: CliRunner) -> None:
        # ``--help`` must never reach the HTTP layer — guards against a
        # regression where the option-default tried to resolve via env.
        result = runner.invoke(app, ["events", "metrics", "--help"])
        assert result.exit_code == 0
        assert "metrics" in result.output


# ---------------------------------------------------------------------------
# 7. Logic-specific
# ---------------------------------------------------------------------------


class TestLogicSpecific:
    def test_run_id_filter_renders_detail_not_summary(self, runner: CliRunner) -> None:
        payload = _payload_two_runs()
        with _patch_httpx_get(_fake_response(payload)):
            result = runner.invoke(
                app, ["events", "metrics", "--run-id", "run-abc12345"],
            )

        # Detail mode features section headings, summary mode does not.
        assert "Emitter:" in result.output
        # Summary header text "Active runs: 2" must NOT appear in
        # detail mode — that's the marker of a regression where we
        # routed to the wrong renderer.
        assert "Active runs:" not in result.output

    def test_api_url_override_propagates_to_request(self, runner: CliRunner) -> None:
        response = _fake_response(_payload_two_runs())
        with patch("httpx.get", return_value=response) as mock_get:
            runner.invoke(
                app,
                ["events", "metrics", "--api-url", "http://example.com:9000/"],
            )
        url = mock_get.call_args.args[0]
        # Trailing slash must be stripped; path appended cleanly.
        assert url == "http://example.com:9000/api/v1/health/events"

    def test_api_url_env_var_used_when_no_flag(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("RYOTENKAI_API_URL", "http://from-env:8123")
        response = _fake_response(_payload_two_runs())
        with patch("httpx.get", return_value=response) as mock_get:
            runner.invoke(app, ["events", "metrics"])
        url = mock_get.call_args.args[0]
        assert url.startswith("http://from-env:8123/")

    def test_format_bytes_for_journal_size(self, runner: CliRunner) -> None:
        # Direct sanity check on the renderer helper — guards against
        # accidental decimal-vs-binary unit swaps (MB vs MiB).
        from ryotenkai_control.cli.commands.events import _format_bytes

        assert _format_bytes(18_734_562) == "18,734,562 (17.9 MiB)"
        assert _format_bytes(0) == "0"
        assert _format_bytes(None) == "-"

    def test_format_age_renders_seconds_minutes_hours(self) -> None:
        from ryotenkai_control.cli.commands.events import _format_age

        assert _format_age(0.3) == "0.3s ago"
        assert _format_age(120) == "2m ago"
        assert _format_age(3700) == "1h ago"
        assert _format_age(None) == "never"
