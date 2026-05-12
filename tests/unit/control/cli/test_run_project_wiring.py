"""CLI tests for ``ryotenkai run start --project`` wiring.

Post-refactor, the CLI parent never constructs ``PipelineOrchestrator``
in-process — it spawns ``python -m src.pipeline.worker`` foreground.
These tests pin the wiring matrix:

| Scenario                   | Behaviour                                         |
|----------------------------|---------------------------------------------------|
| ``run start -c X``         | anonymous spawn, no project resolver call         |
| ``run start -p A``         | resolves project A, RYOTENKAI_PROJECT_ID=A in env |
| ``run start -c X -p A``    | resolves project A with X as override             |
| ``run start`` (nothing)    | helpful error                                     |
| ``run start -p missing``   | ``ProjectNotFoundError`` → clean ``die``          |
| ``-p flag > RYOTENKAI_PROJECT > project use`` | precedence pinned     |

We mock ``subprocess.Popen`` so no real process is spawned; the
subprocess.run wait loop returns rc=0 immediately.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from ryotenkai_control.cli.commands.run import run_app
from ryotenkai_control.workspace.projects.adapter import (
    ProjectNotFoundError,
    ResolvedProject,
)


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def stub_resolved(tmp_path: Path) -> ResolvedProject:
    """Minimal :class:`ResolvedProject` for project-mode tests."""
    config_path = tmp_path / "configs" / "current.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("# stub\n", encoding="utf-8")
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    return ResolvedProject(
        config_path=config_path,
        runs_base_dir=runs_dir,
        env={"PROJECT_KEY": "v"},
        metadata={
            "project_id": "proj-a",
            "actor": "tester",
            "config_version_hash": "abc123",
        },
    )


@pytest.fixture
def fake_popen(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Patch ``subprocess.Popen`` in the run module — captures the
    command + env without actually spawning."""
    calls: dict = {}

    class _FakeProc:
        def __init__(self, cmd, **kwargs):
            calls["cmd"] = cmd
            calls["env"] = kwargs.get("env")
            calls["start_new_session"] = kwargs.get("start_new_session")

        def wait(self):
            return 0

    monkeypatch.setattr("ryotenkai_control.cli.commands.run.subprocess.Popen", _FakeProc)
    return calls


# ---------------------------------------------------------------------------
# Positive — both ``run start`` paths succeed
# ---------------------------------------------------------------------------


class TestPositive:
    def test_run_start_with_project_resolves_and_spawns(
        self,
        runner: CliRunner,
        stub_resolved: ResolvedProject,
        fake_popen: dict,
    ) -> None:
        with patch(
            "ryotenkai_control.workspace.projects.adapter.resolve_project_launch_inputs",
            return_value=stub_resolved,
        ) as resolve_mock:
            result = runner.invoke(run_app, ["start", "--project", "proj-a"])

        assert result.exit_code == 0, result.output
        resolve_mock.assert_called_once()
        # Subprocess command targets the worker module.
        cmd = fake_popen["cmd"]
        assert "ryotenkai_control.pipeline.worker" in cmd
        assert "--config" in cmd
        # Extra-env carries RYOTENKAI_PROJECT_ID + env.json keys.
        env = fake_popen["env"]
        assert env["RYOTENKAI_PROJECT_ID"] == "proj-a"
        assert env["RYOTENKAI_ACTOR"] == "tester"
        assert env["PROJECT_KEY"] == "v"

    def test_run_start_with_config_only_skips_project_resolver(
        self, runner: CliRunner, fake_popen: dict, tmp_path: Path,
    ) -> None:
        config = tmp_path / "x.yaml"
        config.write_text("# stub\n", encoding="utf-8")

        with patch(
            "ryotenkai_control.workspace.projects.adapter.resolve_project_launch_inputs",
        ) as resolve_mock:
            result = runner.invoke(
                run_app, ["start", "--config", str(config)],
            )

        assert result.exit_code == 0, result.output
        # No project resolver call on the anonymous path.
        resolve_mock.assert_not_called()
        cmd = fake_popen["cmd"]
        assert "--config" in cmd
        # No project metadata env vars.
        env = fake_popen["env"]
        assert "RYOTENKAI_PROJECT_ID" not in env

    def test_run_start_config_plus_project_treats_config_as_override(
        self,
        runner: CliRunner,
        stub_resolved: ResolvedProject,
        fake_popen: dict,
        tmp_path: Path,
    ) -> None:
        override = tmp_path / "experimental.yaml"
        override.write_text("# override\n", encoding="utf-8")

        with patch(
            "ryotenkai_control.workspace.projects.adapter.resolve_project_launch_inputs",
            return_value=stub_resolved,
        ) as resolve_mock:
            result = runner.invoke(
                run_app,
                ["start", "--config", str(override), "--project", "proj-a"],
            )

        assert result.exit_code == 0, result.output
        # The config override flowed in as ``config_override`` kwarg.
        called_kwargs = resolve_mock.call_args.kwargs
        assert called_kwargs["config_override"] == override


# ---------------------------------------------------------------------------
# Negative — surfacing clean error messages
# ---------------------------------------------------------------------------


class TestNegative:
    def test_no_args_yields_helpful_error(self, runner: CliRunner) -> None:
        result = runner.invoke(run_app, ["start"])
        assert result.exit_code != 0
        combined = result.output
        assert "missing required argument" in combined.lower() or (
            "config" in combined.lower() and "run-dir" in combined.lower()
        )

    def test_unknown_project_yields_clean_die(
        self, runner: CliRunner,
    ) -> None:
        with patch(
            "ryotenkai_control.workspace.projects.adapter.resolve_project_launch_inputs",
            side_effect=ProjectNotFoundError("ghost"),
        ):
            result = runner.invoke(run_app, ["start", "--project", "ghost"])

        assert result.exit_code != 0
        combined = result.output
        assert "ghost" in combined
        assert "Traceback" not in combined


# ---------------------------------------------------------------------------
# Logic-specific: --project precedence
# ---------------------------------------------------------------------------


class TestPrecedence:
    def test_explicit_flag_wins_over_env_var(
        self,
        runner: CliRunner,
        stub_resolved: ResolvedProject,
        fake_popen: dict,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("RYOTENKAI_PROJECT", "from-env")

        with patch(
            "ryotenkai_control.workspace.projects.adapter.resolve_project_launch_inputs",
            return_value=stub_resolved,
        ) as resolve_mock:
            runner.invoke(
                run_app, ["start", "--project", "explicit-flag"],
            )

        assert resolve_mock.call_args.args[0] == "explicit-flag"

    def test_env_var_used_when_no_flag(
        self,
        runner: CliRunner,
        stub_resolved: ResolvedProject,
        fake_popen: dict,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("RYOTENKAI_PROJECT", "from-env")

        with patch(
            "ryotenkai_control.workspace.projects.adapter.resolve_project_launch_inputs",
            return_value=stub_resolved,
        ) as resolve_mock:
            runner.invoke(run_app, ["start"])

        assert resolve_mock.call_args.args[0] == "from-env"

    def test_persisted_context_used_when_no_flag_no_env(
        self,
        runner: CliRunner,
        stub_resolved: ResolvedProject,
        fake_popen: dict,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("RYOTENKAI_PROJECT", raising=False)

        with (
            patch(
                "ryotenkai_control.cli_state.context_store.get_current_project",
                return_value="from-context",
            ),
            patch(
                "ryotenkai_control.workspace.projects.adapter.resolve_project_launch_inputs",
                return_value=stub_resolved,
            ) as resolve_mock,
        ):
            runner.invoke(run_app, ["start"])

        assert resolve_mock.call_args.args[0] == "from-context"


# ---------------------------------------------------------------------------
# Dry-run path
# ---------------------------------------------------------------------------


class TestDryRun:
    def test_dry_run_with_project_does_not_spawn(
        self, runner: CliRunner, stub_resolved: ResolvedProject,
    ) -> None:
        with (
            patch(
                "ryotenkai_control.workspace.projects.adapter.resolve_project_launch_inputs",
                return_value=stub_resolved,
            ),
            patch("ryotenkai_control.cli.commands.run.subprocess.Popen") as popen_cls,
        ):
            result = runner.invoke(
                run_app, ["start", "--project", "proj-a", "--dry-run"],
            )

        assert result.exit_code == 0, result.output
        # JSON plan is printed, project_id appears in it.
        assert "proj-a" in result.output
        assert '"mode": "start"' in result.output
        # Crucially: subprocess never spawned.
        popen_cls.assert_not_called()
