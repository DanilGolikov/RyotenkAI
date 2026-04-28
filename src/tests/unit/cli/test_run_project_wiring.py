"""CLI tests for ``ryotenkai run start --project`` wiring (Step 8).

These pin the matrix called out in the plan §6.4:

| Scenario                   | Behaviour                                        |
|----------------------------|--------------------------------------------------|
| ``run start -c X``         | anonymous run, no adapter call                   |
| ``run start -p A``         | adapter loads ``A``                              |
| ``run start -c X -p A``    | adapter loads ``A`` with ``X`` as override       |
| ``run start`` (nothing)    | helpful error                                    |
| ``run start -p missing``   | ``ProjectNotFoundError`` → clean ``die``         |
| ``-p flag > RYOTENKAI_PROJECT > project use`` | precedence pinned    |

We use Typer's :class:`CliRunner` and mock the orchestrator + adapter
so tests never touch disk-state. The wiring layer is the unit under
test; orchestration internals are out of scope.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from src.cli.commands.run import run_app
from src.workspace.projects.adapter import (
    ProjectInputs,
    ProjectNotFoundError,
)


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def stub_inputs(tmp_path: Path) -> ProjectInputs:
    """Minimal :class:`ProjectInputs` the orchestrator can swallow."""
    cfg = MagicMock(name="PipelineConfig")
    runs_dir = tmp_path / "project_runs"
    runs_dir.mkdir()
    return ProjectInputs(
        config=cfg,
        runs_base_dir=runs_dir,
        env={"PROJECT_KEY": "v"},
        metadata={"project_id": "proj-a", "actor": "tester"},
    )


# ---------------------------------------------------------------------------
# Positive — both ``run start`` paths succeed
# ---------------------------------------------------------------------------


class TestPositive:
    def test_run_start_with_project_uses_adapter(
        self, runner: CliRunner, stub_inputs: ProjectInputs,
        tmp_path: Path,
    ) -> None:
        config = tmp_path / "x.yaml"
        config.write_text("# stub\n", encoding="utf-8")

        with (
            patch(
                "src.workspace.projects.adapter.load_project_inputs",
                return_value=stub_inputs,
            ) as load_mock,
            patch(
                "src.pipeline.orchestrator.PipelineOrchestrator"
            ) as orch_cls,
        ):
            orch = orch_cls.return_value
            orch.run.return_value.is_success.return_value = True
            result = runner.invoke(run_app, ["start", "--project", "proj-a"])

        assert result.exit_code == 0, result.output
        load_mock.assert_called_once()
        # Orchestrator gets the new keyword shape from Step 3.
        kwargs = orch_cls.call_args.kwargs
        assert kwargs.get("config") is stub_inputs.config
        assert kwargs.get("env") == stub_inputs.env
        assert kwargs.get("metadata") == stub_inputs.metadata

    def test_run_start_with_config_only_skips_adapter(
        self, runner: CliRunner, tmp_path: Path,
    ) -> None:
        config = tmp_path / "x.yaml"
        config.write_text("# stub\n", encoding="utf-8")
        cfg_obj = MagicMock(name="PipelineConfig")

        with (
            patch(
                "src.workspace.projects.adapter.load_project_inputs",
            ) as load_mock,
            patch(
                "src.workspace.integrations.loader.load_pipeline_config",
                return_value=cfg_obj,
            ) as cli_loader,
            patch(
                "src.pipeline.orchestrator.PipelineOrchestrator"
            ) as orch_cls,
        ):
            orch = orch_cls.return_value
            orch.run.return_value.is_success.return_value = True
            result = runner.invoke(run_app, ["start", "--config", str(config)])

        assert result.exit_code == 0, result.output
        # No project adapter call on the anonymous path.
        load_mock.assert_not_called()
        # CLI loader was called (with integration-resolution pass).
        cli_loader.assert_called_once()
        # Orchestrator gets the keyword shape with the loaded config.
        assert orch_cls.call_args.kwargs.get("config") is cfg_obj

    def test_run_start_config_plus_project_treats_config_as_override(
        self, runner: CliRunner, stub_inputs: ProjectInputs, tmp_path: Path,
    ) -> None:
        override = tmp_path / "experimental.yaml"
        override.write_text("# override\n", encoding="utf-8")

        with (
            patch(
                "src.workspace.projects.adapter.load_project_inputs",
                return_value=stub_inputs,
            ) as load_mock,
            patch(
                "src.pipeline.orchestrator.PipelineOrchestrator"
            ) as orch_cls,
        ):
            orch = orch_cls.return_value
            orch.run.return_value.is_success.return_value = True
            result = runner.invoke(
                run_app,
                ["start", "--config", str(override), "--project", "proj-a"],
            )

        assert result.exit_code == 0, result.output
        load_mock.assert_called_once()
        # The config override flowed in as ``config_override`` kwarg.
        called_kwargs = load_mock.call_args.kwargs
        assert called_kwargs["config_override"] == override


# ---------------------------------------------------------------------------
# Negative — surfacing clean error messages
# ---------------------------------------------------------------------------


class TestNegative:
    def test_no_args_yields_helpful_error(self, runner: CliRunner) -> None:
        result = runner.invoke(run_app, ["start"])
        assert result.exit_code != 0
        # ``die`` writes to stderr; the message tells the user what to do.
        combined = result.output
        assert "missing required argument" in combined.lower() or (
            "config" in combined.lower() and "run-dir" in combined.lower()
        )

    def test_unknown_project_yields_clean_die(
        self, runner: CliRunner,
    ) -> None:
        with patch(
            "src.workspace.projects.adapter.load_project_inputs",
            side_effect=ProjectNotFoundError("ghost"),
        ):
            result = runner.invoke(run_app, ["start", "--project", "ghost"])

        assert result.exit_code != 0
        combined = result.output
        assert "ghost" in combined
        assert "Traceback" not in combined  # CLEAN — no traceback


# ---------------------------------------------------------------------------
# Logic-specific: --project precedence
# ---------------------------------------------------------------------------


class TestPrecedence:
    def test_explicit_flag_wins_over_env_var(
        self,
        runner: CliRunner,
        stub_inputs: ProjectInputs,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("RYOTENKAI_PROJECT", "from-env")

        with (
            patch(
                "src.workspace.projects.adapter.load_project_inputs",
                return_value=stub_inputs,
            ) as load_mock,
            patch(
                "src.pipeline.orchestrator.PipelineOrchestrator"
            ) as orch_cls,
        ):
            orch = orch_cls.return_value
            orch.run.return_value.is_success.return_value = True
            runner.invoke(
                run_app, ["start", "--project", "explicit-flag"],
            )

        # The flag wins.
        assert load_mock.call_args.args[0] == "explicit-flag"

    def test_env_var_used_when_no_flag(
        self,
        runner: CliRunner,
        stub_inputs: ProjectInputs,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("RYOTENKAI_PROJECT", "from-env")

        with (
            patch(
                "src.workspace.projects.adapter.load_project_inputs",
                return_value=stub_inputs,
            ) as load_mock,
            patch(
                "src.pipeline.orchestrator.PipelineOrchestrator"
            ) as orch_cls,
        ):
            orch = orch_cls.return_value
            orch.run.return_value.is_success.return_value = True
            runner.invoke(run_app, ["start"])

        assert load_mock.call_args.args[0] == "from-env"

    def test_persisted_context_used_when_no_flag_no_env(
        self,
        runner: CliRunner,
        stub_inputs: ProjectInputs,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("RYOTENKAI_PROJECT", raising=False)

        with (
            patch(
                "src.cli_state.context_store.get_current_project",
                return_value="from-context",
            ),
            patch(
                "src.workspace.projects.adapter.load_project_inputs",
                return_value=stub_inputs,
            ) as load_mock,
            patch(
                "src.pipeline.orchestrator.PipelineOrchestrator"
            ) as orch_cls,
        ):
            orch = orch_cls.return_value
            orch.run.return_value.is_success.return_value = True
            runner.invoke(run_app, ["start"])

        assert load_mock.call_args.args[0] == "from-context"


# ---------------------------------------------------------------------------
# Dry-run path
# ---------------------------------------------------------------------------


class TestDryRun:
    def test_dry_run_with_project_does_not_construct_orchestrator(
        self, runner: CliRunner, stub_inputs: ProjectInputs,
    ) -> None:
        with (
            patch(
                "src.workspace.projects.adapter.load_project_inputs",
                return_value=stub_inputs,
            ),
            patch(
                "src.pipeline.orchestrator.PipelineOrchestrator"
            ) as orch_cls,
        ):
            result = runner.invoke(
                run_app, ["start", "--project", "proj-a", "--dry-run"],
            )

        assert result.exit_code == 0, result.output
        assert "dry-run" in result.output.lower()
        assert "proj-a" in result.output
        # Crucially: orchestrator constructor never called.
        orch_cls.assert_not_called()
