"""Tests for the orchestrator entry boundary.

Pins the keyword-only constructor: ``PipelineOrchestrator(config=...,
metadata=..., env=..., ...)`` accepts a fully-loaded
:class:`PipelineConfig` plus optional adapter context. There is no
legacy ``config_path`` constructor — callers (CLI / API / project
adapter) load the YAML upstream via
:func:`src.workspace.integrations.loader.load_pipeline_config` and
hand the resolved object in.

Test categories: positive, negative, boundary, invariants, regression,
logic-specific.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ryotenkai_control.pipeline.bootstrap import PipelineBootstrap
from ryotenkai_control.pipeline.bootstrap.startup_validator import StartupValidator
from ryotenkai_control.pipeline.execution import StageRegistry
from ryotenkai_control.pipeline.orchestrator import PipelineOrchestrator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _build_mock_config(*, source_path: Path | None = None) -> MagicMock:
    """Mock PipelineConfig stub. Only fields touched by bootstrap are set.

    ``_source_path`` is REQUIRED — bootstrap reads it to record the
    canonical config path on PipelineState. Callers go through
    ``load_pipeline_config`` which sets it; tests mimic that.
    """
    config = MagicMock()
    if source_path is not None:
        config._source_path = source_path
    return config


# ---------------------------------------------------------------------------
# 1. Positive — keyword shape constructs successfully
# ---------------------------------------------------------------------------


class TestPositive:
    def test_keyword_constructor_with_pre_loaded_config(
        self, tmp_path: Path,
    ) -> None:
        config_path = tmp_path / "config.yaml"
        config_path.write_text("model:\n  name: gpt2\n")
        config = _build_mock_config(source_path=config_path)
        secrets = MagicMock()

        with (
            patch("ryotenkai_control.pipeline.bootstrap.pipeline_bootstrap.load_secrets", return_value=secrets),
            patch.object(StartupValidator, "validate"),
            patch("ryotenkai_community.preflight.run_preflight", return_value=MagicMock(ok=True)),
            patch.object(StageRegistry, "_build_stages", return_value=[]),
        ):
            orch = PipelineOrchestrator(config=config)
            # config object passes through; ``config_path`` derived
            # from ``config._source_path`` (set by the loader).
            assert orch.config is config
            assert orch.config_path == config_path

    def test_metadata_from_env_propagates_to_launch_preparator(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Pin: ``RYOTENKAI_*`` env vars reach LaunchPreparator so
        # fresh-run init stamps them onto PipelineState (then mirrored
        # to MLflow as ``meta.*``). The launcher (CLI / API) is the
        # single source of truth for these values.
        monkeypatch.setenv("RYOTENKAI_PROJECT_ID", "helixql-v7")
        monkeypatch.setenv("RYOTENKAI_ACTOR", "human")
        monkeypatch.delenv("RYOTENKAI_CONFIG_VERSION_HASH", raising=False)
        monkeypatch.delenv("RYOTENKAI_CONFIG_OVERRIDE_PATH", raising=False)
        config_path = tmp_path / "config.yaml"
        config_path.write_text("model:\n  name: gpt2\n")
        config = _build_mock_config(source_path=config_path)
        secrets = MagicMock()

        with (
            patch("ryotenkai_control.pipeline.bootstrap.pipeline_bootstrap.load_secrets", return_value=secrets),
            patch.object(StartupValidator, "validate"),
            patch("ryotenkai_community.preflight.run_preflight", return_value=MagicMock(ok=True)),
            patch.object(StageRegistry, "_build_stages", return_value=[]),
        ):
            orch = PipelineOrchestrator(config=config)
            assert orch._launch_preparator._metadata == {
                "project_id": "helixql-v7",
                "actor": "human",
            }


# ---------------------------------------------------------------------------
# 2. Negative — wrong constructor shapes
# ---------------------------------------------------------------------------


class TestNegative:
    def test_legacy_env_kwarg_rejected(self, tmp_path: Path) -> None:
        """Architectural guardrail: ``env=`` was deprecated in step 4 and
        removed in step 6. Passing it must surface a clean ``TypeError``
        so callers updating to subprocess-launch hit a hard signal."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("model:\n  name: gpt2\n")
        config = _build_mock_config(source_path=config_path)
        with pytest.raises(TypeError, match="unexpected keyword argument 'env'"):
            PipelineOrchestrator(config=config, env={"FOO": "bar"})  # type: ignore[call-arg]

    def test_legacy_metadata_kwarg_rejected(self, tmp_path: Path) -> None:
        """Architectural guardrail: ``metadata=`` source moved to
        ``RYOTENKAI_*`` env vars in step 4 and the param was removed in
        step 6. Bootstrap reads from env via :func:`read_metadata_from_env`."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("model:\n  name: gpt2\n")
        config = _build_mock_config(source_path=config_path)
        with pytest.raises(TypeError, match="unexpected keyword argument 'metadata'"):
            PipelineOrchestrator(config=config, metadata={"project_id": "x"})  # type: ignore[call-arg]

    def test_pre_loaded_config_without_source_path_raises(self) -> None:
        # Pin: callers MUST go through ``load_pipeline_config()`` so
        # ``_source_path`` is set. Constructing a bare PipelineConfig
        # object and passing it raises a clear error rather than
        # crashing later in state_store.
        config = MagicMock()
        config._source_path = None  # explicit
        secrets = MagicMock()

        with (
            patch("ryotenkai_control.pipeline.bootstrap.pipeline_bootstrap.load_secrets", return_value=secrets),
            patch.object(StartupValidator, "validate"),
            patch.object(StageRegistry, "_build_stages", return_value=[]),
            pytest.raises(ValueError, match="_source_path"),
        ):
            PipelineOrchestrator(config=config)


# ---------------------------------------------------------------------------
# 3. Boundary — metadata sourced from RYOTENKAI_* env vars
# ---------------------------------------------------------------------------


class TestBoundary:
    def test_no_env_vars_yields_empty_metadata(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Anonymous run (launcher set no project context) → metadata is ``{}``."""
        for var in (
            "RYOTENKAI_PROJECT_ID", "RYOTENKAI_ACTOR",
            "RYOTENKAI_CONFIG_VERSION_HASH", "RYOTENKAI_CONFIG_OVERRIDE_PATH",
        ):
            monkeypatch.delenv(var, raising=False)
        config_path = tmp_path / "config.yaml"
        config_path.write_text("model:\n  name: gpt2\n")
        config = _build_mock_config(source_path=config_path)
        secrets = MagicMock()

        with (
            patch("ryotenkai_control.pipeline.bootstrap.pipeline_bootstrap.load_secrets", return_value=secrets),
            patch.object(StartupValidator, "validate"),
            patch("ryotenkai_community.preflight.run_preflight", return_value=MagicMock(ok=True)),
            patch.object(StageRegistry, "_build_stages", return_value=[]),
        ):
            orch = PipelineOrchestrator(config=config)
            assert orch._launch_preparator._metadata == {}

    def test_env_vars_populate_metadata(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Launcher sets RYOTENKAI_* → bootstrap stamps them onto preparator."""
        monkeypatch.setenv("RYOTENKAI_PROJECT_ID", "my-proj")
        monkeypatch.setenv("RYOTENKAI_ACTOR", "agent:web-ui")
        monkeypatch.setenv("RYOTENKAI_CONFIG_VERSION_HASH", "deadbeef")
        config_path = tmp_path / "config.yaml"
        config_path.write_text("model:\n  name: gpt2\n")
        config = _build_mock_config(source_path=config_path)
        secrets = MagicMock()

        with (
            patch("ryotenkai_control.pipeline.bootstrap.pipeline_bootstrap.load_secrets", return_value=secrets),
            patch.object(StartupValidator, "validate"),
            patch("ryotenkai_community.preflight.run_preflight", return_value=MagicMock(ok=True)),
            patch.object(StageRegistry, "_build_stages", return_value=[]),
        ):
            orch = PipelineOrchestrator(config=config)
            md = orch._launch_preparator._metadata
            assert md["project_id"] == "my-proj"
            assert md["actor"] == "agent:web-ui"
            assert md["config_version_hash"] == "deadbeef"


# ---------------------------------------------------------------------------
# 4. Invariants
# ---------------------------------------------------------------------------


class TestInvariants:
    def test_empty_string_env_vars_skipped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Whitespace-only env vars are treated as unset — no spurious keys
        like ``actor: ""`` leaking into metadata."""
        monkeypatch.setenv("RYOTENKAI_PROJECT_ID", "p1")
        monkeypatch.setenv("RYOTENKAI_ACTOR", "   ")
        monkeypatch.delenv("RYOTENKAI_CONFIG_VERSION_HASH", raising=False)
        config_path = tmp_path / "config.yaml"
        config_path.write_text("model:\n  name: gpt2\n")
        config = _build_mock_config(source_path=config_path)
        secrets = MagicMock()

        with (
            patch("ryotenkai_control.pipeline.bootstrap.pipeline_bootstrap.load_secrets", return_value=secrets),
            patch.object(StartupValidator, "validate"),
            patch("ryotenkai_community.preflight.run_preflight", return_value=MagicMock(ok=True)),
            patch.object(StageRegistry, "_build_stages", return_value=[]),
        ):
            orch = PipelineOrchestrator(config=config)
            md = orch._launch_preparator._metadata
            assert md == {"project_id": "p1"}


# ---------------------------------------------------------------------------
# 5. Logic-specific — bootstrap consumes pre-loaded config as-is
# ---------------------------------------------------------------------------


class TestLogicSpecific:
    def test_pre_loaded_secrets_bypass_load_secrets(self, tmp_path: Path) -> None:
        # Pin: pre-loaded secrets short-circuit load_secrets — caller
        # supplies the secrets, bootstrap uses them as-is.
        config_path = tmp_path / "config.yaml"
        config_path.write_text("model:\n  name: gpt2\n")
        config = _build_mock_config(source_path=config_path)
        custom_secrets = MagicMock()

        load_secrets_mock = MagicMock(return_value=MagicMock())
        with (
            patch("ryotenkai_control.pipeline.bootstrap.pipeline_bootstrap.load_secrets", load_secrets_mock),
            patch.object(StartupValidator, "validate"),
            patch("ryotenkai_community.preflight.run_preflight", return_value=MagicMock(ok=True)),
            patch.object(StageRegistry, "_build_stages", return_value=[]),
        ):
            run_ctx = MagicMock()
            attempt_controller = MagicMock()
            attempt_controller.adopt_state = MagicMock()
            settings = MagicMock()
            PipelineBootstrap.build(
                config=config,
                secrets=custom_secrets,
                run_ctx=run_ctx,
                settings=settings,
                attempt_controller=attempt_controller,
                on_stage_completed=lambda _name: None,
                on_shutdown_signal=lambda _name: None,
            )
        load_secrets_mock.assert_not_called()
