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

from src.pipeline.bootstrap import PipelineBootstrap
from src.pipeline.bootstrap.startup_validator import StartupValidator
from src.pipeline.execution import StageRegistry
from src.pipeline.orchestrator import PipelineOrchestrator


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
            patch("src.pipeline.bootstrap.pipeline_bootstrap.load_secrets", return_value=secrets),
            patch.object(StartupValidator, "validate"),
            patch("src.community.preflight.run_preflight", return_value=MagicMock(ok=True)),
            patch.object(StageRegistry, "_build_stages", return_value=[]),
        ):
            orch = PipelineOrchestrator(
                config=config,
                metadata={"project_id": "x", "actor": "agent:claude"},
            )
            # config object passes through; ``config_path`` derived
            # from ``config._source_path`` (set by the loader).
            assert orch.config is config
            assert orch.config_path == config_path

    def test_metadata_propagates_to_launch_preparator(self, tmp_path: Path) -> None:
        # Pin: ``metadata`` reaches LaunchPreparator so fresh-run init
        # stamps it onto PipelineState (then mirrored to MLflow as
        # ``meta.*``).
        config_path = tmp_path / "config.yaml"
        config_path.write_text("model:\n  name: gpt2\n")
        config = _build_mock_config(source_path=config_path)
        secrets = MagicMock()

        with (
            patch("src.pipeline.bootstrap.pipeline_bootstrap.load_secrets", return_value=secrets),
            patch.object(StartupValidator, "validate"),
            patch("src.community.preflight.run_preflight", return_value=MagicMock(ok=True)),
            patch.object(StageRegistry, "_build_stages", return_value=[]),
        ):
            orch = PipelineOrchestrator(
                config=config,
                metadata={"project_id": "helixql-v7", "actor": "human"},
            )
            assert orch._launch_preparator._metadata == {
                "project_id": "helixql-v7",
                "actor": "human",
            }


# ---------------------------------------------------------------------------
# 2. Negative — wrong constructor shapes
# ---------------------------------------------------------------------------


class TestNegative:
    def test_pre_loaded_config_without_source_path_raises(self) -> None:
        # Pin: callers MUST go through ``load_pipeline_config()`` so
        # ``_source_path`` is set. Constructing a bare PipelineConfig
        # object and passing it raises a clear error rather than
        # crashing later in state_store.
        config = MagicMock()
        config._source_path = None  # explicit
        secrets = MagicMock()

        with (
            patch("src.pipeline.bootstrap.pipeline_bootstrap.load_secrets", return_value=secrets),
            patch.object(StartupValidator, "validate"),
            patch.object(StageRegistry, "_build_stages", return_value=[]),
            pytest.raises(ValueError, match="_source_path"),
        ):
            PipelineOrchestrator(config=config)


# ---------------------------------------------------------------------------
# 3. Boundary — empty / None metadata
# ---------------------------------------------------------------------------


class TestBoundary:
    def test_no_metadata_yields_empty_dict_in_preparator(self, tmp_path: Path) -> None:
        config_path = tmp_path / "config.yaml"
        config_path.write_text("model:\n  name: gpt2\n")
        config = _build_mock_config(source_path=config_path)
        secrets = MagicMock()

        with (
            patch("src.pipeline.bootstrap.pipeline_bootstrap.load_secrets", return_value=secrets),
            patch.object(StartupValidator, "validate"),
            patch("src.community.preflight.run_preflight", return_value=MagicMock(ok=True)),
            patch.object(StageRegistry, "_build_stages", return_value=[]),
        ):
            orch = PipelineOrchestrator(config=config)
            assert orch._launch_preparator._metadata == {}

    def test_explicit_empty_metadata_dict_yields_empty(self, tmp_path: Path) -> None:
        config_path = tmp_path / "config.yaml"
        config_path.write_text("model:\n  name: gpt2\n")
        config = _build_mock_config(source_path=config_path)
        secrets = MagicMock()

        with (
            patch("src.pipeline.bootstrap.pipeline_bootstrap.load_secrets", return_value=secrets),
            patch.object(StartupValidator, "validate"),
            patch("src.community.preflight.run_preflight", return_value=MagicMock(ok=True)),
            patch.object(StageRegistry, "_build_stages", return_value=[]),
        ):
            orch = PipelineOrchestrator(config=config, metadata={})
            assert orch._launch_preparator._metadata == {}


# ---------------------------------------------------------------------------
# 4. Invariants
# ---------------------------------------------------------------------------


class TestInvariants:
    def test_metadata_dict_is_copied_not_referenced(self, tmp_path: Path) -> None:
        # Pin: caller-side mutations of the metadata dict after
        # construction must NOT affect the orchestrator's state.
        config_path = tmp_path / "config.yaml"
        config_path.write_text("model:\n  name: gpt2\n")
        config = _build_mock_config(source_path=config_path)
        secrets = MagicMock()

        caller_dict: dict = {"k": "v"}
        with (
            patch("src.pipeline.bootstrap.pipeline_bootstrap.load_secrets", return_value=secrets),
            patch.object(StartupValidator, "validate"),
            patch("src.community.preflight.run_preflight", return_value=MagicMock(ok=True)),
            patch.object(StageRegistry, "_build_stages", return_value=[]),
        ):
            orch = PipelineOrchestrator(config=config, metadata=caller_dict)

        caller_dict["k"] = "MUTATED"
        caller_dict["new_key"] = "x"
        assert orch._launch_preparator._metadata == {"k": "v"}


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
            patch("src.pipeline.bootstrap.pipeline_bootstrap.load_secrets", load_secrets_mock),
            patch.object(StartupValidator, "validate"),
            patch("src.community.preflight.run_preflight", return_value=MagicMock(ok=True)),
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
