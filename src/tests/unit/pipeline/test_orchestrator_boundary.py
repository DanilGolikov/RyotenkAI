"""Tests for the Variant 1 orchestrator entry boundary.

Pins the dual-shape constructor: orchestrator accepts EITHER a
``config_path`` (legacy path-based) OR a pre-loaded ``PipelineConfig``
object plus ``metadata``/``env`` (Variant 1 hexagonal boundary).

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

    ``_source_path`` matters because the Variant 1 boundary derives
    ``config_path`` from it (downstream consumers like state_store
    record it on PipelineState).
    """
    config = MagicMock()
    if source_path is not None:
        config._source_path = source_path
    return config


# ---------------------------------------------------------------------------
# 1. Positive — both shapes construct successfully
# ---------------------------------------------------------------------------


class TestPositive:
    def test_legacy_config_path_constructor(self, tmp_path: Path) -> None:
        config_path = tmp_path / "config.yaml"
        config_path.write_text("model:\n  name: gpt2\n")
        config = _build_mock_config()
        secrets = MagicMock()

        with (
            patch("src.pipeline.bootstrap.pipeline_bootstrap.load_config", return_value=config),
            patch("src.pipeline.bootstrap.pipeline_bootstrap.load_secrets", return_value=secrets),
            patch.object(StartupValidator, "validate"),
            patch("src.community.preflight.run_preflight", return_value=MagicMock(ok=True)),
            patch.object(StageRegistry, "_build_stages", return_value=[]),
        ):
            orch = PipelineOrchestrator(config_path)
            assert orch.config is config
            assert orch.config_path == config_path

    def test_variant1_pre_loaded_config_constructor(self, tmp_path: Path) -> None:
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
            # config object passes through, ``config_path`` derived from
            # ``config._source_path`` (set by ``load_config``).
            assert orch.config is config
            assert orch.config_path == config_path

    def test_metadata_propagates_to_launch_preparator(self, tmp_path: Path) -> None:
        # Pin: ``metadata`` provided at construction time reaches
        # ``LaunchPreparator`` so that fresh-run init stamps it onto
        # PipelineState. (Where it then mirrors to MLflow as ``meta.*``.)
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
    def test_neither_config_nor_config_path_raises(self) -> None:
        with pytest.raises(ValueError, match="exactly one of"):
            PipelineOrchestrator()

    def test_both_config_and_config_path_raises(self, tmp_path: Path) -> None:
        config = _build_mock_config(source_path=tmp_path / "c.yaml")
        with pytest.raises(ValueError, match="exactly one of"):
            PipelineOrchestrator(config_path=tmp_path / "c.yaml", config=config)

    def test_pre_loaded_config_without_source_path_raises(self) -> None:
        # Pin: callers MUST go through ``load_config()`` so ``_source_path``
        # is set. Constructing a bare PipelineConfig object and passing
        # it raises a clear error rather than crashing in state_store
        # later.
        config = MagicMock()
        # Explicitly remove _source_path so getattr returns None.
        config._source_path = None
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

    def test_bootstrap_validates_mutual_exclusion(self, tmp_path: Path) -> None:
        # Pin: bootstrap-level check fires before orchestrator-level
        # check (defence in depth).
        run_ctx = MagicMock()
        attempt_controller = MagicMock()
        settings = MagicMock()
        with pytest.raises(ValueError, match="exactly one of"):
            PipelineBootstrap.build(
                run_ctx=run_ctx,
                settings=settings,
                attempt_controller=attempt_controller,
                on_stage_completed=lambda _name: None,
                on_shutdown_signal=lambda _name: None,
            )


# ---------------------------------------------------------------------------
# 5. Regression — existing tests using config_path keep passing
# ---------------------------------------------------------------------------


class TestRegression:
    def test_legacy_path_construction_still_works(self, tmp_path: Path) -> None:
        # Pin: the path-based constructor (positional, no kwargs)
        # remains valid throughout the deprecation window.
        config_path = tmp_path / "config.yaml"
        config_path.write_text("model:\n  name: gpt2\n")
        config = _build_mock_config()
        secrets = MagicMock()

        with (
            patch("src.pipeline.bootstrap.pipeline_bootstrap.load_config", return_value=config),
            patch("src.pipeline.bootstrap.pipeline_bootstrap.load_secrets", return_value=secrets),
            patch.object(StartupValidator, "validate"),
            patch("src.community.preflight.run_preflight", return_value=MagicMock(ok=True)),
            patch.object(StageRegistry, "_build_stages", return_value=[]),
        ):
            # Old positional-style construction.
            orch = PipelineOrchestrator(config_path)
            assert orch is not None


# ---------------------------------------------------------------------------
# 6. Logic-specific — pre-loaded config bypasses load_config
# ---------------------------------------------------------------------------


class TestLogicSpecific:
    def test_variant1_path_does_not_call_load_config(self, tmp_path: Path) -> None:
        # Pin: when the caller provides a pre-loaded config, bootstrap
        # MUST NOT re-invoke load_config (would double-load YAML and
        # potentially re-resolve integrations against a different
        # registry, defeating the boundary).
        config_path = tmp_path / "config.yaml"
        config_path.write_text("model:\n  name: gpt2\n")
        config = _build_mock_config(source_path=config_path)
        secrets = MagicMock()

        load_config_mock = MagicMock(return_value=config)
        with (
            patch("src.pipeline.bootstrap.pipeline_bootstrap.load_config", load_config_mock),
            patch("src.pipeline.bootstrap.pipeline_bootstrap.load_secrets", return_value=secrets),
            patch.object(StartupValidator, "validate"),
            patch("src.community.preflight.run_preflight", return_value=MagicMock(ok=True)),
            patch.object(StageRegistry, "_build_stages", return_value=[]),
        ):
            PipelineOrchestrator(config=config, metadata={"project_id": "x"})

        # load_config was NOT called by bootstrap.
        load_config_mock.assert_not_called()

    def test_legacy_path_invokes_load_config(self, tmp_path: Path) -> None:
        # Sibling pin: legacy path SHOULD call load_config (since
        # caller didn't pre-load).
        config_path = tmp_path / "config.yaml"
        config_path.write_text("model:\n  name: gpt2\n")
        config = _build_mock_config()
        secrets = MagicMock()

        load_config_mock = MagicMock(return_value=config)
        with (
            patch("src.pipeline.bootstrap.pipeline_bootstrap.load_config", load_config_mock),
            patch("src.pipeline.bootstrap.pipeline_bootstrap.load_secrets", return_value=secrets),
            patch.object(StartupValidator, "validate"),
            patch("src.community.preflight.run_preflight", return_value=MagicMock(ok=True)),
            patch.object(StageRegistry, "_build_stages", return_value=[]),
        ):
            PipelineOrchestrator(config_path)

        load_config_mock.assert_called_once_with(config_path)

    def test_pre_loaded_secrets_bypass_load_secrets(self, tmp_path: Path) -> None:
        # Pin: pre-loaded secrets ALSO short-circuit load_secrets.
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
            # Bootstrap-level direct call to test the secrets shortcut.
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
