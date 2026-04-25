"""Tests for stale-plugin detection (PR14 / E1).

Stale = a plugin id referenced in the YAML config that's no longer in
the community catalog (manifest deleted, folder renamed, plugin renamed
in code without a config migration). The detector lets the UI render a
"Remove from config" button per stale row instead of failing the run
mid-pipeline with a clear-but-late "plugin not found" error.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest
import yaml

from src.community.catalog import CommunityCatalog
from src.community.stale_plugins import StalePluginRef, find_stale_plugins


@pytest.fixture
def patched_catalog(tmp_community_root, monkeypatch):
    """Patch both the module-level singleton and the package re-export
    so callers that imported ``catalog`` earlier see the temp instance.
    Same shape as in test_preflight / test_instance_validation."""
    import sys
    import src.community as community_pkg

    cat = CommunityCatalog(root=tmp_community_root)
    catalog_module = sys.modules["src.community.catalog"]
    monkeypatch.setattr(catalog_module, "catalog", cat)
    monkeypatch.setattr(community_pkg, "catalog", cat)
    return cat


def _load_pipeline_config(yaml_text: str):
    """Load the fixture YAML and force ``reports.sections=[]`` so tests
    that focus on validation/evaluation/reward don't inherit the
    default report-sections list (whose 13 built-in plugins live under
    the *real* community/ root, not the temp catalog these tests use).
    Tests that exercise reports staleness override the field
    explicitly.
    """
    from src.config.reports.schema import ReportsConfig
    from src.utils.config import PipelineConfig

    config = PipelineConfig.model_validate(yaml.safe_load(yaml_text))
    config.reports = ReportsConfig(sections=[])
    return config


def _fixture_config_yaml() -> str:
    fixture = (
        Path(__file__).resolve().parents[3]
        / "tests/fixtures/configs/test_pipeline.yaml"
    )
    return fixture.read_text(encoding="utf-8")


def _make_eval_plugin(make_plugin_dir, plugin_id: str) -> None:
    plugin_source = dedent(f"""
        from src.evaluation.plugins.base import EvalResult, EvaluatorPlugin

        class {plugin_id.title().replace('_', '')}Plugin(EvaluatorPlugin):
            def evaluate(self, samples):
                return EvalResult(plugin_name="{plugin_id}", passed=True)

            def get_recommendations(self, result):
                return []
    """)
    make_plugin_dir(
        "evaluation",
        plugin_id,
        plugin_source=plugin_source,
        class_name=f"{plugin_id.title().replace('_', '')}Plugin",
    )


def _attach_eval_instance(config, plugin_id: str, *, instance_id: str = "main"):
    from src.config.evaluation.schema import (
        EvaluationDatasetConfig,
        EvaluatorPluginConfig,
    )

    config.inference.enabled = True
    config.evaluation.enabled = True
    if config.evaluation.dataset is None:
        config.evaluation.dataset = EvaluationDatasetConfig(path="data/eval.jsonl")
    config.evaluation.evaluators.plugins = [
        EvaluatorPluginConfig(id=instance_id, plugin=plugin_id, enabled=True)
    ]


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_no_references_no_stale(patched_catalog) -> None:
    config = _load_pipeline_config(_fixture_config_yaml())
    assert find_stale_plugins(config) == []


def test_registered_plugin_not_flagged(
    patched_catalog,
    make_plugin_dir,
) -> None:
    _make_eval_plugin(make_plugin_dir, "scorer")
    patched_catalog.reload()

    config = _load_pipeline_config(_fixture_config_yaml())
    _attach_eval_instance(config, "scorer")
    assert find_stale_plugins(config) == []


# ---------------------------------------------------------------------------
# Per-kind stale detection
# ---------------------------------------------------------------------------


def test_evaluation_plugin_missing_from_catalog_flagged(
    patched_catalog,
) -> None:
    """An evaluator plugin referenced but not on disk surfaces with the
    full deep-link location so the UI can highlight the right row."""
    config = _load_pipeline_config(_fixture_config_yaml())
    _attach_eval_instance(config, "ghost_evaluator", instance_id="judge")

    stale = find_stale_plugins(config)
    assert len(stale) == 1
    only = stale[0]
    assert only.plugin_kind == "evaluation"
    assert only.plugin_name == "ghost_evaluator"
    assert only.instance_id == "judge"
    assert "evaluation.evaluators.plugins" in only.location


def test_validation_plugin_stale_flagged(
    patched_catalog,
) -> None:
    config = _load_pipeline_config(_fixture_config_yaml())
    primary = next(iter(config.datasets.values()))
    primary_id = next(iter(config.datasets.keys()))
    from src.config.datasets.validation import (
        DatasetValidationPluginConfig,
        DatasetValidationsConfig,
    )

    primary.validations = DatasetValidationsConfig(
        plugins=[
            DatasetValidationPluginConfig(
                id="missing_main",
                plugin="ghost_validator",
            )
        ]
    )

    stale = find_stale_plugins(config)
    assert any(
        s.plugin_kind == "validation"
        and s.plugin_name == "ghost_validator"
        and primary_id in s.location
        for s in stale
    )


def test_reward_plugin_stale_flagged(
    patched_catalog,
) -> None:
    config = _load_pipeline_config(_fixture_config_yaml())
    # Replace the SFT phase with a GRPO one referencing a ghost reward.
    from src.config.training.strategies.phase import StrategyPhaseConfig

    from src.config.training.hyperparams import PhaseHyperparametersConfig

    config.training.strategies = [
        StrategyPhaseConfig(
            strategy_type="grpo",
            dataset="default",
            hyperparams=PhaseHyperparametersConfig(
                epochs=1,
                learning_rate=2e-4,
                max_prompt_length=1024,
                max_completion_length=512,
            ),
            params={
                "reward_plugin": "ghost_reward",
                "reward_params": {},
            },
        )
    ]

    stale = find_stale_plugins(config)
    assert any(
        s.plugin_kind == "reward" and s.plugin_name == "ghost_reward"
        for s in stale
    )


def test_report_section_stale_flagged(
    patched_catalog,
) -> None:
    config = _load_pipeline_config(_fixture_config_yaml())
    from src.config.reports.schema import ReportsConfig

    config.reports = ReportsConfig(sections=["ghost_report"])
    stale = find_stale_plugins(config)
    assert any(
        s.plugin_kind == "reports" and s.plugin_name == "ghost_report"
        for s in stale
    )


def test_multiple_stale_refs_returned_together(
    patched_catalog,
) -> None:
    """A single config with stale entries across kinds returns one
    flat list — the UI renders it as a single banner with per-row
    deep-links."""
    config = _load_pipeline_config(_fixture_config_yaml())
    from src.config.reports.schema import ReportsConfig

    _attach_eval_instance(config, "ghost_eval", instance_id="judge")
    config.reports = ReportsConfig(sections=["ghost_section"])

    stale = find_stale_plugins(config)
    kinds = sorted({s.plugin_kind for s in stale})
    assert kinds == ["evaluation", "reports"]


# ---------------------------------------------------------------------------
# Type sanity
# ---------------------------------------------------------------------------


def test_returns_dataclass_instances(patched_catalog) -> None:
    config = _load_pipeline_config(_fixture_config_yaml())
    _attach_eval_instance(config, "ghost_eval")
    [only] = find_stale_plugins(config)
    assert isinstance(only, StalePluginRef)
