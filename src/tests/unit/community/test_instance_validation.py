"""Tests for the preflight instance-shape gate (B3).

Hand-edited YAML configs can wire a plugin with ``params`` /
``thresholds`` blocks that violate the manifest's ``params_schema``
(wrong type, out-of-range value, unknown enum option, etc.). The
runtime path then crashes mid-pipeline with whatever error the
plugin's ``__init__`` happens to assert. Preflight catches all of
these at second 0.
"""

from __future__ import annotations

from textwrap import dedent
from typing import TYPE_CHECKING

import pytest
import yaml

from src.community.catalog import CommunityCatalog
from src.community.preflight import (
    LaunchAbortedError,
    run_preflight,
    validate_instances,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def patched_catalog(tmp_community_root, monkeypatch):
    """Same shape as the fixture in ``test_preflight.py`` — patches both
    the module-level singleton and the package re-export so callers
    that imported ``catalog`` earlier see the temp instance."""
    import sys
    import src.community as community_pkg

    cat = CommunityCatalog(root=tmp_community_root)
    catalog_module = sys.modules["src.community.catalog"]
    monkeypatch.setattr(catalog_module, "catalog", cat)
    monkeypatch.setattr(community_pkg, "catalog", cat)
    return cat


def _load_pipeline_config(yaml_text: str):
    from src.utils.config import PipelineConfig

    return PipelineConfig.model_validate(yaml.safe_load(yaml_text))


def _fixture_config_yaml() -> str:
    """Reads the canonical test config — already validated by the
    main config-loader test suite, so we don't have to keep it in
    lockstep with schema evolution by hand."""
    from pathlib import Path as _Path

    fixture = (
        _Path(__file__).resolve().parents[3]
        / "tests/fixtures/configs/test_pipeline.yaml"
    )
    return fixture.read_text(encoding="utf-8")


def _make_eval_plugin_with_schema(make_plugin_dir, plugin_id: str) -> None:
    """Drop an evaluation plugin with a non-trivial params/thresholds
    schema so the validator has something to assert against."""
    extras = dedent("""
        [params_schema.timeout_seconds]
        type = "integer"
        min = 1
        max = 60
        default = 30

        [params_schema.mode]
        type = "enum"
        options = ["fast", "thorough"]
        default = "fast"

        [thresholds_schema.min_score]
        type = "number"
        min = 0.0
        max = 1.0
        default = 0.5
    """)
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
        manifest_extras=extras,
        plugin_source=plugin_source,
        class_name=f"{plugin_id.title().replace('_', '')}Plugin",
    )


def _attach_eval_instance(config, plugin_id: str, *, params=None, thresholds=None):
    from src.config.evaluation.schema import (
        EvaluationDatasetConfig,
        EvaluatorPluginConfig,
    )

    config.inference.enabled = True
    config.evaluation.enabled = True
    if config.evaluation.dataset is None:
        config.evaluation.dataset = EvaluationDatasetConfig(path="data/eval.jsonl")
    config.evaluation.evaluators.plugins = [
        EvaluatorPluginConfig(
            id="judge",
            plugin=plugin_id,
            enabled=True,
            params=params or {},
            thresholds=thresholds or {},
        )
    ]


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_valid_instance_produces_no_errors(
    patched_catalog,
    make_plugin_dir,
) -> None:
    _make_eval_plugin_with_schema(make_plugin_dir, "scored")
    patched_catalog.reload()

    config = _load_pipeline_config(_fixture_config_yaml())
    _attach_eval_instance(
        config, "scored",
        params={"timeout_seconds": 10, "mode": "fast"},
        thresholds={"min_score": 0.7},
    )
    assert validate_instances(config) == []


# ---------------------------------------------------------------------------
# Each violation class
# ---------------------------------------------------------------------------


def test_wrong_type_surfaces_error(
    patched_catalog,
    make_plugin_dir,
) -> None:
    _make_eval_plugin_with_schema(make_plugin_dir, "scored")
    patched_catalog.reload()

    config = _load_pipeline_config(_fixture_config_yaml())
    _attach_eval_instance(
        config, "scored",
        params={"timeout_seconds": "not-an-int", "mode": "fast"},
    )
    errors = validate_instances(config)
    assert any(e.location == "params.timeout_seconds" for e in errors), errors


def test_out_of_range_surfaces_error(
    patched_catalog,
    make_plugin_dir,
) -> None:
    _make_eval_plugin_with_schema(make_plugin_dir, "scored")
    patched_catalog.reload()

    config = _load_pipeline_config(_fixture_config_yaml())
    _attach_eval_instance(
        config, "scored",
        params={"timeout_seconds": 999, "mode": "fast"},
    )
    errors = validate_instances(config)
    assert any(e.location == "params.timeout_seconds" for e in errors)


def test_unknown_enum_option_surfaces_error(
    patched_catalog,
    make_plugin_dir,
) -> None:
    _make_eval_plugin_with_schema(make_plugin_dir, "scored")
    patched_catalog.reload()

    config = _load_pipeline_config(_fixture_config_yaml())
    _attach_eval_instance(
        config, "scored",
        params={"timeout_seconds": 10, "mode": "ludicrous"},
    )
    errors = validate_instances(config)
    assert any(e.location == "params.mode" for e in errors)


def test_thresholds_violations_are_namespaced(
    patched_catalog,
    make_plugin_dir,
) -> None:
    """thresholds errors carry the ``thresholds.`` prefix so the UI
    can route them to the right Configure modal section."""
    _make_eval_plugin_with_schema(make_plugin_dir, "scored")
    patched_catalog.reload()

    config = _load_pipeline_config(_fixture_config_yaml())
    _attach_eval_instance(
        config, "scored",
        params={"timeout_seconds": 10, "mode": "fast"},
        thresholds={"min_score": 5.0},
    )
    errors = validate_instances(config)
    assert any(e.location == "thresholds.min_score" for e in errors)


def test_unknown_field_rejected(
    patched_catalog,
    make_plugin_dir,
) -> None:
    """The schema is closed (``additionalProperties: false``) so an
    unknown key trips the validator — protects against silent typos."""
    _make_eval_plugin_with_schema(make_plugin_dir, "scored")
    patched_catalog.reload()

    config = _load_pipeline_config(_fixture_config_yaml())
    _attach_eval_instance(
        config, "scored",
        params={"timeout_seconds": 10, "mode": "fast", "typoed_param": True},
    )
    errors = validate_instances(config)
    # Unknown keys land at the params-block level (no specific path).
    assert any(e.location == "params" for e in errors)


# ---------------------------------------------------------------------------
# Multiple errors per pass
# ---------------------------------------------------------------------------


def test_iter_errors_reports_all_violations(
    patched_catalog,
    make_plugin_dir,
) -> None:
    """Use ``Draft7Validator.iter_errors`` not ``validate`` so the user
    sees every problem at once instead of fixing them one by one."""
    _make_eval_plugin_with_schema(make_plugin_dir, "scored")
    patched_catalog.reload()

    config = _load_pipeline_config(_fixture_config_yaml())
    _attach_eval_instance(
        config, "scored",
        params={"timeout_seconds": 999, "mode": "ludicrous"},
        thresholds={"min_score": 5.0},
    )
    errors = validate_instances(config)
    locations = {e.location for e in errors}
    assert {"params.timeout_seconds", "params.mode", "thresholds.min_score"} <= locations


# ---------------------------------------------------------------------------
# Combined report
# ---------------------------------------------------------------------------


def test_run_preflight_combines_envs_and_instances(
    patched_catalog,
    make_plugin_dir,
    monkeypatch,
) -> None:
    """``run_preflight`` returns one dataclass with both halves."""
    _make_eval_plugin_with_schema(make_plugin_dir, "scored")
    patched_catalog.reload()

    config = _load_pipeline_config(_fixture_config_yaml())
    _attach_eval_instance(
        config, "scored",
        params={"timeout_seconds": -5, "mode": "fast"},
    )
    report = run_preflight(config)
    assert report.ok is False
    assert report.missing_envs == []
    assert any(e.location == "params.timeout_seconds" for e in report.instance_errors)


def test_launch_aborted_error_carries_both_lists(
    patched_catalog,
    make_plugin_dir,
) -> None:
    _make_eval_plugin_with_schema(make_plugin_dir, "scored")
    patched_catalog.reload()

    config = _load_pipeline_config(_fixture_config_yaml())
    _attach_eval_instance(
        config, "scored",
        params={"timeout_seconds": 99999, "mode": "fast"},
    )
    report = run_preflight(config)
    err = LaunchAbortedError(
        missing=report.missing_envs,
        instance_errors=report.instance_errors,
    )
    assert err.instance_errors == report.instance_errors
    assert "scored:params.timeout_seconds" in str(err)
