"""Tests for the pre-launch env gate (A6 in cozy-booping-walrus).

Builds tiny synthetic catalogs via the D2 fixtures and a minimal
:class:`PipelineConfig` so the gate is exercised in isolation —
without spinning up real plugins, the orchestrator, or the API stack.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest
import yaml

from src.community.catalog import CommunityCatalog
from src.community.preflight import (
    LaunchAbortedError,
    MissingEnv,
    validate_required_env,
)


_FIXTURE_CONFIG_PATH = Path(__file__).resolve().parents[3] / "tests/fixtures/configs/test_pipeline.yaml"


def _minimal_pipeline_config_yaml() -> str:
    """Return the canonical test pipeline YAML — already a valid
    :class:`PipelineConfig` — for tests to extend with plugin blocks."""
    return _FIXTURE_CONFIG_PATH.read_text(encoding="utf-8")


def _load_pipeline_config(yaml_text: str):
    from src.utils.config import PipelineConfig

    return PipelineConfig.model_validate(yaml.safe_load(yaml_text))


@pytest.fixture
def patched_catalog(tmp_community_root, monkeypatch):
    """Swap ``src.community.catalog.catalog`` for a temp-rooted instance.

    Lets every test build a fresh on-disk plugin tree under
    :func:`tmp_community_root` and run preflight against it without
    polluting the real catalog. The fixture restores the original
    singleton on teardown.
    """
    cat = CommunityCatalog(root=tmp_community_root)
    # In ``src/community/__init__.py`` the submodule ``catalog`` is
    # shadowed by the singleton instance also named ``catalog``. Using
    # ``sys.modules`` is the unambiguous way to grab the actual module
    # and patch its top-level singleton; the package re-export gets
    # patched separately so readers that did
    # ``from src.community import catalog`` see the new instance too.
    import sys
    import src.community as community_pkg

    catalog_module = sys.modules["src.community.catalog"]
    monkeypatch.setattr(catalog_module, "catalog", cat)
    monkeypatch.setattr(community_pkg, "catalog", cat)
    return cat


def _make_eval_plugin(make_plugin_dir, plugin_id: str, env_name: str, *, optional: bool = False, secret: bool = True):
    """Drop a minimal evaluation plugin that declares one required env."""
    extras = dedent(f"""
        [[required_env]]
        name = "{env_name}"
        description = "API key for {plugin_id}"
        optional = {str(optional).lower()}
        secret = {str(secret).lower()}
        managed_by = ""
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


def _attach_eval_plugin(config, plugin_id: str, *, instance_id: str = "main", enabled: bool = True) -> None:
    """Mutate ``config`` so its evaluation block runs the given plugin.

    Operates on the already-validated :class:`PipelineConfig` rather
    than the YAML text — appending an ``evaluation:`` block via string
    concatenation collides with the fixture's existing inference/eval
    sections and is awkward to keep in sync with schema evolution.
    """
    from src.config.evaluation.schema import EvaluatorPluginConfig

    config.inference.enabled = True
    config.evaluation.enabled = True
    if config.evaluation.dataset is None:
        from src.config.evaluation.schema import EvaluationDatasetConfig

        config.evaluation.dataset = EvaluationDatasetConfig(path="data/eval.jsonl")
    config.evaluation.evaluators.plugins = [
        EvaluatorPluginConfig(
            id=instance_id, plugin=plugin_id, enabled=enabled
        )
    ]


# ---------------------------------------------------------------------------
# Validation logic — tested via the evaluation kind because it has the
# easiest config shape (validation/reward/reports add complications that
# don't change the gate's logic).
# ---------------------------------------------------------------------------


def test_no_plugins_means_no_missing(patched_catalog) -> None:
    config = _load_pipeline_config(_minimal_pipeline_config_yaml())
    assert validate_required_env(config) == []


def test_missing_required_env_surfaces(
    patched_catalog,
    make_plugin_dir,
    monkeypatch,
) -> None:
    monkeypatch.delenv("EVAL_FAKE_KEY", raising=False)
    _make_eval_plugin(make_plugin_dir, "needs_key", "EVAL_FAKE_KEY")
    patched_catalog.reload()

    config = _load_pipeline_config(_minimal_pipeline_config_yaml())
    _attach_eval_plugin(config, "needs_key", instance_id="judge")
    missing = validate_required_env(config)
    assert len(missing) == 1
    only = missing[0]
    assert isinstance(only, MissingEnv)
    assert only.plugin_kind == "evaluation"
    assert only.plugin_name == "needs_key"
    assert only.plugin_instance_id == "judge"
    assert only.name == "EVAL_FAKE_KEY"
    assert only.secret is True


def test_present_in_process_env_passes(
    patched_catalog,
    make_plugin_dir,
    monkeypatch,
) -> None:
    monkeypatch.setenv("EVAL_FAKE_KEY", "abc123")
    _make_eval_plugin(make_plugin_dir, "needs_key", "EVAL_FAKE_KEY")
    patched_catalog.reload()

    config = _load_pipeline_config(_minimal_pipeline_config_yaml())
    _attach_eval_plugin(config, "needs_key")
    assert validate_required_env(config) == []


def test_present_in_project_env_passes(
    patched_catalog,
    make_plugin_dir,
    monkeypatch,
) -> None:
    monkeypatch.delenv("EVAL_FAKE_KEY", raising=False)
    _make_eval_plugin(make_plugin_dir, "needs_key", "EVAL_FAKE_KEY")
    patched_catalog.reload()

    config = _load_pipeline_config(_minimal_pipeline_config_yaml())
    _attach_eval_plugin(config, "needs_key")
    project_env = {"EVAL_FAKE_KEY": "from-env-json"}
    assert validate_required_env(config, project_env=project_env) == []


def test_optional_envs_are_ignored(
    patched_catalog,
    make_plugin_dir,
    monkeypatch,
) -> None:
    """``optional=true`` envs are surfaced in the UI but don't block launch."""
    monkeypatch.delenv("EVAL_OPTIONAL_HINT", raising=False)
    _make_eval_plugin(
        make_plugin_dir, "soft_dep", "EVAL_OPTIONAL_HINT", optional=True
    )
    patched_catalog.reload()

    config = _load_pipeline_config(_minimal_pipeline_config_yaml())
    _attach_eval_plugin(config, "soft_dep")
    assert validate_required_env(config) == []


def test_disabled_plugin_skipped(
    patched_catalog,
    make_plugin_dir,
    monkeypatch,
) -> None:
    """``enabled: false`` evaluator instances don't trigger preflight."""
    monkeypatch.delenv("EVAL_FAKE_KEY", raising=False)
    _make_eval_plugin(make_plugin_dir, "needs_key", "EVAL_FAKE_KEY")
    patched_catalog.reload()

    config = _load_pipeline_config(_minimal_pipeline_config_yaml())
    _attach_eval_plugin(config, "needs_key", enabled=False)
    assert validate_required_env(config) == []


def test_eval_block_disabled_skipped(
    patched_catalog,
    make_plugin_dir,
    monkeypatch,
) -> None:
    """``evaluation.enabled=false`` skips evaluator plugins entirely."""
    monkeypatch.delenv("EVAL_FAKE_KEY", raising=False)
    _make_eval_plugin(make_plugin_dir, "needs_key", "EVAL_FAKE_KEY")
    patched_catalog.reload()

    config = _load_pipeline_config(_minimal_pipeline_config_yaml())
    _attach_eval_plugin(config, "needs_key")
    config.evaluation.enabled = False
    assert validate_required_env(config) == []


def test_unknown_plugin_silently_skipped(
    patched_catalog,
    monkeypatch,
) -> None:
    """Plugins referenced in config but not registered in the catalog
    are silently skipped — the loader path raises a clearer error at
    instantiate-time, so duplicating it here would be noise."""
    monkeypatch.delenv("EVAL_FAKE_KEY", raising=False)
    config = _load_pipeline_config(_minimal_pipeline_config_yaml())
    _attach_eval_plugin(config, "ghost_plugin")
    assert validate_required_env(config) == []


def test_secrets_model_extra_is_consulted(
    patched_catalog,
    make_plugin_dir,
    monkeypatch,
    fake_secrets,
) -> None:
    """Plugin secrets stored in ``Secrets.model_extra`` (lowercase) are
    consulted under their uppercase env-name spelling."""
    monkeypatch.delenv("EVAL_FAKE_KEY", raising=False)
    _make_eval_plugin(make_plugin_dir, "needs_key", "EVAL_FAKE_KEY")
    patched_catalog.reload()

    config = _load_pipeline_config(_minimal_pipeline_config_yaml())
    _attach_eval_plugin(config, "needs_key")
    secrets = fake_secrets(EVAL_FAKE_KEY="abc123")
    assert validate_required_env(config, secrets=secrets) == []


def test_managed_by_envs_are_skipped(
    patched_catalog,
    make_plugin_dir,
    monkeypatch,
) -> None:
    """``managed_by="integrations" | "providers"`` envs are checked per-
    resource by Settings, not by preflight — the Configure modal hands
    the user off to the right tab instead of nudging them at env.json."""
    monkeypatch.delenv("EVAL_HF_TOKEN_PROXY", raising=False)
    extras = dedent("""
        [[required_env]]
        name = "EVAL_HF_TOKEN_PROXY"
        description = ""
        optional = false
        secret = true
        managed_by = "integrations"
    """)
    plugin_source = dedent("""
        from src.evaluation.plugins.base import EvalResult, EvaluatorPlugin

        class ManagedPlugin(EvaluatorPlugin):
            def evaluate(self, samples):
                return EvalResult(plugin_name="managed", passed=True)

            def get_recommendations(self, result):
                return []
    """)
    make_plugin_dir(
        "evaluation", "managed", manifest_extras=extras,
        plugin_source=plugin_source, class_name="ManagedPlugin",
    )
    patched_catalog.reload()

    config = _load_pipeline_config(_minimal_pipeline_config_yaml())
    _attach_eval_plugin(config, "managed")
    assert validate_required_env(config) == []


def test_launch_aborted_error_carries_missing_list(
    patched_catalog,
    make_plugin_dir,
    monkeypatch,
) -> None:
    monkeypatch.delenv("EVAL_FAKE_KEY", raising=False)
    _make_eval_plugin(make_plugin_dir, "needs_key", "EVAL_FAKE_KEY")
    patched_catalog.reload()

    config = _load_pipeline_config(_minimal_pipeline_config_yaml())
    _attach_eval_plugin(config, "needs_key")
    missing = validate_required_env(config)

    err = LaunchAbortedError(missing)
    assert err.missing == missing
    assert "needs_key:EVAL_FAKE_KEY" in str(err)
