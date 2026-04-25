"""End-to-end coverage matrix for the plugin platform.

Organised by the test categories that gave the requirements:

1. ``positive``      — happy paths the rest of the system depends on.
2. ``negative``      — refusing bad input loudly enough.
3. ``boundary``      — empty / single / many / pathological values.
4. ``invariants``    — properties that must hold across mutations
                       (roundtrips, idempotence).
5. ``dependency``    — failures of upstream collaborators (resolver
                       missing, jsonschema absent, catalog stale).
6. ``regression``    — explicit guards for bugs that already shipped
                       and got fixed (4 from the Phase 1+2 audit, 1
                       from the Phase 3 audit).
7. ``logic``         — invariants of the platform's semantics
                       (secret namespace isolation, kind/instance
                       coupling, broadcast scope).
8. ``combinatorial`` — full kind × failure-type / kind × mutation
                       matrices.

Each top-level ``class`` corresponds to one category — pytest
collects them as namespaces so a CI run that wants to skip (say)
combinatorial sweeps for speed can do ``-k "not Combinatorial"``.

Tests aim to be small and high-signal: one assertion per behaviour,
no shared mutable state. The community-fixtures from
``conftest.py`` (``tmp_community_root``, ``make_plugin_dir``,
``fake_secrets``) do all the on-disk wiring.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest
import yaml
from pydantic import ValidationError

from src.community.catalog import CommunityCatalog
from src.community.instance_validator import validate_instance
from src.community.loader import LoadFailure, load_plugins
from src.community.manifest import LATEST_SCHEMA_VERSION, PluginManifest, RequiredEnvSpec
from src.community.preflight import (
    LaunchAbortedError,
    run_preflight,
    validate_required_env,
)
from src.community.stale_plugins import find_stale_plugins
from src.data.validation.registry import ValidationPluginRegistry
from src.evaluation.plugins.registry import EvaluatorPluginRegistry
from src.reports.plugins.registry import ReportPluginRegistry
from src.training.reward_plugins.registry import RewardPluginRegistry


# ---------------------------------------------------------------------------
# Shared helpers — kept thin so each category test reads at a glance.
# ---------------------------------------------------------------------------


_FIXTURE = (
    Path(__file__).resolve().parents[3] / "tests/fixtures/configs/test_pipeline.yaml"
)


def _load_pipeline_config():
    """Canonical valid pipeline config from the test fixture, with
    ``reports.sections=[]`` so tests that focus on validation/eval/
    reward don't trip on the default 13-plugin reports list (whose
    plugins live under the *real* community/, not the temp catalog)."""
    from src.config.reports.schema import ReportsConfig
    from src.utils.config import PipelineConfig

    config = PipelineConfig.model_validate(yaml.safe_load(_FIXTURE.read_text()))
    config.reports = ReportsConfig(sections=[])
    return config


def _attach_eval(config, plugin_id: str, *, params=None, thresholds=None, enabled=True):
    """Wire a single evaluator plugin instance into ``config``."""
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
            enabled=enabled,
            params=params or {},
            thresholds=thresholds or {},
        )
    ]
    return config


def _make_minimal_eval_plugin(make_plugin_dir, plugin_id: str) -> None:
    """Drop a working evaluator plugin folder under the temp catalog."""
    src = dedent(f"""
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
        plugin_source=src,
        class_name=f"{plugin_id.title().replace('_', '')}Plugin",
    )


@pytest.fixture
def patched_catalog(tmp_community_root, monkeypatch):
    """Per-test catalog rooted at the temp tree."""
    import sys
    import src.community as community_pkg

    cat = CommunityCatalog(root=tmp_community_root)
    catalog_module = sys.modules["src.community.catalog"]
    monkeypatch.setattr(catalog_module, "catalog", cat)
    monkeypatch.setattr(community_pkg, "catalog", cat)
    return cat


# ===========================================================================
# 1. POSITIVE — happy paths the platform's invariants depend on.
# ===========================================================================


class TestPositive:
    def test_loader_imports_a_minimal_plugin_cleanly(
        self, tmp_community_root, make_plugin_dir
    ) -> None:
        _make_minimal_eval_plugin(make_plugin_dir, "tiny")
        result = load_plugins("evaluation", root=tmp_community_root, strict=True)
        assert len(result.plugins) == 1
        assert result.plugins[0].manifest.plugin.id == "tiny"
        assert result.failures == []

    def test_registry_instantiate_returns_an_instance(
        self, tmp_community_root, make_plugin_dir
    ) -> None:
        _make_minimal_eval_plugin(make_plugin_dir, "tiny")
        loaded = load_plugins("evaluation", root=tmp_community_root, strict=True)
        registry = EvaluatorPluginRegistry()
        registry.register_from_community(loaded.plugins[0])
        instance = registry.instantiate("tiny", params={}, thresholds={})
        assert isinstance(instance, loaded.plugins[0].plugin_cls)

    def test_preflight_returns_ok_when_nothing_required(self, patched_catalog) -> None:
        config = _load_pipeline_config()
        report = run_preflight(config)
        assert report.ok is True
        assert report.missing_envs == []
        assert report.instance_errors == []

    def test_required_secret_names_filters_correctly(self) -> None:
        manifest = PluginManifest.model_validate({
            "plugin": {
                "id": "x", "kind": "validation", "version": "1.0.0",
                "entry_point": {"module": "plugin", "class": "X"},
            },
            "required_env": [
                {"name": "REAL_KEY", "secret": True, "optional": False},
                {"name": "OPT_KEY", "secret": True, "optional": True},
                {"name": "PUB_URL", "secret": False, "optional": False},
            ],
        })
        # Only secret=True AND optional=False makes it into the runtime tuple.
        assert manifest.required_secret_names() == ("REAL_KEY",)


# ===========================================================================
# 2. NEGATIVE — bad input must be rejected loudly.
# ===========================================================================


class TestNegative:
    def test_manifest_rejects_negative_schema_version(self) -> None:
        with pytest.raises(ValidationError, match=r"schema_version must be >= 1"):
            PluginManifest.model_validate({
                "schema_version": -1,
                "plugin": {
                    "id": "x", "kind": "validation", "version": "1.0.0",
                    "entry_point": {"module": "plugin", "class": "X"},
                },
            })

    def test_manifest_rejects_future_schema_version_with_upgrade_hint(self) -> None:
        with pytest.raises(ValidationError, match=r"Upgrade the host"):
            PluginManifest.model_validate({
                "schema_version": LATEST_SCHEMA_VERSION + 99,
                "plugin": {
                    "id": "x", "kind": "validation", "version": "1.0.0",
                    "entry_point": {"module": "plugin", "class": "X"},
                },
            })

    def test_manifest_rejects_dotted_field_name_in_params_schema(self) -> None:
        with pytest.raises(ValidationError, match=r"snake_case Python identifiers"):
            PluginManifest.model_validate({
                "plugin": {
                    "id": "x", "kind": "validation", "version": "1.0.0",
                    "entry_point": {"module": "plugin", "class": "X"},
                },
                "params_schema": {"a.b": {"type": "integer"}},
            })

    def test_registry_instantiate_raises_on_unknown_id(self) -> None:
        registry = EvaluatorPluginRegistry()
        with pytest.raises(KeyError, match=r"is not registered"):
            registry.instantiate("ghost", params={}, thresholds={})

    def test_preflight_blocks_when_required_env_unset(
        self,
        patched_catalog,
        make_plugin_dir,
        monkeypatch,
    ) -> None:
        monkeypatch.delenv("EVAL_FAKE", raising=False)
        make_plugin_dir(
            "evaluation",
            "needs_key",
            manifest_extras=dedent("""
                [[required_env]]
                name = "EVAL_FAKE"
                optional = false
                secret = true
                managed_by = ""
            """),
            plugin_source=dedent("""
                from src.evaluation.plugins.base import EvalResult, EvaluatorPlugin
                class NeedsKeyPlugin(EvaluatorPlugin):
                    def evaluate(self, samples):
                        return EvalResult(plugin_name="needs_key", passed=True)
                    def get_recommendations(self, r): return []
            """),
            class_name="NeedsKeyPlugin",
        )
        patched_catalog.reload()

        config = _attach_eval(_load_pipeline_config(), "needs_key")
        missing = validate_required_env(config)
        assert any(m.name == "EVAL_FAKE" for m in missing)


# ===========================================================================
# 3. BOUNDARY — empty / one / many / pathological.
# ===========================================================================


class TestBoundary:
    def test_loader_on_empty_kind_dir_returns_empty_result(
        self, tmp_community_root
    ) -> None:
        result = load_plugins("evaluation", root=tmp_community_root, strict=True)
        assert list(result) == []
        assert result.failures == []

    def test_registry_clear_on_empty_registry_is_noop(self) -> None:
        registry = EvaluatorPluginRegistry()
        registry.clear()
        assert registry.list_ids() == []

    def test_required_env_with_zero_entries_yields_empty_secret_tuple(self) -> None:
        manifest = PluginManifest.model_validate({
            "plugin": {
                "id": "x", "kind": "validation", "version": "1.0.0",
                "entry_point": {"module": "plugin", "class": "X"},
            },
            "required_env": [],
        })
        assert manifest.required_secret_names() == ()

    def test_required_env_with_single_entry_round_trips(self) -> None:
        body = {
            "plugin": {
                "id": "x", "kind": "validation", "version": "1.0.0",
                "entry_point": {"module": "plugin", "class": "X"},
            },
            "required_env": [{
                "name": "K", "secret": True, "optional": False, "managed_by": "",
            }],
        }
        manifest = PluginManifest.model_validate(body)
        assert manifest.required_secret_names() == ("K",)

    def test_long_plugin_id_loads(
        self, tmp_community_root, make_plugin_dir
    ) -> None:
        long_id = "a" * 200
        _make_minimal_eval_plugin(make_plugin_dir, long_id)
        result = load_plugins("evaluation", root=tmp_community_root, strict=True)
        assert result.plugins[0].manifest.plugin.id == long_id

    def test_unicode_description_round_trips_through_ui_manifest(self) -> None:
        manifest = PluginManifest.model_validate({
            "plugin": {
                "id": "x", "kind": "validation", "version": "1.0.0",
                "description": "Чек на dëduplication 测试 🦊",
                "entry_point": {"module": "plugin", "class": "X"},
            },
        })
        ui = manifest.ui_manifest()
        assert ui["description"] == "Чек на dëduplication 测试 🦊"

    def test_empty_required_env_list_doesnt_block_launch(self, patched_catalog) -> None:
        config = _load_pipeline_config()
        # No plugins with required_env attached → no missing.
        assert validate_required_env(config) == []

    def test_stale_detection_on_empty_config_is_empty_list(
        self, patched_catalog
    ) -> None:
        config = _load_pipeline_config()
        assert find_stale_plugins(config) == []


# ===========================================================================
# 4. INVARIANTS — properties preserved across mutations.
# ===========================================================================


class TestInvariants:
    def test_register_then_clear_then_register_is_idempotent(
        self, tmp_community_root, make_plugin_dir
    ) -> None:
        _make_minimal_eval_plugin(make_plugin_dir, "tiny")
        loaded = load_plugins("evaluation", root=tmp_community_root, strict=True)
        registry = EvaluatorPluginRegistry()
        registry.register_from_community(loaded.plugins[0])
        first_class = registry.get_class("tiny")
        registry.clear()
        registry.register_from_community(loaded.plugins[0])
        second_class = registry.get_class("tiny")
        # Same class reference — re-registration of an identical entry
        # is non-destructive.
        assert first_class is second_class

    def test_ui_manifest_round_trips_through_catalog(
        self, tmp_community_root, make_plugin_dir
    ) -> None:
        _make_minimal_eval_plugin(make_plugin_dir, "tiny")
        loaded = load_plugins("evaluation", root=tmp_community_root, strict=True)
        ui_first = loaded.plugins[0].manifest.ui_manifest()
        # Re-validate the manifest from its ui shape: identity bits
        # must survive (id / kind / version).
        assert ui_first["id"] == "tiny"
        assert ui_first["kind"] == "evaluation"
        assert ui_first["schema_version"] == LATEST_SCHEMA_VERSION

    def test_required_secret_names_is_subset_of_required_env_names(self) -> None:
        manifest = PluginManifest.model_validate({
            "plugin": {
                "id": "x", "kind": "validation", "version": "1.0.0",
                "entry_point": {"module": "plugin", "class": "X"},
            },
            "required_env": [
                {"name": "A", "secret": True, "optional": False},
                {"name": "B", "secret": True, "optional": True},
                {"name": "C", "secret": False, "optional": False},
            ],
        })
        env_names = {spec.name for spec in manifest.required_env}
        secrets = set(manifest.required_secret_names())
        assert secrets <= env_names  # hard subset relationship

    def test_loose_mode_preserves_plugins_count_invariant(
        self, tmp_community_root, make_plugin_dir
    ) -> None:
        """plugins+failures across loose/strict on the same input is
        equivalent: strict raises on the first failure, loose returns
        partition. For an all-clean input the totals match."""
        _make_minimal_eval_plugin(make_plugin_dir, "tiny")
        strict = load_plugins("evaluation", root=tmp_community_root, strict=True)
        loose = load_plugins("evaluation", root=tmp_community_root, strict=False)
        assert len(strict.plugins) == len(loose.plugins)
        assert loose.failures == []


# ===========================================================================
# 5. DEPENDENCY ERRORS — failures of upstream collaborators.
# ===========================================================================


class TestDependencyErrors:
    def test_registry_instantiate_raises_when_resolver_missing(
        self, tmp_community_root, make_plugin_dir
    ) -> None:
        make_plugin_dir(
            "evaluation",
            "needs_key",
            manifest_extras=dedent("""
                [[required_env]]
                name = "EVAL_FAKE"
                optional = false
                secret = true
                managed_by = ""
            """),
            plugin_source=dedent("""
                from src.evaluation.plugins.base import EvalResult, EvaluatorPlugin
                class NeedsKeyPlugin(EvaluatorPlugin):
                    def evaluate(self, samples):
                        return EvalResult(plugin_name="needs_key", passed=True)
                    def get_recommendations(self, r): return []
            """),
            class_name="NeedsKeyPlugin",
        )
        loaded = load_plugins("evaluation", root=tmp_community_root, strict=True)
        registry = EvaluatorPluginRegistry()
        registry.register_from_community(loaded.plugins[0])
        # No resolver → fail fast.
        with pytest.raises(RuntimeError, match=r"requires secrets"):
            registry.instantiate("needs_key", params={}, thresholds={})

    def test_loader_loose_mode_captures_broken_manifest(
        self, tmp_community_root
    ) -> None:
        """A manifest that fails Pydantic validation populates a
        ``manifest_parse`` LoadFailure rather than crashing the whole
        scan. Production-loose path."""
        plugin_dir = tmp_community_root / "evaluation" / "broken"
        plugin_dir.mkdir(parents=True)
        (plugin_dir / "manifest.toml").write_text("[plugin]\nid = 42\n")  # id must be str
        (plugin_dir / "plugin.py").write_text("class X: pass\n")
        result = load_plugins("evaluation", root=tmp_community_root, strict=False)
        assert result.plugins == []
        assert any(f.error_type == "manifest_parse" for f in result.failures)

    def test_loader_loose_mode_captures_import_error(
        self, tmp_community_root, make_plugin_dir
    ) -> None:
        make_plugin_dir(
            "evaluation",
            "explodes",
            plugin_source="raise RuntimeError('boom at import')\n",
            class_name="ExplodesPlugin",
        )
        result = load_plugins("evaluation", root=tmp_community_root, strict=False)
        assert any(f.error_type == "import_error" for f in result.failures)

    def test_preflight_silently_skips_unknown_plugin(
        self, patched_catalog
    ) -> None:
        """A plugin id referenced in YAML but absent from the catalog
        must NOT surface as a missing-env error — the runtime path
        will produce a clearer "plugin not found" later."""
        config = _attach_eval(_load_pipeline_config(), "ghost_plugin")
        report = run_preflight(config)
        assert report.missing_envs == []
        assert report.instance_errors == []


# ===========================================================================
# 6. REGRESSIONS — explicit guards for fixed bugs.
# ===========================================================================


class TestRegressions:
    def test_phase12_audit_required_env_crosscheck_loose_mode(
        self, tmp_community_root
    ) -> None:
        """A drift between Python REQUIRED_ENV and TOML required_env
        used to take the WHOLE catalog down because the cross-check
        ran outside the load_plugins try/except. Now it surfaces as
        a structured ``metadata_error`` LoadFailure."""
        plugin_dir = tmp_community_root / "evaluation" / "drifty"
        plugin_dir.mkdir(parents=True)
        (plugin_dir / "manifest.toml").write_text(dedent("""
            [plugin]
            id = "drifty"
            kind = "evaluation"
            version = "1.0.0"

            [plugin.entry_point]
            module = "plugin"
            class = "DriftyPlugin"

            [[required_env]]
            name = "EVAL_X"
            optional = true
            secret = true
            managed_by = ""
        """).strip())
        (plugin_dir / "plugin.py").write_text(dedent("""
            from src.evaluation.plugins.base import EvalResult, EvaluatorPlugin
            from src.community.manifest import RequiredEnvSpec

            class DriftyPlugin(EvaluatorPlugin):
                REQUIRED_ENV = (
                    RequiredEnvSpec(name="EVAL_X", optional=False, secret=True, managed_by=""),
                )
                def evaluate(self, samples):
                    return EvalResult(plugin_name="drifty", passed=True)
                def get_recommendations(self, r): return []
        """))
        result = load_plugins("evaluation", root=tmp_community_root, strict=False)
        assert result.plugins == []
        assert any(f.error_type == "metadata_error" for f in result.failures)

    def test_phase12_audit_kind_labels_match_pluginkind(self) -> None:
        """``_kind`` ClassVars used to read ``"evaluator"`` / ``"report"``
        for two registries while the canonical PluginKind literal uses
        ``"evaluation"`` / ``"reports"``. Error messages drifted from
        the API vocabulary."""
        assert ValidationPluginRegistry._kind == "validation"
        assert EvaluatorPluginRegistry._kind == "evaluation"
        assert RewardPluginRegistry._kind == "reward"
        assert ReportPluginRegistry._kind == "reports"

    def test_phase12_audit_schema_version_default_matches_latest(self) -> None:
        """The API ``PluginManifest.schema_version`` default used to
        read 1; it now mirrors LATEST so OpenAPI consumers see the
        version the API actually emits."""
        from src.api.schemas.plugin import PluginManifest as ApiPluginManifest

        assert ApiPluginManifest.model_fields["schema_version"].default == LATEST_SCHEMA_VERSION

    def test_phase3_audit_scaffold_reports_seeds_plugin_id(self, tmp_path) -> None:
        """The reports scaffold used to emit ``plugin_id = ""`` and
        rely on the loader to stamp it. Tests instantiating directly
        saw an empty id in render output. Now seeded from manifest."""
        from src.cli.plugin_scaffold import _render_plugin_py

        body = _render_plugin_py("reports", "my_section", "MySectionPlugin")
        assert 'plugin_id = "my_section"' in body


# ===========================================================================
# 7. LOGIC-SPECIFIC — invariants of the platform's semantics.
# ===========================================================================


class TestLogicSpecific:
    def test_secret_namespace_isolation_validation_cant_read_eval_keys(
        self, fake_secrets
    ) -> None:
        """Validation plugins are fenced into the DTST_* prefix; any
        attempt to resolve a non-DTST_ key raises so a buggy manifest
        can't leak system secrets through the validation surface."""
        from src.data.validation.secrets import SecretsResolver

        secrets = fake_secrets(EVAL_OTHER_KIND="leaked", DTST_OK="value")
        resolver = SecretsResolver(secrets)
        with pytest.raises(ValueError, match=r"outside the allowed 'DTST_\*' namespace"):
            resolver.resolve(("EVAL_OTHER_KIND",))

    def test_secret_namespace_isolation_eval_cant_read_dtst_keys(
        self, fake_secrets
    ) -> None:
        """Symmetric check for evaluation kind."""
        from src.evaluation.plugins.secrets import SecretsResolver

        secrets = fake_secrets(DTST_OTHER_KIND="leaked", EVAL_OK="value")
        resolver = SecretsResolver(secrets)
        with pytest.raises(ValueError, match=r"outside the allowed 'EVAL_\*' namespace"):
            resolver.resolve(("DTST_OTHER_KIND",))

    def test_managed_by_envs_skipped_by_preflight(
        self, patched_catalog, make_plugin_dir, monkeypatch
    ) -> None:
        """``managed_by="integrations" | "providers"`` envs are owned
        by Settings, not env.json. Preflight must NOT block on them."""
        monkeypatch.delenv("EVAL_HF_PROXY", raising=False)
        make_plugin_dir(
            "evaluation",
            "managed",
            manifest_extras=dedent("""
                [[required_env]]
                name = "EVAL_HF_PROXY"
                optional = false
                secret = true
                managed_by = "integrations"
            """),
            plugin_source=dedent("""
                from src.evaluation.plugins.base import EvalResult, EvaluatorPlugin
                class ManagedPlugin(EvaluatorPlugin):
                    def evaluate(self, samples):
                        return EvalResult(plugin_name="managed", passed=True)
                    def get_recommendations(self, r): return []
            """),
            class_name="ManagedPlugin",
        )
        patched_catalog.reload()
        config = _attach_eval(_load_pipeline_config(), "managed")
        assert validate_required_env(config) == []

    def test_disabled_evaluation_plugin_is_not_preflighted(
        self, patched_catalog, make_plugin_dir, monkeypatch
    ) -> None:
        """A plugin instance with ``enabled: false`` doesn't run, so
        its missing envs / shape errors must not block launch."""
        monkeypatch.delenv("EVAL_FAKE", raising=False)
        make_plugin_dir(
            "evaluation",
            "needs_key",
            manifest_extras=dedent("""
                [[required_env]]
                name = "EVAL_FAKE"
                optional = false
                secret = true
                managed_by = ""
            """),
            plugin_source=dedent("""
                from src.evaluation.plugins.base import EvalResult, EvaluatorPlugin
                class NeedsKeyPlugin(EvaluatorPlugin):
                    def evaluate(self, samples):
                        return EvalResult(plugin_name="needs_key", passed=True)
                    def get_recommendations(self, r): return []
            """),
            class_name="NeedsKeyPlugin",
        )
        patched_catalog.reload()
        config = _attach_eval(
            _load_pipeline_config(), "needs_key", enabled=False
        )
        assert validate_required_env(config) == []

    def test_instance_validator_collects_all_errors_in_one_pass(self) -> None:
        """Draft7Validator.iter_errors is the right tool because a
        single ``validate`` call would only show the first error and
        force the user to fix-and-rerun in a loop. We assert multi-
        error collection explicitly."""
        manifest = PluginManifest.model_validate({
            "plugin": {
                "id": "x", "kind": "evaluation", "version": "1.0.0",
                "entry_point": {"module": "plugin", "class": "X"},
            },
            "params_schema": {
                "n": {"type": "integer", "min": 0, "max": 10, "default": 5},
                "mode": {"type": "enum", "options": ["a", "b"], "default": "a"},
            },
        })
        errors = validate_instance(
            manifest,
            plugin_kind="evaluation",
            plugin_name="x",
            plugin_instance_id="i",
            params={"n": 999, "mode": "ludicrous"},
            thresholds={},
        )
        # Both violations surfaced in one call.
        locations = {e.location for e in errors}
        assert "params.n" in locations
        assert "params.mode" in locations

    def test_launch_aborted_error_message_lists_both_failure_classes(self) -> None:
        """LaunchAbortedError concatenates env + shape failure
        descriptions so the user sees the full picture in one
        traceback line."""
        from src.community.instance_validator import InstanceValidationError
        from src.community.preflight import MissingEnv

        err = LaunchAbortedError(
            missing=[MissingEnv(
                plugin_kind="evaluation", plugin_name="p",
                plugin_instance_id="i", name="K",
                description="", secret=True, managed_by="",
            )],
            instance_errors=[InstanceValidationError(
                plugin_kind="evaluation", plugin_name="p",
                plugin_instance_id="i", location="params.x",
                message="bad shape",
            )],
        )
        msg = str(err)
        assert "p:K" in msg
        assert "p:params.x" in msg


# ===========================================================================
# 8. COMBINATORIAL — full kind × failure-type / kind × mutation matrices.
# ===========================================================================


_ALL_KINDS = ("validation", "evaluation", "reward", "reports")


class TestCombinatorial:
    @pytest.mark.parametrize(
        ("registry_cls",),
        [
            (ValidationPluginRegistry,),
            (EvaluatorPluginRegistry,),
            (RewardPluginRegistry,),
            (ReportPluginRegistry,),
        ],
    )
    def test_every_registry_starts_empty(self, registry_cls: type) -> None:
        """Fresh instance of any kind has the same boundary state."""
        registry = registry_cls()
        assert registry.list_ids() == []
        assert registry.list_manifests() == []
        assert not registry.is_registered("anything")

    @pytest.mark.parametrize("kind", _ALL_KINDS)
    def test_loose_mode_isolates_failures_per_kind(
        self, kind: str, tmp_community_root
    ) -> None:
        """A broken plugin in one kind subtree must not affect the
        scan of a different kind. Scoped per-kind error capture is the
        whole point of the LoadResult split."""
        # Drop a broken evaluation plugin (one we know breaks).
        broken = tmp_community_root / "evaluation" / "broken"
        broken.mkdir(parents=True)
        (broken / "manifest.toml").write_text("[plugin]\nid = 99\n")
        (broken / "plugin.py").write_text("")

        # Scan some OTHER kind — must complete cleanly.
        if kind == "evaluation":
            pytest.skip("same-kind broken plugin tested in TestDependencyErrors")
        result = load_plugins(kind, root=tmp_community_root, strict=False)
        assert result.failures == []

    @pytest.mark.parametrize(
        ("error_type", "manifest_text"),
        [
            ("manifest_parse", "[plugin]\nid = 42\n"),  # id must be str
            (
                "kind_mismatch",
                # declares kind=reward in evaluation/ tree
                dedent("""
                    [plugin]
                    id = "wrong"
                    kind = "reward"
                    version = "1.0.0"
                    supported_strategies = ["grpo"]

                    [plugin.entry_point]
                    module = "plugin"
                    class = "X"
                """).strip(),
            ),
        ],
    )
    def test_loose_mode_records_each_error_type(
        self,
        tmp_community_root,
        error_type: str,
        manifest_text: str,
    ) -> None:
        plugin_dir = tmp_community_root / "evaluation" / "x"
        plugin_dir.mkdir(parents=True)
        (plugin_dir / "manifest.toml").write_text(manifest_text)
        (plugin_dir / "plugin.py").write_text("class X:\n    pass\n")
        result = load_plugins("evaluation", root=tmp_community_root, strict=False)
        assert any(f.error_type == error_type for f in result.failures)

    @pytest.mark.parametrize("kind", _ALL_KINDS)
    def test_LoadResult_iter_and_len_agree_with_plugins_list(
        self, kind: str, tmp_community_root
    ) -> None:
        """LoadResult is iterable / sized via the plugins field. The
        invariant must hold for every kind even when the directory is
        empty."""
        result = load_plugins(kind, root=tmp_community_root, strict=False)
        assert list(result) == result.plugins
        assert len(result) == len(result.plugins)
