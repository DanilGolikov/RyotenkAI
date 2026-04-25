"""Contract tests for ``PluginRegistry[T]``.

Single source of truth for the unified-registry behaviour: every kind
should agree on ``register_from_community`` / ``instantiate`` /
``get_class`` / ``manifest`` / ``list_ids`` / ``is_registered`` /
``clear``. Parametrised across all four kinds + driven by the disk-
backed fixtures from ``conftest.py``, so the tests double as a smoke
check that the loader → registry hand-off keeps working end-to-end.
"""

from __future__ import annotations

from textwrap import dedent
from typing import TYPE_CHECKING, Any

import pytest

from src.community.loader import load_plugins
from src.data.validation.registry import ValidationPluginRegistry
from src.evaluation.plugins.registry import EvaluatorPluginRegistry
from src.reports.plugins.registry import ReportPluginRegistry
from src.training.reward_plugins.registry import RewardPluginRegistry

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from src.community.registry_base import PluginRegistry


# ---------------------------------------------------------------------------
# Per-kind plugin source — minimal but actually-instantiable. Each kind
# has slightly different __init__ shape, so we hand-craft sources to
# match what the registry's _make_init_kwargs will pass.
# ---------------------------------------------------------------------------

VALIDATION_SRC = dedent("""
    from src.data.validation.base import ValidationPlugin, ValidationResult


    class TinyValidationPlugin(ValidationPlugin):
        def validate(self, dataset):
            return ValidationResult(
                plugin_name="tiny",
                passed=True,
                params=self.params,
                thresholds=self.thresholds,
                metrics={},
                warnings=[],
                errors=[],
                execution_time_ms=0.0,
            )

        def get_recommendations(self, result):
            return []
""")

EVALUATION_SRC = dedent("""
    from src.evaluation.plugins.base import EvalResult, EvaluatorPlugin


    class TinyEvalPlugin(EvaluatorPlugin):
        def evaluate(self, samples):
            return EvalResult(plugin_name="tiny", passed=True)

        def get_recommendations(self, result):
            return []
""")

REWARD_SRC = dedent("""
    from src.training.reward_plugins.base import RewardPlugin


    class TinyRewardPlugin(RewardPlugin):
        def build_trainer_kwargs(self, *, train_dataset, phase_config, pipeline_config):
            return {}
""")

REPORT_SRC = dedent("""
    from src.reports.plugins.interfaces import ReportBlock, ReportPlugin


    class TinyReportPlugin(ReportPlugin):
        plugin_id = "tiny"
        title = "Tiny"
        order = 0

        def render(self, ctx):
            return ReportBlock(
                block_id=self.plugin_id,
                title=self.title,
                order=self.order,
                nodes=[],
            )
""")


# Reward manifests require ``supported_strategies`` inside the ``[plugin]``
# block (UI uses it to know which strategies can attach the plugin) —
# threaded in via ``plugin_extras`` only for the reward case.
REWARD_PLUGIN_EXTRAS = 'supported_strategies = ["grpo"]'


# (kind, registry-cls, plugin-source, class-name, expected-init-kwargs, plugin_extras)
KIND_CASES = [
    pytest.param(
        "validation",
        ValidationPluginRegistry,
        VALIDATION_SRC,
        "TinyValidationPlugin",
        {"params": {"x": 1}, "thresholds": {"y": 2}},
        "",
        id="validation",
    ),
    pytest.param(
        "evaluation",
        EvaluatorPluginRegistry,
        EVALUATION_SRC,
        "TinyEvalPlugin",
        {"params": {"x": 1}, "thresholds": {"y": 2}},
        "",
        id="evaluation",
    ),
    pytest.param(
        "reward",
        RewardPluginRegistry,
        REWARD_SRC,
        "TinyRewardPlugin",
        {"params": {"x": 1}},
        REWARD_PLUGIN_EXTRAS,
        id="reward",
    ),
    pytest.param(
        "reports",
        ReportPluginRegistry,
        REPORT_SRC,
        "TinyReportPlugin",
        {},
        "",
        id="reports",
    ),
]


@pytest.fixture
def registry_factory() -> Callable[[type], PluginRegistry[Any]]:
    """Return a fresh registry instance for the parametrised case.

    Per-test fresh instances avoid global-state leakage between cases —
    the production singletons (``validation_registry`` etc.) stay
    untouched.
    """

    def _factory(registry_cls: type) -> PluginRegistry[Any]:
        return registry_cls()

    return _factory


@pytest.mark.parametrize(
    ("kind", "registry_cls", "plugin_src", "class_name", "init_kwargs", "plugin_extras"),
    KIND_CASES,
)
def test_register_then_instantiate(
    kind: str,
    registry_cls: type,
    plugin_src: str,
    class_name: str,
    init_kwargs: dict[str, Any],
    plugin_extras: str,
    tmp_community_root: Path,
    make_plugin_dir,
    registry_factory,
) -> None:
    make_plugin_dir(
        kind, "tiny",
        plugin_source=plugin_src,
        class_name=class_name,
        plugin_extras=plugin_extras,
    )
    loaded = load_plugins(kind, root=tmp_community_root)
    assert len(loaded) == 1

    registry = registry_factory(registry_cls)
    registry.register_from_community(loaded[0])

    assert registry.list_ids() == ["tiny"]
    assert registry.is_registered("tiny")
    assert registry.get_class("tiny") is loaded[0].plugin_cls

    instance = registry.instantiate("tiny", **init_kwargs)
    assert isinstance(instance, loaded[0].plugin_cls)


@pytest.mark.parametrize(
    ("kind", "registry_cls", "plugin_src", "class_name", "init_kwargs", "plugin_extras"),
    KIND_CASES,
)
def test_clear_empties_registry(
    kind: str,
    registry_cls: type,
    plugin_src: str,
    class_name: str,
    init_kwargs: dict[str, Any],
    plugin_extras: str,
    tmp_community_root: Path,
    make_plugin_dir,
    registry_factory,
) -> None:
    _ = init_kwargs  # not used in this test
    make_plugin_dir(
        kind, "tiny",
        plugin_source=plugin_src,
        class_name=class_name,
        plugin_extras=plugin_extras,
    )
    loaded = load_plugins(kind, root=tmp_community_root)
    registry = registry_factory(registry_cls)
    registry.register_from_community(loaded[0])
    assert registry.is_registered("tiny")

    registry.clear()
    assert registry.list_ids() == []
    assert not registry.is_registered("tiny")
    with pytest.raises(KeyError, match=f"{registry._kind} plugin 'tiny' is not registered"):
        registry.get_class("tiny")


@pytest.mark.parametrize(
    ("kind", "registry_cls", "plugin_src", "class_name", "init_kwargs", "plugin_extras"),
    KIND_CASES,
)
def test_unknown_id_lookup_raises(
    kind: str,
    registry_cls: type,
    plugin_src: str,
    class_name: str,
    init_kwargs: dict[str, Any],
    plugin_extras: str,
    registry_factory,
) -> None:
    _ = plugin_src, class_name, init_kwargs, plugin_extras
    registry = registry_factory(registry_cls)
    with pytest.raises(KeyError, match=f"{registry._kind} plugin 'nope' is not registered"):
        registry.get_class("nope")
    with pytest.raises(KeyError, match=f"{registry._kind} plugin 'nope' is not registered"):
        registry.manifest("nope")


@pytest.mark.parametrize(
    ("kind", "registry_cls", "plugin_src", "class_name", "init_kwargs", "plugin_extras"),
    KIND_CASES,
)
def test_register_idempotent_same_class(
    kind: str,
    registry_cls: type,
    plugin_src: str,
    class_name: str,
    init_kwargs: dict[str, Any],
    plugin_extras: str,
    tmp_community_root: Path,
    make_plugin_dir,
    registry_factory,
) -> None:
    """Re-registering the same class under the same id is allowed.

    The catalog re-runs ``register_from_community`` after every
    fingerprint change; the operation should be idempotent so reloads
    don't blow up.
    """
    _ = init_kwargs
    make_plugin_dir(
        kind, "tiny",
        plugin_source=plugin_src,
        class_name=class_name,
        plugin_extras=plugin_extras,
    )
    loaded = load_plugins(kind, root=tmp_community_root)
    registry = registry_factory(registry_cls)
    registry.register_from_community(loaded[0])
    # Second call with the *same* loaded entry should be a no-op.
    registry.register_from_community(loaded[0])
    assert registry.list_ids() == ["tiny"]


def test_secret_injection_requires_resolver(
    tmp_community_root: Path,
    make_plugin_dir,
) -> None:
    """A plugin declaring required secrets must get a resolver at instantiate time."""
    make_plugin_dir(
        "evaluation",
        "needs_secret",
        manifest_extras=dedent("""
            [secrets]
            required = ["EVAL_FAKE_KEY"]
        """),
        plugin_source=EVALUATION_SRC.replace("TinyEvalPlugin", "NeedsSecretEvalPlugin"),
        class_name="NeedsSecretEvalPlugin",
    )
    loaded = load_plugins("evaluation", root=tmp_community_root)
    registry = EvaluatorPluginRegistry()
    registry.register_from_community(loaded[0])

    with pytest.raises(RuntimeError, match=r"requires secrets \['EVAL_FAKE_KEY'\]"):
        registry.instantiate("needs_secret", params={}, thresholds={})


def test_secret_injection_attaches_resolved_dict(
    tmp_community_root: Path,
    make_plugin_dir,
    fake_secrets,
) -> None:
    """A plugin declaring required secrets gets ``_secrets`` populated."""
    from src.evaluation.plugins.secrets import SecretsResolver

    make_plugin_dir(
        "evaluation",
        "needs_secret",
        manifest_extras=dedent("""
            [secrets]
            required = ["EVAL_FAKE_KEY"]
        """),
        plugin_source=EVALUATION_SRC.replace("TinyEvalPlugin", "NeedsSecretEvalPlugin"),
        class_name="NeedsSecretEvalPlugin",
    )
    loaded = load_plugins("evaluation", root=tmp_community_root)
    registry = EvaluatorPluginRegistry()
    registry.register_from_community(loaded[0])

    secrets = fake_secrets(EVAL_FAKE_KEY="abc123")
    resolver = SecretsResolver(secrets)
    instance = registry.instantiate(
        "needs_secret",
        resolver=resolver,
        params={},
        thresholds={},
    )
    assert getattr(instance, "_secrets", None) == {"EVAL_FAKE_KEY": "abc123"}


def test_reports_registry_rejects_init_kwargs(
    tmp_community_root: Path,
    make_plugin_dir,
) -> None:
    """Reports plugins take no constructor args — passing some surfaces a clear error."""
    make_plugin_dir(
        "reports", "tiny",
        plugin_source=REPORT_SRC,
        class_name="TinyReportPlugin",
    )
    loaded = load_plugins("reports", root=tmp_community_root)
    registry = ReportPluginRegistry()
    registry.register_from_community(loaded[0])

    with pytest.raises(TypeError, match="report plugin instantiation does not accept init_kwargs"):
        registry.instantiate("tiny", params={"unexpected": True})
