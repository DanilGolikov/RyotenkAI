"""Render-helpers for scaffolding a new community plugin folder.

Pure-python — no Typer, no CLI imports. Used by
:mod:`src.cli.commands.plugin` (the ``ryotenkai plugin scaffold`` verb)
and by tests that want to verify the rendered skeletons compile + load
through the catalog.

The output is deliberately minimal: the manifest is the smallest valid
schema-v4 document for the kind, the ``plugin.py`` skeleton inherits
the right ABC and stubs the contract methods with TODOs, and the
``tests/test_plugin.py`` smoke test imports the class so a
``pytest community/<kind>/<id>`` run flags syntax errors immediately.
"""

from __future__ import annotations

import re
from typing import Final, Literal

from src.community.manifest import LATEST_SCHEMA_VERSION

#: Plugin id format. Must be ``snake_case`` for the
#: ``manifest_id == folder_name == python_module`` invariants to hold.
PLUGIN_ID_RE: Final[re.Pattern[str]] = re.compile(r"^[a-z][a-z0-9_]*$")

ScaffoldKind = Literal["validation", "evaluation", "reward", "reports"]

#: ``kind → (import-path, ABC-class)`` of the base class plugins extend.
_KIND_BASE: Final[dict[str, tuple[str, str]]] = {
    "validation": ("src.data.validation.base", "ValidationPlugin"),
    "evaluation": ("src.evaluation.plugins.base", "EvaluatorPlugin"),
    "reward":     ("src.training.reward_plugins.base", "RewardPlugin"),
    "reports":    ("src.reports.plugins.interfaces", "ReportPlugin"),
}


def validate_plugin_id(plugin_id: str) -> None:
    """Reject ids that won't round-trip cleanly through manifest/folder/module."""
    if not PLUGIN_ID_RE.match(plugin_id):
        raise ValueError(
            f"plugin id must match {PLUGIN_ID_RE.pattern!r} "
            f"(snake_case, lowercase, starts with a letter); got {plugin_id!r}"
        )


def class_name_from_id(plugin_id: str) -> str:
    """``hello_world`` → ``HelloWorldPlugin``."""
    parts = [p for p in plugin_id.split("_") if p]
    return "".join(p.capitalize() for p in parts) + "Plugin"


def kind_class_base(kind: ScaffoldKind) -> tuple[str, str]:
    """Return ``(module, ABC-class)`` for the kind."""
    return _KIND_BASE[kind]


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------


def render_manifest(plugin_id: str, kind: ScaffoldKind, class_name: str) -> str:
    """Render a minimum-valid schema-v4 manifest for the kind."""
    lines = [
        f"schema_version = {LATEST_SCHEMA_VERSION}",
        "",
        "[plugin]",
        f'id = "{plugin_id}"',
        f'kind = "{kind}"',
        f'name = "{plugin_id.replace("_", " ").title()}"',
        'version = "0.1.0"',
        'category = ""',
        'stability = "experimental"',
        f'description = "TODO: one-line description of what this {kind} plugin does."',
    ]
    if kind == "reward":
        lines.append('supported_strategies = []  # TODO: e.g. ["grpo", "sapo"]')
    lines += [
        "",
        "[plugin.entry_point]",
        'module = "plugin"',
        f'class = "{class_name}"',
        "",
    ]
    return "\n".join(lines)


def render_plugin_py(kind: ScaffoldKind, plugin_id: str, class_name: str) -> str:
    """Render the ``plugin.py`` body — picks the kind-specific stub."""
    base_module, base_class = kind_class_base(kind)
    if kind == "validation":
        return _validation_body(class_name, base_module, base_class)
    if kind == "evaluation":
        return _evaluation_body(class_name, base_module, base_class)
    if kind == "reward":
        return _reward_body(class_name, base_module, base_class)
    if kind == "reports":
        return _reports_body(plugin_id, class_name, base_module, base_class)
    raise ValueError(f"unknown kind: {kind}")


def _validation_body(class_name: str, base_module: str, base_class: str) -> str:
    return (
        '"""TODO: one-line module docstring."""\n'
        "\n"
        "from __future__ import annotations\n"
        "\n"
        "import time\n"
        "from typing import TYPE_CHECKING\n"
        "\n"
        f"from {base_module} import {base_class}, ValidationResult\n"
        "\n"
        "if TYPE_CHECKING:\n"
        "    from datasets import Dataset, IterableDataset\n"
        "\n"
        "\n"
        f"class {class_name}({base_class}):\n"
        '    """TODO: class docstring describing what gets validated."""\n'
        "\n"
        "    expensive = False\n"
        "    supports_streaming = True\n"
        "\n"
        '    def validate(self, dataset: "Dataset | IterableDataset") -> ValidationResult:\n'
        "        start = time.time()\n"
        "        # TODO: implement the actual check\n"
        "        passed = True\n"
        "        return ValidationResult(\n"
        "            plugin_name=self.name,\n"
        "            passed=passed,\n"
        "            params=dict(self.params),\n"
        "            thresholds=dict(self.thresholds),\n"
        "            metrics={},\n"
        "            warnings=[],\n"
        "            errors=[],\n"
        "            execution_time_ms=(time.time() - start) * 1000,\n"
        "        )\n"
        "\n"
        "    def get_recommendations(self, result: ValidationResult) -> list[str]:\n"
        "        return [] if result.passed else [\"TODO: actionable recommendation\"]\n"
    )


def _evaluation_body(class_name: str, base_module: str, base_class: str) -> str:
    return (
        '"""TODO: one-line module docstring."""\n'
        "\n"
        "from __future__ import annotations\n"
        "\n"
        f"from {base_module} import EvalResult, EvalSample, {base_class}\n"
        "\n"
        "\n"
        f"class {class_name}({base_class}):\n"
        '    """TODO: class docstring describing what gets scored."""\n'
        "\n"
        "    requires_expected_answer = False\n"
        "\n"
        "    def evaluate(self, samples: list[EvalSample]) -> EvalResult:\n"
        "        # TODO: implement scoring; populate metrics with the numbers\n"
        "        # that drive your pass/fail decision.\n"
        "        return EvalResult(\n"
        "            plugin_name=self.name,\n"
        "            passed=True,\n"
        "            metrics={},\n"
        "            sample_count=len(samples),\n"
        "        )\n"
        "\n"
        "    def get_recommendations(self, result: EvalResult) -> list[str]:\n"
        "        return [] if result.passed else [\"TODO: actionable recommendation\"]\n"
    )


def _reward_body(class_name: str, base_module: str, base_class: str) -> str:
    return (
        '"""TODO: one-line module docstring."""\n'
        "\n"
        "from __future__ import annotations\n"
        "\n"
        "from typing import TYPE_CHECKING, Any\n"
        "\n"
        f"from {base_module} import {base_class}\n"
        "\n"
        "if TYPE_CHECKING:\n"
        "    from datasets import Dataset\n"
        "\n"
        "    from src.utils.config import PipelineConfig, StrategyPhaseConfig\n"
        "\n"
        "\n"
        f"class {class_name}({base_class}):\n"
        '    """TODO: class docstring describing the reward signal."""\n'
        "\n"
        "    def build_trainer_kwargs(\n"
        "        self,\n"
        "        *,\n"
        '        train_dataset: "Dataset",\n'
        '        phase_config: "StrategyPhaseConfig",\n'
        '        pipeline_config: "PipelineConfig",\n'
        "    ) -> dict[str, Any]:\n"
        "        # TODO: return TRL Trainer kwargs (e.g. {'reward_funcs': [...]}).\n"
        "        # See community/reward/README.md for the contract.\n"
        "        return {}\n"
    )


def _reports_body(plugin_id: str, class_name: str, base_module: str, base_class: str) -> str:
    return (
        '"""TODO: one-line module docstring."""\n'
        "\n"
        "from __future__ import annotations\n"
        "\n"
        "from src.reports.document.nodes import DocBlock, Heading, Paragraph, inlines, txt\n"
        f"from {base_module} import ReportBlock, {base_class}, ReportPluginContext\n"
        "\n"
        "\n"
        f"class {class_name}({base_class}):\n"
        '    """TODO: class docstring describing the section."""\n'
        "\n"
        f'    plugin_id = "{plugin_id}"\n'
        '    title = "TODO: section title"\n'
        "    order = 0  # overwritten by build_report_plugins from reports.sections\n"
        "\n"
        "    def render(self, ctx: ReportPluginContext) -> ReportBlock:\n"
        "        nodes: list[DocBlock] = [\n"
        "            Heading(2, inlines(txt(self.title))),\n"
        "            Paragraph(inlines(txt(\"TODO: build the section body from ctx.\"))),\n"
        "        ]\n"
        "        return ReportBlock(\n"
        "            block_id=self.plugin_id,\n"
        "            title=self.title,\n"
        "            order=self.order,\n"
        "            nodes=nodes,\n"
        "        )\n"
    )


def render_readme(plugin_id: str, kind: ScaffoldKind, class_name: str) -> str:
    """Per-plugin README — points at the kind's full authoring guide."""
    return (
        f"# {plugin_id}\n"
        "\n"
        f"Scaffolded by `ryotenkai plugin scaffold {kind} {plugin_id}`.\n"
        "\n"
        "## Status\n"
        "\n"
        f"Stability: **experimental**. Class: `{class_name}`. Edit "
        "`plugin.py` to flesh out the contract; run `ryotenkai plugin "
        f"sync community/{kind}/{plugin_id}` afterwards to refresh the "
        "generated bits in `manifest.toml`.\n"
        "\n"
        "## Author guide\n"
        "\n"
        f"See `community/{kind}/README.md` for the full contract this kind "
        "has to honour. The repo-level `community/README.md` covers "
        "lifecycle, secret/env handling, and the deprecation policy.\n"
    )


def render_smoke_test(class_name: str) -> str:
    """Smoke test that imports the class — guards import-time regressions."""
    return (
        '"""Smoke test for the scaffolded plugin."""\n'
        "\n"
        "from __future__ import annotations\n"
        "\n"
        "\n"
        "def test_class_imports_cleanly() -> None:\n"
        '    """Smoke check that ``plugin.py`` parses + the class is exported."""\n'
        "    from plugin import (  # noqa: PLC0415\n"
        f"        {class_name},\n"
        "    )\n"
        "\n"
        f"    assert {class_name} is not None\n"
    )


__all__ = [
    "PLUGIN_ID_RE",
    "ScaffoldKind",
    "class_name_from_id",
    "kind_class_base",
    "render_manifest",
    "render_plugin_py",
    "render_readme",
    "render_smoke_test",
    "validate_plugin_id",
]
