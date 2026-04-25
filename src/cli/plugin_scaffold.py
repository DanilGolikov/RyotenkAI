"""``ryotenkai plugin scaffold <kind> <id>`` — bootstrap a new plugin folder.

The CLI sub-app this module exposes is added to the main ``ryotenkai``
typer tree by :mod:`src.main`. Running

    ryotenkai plugin scaffold validation hello_world

creates ``community/validation/hello_world/`` with:

- ``manifest.toml`` — minimum valid manifest at schema_version 4.
- ``plugin.py`` — class skeleton inheriting the right ABC for the kind,
  with TODO comments for the methods the author has to implement.
- ``README.md`` — short orientation pointing at the kind's full guide.
- ``tests/test_plugin.py`` — smoke test that imports the class.

The output is *deliberately* minimal — it loads cleanly through the
community catalog without further edits, so ``ryotenkai community
sync`` works end-to-end on the fresh folder. Authors then flesh out
``__init__`` defaults, params/thresholds schemas, and the kind-specific
method bodies.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated, Literal

import typer

from src.community.constants import COMMUNITY_ROOT, PLUGIN_KIND_DIRS
from src.community.manifest import LATEST_SCHEMA_VERSION

ScaffoldKind = Literal["validation", "evaluation", "reward", "reports"]

plugin_app = typer.Typer(
    no_args_is_help=True,
    help=(
        "Bootstrap new plugin folders under community/. Use this when you "
        "want a working skeleton from scratch — `community scaffold` is "
        "for refreshing an existing folder's manifest.toml."
    ),
    rich_markup_mode=None,
    context_settings={"help_option_names": ["-h", "--help"]},
    pretty_exceptions_enable=False,
    pretty_exceptions_show_locals=False,
    pretty_exceptions_short=True,
)


@plugin_app.callback()
def _plugin_callback() -> None:
    """Forces typer to keep ``scaffold`` as a subcommand even though
    it's currently the only one. Without this, typer collapses
    single-command apps and ``ryotenkai plugin scaffold …`` would route
    the literal ``"scaffold"`` into the kind argument."""
    return None


_PLUGIN_ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def _kind_class_base(kind: ScaffoldKind) -> tuple[str, str]:
    """Return (import-path, class-name) of the ABC the plugin must extend."""
    return {
        "validation": ("src.data.validation.base", "ValidationPlugin"),
        "evaluation": ("src.evaluation.plugins.base", "EvaluatorPlugin"),
        "reward": ("src.training.reward_plugins.base", "RewardPlugin"),
        "reports": ("src.reports.plugins.interfaces", "ReportPlugin"),
    }[kind]


def _class_name_from_id(plugin_id: str) -> str:
    """``hello_world`` → ``HelloWorldPlugin``."""
    parts = [p for p in plugin_id.split("_") if p]
    return "".join(p.capitalize() for p in parts) + "Plugin"


def _render_manifest(plugin_id: str, kind: ScaffoldKind, class_name: str) -> str:
    """Render a minimal valid manifest.toml for the kind."""
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
        # Reward manifests must declare supported_strategies — emit a
        # TODO so the loader's PluginSpec validator nudges the author
        # before the plugin is usable.
        lines.append('supported_strategies = []  # TODO: e.g. ["grpo", "sapo"]')
    lines += [
        "",
        "[plugin.entry_point]",
        'module = "plugin"',
        f'class = "{class_name}"',
        "",
    ]
    return "\n".join(lines)


def _render_plugin_py(kind: ScaffoldKind, plugin_id: str, class_name: str) -> str:
    """Render the plugin.py skeleton with the right ABC + TODO bodies."""
    base_module, base_class = _kind_class_base(kind)
    if kind == "validation":
        body = _validation_body(class_name, base_module, base_class)
    elif kind == "evaluation":
        body = _evaluation_body(class_name, base_module, base_class)
    elif kind == "reward":
        body = _reward_body(class_name, base_module, base_class)
    elif kind == "reports":
        body = _reports_body(plugin_id, class_name, base_module, base_class)
    else:
        # ScaffoldKind is a Literal — but mypy doesn't know that exhaustive
        # match here is total without a fallback raise.
        raise ValueError(f"unknown kind: {kind}")
    return body


def _validation_body(class_name: str, base_module: str, base_class: str) -> str:
    return (
        f'"""TODO: one-line module docstring."""\n'
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
        f'"""TODO: one-line module docstring."""\n'
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
        f'"""TODO: one-line module docstring."""\n'
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
        f'"""TODO: one-line module docstring."""\n'
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
        # Hardcoded plugin_id matches the manifest id the loader will
        # also stamp at registration time. Setting it here makes the
        # class self-contained for tests that instantiate the plugin
        # outside the catalog path (the most common authoring flow).
        f'    plugin_id = "{plugin_id}"\n'
        f'    title = "TODO: section title"\n'
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


def _render_readme(plugin_id: str, kind: ScaffoldKind, class_name: str) -> str:
    """Per-plugin README — points at the kind's full authoring guide."""
    return (
        f"# {plugin_id}\n"
        "\n"
        f"Scaffolded by `ryotenkai plugin scaffold {kind} {plugin_id}`.\n"
        "\n"
        "## Status\n"
        "\n"
        f"Stability: **experimental**. Class: `{class_name}`. Edit "
        f"`plugin.py` to flesh out the contract; run `ryotenkai community "
        f"sync community/{kind}/{plugin_id}` afterwards to refresh the "
        "generated bits in `manifest.toml`.\n"
        "\n"
        "## Author guide\n"
        "\n"
        f"See `community/{kind}/README.md` for the full contract this kind "
        "has to honour. The repo-level `community/README.md` covers "
        "lifecycle, secret/env handling, and the deprecation policy.\n"
    )


def _render_smoke_test(class_name: str) -> str:
    """Smoke test that imports the class — guards against import-time
    regressions and gives the author a place to add real tests."""
    return (
        '"""Smoke test for the scaffolded plugin."""\n'
        "\n"
        "from __future__ import annotations\n"
        "\n"
        "\n"
        "def test_class_imports_cleanly() -> None:\n"
        f'    """Smoke check that ``plugin.py`` parses + the class is exported."""\n'
        "    from plugin import (  # noqa: PLC0415\n"
        f"        {class_name},\n"
        "    )\n"
        "\n"
        f"    assert {class_name} is not None\n"
    )


@plugin_app.command("scaffold")
def scaffold_cmd(
    kind: Annotated[
        ScaffoldKind,
        typer.Argument(
            help='Plugin kind: "validation" | "evaluation" | "reward" | "reports".',
        ),
    ],
    plugin_id: Annotated[
        str,
        typer.Argument(
            help='Plugin id (snake_case, e.g. "hello_world"). Becomes the folder name and the manifest "id" field.',
        ),
    ],
    root: Annotated[
        Path,
        typer.Option(
            "--root",
            help="Override the community/ root (used in tests).",
            resolve_path=True,
        ),
    ] = COMMUNITY_ROOT,
    force: Annotated[
        bool,
        typer.Option("--force", help="Overwrite an existing plugin folder."),
    ] = False,
) -> None:
    """Bootstrap a working plugin folder under community/<kind>/<plugin_id>/.

    \b
    Examples:
      ryotenkai plugin scaffold validation hello_world
      ryotenkai plugin scaffold reward my_reward
    """
    if not _PLUGIN_ID_RE.match(plugin_id):
        typer.echo(
            f"error: plugin id must match {_PLUGIN_ID_RE.pattern!r} "
            f"(snake_case, lowercase, starting with a letter); got {plugin_id!r}",
            err=True,
        )
        raise typer.Exit(code=1)

    kind_dir = root / PLUGIN_KIND_DIRS[kind]
    plugin_dir = kind_dir / plugin_id
    if plugin_dir.exists() and not force:
        typer.echo(
            f"error: {plugin_dir} already exists; pass --force to overwrite",
            err=True,
        )
        raise typer.Exit(code=1)

    class_name = _class_name_from_id(plugin_id)
    plugin_dir.mkdir(parents=True, exist_ok=True)
    (plugin_dir / "manifest.toml").write_text(
        _render_manifest(plugin_id, kind, class_name)
    )
    (plugin_dir / "plugin.py").write_text(_render_plugin_py(kind, plugin_id, class_name))
    (plugin_dir / "README.md").write_text(_render_readme(plugin_id, kind, class_name))

    tests_dir = plugin_dir / "tests"
    tests_dir.mkdir(exist_ok=True)
    (tests_dir / "__init__.py").write_text("")
    (tests_dir / "test_plugin.py").write_text(_render_smoke_test(class_name))

    typer.echo(f"✓ scaffolded {plugin_dir.relative_to(root.parent)}")
    typer.echo("  - manifest.toml")
    typer.echo("  - plugin.py")
    typer.echo("  - README.md")
    typer.echo("  - tests/test_plugin.py")
    typer.echo("")
    typer.echo("Next steps:")
    typer.echo(
        f"  1. Edit plugin.py — replace TODO bodies with your logic."
    )
    typer.echo(
        f"  2. Add params/thresholds schemas to manifest.toml when you have them."
    )
    typer.echo(
        f"  3. Run `ryotenkai community sync community/{kind}/{plugin_id}` "
        "to refresh the generated bits."
    )


__all__ = ["plugin_app", "scaffold_cmd"]
