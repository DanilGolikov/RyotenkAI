"""``ryotenkai preset <verb>`` — discover, inspect, apply, diff presets.

Presets are versioned, manifest-described YAML snippets that overlay a
project's pipeline config (``[preset.scope]`` controls which top-level
keys get replaced vs. preserved). Apply through
:func:`src.community.preset_apply.apply_preset` so the CLI sees exactly
what the API returns to the Web preview modal.

Preset commands deliberately bypass :class:`CommunityCatalog` and call
:func:`src.community.loader.load_presets` directly. Going through the
catalog would trigger ``_populate_registries`` (which imports every
plugin kind, and transitively the training-strategies factory) — a
needless 2 s of import noise for a read-only ``preset ls``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated

import typer
import yaml

from src.cli.common_options import DryRunOpt, RequiredConfigOpt
from src.cli.context import CLIContext
from src.cli.errors import die
from src.cli.renderer import get_renderer

preset_app = typer.Typer(
    no_args_is_help=True,
    help="List, show, apply, and diff community presets.",
    rich_markup_mode=None,
    context_settings={"help_option_names": ["-h", "--help"]},
    pretty_exceptions_enable=False,
)


@lru_cache(maxsize=1)
def _loaded_presets() -> list:  # type: ignore[type-arg]
    """Pure preset scan — no plugin imports, no registry population.

    Cached for the lifetime of the process so two preset verbs in a row
    (e.g. ``preset show`` after ``preset ls``) don't re-walk the disk.
    """
    from src.community.loader import load_presets

    return list(load_presets().presets)


def _find_preset(preset_id: str):  # type: ignore[no-untyped-def]
    return next(
        (p for p in _loaded_presets() if p.manifest.preset.id == preset_id),
        None,
    )


@preset_app.command("ls")
def ls_cmd(ctx: typer.Context) -> None:
    """List installed presets."""
    state = ctx.ensure_object(CLIContext)
    renderer = get_renderer(state)
    presets = _loaded_presets()

    rows = [
        {
            "id": p.manifest.preset.id,
            "name": p.manifest.preset.name,
            "version": p.manifest.preset.version,
            "size_tier": p.manifest.preset.size_tier,
        }
        for p in presets
    ]

    if state.is_machine_readable:
        renderer.emit(rows)
    elif not rows:
        renderer.text("No presets installed.")
    else:
        renderer.table(
            headers=["ID", "Name", "Version", "Size"],
            rows=[(r["id"], r["name"], r["version"], r["size_tier"] or "-") for r in rows],
        )
    renderer.flush()


@preset_app.command("show")
def show_cmd(
    ctx: typer.Context,
    preset_id: Annotated[str, typer.Argument(help="Preset id.")],
) -> None:
    """Show full preset manifest + body."""
    state = ctx.ensure_object(CLIContext)
    renderer = get_renderer(state)
    match = _find_preset(preset_id)
    if match is None:
        raise die(f"preset not found: {preset_id}")

    renderer.emit({
        "manifest": match.manifest.model_dump(),
        "yaml_body": match.yaml_text,
    })
    renderer.flush()


@preset_app.command("apply")
def apply_cmd(
    ctx: typer.Context,
    preset_id: Annotated[str, typer.Argument(help="Preset id.")],
    config: RequiredConfigOpt,
    output: Annotated[
        Path | None, typer.Option(
            "--output", "-o-out",
            help="Where to write the merged config (default: stdout / specified by -o).",
            dir_okay=False,
        ),
    ] = None,
    dry_run: DryRunOpt = False,
) -> None:
    """Apply a preset to a config file — print or write the merged result."""
    from src.community.preset_apply import apply_preset

    state = ctx.ensure_object(CLIContext)
    renderer = get_renderer(state)
    preset = _find_preset(preset_id)
    if preset is None:
        raise die(f"preset not found: {preset_id}")

    current = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
    if not isinstance(current, dict):
        raise die(f"config YAML must be a mapping, got {type(current).__name__}")

    preview = apply_preset(current, preset)

    if dry_run or output is None:
        if state.is_machine_readable:
            renderer.emit({
                "preset_id": preset_id,
                "diff": [d.__dict__ for d in preview.diff],
                "warnings": preview.warnings,
                "resulting_config": preview.resulting_config,
            })
        else:
            for d in preview.diff:
                renderer.text(f"  {d.kind} {d.key}")
            for w in preview.warnings:
                renderer.text(f"warning: {w}")
            if dry_run:
                renderer.text("[dry-run] not written")
            else:
                renderer.text("(no --output provided; pass --output PATH to write)")
        renderer.flush()
        return

    output.write_text(yaml.safe_dump(preview.resulting_config, sort_keys=False),
                      encoding="utf-8")
    typer.echo(f"wrote merged config: {output}")


@preset_app.command("diff")
def diff_cmd(
    ctx: typer.Context,
    preset_id: Annotated[str, typer.Argument(help="Preset id.")],
    config: RequiredConfigOpt,
) -> None:
    """Print the per-key diff a preset would apply to ``config``."""
    from src.community.preset_apply import apply_preset

    state = ctx.ensure_object(CLIContext)
    renderer = get_renderer(state)
    preset = _find_preset(preset_id)
    if preset is None:
        raise die(f"preset not found: {preset_id}")

    current = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
    preview = apply_preset(current, preset)

    if state.is_machine_readable:
        renderer.emit([d.__dict__ for d in preview.diff])
    elif not preview.diff:
        renderer.text("No changes — preset is already a no-op against this config.")
    else:
        renderer.table(
            headers=["Kind", "Key", "Reason"],
            rows=[(d.kind, d.key, d.reason) for d in preview.diff],
        )
    renderer.flush()


__all__ = ["preset_app"]
