"""``python -m src.main community …`` — scaffold, sync and pack manifests."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from src.community.pack import pack_community_folder
from src.community.scaffold import scaffold_plugin_manifest, scaffold_preset_manifest
from src.community.sync import sync_plugin_manifest, sync_preset_manifest

community_app = typer.Typer(
    no_args_is_help=True,
    help="Scaffold, sync and pack community/ plugin and preset manifests.",
)


def _detect_is_preset(path: Path, *, override: str | None = None) -> bool:
    if override in ("plugin", "preset"):
        return override == "preset"
    if override is not None and override != "auto":
        raise typer.BadParameter("--kind must be 'auto', 'plugin' or 'preset'")
    # Auto: <community>/presets/<id>  →  preset;  <community>/<kind>/<id>  →  plugin.
    return path.parent.name == "presets"


@community_app.command("scaffold")
def scaffold_cmd(
    path: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=True,
            file_okay=False,
            resolve_path=True,
            help="Folder containing the plugin (with plugin.py) or preset (with *.yaml).",
        ),
    ],
    force: Annotated[
        bool,
        typer.Option("--force", "-f", help="Overwrite existing manifest.toml."),
    ] = False,
    kind: Annotated[
        str,
        typer.Option(
            "--kind",
            help="'auto' (default) | 'plugin' | 'preset'. Override if folder lives outside community/.",
        ),
    ] = "auto",
) -> None:
    """Generate a fresh manifest.toml for a plugin or preset folder.

    Fields that can be inferred (entry_point, kind, params/thresholds schema,
    required secrets, docstring → description) are filled in automatically.
    Fields that need human judgement (category, stability, preset description)
    are emitted with a ``# TODO: fill in`` marker so nothing sneaks past the
    author.
    """
    is_preset = _detect_is_preset(path, override=kind)
    text = scaffold_preset_manifest(path) if is_preset else scaffold_plugin_manifest(path)
    target = path / "manifest.toml"
    if target.exists() and not force:
        typer.echo(
            f"error: {target} already exists; pass --force to overwrite",
            err=True,
        )
        raise typer.Exit(code=1)
    target.write_text(text, encoding="utf-8")
    typer.echo(f"wrote {target}")


@community_app.command("sync")
def sync_cmd(
    path: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=True,
            file_okay=False,
            resolve_path=True,
            help="Folder containing the plugin/preset and its manifest.toml.",
        ),
    ],
    bump: Annotated[
        str,
        typer.Option(
            "--bump",
            "-b",
            help="Version increment: patch | minor | major (default: patch).",
        ),
    ] = "patch",
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Print the diff but do not modify manifest.toml.",
        ),
    ] = False,
    kind: Annotated[
        str,
        typer.Option("--kind", help="'auto' (default) | 'plugin' | 'preset'."),
    ] = "auto",
) -> None:
    """Re-run inference over the code, 3-way-merge with the existing manifest,
    bump the version, and print/apply the diff.

    Always review the diff first — deletions in ``params_schema`` happen when
    a parameter is no longer referenced from the plugin's source code.
    """
    if bump not in ("patch", "minor", "major"):
        raise typer.BadParameter("bump must be one of: patch, minor, major")
    is_preset = _detect_is_preset(path, override=kind)
    runner = sync_preset_manifest if is_preset else sync_plugin_manifest
    result = runner(path, bump=bump)

    if not result.changed:
        typer.echo(f"{path}/manifest.toml already in sync (no changes)")
        return

    typer.echo(result.diff)
    if dry_run:
        typer.echo("[dry-run] manifest.toml not modified")
        return
    (path / "manifest.toml").write_text(result.new_text, encoding="utf-8")
    typer.echo(f"updated {path}/manifest.toml")


@community_app.command("pack")
def pack_cmd(
    path: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=True,
            file_okay=False,
            resolve_path=True,
            help="Plugin or preset folder to zip.",
        ),
    ],
    force: Annotated[
        bool,
        typer.Option("--force", "-f", help="Overwrite existing <folder>.zip."),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="List files but do not create the archive."),
    ] = False,
) -> None:
    """Pack the folder into ``<folder>.zip`` next to it.

    The archive is placed alongside the source folder (inside its kind
    directory) so the community loader picks it up automatically — you
    can delete the source folder and everything keeps working. Build
    caches (``__pycache__`` / ``.pytest_cache`` / …) are filtered out.
    The manifest is validated before packing: a broken manifest blocks
    the zip rather than shipping a bad archive.
    """
    try:
        result = pack_community_folder(path, force=force, dry_run=dry_run)
    except FileExistsError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    size_kb = result.total_bytes / 1024
    prefix = "[dry-run] would write" if dry_run else "wrote"
    typer.echo(f"{prefix} {result.archive_path}  ({len(result.files)} files, {size_kb:.1f} KiB uncompressed)")
    if dry_run:
        for name in result.files:
            typer.echo(f"  {name}")


__all__ = ["community_app"]
