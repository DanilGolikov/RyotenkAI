"""``ryotenkai plugin <verb>`` — community plugin lifecycle.

The merged surface of the legacy ``community`` group + the standalone
``plugin_scaffold`` Typer + new install / preflight / stale verbs.

Verb summary:
- **ls / show** — discovery via ``CommunityCatalog``.
- **scaffold** — bootstrap a fresh plugin folder using the pure-python
  render helpers in :mod:`src.community.scaffold_template`.
- **sync / sync-envs / pack** — author-side toolkit (re-exported from
  :mod:`src.community.{sync,pack}`).
- **validate** — standalone manifest validator (no Python import).
- **install** — clone / unzip / copy a plugin into ``community/<kind>/``.
- **preflight** — pre-launch missing-env + instance-shape gate.
- **stale** — find references to plugins not in the catalog.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import typer

from src.cli.common_options import DryRunOpt, ForceOpt, RequiredConfigOpt
from src.cli.context import CLIContext
from src.cli.errors import die
from src.cli.renderer import get_renderer
from src.community.constants import ALL_PLUGIN_KINDS

plugin_app = typer.Typer(
    no_args_is_help=True,
    help="Discover, install, validate, scaffold, sync, pack, and preflight plugins.",
    rich_markup_mode=None,
    context_settings={"help_option_names": ["-h", "--help"]},
    pretty_exceptions_enable=False,
)


# ---------------------------------------------------------------------------
# Discovery: ls / show
# ---------------------------------------------------------------------------


@plugin_app.command("ls")
def ls_cmd(
    ctx: typer.Context,
    kind: Annotated[
        str,
        typer.Option(
            "--kind",
            help=f"Filter by plugin kind: {' | '.join(ALL_PLUGIN_KINDS)}.",
        ),
    ],
) -> None:
    """List installed plugins of the given kind."""
    from src.api.services import plugin_service

    if kind not in ALL_PLUGIN_KINDS:
        raise die(
            f"--kind must be one of {sorted(ALL_PLUGIN_KINDS)}; got {kind!r}",
        )

    state = ctx.ensure_object(CLIContext)
    renderer = get_renderer(state)
    response = plugin_service.list_plugins(kind)  # type: ignore[arg-type]

    if state.is_machine_readable:
        renderer.emit(response.model_dump())
    elif not response.plugins:
        renderer.text(f"No {kind} plugins installed.")
    else:
        renderer.table(
            headers=["ID", "Name", "Version", "Stability"],
            rows=[
                (p.id, p.name, p.version, p.stability)
                for p in response.plugins
            ],
        )
        if response.failures:
            renderer.text("")
            renderer.text(f"warning: {len(response.failures)} plugin(s) failed to load:")
            for fail in response.failures:
                renderer.text(f"  - {fail.entry_name}: {fail.error_type} — {fail.message}")
    renderer.flush()


@plugin_app.command("show")
def show_cmd(
    ctx: typer.Context,
    kind: Annotated[str, typer.Argument(help="Plugin kind.")],
    plugin_id: Annotated[str, typer.Argument(help="Plugin id.")],
) -> None:
    """Show full manifest for a plugin."""
    from src.community.catalog import catalog

    if kind not in ALL_PLUGIN_KINDS:
        raise die(f"unknown kind: {kind!r}")

    state = ctx.ensure_object(CLIContext)
    renderer = get_renderer(state)
    catalog.ensure_loaded()
    try:
        loaded = catalog.get(kind, plugin_id)  # type: ignore[arg-type]
    except KeyError:
        raise die(f"plugin not found: {kind}/{plugin_id}")
    renderer.emit(loaded.manifest.ui_manifest())
    renderer.flush()


# ---------------------------------------------------------------------------
# Authoring: scaffold / sync / sync-envs / pack
# ---------------------------------------------------------------------------


@plugin_app.command("scaffold")
def scaffold_cmd(
    kind: Annotated[
        str,
        typer.Argument(
            help=f"Plugin kind: {' | '.join(ALL_PLUGIN_KINDS)}.",
        ),
    ],
    plugin_id: Annotated[str, typer.Argument(help="Plugin id (snake_case).")],
    root: Annotated[
        Path | None,
        typer.Option(
            "--root", help="Override the community/ root (used in tests).",
            file_okay=False, dir_okay=True, resolve_path=True,
        ),
    ] = None,
    force: ForceOpt = False,
) -> None:
    """Bootstrap a new plugin folder with a minimum-valid manifest + skeleton."""
    from src.community.constants import COMMUNITY_ROOT, PLUGIN_KIND_DIRS
    from src.community.scaffold_template import (
        ScaffoldKind,
        class_name_from_id,
        render_manifest,
        render_plugin_py,
        render_readme,
        render_smoke_test,
        validate_plugin_id,
    )

    if kind not in ALL_PLUGIN_KINDS:
        raise die(
            f"--kind must be one of {sorted(ALL_PLUGIN_KINDS)}; got {kind!r}",
        )
    try:
        validate_plugin_id(plugin_id)
    except ValueError as exc:
        raise die(str(exc))

    scaffold_kind: ScaffoldKind = kind  # type: ignore[assignment]
    root_dir = root if root is not None else COMMUNITY_ROOT
    target = root_dir / PLUGIN_KIND_DIRS[kind] / plugin_id
    if target.exists() and not force:
        raise die(
            f"target already exists: {target}",
            hint="pass --force to overwrite",
        )

    target.mkdir(parents=True, exist_ok=True)
    class_name = class_name_from_id(plugin_id)
    (target / "manifest.toml").write_text(
        render_manifest(plugin_id, scaffold_kind, class_name), encoding="utf-8",
    )
    (target / "plugin.py").write_text(
        render_plugin_py(scaffold_kind, plugin_id, class_name), encoding="utf-8",
    )
    (target / "README.md").write_text(
        render_readme(plugin_id, scaffold_kind, class_name), encoding="utf-8",
    )
    tests_dir = target / "tests"
    tests_dir.mkdir(exist_ok=True)
    (tests_dir / "__init__.py").write_text("", encoding="utf-8")
    (tests_dir / "test_plugin.py").write_text(
        render_smoke_test(class_name), encoding="utf-8",
    )
    rel = target.relative_to(root_dir.parent) if root_dir.parent in target.parents or root_dir.parent == target.parent.parent else target
    typer.echo(f"scaffolded {rel}")
    typer.echo("  - manifest.toml")
    typer.echo("  - plugin.py")
    typer.echo("  - README.md")
    typer.echo("  - tests/test_plugin.py")


@plugin_app.command("sync")
def sync_cmd(
    path: Annotated[
        Path,
        typer.Argument(
            help="Plugin or preset folder.",
            exists=True, file_okay=False, dir_okay=True, resolve_path=True,
        ),
    ],
    bump: Annotated[
        str, typer.Option("--bump", "-b", help="Version increment: patch | minor | major."),
    ] = "patch",
    dry_run: DryRunOpt = False,
) -> None:
    """Re-run manifest inference, 3-way-merge with on-disk file, bump version."""
    from src.community.sync import sync_plugin_manifest, sync_preset_manifest

    if bump not in ("patch", "minor", "major"):
        raise typer.BadParameter("--bump must be one of: patch, minor, major")

    is_preset = path.parent.name == "presets" or (path / "preset.yaml").exists()
    runner = sync_preset_manifest if is_preset else sync_plugin_manifest
    try:
        result = runner(path, bump=bump)
    except FileNotFoundError as exc:
        raise die(str(exc))
    if not result.changed:
        typer.echo(f"in sync: {path}")
        return
    typer.echo(result.diff)
    if dry_run:
        typer.echo(f"[dry-run] {path} not modified")
        return
    (path / "manifest.toml").write_text(result.new_text, encoding="utf-8")
    typer.echo(f"updated: {path}/manifest.toml")


@plugin_app.command("sync-envs")
def sync_envs_cmd(
    path: Annotated[
        Path,
        typer.Argument(
            help="Plugin folder — manifest.toml [[required_env]] is rewritten "
                 "from the plugin class's REQUIRED_ENV ClassVar.",
            exists=True, file_okay=False, dir_okay=True, resolve_path=True,
        ),
    ],
    dry_run: DryRunOpt = False,
) -> None:
    """Re-render manifest's ``[[required_env]]`` from ``REQUIRED_ENV``."""
    from src.community.sync import sync_plugin_envs

    manifest_path = path / "manifest.toml"
    if not manifest_path.is_file():
        raise die(f"no manifest.toml in {path}")

    try:
        result = sync_plugin_envs(path)
    except Exception as exc:
        raise die(str(exc))

    if not result.changed:
        typer.echo(f"in sync: {manifest_path}")
        return

    typer.echo(result.diff)
    if dry_run:
        typer.echo(f"[dry-run] {manifest_path} not modified")
        return
    manifest_path.write_text(result.new_text, encoding="utf-8")
    typer.echo(f"updated: {manifest_path}")


@plugin_app.command("pack")
def pack_cmd(
    path: Annotated[
        Path,
        typer.Argument(
            help="Plugin or preset folder to zip.",
            exists=True, file_okay=False, dir_okay=True, resolve_path=True,
        ),
    ],
    force: ForceOpt = False,
    dry_run: DryRunOpt = False,
) -> None:
    """Zip a plugin/preset folder into ``<folder>.zip`` next to it."""
    from src.community.pack import pack_community_folder

    try:
        result = pack_community_folder(path, force=force, dry_run=dry_run)
    except (FileExistsError, FileNotFoundError, NotADirectoryError, ValueError) as exc:
        raise die(str(exc))
    typer.echo(
        f"{'would write' if dry_run else 'wrote'} {result.archive_path}  "
        f"({len(result.files)} files, {result.total_bytes / 1024:.1f} KiB)"
    )


# ---------------------------------------------------------------------------
# Validate manifest (no Python import)
# ---------------------------------------------------------------------------


@plugin_app.command("validate")
def validate_cmd(
    ctx: typer.Context,
    path: Annotated[
        Path,
        typer.Argument(
            help="Plugin / preset folder OR direct path to manifest.toml.",
            exists=True, resolve_path=True,
        ),
    ],
    strict: Annotated[
        bool, typer.Option("--strict", help="Treat warnings as errors."),
    ] = False,
) -> None:
    """Validate a manifest.toml (TOML + Pydantic) without importing the plugin."""
    from src.community.validate_manifest import (
        validate_manifest_dir,
        validate_manifest_file,
    )

    state = ctx.ensure_object(CLIContext)
    renderer = get_renderer(state)
    result = (
        validate_manifest_dir(path) if path.is_dir() else validate_manifest_file(path)
    )

    if state.is_machine_readable:
        renderer.emit({
            "path": str(result.path),
            "kind": result.kind,
            "manifest_id": result.manifest_id,
            "schema_version": result.schema_version,
            "is_valid": result.is_valid,
            "passes_strict": result.passes(strict=True),
            "issues": [
                {"severity": i.severity, "code": i.code,
                 "location": i.location, "message": i.message}
                for i in result.issues
            ],
        })
    else:
        for issue in result.issues:
            prefix = "warning" if issue.severity == "warning" else "error  "
            loc = f"[{issue.location}] " if issue.location else ""
            renderer.text(f"{prefix}: {loc}{issue.message}")
        renderer.text("")
        if result.passes(strict=strict):
            renderer.text("OK")
        else:
            renderer.text("FAIL")
    renderer.flush()
    if not result.passes(strict=strict):
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------


@plugin_app.command("install")
def install_cmd(
    source: Annotated[
        str | None,
        typer.Argument(
            help="Local folder or .zip archive (omit when using --git).",
        ),
    ] = None,
    git: Annotated[
        str | None,
        typer.Option("--git", help="Git URL to clone (requires --ref)."),
    ] = None,
    ref: Annotated[
        str | None,
        typer.Option("--ref", help="Git commit SHA (or branch/tag with --allow-untrusted)."),
    ] = None,
    expected_kind: Annotated[
        str | None,
        typer.Option(
            "--kind",
            help="Expected plugin kind — refuses install on mismatch.",
        ),
    ] = None,
    allow_untrusted: Annotated[
        bool,
        typer.Option(
            "--allow-untrusted",
            help="Allow git installs from a branch / tag (mutable on remote).",
        ),
    ] = False,
    force: ForceOpt = False,
) -> None:
    """Install a plugin from a local folder, .zip archive, or git URL."""
    from src.community.install import (
        InstallError,
        install_git,
        install_local,
    )

    if (source is None) == (git is None):
        raise die(
            "specify exactly one of <source> or --git",
            hint="ryotenkai plugin install ./path  OR  --git URL --ref <sha>",
        )

    try:
        if git is not None:
            if ref is None:
                raise die("--git requires --ref <commit-sha>")
            result = install_git(
                git, ref=ref, expected_kind=expected_kind,
                allow_untrusted=allow_untrusted, force=force,
            )
        else:
            assert source is not None  # mutual exclusivity above
            result = install_local(
                Path(source), expected_kind=expected_kind, force=force,
            )
    except InstallError as exc:
        raise die(exc.message, hint=f"code={exc.code}")

    action = "overwrote" if result.overwritten else "installed"
    typer.echo(
        f"{action}: {result.kind}/{result.plugin_id} → {result.target_path} "
        f"(source={result.source_kind})",
    )


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------


_PREFLIGHT_EXIT: dict[Literal["ok", "missing_envs", "instance_errors", "catalog_error"], int] = {
    "ok": 0,
    "missing_envs": 1,
    "instance_errors": 2,
    "catalog_error": 3,
}


@plugin_app.command("preflight")
def preflight_cmd(
    ctx: typer.Context,
    config: RequiredConfigOpt,
    strict: Annotated[
        bool, typer.Option("--strict", help="Treat any catalog load failure as fatal."),
    ] = False,
) -> None:
    """Pre-launch gate: missing envs + instance-shape errors for a config."""
    from src.community.preflight import run_preflight
    from src.utils.config import load_config

    state = ctx.ensure_object(CLIContext)
    renderer = get_renderer(state)

    try:
        cfg = load_config(config)
    except Exception as exc:
        raise die(f"cannot load config: {exc}", code=_PREFLIGHT_EXIT["catalog_error"])

    try:
        report = run_preflight(cfg)
    except Exception as exc:
        raise die(f"preflight failed: {exc}", code=_PREFLIGHT_EXIT["catalog_error"])

    payload = {
        "ok": report.ok,
        "missing_envs": [
            {"plugin_kind": e.plugin_kind, "plugin_name": e.plugin_name,
             "plugin_instance_id": e.plugin_instance_id, "name": e.name,
             "description": e.description, "secret": e.secret,
             "managed_by": e.managed_by}
            for e in report.missing_envs
        ],
        "instance_errors": [
            {"plugin_kind": e.plugin_kind, "plugin_name": e.plugin_name,
             "plugin_instance_id": e.plugin_instance_id,
             "location": e.location, "message": e.message}
            for e in report.instance_errors
        ],
    }
    if state.is_machine_readable:
        renderer.emit(payload)
    else:
        if report.missing_envs:
            renderer.text(f"Missing envs ({len(report.missing_envs)}):")
            for e in report.missing_envs:
                renderer.text(f"  - {e.plugin_name} [{e.plugin_kind}]: {e.name}")
        if report.instance_errors:
            renderer.text(f"Instance errors ({len(report.instance_errors)}):")
            for e in report.instance_errors:
                renderer.text(
                    f"  - {e.plugin_name} [{e.plugin_kind}].{e.location}: {e.message}",
                )
        if report.ok:
            renderer.text("preflight OK — ready to launch")
    renderer.flush()
    _ = strict  # reserved for future "fail on any catalog warning"

    if report.missing_envs and not report.instance_errors:
        raise typer.Exit(code=_PREFLIGHT_EXIT["missing_envs"])
    if report.instance_errors:
        raise typer.Exit(code=_PREFLIGHT_EXIT["instance_errors"])


# ---------------------------------------------------------------------------
# Stale
# ---------------------------------------------------------------------------


@plugin_app.command("stale")
def stale_cmd(
    ctx: typer.Context,
    config: RequiredConfigOpt,
) -> None:
    """List references to plugins absent from the catalog."""
    from src.community.stale_plugins import find_stale_plugins
    from src.utils.config import load_config

    state = ctx.ensure_object(CLIContext)
    renderer = get_renderer(state)
    cfg = load_config(config)
    stale = find_stale_plugins(cfg)
    payload = [
        {"plugin_kind": s.plugin_kind, "plugin_name": s.plugin_name,
         "instance_id": s.instance_id, "location": s.location}
        for s in stale
    ]
    if state.is_machine_readable:
        renderer.emit(payload)
    elif not stale:
        renderer.text("No stale plugin references found.")
    else:
        renderer.table(
            headers=["Kind", "Plugin", "Instance", "Location"],
            rows=[
                (s.plugin_kind, s.plugin_name, s.instance_id, s.location)
                for s in stale
            ],
        )
    renderer.flush()
    if stale:
        raise typer.Exit(code=1)


__all__ = ["plugin_app"]
