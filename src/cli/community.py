"""``ryotenkai community …`` — scaffold, sync and pack community manifests.

The CLI accepts three kinds of path targets and picks the right action
automatically:

* **Leaf folder** — one plugin or preset (``community/validation/avg_length``)
* **Kind folder** — every plugin/preset inside
  (``community/validation`` → all 10 validators; ``community/presets`` →
  all 3 presets)
* **Community root** — every plugin and preset under ``community/``

Run ``ryotenkai community help`` to see the full command list. Each
command also accepts ``--help`` for its own flags.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

import typer

from src.community.constants import (
    ALL_PLUGIN_KINDS,
    COMMUNITY_ROOT,
    PRESET_DIR_NAME,
)
from src.community.pack import PackResult, pack_community_folder
from src.community.scaffold import scaffold_plugin_manifest, scaffold_preset_manifest
from src.community.sync import SyncResult, sync_plugin_manifest, sync_preset_manifest

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

community_app = typer.Typer(
    no_args_is_help=True,
    help=(
        "Scaffold, sync and pack community/ plugin and preset manifests.\n\n"
        "Accepts a single plugin/preset folder, an entire kind folder "
        "(e.g. community/validation/), or the community/ root."
    ),
    rich_markup_mode="rich",
    context_settings={"help_option_names": ["-h", "--help"]},
)


# ---------------------------------------------------------------------------
# Path classification & batching
# ---------------------------------------------------------------------------


PathRole = Literal[
    "plugin",           # leaf folder with plugin.py / plugin/ package
    "preset",           # leaf folder with *.yaml under community/presets/
    "kind_dir",         # community/{validation,evaluation,reward,reports,presets}
    "community_root",   # community/ itself
    "unknown",          # anything else
]


def _classify_path(path: Path) -> PathRole:
    """Decide what kind of target ``path`` is, without touching the filesystem
    beyond ``is_dir`` / ``is_file`` probes.
    """
    if not path.is_dir():
        return "unknown"

    resolved = path.resolve()

    # community/ itself — either the bundled path, or any folder named
    # "community" that holds at least one known kind subdir (so the
    # command works from any checkout / worktree without a CWD dance).
    try:
        if resolved == COMMUNITY_ROOT.resolve():
            return "community_root"
    except FileNotFoundError:
        pass
    if resolved.name == "community" and _looks_like_community_root(resolved):
        return "community_root"

    # Kind dir — direct child of a "community" folder with a known name.
    if (
        resolved.parent.name == "community"
        and resolved.name in (*ALL_PLUGIN_KINDS, PRESET_DIR_NAME)
    ):
        return "kind_dir"

    # Preset leaf — parent is "presets" and folder contains at least one yaml.
    if resolved.parent.name == PRESET_DIR_NAME and _has_yaml(resolved):
        return "preset"

    # Plugin leaf — folder contains plugin.py or plugin/__init__.py.
    if (resolved / "plugin.py").is_file():
        return "plugin"
    if (resolved / "plugin" / "__init__.py").is_file():
        return "plugin"

    # Fallback: manifest present + yaml present → treat as preset (e.g. user
    # dropped a preset into a non-standard parent and used --kind preset).
    if (resolved / "manifest.toml").is_file() and _has_yaml(resolved):
        return "preset"

    return "unknown"


def _has_yaml(path: Path) -> bool:
    return any(path.glob("*.yaml"))


def _looks_like_community_root(path: Path) -> bool:
    """True if ``path`` has at least one known kind subdirectory."""
    known = {*ALL_PLUGIN_KINDS, PRESET_DIR_NAME}
    return any((path / name).is_dir() for name in known)


def _iter_subfolders(path: Path) -> list[Path]:
    return sorted(
        p for p in path.iterdir()
        if p.is_dir() and not p.name.startswith(".") and not p.name.startswith("__")
    )


@dataclass(frozen=True, slots=True)
class Target:
    path: Path
    role: Literal["plugin", "preset"]


def _collect_targets(path: Path, *, kind_override: str | None = None) -> list[Target]:
    """Expand the user-supplied path into a flat list of leaf targets.

    * Leaf folder → [Target(that folder)]
    * Kind folder → every plausible leaf inside
    * Community root → every leaf under every kind folder
    """
    role_override = _role_from_kind_flag(kind_override)

    classification = _classify_path(path)

    if classification in ("plugin", "preset"):
        role = role_override or classification
        return [Target(path=path, role=role)]  # type: ignore[arg-type]

    if classification == "kind_dir":
        return _leaves_in_kind_dir(path, role_override=role_override)

    if classification == "community_root":
        targets: list[Target] = []
        for kind_dir in _iter_subfolders(path):
            if _classify_path(kind_dir) == "kind_dir":
                targets.extend(_leaves_in_kind_dir(kind_dir))
        return targets

    # Unknown: caller decides how to surface the error.
    if role_override is not None:
        return [Target(path=path, role=role_override)]
    return []


def _role_from_kind_flag(kind: str | None) -> Literal["plugin", "preset"] | None:
    if kind in (None, "auto"):
        return None
    if kind in ("plugin", "preset"):
        return kind  # type: ignore[return-value]
    raise typer.BadParameter("--kind must be 'auto', 'plugin' or 'preset'")


def _leaves_in_kind_dir(
    kind_dir: Path,
    *,
    role_override: Literal["plugin", "preset"] | None = None,
) -> list[Target]:
    default_role: Literal["plugin", "preset"] = (
        "preset" if kind_dir.name == PRESET_DIR_NAME else "plugin"
    )
    role = role_override or default_role
    targets: list[Target] = []
    for leaf in _iter_subfolders(kind_dir):
        classification = _classify_path(leaf)
        if classification in ("plugin", "preset"):
            targets.append(Target(path=leaf, role=classification))
        else:
            # Not a valid leaf but the user asked for the whole kind dir;
            # include it with the default role so sync/scaffold can fail
            # loudly with a useful message rather than silently skipping.
            targets.append(Target(path=leaf, role=role))
    return targets


# ---------------------------------------------------------------------------
# Pretty output helpers
# ---------------------------------------------------------------------------


def _style_ok(text: str) -> str:
    return typer.style(text, fg=typer.colors.GREEN)


def _style_dim(text: str) -> str:
    return typer.style(text, dim=True)


def _style_warn(text: str) -> str:
    return typer.style(text, fg=typer.colors.YELLOW)


def _style_err(text: str) -> str:
    return typer.style(text, fg=typer.colors.RED)


def _style_label(text: str) -> str:
    return typer.style(text, bold=True)


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def _usage_hint() -> str:
    return (
        "\nExpected one of:\n"
        "  • a plugin folder with plugin.py "
        "(e.g. community/validation/min_samples)\n"
        "  • a preset folder with *.yaml "
        "(e.g. community/presets/01-small)\n"
        "  • a kind folder (community/validation, community/presets, …)\n"
        "  • the community/ root\n"
    )


def _die(message: str) -> None:
    typer.echo(_style_err(f"error: {message}"), err=True)
    raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# help alias (so `ryotenkai community help` works alongside `--help`)
# ---------------------------------------------------------------------------


@community_app.command("help", hidden=True)
def _help_cmd(ctx: typer.Context) -> None:
    """Show this help message."""
    parent = ctx.parent
    typer.echo(parent.get_help() if parent is not None else ctx.get_help())


# ---------------------------------------------------------------------------
# scaffold
# ---------------------------------------------------------------------------


@community_app.command("scaffold")
def scaffold_cmd(
    path: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=True,
            file_okay=False,
            resolve_path=True,
            help=(
                "A single plugin/preset folder, a kind folder "
                "(community/validation), or community/ itself."
            ),
        ),
    ],
    force: Annotated[
        bool,
        typer.Option("--force", "-f", help="Overwrite existing manifest.toml files."),
    ] = False,
    kind: Annotated[
        str,
        typer.Option(
            "--kind",
            help="'auto' (default) | 'plugin' | 'preset'. Override when the folder lives outside community/.",
        ),
    ] = "auto",
) -> None:
    """Generate fresh manifest.toml files for plugin/preset folders.

    Fields that can be inferred — entry_point, kind, params/thresholds
    schema, required secrets, description from class docstring — are
    filled in automatically. Fields that need human judgement (category,
    stability, preset description) are emitted with a [bold]# TODO[/bold]
    marker so nothing sneaks past the author.

    [bold]Examples[/bold]

        ryotenkai community scaffold community/validation/my_plugin
        ryotenkai community scaffold community/validation   # every plugin without a manifest
        ryotenkai community scaffold community/ --force     # regenerate everything
    """
    targets = _resolve_targets_or_die(path, kind_override=kind)
    is_batch = len(targets) > 1 or _classify_path(path) in ("kind_dir", "community_root")

    if is_batch:
        _scaffold_batch(targets, path=path, force=force)
    else:
        _scaffold_single(targets[0], force=force)


def _scaffold_single(target: Target, *, force: bool) -> None:
    manifest_path = target.path / "manifest.toml"
    if manifest_path.exists() and not force:
        typer.echo(
            _style_err(
                f"error: {_rel(manifest_path)} already exists; pass --force to overwrite"
            ),
            err=True,
        )
        raise typer.Exit(code=1)
    try:
        text = (
            scaffold_preset_manifest(target.path)
            if target.role == "preset"
            else scaffold_plugin_manifest(target.path)
        )
    except FileNotFoundError as exc:
        typer.echo(_style_err(f"error: {exc}"), err=True)
        typer.echo(_usage_hint(), err=True)
        raise typer.Exit(code=1) from exc
    manifest_path.write_text(text, encoding="utf-8")
    typer.echo(f"{_style_ok('wrote')} {_rel(manifest_path)}")
    typer.echo(
        _style_dim(
            f"  next: edit TODO fields, then `ryotenkai community sync {_rel(target.path)}`"
        )
    )


def _scaffold_batch(targets: list[Target], *, path: Path, force: bool) -> None:
    if not targets:
        _die(f"no plugin or preset folders found under {_rel(path)}")

    typer.echo(
        _style_label(
            f"Scaffolding manifests under {_rel(path)} "
            f"({len(targets)} folder{'s' if len(targets) != 1 else ''}):"
        )
    )
    created = 0
    skipped = 0
    errors = 0
    for target in targets:
        manifest_path = target.path / "manifest.toml"
        if manifest_path.exists() and not force:
            typer.echo(f"  {_style_dim('·')} {_rel(target.path)}  skipped (has manifest.toml)")
            skipped += 1
            continue
        try:
            text = (
                scaffold_preset_manifest(target.path)
                if target.role == "preset"
                else scaffold_plugin_manifest(target.path)
            )
        except FileNotFoundError as exc:
            typer.echo(
                f"  {_style_err('✗')} {_rel(target.path)}  {exc}",
                err=True,
            )
            errors += 1
            continue
        action = "overwrote" if manifest_path.exists() else "wrote"
        manifest_path.write_text(text, encoding="utf-8")
        typer.echo(f"  {_style_ok('✓')} {_rel(target.path)}  {action}")
        created += 1
    typer.echo(
        _style_label(
            f"\nSummary: {created} written, {skipped} skipped, {errors} error"
            f"{'s' if errors != 1 else ''}."
        )
    )
    if errors:
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# sync
# ---------------------------------------------------------------------------


@community_app.command("sync")
def sync_cmd(
    path: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=True,
            file_okay=False,
            resolve_path=True,
            help=(
                "A single plugin/preset folder, a kind folder "
                "(community/validation), or community/ itself."
            ),
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
            help="Print what would change; do not modify any manifest.toml.",
        ),
    ] = False,
    kind: Annotated[
        str,
        typer.Option("--kind", help="'auto' (default) | 'plugin' | 'preset'."),
    ] = "auto",
) -> None:
    """Re-run inference over source code, 3-way-merge with the existing
    manifest, bump the version, and print/apply the diff.

    Always review the diff first — deletions in ``params_schema`` happen
    when a parameter is no longer referenced from the plugin's source
    code.

    [bold]Examples[/bold]

        ryotenkai community sync community/validation/min_samples
        ryotenkai community sync community/validation --dry-run
        ryotenkai community sync community/ --bump minor
    """
    if bump not in ("patch", "minor", "major"):
        raise typer.BadParameter("--bump must be one of: patch, minor, major")

    targets = _resolve_targets_or_die(path, kind_override=kind)
    is_batch = len(targets) > 1 or _classify_path(path) in ("kind_dir", "community_root")

    if is_batch:
        _sync_batch(targets, path=path, bump=bump, dry_run=dry_run)
    else:
        _sync_single(targets[0], bump=bump, dry_run=dry_run)


def _run_sync(target: Target, *, bump: str) -> SyncResult:
    runner = sync_preset_manifest if target.role == "preset" else sync_plugin_manifest
    return runner(target.path, bump=bump)  # type: ignore[arg-type]


def _sync_single(target: Target, *, bump: str, dry_run: bool) -> None:
    manifest_path = target.path / "manifest.toml"
    if not manifest_path.is_file():
        _missing_manifest_error(target)

    try:
        result = _run_sync(target, bump=bump)
    except FileNotFoundError as exc:
        typer.echo(_style_err(f"error: {exc}"), err=True)
        typer.echo(_usage_hint(), err=True)
        raise typer.Exit(code=1) from exc

    if not result.changed:
        typer.echo(
            f"{_style_dim('·')} {_rel(manifest_path)} already in sync (no changes)"
        )
        return

    typer.echo(result.diff)
    if dry_run:
        typer.echo(_style_warn("[dry-run]") + f" {_rel(manifest_path)} not modified")
        return

    manifest_path.write_text(result.new_text, encoding="utf-8")
    typer.echo(f"{_style_ok('✓')} updated {_rel(manifest_path)}")


def _sync_batch(
    targets: list[Target], *, path: Path, bump: str, dry_run: bool
) -> None:
    if not targets:
        _die(f"no plugin or preset folders found under {_rel(path)}")

    banner = (
        f"Syncing manifests under {_rel(path)} "
        f"({len(targets)} folder{'s' if len(targets) != 1 else ''}, bump={bump}"
        + (", dry-run" if dry_run else "")
        + "):"
    )
    typer.echo(_style_label(banner))

    updated = 0
    unchanged = 0
    missing = 0
    errors = 0
    for target in targets:
        rel = _rel(target.path)
        manifest_path = target.path / "manifest.toml"
        if not manifest_path.is_file():
            typer.echo(
                f"  {_style_warn('⚠')} {rel}  no manifest.toml — run `scaffold` first"
            )
            missing += 1
            continue
        try:
            result = _run_sync(target, bump=bump)
        except (FileNotFoundError, ValueError) as exc:
            typer.echo(f"  {_style_err('✗')} {rel}  {exc}")
            errors += 1
            continue

        if not result.changed:
            typer.echo(f"  {_style_dim('·')} {rel}  in sync")
            unchanged += 1
            continue

        old_version = _extract_version(result.old_text)
        new_version = _extract_version(result.new_text)
        label = (
            f"{old_version} → {new_version}"
            if old_version and new_version
            else "will change"
        )
        if dry_run:
            typer.echo(f"  {_style_warn('~')} {rel}  {label} [dry-run]")
        else:
            manifest_path.write_text(result.new_text, encoding="utf-8")
            typer.echo(f"  {_style_ok('✓')} {rel}  {label}")
        updated += 1

    typer.echo(
        _style_label(
            f"\nSummary: {updated} changed, {unchanged} in sync, "
            f"{missing} missing manifest, {errors} error"
            f"{'s' if errors != 1 else ''}."
        )
    )
    if dry_run and updated:
        typer.echo(
            _style_dim(
                "  tip: re-run without --dry-run to apply, "
                "or run sync on a single folder to see the full diff."
            )
        )
    if errors:
        raise typer.Exit(code=1)


def _extract_version(text: str) -> str | None:
    """Cheap line scan — avoids a full TOML parse for summary output."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("version"):
            # version = "1.2.3"
            _, _, value = stripped.partition("=")
            return value.strip().strip('"').strip("'") or None
    return None


def _missing_manifest_error(target: Target) -> None:
    """Produce a useful error when a single-folder sync points at a folder
    without ``manifest.toml``. Covers the three common user mistakes.
    """
    path = target.path
    looks_like_plugin = (path / "plugin.py").is_file() or (path / "plugin").is_dir()
    looks_like_preset = _has_yaml(path)

    lines = [f"no manifest.toml in {_rel(path)}"]
    if looks_like_plugin or looks_like_preset:
        lines.append(
            f"  hint: run `ryotenkai community scaffold {_rel(path)}` first "
            "to create the manifest."
        )
    else:
        lines.append(_usage_hint().rstrip())
    _die("\n".join(lines))


# ---------------------------------------------------------------------------
# pack
# ---------------------------------------------------------------------------


@community_app.command("pack")
def pack_cmd(
    path: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=True,
            file_okay=False,
            resolve_path=True,
            help=(
                "A plugin/preset folder, a kind folder, or community/ "
                "to pack every subfolder."
            ),
        ),
    ],
    force: Annotated[
        bool,
        typer.Option("--force", "-f", help="Overwrite existing <folder>.zip archives."),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="List files, do not create archives."),
    ] = False,
) -> None:
    """Pack plugin/preset folders into zip archives next to each folder.

    Each archive is placed alongside the source folder (inside its kind
    directory) so the community loader picks it up automatically — the
    source folder can then be deleted and everything keeps working.
    Build caches (``__pycache__`` / ``.pytest_cache`` / …) are filtered
    out. The manifest is validated before packing.

    [bold]Examples[/bold]

        ryotenkai community pack community/validation/min_samples
        ryotenkai community pack community/validation
        ryotenkai community pack community/ --force
    """
    targets = _resolve_targets_or_die(path, kind_override=None)
    is_batch = len(targets) > 1 or _classify_path(path) in ("kind_dir", "community_root")

    if is_batch:
        _pack_batch(targets, path=path, force=force, dry_run=dry_run)
    else:
        _pack_single(targets[0].path, force=force, dry_run=dry_run)


def _pack_single(folder: Path, *, force: bool, dry_run: bool) -> None:
    try:
        result = pack_community_folder(folder, force=force, dry_run=dry_run)
    except FileExistsError as exc:
        _die(str(exc))
        return
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        typer.echo(_style_err(f"error: {exc}"), err=True)
        raise typer.Exit(code=1) from exc

    _print_pack_line(result, dry_run=dry_run, prefix="")
    if dry_run:
        for name in result.files:
            typer.echo(f"    {name}")


def _pack_batch(
    targets: list[Target], *, path: Path, force: bool, dry_run: bool
) -> None:
    if not targets:
        _die(f"no plugin or preset folders found under {_rel(path)}")

    banner = (
        f"Packing folders under {_rel(path)} "
        f"({len(targets)} folder{'s' if len(targets) != 1 else ''}"
        + (", dry-run" if dry_run else "")
        + "):"
    )
    typer.echo(_style_label(banner))
    packed = 0
    skipped = 0
    errors = 0
    for target in targets:
        try:
            result = pack_community_folder(target.path, force=force, dry_run=dry_run)
        except FileExistsError:
            typer.echo(
                f"  {_style_dim('·')} {_rel(target.path)}  zip exists "
                f"(use --force to overwrite)"
            )
            skipped += 1
            continue
        except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
            typer.echo(f"  {_style_err('✗')} {_rel(target.path)}  {exc}")
            errors += 1
            continue
        _print_pack_line(result, dry_run=dry_run, prefix="  ")
        packed += 1
    typer.echo(
        _style_label(
            f"\nSummary: {packed} packed, {skipped} skipped, {errors} error"
            f"{'s' if errors != 1 else ''}."
        )
    )
    if errors:
        raise typer.Exit(code=1)


def _print_pack_line(result: PackResult, *, dry_run: bool, prefix: str) -> None:
    size_kib = result.total_bytes / 1024
    marker = _style_warn("~") if dry_run else _style_ok("✓")
    verb = "would write" if dry_run else "wrote"
    suffix = (
        f"{_rel(result.archive_path)}  "
        f"({len(result.files)} files, {size_kib:.1f} KiB)"
    )
    typer.echo(f"{prefix}{marker} {verb} {suffix}")


# ---------------------------------------------------------------------------
# Common: resolve targets with a good error message
# ---------------------------------------------------------------------------


def _resolve_targets_or_die(
    path: Path, *, kind_override: str | None
) -> list[Target]:
    targets = _collect_targets(path, kind_override=kind_override)
    if targets:
        return targets

    # Empty → classify why and surface a pointed message.
    classification = _classify_path(path)
    if classification in ("community_root", "kind_dir"):
        _die(f"{_rel(path)} contains no plugin or preset folders")
    typer.echo(
        _style_err(
            f"error: cannot tell whether {_rel(path)} is a plugin or a preset"
        ),
        err=True,
    )
    typer.echo(_usage_hint(), err=True)
    typer.echo(
        _style_dim(
            "  hint: pass --kind plugin or --kind preset to force a role."
        ),
        err=True,
    )
    raise typer.Exit(code=1)


# Legacy single-path detection kept for the tests that exercise it directly.
def _detect_is_preset(path: Path, *, override: str | None = None) -> bool:
    role_override = _role_from_kind_flag(override)
    if role_override is not None:
        return role_override == "preset"
    return _classify_path(path) == "preset"


# Convenience re-export for tests/external callers that want a single entry.
def _iter_target_paths(path: Path) -> Iterable[Path]:
    for target in _collect_targets(path, kind_override=None):
        yield target.path


__all__ = ["community_app"]
