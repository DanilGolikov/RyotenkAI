"""Install a community plugin or preset into ``community/<kind>/<id>/``.

Three sources are supported:

- **local folder** — copy ``source/`` into ``community/<kind>/<id>/``.
- **zip archive** — extract, then handle as a local folder.
- **git URL** — shallow clone, checkout a pinned commit-sha, then handle
  as a local folder. Branches and tags require ``allow_untrusted=True``
  because they are mutable on the remote.

The installer is **strict by default**:

1. The source's ``manifest.toml`` must validate (via
   :mod:`src.community.validate_manifest`) before anything is copied.
2. The kind / id derived from the manifest decide the destination path —
   the user can pin ``expected_kind`` to refuse silent kind drift.
3. Existing destinations are protected: ``force=False`` raises rather
   than overwriting a user-edited folder.

The function never imports the plugin's Python module — manifest
validation is enough to gate the copy. Importability is the loader's
problem at runtime; surfacing it during ``install`` would force the user
to ``pip install`` the plugin's deps before any directory move.

This module has **no Typer-level concerns** — the CLI command in
``src/cli/commands/plugin.py`` is a thin wrapper that maps
:class:`InstallError` to typer exit codes and formats :class:`InstallResult`
through the renderer.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from src.community.archive import resolve_extraction_root
from src.community.constants import (
    ALL_PLUGIN_KINDS,
    COMMUNITY_ROOT,
    PRESET_DIR_NAME,
)
from src.community.validate_manifest import (
    ManifestValidationResult,
    validate_manifest_dir,
)
from src.utils.logger import logger

#: Accepts either a 40-char SHA-1 (full commit hash) or a >=7-char hex
#: short hash. Matches the form ``git rev-parse <ref>`` would emit.
_COMMIT_SHA_RE: re.Pattern[str] = re.compile(r"^[0-9a-f]{7,40}$")

#: Subset of :data:`ALL_PLUGIN_KINDS` plus ``"preset"``. Validated here
#: instead of letting downstream rejects fire — the user expects a
#: prompt error for ``--kind nonsense``.
ALLOWED_KINDS: frozenset[str] = frozenset({*ALL_PLUGIN_KINDS, "preset"})

InstallSourceKind = Literal["folder", "archive", "git"]


@dataclass(frozen=True, slots=True)
class InstallResult:
    """Outcome of a successful install."""

    plugin_id: str
    kind: str
    source_kind: InstallSourceKind
    target_path: Path
    overwritten: bool


class InstallError(Exception):
    """Raised when an install cannot proceed.

    ``code`` is a stable, machine-readable category — CLI maps it to
    exit codes, callers can match on it for programmatic handling.
    """

    Code = Literal[
        "source_not_found",
        "source_invalid",
        "manifest_invalid",
        "kind_mismatch",
        "kind_unknown",
        "destination_exists",
        "git_unavailable",
        "git_clone_failed",
        "git_checkout_failed",
        "git_ref_drift",
        "untrusted_git_ref",
    ]

    def __init__(self, code: Code, message: str) -> None:
        super().__init__(message)
        self.code: InstallError.Code = code
        self.message: str = message


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def install_local(
    source: Path,
    *,
    expected_kind: str | None = None,
    force: bool = False,
    community_root: Path | None = None,
) -> InstallResult:
    """Install a local folder or ``.zip`` archive.

    ``expected_kind`` (when set) must match the manifest's kind — a
    safety net for users who think they're installing a validation
    plugin but the manifest declares ``reward``.
    """
    _check_kind_value(expected_kind)
    if not source.exists():
        raise InstallError("source_not_found", f"source does not exist: {source}")

    if source.is_file() and source.suffix == ".zip":
        with _extracted_archive(source) as folder:
            return _install_from_folder(
                folder, source_kind="archive", expected_kind=expected_kind,
                force=force, community_root=community_root,
            )
    if source.is_dir():
        return _install_from_folder(
            source, source_kind="folder", expected_kind=expected_kind,
            force=force, community_root=community_root,
        )
    raise InstallError(
        "source_invalid",
        f"source must be a directory or .zip archive: {source}",
    )


def install_git(
    url: str,
    *,
    ref: str,
    expected_kind: str | None = None,
    allow_untrusted: bool = False,
    force: bool = False,
    community_root: Path | None = None,
) -> InstallResult:
    """Install from a git URL pinned to a commit sha.

    Branches and tags are mutable on the remote (force-push, retag) —
    they are rejected unless ``allow_untrusted=True``. After clone +
    checkout the installer compares the resolved sha against ``ref``;
    a mismatch (which can happen if the remote rewrote history during
    the clone window) aborts with ``git_ref_drift``.

    The clone is shallow (``--depth 1`` then ``fetch <ref>``) so even
    huge plugin repos take a single network round-trip.
    """
    _check_kind_value(expected_kind)
    _ensure_git_available()

    is_pinned_sha = bool(_COMMIT_SHA_RE.fullmatch(ref))
    if not is_pinned_sha and not allow_untrusted:
        raise InstallError(
            "untrusted_git_ref",
            f"ref {ref!r} is not a commit sha; pass --allow-untrusted to "
            f"install from a mutable branch or tag",
        )

    with tempfile.TemporaryDirectory(prefix="ryotenkai-install-git-") as tmp:
        clone_dir = Path(tmp) / "clone"
        _git_clone(url, clone_dir)
        _git_checkout(clone_dir, ref)
        if is_pinned_sha:
            resolved = _git_rev_parse(clone_dir, "HEAD")
            if not resolved.startswith(ref) and not ref.startswith(resolved):
                raise InstallError(
                    "git_ref_drift",
                    f"checked-out HEAD ({resolved}) does not match the "
                    f"requested ref ({ref}); the remote may have rewritten history",
                )
        return _install_from_folder(
            clone_dir, source_kind="git", expected_kind=expected_kind,
            force=force, community_root=community_root,
        )


# ---------------------------------------------------------------------------
# Internals — folder install
# ---------------------------------------------------------------------------


def _install_from_folder(
    folder: Path,
    *,
    source_kind: InstallSourceKind,
    expected_kind: str | None,
    force: bool,
    community_root: Path | None,
) -> InstallResult:
    validation = validate_manifest_dir(folder)
    if not validation.is_valid:
        raise InstallError(
            "manifest_invalid",
            _format_manifest_errors(validation),
        )

    kind, plugin_id = _resolve_kind_and_id(validation)
    if expected_kind is not None and expected_kind != kind:
        raise InstallError(
            "kind_mismatch",
            f"manifest declares kind={kind!r}, but caller expected "
            f"kind={expected_kind!r}",
        )

    root = (community_root or COMMUNITY_ROOT).expanduser().resolve()
    target = root / _kind_subdir(kind) / plugin_id
    overwritten = target.exists()
    if overwritten and not force:
        raise InstallError(
            "destination_exists",
            f"target already exists: {target}; pass force=True to overwrite",
        )

    if overwritten:
        shutil.rmtree(target)

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(folder, target, dirs_exist_ok=False, ignore=_ignore_caches)
    logger.info(
        "[community.install] %s %r installed (kind=%s) → %s "
        "(source_kind=%s, overwritten=%s)",
        "preset" if kind == "preset" else "plugin",
        plugin_id, kind, target, source_kind, overwritten,
    )
    return InstallResult(
        plugin_id=plugin_id, kind=kind, source_kind=source_kind,
        target_path=target, overwritten=overwritten,
    )


def _resolve_kind_and_id(
    validation: ManifestValidationResult,
) -> tuple[str, str]:
    """Pull (kind, id) out of a validated manifest result.

    For plugin manifests the kind comes from ``[plugin] kind`` (re-parsed
    from the file because the validator only exposes the manifest_id);
    presets always have ``kind="preset"``.
    """
    if validation.manifest_id is None:
        # Should never happen for an is_valid result, but bail loudly.
        raise InstallError(
            "manifest_invalid",
            "manifest has no id after validation — internal contract bug",
        )
    if validation.kind == "preset":
        return "preset", validation.manifest_id
    if validation.kind == "plugin":
        # Re-read the file just to grab the kind — cheap and avoids
        # threading another field through ManifestValidationResult.
        import tomllib
        payload = tomllib.loads(validation.path.read_text(encoding="utf-8"))
        plugin_kind = payload.get("plugin", {}).get("kind")
        if not isinstance(plugin_kind, str):
            raise InstallError(
                "manifest_invalid",
                "plugin manifest missing [plugin].kind",
            )
        return plugin_kind, validation.manifest_id
    raise InstallError(
        "manifest_invalid",
        f"unknown manifest kind: {validation.kind}",
    )


def _kind_subdir(kind: str) -> str:
    """Map kind → community subdirectory name.

    Plugin kinds map 1:1 (``"validation"`` → ``"validation"``);
    presets live under :data:`PRESET_DIR_NAME`.
    """
    if kind == "preset":
        return PRESET_DIR_NAME
    if kind in ALL_PLUGIN_KINDS:
        return kind
    raise InstallError("kind_unknown", f"unknown kind: {kind!r}")


def _check_kind_value(expected_kind: str | None) -> None:
    if expected_kind is None:
        return
    if expected_kind not in ALLOWED_KINDS:
        raise InstallError(
            "kind_unknown",
            f"--kind must be one of {sorted(ALLOWED_KINDS)}; got {expected_kind!r}",
        )


def _ignore_caches(_dir: str, names: list[str]) -> list[str]:
    """``shutil.copytree`` ignore-callback: skip caches & VCS noise."""
    skip = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
            ".git", ".github", ".venv", "node_modules", ".DS_Store"}
    return [n for n in names if n in skip]


def _format_manifest_errors(validation: ManifestValidationResult) -> str:
    """Render the first few errors compactly for the InstallError message."""
    lines = [f"manifest at {validation.path} failed validation:"]
    for issue in validation.errors[:5]:
        loc = f" [{issue.location}]" if issue.location else ""
        lines.append(f"  - {issue.code}{loc}: {issue.message}")
    if len(validation.errors) > 5:
        lines.append(f"  ... ({len(validation.errors) - 5} more)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Archive helper
# ---------------------------------------------------------------------------


@contextmanager
def _extracted_archive(zip_path: Path):
    """Yield the manifest-bearing root inside an extracted ``.zip``.

    Reuses :func:`src.community.archive.resolve_extraction_root` so a
    zip that wraps the plugin in a single top-level folder is handled
    the same way the loader handles cached extractions.
    """
    with tempfile.TemporaryDirectory(prefix="ryotenkai-install-zip-") as tmp:
        target = Path(tmp) / "extracted"
        try:
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(target)
        except zipfile.BadZipFile as exc:
            raise InstallError("source_invalid", f"bad zip archive: {exc}") from exc
        yield resolve_extraction_root(target)


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------


def _ensure_git_available() -> None:
    if shutil.which("git") is None:
        raise InstallError(
            "git_unavailable",
            "git executable not found in PATH; install git or use a "
            "local source",
        )


def _git_clone(url: str, target: Path) -> None:
    """Shallow-clone ``url`` to ``target``.

    ``--no-single-branch --depth 50`` is the smallest invocation that
    still lets ``git checkout <commit>`` succeed for typical small
    plugin repos. If a 50-commit window doesn't reach the requested
    commit the install fails on checkout — the user pins a deeper ref
    or upgrades the install to a full clone (future flag).
    """
    cmd = ["git", "clone", "--depth", "50", "--no-single-branch", url, str(target)]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise InstallError(
            "git_clone_failed",
            f"git clone {url!r} failed: {proc.stderr.strip() or proc.stdout.strip()}",
        )


def _git_checkout(repo: Path, ref: str) -> None:
    proc = subprocess.run(
        ["git", "-C", str(repo), "checkout", "--detach", ref],
        capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0:
        raise InstallError(
            "git_checkout_failed",
            f"git checkout {ref!r} failed: {proc.stderr.strip() or proc.stdout.strip()}",
        )


def _git_rev_parse(repo: Path, ref: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", ref],
        capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0:
        raise InstallError(
            "git_checkout_failed",
            f"git rev-parse {ref!r} failed: {proc.stderr.strip() or proc.stdout.strip()}",
        )
    return proc.stdout.strip()


__all__ = [
    "ALLOWED_KINDS",
    "InstallError",
    "InstallResult",
    "InstallSourceKind",
    "install_git",
    "install_local",
]
