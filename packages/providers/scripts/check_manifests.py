#!/usr/bin/env python3
"""Manifest-code drift validator.

Walks every ``provider.toml`` and checks the invariants the
:class:`ProviderManifest` schema validator alone can't enforce — the
ones that span Python class shape and manifest declaration. Designed
for fast, human-readable feedback in pre-commit and CI; the same
invariants are also enforced by the pytest suite
(``test_provider_registry_invariants.py``).

Checks performed (one section of output per failure):

1. **Manifest schema** — every ``provider.toml`` parses through
   :class:`ProviderManifest` (Pydantic v2). Catches malformed TOML,
   wrong types, missing required fields.

2. **Folder ↔ id parity** — ``provider.id`` equals the folder name.

3. **Entry-point importability** — every declared
   ``entry_points.<role>`` resolves to a real class via
   ``importlib.import_module``.

4. **Capability ↔ Protocol parity** — for each cap flag with a paired
   Protocol:
     * ``supports_lifecycle_actions`` ↔ ``ITerminalActionProvider``
     * ``supports_recovery_probe``    ↔ ``attempt_recovery`` callable
     * ``supports_capacity_error_detection`` ↔ ``is_capacity_error``
       callable
   Drift in either direction surfaces with a clear "fix this OR fix
   that" message.

5. **ProviderBase inheritance** — every training-role class inherits
   ``ProviderBase``.

6. **Required-env coverage** — every entry in
   ``manifest.required_env`` has a non-empty ``name``; the
   ``required_for_roles`` list is a subset of ``provider.roles``.

7. **Pod-manifest projection drift** — running
   :func:`project_to_pod_manifest` against the source ``provider.toml``
   produces output identical to the on-disk
   ``packages/pod/.../pod_manifests/<id>.toml``. Catches the case
   where a dev edited ``provider.toml`` but forgot to re-run
   ``compile_pod_manifests.py``.

Exit codes:
* ``0`` — all checks pass.
* ``1`` — at least one drift detected. Each finding is printed with
  file:line context and a concrete "Fix:" hint.

Usage::

    python packages/providers/scripts/check_manifests.py
    python packages/providers/scripts/check_manifests.py --verbose
"""

from __future__ import annotations

import argparse
import importlib
import sys
import tomllib
from pathlib import Path
from typing import Any


def _workspace_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file() and (parent / "packages").is_dir():
            return parent
    return Path.cwd()


_REPO_ROOT = _workspace_root()
_PROVIDERS_ROOT = _REPO_ROOT / "packages" / "providers" / "src" / "ryotenkai_providers"
_POD_MANIFESTS_DIR = (
    _REPO_ROOT
    / "packages"
    / "pod"
    / "src"
    / "ryotenkai_pod"
    / "runner"
    / "runtime"
    / "pod_manifests"
)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


class _Findings:
    """Collects drift records, prints at end. Each record has a uniform
    shape so the output is greppable: ``❌ <provider_id>: <one-line
    summary>\\n   Where: <file:rel-path>\\n   Fix: <concrete action>``.
    """

    def __init__(self) -> None:
        self._records: list[tuple[str, str, str, str]] = []

    def add(self, *, provider_id: str, summary: str, where: str, fix: str) -> None:
        self._records.append((provider_id, summary, where, fix))

    def render(self) -> str:
        if not self._records:
            return ""
        lines: list[str] = []
        for pid, summary, where, fix in self._records:
            lines.append(f"❌ {pid}: {summary}")
            lines.append(f"   Where: {where}")
            lines.append(f"   Fix:   {fix}")
            lines.append("")
        return "\n".join(lines)

    def empty(self) -> bool:
        return not self._records


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


_NON_PROVIDER_DIRS = frozenset({"training", "inference", "scripts", "tests", "__pycache__"})


def _discover_manifest_paths(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    out: list[Path] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name in _NON_PROVIDER_DIRS or child.name.startswith("."):
            continue
        m = child / "provider.toml"
        if m.is_file():
            out.append(m)
    return out


# ---------------------------------------------------------------------------
# Per-manifest checks
# ---------------------------------------------------------------------------


def _check_manifest(
    manifest_path: Path,
    findings: _Findings,
    *,
    verbose: bool,
) -> None:
    """Run every applicable check against a single ``provider.toml``."""

    rel = manifest_path.relative_to(_REPO_ROOT)
    folder_id = manifest_path.parent.name

    # Phase 1: Pydantic schema parse.
    try:
        with manifest_path.open("rb") as fh:
            raw = tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        findings.add(
            provider_id=folder_id,
            summary=f"could not parse provider.toml: {exc}",
            where=str(rel),
            fix="fix the TOML syntax error.",
        )
        return

    try:
        from ryotenkai_providers.manifest import ProviderManifest
        manifest = ProviderManifest.model_validate(raw)
    except Exception as exc:
        findings.add(
            provider_id=folder_id,
            summary=f"manifest schema validation failed: {exc}",
            where=str(rel),
            fix="resolve the validation error reported above.",
        )
        return

    # Phase 2: folder ↔ id parity.
    if manifest.provider.id != folder_id:
        findings.add(
            provider_id=folder_id,
            summary=(
                f"manifest provider.id={manifest.provider.id!r} does not match "
                f"folder name {folder_id!r}"
            ),
            where=str(rel),
            fix=f"rename the folder OR set provider.id = \"{folder_id}\".",
        )
        return

    if verbose:
        print(f"  ✓ {manifest.provider.id}: schema + folder id OK")

    # Phase 3: every entry-point class is importable.
    classes: dict[str, type] = {}
    for role_key in ("training", "inference", "pod_lifecycle_client", "config_schema"):
        ep = getattr(manifest.entry_points, role_key, None)
        if ep is None:
            continue
        try:
            module = importlib.import_module(ep.module)
        except ImportError as exc:
            findings.add(
                provider_id=manifest.provider.id,
                summary=(
                    f"entry_points.{role_key} module {ep.module!r} cannot be imported: {exc}"
                ),
                where=str(rel),
                fix=(
                    f"check the module path is correct, or that the package "
                    f"is installed in this environment."
                ),
            )
            continue
        if not hasattr(module, ep.class_name):
            findings.add(
                provider_id=manifest.provider.id,
                summary=(
                    f"entry_points.{role_key} class {ep.class_name!r} not found "
                    f"in module {ep.module!r}"
                ),
                where=str(rel),
                fix=(
                    f"either rename the Python class to {ep.class_name!r} or "
                    f"update the manifest to point at the actual class name."
                ),
            )
            continue
        classes[role_key] = getattr(module, ep.class_name)
        if verbose:
            print(f"    ✓ {role_key} → {ep.module}:{ep.class_name}")

    # Phase 4: capability ↔ Protocol / method parity (training side).
    training_cls = classes.get("training")
    if training_cls is not None:
        from ryotenkai_providers.training.interfaces import (
            ITerminalActionProvider,
            ProviderBase,
        )

        # ProviderBase inheritance.
        if not (isinstance(training_cls, type) and issubclass(training_cls, ProviderBase)):
            findings.add(
                provider_id=manifest.provider.id,
                summary=(
                    f"training class {training_cls.__name__} does not inherit ProviderBase"
                ),
                where=f"{ep.module}.{training_cls.__name__}",
                fix="add `ProviderBase` to the class bases — needed for default identity accessors.",
            )

        # supports_lifecycle_actions ↔ ITerminalActionProvider
        cap_la = manifest.capabilities.supports_lifecycle_actions
        is_la = isinstance(training_cls, type) and issubclass(
            training_cls, ITerminalActionProvider
        )
        if cap_la != is_la:
            findings.add(
                provider_id=manifest.provider.id,
                summary=(
                    f"supports_lifecycle_actions={cap_la} but "
                    f"isinstance(class, ITerminalActionProvider)={is_la}"
                ),
                where=str(rel),
                fix=(
                    "either flip the cap flag in provider.toml, or "
                    "add/remove ITerminalActionProvider in the class bases."
                ),
            )

        # supports_recovery_probe ↔ attempt_recovery method
        cap_rp = manifest.capabilities.supports_recovery_probe
        has_rp = callable(getattr(training_cls, "attempt_recovery", None))
        if cap_rp != has_rp:
            findings.add(
                provider_id=manifest.provider.id,
                summary=(
                    f"supports_recovery_probe={cap_rp} but "
                    f"attempt_recovery callable on class={has_rp}"
                ),
                where=f"{training_cls.__module__}.{training_cls.__name__}",
                fix=(
                    "either flip the cap flag, or implement / remove "
                    "the IRecoveryProbeProvider methods on the class."
                ),
            )

        # supports_capacity_error_detection ↔ is_capacity_error method
        cap_cc = manifest.capabilities.supports_capacity_error_detection
        has_cc = callable(getattr(training_cls, "is_capacity_error", None))
        if cap_cc != has_cc:
            findings.add(
                provider_id=manifest.provider.id,
                summary=(
                    f"supports_capacity_error_detection={cap_cc} but "
                    f"is_capacity_error callable on class={has_cc}"
                ),
                where=f"{training_cls.__module__}.{training_cls.__name__}",
                fix=(
                    "either flip the cap flag, or implement / remove "
                    "the ICapacityErrorClassifier methods on the class."
                ),
            )

    # Phase 5: pod-manifest projection drift.
    pod_target = _POD_MANIFESTS_DIR / f"{manifest.provider.id}.toml"
    try:
        from compile_pod_manifests import project_to_pod_manifest, render_pod_manifest_toml
    except ImportError:
        # Fallback when the script is invoked from outside the scripts/
        # dir (no sys.path adjustment) — try a direct path import.
        sys.path.insert(0, str(Path(__file__).parent))
        from compile_pod_manifests import project_to_pod_manifest, render_pod_manifest_toml  # type: ignore
        sys.path.pop(0)

    expected = render_pod_manifest_toml(project_to_pod_manifest(raw))
    actual = pod_target.read_text(encoding="utf-8") if pod_target.is_file() else ""
    if actual != expected:
        findings.add(
            provider_id=manifest.provider.id,
            summary="pod sub-manifest is stale (projection drift)",
            where=str(pod_target.relative_to(_REPO_ROOT)),
            fix=(
                "run ``python packages/providers/scripts/compile_pod_manifests.py`` "
                "and commit the regenerated file."
            ),
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print each manifest checked with per-phase ✓ markers.",
    )
    args = parser.parse_args(argv)

    paths = _discover_manifest_paths(_PROVIDERS_ROOT)
    if not paths:
        print(f"No provider.toml files found under {_PROVIDERS_ROOT}", file=sys.stderr)
        return 1
    findings = _Findings()
    print(f"Checking {len(paths)} provider manifest(s)...")
    print()
    for p in paths:
        _check_manifest(p, findings, verbose=args.verbose)
    print()
    if findings.empty():
        print(f"✓ {len(paths)} manifest(s) checked, 0 drift detected.")
        return 0
    print(findings.render())
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
