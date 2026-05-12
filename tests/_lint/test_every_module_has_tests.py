"""Sentinel: every production module must be referenced by at least one test file.

Catches the "agent added a new module without tests" failure mode. The
contract:

- For every ``packages/*/src/<pkg>/**/*.py`` (excluding ``__init__.py``,
  ``py.typed``, and other allowlisted paths), there must exist at least
  one ``tests/**/*.py`` file that imports or references the module.

- "References" is liberal: any line in any test file that contains the
  import-style dotted name of the module (e.g. ``ryotenkai_control.foo.bar``)
  satisfies the gate. Direct ``from ... import ...``, indirect imports
  via parent module, and even string mentions in monkeypatch / patch
  targets all count.

- Allowlist (``tests/_lint/no_test_required.yaml``) excuses modules that
  legitimately don't need a dedicated test (pure constants, CLI
  entrypoints exercised by e2e, etc.). Every allowlist entry has a
  ``reason`` that the sentinel does NOT verify but humans should.

Why a sentinel:
- An agent can land a new production module and forget to add tests;
  line-coverage alone won't catch this (coverage is computed only over
  imported modules).
- Mutation testing on hotspots (separate gate) catches *weak* tests, but
  not *absent* tests on non-hotspots. This is the absent-test gate.
"""

from __future__ import annotations

import fnmatch
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGES_ROOT = REPO_ROOT / "packages"
TESTS_ROOT = REPO_ROOT / "tests"
ALLOWLIST_PATH = Path(__file__).parent / "no_test_required.yaml"

SKIP_NAMES = {"__init__.py", "py.typed"}
# Reserved files that are NOT user source code (markers, compiled, etc.)
SKIP_SUFFIXES = (".pyi",)


def _load_allowlist() -> tuple[set[str], list[str]]:
    if not ALLOWLIST_PATH.exists():
        return set(), []
    data = yaml.safe_load(ALLOWLIST_PATH.read_text(encoding="utf-8"))
    explicit = {entry["path"] for entry in data.get("allow", [])}
    patterns = [entry["pattern"] for entry in data.get("allow_patterns", [])]
    return explicit, patterns


def _production_modules() -> list[Path]:
    """Yield every `.py` under `packages/*/src/` (skipping aggregators)."""
    out: list[Path] = []
    if not PACKAGES_ROOT.exists():
        return out
    for src_root in PACKAGES_ROOT.glob("*/src"):
        for py in src_root.rglob("*.py"):
            if py.name in SKIP_NAMES:
                continue
            if py.suffix in SKIP_SUFFIXES:
                continue
            out.append(py)
    return out


def _import_name(path: Path) -> str:
    """Convert ``packages/control/src/ryotenkai_control/foo/bar.py`` →
    ``ryotenkai_control.foo.bar``.
    """
    rel = path.relative_to(PACKAGES_ROOT)
    # rel = <pkg>/src/ryotenkai_<pkg>/foo/bar.py
    parts = rel.parts
    if len(parts) < 4 or parts[1] != "src":
        return ""
    # Drop <pkg>/src; remove .py suffix
    dotted_parts = list(parts[2:])
    dotted_parts[-1] = dotted_parts[-1][:-3]  # strip .py
    return ".".join(dotted_parts)


def _test_files() -> list[Path]:
    if not TESTS_ROOT.exists():
        return []
    return [p for p in TESTS_ROOT.rglob("*.py") if "/_lint/" not in str(p)]


def _all_test_text() -> str:
    """One big string of every test file's content — fast substring search."""
    chunks: list[str] = []
    for p in _test_files():
        try:
            chunks.append(p.read_text(encoding="utf-8"))
        except OSError:
            continue
    return "\n".join(chunks)


def _matches_allowlist(rel_path: str, explicit: set[str], patterns: list[str]) -> bool:
    if rel_path in explicit:
        return True
    for pat in patterns:
        if fnmatch.fnmatch(rel_path, pat):
            return True
    return False


def _test_file_basenames() -> set[str]:
    """Set of every test file's basename without prefix/suffix.

    ``tests/unit/foo/test_bar.py`` → ``bar``. Used for the loose match
    that catches dedicated ``test_<module>.py`` files even when the
    import is via a parent module.
    """
    out: set[str] = set()
    for p in _test_files():
        name = p.stem  # "test_bar"
        if name.startswith("test_"):
            out.add(name[5:])
    return out


def _is_referenced(dotted: str, basename: str, test_blob: str, test_stems: set[str]) -> bool:
    """A module is "referenced" if any of these hold:

    1. The full dotted path appears in any test file (strict match).
    2. A non-trivial dotted-name SUFFIX (e.g. ``trainer.strategies.grpo``
       → also tries ``strategies.grpo`` and ``trainer.strategies``)
       appears in any test file — covers indirect imports via parent
       modules.
    3. There is a test file named ``test_<basename>.py``.
    """
    if dotted in test_blob:
        return True
    # Try longer-than-3-segment suffixes
    parts = dotted.split(".")
    for start in range(1, max(1, len(parts) - 1)):
        suffix = ".".join(parts[start:])
        if len(suffix) > 6 and suffix in test_blob:
            return True
    return basename in test_stems


def test_every_production_module_has_at_least_one_test_reference() -> None:
    """A new ``.py`` under ``packages/*/src/`` must be referenced by ≥ 1 test."""
    explicit, patterns = _load_allowlist()
    test_blob = _all_test_text()
    test_stems = _test_file_basenames()

    untested: list[str] = []
    for mod in _production_modules():
        rel = str(mod.relative_to(REPO_ROOT))
        if _matches_allowlist(rel, explicit, patterns):
            continue
        dotted = _import_name(mod)
        if not dotted:
            continue
        basename = mod.stem
        if _is_referenced(dotted, basename, test_blob, test_stems):
            continue
        untested.append(f"{rel}  (importable as `{dotted}`)")

    if untested:
        sample = "\n  ".join(untested[:30])
        more = "" if len(untested) <= 30 else f"\n  …and {len(untested) - 30} more"
        pytest.fail(
            f"{len(untested)} production module(s) have no test references.\n"
            f"Add at least one ``tests/**/*.py`` that imports them, OR add an "
            f"entry to ``tests/_lint/no_test_required.yaml`` with a reason.\n\n"
            f"  {sample}{more}"
        )
