"""Sentinel: no test files may live under ``packages/*/tests/``.

The legacy per-package test layout was migrated into the centralised
``tests/`` root during Batches 7a-7c. After the cleanup commit that
follows this sentinel, the ``packages/*/tests/`` directories are
removed entirely. This guard prevents them from reappearing.

Why the guard matters:

* pytest's discovery walks the whole repo (``testpaths = tests`` in
  ``tests/pytest.ini``). A stray test file under ``packages/`` would
  either be invisible (bad — silently uncovered) or, if a future
  contributor pointed pytest at the package, it would resurrect the
  duplicate-fixture problem we just solved.
* New contributors / agents may not know the convention. The sentinel
  fails loudly with the right pointer.

Recovery path when the sentinel fires:

* Move the new test into ``tests/<layer>/<package>/`` mirroring the
  production module path.
* Use canonical fakes from ``tests/_fakes/`` and factories from
  ``tests/_factories/`` instead of rolling inline stubs.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGES_ROOT = REPO_ROOT / "packages"


def test_no_packages_tests_dirs() -> None:
    """``packages/*/tests/`` MUST NOT exist."""
    offenders: list[str] = []
    if PACKAGES_ROOT.exists():
        for pkg_dir in PACKAGES_ROOT.iterdir():
            if not pkg_dir.is_dir():
                continue
            tests_dir = pkg_dir / "tests"
            if tests_dir.exists():
                offenders.append(str(tests_dir.relative_to(REPO_ROOT)))
    if offenders:
        pytest.fail(
            "Legacy ``packages/*/tests/`` directory found — all tests "
            "live under the central ``tests/`` tree post-Batch-7c.\n"
            "Move test files into ``tests/<unit|integration|e2e>/"
            "<package>/...`` and use canonical fakes from "
            "``tests/_fakes/``.\n\nOffending paths:\n  - "
            + "\n  - ".join(offenders)
        )


def test_no_test_files_under_packages() -> None:
    """Defence-in-depth: no ``test_*.py`` files anywhere under packages.

    The previous test catches the *directory* form; this one catches a
    stray file dropped into an arbitrary subdirectory (e.g.
    ``packages/control/src/.../test_inline.py``).
    """
    if not PACKAGES_ROOT.exists():
        return
    offenders: list[str] = []
    for p in PACKAGES_ROOT.rglob("test_*.py"):
        # Test fixtures or config files named "test_*.yaml" don't apply
        # — rglob with "test_*.py" already filters to Python.
        offenders.append(str(p.relative_to(REPO_ROOT)))
    if offenders:
        sample = "\n  - ".join(offenders[:20])
        more = "" if len(offenders) <= 20 else f"\n  …and {len(offenders) - 20} more"
        pytest.fail(
            f"{len(offenders)} ``test_*.py`` file(s) found under "
            f"``packages/`` — all tests belong in the central "
            f"``tests/`` tree.\n\n  - {sample}{more}"
        )
