"""Unit tests for debug-bundle plugin."""

from __future__ import annotations

import tarfile
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest

pytest_plugins = ["pytester"]


def test_debug_bundle_created_for_l5_failure(pytester: pytest.Pytester) -> None:
    repo_root = str(__import__("pathlib").Path(__file__).resolve().parents[2])
    pytester.makepyfile(
        conftest=f"""
import sys
sys.path.insert(0, {repo_root!r})
from pathlib import Path
from tests._harness import debug_bundle


def pytest_configure(config):
    out = Path(config.rootpath) / "tests" / ".debug_bundles"
    debug_bundle.register(config, output_dir=out)
"""
    )
    e2e_dir = pytester.path / "tests" / "e2e"
    e2e_dir.mkdir(parents=True)
    (pytester.path / "tests" / "__init__.py").write_text("")
    (pytester.path / "tests" / "e2e" / "__init__.py").write_text("")
    (e2e_dir / "test_l5_fail.py").write_text(
        "def test_boom():\n    assert False\n",
        encoding="utf-8",
    )

    result = pytester.runpytest("-p", "no:cacheprovider", "--import-mode=importlib", "tests/e2e/")
    assert result.ret != 0, result.outlines

    bundles = list((pytester.path / "tests" / ".debug_bundles").glob("*.tar.gz"))
    assert len(bundles) == 1, bundles
    with tarfile.open(bundles[0], "r:gz") as tar:
        names = set(tar.getnames())
    assert "report.txt" in names
    assert "logs/captured.txt" in names
    assert "fake_state.json" in names
    assert "journal.txt" in names


def test_debug_bundle_skipped_for_unit_failure(pytester: pytest.Pytester) -> None:
    repo_root = str(__import__("pathlib").Path(__file__).resolve().parents[2])
    pytester.makepyfile(
        conftest=f"""
import sys
sys.path.insert(0, {repo_root!r})
from pathlib import Path
from tests._harness import debug_bundle


def pytest_configure(config):
    out = Path(config.rootpath) / "tests" / ".debug_bundles"
    debug_bundle.register(config, output_dir=out)
"""
    )
    unit_dir = pytester.path / "tests" / "unit"
    unit_dir.mkdir(parents=True)
    (pytester.path / "tests" / "__init__.py").write_text("")
    (pytester.path / "tests" / "unit" / "__init__.py").write_text("")
    (unit_dir / "test_l1_fail.py").write_text(
        "def test_boom():\n    assert False\n",
        encoding="utf-8",
    )

    result = pytester.runpytest("-p", "no:cacheprovider", "--import-mode=importlib", "tests/unit/")
    assert result.ret != 0, result.outlines
    bundles_dir = pytester.path / "tests" / ".debug_bundles"
    assert not bundles_dir.exists() or not list(bundles_dir.glob("*.tar.gz"))
