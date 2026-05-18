"""Unit tests for ``scripts/lint/mlflow_rules.py`` (Phase M7).

Covers each of the four rules with synthetic Python source written to
``tmp_path`` plus the allowlist semantics. Uses ``importlib`` to load
the script as a module so the AST visitor + walker logic can be
exercised directly without subprocessing.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Load the script as a module (it lives outside ``packages/`` so it has no
# package entry; importlib gives us direct access).
# ---------------------------------------------------------------------------

_SCRIPT_PATH = Path(__file__).resolve().parents[4] / "scripts" / "lint" / "mlflow_rules.py"
_spec = importlib.util.spec_from_file_location("mlflow_rules", _SCRIPT_PATH)
assert _spec is not None
_mlflow_rules = importlib.util.module_from_spec(_spec)
sys.modules["mlflow_rules"] = _mlflow_rules
assert _spec.loader is not None
_spec.loader.exec_module(_mlflow_rules)


@pytest.fixture()
def packages_root(tmp_path: Path) -> Path:
    """Create a fake ``packages/`` tree under tmp_path."""
    (tmp_path / "packages" / "shared" / "src" / "ryotenkai_shared" / "infrastructure" / "mlflow").mkdir(
        parents=True,
    )
    (tmp_path / "packages" / "control" / "src" / "ryotenkai_control" / "pipeline" / "mlflow" / "read").mkdir(
        parents=True,
    )
    (tmp_path / "packages" / "pod" / "src" / "ryotenkai_pod" / "trainer" / "subdir").mkdir(
        parents=True,
    )
    return tmp_path


def _scan(packages_root: Path, paths: list[str]) -> list:
    # Use the production walker so _SKIP_DIRS semantics are exercised.
    targets = [packages_root / p for p in paths]
    files = _mlflow_rules._iter_python_files(packages_root, targets)
    violations = []
    for f in files:
        violations.extend(_mlflow_rules._scan_file(f, packages_root))
    return violations


class TestNoAutolog:
    def test_flags_mlflow_autolog(self, packages_root: Path) -> None:
        target = packages_root / "packages" / "shared" / "src" / "snippet.py"
        target.write_text("import mlflow\nmlflow.autolog()\n")
        violations = _scan(packages_root, ["packages"])
        assert any(v.rule == "NO_AUTOLOG" for v in violations)

    def test_flags_mlflow_transformers_autolog(self, packages_root: Path) -> None:
        target = packages_root / "packages" / "shared" / "src" / "snippet.py"
        target.write_text("import mlflow\nmlflow.transformers.autolog(log_models=False)\n")
        violations = _scan(packages_root, ["packages"])
        assert any(v.rule == "NO_AUTOLOG" for v in violations)

    def test_ignores_other_autolog_attributes(self, packages_root: Path) -> None:
        target = packages_root / "packages" / "shared" / "src" / "snippet.py"
        target.write_text("class X:\n    def autolog(self): pass\nX().autolog()\n")
        violations = _scan(packages_root, ["packages"])
        assert not any(v.rule == "NO_AUTOLOG" for v in violations)


class TestNoSetTrackingUri:
    def test_flags_set_tracking_uri_outside_allowlist(self, packages_root: Path) -> None:
        target = (
            packages_root / "packages" / "control" / "src" / "ryotenkai_control" / "snippet.py"
        )
        target.write_text("import mlflow\nmlflow.set_tracking_uri('http://x')\n")
        violations = _scan(packages_root, ["packages"])
        assert any(v.rule == "NO_SET_TRACKING_URI_GLOBAL" for v in violations)

    def test_allowlisted_transport_module_passes(self, packages_root: Path) -> None:
        target = (
            packages_root
            / "packages"
            / "shared"
            / "src"
            / "ryotenkai_shared"
            / "infrastructure"
            / "mlflow"
            / "transport.py"
        )
        target.write_text("import mlflow\nmlflow.set_tracking_uri('http://x')\n")
        violations = _scan(packages_root, ["packages"])
        assert not any(v.rule == "NO_SET_TRACKING_URI_GLOBAL" for v in violations)


class TestNoAdHocMlflowClient:
    def test_flags_mlflow_tracking_client(self, packages_root: Path) -> None:
        target = (
            packages_root / "packages" / "control" / "src" / "ryotenkai_control" / "snippet.py"
        )
        target.write_text("from mlflow.tracking import MlflowClient\nMlflowClient()\n")
        violations = _scan(packages_root, ["packages"])
        assert any(v.rule == "NO_AD_HOC_MLFLOW_CLIENT" for v in violations)

    def test_flags_mlflow_dot_mlflow_client(self, packages_root: Path) -> None:
        target = (
            packages_root / "packages" / "control" / "src" / "ryotenkai_control" / "snippet.py"
        )
        target.write_text("import mlflow\nmlflow.MlflowClient()\n")
        violations = _scan(packages_root, ["packages"])
        assert any(v.rule == "NO_AD_HOC_MLFLOW_CLIENT" for v in violations)

    def test_allowlisted_read_client_passes(self, packages_root: Path) -> None:
        target = (
            packages_root
            / "packages"
            / "control"
            / "src"
            / "ryotenkai_control"
            / "pipeline"
            / "mlflow"
            / "read"
            / "client.py"
        )
        target.write_text("from mlflow.tracking import MlflowClient\nMlflowClient(tracking_uri='x')\n")
        violations = _scan(packages_root, ["packages"])
        assert not any(v.rule == "NO_AD_HOC_MLFLOW_CLIENT" for v in violations)


class TestNoStartRunInTrainer:
    def test_flags_start_run_in_trainer(self, packages_root: Path) -> None:
        target = (
            packages_root
            / "packages"
            / "pod"
            / "src"
            / "ryotenkai_pod"
            / "trainer"
            / "subdir"
            / "snippet.py"
        )
        target.write_text("import mlflow\nmlflow.start_run()\n")
        violations = _scan(packages_root, ["packages"])
        assert any(v.rule == "NO_START_RUN_IN_TRAINER" for v in violations)

    def test_ignores_start_run_outside_trainer(self, packages_root: Path) -> None:
        target = (
            packages_root / "packages" / "control" / "src" / "ryotenkai_control" / "snippet.py"
        )
        target.write_text("import mlflow\nmlflow.start_run()\n")
        violations = _scan(packages_root, ["packages"])
        assert not any(v.rule == "NO_START_RUN_IN_TRAINER" for v in violations)


class TestAttributeChainHelper:
    def test_chain_match_positive(self) -> None:
        import ast as _ast

        tree = _ast.parse("a.b.c", mode="eval")
        assert _mlflow_rules._is_attr_chain(tree.body, ("a", "b", "c")) is True

    def test_chain_match_negative(self) -> None:
        import ast as _ast

        tree = _ast.parse("a.b", mode="eval")
        assert _mlflow_rules._is_attr_chain(tree.body, ("a", "b", "c")) is False

    def test_chain_with_non_attribute_root(self) -> None:
        import ast as _ast

        tree = _ast.parse("foo()", mode="eval")
        assert _mlflow_rules._is_attr_chain(tree.body, ("foo",)) is False


class TestSkipDirs:
    def test_skips_pycache(self, packages_root: Path) -> None:
        pycache = packages_root / "packages" / "shared" / "src" / "__pycache__"
        pycache.mkdir()
        bad = pycache / "snippet.py"
        bad.write_text("import mlflow\nmlflow.autolog()\n")
        violations = _scan(packages_root, ["packages"])
        assert not violations
