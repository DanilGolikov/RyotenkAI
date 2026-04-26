"""
Architectural guardrails — CI-enforced structural constraints.

These tests encode decisions made during the Phase 4 refactor so that
future changes that accidentally violate the architecture fail fast in CI
rather than silently accumulating debt.

Guardrails covered:
1. File size limits — hotspot files must not grow beyond agreed thresholds
2. Import cycle detection — key module pairs must not form circular imports
3. Co-change contracts — files that historically drift together must expose
   compatible interfaces (checked via import + basic attribute presence)
4. Extraction contracts — extracted modules must remain importable and
   expose their public API
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parents[3]  # repo root (src/tests/unit/test_*.py → 3 levels up)
SRC = ROOT / "src"


def _line_count(rel_path: str) -> int:
    """Return the number of lines in a source file relative to repo root."""
    return len((ROOT / rel_path).read_text(encoding="utf-8").splitlines())


# ---------------------------------------------------------------------------
# 1. FILE SIZE LIMITS
# ---------------------------------------------------------------------------

class TestFileSizeLimits:
    """Hotspot files that have historically accumulated must not grow further.

    Thresholds are set at ~110 % of the current line count so that small
    additions don't trip the guard, but runaway growth is caught early.
    """

    @pytest.mark.parametrize("rel_path,max_lines", [
        # orchestrator.py: 2062 → 1385 after Phase-3 decomposition; target ≤200 LOC
        # after Phase-A architectural refactor. Ratchet down after each PR-AN.
        ("src/pipeline/orchestrator.py", 1_400),
        # MLflowManager facade — keep from growing; currently ~628 lines.
        ("src/training/managers/mlflow_manager/manager.py", 720),
        # setup.py mixin — currently ~285 lines; cap at 320.
        ("src/training/managers/mlflow_manager/setup.py", 320),
        # training_monitor.py — 979 lines currently; cap at 1000 pending future extraction.
        ("src/pipeline/stages/training_monitor.py", 1_000),
    ])
    def test_file_not_exceeding_line_limit(self, rel_path: str, max_lines: int) -> None:
        actual = _line_count(rel_path)
        assert actual <= max_lines, (
            f"{rel_path} has {actual} lines, exceeding the guardrail of {max_lines}. "
            "Either refactor the file or raise the limit with a documented rationale."
        )


# ---------------------------------------------------------------------------
# 2. IMPORT CYCLE DETECTION
# ---------------------------------------------------------------------------

class TestNoCycles:
    """Critical module pairs must not form import cycles.

    We verify this by importing each module in isolation: if a circular
    import exists Python raises ImportError during the import.
    """

    @pytest.mark.parametrize("module_path", [
        "src.pipeline.stages.dataset_validator.artifact_manager",
        "src.pipeline.state.transitioner",
        "src.training.managers.mlflow_manager.manager",
        "src.training.mlflow.resilient_transport",
        "src.training.mlflow.metrics_buffer",
    ])
    def test_module_imports_cleanly(self, module_path: str) -> None:
        """Module must be importable without circular-import errors."""
        # Use a subprocess so we get a truly fresh interpreter state.
        result = subprocess.run(
            [sys.executable, "-c", f"import {module_path}"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, (
            f"Importing {module_path} failed — possible circular import:\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

    def test_validation_artifact_manager_does_not_import_orchestrator(self) -> None:
        """ValidationArtifactManager must not import the orchestrator (would re-create the cycle)."""
        import_lines = [
            line
            for line in (SRC / "pipeline/stages/dataset_validator/artifact_manager.py")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.startswith(("import ", "from "))
        ]
        assert not any("orchestrator" in line for line in import_lines), (
            "artifact_manager.py imports from orchestrator — this creates a cycle. "
            "Move shared types to a separate module instead."
        )

    def test_state_transitioner_does_not_import_orchestrator(self) -> None:
        """StateTransitioner functions must not import the orchestrator."""
        import_lines = [
            line
            for line in (SRC / "pipeline/state/transitioner.py")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.startswith(("import ", "from "))
        ]
        assert not any("orchestrator" in line for line in import_lines)


# ---------------------------------------------------------------------------
# 3. CO-CHANGE CONTRACT: MLflowDatasetLogger constructor shape
# ---------------------------------------------------------------------------

class TestMLflowDatasetLoggerContract:
    """MLflowDatasetLogger must remain constructable via _make_dataset_logger().

    The factory method in MLflowManager.__init__ is the single source of truth
    for construction.  We verify the constructor still accepts the expected
    keyword arguments so that refactors to MLflowDatasetLogger don't silently
    break MLflowManager.
    """

    def test_dataset_logger_constructor_accepts_expected_kwargs(self) -> None:
        import inspect
        from src.training.mlflow.dataset_logger import MLflowDatasetLogger

        sig = inspect.signature(MLflowDatasetLogger.__init__)
        params = set(sig.parameters.keys()) - {"self"}
        required = {"mlflow_module", "primitives", "has_active_run"}
        missing = required - params
        assert not missing, (
            f"MLflowDatasetLogger.__init__ is missing expected parameters: {missing}. "
            "Update _make_dataset_logger() in manager.py to match."
        )

    def test_make_dataset_logger_is_the_only_constructor_site(self) -> None:
        """MLflowDatasetLogger(...) must only be constructed via _make_dataset_logger."""
        mgr_dir = SRC / "training/managers/mlflow_manager"
        for py_file in mgr_dir.glob("*.py"):
            if py_file.name == "manager.py":
                continue  # _make_dataset_logger itself lives here
            src = py_file.read_text(encoding="utf-8")
            assert "MLflowDatasetLogger(" not in src, (
                f"{py_file.name} constructs MLflowDatasetLogger directly. "
                "Use self._make_dataset_logger() instead to keep construction centralised."
            )


# ---------------------------------------------------------------------------
# 4. EXTRACTION CONTRACTS — public APIs must remain stable
# ---------------------------------------------------------------------------

class TestExtractionContracts:
    """Extracted modules must expose their documented public API."""

    def test_validation_artifact_manager_public_api(self) -> None:
        from src.pipeline.stages.dataset_validator.artifact_manager import ValidationArtifactManager

        expected_methods = [
            "on_dataset_scheduled",
            "on_dataset_loaded",
            "on_validation_completed",
            "on_validation_failed",
            "on_plugin_start",
            "on_plugin_complete",
            "on_plugin_failed",
            "flush_validation_artifact",
            "build_dataset_validation_state_outputs",
        ]
        for method in expected_methods:
            assert hasattr(ValidationArtifactManager, method), (
                f"ValidationArtifactManager is missing method '{method}'. "
                "Update the orchestrator delegation and this contract test."
            )

    def test_state_transitioner_public_api(self) -> None:
        from src.pipeline.state import transitioner

        expected_functions = [
            "mark_stage_running",
            "mark_stage_completed",
            "mark_stage_failed",
            "mark_stage_skipped",
            "mark_stage_interrupted",
            "finalize_attempt_state",
        ]
        for fn_name in expected_functions:
            assert hasattr(transitioner, fn_name), (
                f"transitioner module is missing function '{fn_name}'. "
                "Update the orchestrator static-method delegates and this contract test."
            )

    def test_pipeline_validation_package_importable(self) -> None:
        mod = importlib.import_module("src.pipeline.stages.dataset_validator")
        assert hasattr(mod, "ValidationArtifactManager")

    def test_resilient_transport_public_api(self) -> None:
        from src.training.mlflow.resilient_transport import ResilientMLflowTransport

        expected = ["install", "uninstall", "breaker_state"]
        for attr in expected:
            assert hasattr(ResilientMLflowTransport, attr), (
                f"ResilientMLflowTransport missing '{attr}'"
            )


# ---------------------------------------------------------------------------
# 5. NO-VAMPIRE-REF — collaborators never import PipelineOrchestrator
# ---------------------------------------------------------------------------

class TestNoOrchestratorImports:
    """Collaborators must not import PipelineOrchestrator.

    Core invariant of the Phase-A architecture: state and behaviour flows
    DOWN the dependency tree (orchestrator → collaborators → utils), never
    back up. An ``import PipelineOrchestrator`` outside the allow-list
    indicates a vampire reference that will break testability.
    """

    _ALLOWED_IMPORTERS = frozenset({
        "src/pipeline/orchestrator.py",         # self
        "src/main.py",                          # CLI entry point
        "src/api/services/run_service.py",      # web backend entry
    })

    def test_no_collaborator_imports_pipeline_orchestrator(self) -> None:
        import re

        forbidden = (
            re.compile(r"^from\s+src\.pipeline\.orchestrator\s+import\s+PipelineOrchestrator", re.M),
            re.compile(r"^import\s+src\.pipeline\.orchestrator\b", re.M),
        )
        pipeline_dir = SRC / "pipeline"
        violations: list[str] = []
        for py_file in pipeline_dir.rglob("*.py"):
            if "__pycache__" in py_file.parts:
                continue
            rel = py_file.relative_to(ROOT).as_posix()
            if rel in self._ALLOWED_IMPORTERS:
                continue
            content = py_file.read_text(encoding="utf-8")
            for pattern in forbidden:
                match = pattern.search(content)
                if match:
                    violations.append(f"{rel}: {match.group(0)}")
        assert not violations, (
            "Collaborators must not import PipelineOrchestrator. Offenders:\n  "
            + "\n  ".join(violations)
        )


# ---------------------------------------------------------------------------
# 6. NO-PRIVATE-PROBE — tests must not poke migrated private attrs
# ---------------------------------------------------------------------------

class TestNoMigratedPrivateProbes:
    """Each PR-AN that extracts a private into a component adds its private
    attr to this list. Future tests that reach for it fail CI — tests must
    exercise the new public component API instead.
    """

    # Pattern → reason. Patterns are regex; they match a WRITE/ASSIGN form
    # (e.g. ``orch._run_lock =`` or ``orchestrator._run_lock =``), not reads.
    _BANNED_PATTERNS = (
        # PR-A1: run_lock lifecycle owned by RunLockGuard
        (r"(?:orch|orchestrator)\._run_lock\s*=",
         "PR-A1: use _install_fake_lock_guard(orch, tmp_path) helper — RunLockGuard owns the lock now"),
    )

    @pytest.mark.parametrize("pattern, reason", _BANNED_PATTERNS)
    def test_no_test_writes_to_migrated_private(self, pattern: str, reason: str) -> None:
        import re

        regex = re.compile(pattern)
        tests_dir = SRC / "tests"
        violations: list[str] = []
        for py_file in tests_dir.rglob("*.py"):
            if "__pycache__" in py_file.parts:
                continue
            # Skip this guardrail file itself
            if py_file.resolve() == Path(__file__).resolve():
                continue
            for ln_no, line in enumerate(
                py_file.read_text(encoding="utf-8").splitlines(), start=1
            ):
                if regex.search(line):
                    rel = py_file.relative_to(ROOT).as_posix()
                    violations.append(f"{rel}:{ln_no}: {line.strip()}")
        assert not violations, f"{reason}\nOffenders:\n  " + "\n  ".join(violations)
