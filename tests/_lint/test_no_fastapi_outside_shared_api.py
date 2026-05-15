"""Sentinel: FastAPI may only be imported under ``shared/api/``.

Phase B (sharded-stargazing-wigderson, 2026-05-16) added
``fastapi>=0.115.0,<1.0.0`` to ``packages/shared/pyproject.toml`` so
``shared/api/error_handlers.py`` can return ``JSONResponse`` and
adapt ``HTTPException`` / ``RequestValidationError``. The dependency
addition risks leaking FastAPI types into modules that have no
business knowing about HTTP frameworks:

* ``shared/contracts/`` -- wire models, used on both ends; must
  remain framework-agnostic (Pydantic only).
* ``shared/utils/`` -- pure helpers; no framework coupling.
* ``shared/infrastructure/`` -- Protocols for MLflow/PodLifecycle;
  no framework coupling.
* ``shared/config/``, ``shared/constants/``, ``shared/errors/``,
  ``shared/inference/``, ``shared/observability/``,
  ``shared/pipeline_context/`` -- same rule.

The only allowed location is
``packages/shared/src/ryotenkai_shared/api/`` (this is the FastAPI-
facing subpackage holding handlers + middleware). Adding ``api/``
files that import FastAPI is fine; adding such imports anywhere else
in ``shared`` fails this sentinel and forces the author to either
move the code into ``shared/api/`` or rethink the import.

The sentinel walks every ``packages/shared/src/**/*.py`` and rejects
both ``from fastapi[.foo] import bar`` and ``import fastapi[.foo]``.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SHARED_SRC = REPO_ROOT / "packages" / "shared" / "src" / "ryotenkai_shared"
# The single allowed subdirectory under shared (relative to SHARED_SRC).
ALLOWED_SUBDIR = "api"


def _is_fastapi_import_from(node: ast.AST) -> bool:
    """Return True for ``from fastapi[.subpkg] import X``."""
    if not isinstance(node, ast.ImportFrom):
        return False
    if node.module is None:
        return False
    return node.module == "fastapi" or node.module.startswith("fastapi.")


def _is_fastapi_import(node: ast.AST) -> bool:
    """Return True for ``import fastapi[.subpkg][ as X]``."""
    if not isinstance(node, ast.Import):
        return False
    for alias in node.names:
        if alias.name == "fastapi" or alias.name.startswith("fastapi."):
            return True
    return False


def _scan_file(path: Path) -> list[str]:
    """Return human-readable violation lines for ``path``."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, OSError):
        return []
    violations: list[str] = []
    for node in ast.walk(tree):
        if _is_fastapi_import_from(node):
            assert isinstance(node, ast.ImportFrom)  # for the type-checker
            violations.append(
                f"{path}:{node.lineno}: forbidden `from {node.module} import ...` "
                "outside `shared/api/` -- move to shared/api/ or drop the import"
            )
        elif _is_fastapi_import(node):
            assert isinstance(node, ast.Import)
            for alias in node.names:
                if alias.name == "fastapi" or alias.name.startswith("fastapi."):
                    violations.append(
                        f"{path}:{node.lineno}: forbidden `import {alias.name}` "
                        "outside `shared/api/` -- move to shared/api/ or drop"
                    )
    return violations


def _iter_target_files() -> list[Path]:
    """Walk ``packages/shared/src/`` for ``*.py`` files outside the
    allowed ``api/`` subdirectory. ``__pycache__`` is skipped.
    """
    if not SHARED_SRC.exists():
        return []
    files: list[Path] = []
    for py in SHARED_SRC.rglob("*.py"):
        if "__pycache__" in py.parts:
            continue
        # Allow anything under shared/src/ryotenkai_shared/api/.
        try:
            rel = py.relative_to(SHARED_SRC)
        except ValueError:
            continue
        if rel.parts and rel.parts[0] == ALLOWED_SUBDIR:
            continue
        files.append(py)
    return files


def _scan_all() -> list[str]:
    violations: list[str] = []
    for py in _iter_target_files():
        violations.extend(_scan_file(py))
    return violations


def test_no_fastapi_outside_shared_api() -> None:
    """No file under ``packages/shared/src/ryotenkai_shared/`` may
    import ``fastapi`` (or any submodule) except those under
    ``ryotenkai_shared/api/``.
    """
    violations = _scan_all()
    assert not violations, (
        "FastAPI is only allowed under `shared/src/ryotenkai_shared/api/`. "
        "Other shared modules must remain framework-agnostic so they can be "
        "consumed unchanged on the Mac side (which doesn't need an HTTP "
        "server). Move the code into `shared/api/` or rethink the import.\n\n"
        "Offenders:\n  " + "\n  ".join(violations)
    )


def test_sentinel_catches_synthetic_import_from(tmp_path: Path) -> None:
    """Synthetic ``from fastapi import X`` is flagged."""
    bad = tmp_path / "bad_from.py"
    bad.write_text("from fastapi import HTTPException\n", encoding="utf-8")
    findings = _scan_file(bad)
    assert findings, "Sentinel must catch `from fastapi import ...`"
    assert "fastapi" in findings[0]


def test_sentinel_catches_synthetic_submodule_import_from(tmp_path: Path) -> None:
    """Synthetic ``from fastapi.responses import JSONResponse`` is flagged."""
    bad = tmp_path / "bad_sub.py"
    bad.write_text(
        "from fastapi.responses import JSONResponse\n", encoding="utf-8",
    )
    findings = _scan_file(bad)
    assert findings, "Sentinel must catch `from fastapi.SUBMOD import ...`"


def test_sentinel_catches_synthetic_import(tmp_path: Path) -> None:
    """Synthetic ``import fastapi`` is flagged."""
    bad = tmp_path / "bad_import.py"
    bad.write_text("import fastapi\n", encoding="utf-8")
    findings = _scan_file(bad)
    assert findings, "Sentinel must catch `import fastapi`"


def test_sentinel_ignores_unrelated_imports(tmp_path: Path) -> None:
    """Imports that merely *look* like FastAPI must NOT be flagged."""
    ok = tmp_path / "ok.py"
    ok.write_text(
        "from pydantic import BaseModel\n"
        "from ryotenkai_shared.contracts.problem_details import ErrorCode\n"
        "import json\n",
        encoding="utf-8",
    )
    findings = _scan_file(ok)
    assert not findings, findings


def test_allowed_subdirectory_files_skipped() -> None:
    """Files under ``shared/api/`` are not scanned; they MAY import
    FastAPI. The sentinel must not include them in its target set."""
    targets = _iter_target_files()
    for path in targets:
        rel = path.relative_to(SHARED_SRC)
        assert rel.parts[0] != ALLOWED_SUBDIR, (
            f"sentinel must skip files under `{ALLOWED_SUBDIR}/`, got: {rel}"
        )
