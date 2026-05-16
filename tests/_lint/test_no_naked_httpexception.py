"""Sentinel: no ``raise HTTPException(...)`` in production code.

Post-Phase G fix-up #2: every in-repo raise site MUST raise a typed
:class:`RyotenkAIError` subclass instead of a raw FastAPI
:class:`fastapi.HTTPException`. The unified ``http_exception_handler``
adapter is kept ONLY for FastAPI's own internal raises (router 405,
path-param coercion failures bypassing the validation handler,
third-party middleware that hasn't been migrated).

Rationale:

* Typed errors carry a machine-readable ``ErrorCode`` from the
  unified registry. Raw ``HTTPException`` only carries an integer
  status; the adapter has to fall back to ``ErrorCode.INTERNAL_ERROR``
  when the detail is a plain string. Clients (frontend, CLI) pin on
  ``code``, not on ``status`` (one status often maps to multiple
  semantics, e.g. 422 -> JOB_SPEC_INVALID vs PLUGIN_UNPACK_FAILED).
* Typed errors carry ``context: dict`` for structured per-occurrence
  metadata. Dict-detail ``HTTPException`` does the same thing in an
  ad-hoc way that's hard to schema-check.
* Typed errors plug into the same wire-rendering path as the pod
  runner (``ryotenkai_error_handler`` in ``shared/api/error_handlers``)
  so the boundary protocol stays uniform across services.

Allowlist: an entry is allowed only for the catch-handler in
``shared/api/error_handlers.http_exception_handler`` (it CATCHES
HTTPException; it does not raise). Add new entries only with a
code-owner review.
"""

from __future__ import annotations

import ast
from pathlib import Path

# Files allowed to mention ``HTTPException`` in a ``raise`` position
# at the AST level. Empty by design after the post-Phase G fix-up.
_ALLOWLIST: frozenset[str] = frozenset()


def _packages_root() -> Path:
    """Return ``packages/`` directory at the worktree root.

    Mirrors :mod:`tests/_lint/test_every_module_has_tests.py`'s
    resolution helper so the sentinel walks the same tree.
    """
    here = Path(__file__).resolve()
    # tests/_lint/test_no_naked_httpexception.py -> ../../ = repo root
    return here.parents[2] / "packages"


def _walk_production_py_files() -> list[Path]:
    """Walk every ``packages/*/src/**/*.py`` file in the repo."""
    root = _packages_root()
    if not root.exists():
        return []
    return sorted(p for p in root.glob("*/src/**/*.py") if "__pycache__" not in p.parts)


def _file_raises_httpexception(path: Path) -> list[int]:
    """Return line numbers of ``raise HTTPException(...)`` in ``path``.

    Uses AST walking so we never mis-flag string literals or comments
    that contain the substring. The pattern matched is a ``Raise``
    statement whose ``exc`` slot is a ``Call`` whose ``func`` is the
    bare ``Name`` ``HTTPException`` (the standard FastAPI import).
    Attribute access (e.g. ``fastapi.HTTPException(...)``) is also
    matched defensively.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return []

    offending: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Raise) or node.exc is None:
            continue
        exc = node.exc
        # Accept either ``raise HTTPException(...)`` or ``raise HTTPException``
        # (the latter is a bare-class raise — also flagged).
        if isinstance(exc, ast.Call):
            func = exc.func
        else:
            func = exc
        target_name: str | None = None
        if isinstance(func, ast.Name):
            target_name = func.id
        elif isinstance(func, ast.Attribute):
            target_name = func.attr
        if target_name == "HTTPException":
            offending.append(node.lineno)
    return offending


def test_no_raise_httpexception_in_production_code() -> None:
    """No ``raise HTTPException(...)`` in ``packages/*/src/``."""
    violations: list[str] = []
    for path in _walk_production_py_files():
        rel = path.relative_to(_packages_root().parent)
        rel_str = str(rel)
        if rel_str in _ALLOWLIST:
            continue
        lines = _file_raises_httpexception(path)
        for lineno in lines:
            violations.append(f"{rel_str}:{lineno}")
    assert not violations, (
        "Naked ``raise HTTPException(...)`` is forbidden. Raise a typed "
        ":class:`RyotenkAIError` subclass instead (see "
        "``packages/shared/src/ryotenkai_shared/errors/__init__.py`` "
        "for the catalog).\n\n"
        "Offenders:\n  " + "\n  ".join(violations)
    )


# ---------------------------------------------------------------------------
# Synthetic self-tests
# ---------------------------------------------------------------------------


def test_detector_flags_bare_name_call(tmp_path: Path) -> None:
    """``raise HTTPException(status_code=...)`` is detected."""
    file = tmp_path / "x.py"
    file.write_text(
        "from fastapi import HTTPException\n"
        "def f():\n"
        "    raise HTTPException(status_code=400, detail='bad')\n",
        encoding="utf-8",
    )
    assert _file_raises_httpexception(file) == [3]


def test_detector_flags_attribute_call(tmp_path: Path) -> None:
    """``raise fastapi.HTTPException(...)`` (attribute access) is detected."""
    file = tmp_path / "x.py"
    file.write_text(
        "import fastapi\n"
        "def f():\n"
        "    raise fastapi.HTTPException(status_code=404)\n",
        encoding="utf-8",
    )
    assert _file_raises_httpexception(file) == [3]


def test_detector_flags_bare_class_raise(tmp_path: Path) -> None:
    """``raise HTTPException`` (no call) is also detected."""
    file = tmp_path / "x.py"
    file.write_text(
        "from fastapi import HTTPException\n"
        "def f():\n"
        "    raise HTTPException\n",
        encoding="utf-8",
    )
    assert _file_raises_httpexception(file) == [3]


def test_detector_ignores_except_clause(tmp_path: Path) -> None:
    """``except HTTPException`` is not a raise -- not flagged."""
    file = tmp_path / "x.py"
    file.write_text(
        "from fastapi import HTTPException\n"
        "def f():\n"
        "    try:\n"
        "        pass\n"
        "    except HTTPException:\n"
        "        pass\n",
        encoding="utf-8",
    )
    assert _file_raises_httpexception(file) == []


def test_detector_ignores_string_literal(tmp_path: Path) -> None:
    """A string literal mentioning ``HTTPException`` is not a raise."""
    file = tmp_path / "x.py"
    file.write_text(
        "DOC = 'use HTTPException for FastAPI errors'\n"
        "def f():\n"
        "    return DOC\n",
        encoding="utf-8",
    )
    assert _file_raises_httpexception(file) == []


def test_detector_ignores_typed_raise(tmp_path: Path) -> None:
    """``raise JobNotFoundError(...)`` -- the canonical migration target."""
    file = tmp_path / "x.py"
    file.write_text(
        "from ryotenkai_shared.errors import JobNotFoundError\n"
        "def f():\n"
        "    raise JobNotFoundError(detail='gone')\n",
        encoding="utf-8",
    )
    assert _file_raises_httpexception(file) == []


def test_detector_handles_syntax_error(tmp_path: Path) -> None:
    """Files that can't be parsed return ``[]`` (don't crash the sentinel)."""
    file = tmp_path / "broken.py"
    file.write_text("def f(:\n    pass\n", encoding="utf-8")
    assert _file_raises_httpexception(file) == []


def test_packages_root_resolves() -> None:
    """The packages root resolves to an existing directory in the repo."""
    root = _packages_root()
    assert root.exists() and root.is_dir(), root
    # Sanity check: at least one src subtree exists.
    src_dirs = list(root.glob("*/src"))
    assert src_dirs, "expected packages/<pkg>/src directories"
