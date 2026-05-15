"""Sentinel: ``ryotenkai_shared.utils.result`` must never be imported again.

Phase A2 finale (sharded-stargazing-wigderson, 2026-05-16) deleted the
legacy ``Result[T, AppError]`` module and its dataclass hierarchy in
favour of the unified :class:`RyotenkAIError` raise-based system in
:mod:`ryotenkai_shared.errors`. To make the migration permanent, this
sentinel blocks every reintroduction:

* ``from ryotenkai_shared.utils.result import ...`` — any name.
* ``import ryotenkai_shared.utils.result``.
* Attribute access ``ryotenkai_shared.utils.result.X`` (e.g. via a
  fully-qualified module reference).

Scope:

* Walks ``packages/*/src/**/*.py`` AND ``tests/**/*.py``.
* Self-excludes ``tests/_lint/`` (this file documents the forbidden
  module in its body and the policy docs cross-reference it; treating
  those strings as imports would create a paradox).

Permanent — there is no allowlist. If you genuinely need to express
a result-style return shape, raise a :class:`RyotenkAIError` subclass
from ``ryotenkai_shared.errors`` or, for non-error returns, just use
a typed dict / tuple / dataclass.

See:

* ``docs/plans/sharded-stargazing-wigderson.md`` (Phase A2).
* ``docs/testing/mock_policy.md`` for the broader testing policy.
"""

from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGES_ROOT = REPO_ROOT / "packages"
TESTS_ROOT = REPO_ROOT / "tests"
LINT_DIR = TESTS_ROOT / "_lint"

# The forbidden module's fully-qualified dotted path.
_FORBIDDEN_MODULE = "ryotenkai_shared.utils.result"


def _is_forbidden_import_from(node: ast.AST) -> bool:
    """Return True for ``from ryotenkai_shared.utils.result import ...``.

    Also catches relative-resolution noise: we anchor on the literal
    dotted ``module`` string, since the result lived at
    ``packages/shared/src/ryotenkai_shared/utils/result.py``.
    """
    return isinstance(node, ast.ImportFrom) and node.module == _FORBIDDEN_MODULE


def _is_forbidden_import(node: ast.AST) -> bool:
    """Return True for ``import ryotenkai_shared.utils.result[ as X]``."""
    if not isinstance(node, ast.Import):
        return False
    for alias in node.names:
        if alias.name == _FORBIDDEN_MODULE:
            return True
    return False


def _is_forbidden_attribute(node: ast.AST) -> bool:
    """Return True for ``ryotenkai_shared.utils.result.SOMETHING`` access.

    Matches both ``a.b.c.X`` (``Attribute(value=Attribute(...))``) and
    ``a.b.c`` referenced directly (rare; covered by import). We unparse
    the ``ast.Attribute`` chain and check the dotted prefix.
    """
    if not isinstance(node, ast.Attribute):
        return False
    # Build the dotted name from the leaf up.
    parts: list[str] = []
    cur: ast.AST = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if not isinstance(cur, ast.Name):
        return False
    parts.append(cur.id)
    dotted = ".".join(reversed(parts))
    # Match the forbidden module as a prefix (``...result`` itself OR
    # ``...result.X``); never partial-match against an unrelated module
    # whose name happens to start with the same chars.
    if dotted == _FORBIDDEN_MODULE:
        return True
    return dotted.startswith(_FORBIDDEN_MODULE + ".")


def _scan_file(path: Path) -> list[str]:
    """Return human-readable violation lines for ``path``."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, OSError):
        return []

    violations: list[str] = []
    for node in ast.walk(tree):
        if _is_forbidden_import_from(node):
            violations.append(
                f"{path}:{node.lineno}: forbidden `from {_FORBIDDEN_MODULE} "
                f"import ...` — use ryotenkai_shared.errors instead"
            )
        elif _is_forbidden_import(node):
            violations.append(
                f"{path}:{node.lineno}: forbidden `import {_FORBIDDEN_MODULE}` "
                f"— module deleted in Phase A2 finale"
            )
        elif _is_forbidden_attribute(node):
            violations.append(
                f"{path}:{node.lineno}: forbidden attribute access on "
                f"`{_FORBIDDEN_MODULE}.*` — module deleted in Phase A2 finale"
            )
    return violations


def _iter_target_files() -> list[Path]:
    """Walk ``packages/*/src/`` and ``tests/`` for ``*.py`` files,
    excluding the sentinel directory itself and ``__pycache__``."""
    files: list[Path] = []
    if PACKAGES_ROOT.exists():
        for src_root in PACKAGES_ROOT.glob("*/src"):
            for py in src_root.rglob("*.py"):
                if "__pycache__" in py.parts:
                    continue
                files.append(py)
    if TESTS_ROOT.exists():
        for py in TESTS_ROOT.rglob("*.py"):
            if "__pycache__" in py.parts:
                continue
            # Self-exclude: this file references the forbidden module
            # by name in its docstring; treating those strings as
            # imports would be a self-defeating paradox.
            try:
                py.relative_to(LINT_DIR)
            except ValueError:
                pass
            else:
                continue
            files.append(py)
    return files


def _scan_all() -> list[str]:
    violations: list[str] = []
    for py in _iter_target_files():
        violations.extend(_scan_file(py))
    return violations


def test_no_apperror_or_result_imports() -> None:
    """No production source or test may import the deleted
    ``ryotenkai_shared.utils.result`` module. Phase A2 finale, permanent.
    """
    violations = _scan_all()
    assert not violations, (
        "The legacy Result[T, AppError] module was deleted in Phase A2 "
        "finale (sharded-stargazing-wigderson). Any new dependency on it "
        "is forbidden. Use raise-based contracts on `RyotenkAIError` "
        "subclasses from `ryotenkai_shared.errors` instead.\n\n"
        "Offenders:\n  " + "\n  ".join(violations)
    )


def test_sentinel_catches_synthetic_import_from(tmp_path: Path) -> None:
    """Synthetic ``from ryotenkai_shared.utils.result import Ok`` is flagged."""
    bad = tmp_path / "bad_from.py"
    bad.write_text(
        "from ryotenkai_shared.utils.result import Ok\n",
        encoding="utf-8",
    )
    findings = _scan_file(bad)
    assert findings, "Sentinel must catch `from ryotenkai_shared.utils.result import ...`"
    assert _FORBIDDEN_MODULE in findings[0]


def test_sentinel_catches_synthetic_import(tmp_path: Path) -> None:
    """Synthetic ``import ryotenkai_shared.utils.result`` is flagged."""
    bad = tmp_path / "bad_import.py"
    bad.write_text(
        "import ryotenkai_shared.utils.result\n",
        encoding="utf-8",
    )
    findings = _scan_file(bad)
    assert findings, "Sentinel must catch `import ryotenkai_shared.utils.result`"


def test_sentinel_catches_synthetic_attribute_access(tmp_path: Path) -> None:
    """Synthetic ``ryotenkai_shared.utils.result.Ok`` attribute access is flagged.

    Note: a bare ``ryotenkai_shared.utils.result.Ok`` reference requires
    ``ryotenkai_shared`` to be importable in scope — typically via
    ``import ryotenkai_shared``. The sentinel catches both the import
    AND the attribute access regardless of order.
    """
    bad = tmp_path / "bad_attr.py"
    bad.write_text(
        "import ryotenkai_shared\n"
        "_ = ryotenkai_shared.utils.result.Ok\n",
        encoding="utf-8",
    )
    findings = _scan_file(bad)
    assert findings, "Sentinel must catch `ryotenkai_shared.utils.result.X` access"
    assert any("attribute access" in v for v in findings), findings


def test_sentinel_ignores_unrelated_dotted_name(tmp_path: Path) -> None:
    """An unrelated module whose name shares a prefix must NOT be flagged."""
    ok = tmp_path / "ok.py"
    ok.write_text(
        "from ryotenkai_shared.utils.logger import logger\n"
        "from ryotenkai_shared.errors import RyotenkAIError\n",
        encoding="utf-8",
    )
    findings = _scan_file(ok)
    assert not findings, findings


def test_sentinel_ignores_string_literals(tmp_path: Path) -> None:
    """Regression-pin test: a regression assertion like
    ``assert \"from ryotenkai_shared.utils.result\" not in content`` is a
    string literal, NOT an import — the sentinel must not flag it."""
    ok = tmp_path / "ok_string.py"
    ok.write_text(
        "MARKER = \"from ryotenkai_shared.utils.result\"\n"
        "assert MARKER == \"from ryotenkai_shared.utils.result\"\n",
        encoding="utf-8",
    )
    findings = _scan_file(ok)
    assert not findings, findings
