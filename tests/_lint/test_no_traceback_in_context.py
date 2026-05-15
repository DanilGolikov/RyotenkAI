"""Sentinel: ``traceback.format_exc()`` must never land inside an error context.

Phase A1 (sharded-stargazing-wigderson, 2026-05-14) sets up the
unified error hierarchy. A recurring failure mode in the legacy
``AppError`` dataclass was stuffing the full traceback into the
``details`` dict, which then leaked into the wire body and into logs
where the same traceback was already being emitted separately --
duplicating noise AND smuggling filesystem paths through the public
API.

This sentinel walks ``packages/*/src/**/*.py`` and flags any call to
``traceback.format_exc(...)`` whose result is placed into a position
that looks like an error context. Specifically, it flags:

* Use as a value in a dict literal under the keys ``context``,
  ``details``, ``traceback`` (any-position), or ``traceback_summary``.
* Use as a keyword argument named ``context`` (or those four
  alternatives) to any call.
* Plain assignment ``something.context = traceback.format_exc()``
  where the attribute name is in the bad-attr set.

False positives are fine if the code is acceptable: add the file to
``_LEGACY_ALLOWLIST`` below until the migration cleans it up. Phase
A1 ships ``utils/result.py`` in the allowlist (it'll be deleted in
Phase A2 anyway).
"""

from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGES_ROOT = REPO_ROOT / "packages"

# Attribute / dict-key / kwarg names that, when paired with
# ``traceback.format_exc()``, represent leakage into an error context.
_BAD_KEYS: frozenset[str] = frozenset({
    "context",
    "details",
    "traceback",
    "traceback_summary",
})


# Files that pre-date Phase A1 and legitimately use ``format_exc`` in
# a way that this sentinel cannot easily distinguish from the bad
# leakage pattern (e.g. populating a value-object dataclass used by
# the community plugin loader, not a raised exception).
#
# Entries:
#
# * ``community/.../loader.py`` and ``libs.py`` -- the plugin loader
#   populates a ``LoadFailure`` dataclass (NOT a raised exception)
#   with ``traceback=...`` for developer drilldown. The field is
#   internal to the community loader's failure-collection API and
#   never crosses an HTTP boundary; the sentinel's protection is
#   moot for this surface. If/when LoadFailure is replaced by a
#   typed exception (Phase F), this entry can be removed.
#
# Phase A2 finale (2026-05-16): the ``shared/.../utils/result.py``
# entry was retired together with the file it pinned.
_LEGACY_ALLOWLIST: frozenset[str] = frozenset({
    # Path relative to packages/ root.
    "community/src/ryotenkai_community/loader.py",
    "community/src/ryotenkai_community/libs.py",
})


def _is_format_exc_call(node: ast.AST) -> bool:
    """Return True if ``node`` is a call to ``traceback.format_exc(...)``.

    Accepts both ``traceback.format_exc()`` and bare ``format_exc()``
    (works under ``from traceback import format_exc`` aliasing).
    """
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Attribute) and func.attr == "format_exc":
        return True
    if isinstance(func, ast.Name) and func.id == "format_exc":
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
        # --- dict literal: {"context": format_exc()}  --------------------
        if isinstance(node, ast.Dict):
            for key_node, value_node in zip(node.keys, node.values):
                if not _is_format_exc_call(value_node):
                    continue
                if isinstance(key_node, ast.Constant) and isinstance(key_node.value, str):
                    if key_node.value in _BAD_KEYS:
                        violations.append(
                            f"{path}:{value_node.lineno}: traceback.format_exc() inside "
                            f"dict literal under key {key_node.value!r}"
                        )

        # --- function call kwargs: foo(context=format_exc()) -------------
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg is None:
                    continue
                if kw.arg in _BAD_KEYS and _is_format_exc_call(kw.value):
                    violations.append(
                        f"{path}:{kw.value.lineno}: traceback.format_exc() passed as "
                        f"keyword {kw.arg!r}"
                    )

        # --- attribute assignment: self.context = format_exc()  ----------
        if isinstance(node, ast.Assign):
            if not _is_format_exc_call(node.value):
                continue
            for target in node.targets:
                if isinstance(target, ast.Attribute) and target.attr in _BAD_KEYS:
                    violations.append(
                        f"{path}:{node.lineno}: traceback.format_exc() assigned to "
                        f"attribute {target.attr!r}"
                    )

    return violations


def _scan_packages() -> list[str]:
    """Walk ``packages/*/src/`` and collect all violations not in the allowlist."""
    if not PACKAGES_ROOT.exists():
        return []
    violations: list[str] = []
    for src_root in PACKAGES_ROOT.glob("*/src"):
        for py in src_root.rglob("*.py"):
            if "__pycache__" in py.parts:
                continue
            rel = py.relative_to(PACKAGES_ROOT).as_posix()
            if rel in _LEGACY_ALLOWLIST:
                continue
            violations.extend(_scan_file(py))
    return violations


def test_no_traceback_in_error_context() -> None:
    """No production code may stuff ``traceback.format_exc()`` into an error context.

    Tracebacks belong in the structured log line emitted by the
    handler, NOT in the wire payload's ``context`` / ``details`` dict
    or anywhere they could leak filesystem paths to the client.
    """
    violations = _scan_packages()
    assert not violations, (
        "traceback.format_exc() must never land in an error context/details "
        "dict -- it leaks filesystem paths and duplicates log noise.\n"
        "Move the traceback to ``logger.exception(...)`` or "
        "``logger.error(..., exc_info=exc)``.\n\n"
        "Offenders:\n  " + "\n  ".join(violations)
    )


def test_sentinel_catches_synthetic_dict_violation(tmp_path: Path) -> None:
    """A fake offending file with ``{"context": format_exc()}`` is flagged."""
    bad = tmp_path / "bad.py"
    bad.write_text(
        "import traceback\n"
        "def boom():\n"
        "    raise RuntimeError('x', {'context': traceback.format_exc()})\n",
        encoding="utf-8",
    )
    findings = _scan_file(bad)
    assert any("context" in line for line in findings), findings


def test_sentinel_catches_synthetic_kwarg_violation(tmp_path: Path) -> None:
    """A fake offending file with ``Error(context=format_exc())`` is flagged."""
    bad = tmp_path / "bad_kwarg.py"
    bad.write_text(
        "import traceback\n"
        "def make():\n"
        "    return ValueError(context=traceback.format_exc())\n",
        encoding="utf-8",
    )
    findings = _scan_file(bad)
    assert any("context" in line for line in findings), findings


def test_sentinel_catches_synthetic_attr_assignment_violation(tmp_path: Path) -> None:
    """A fake offending file with ``self.context = format_exc()`` is flagged."""
    bad = tmp_path / "bad_attr.py"
    bad.write_text(
        "import traceback\n"
        "class X:\n"
        "    def boom(self):\n"
        "        self.context = traceback.format_exc()\n",
        encoding="utf-8",
    )
    findings = _scan_file(bad)
    assert any("context" in line for line in findings), findings


def test_sentinel_ignores_format_exc_in_safe_position(tmp_path: Path) -> None:
    """``logger.error("...", traceback.format_exc())`` is fine -- not flagged."""
    ok = tmp_path / "ok.py"
    ok.write_text(
        "import logging\nimport traceback\n"
        "log = logging.getLogger(__name__)\n"
        "def boom():\n"
        "    log.error('crash', traceback.format_exc())\n",
        encoding="utf-8",
    )
    findings = _scan_file(ok)
    assert not findings, findings


def test_sentinel_ignores_non_bad_key_in_dict(tmp_path: Path) -> None:
    """``{"some_other_key": format_exc()}`` is fine (key is not in the bad set)."""
    ok = tmp_path / "ok2.py"
    ok.write_text(
        "import traceback\n"
        "def boom():\n"
        "    return {'tb_only': traceback.format_exc()}\n",
        encoding="utf-8",
    )
    findings = _scan_file(ok)
    assert not findings, findings
