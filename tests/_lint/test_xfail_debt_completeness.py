"""Sentinel: every strict-True xfail must carry an ``xfail-debt:<id>`` token.

Agent-driven development requires that every strict-True xfail be
traceable to a documented removal path. The contract this sentinel
enforces:

1. Every ``@pytest.mark.xfail(strict=True, ...)`` decorator (or
   ``pytestmark = pytest.mark.xfail(strict=True, ...)`` module-level
   marker) MUST have a ``reason`` whose text contains
   ``xfail-debt:<id>`` for some non-empty ``<id>``.
2. Every ``<id>`` referenced from test code MUST appear in
   ``docs/migration/xfail_debt.md`` (as a code-fenced backtick token).
3. Escape hatch: tokens of the form ``ad-hoc-<timestamp>`` (e.g.
   ``ad-hoc-20260512T203000``) are accepted without a ledger row.
   They are intended for one-time emergency unblocks and should be
   resolved within 30 days.

Why a sentinel (and not just CI lint):
- Agents writing tests can add xfails inadvertently; the sentinel
  prevents the debt heap from growing silently.
- The ledger row (``docs/migration/xfail_debt.md``) carries owner +
  trigger so future contributors know what unlocks the fix.

Mechanism:
- AST walk over ``tests/**/test_*.py`` (excluding ``tests/_lint/`` to
  avoid self-reference).
- Parse the reason string from every strict-True xfail marker; extract
  ``xfail-debt:<id>`` tokens.
- Cross-reference against ``docs/migration/xfail_debt.md``.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TESTS_ROOT = REPO_ROOT / "tests"
LEDGER_PATH = REPO_ROOT / "docs" / "migration" / "xfail_debt.md"

TOKEN_RE = re.compile(r"xfail-debt:([a-zA-Z0-9_\-]+)")
AD_HOC_RE = re.compile(r"^ad-hoc-\d{8,}$")


def _iter_test_files() -> list[Path]:
    """Every ``test_*.py`` under ``tests/`` except ``tests/_lint/``."""
    out: list[Path] = []
    for p in TESTS_ROOT.rglob("test_*.py"):
        rel = p.relative_to(TESTS_ROOT)
        if rel.parts and rel.parts[0] == "_lint":
            continue
        out.append(p)
    return out


def _reason_text(call: ast.Call) -> str | None:
    """Return the literal `reason=` text from an xfail call, or None."""
    for kw in call.keywords:
        if kw.arg != "reason":
            continue
        # The value may be a string literal, a JoinedStr (f-string),
        # a parenthesised tuple of strings ("a" "b"), or a Call (rare).
        # Use ast.unparse to get a stable textual representation, then
        # strip quotes and concatenate adjacent strings.
        try:
            return ast.unparse(kw.value)
        except Exception:  # noqa: BLE001
            return None
    return None


def _is_strict_true(call: ast.Call) -> bool:
    for kw in call.keywords:
        if kw.arg == "strict" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
            return True
    return False


def _is_xfail_call(node: ast.AST) -> bool:
    """True if ``node`` is a ``pytest.mark.xfail(...)`` call."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    # Match attribute chain `.xfail`
    if not isinstance(func, ast.Attribute) or func.attr != "xfail":
        return False
    return True


def _strict_xfails_in_file(path: Path) -> list[tuple[int, str | None]]:
    """Return (line_number, reason_text) for every strict-True xfail in the file."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return []
    out: list[tuple[int, str | None]] = []
    for node in ast.walk(tree):
        if _is_xfail_call(node) and _is_strict_true(node):
            out.append((node.lineno, _reason_text(node)))
    return out


def _ledger_tokens() -> set[str]:
    """Parse the ledger and extract every documented xfail-debt id.

    A token is documented when it appears verbatim inside backticks
    (``\`xfail-debt:<id>\``` is NOT used; instead the ledger lists the
    bare id in the first column inside backticks: ``\`<id>\```).
    """
    if not LEDGER_PATH.exists():
        return set()
    text = LEDGER_PATH.read_text(encoding="utf-8")
    out: set[str] = set()
    # Match either `<id>` (backtick id) or xfail-debt:<id>
    for m in re.finditer(r"`([a-zA-Z0-9_\-]+)`", text):
        out.add(m.group(1))
    for m in TOKEN_RE.finditer(text):
        out.add(m.group(1))
    return out


def test_every_strict_xfail_has_token() -> None:
    """Every strict-True xfail must carry an ``xfail-debt:<id>`` token in its reason."""
    violations: list[str] = []
    for path in _iter_test_files():
        for lineno, reason in _strict_xfails_in_file(path):
            if reason is None:
                violations.append(f"{path.relative_to(REPO_ROOT)}:{lineno}: no `reason=` argument")
                continue
            if not TOKEN_RE.search(reason):
                snippet = reason.replace("\n", " ")[:120]
                violations.append(
                    f"{path.relative_to(REPO_ROOT)}:{lineno}: reason missing `xfail-debt:<id>` token | {snippet}"
                )
    if violations:
        msg = "\n  ".join(["Strict-True xfails missing xfail-debt token:"] + violations)
        pytest.fail(msg)


def test_every_token_has_ledger_entry() -> None:
    """Every ``xfail-debt:<id>`` in test code must appear in ``xfail_debt.md``."""
    tokens_in_code: dict[str, list[str]] = {}
    for path in _iter_test_files():
        for lineno, reason in _strict_xfails_in_file(path):
            if reason is None:
                continue
            for m in TOKEN_RE.finditer(reason):
                tid = m.group(1)
                tokens_in_code.setdefault(tid, []).append(
                    f"{path.relative_to(REPO_ROOT)}:{lineno}"
                )

    documented = _ledger_tokens()
    missing: list[str] = []
    for tid, sites in sorted(tokens_in_code.items()):
        if AD_HOC_RE.match(tid):
            # ad-hoc escape hatch — accepted without ledger row
            continue
        if tid not in documented:
            missing.append(f"{tid}: referenced at {', '.join(sites)} but not documented in {LEDGER_PATH.relative_to(REPO_ROOT)}")
    if missing:
        msg = "\n  ".join(["Undocumented xfail-debt tokens:"] + missing)
        pytest.fail(msg)
