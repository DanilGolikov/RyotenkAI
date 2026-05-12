"""Quarantine-SLA sentinel.

Walks the greenfield tests tree and asserts no test has been
quarantined (``@pytest.mark.flaky`` or
``@pytest.mark.skip(reason="flaky: ...")``) for more than 30 days
without a follow-up. The contract: every quarantine decorator is
accompanied by a ``# quarantined_at: YYYY-MM-DD`` comment within 3
lines above or on the decorator line; the parser locates the date and
fails CI if today is more than 30 days past.

Enforces Decision 6 of
``docs/plans/structured-hopping-starfish.md`` (flake management — 30
day quarantine SLA).

No tests are quarantined yet, so this sentinel passes trivially.
"""

from __future__ import annotations

import ast
import datetime as _dt
import re
from pathlib import Path

import pytest

_QUARANTINE_DAYS = 30
_ROOT = Path(__file__).resolve().parents[1]  # tests/

# We accept any of these as a "flake quarantine" decorator pattern:
_QUARANTINE_MARKERS = ("flaky",)
_QUARANTINE_SKIP_REASON_RE = re.compile(r"flaky\s*:", re.IGNORECASE)
_DATE_RE = re.compile(r"#\s*quarantined_at\s*:\s*(\d{4}-\d{2}-\d{2})")


def _decorator_targets_flaky(decorator: ast.expr) -> bool:
    """Detect ``@pytest.mark.flaky`` or ``@pytest.mark.skip(reason='flaky:...')``."""
    # @pytest.mark.flaky / @pytest.mark.flaky(...)
    target = decorator.func if isinstance(decorator, ast.Call) else decorator
    if isinstance(target, ast.Attribute):
        if target.attr in _QUARANTINE_MARKERS:
            return True
        # @pytest.mark.skip(reason="flaky: ...")
        if target.attr == "skip" and isinstance(decorator, ast.Call):
            for kw in decorator.keywords:
                if kw.arg == "reason" and isinstance(kw.value, ast.Constant):
                    if isinstance(kw.value.value, str) and _QUARANTINE_SKIP_REASON_RE.search(
                        kw.value.value,
                    ):
                        return True
    return False


def _find_quarantined_at(lines: list[str], decorator_lineno: int) -> str | None:
    """Look for ``# quarantined_at: YYYY-MM-DD`` on the decorator line or 3 lines above."""
    for i in range(max(0, decorator_lineno - 3), decorator_lineno):
        match = _DATE_RE.search(lines[i])
        if match:
            return match.group(1)
    return None


def _iter_python_files() -> list[Path]:
    return [
        p for p in _ROOT.rglob("*.py")
        if "/.venv" not in str(p)
        and "/__pycache__" not in str(p)
        # Don't sentinel-check this very file.
        and p.resolve() != Path(__file__).resolve()
    ]


def test_no_test_has_been_quarantined_too_long() -> None:
    today = _dt.date.today()
    violations: list[str] = []
    missing_date: list[str] = []

    for path in _iter_python_files():
        try:
            source = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        lines = source.splitlines()
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            for decorator in node.decorator_list:
                if not _decorator_targets_flaky(decorator):
                    continue
                # decorator.lineno is 1-indexed and points at the @ line.
                start = decorator.lineno - 1
                date_str = _find_quarantined_at(lines, start)
                if date_str is None:
                    missing_date.append(
                        f"{path}:{decorator.lineno} {node.name!r} "
                        f"is quarantined without a # quarantined_at: YYYY-MM-DD comment",
                    )
                    continue
                try:
                    quarantined_at = _dt.date.fromisoformat(date_str)
                except ValueError:
                    missing_date.append(
                        f"{path}:{decorator.lineno} unparseable quarantined_at: {date_str!r}",
                    )
                    continue
                age = (today - quarantined_at).days
                if age > _QUARANTINE_DAYS:
                    violations.append(
                        f"{path}:{decorator.lineno} {node.name!r} has been "
                        f"quarantined for {age} days (since {date_str}); "
                        f"max {_QUARANTINE_DAYS}",
                    )

    if missing_date:
        pytest.fail(
            "quarantined tests must have a # quarantined_at: YYYY-MM-DD comment:\n  "
            + "\n  ".join(missing_date),
        )
    if violations:
        pytest.fail(
            "quarantined tests have exceeded the 30-day SLA:\n  "
            + "\n  ".join(violations),
        )


__all__: list[str] = []
