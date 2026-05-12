"""Sentinel: запрещаем mock над Protocol-интерфейсами в greenfield ``tests/``.

Greenfield-директория — никаких legacy-исключений. Если кто-то попытается
обмокать ``IMLflowManager``/``IPodLifecycleClient``/etc., тест падает.
Legacy ``packages/<pkg>/tests/`` сюда не попадает по построению.
"""

from __future__ import annotations

import ast
from pathlib import Path

_PROTOCOLS: frozenset[str] = frozenset(
    {
        "IMLflowManager",
        "IPodLifecycleClient",
        "IRunPodAPI",
        "ITrainerSpawner",
        "ISSHClient",
        "IHFHubClient",
        "IJobClient",
        "Clock",
    }
)

_PROTOCOL_TARGET_RE_SUFFIXES = tuple(f".{name}" for name in _PROTOCOLS)


def _tests_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _is_self_or_lint_file(path: Path, tests_root: Path) -> bool:
    rel = path.relative_to(tests_root)
    parts = rel.parts
    if not parts:
        return True
    return parts[0] == "_lint"


def _matches_patch_target(value: str) -> bool:
    if value in _PROTOCOLS:
        return True
    return value.endswith(_PROTOCOL_TARGET_RE_SUFFIXES)


def _attr_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return None


def _func_name(node: ast.Call) -> str | None:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _collect_violations(tree: ast.AST, path: Path) -> list[str]:
    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fname = _func_name(node)
        if fname is None:
            continue

        if fname == "patch":
            target_arg = next(iter(node.args), None)
            if (
                isinstance(target_arg, ast.Constant)
                and isinstance(target_arg.value, str)
                and _matches_patch_target(target_arg.value)
            ):
                violations.append(
                    f"{path}:{node.lineno}: mock.patch('{target_arg.value}') targets a Protocol"
                )

        if fname in {"MagicMock", "create_autospec", "Mock", "AsyncMock", "NonCallableMock"}:
            spec_target: ast.AST | None = None
            if fname == "create_autospec" and node.args:
                spec_target = node.args[0]
            for kw in node.keywords:
                if kw.arg == "spec":
                    spec_target = kw.value
                    break
            if spec_target is not None:
                name = _attr_name(spec_target)
                if name is not None and name in _PROTOCOLS:
                    violations.append(f"{path}:{node.lineno}: {fname}(spec={name}) targets a Protocol")
    return violations


def _scan(root: Path) -> list[str]:
    tests_root = _tests_root()
    violations: list[str] = []
    for path in root.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        if _is_self_or_lint_file(path, tests_root):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        violations.extend(_collect_violations(tree, path))
    return violations


def test_no_mocking_of_protocols_in_tests() -> None:
    root = _tests_root()
    assert root.exists(), f"sentinel mis-anchored: {root} is missing"
    violations = _scan(root)
    assert not violations, "Mocking Protocols is forbidden in tests/. Use canonical fakes from tests/_fakes/.\n  " + "\n  ".join(
        violations
    )


def test_sentinel_detects_synthetic_violation(tmp_path: Path) -> None:
    fake_root = tmp_path / "tests" / "unit"
    fake_root.mkdir(parents=True)
    bad = fake_root / "test_bad.py"
    bad.write_text(
        "from unittest.mock import patch\n"
        "@patch('ryotenkai_shared.infrastructure.mlflow.protocol.IMLflowManager')\n"
        "def test_x(_): pass\n",
        encoding="utf-8",
    )
    tree = ast.parse(bad.read_text(encoding="utf-8"))
    found = _collect_violations(tree, bad)
    assert any("IMLflowManager" in v for v in found), found

    bad2 = fake_root / "test_bad2.py"
    bad2.write_text(
        "from unittest.mock import MagicMock\n"
        "class IMLflowManager: pass\n"
        "_ = MagicMock(spec=IMLflowManager)\n",
        encoding="utf-8",
    )
    tree2 = ast.parse(bad2.read_text(encoding="utf-8"))
    found2 = _collect_violations(tree2, bad2)
    assert any("MagicMock(spec=IMLflowManager)" in v for v in found2), found2
