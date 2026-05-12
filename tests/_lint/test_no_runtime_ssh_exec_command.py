"""Sentinel — Mac-side runtime modules MUST NOT issue ``ssh.exec_command``.

Phase 3 PR-3.1 of transport-unification-v2. ``importlinter`` is a
static-graph check on **import statements**; it can forbid
``ryotenkai_shared.utils.ssh_client`` imports from the runtime
path but it CANNOT see *runtime call expressions*. A module that
lawfully imports ``SSHClient`` for one bootstrap-time call could
re-use the same instance to do an arbitrary runtime call —
importlinter would still pass. This sentinel closes that gap by
walking the AST and flagging every ``.exec_command(`` call site
outside the explicit allowlist.

Companion docs:
    docs/architecture/SSH_SURFACE.md
    packages/control/tests/sentinel/bootstrap_allowlist.py

The allowlist is intentionally in a separate module so that
adding a new SSH call site requires a PR review against
``bootstrap_allowlist.py`` with a written justification (RP22).
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path


def _load_allowlist() -> frozenset[str]:
    """Load :data:`ALLOWLIST` directly from ``bootstrap_allowlist.py``
    sitting next to this file. Avoids the ``from packages.control....``
    import path which doesn't work — ``packages/`` is not a Python
    package directory."""
    here = Path(__file__).resolve()
    spec = importlib.util.spec_from_file_location(
        "_bootstrap_allowlist",
        here.parent / "bootstrap_allowlist.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ALLOWLIST


ALLOWLIST: frozenset[str] = _load_allowlist()


def _control_src() -> Path:
    """Resolve ``packages/control/src/ryotenkai_control/`` from this file.

    File is at ``tests/_lint/test_*.py`` → parents[2] = worktree root.
    """
    here = Path(__file__).resolve()
    return (
        here.parents[2]
        / "packages"
        / "control"
        / "src"
        / "ryotenkai_control"
    )


def _module_name(path: Path, src_root: Path) -> str:
    """Convert ``packages/control/src/ryotenkai_control/foo/bar.py``
    → ``ryotenkai_control.foo.bar``."""
    rel = path.relative_to(src_root.parent)
    parts = list(rel.parts)
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    else:
        parts[-1] = parts[-1].removesuffix(".py")
    return ".".join(parts)


def _is_exec_command_attr_call(node: ast.AST) -> bool:
    """Match ``<anything>.exec_command(...)`` AST shapes.

    The ``ssh_client.exec_command(...)`` and
    ``self.ssh.exec_command(...)`` patterns both surface as
    ``ast.Call(func=ast.Attribute(attr='exec_command'))``. We
    deliberately do NOT scope to ``SSHClient.exec_command`` because
    the type information isn't reliably available at AST level —
    matching by attribute name is over-broad but the false positive
    rate is acceptable (no other class in the codebase has a method
    with this exact name).
    """
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if not isinstance(func, ast.Attribute):
        return False
    return func.attr == "exec_command"


def test_no_exec_command_outside_allowlist() -> None:
    """For every ``.py`` under ``packages/control/src/`` that is NOT
    on :data:`ALLOWLIST`, assert no ``.exec_command(...)`` call site
    exists in the AST.
    """
    src = _control_src()
    assert src.exists(), f"sentinel mis-anchored: {src}"

    violations: list[str] = []
    for path in src.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        modname = _module_name(path, src)
        if modname in ALLOWLIST:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as exc:
            raise AssertionError(f"{path}: syntax error: {exc}") from exc
        for node in ast.walk(tree):
            if _is_exec_command_attr_call(node):
                violations.append(
                    f"{path.relative_to(src.parent.parent)}:{node.lineno}",
                )

    assert not violations, (
        "Mac-side modules MUST NOT issue .exec_command() outside the "
        "bootstrap / pod-pull allowlist. Migrate the call to HTTP via "
        "JobClient (see docs/architecture/SSH_SURFACE.md) or — if it "
        "is genuinely a bootstrap/data-pull concern — add the module "
        "to bootstrap_allowlist.py with a justification.\n  "
        + "\n  ".join(violations)
    )


def test_allowlist_modules_actually_exist() -> None:
    """Each entry in the allowlist must resolve to an existing
    file. Catches typos / file renames that would silently disable
    enforcement on the renamed module."""
    src = _control_src()
    missing: list[str] = []
    for modname in ALLOWLIST:
        if not modname.startswith("ryotenkai_control."):
            missing.append(f"{modname}: must start with ryotenkai_control.")
            continue
        rel = modname.replace("ryotenkai_control.", "", 1).replace(".", "/")
        candidates = [src / f"{rel}.py", src / rel / "__init__.py"]
        if not any(c.exists() for c in candidates):
            missing.append(f"{modname}: no file found at {candidates}")
    assert not missing, "\n  ".join(["allowlist drift:", *missing])
