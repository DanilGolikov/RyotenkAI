"""Categorize ``AsyncMock`` usages in the test suite.

For each ``AsyncMock(...)`` call in ``tests/``, emit a CSV row with:

* file:line
* nearby variable name (target of assignment / attribute or keyword
  the literal participates in)
* "shape" — a coarse classification of the call
  (`return_value` / `side_effect` / `bare`)
* whether the surrounding test uses interaction-style assertions
  (``assert_awaited`` / ``await_count`` / ``await_args``) — if yes,
  KEEP as a legitimate interaction test
* guessed replacement Fake or KEEP

The output is written to stdout as CSV.  No file IO besides reading
the test sources.
"""

from __future__ import annotations

import ast
import csv
import re
import sys
from pathlib import Path


_INTERACTION_PATTERNS = re.compile(
    r"\b("
    r"assert_(awaited|called|not_awaited|not_called|awaited_(once|with|once_with))|"
    r"await_(args|count)|"
    r"call_(args|count)"
    r")\b",
)


def _interaction_var_names(source: str) -> set[str]:
    """Return variable names that appear in interaction-style asserts."""
    names: set[str] = set()
    for line in source.splitlines():
        m = re.findall(
            r"(\w+(?:\.\w+)*)\.(assert_awaited|assert_awaited_once|assert_awaited_once_with|assert_awaited_with|assert_not_awaited|assert_called|assert_called_once|assert_not_called|await_count|await_args|call_args|call_count)",
            line,
        )
        for var, _attr in m:
            # Track the root variable.
            names.add(var.split(".")[0])
            # Also track the dotted path (so ``client.request_stop``
            # matches when the test does
            # ``client.request_stop.assert_awaited_once()``).
            names.add(var)
    return names


def _kw_shape(call: ast.Call) -> str:
    """Classify the AsyncMock(...) call shape."""
    if not call.args and not call.keywords:
        return "bare"
    kwargs = {k.arg for k in call.keywords if k.arg is not None}
    if "side_effect" in kwargs:
        return "side_effect"
    if "return_value" in kwargs:
        return "return_value"
    if "spec" in kwargs:
        return "spec"
    return "other"


def _enclosing_assignment(tree: ast.AST, call: ast.Call) -> str:
    """Find the var/attribute/keyword the AsyncMock literal binds to."""

    hit: list[str] = []
    for node in ast.walk(tree):
        # Direct assignment: ``x = AsyncMock(...)``
        if isinstance(node, ast.Assign) and node.value is call:
            target = node.targets[0]
            hit.append(ast.unparse(target))
        # ``x: T = AsyncMock(...)``
        if isinstance(node, ast.AnnAssign) and node.value is call:
            hit.append(ast.unparse(node.target))
        # Call-level keywords / positional args.
        if isinstance(node, ast.Call):
            for k in node.keywords:
                if k.value is call:
                    hit.append(f"<kw>{k.arg}")
            for a in node.args:
                if a is call:
                    func_name = ast.unparse(node.func)
                    hit.append(f"<arg>{func_name}")
    return hit[0] if hit else "<unknown>"


def _guess_replacement(
    var_name: str,
    is_interaction: bool,
    shape: str,
    is_module_patch: bool,
) -> str:
    """Heuristic mapping. Always conservative — when unsure, KEEP."""
    if is_interaction:
        return "KEEP (interaction test)"
    if is_module_patch:
        return "KEEP (patch.new=AsyncMock for module-level async fn)"

    lower = var_name.lower()
    # ``<arg>...`` means the AsyncMock is a positional argument inside a
    # constructor — typically a "placeholder coroutine callable" use.
    # Apply that classification first (before name-based guesses).
    if var_name.startswith("<arg>") and shape == "bare":
        return "Replace with async def _noop(*a, **k): pass"
    if "job_client" in lower or var_name == "<kw>job_client":
        return "FakeJobClient"
    if "lifecycle" in lower or "pod_client" in lower:
        return "FakePodLifecycleClient"
    if "mlflow" in lower:
        return "FakeMLflowManager"
    if "ssh" in lower:
        return "FakeSSHClient"
    if "hf_hub" in lower or "hfhub" in lower:
        return "FakeHFHubClient"
    if "runpod_api" in lower or "runpod" in lower and "api" in lower:
        return "FakeRunPodAPI"
    if "trainer_spawner" in lower:
        return "FakeTrainerSpawner"

    if shape == "bare":
        # Bare AsyncMock() used as a placeholder coroutine callable.
        return "Replace with async def _noop(*a, **k): pass"
    if shape == "return_value":
        # Single async method stub.
        return "KEEP (per-method async stub)"
    if shape == "side_effect":
        return "KEEP (per-method async stub w/ side effect)"
    return "REVIEW"


def _is_module_patch_new(call: ast.Call, tree: ast.AST) -> bool:
    """True if this AsyncMock(...) is the ``new=`` arg of patch(...)."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            target = (
                func.attr if isinstance(func, ast.Attribute) else
                func.id if isinstance(func, ast.Name) else None
            )
            if target == "patch":
                for kw in node.keywords:
                    if kw.arg == "new" and kw.value is call:
                        return True
    return False


def iter_asyncmock_calls(path: Path):
    """Yield ``(line, call, tree, source)`` for every AsyncMock call."""
    src = path.read_text()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = (
                func.id if isinstance(func, ast.Name) else
                func.attr if isinstance(func, ast.Attribute) else None
            )
            if name == "AsyncMock":
                yield node, tree, src


def main(argv: list[str] | None = None) -> int:
    root = Path("tests")
    writer = csv.writer(sys.stdout)
    writer.writerow(
        ["file", "line", "var", "shape", "module_patch_new",
         "interaction_var", "guessed_replacement"],
    )

    for f in sorted(root.rglob("test_*.py")):
        # Skip telemetry / sentinel allowlist.
        if ".telemetry" in f.parts:
            continue
        if "_lint" in f.parts:
            continue
        src = f.read_text()
        interaction_vars = _interaction_var_names(src)

        for call, tree, _src in iter_asyncmock_calls(f):
            var = _enclosing_assignment(tree, call)
            shape = _kw_shape(call)
            module_patch = _is_module_patch_new(call, tree)
            # Interaction match: either the root var name is in the set
            # of interaction-asserted vars, or any prefix of the var
            # (``client.request_stop``) matches.
            is_interaction = False
            stripped = var.lstrip("<kw><arg>").split(".")[0]
            if stripped in interaction_vars:
                is_interaction = True
            # Also check if var itself appears in interaction patterns.
            if var in interaction_vars:
                is_interaction = True

            replacement = _guess_replacement(
                var, is_interaction, shape, module_patch,
            )
            writer.writerow(
                [
                    str(f),
                    call.lineno,
                    var,
                    shape,
                    "yes" if module_patch else "",
                    "yes" if is_interaction else "",
                    replacement,
                ],
            )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
