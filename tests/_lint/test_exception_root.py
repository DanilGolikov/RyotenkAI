"""Sentinel: every ``*Error`` class in production code must root in :class:`RyotenkAIError`.

Phase A1 (sharded-stargazing-wigderson, 2026-05-14) introduces a single
unified exception root. New ``class FooError(Exception)``-style
declarations are blocked by this sentinel unless the class is in the
allowlist (``tests/_lint/exception_root_allowlist.yaml``).

The Phase A1 deliverable is the **sentinel + the allowlist** -- existing
classes pre-Phase-A1 are pinned with ``phase-A1-pending-migration``
reasons. Each later phase (A2, B, C, D, E, F) shrinks the allowlist as
it migrates classes to the new root.

Scope and detection:

* Walks ``packages/*/src/**/*.py``.
* For every ``ClassDef`` whose ``name`` ends with ``Error``:
  - Resolves bases at the AST level (Name/Attribute) -- best-effort,
    no import.
  - If any base resolves to a non-RyotenkAI generic exception name
    (``Exception``, ``RuntimeError``, ``LookupError``, ``ValueError``,
    ``TypeError``, ``TimeoutError``, ``OSError``, ``BaseException``,
    ``IOError``), the class must transitively root in
    :class:`RyotenkAIError`.
  - Subclasses of locally-defined types (e.g.
    ``class FooError(BarError)``) are accepted *if* ``BarError`` is
    itself in (or transitively rooted in) the allowed/known set: we
    walk the in-tree class graph and infer the transitive base.
* Non-RyotenkAI Pydantic/BaseModel classes named ``...Error`` are
  ignored (their base is ``BaseModel``, not an exception type).

Maintenance:

* When a class is migrated to ``RyotenkAIError``, remove its row from
  the YAML. The sentinel will then pass without it.
* New ``*Error(Exception)`` raise sites are blocked by this test.
"""

from __future__ import annotations

import ast
from functools import lru_cache
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGES_ROOT = REPO_ROOT / "packages"
ALLOWLIST_PATH = Path(__file__).parent / "exception_root_allowlist.yaml"


# Generic Python exception base names that this sentinel rejects (the
# point of the check is: "don't root your *Error in these directly").
_GENERIC_EXCEPTION_BASES = frozenset({
    "Exception",
    "BaseException",
    "RuntimeError",
    "LookupError",
    "ValueError",
    "TypeError",
    "TimeoutError",
    "OSError",
    "IOError",
    "ArithmeticError",
    "AttributeError",
})


# The RyotenkAI root + its abstract markers. Anything inheriting
# transitively from one of these is OK.
_RYOTENKAI_ROOT_NAMES = frozenset({
    "RyotenkAIError",
    "DomainError",
    "InfrastructureError",
    "InternalError",
    "TransportError",
})


# The root class itself MUST inherit from ``Exception`` (it IS the root
# of the hierarchy), so the sentinel skips it. Other classes inside
# ``ryotenkai_shared.errors`` are abstract markers / concretes already
# rooted via the chain, and are filtered naturally by the transitive
# walk -- no special-case needed there.
_HIERARCHY_ROOT_QUALNAME = "ryotenkai_shared.errors.base.RyotenkAIError"


# Bases that the sentinel ignores entirely (these are NOT exception
# types, even if the class name happens to end in Error -- e.g.
# Pydantic ``BaseModel`` subclasses, StrEnum members).
_NON_EXCEPTION_BASES = frozenset({
    "BaseModel",
    "StrEnum",
    "Enum",
    "TypedDict",
    "Protocol",
})


def _load_allowlist() -> set[str]:
    """Parse the YAML allowlist into a set of dotted ``module.ClassName`` ids."""
    if not ALLOWLIST_PATH.exists():
        return set()
    raw = yaml.safe_load(ALLOWLIST_PATH.read_text(encoding="utf-8")) or {}
    entries = raw.get("allow", []) or []
    out: set[str] = set()
    for entry in entries:
        if isinstance(entry, dict) and "path" in entry:
            out.add(entry["path"])
        elif isinstance(entry, str):
            out.add(entry)
    return out


@lru_cache(maxsize=1)
def _collect_class_graph() -> dict[str, tuple[str, list[str], int]]:
    """Walk packages/*/src/ and harvest every class definition.

    Returns a dict keyed on ``"<dotted.module>.ClassName"``, value
    ``(module_dotted, [base_names], lineno)``. Bases are AST-level
    name strings (Name.id or Attribute.attr); unresolved bases keep
    their AST surface form.
    """
    graph: dict[str, tuple[str, list[str], int]] = {}
    if not PACKAGES_ROOT.exists():
        return graph
    for src_root in PACKAGES_ROOT.glob("*/src"):
        for py in src_root.rglob("*.py"):
            if "__pycache__" in py.parts:
                continue
            try:
                tree = ast.parse(py.read_text(encoding="utf-8"))
            except SyntaxError:
                continue
            rel = py.relative_to(PACKAGES_ROOT)
            # rel = <pkg>/src/<import_name>/.../<file>.py
            parts = list(rel.parts)
            if len(parts) < 4 or parts[1] != "src":
                continue
            mod_parts = parts[2:]
            mod_parts[-1] = mod_parts[-1].removesuffix(".py")
            dotted_mod = ".".join(mod_parts)
            for node in ast.walk(tree):
                if not isinstance(node, ast.ClassDef):
                    continue
                bases: list[str] = []
                for b in node.bases:
                    if isinstance(b, ast.Name):
                        bases.append(b.id)
                    elif isinstance(b, ast.Attribute):
                        bases.append(b.attr)
                    elif isinstance(b, ast.Subscript):
                        v = b.value
                        if isinstance(v, ast.Name):
                            bases.append(v.id)
                        elif isinstance(v, ast.Attribute):
                            bases.append(v.attr)
                full = f"{dotted_mod}.{node.name}"
                graph[full] = (dotted_mod, bases, node.lineno)
    return graph


def _index_by_local_name(graph: dict[str, tuple[str, list[str], int]]) -> dict[str, list[str]]:
    """For each bare class name, list the fully-qualified ids declaring it.

    Used to follow ``class FooError(BarError):`` where ``BarError`` is
    defined elsewhere in-tree. Ambiguous (multiple classes with the
    same name) -- we keep all candidates and consider the class OK if
    ANY candidate is rooted.
    """
    by_name: dict[str, list[str]] = {}
    for full, (_mod, _bases, _lineno) in graph.items():
        name = full.rsplit(".", 1)[-1]
        by_name.setdefault(name, []).append(full)
    return by_name


def _is_rooted_in_ryotenkai(
    full_id: str,
    graph: dict[str, tuple[str, list[str], int]],
    by_name: dict[str, list[str]],
    visited: set[str] | None = None,
) -> bool:
    """Return True if ``full_id`` transitively inherits from a RyotenkAI root.

    Walks the in-tree class graph by base **name** (no imports). Each
    base name is resolved by ``by_name`` lookup; ambiguous names match
    if any candidate is rooted.
    """
    if visited is None:
        visited = set()
    if full_id in visited:
        return False
    visited.add(full_id)
    entry = graph.get(full_id)
    if entry is None:
        return False
    _, bases, _ = entry
    for base in bases:
        if base in _RYOTENKAI_ROOT_NAMES:
            return True
        # Walk into in-tree definitions of the base name.
        for candidate in by_name.get(base, []):
            if candidate == full_id:
                continue
            if _is_rooted_in_ryotenkai(candidate, graph, by_name, visited):
                return True
    return False


def _collect_violations() -> list[str]:
    """Return human-readable violation lines for unallowlisted offenders."""
    graph = _collect_class_graph()
    by_name = _index_by_local_name(graph)
    allowlist = _load_allowlist()

    violations: list[str] = []
    for full, (_mod, bases, lineno) in graph.items():
        name = full.rsplit(".", 1)[-1]
        if not name.endswith("Error"):
            continue
        if not bases:
            continue
        # The hierarchy's own root is exempt -- it IS the root, so it
        # must inherit from ``Exception`` by design.
        if full == _HIERARCHY_ROOT_QUALNAME:
            continue
        # Skip clearly non-exception bases (BaseModel, StrEnum, ...).
        if all(b in _NON_EXCEPTION_BASES for b in bases):
            continue
        # If any base looks like a generic exception OR an in-tree
        # class whose name ends in Error, we require RyotenkAI rooting.
        looks_exception_like = False
        for b in bases:
            if b in _GENERIC_EXCEPTION_BASES:
                looks_exception_like = True
                break
            if b in _RYOTENKAI_ROOT_NAMES:
                # Already rooted at this level -- short-circuit OK.
                looks_exception_like = False
                break
            if b in by_name:
                # In-tree base name -- treat as exception-like (the
                # transitive walk below will give the final verdict).
                looks_exception_like = True
        if not looks_exception_like:
            continue
        if _is_rooted_in_ryotenkai(full, graph, by_name):
            continue
        if full in allowlist:
            continue
        base_str = ",".join(bases)
        violations.append(f"{full}:{lineno} base={base_str}")
    return sorted(violations)


def test_every_error_class_roots_in_ryotenkai_error() -> None:
    """Every ``*Error`` in production must root in :class:`RyotenkAIError`.

    Existing classes pre-Phase-A1 are pinned in the YAML allowlist.
    Adding a new ``Foo(Exception)`` raise site outside the allowlist
    is the failure mode this sentinel catches.
    """
    violations = _collect_violations()
    assert not violations, (
        "These ``*Error`` classes inherit from a non-RyotenkAI base and "
        "are NOT in tests/_lint/exception_root_allowlist.yaml. Either:\n"
        "  - reroot the class on RyotenkAIError / DomainError / "
        "InfrastructureError (preferred), or\n"
        "  - add an allowlist entry with a documented migration reason.\n\n"
        "Offenders:\n  " + "\n  ".join(violations)
    )


def test_allowlist_yaml_is_valid_and_non_empty() -> None:
    """Sanity-check the allowlist parses and is non-empty in Phase A1."""
    assert ALLOWLIST_PATH.exists(), f"Allowlist missing: {ALLOWLIST_PATH}"
    entries = _load_allowlist()
    assert entries, (
        "Phase A1 expects the allowlist to be seeded with pre-existing "
        "exception classes pending migration."
    )
    # Every entry should be a dotted path containing at least one '.'.
    for path in entries:
        assert "." in path, f"Malformed allowlist path: {path!r}"


def test_sentinel_catches_synthetic_violation(tmp_path: Path) -> None:
    """A fake offending class file must be reported by the scanner.

    This is the regression test that pins the sentinel's detection logic
    so refactors can't accidentally make it permissive.
    """
    # Build a tiny in-memory graph + verify _is_rooted_in_ryotenkai.
    fake_graph = {
        "fake_pkg.foo.BadError": ("fake_pkg.foo", ["Exception"], 1),
        "fake_pkg.foo.OkError": ("fake_pkg.foo", ["RyotenkAIError"], 2),
        "fake_pkg.foo.ChainedError": ("fake_pkg.foo", ["OkError"], 3),
        "fake_pkg.foo.UnrootedChainedError": ("fake_pkg.foo", ["BadError"], 4),
    }
    by_name = _index_by_local_name(fake_graph)
    assert not _is_rooted_in_ryotenkai("fake_pkg.foo.BadError", fake_graph, by_name)
    assert _is_rooted_in_ryotenkai("fake_pkg.foo.OkError", fake_graph, by_name)
    assert _is_rooted_in_ryotenkai("fake_pkg.foo.ChainedError", fake_graph, by_name)
    assert not _is_rooted_in_ryotenkai(
        "fake_pkg.foo.UnrootedChainedError", fake_graph, by_name
    )


def test_sentinel_ignores_basemodel_named_error(tmp_path: Path) -> None:
    """A ``FooError(BaseModel)`` must NOT be flagged (Pydantic models)."""
    fake_graph = {
        "fake_pkg.foo.PluginLoadError": ("fake_pkg.foo", ["BaseModel"], 1),
    }
    by_name = _index_by_local_name(fake_graph)
    # _is_rooted returns False (rightfully), but the per-class scan
    # skips this case because BaseModel is in _NON_EXCEPTION_BASES.
    # We verify via the public bases-resolver.
    bases = fake_graph["fake_pkg.foo.PluginLoadError"][1]
    assert all(b in _NON_EXCEPTION_BASES for b in bases)
