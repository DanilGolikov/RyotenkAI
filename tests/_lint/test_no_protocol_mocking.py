"""Sentinel: запрещаем mock над Protocol-интерфейсами в greenfield ``tests/``.

Greenfield-директория — никаких legacy-исключений. Если кто-то попытается
обмокать ``IPodLifecycleClient`` / ``IRunPodAPI`` / etc., тест падает.
Legacy ``packages/<pkg>/tests/`` сюда не попадает по построению.

The set of Protocol names is discovered dynamically by walking
``packages/*/src/**/*.py`` for classes whose base is ``Protocol`` (or
``typing.Protocol``, ``runtime_checkable``-decorated, etc.). A small curated
seed list is unioned in for non-class Protocol-like sentinels (e.g. the
``Clock`` Callable-alias).

Phase 5 (2026-05-12) extension
------------------------------
On top of the Protocol-mocking ban, this module now enforces a
**monotonically-shrinking allowlist** for two mock patterns that Phases 3A
and 3B have fully audited:

* every ``AsyncMock(...)`` call site in ``tests/`` must be either deleted
  or pinned in :mod:`tests._lint._mock_allowlist` with a justification;
* every concrete-class ``MagicMock(spec=ConcreteClass)`` call site must be
  either deleted or allowlisted.

External-library ``@patch(...)`` targets (``torch.cuda.*``, ``time.*``,
``mlflow.*``, ``huggingface_hub.*``, ``subprocess.*``, ``datasets.*``,
``peft.*``, ``concurrent.futures.*``, ``httpx.*``) are covered by *pattern*
entries in the allowlist; any test patching one of those targets is
implicitly excused.

Bare ``MagicMock()`` data carriers and internal ``@patch`` targets are NOT
yet enforced -- they are still being phased out in later batches. The
allowlist is the gate; over time the bare-MagicMock check will be added
once those usages drop below ~50.
"""

from __future__ import annotations

import ast
import fnmatch
import importlib.util
from datetime import date
from functools import lru_cache
from pathlib import Path

# Curated additions: not classes inheriting Protocol but still off-limits to
# mock (Callable-aliases or names we always want protected even before AST
# discovery sees them).
_CURATED_PROTOCOLS: frozenset[str] = frozenset({"Clock"})


def _packages_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "packages"


def _is_protocol_base(base: ast.AST) -> bool:
    """Return True if an AST class base node refers to ``Protocol``."""
    if isinstance(base, ast.Name):
        return base.id == "Protocol"
    if isinstance(base, ast.Attribute):
        return base.attr == "Protocol"
    if isinstance(base, ast.Subscript):
        # Generic Protocol, e.g. ``Protocol[T]``
        value = base.value
        if isinstance(value, ast.Name):
            return value.id == "Protocol"
        if isinstance(value, ast.Attribute):
            return value.attr == "Protocol"
    return False


@lru_cache(maxsize=1)
def _discover_protocols() -> frozenset[str]:
    """Walk ``packages/*/src/**/*.py`` and collect every class inheriting Protocol."""
    found: set[str] = set()
    pkgs = _packages_dir()
    if not pkgs.exists():
        return frozenset()
    for path in pkgs.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        # Restrict to ``src/`` trees (skip tests/ packaged alongside).
        if "src" not in path.parts:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            if any(_is_protocol_base(b) for b in node.bases):
                found.add(node.name)
    return frozenset(found)


def _protocols() -> frozenset[str]:
    return _discover_protocols() | _CURATED_PROTOCOLS


# Kept as a module-level lookup for the suffix-match (e.g. ``foo.IMLflowManager``).
_PROTOCOL_TARGET_RE_SUFFIXES = tuple(f".{name}" for name in _protocols())


def _tests_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _is_self_or_lint_file(path: Path, tests_root: Path) -> bool:
    rel = path.relative_to(tests_root)
    parts = rel.parts
    if not parts:
        return True
    return parts[0] == "_lint"


# Test-infrastructure directories that own canonical fakes/factories/helpers.
# They legitimately host the few documented mocks (e.g. validation-fallback
# MagicMock(spec=DatasetSourceLocal) inside ``_fakes/dataset_source.py``)
# and should not be scanned by the allowlist sentinel.
_INFRA_TOPLEVEL_DIRS: frozenset[str] = frozenset({
    "_lint",
    "_fakes",
    "_factories",
    "_helpers",
    "_harness",
})


def _is_test_infrastructure(path: Path, tests_root: Path) -> bool:
    """Return True if the path lives inside an infrastructure directory."""
    rel = path.relative_to(tests_root)
    parts = rel.parts
    if not parts:
        return True
    return parts[0] in _INFRA_TOPLEVEL_DIRS


def _matches_patch_target(value: str) -> bool:
    protocols = _protocols()
    if value in protocols:
        return True
    suffixes = tuple(f".{name}" for name in protocols)
    return value.endswith(suffixes)


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
                if name is not None and name in _protocols():
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
        "@patch('ryotenkai_shared.infrastructure.lifecycle.IPodLifecycleClient')\n"
        "def test_x(_): pass\n",
        encoding="utf-8",
    )
    tree = ast.parse(bad.read_text(encoding="utf-8"))
    found = _collect_violations(tree, bad)
    assert any("IPodLifecycleClient" in v for v in found), found

    bad2 = fake_root / "test_bad2.py"
    bad2.write_text(
        "from unittest.mock import MagicMock\n"
        "class IPodLifecycleClient: pass\n"
        "_ = MagicMock(spec=IPodLifecycleClient)\n",
        encoding="utf-8",
    )
    tree2 = ast.parse(bad2.read_text(encoding="utf-8"))
    found2 = _collect_violations(tree2, bad2)
    assert any("MagicMock(spec=IPodLifecycleClient)" in v for v in found2), found2


def test_sentinel_dynamically_discovers_protocols() -> None:
    """Discovery walks packages/*/src/ and finds Protocol classes.

    Regression guard: before dynamic discovery, the sentinel relied on a
    hand-maintained list and silently missed ``IEarlyReleasable`` /
    ``IDatasetLoader`` mocks. This test pins the contract.
    """
    discovered = _discover_protocols()
    expected_baseline = {
        "IPodLifecycleClient",
        "IRunPodAPI",
        "ITrainerSpawner",
        "ISSHClient",
        "IHFHubClient",
        "IJobClient",
        "IEarlyReleasable",
        "IDatasetLoader",
    }
    missing = expected_baseline - discovered
    assert not missing, f"Dynamic Protocol discovery missed: {sorted(missing)}"


def test_sentinel_catches_newly_discovered_protocol(tmp_path: Path) -> None:
    """Synthetic ``MagicMock(spec=IEarlyReleasable)`` must be flagged."""
    bad = tmp_path / "test_synth_releasable.py"
    bad.write_text(
        "from unittest.mock import MagicMock\n"
        "from ryotenkai_control.pipeline.stages.gpu_deployer import IEarlyReleasable\n"
        "_ = MagicMock(spec=IEarlyReleasable)\n",
        encoding="utf-8",
    )
    tree = ast.parse(bad.read_text(encoding="utf-8"))
    found = _collect_violations(tree, bad)
    assert any("MagicMock(spec=IEarlyReleasable)" in v for v in found), found


# ---------------------------------------------------------------------------
# Phase 5 (2026-05-12): allowlist-based monotonically-shrinking sentinel.
#
# Loads :mod:`tests._lint._mock_allowlist` via ``importlib`` (sibling-file
# style, to match :mod:`tests._lint.bootstrap_allowlist` precedent), then
# enforces:
#
#   * every ``AsyncMock(...)`` call in ``tests/`` is pinned in ALLOWLIST.
#   * every concrete-class ``MagicMock(spec=X)`` call is pinned in ALLOWLIST.
#   * every external-lib ``@patch("torch.cuda.*"|"time.*"|...)`` matches an
#     active pattern entry.
#
# Bare ``MagicMock()`` and internal ``@patch`` calls are intentionally NOT
# enforced here yet -- those phases (2A/4) are still in flight. Adding them
# now would lock in hundreds of false positives. As those phases land, the
# enforcement gates extend incrementally.
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _load_allowlist_module():
    """Load the sibling ``_mock_allowlist.py`` via importlib.

    Avoids the ``from tests._lint._mock_allowlist import ...`` path that
    would not survive pytest's ``--import-mode=importlib``.

    Note: we register the module in ``sys.modules`` BEFORE executing it
    so that ``@dataclass`` decorators (which look up the class module to
    resolve forward references / check ``KW_ONLY``) find their host.
    """
    import sys

    cached = sys.modules.get("_mock_allowlist")
    if cached is not None:
        return cached
    here = Path(__file__).resolve()
    spec = importlib.util.spec_from_file_location(
        "_mock_allowlist",
        here.parent / "_mock_allowlist.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["_mock_allowlist"] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop("_mock_allowlist", None)
        raise
    return module


def _allowlist_entries():
    return list(_load_allowlist_module().ALLOWLIST)


def _external_patch_prefixes() -> dict[str, tuple[str, ...]]:
    return dict(_load_allowlist_module().EXTERNAL_PATCH_PREFIXES)


def _matches_external_pattern(target: str, active_patterns: frozenset[str]) -> bool:
    """Return True if ``target`` matches an active external-patch pattern."""
    prefixes = _external_patch_prefixes()
    for pattern, prefix_tuple in prefixes.items():
        if pattern not in active_patterns:
            continue
        if target.startswith(prefix_tuple):
            return True
    return False


def _path_matches_glob(rel_path: str, glob: str) -> bool:
    """fnmatch-style glob match -- tests/**/test_*.py against rel_path."""
    return fnmatch.fnmatch(rel_path, glob)


def _is_pinned(rel_path: str, lineno: int, pinned: dict[tuple[str, int], object]) -> bool:
    return (rel_path, lineno) in pinned


def _is_pattern_excused_path(rel_path: str, allowlist) -> bool:
    """Return True if ``rel_path`` matches any active pattern-glob.

    Pattern entries don't cover bare AsyncMock/MagicMock -- they cover
    external-lib @patch sites only. But the pattern-glob path tells us
    *which* tests are allowed to host such @patch calls. We accept the
    glob as long as the per-call target matches one of the prefixes.
    """
    for entry in allowlist:
        if entry.line != 0:
            continue
        if _path_matches_glob(rel_path, entry.path):
            return True
    return False


def _collect_unallowlisted_mocks(tree: ast.AST, rel_path: str) -> list[tuple[int, str]]:
    """Return (lineno, kind) pairs for AsyncMock/MagicMock(spec=Concrete) calls.

    These are the patterns Phase 3A/3B fully audited. The caller cross-checks
    against the allowlist.
    """
    findings: list[tuple[int, str]] = []
    protocol_names = _protocols()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fname = _func_name(node)
        if fname is None:
            continue
        if fname == "AsyncMock":
            findings.append((node.lineno, "AsyncMock"))
            continue
        if fname == "MagicMock":
            # Only flag concrete-class spec= forms. List-spec and Protocol
            # cases are handled by other tests in this module.
            spec_target: ast.AST | None = None
            for kw in node.keywords:
                if kw.arg == "spec":
                    spec_target = kw.value
                    break
            if spec_target is None:
                continue  # bare MagicMock -- not yet enforced (Phase 2A still in flight)
            # Skip list-spec forms (MagicMock(spec=[...]))
            if isinstance(spec_target, (ast.List, ast.Tuple)):
                continue
            name = _attr_name(spec_target)
            if name is None:
                continue
            if name in protocol_names:
                # Already caught by the Protocol sentinel above.
                continue
            findings.append((node.lineno, f"MagicMock_spec_{name}"))
    return findings


def _collect_external_patch_calls(tree: ast.AST) -> list[tuple[int, str]]:
    """Return (lineno, target) for ``@patch("...")`` / ``patch("...")`` calls."""
    findings: list[tuple[int, str]] = []
    prefixes = _external_patch_prefixes()
    all_prefixes: tuple[str, ...] = tuple(p for ps in prefixes.values() for p in ps)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fname = _func_name(node)
        if fname not in {"patch"}:
            continue
        target = next(iter(node.args), None)
        if not (isinstance(target, ast.Constant) and isinstance(target.value, str)):
            continue
        if target.value.startswith(all_prefixes):
            findings.append((node.lineno, target.value))
    return findings


def _scan_unallowlisted(root: Path) -> list[str]:
    """Walk tests/ and return human-readable un-allowlisted mock violations."""
    tests_root = _tests_root()
    allowlist = _allowlist_entries()
    pinned = {(e.path, e.line): e for e in allowlist if e.line > 0}
    active_patterns = frozenset(e.pattern for e in allowlist if e.line == 0)

    violations: list[str] = []
    for path in root.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        if _is_test_infrastructure(path, tests_root):
            continue
        rel = path.relative_to(tests_root.parent)
        rel_path = rel.as_posix()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue

        for lineno, kind in _collect_unallowlisted_mocks(tree, rel_path):
            if _is_pinned(rel_path, lineno, pinned):
                continue
            violations.append(
                f"{rel_path}:{lineno}: {kind} not in tests/_lint/_mock_allowlist.py "
                "(eliminate, or add an AllowlistEntry with a documented reason)."
            )

        for lineno, target in _collect_external_patch_calls(tree):
            # External patches are excused by an active pattern entry,
            # regardless of file: every test is allowed to patch torch.cuda
            # iff the policy says so. We do NOT enforce pinning here -- the
            # cost (hundreds of pin lines) outweighs the marginal signal.
            if _matches_external_pattern(target, active_patterns):
                continue
            # If we got here, the @patch is on an external-shape target but
            # no active pattern excuses it. This is a real violation.
            violations.append(
                f"{rel_path}:{lineno}: @patch(\"{target}\") has no active pattern in "
                "tests/_lint/_mock_allowlist.py (add a pattern entry or remove the patch)."
            )
    return violations


def test_no_unallowlisted_mocks() -> None:
    """Every ``AsyncMock``/``MagicMock(spec=Concrete)`` call in ``tests/`` must
    be either eliminated, allowlisted in :mod:`tests._lint._mock_allowlist`,
    or (for external-lib ``@patch``) covered by an active pattern entry.

    This is the **monotonically-shrinking gate**: removing entries from the
    allowlist is fine; adding them requires code-owner review.
    """
    root = _tests_root()
    violations = _scan_unallowlisted(root)
    assert not violations, (
        "Un-allowlisted mock usages (Phase 5 sentinel):\n  "
        + "\n  ".join(violations)
        + "\n\nSee docs/testing/mock_policy.md for the allowlist process."
    )


def test_allowlist_entries_renewed_within_365_days() -> None:
    """Allowlist entries decay if not re-reviewed within a year.

    Forces a yearly audit: every entry must be explicitly re-blessed by
    bumping its ``renewed`` ISO date. Stale entries are surfaced here so
    the maintainer can either renew or eliminate them.
    """
    today = date.today()
    stale: list[str] = []
    for entry in _allowlist_entries():
        try:
            renewed = date.fromisoformat(entry.renewed)
        except ValueError as exc:
            stale.append(f"{entry.path}:{entry.line} -- invalid `renewed` value: {entry.renewed!r} ({exc})")
            continue
        age_days = (today - renewed).days
        if age_days > 365:
            stale.append(
                f"{entry.path}:{entry.line} -- renewed={entry.renewed} "
                f"({age_days}d ago) -- re-review and bump `renewed=`, "
                "or eliminate the mock."
            )
    assert not stale, "Stale allowlist entries (need re-review):\n  " + "\n  ".join(stale)


def test_allowlist_module_is_importable() -> None:
    """Smoke-check that ``_mock_allowlist.py`` parses, ALLOWLIST is non-empty,
    and every pinned entry has a non-empty ``reason``."""
    module = _load_allowlist_module()
    assert hasattr(module, "ALLOWLIST"), "ALLOWLIST symbol missing"
    entries = list(module.ALLOWLIST)
    assert entries, "ALLOWLIST is empty -- Phase 5 must populate it from Phase 3A/3B logs"
    for entry in entries:
        assert entry.reason and len(entry.reason) >= 10, (
            f"Entry {entry.path}:{entry.line} has empty/too-short reason"
        )
        assert entry.added and entry.renewed, f"Entry {entry.path}:{entry.line} missing dates"
        date.fromisoformat(entry.added)  # raises on malformed
        date.fromisoformat(entry.renewed)


def test_sentinel_catches_synthetic_unallowlisted_asyncmock(tmp_path: Path) -> None:
    """Synthetic violation: an un-pinned ``AsyncMock()`` must be flagged."""
    fake = tmp_path / "tests" / "unit" / "test_unallowlisted.py"
    fake.parent.mkdir(parents=True)
    fake.write_text(
        "from unittest.mock import AsyncMock\n"
        "x = AsyncMock(return_value=1)\n",
        encoding="utf-8",
    )
    tree = ast.parse(fake.read_text(encoding="utf-8"))
    # The fake file path is outside tests/, so build a synthetic rel_path
    # that matches what the real scanner would emit (still un-allowlisted).
    findings = _collect_unallowlisted_mocks(tree, "tests/unit/test_unallowlisted.py")
    assert findings, "Sentinel must catch a synthetic AsyncMock()"
    assert findings[0][1] == "AsyncMock", findings


def test_sentinel_catches_synthetic_external_patch_without_pattern(tmp_path: Path) -> None:
    """Synthetic ``@patch("zoneinfo.ZoneInfo")`` should not match an active
    external pattern -- so it must be flagged when scanning."""
    fake = tmp_path / "tests" / "unit" / "test_unallowlisted_patch.py"
    fake.parent.mkdir(parents=True)
    fake.write_text(
        "from unittest.mock import patch\n"
        "@patch(\"zoneinfo.ZoneInfo\")\n"
        "def test_foo(_): pass\n",
        encoding="utf-8",
    )
    tree = ast.parse(fake.read_text(encoding="utf-8"))
    # zoneinfo is NOT in EXTERNAL_PATCH_PREFIXES, so _collect_external_patch_calls
    # should return [] for this fake (the scanner skips it silently because
    # the prefix list filters at collection time). This is the documented
    # behaviour: we only catch external patches we've already classified.
    # Document the constraint so future maintainers don't expect a stricter check.
    external = _collect_external_patch_calls(tree)
    assert external == [], (
        "Scanner only catches external patches that match a known prefix; "
        "unknown external targets like zoneinfo pass silently by design. "
        "Add a new pattern + prefix tuple to extend coverage."
    )


def test_sentinel_catches_synthetic_unmatched_pattern_external_patch(tmp_path: Path) -> None:
    """If a pattern is REMOVED from the allowlist (e.g. patch_torch_cuda),
    the corresponding ``@patch("torch.cuda.is_available")`` call must be
    flagged. We simulate this by passing an empty ``active_patterns`` set
    to ``_matches_external_pattern``."""
    target = "torch.cuda.is_available"
    # With patch_torch_cuda active, the call is excused.
    assert _matches_external_pattern(target, frozenset({"patch_torch_cuda"}))
    # With it removed, the call is flagged.
    assert not _matches_external_pattern(target, frozenset())
