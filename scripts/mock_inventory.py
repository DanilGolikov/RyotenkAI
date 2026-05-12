#!/usr/bin/env python3
"""AST-based mock pattern inventory scanner for RyotenkAI tests/.

Walks tests/ recursively, parses each .py file, and emits structured
findings about unittest.mock usage. Designed to feed the downstream
conversion plan in docs/migration/mock_inventory.md.

Outputs:
    - CSV to stdout (or use --csv FILE to write)
    - Optionally, --md FILE writes the summary markdown.

Patterns detected:
    - MagicMock(...) — bare, with spec=, or with other kwargs
    - AsyncMock(...)  — bare or with spec=
    - Mock(...) and NonCallableMock(...)
    - create_autospec(...)
    - mock_open(...)
    - @patch("...") decorators
    - @patch.object(...) decorators
    - @patch.dict(...) decorators
    - with patch("...")  context-managers
    - with patch.object(...) context-managers
    - with patch.dict(...) context-managers
    - PropertyMock(...)
    - call(...) / ANY usage (informational)

Heuristics for categorization (per spec in the task):
    See ``categorize`` below.
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
TESTS_ROOT = REPO_ROOT / "tests"

# Pattern names (kept stable for the CSV / markdown).
P_MAGICMOCK_BARE = "MagicMock_bare"
P_MAGICMOCK_SPEC = "MagicMock_spec"
P_MAGICMOCK_OTHER = "MagicMock_other"  # MagicMock(return_value=...) etc, no spec
P_ASYNCMOCK_BARE = "AsyncMock_bare"
P_ASYNCMOCK_SPEC = "AsyncMock_spec"
P_ASYNCMOCK_OTHER = "AsyncMock_other"
P_MOCK_BARE = "Mock_bare"
P_MOCK_SPEC = "Mock_spec"
P_MOCK_OTHER = "Mock_other"
P_NONCALLABLEMOCK = "NonCallableMock"
P_PROPERTYMOCK = "PropertyMock"
P_CREATE_AUTOSPEC = "create_autospec"
P_MOCK_OPEN = "mock_open"
P_PATCH_DECORATOR = "patch_decorator"
P_PATCH_OBJECT_DECORATOR = "patch_object_decorator"
P_PATCH_DICT_DECORATOR = "patch_dict_decorator"
P_PATCH_MULTIPLE_DECORATOR = "patch_multiple_decorator"
P_PATCH_CM = "patch_context_manager"
P_PATCH_OBJECT_CM = "patch_object_context_manager"
P_PATCH_DICT_CM = "patch_dict_context_manager"
P_PATCH_MULTIPLE_CM = "patch_multiple_context_manager"
P_PATCH_START = "patch_start"  # x = patch(...); x.start()
# Informational / non-counted toward conversions but useful:
P_MOCK_ANY = "ANY"
P_MOCK_CALL = "call"


@dataclass
class Hit:
    file: str  # repo-relative
    pattern: str
    target: str  # spec= class name, patch path, etc. "" if N/A.
    line_number: int
    category: str  # ACCEPTABLE / MECHANICAL / JUDGMENT / REFACTOR / INFO
    rationale: str = ""


# -- Heuristic helpers ---------------------------------------------------------


# Lightweight checks: we don't import the production modules to inspect them,
# we use lexical signals (suffix Protocol, dataclass / BaseModel naming, etc.).
PYDANTIC_HINTS = {
    # known Pydantic models in the project (extend by inspecting types if needed)
    "RunInfo",
    "RunSpec",
    "TrainingConfig",
    "EvaluationConfig",
    "EvaluationResults",
    "DatasetSpec",
    "DeploymentResult",
    "ResolvedPodSpec",
    "PhaseResult",
    "PlanAction",
    "PostmortemReport",
    "PreparedModel",
    "PhaseConfig",
    "PhaseSpec",
}

PROTOCOL_HINTS_SUFFIX = ("Protocol", "Client")
PROTOCOL_HINTS_PREFIX = ("I",)


def classify_spec_target(name: str) -> str:
    """Classify spec=X target by name heuristics."""
    if not name:
        return "unknown"
    # spec=[...] list literal — spec_set / attribute names only.
    if name.startswith("[") and name.endswith("]"):
        return "list-spec"
    if name in PYDANTIC_HINTS:
        return "pydantic"
    # Crude: names starting with capital I followed by capital letter are
    # often Protocols (IPodLifecycleClient, IMLflowManager). Endswith
    # 'Protocol' is also strong.
    if name.endswith("Protocol"):
        return "protocol"
    if (
        len(name) >= 2
        and name[0] == "I"
        and name[1].isupper()
        and not name.startswith("Iso")  # avoid false positives
    ):
        return "protocol"
    # External standard libs frequently used as spec=
    if name.startswith(("threading.", "subprocess.", "asyncio.", "queue.")):
        return "external"
    return "concrete"


# Things considered "external libs" when patched.
EXTERNAL_PREFIXES = (
    "subprocess",
    "time",
    "os.",
    "pathlib",
    "shutil",
    "tempfile",
    "socket",
    "requests",
    "httpx",
    "urllib",
    "boto3",
    "openai",
    "anthropic",
    "mlflow",
    "logging",
    "random",
    "datetime",
    "uuid",
    "hashlib",
    "json",
    "sys",
    "io",
    "asyncio",
    "threading",
    "multiprocessing",
)


def classify_patch_target(path: str) -> str:
    """Classify @patch("a.b.c") target string."""
    if not path:
        return "unknown"
    head = path.split(".")[0]
    if head in {"subprocess", "time", "os", "shutil", "logging", "random", "uuid",
                "datetime", "tempfile", "socket", "requests", "httpx", "urllib",
                "boto3", "openai", "anthropic", "mlflow", "asyncio", "threading",
                "multiprocessing", "io", "sys", "pathlib", "hashlib", "json",
                "torch", "huggingface_hub", "transformers", "datasets",
                "numpy", "pandas", "yaml", "tomllib", "tomlkit", "pydantic",
                "fastapi", "starlette", "httpcore", "paramiko", "fabric"}:
        return "external"
    # Internal: starts with one of our package names.
    if head.startswith("ryotenkai_") or head.startswith("packages"):
        return "internal"
    # ``community.``, ``tests.``, etc — internal-ish.
    if head in {"community", "tests", "scripts", "src"}:
        return "internal"
    # Sometimes patched as plain module path inside the test (e.g.
    # ``module._private`` where module is the SUT). Treat as internal.
    return "internal_or_unknown"


def categorize(pattern: str, target: str, src_line: str | None = None) -> tuple[str, str]:
    """Return (category, rationale) for the canonical pattern + target."""
    # External I/O patches → ACCEPTABLE
    if pattern in {
        P_PATCH_DECORATOR,
        P_PATCH_CM,
    }:
        cls = classify_patch_target(target)
        if cls == "external":
            return ("ACCEPTABLE", "external syscall/IO/log/clock — legit boundary")
        # Internal: depends on what's patched.
        # Private function or _underscore method → REFACTOR (test smell).
        last = target.rsplit(".", 1)[-1]
        if last.startswith("_"):
            return ("REFACTOR", "patches private internal — production needs DI")
        # Patching a Protocol/Client interface inside SUT → already banned by sentinel
        if last.endswith("Protocol") or (len(last) > 1 and last[0] == "I" and last[1].isupper()):
            return ("REFACTOR", "patches Protocol; should use Fake")
        # Otherwise: a module-level constant or class; needs per-file decision.
        return ("JUDGMENT", "internal target; consider dep-injection or Fake")
    if pattern == P_PATCH_OBJECT_DECORATOR or pattern == P_PATCH_OBJECT_CM:
        # patch.object(x, "y") — often a code smell when y is private
        if target.startswith("_") or "._" in target:
            return ("REFACTOR", "patch.object on private attr — test smell")
        return ("JUDGMENT", "patch.object — usually replaceable with Fake")
    if pattern == P_PATCH_DICT_DECORATOR or pattern == P_PATCH_DICT_CM:
        if target.startswith("os.environ") or target == "os.environ":
            return ("MECHANICAL", "patch.dict(os.environ) -> monkeypatch.setenv()")
        return ("MECHANICAL", "patch.dict -> targeted fixture / setattr")
    if pattern == P_MAGICMOCK_BARE:
        return ("MECHANICAL", "bare MagicMock data carrier -> SimpleNamespace")
    if pattern == P_MAGICMOCK_OTHER:
        return ("MECHANICAL", "MagicMock(return_value=...) -> small Fake or factory")
    if pattern == P_MAGICMOCK_SPEC:
        cls = classify_spec_target(target)
        if cls == "pydantic":
            return ("MECHANICAL", "MagicMock(spec=PydanticModel) -> real PydanticModel(...)")
        if cls == "protocol":
            return ("REFACTOR", "MagicMock(spec=Protocol) -> Fake implementation")
        return ("JUDGMENT", "MagicMock(spec=ConcreteClass) -> Fake or real construct")
    if pattern == P_ASYNCMOCK_BARE:
        return ("JUDGMENT", "AsyncMock — prefer Fake async client when available")
    if pattern == P_ASYNCMOCK_OTHER:
        return ("JUDGMENT", "AsyncMock(return_value=...) — prefer Fake async client")
    if pattern == P_ASYNCMOCK_SPEC:
        cls = classify_spec_target(target)
        if cls == "protocol":
            return ("REFACTOR", "AsyncMock(spec=Protocol) -> Fake async impl")
        return ("JUDGMENT", "AsyncMock(spec=...) — prefer Fake async client")
    if pattern == P_MOCK_BARE:
        return ("MECHANICAL", "Mock() bare -> SimpleNamespace")
    if pattern == P_MOCK_OTHER:
        return ("MECHANICAL", "Mock(return_value=...) -> small Fake")
    if pattern == P_MOCK_SPEC:
        return categorize(P_MAGICMOCK_SPEC, target, src_line)
    if pattern == P_NONCALLABLEMOCK:
        return ("MECHANICAL", "NonCallableMock -> SimpleNamespace")
    if pattern == P_PROPERTYMOCK:
        return ("JUDGMENT", "PropertyMock — rewrite via @property in Fake")
    if pattern == P_CREATE_AUTOSPEC:
        return ("JUDGMENT", "create_autospec -> Fake or real construct")
    if pattern == P_MOCK_OPEN:
        return ("MECHANICAL", "mock_open -> tmp_path file fixture")
    if pattern == P_PATCH_START:
        return ("REFACTOR", "patch().start() leaks; replace with Fake or fixture")
    if pattern == P_PATCH_MULTIPLE_DECORATOR or pattern == P_PATCH_MULTIPLE_CM:
        return ("REFACTOR", "patch.multiple — heavy mocking; refactor with DI")
    if pattern in (P_MOCK_ANY, P_MOCK_CALL):
        return ("INFO", "ANY/call — assertion helpers, not a mock object")
    return ("JUDGMENT", "unknown pattern")


# -- AST extraction ------------------------------------------------------------


def get_call_fullname(node: ast.AST) -> str:
    """Return dotted name for Attribute/Name nodes; '' for others."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = get_call_fullname(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return ""


def literal_str(node: ast.AST) -> str | None:
    """Extract string literal if node is a Constant str."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def first_arg_class_name(call: ast.Call) -> str:
    """For MagicMock(spec=X) extract X's name from kwarg spec=, else from arg[0]
    if it looks like spec usage."""
    for kw in call.keywords:
        if kw.arg == "spec":
            return get_call_fullname(kw.value) or _repr(kw.value)
    return ""


def _repr(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:
        return f"<{type(node).__name__}>"


@dataclass
class FileScan:
    rel_path: str
    hits: list[Hit] = field(default_factory=list)
    has_unittest_mock_import: bool = False


def _patch_subform(call: ast.Call) -> str | None:
    """Distinguish patch / patch.object / patch.dict / patch.multiple.

    Returns one of: 'patch', 'patch.object', 'patch.dict', 'patch.multiple'
    if call.func resolves to that. Else None.
    """
    name = get_call_fullname(call.func)
    # Strip leading module if `from unittest import mock; mock.patch.dict(...)`.
    # Examples we may see:
    #   patch / patch.object / patch.dict / patch.multiple
    #   mock.patch / mock.patch.object / ...
    parts = name.split(".")
    if not parts:
        return None
    if "patch" not in parts:
        return None
    # find the patch token and look at what follows
    idx = parts.index("patch")
    tail = parts[idx + 1:] if len(parts) > idx + 1 else []
    if not tail:
        return "patch"
    head = tail[0]
    if head in {"object", "dict", "multiple"}:
        return f"patch.{head}"
    return "patch"  # something like patch().start


def _patch_target_string(call: ast.Call, subform: str) -> str:
    """Extract a useful target string for the CSV.

    - patch("a.b.c") → "a.b.c"
    - patch.object(obj, "method") → "obj.method" (textual)
    - patch.dict(os.environ, {...}) → "os.environ"
    - patch.multiple("a.b", ...) → "a.b"
    """
    if subform in {"patch", "patch.multiple"}:
        if call.args:
            s = literal_str(call.args[0])
            if s is not None:
                return s
            return _repr(call.args[0])[:160]
        return ""
    if subform == "patch.object":
        if len(call.args) >= 1:
            target_obj = _repr(call.args[0])
            attr = ""
            if len(call.args) >= 2:
                attr = literal_str(call.args[1]) or _repr(call.args[1])
            return f"{target_obj}.{attr}" if attr else target_obj
        return ""
    if subform == "patch.dict":
        if call.args:
            return _repr(call.args[0])[:160]
        return ""
    return ""


def scan_file(path: Path) -> FileScan:
    rel = str(path.relative_to(REPO_ROOT))
    fs = FileScan(rel_path=rel)
    try:
        source = path.read_text(encoding="utf-8")
    except Exception:
        return fs
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return fs

    # Imports — flag unittest.mock usage.
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module and node.module.startswith("unittest"):
                fs.has_unittest_mock_import = True
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "unittest" or alias.name.startswith("unittest"):
                    fs.has_unittest_mock_import = True

    # Walk for calls and decorators.
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            _handle_call(node, fs)
        # Decorators on functions / async functions / classes.
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            for dec in node.decorator_list:
                _handle_decorator(dec, fs)
        # ``with patch(...) as x:`` is handled via Call walk (the call itself
        # is in the With node's items[].context_expr). But Call walk visits
        # those too, so we don't double-count.

    return fs


def _handle_call(call: ast.Call, fs: FileScan) -> None:
    fname = get_call_fullname(call.func)
    if not fname:
        return
    last = fname.split(".")[-1]

    # --- Mock constructors ---------------------------------------------------
    if last == "MagicMock":
        target = first_arg_class_name(call)
        if target:
            pattern = P_MAGICMOCK_SPEC
        elif not call.args and not call.keywords:
            pattern = P_MAGICMOCK_BARE
        else:
            pattern = P_MAGICMOCK_OTHER
        _add(fs, pattern, target, call.lineno)
        return
    if last == "AsyncMock":
        target = first_arg_class_name(call)
        if target:
            pattern = P_ASYNCMOCK_SPEC
        elif not call.args and not call.keywords:
            pattern = P_ASYNCMOCK_BARE
        else:
            pattern = P_ASYNCMOCK_OTHER
        _add(fs, pattern, target, call.lineno)
        return
    if last == "Mock":
        # Could be unittest.mock.Mock OR pytest_mock's Mock — assume the former
        # when unittest is imported in this file (filtered downstream too).
        target = first_arg_class_name(call)
        if target:
            pattern = P_MOCK_SPEC
        elif not call.args and not call.keywords:
            pattern = P_MOCK_BARE
        else:
            pattern = P_MOCK_OTHER
        _add(fs, pattern, target, call.lineno)
        return
    if last == "NonCallableMock":
        _add(fs, P_NONCALLABLEMOCK, first_arg_class_name(call), call.lineno)
        return
    if last == "PropertyMock":
        _add(fs, P_PROPERTYMOCK, "", call.lineno)
        return
    if last == "create_autospec":
        target = ""
        if call.args:
            target = get_call_fullname(call.args[0]) or _repr(call.args[0])
        _add(fs, P_CREATE_AUTOSPEC, target, call.lineno)
        return
    if last == "mock_open":
        _add(fs, P_MOCK_OPEN, "", call.lineno)
        return

    # --- patch / patch.object / patch.dict / patch.multiple ------------------
    subform = _patch_subform(call)
    if subform is None:
        return
    target = _patch_target_string(call, subform)
    if subform == "patch":
        _add(fs, P_PATCH_CM, target, call.lineno)
    elif subform == "patch.object":
        _add(fs, P_PATCH_OBJECT_CM, target, call.lineno)
    elif subform == "patch.dict":
        _add(fs, P_PATCH_DICT_CM, target, call.lineno)
    elif subform == "patch.multiple":
        _add(fs, P_PATCH_MULTIPLE_CM, target, call.lineno)


def _handle_decorator(dec: ast.AST, fs: FileScan) -> None:
    # Decorators may be Call (e.g. @patch("...")) or Attribute / Name.
    if isinstance(dec, ast.Call):
        fname = get_call_fullname(dec.func)
        if not fname:
            return
        if "patch" not in fname.split("."):
            return
        subform = _patch_subform(dec)
        if subform is None:
            return
        target = _patch_target_string(dec, subform)
        # Re-tag CM hits emitted by the Call walk as decorators if line matches.
        # Simpler: emit a separate "decorator" hit and rely on dedup downstream
        # (we'll dedup by line+pattern when computing final counts).
        if subform == "patch":
            _add_dedup(fs, P_PATCH_DECORATOR, target, dec.lineno,
                       supersedes=P_PATCH_CM)
        elif subform == "patch.object":
            _add_dedup(fs, P_PATCH_OBJECT_DECORATOR, target, dec.lineno,
                       supersedes=P_PATCH_OBJECT_CM)
        elif subform == "patch.dict":
            _add_dedup(fs, P_PATCH_DICT_DECORATOR, target, dec.lineno,
                       supersedes=P_PATCH_DICT_CM)
        elif subform == "patch.multiple":
            _add_dedup(fs, P_PATCH_MULTIPLE_DECORATOR, target, dec.lineno,
                       supersedes=P_PATCH_MULTIPLE_CM)


def _add(fs: FileScan, pattern: str, target: str, lineno: int) -> None:
    cat, why = categorize(pattern, target)
    fs.hits.append(Hit(file=fs.rel_path, pattern=pattern, target=target,
                      line_number=lineno, category=cat, rationale=why))


def _add_dedup(fs: FileScan, pattern: str, target: str, lineno: int,
               supersedes: str) -> None:
    """Append a decorator hit and remove the matching CM hit at same line."""
    # Remove duplicate CM hit at same line (the Call walk has already added it).
    for i, h in enumerate(fs.hits):
        if h.line_number == lineno and h.pattern == supersedes:
            fs.hits.pop(i)
            break
    _add(fs, pattern, target, lineno)


# -- Aggregate + report --------------------------------------------------------


def collect() -> tuple[list[FileScan], list[Hit]]:
    scans: list[FileScan] = []
    all_hits: list[Hit] = []
    py_files = sorted(TESTS_ROOT.rglob("*.py"))
    for p in py_files:
        # Skip __pycache__ — rglob already skips dirs starting with __pycache__
        # only via name patterns, not contents.
        if "__pycache__" in p.parts:
            continue
        scan = scan_file(p)
        if not scan.has_unittest_mock_import:
            # Still scan, but only count hits we found (rare to have raw
            # MagicMock without import — likely re-export). Keep them.
            pass
        if scan.hits or scan.has_unittest_mock_import:
            scans.append(scan)
            all_hits.extend(scan.hits)
    return scans, all_hits


def write_csv(hits: Iterable[Hit], out_path: Path) -> None:
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "pattern", "target", "line_number",
                    "suggested_category", "rationale"])
        for h in sorted(hits, key=lambda x: (x.file, x.line_number)):
            w.writerow([h.file, h.pattern, h.target, h.line_number,
                        h.category, h.rationale])


def percentile_buckets(counts: list[int]) -> dict[str, int]:
    if not counts:
        return {}
    counts = sorted(counts)
    return {
        "min": counts[0],
        "p50": counts[len(counts) // 2],
        "p90": counts[int(len(counts) * 0.9)],
        "max": counts[-1],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path,
                        default=REPO_ROOT / "docs/migration/mock_inventory.csv")
    parser.add_argument("--md", type=Path,
                        default=REPO_ROOT / "docs/migration/mock_inventory.md")
    parser.add_argument("--stdout", action="store_true",
                        help="Also write CSV to stdout")
    args = parser.parse_args()

    scans, hits = collect()

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    write_csv(hits, args.csv)

    # Aggregates.
    pattern_counts: Counter[str] = Counter(h.pattern for h in hits)
    category_counts: Counter[str] = Counter(h.category for h in hits)
    per_file: dict[str, Counter[str]] = defaultdict(Counter)
    for h in hits:
        per_file[h.file][h.pattern] += 1

    # Top spec targets.
    spec_targets: Counter[str] = Counter()
    for h in hits:
        if h.pattern in {P_MAGICMOCK_SPEC, P_ASYNCMOCK_SPEC, P_MOCK_SPEC}:
            spec_targets[h.target] += 1
    spec_targets.pop("", None)

    # Top patch targets.
    patch_targets: Counter[str] = Counter()
    for h in hits:
        if h.pattern in {P_PATCH_DECORATOR, P_PATCH_CM}:
            patch_targets[h.target] += 1
    patch_object_targets: Counter[str] = Counter()
    for h in hits:
        if h.pattern in {P_PATCH_OBJECT_DECORATOR, P_PATCH_OBJECT_CM}:
            patch_object_targets[h.target] += 1

    # Top files.
    file_totals = [(f, sum(c.values())) for f, c in per_file.items()]
    file_totals.sort(key=lambda x: x[1], reverse=True)

    # Mechanical conversion leverage: count × ease (ease = 3 for trivially
    # mechanical, 2 for moderate, 1 for hard). We map by pattern.
    EASE = {
        P_MAGICMOCK_BARE: 3,
        P_PATCH_DICT_DECORATOR: 3,
        P_PATCH_DICT_CM: 3,
        P_MAGICMOCK_OTHER: 2,
        P_MOCK_BARE: 3,
        P_MOCK_OTHER: 2,
        P_MOCK_OPEN: 3,
        P_NONCALLABLEMOCK: 3,
        P_MAGICMOCK_SPEC: 2,
        P_MOCK_SPEC: 2,
        P_ASYNCMOCK_BARE: 1,
        P_ASYNCMOCK_OTHER: 1,
        P_ASYNCMOCK_SPEC: 1,
        P_PATCH_DECORATOR: 1,
        P_PATCH_CM: 1,
        P_PATCH_OBJECT_DECORATOR: 1,
        P_PATCH_OBJECT_CM: 1,
        P_PROPERTYMOCK: 1,
        P_CREATE_AUTOSPEC: 1,
        P_PATCH_MULTIPLE_DECORATOR: 1,
        P_PATCH_MULTIPLE_CM: 1,
    }
    leverage: list[tuple[str, int, int, int]] = []  # pattern, count, ease, score
    for pat, cnt in pattern_counts.items():
        ease = EASE.get(pat, 1)
        leverage.append((pat, cnt, ease, cnt * ease))
    leverage.sort(key=lambda x: x[3], reverse=True)

    # Refactor candidates: internal patch targets that appear most often.
    refactor_targets: Counter[str] = Counter()
    for h in hits:
        if h.pattern in {P_PATCH_DECORATOR, P_PATCH_CM} and h.category == "REFACTOR":
            refactor_targets[h.target] += 1
        if h.pattern in {P_PATCH_OBJECT_DECORATOR, P_PATCH_OBJECT_CM} and h.category == "REFACTOR":
            refactor_targets[h.target] += 1

    # --- Write markdown ----------------------------------------------------
    md = render_markdown(
        scans=scans,
        hits=hits,
        pattern_counts=pattern_counts,
        category_counts=category_counts,
        per_file=per_file,
        spec_targets=spec_targets,
        patch_targets=patch_targets,
        patch_object_targets=patch_object_targets,
        file_totals=file_totals,
        leverage=leverage,
        refactor_targets=refactor_targets,
    )
    args.md.parent.mkdir(parents=True, exist_ok=True)
    args.md.write_text(md, encoding="utf-8")

    # Summary to stdout.
    print(f"Scanned {len(scans)} test files with hits.")
    print(f"Total hits: {sum(pattern_counts.values())}")
    print(f"CSV written to: {args.csv}")
    print(f"MD written to:  {args.md}")
    print()
    print("Pattern totals:")
    for pat, cnt in pattern_counts.most_common():
        print(f"  {pat:32s} {cnt}")
    print()
    print("Category totals:")
    for cat, cnt in category_counts.most_common():
        print(f"  {cat:14s} {cnt}")

    return 0


def _bar(n: int, max_n: int, width: int = 24) -> str:
    if max_n <= 0:
        return ""
    filled = int(round(n / max_n * width))
    return "█" * filled + "·" * (width - filled)


def render_markdown(
    *,
    scans: list[FileScan],
    hits: list[Hit],
    pattern_counts: Counter[str],
    category_counts: Counter[str],
    per_file: dict[str, Counter[str]],
    spec_targets: Counter[str],
    patch_targets: Counter[str],
    patch_object_targets: Counter[str],
    file_totals: list[tuple[str, int]],
    leverage: list[tuple[str, int, int, int]],
    refactor_targets: Counter[str],
) -> str:
    total_hits = sum(pattern_counts.values())
    n_files = len(scans)
    max_pat = pattern_counts.most_common(1)[0][1] if pattern_counts else 1

    out: list[str] = []
    out.append("# Mock Pattern Inventory (RyotenkAI tests/)")
    out.append("")
    out.append("Generated by `scripts/mock_inventory.py` (AST-based).")
    out.append("")
    out.append("## Executive Summary")
    out.append("")
    out.append(f"- Files with `unittest.mock` usage: **{n_files}**")
    out.append(f"- Total mock-pattern hits across `tests/`: **{total_hits}**")
    out.append("")
    out.append("### Top-level pattern totals")
    out.append("")
    out.append("| Pattern | Count | Share |")
    out.append("|---|---:|---|")
    for pat, cnt in pattern_counts.most_common():
        out.append(f"| `{pat}` | {cnt} | {_bar(cnt, max_pat)} |")
    out.append("")
    out.append("### Category breakdown")
    out.append("")
    out.append("| Category | Count |")
    out.append("|---|---:|")
    for cat, cnt in category_counts.most_common():
        out.append(f"| {cat} | {cnt} |")
    out.append("")
    out.append("Definitions:")
    out.append("")
    out.append("- **ACCEPTABLE** — legitimate interaction test or pytest-native")
    out.append("  pattern; keep as-is.")
    out.append("- **MECHANICAL** — high-volume, bulk-convertible (e.g. bare")
    out.append("  `MagicMock()` data carriers, `patch.dict(os.environ)`).")
    out.append("- **JUDGMENT** — needs per-file decision (Fake vs real construct).")
    out.append("- **REFACTOR** — test smell that signals a production change")
    out.append("  (dep injection) is needed.")
    out.append("")

    # Top 50 files
    out.append("## Top 50 files by mock-pattern count")
    out.append("")
    cols = [
        P_MAGICMOCK_BARE,
        P_MAGICMOCK_SPEC,
        P_MAGICMOCK_OTHER,
        P_ASYNCMOCK_BARE,
        P_ASYNCMOCK_OTHER,
        P_ASYNCMOCK_SPEC,
        P_PATCH_DECORATOR,
        P_PATCH_OBJECT_DECORATOR,
        P_PATCH_DICT_DECORATOR,
        P_PATCH_CM,
        P_PATCH_OBJECT_CM,
        P_PATCH_DICT_CM,
        P_CREATE_AUTOSPEC,
    ]
    header = "| file | total | " + " | ".join(c for c in cols) + " |"
    out.append(header)
    out.append("|" + "|".join(["---"] * (2 + len(cols))) + "|")
    for f, total in file_totals[:50]:
        c = per_file[f]
        row = [f"`{f}`", str(total)]
        for col in cols:
            row.append(str(c.get(col, 0)) or "·")
        out.append("| " + " | ".join(row) + " |")
    out.append("")

    # Top spec targets
    out.append("## Top spec= targets")
    out.append("")
    out.append("| Class | Count | Kind |")
    out.append("|---|---:|---|")
    for cls, cnt in spec_targets.most_common(30):
        kind = classify_spec_target(cls.split(".")[-1])
        out.append(f"| `{cls}` | {cnt} | {kind} |")
    out.append("")

    # Top patch targets
    out.append("## Top `@patch(...)` / `patch(...)` targets")
    out.append("")
    out.append("| Path | Count | Kind |")
    out.append("|---|---:|---|")
    for path, cnt in patch_targets.most_common(30):
        kind = classify_patch_target(path)
        out.append(f"| `{path}` | {cnt} | {kind} |")
    out.append("")
    out.append("### Top `patch.object(...)` targets")
    out.append("")
    out.append("| Target | Count |")
    out.append("|---|---:|")
    for target, cnt in patch_object_targets.most_common(30):
        out.append(f"| `{target}` | {cnt} |")
    out.append("")

    # Mechanical leverage
    out.append("## Mechanical-conversion leverage (top 10)")
    out.append("")
    out.append("Score = count × ease (ease ∈ {1,2,3} per the heuristics in")
    out.append("`scripts/mock_inventory.py::EASE`).")
    out.append("")
    out.append("| Pattern | Count | Ease | Score |")
    out.append("|---|---:|---:|---:|")
    for pat, cnt, ease, score in leverage[:10]:
        out.append(f"| `{pat}` | {cnt} | {ease} | {score} |")
    out.append("")

    # Refactor candidates
    out.append("## Refactor candidates (top 20)")
    out.append("")
    out.append("These are internal `@patch(...)` / `patch.object(...)` targets")
    out.append("classified as REFACTOR — production likely needs dependency")
    out.append("injection.")
    out.append("")
    out.append("| Target | Count |")
    out.append("|---|---:|")
    for tgt, cnt in refactor_targets.most_common(20):
        out.append(f"| `{tgt}` | {cnt} |")
    out.append("")

    # Suggested batches.
    out.append("## Suggested batch plan for downstream conversion")
    out.append("")
    out.append("Ordered by leverage (count × ease). Each batch ≈ 200–400 hits.")
    out.append("")
    batches = suggest_batches(pattern_counts, hits)
    for i, (label, count, what) in enumerate(batches, 1):
        out.append(f"{i}. **{label}** — ~{count} hits — {what}")
    out.append("")

    out.append("## How to reproduce")
    out.append("")
    out.append("```bash")
    out.append("python scripts/mock_inventory.py \\")
    out.append("    --csv docs/migration/mock_inventory.csv \\")
    out.append("    --md  docs/migration/mock_inventory.md")
    out.append("```")
    out.append("")
    return "\n".join(out)


def suggest_batches(pattern_counts: Counter[str], hits: list[Hit]):
    """Suggest mechanical conversion batches grouped by pattern + target."""
    batches: list[tuple[str, int, str]] = []
    # Batch 1: bare MagicMock → SimpleNamespace
    n = pattern_counts.get(P_MAGICMOCK_BARE, 0)
    if n:
        batches.append((
            "Batch 1: bare `MagicMock()` → `SimpleNamespace`",
            n,
            "Mechanical. Group by file; each replacement is local.",
        ))
    # Batch 2: MagicMock(spec=Pydantic) → real model
    n2 = sum(1 for h in hits
             if h.pattern == P_MAGICMOCK_SPEC
             and classify_spec_target(h.target.split(".")[-1]) == "pydantic")
    if n2:
        batches.append((
            "Batch 2: `MagicMock(spec=PydanticModel)` → real model factory",
            n2,
            "Mechanical. Build factories in tests/_fixtures.",
        ))
    # Batch 3: MagicMock(...) with kwargs → small Fake
    n3 = pattern_counts.get(P_MAGICMOCK_OTHER, 0)
    if n3:
        batches.append((
            "Batch 3: `MagicMock(return_value=...)` → tiny Fake or factory",
            n3,
            "Mechanical-ish. Replace per-file with a small dataclass.",
        ))
    # Batch 4: patch.dict(os.environ) → monkeypatch.setenv
    n4 = pattern_counts.get(P_PATCH_DICT_DECORATOR, 0) + pattern_counts.get(P_PATCH_DICT_CM, 0)
    if n4:
        batches.append((
            "Batch 4: `patch.dict(os.environ, ...)` → `monkeypatch.setenv`",
            n4,
            "Mechanical. Pure rewrite.",
        ))
    # Batch 5: MagicMock(spec=Concrete) → Fake
    n5 = sum(1 for h in hits
             if h.pattern == P_MAGICMOCK_SPEC
             and classify_spec_target(h.target.split(".")[-1]) == "concrete")
    if n5:
        batches.append((
            "Batch 5: `MagicMock(spec=ConcreteClass)` → Fake or real construct",
            n5,
            "Judgment per class. Build Fakes in tests/_fakes.",
        ))
    # Batch 6: AsyncMock → Fake async client
    nA = (pattern_counts.get(P_ASYNCMOCK_BARE, 0)
          + pattern_counts.get(P_ASYNCMOCK_OTHER, 0)
          + pattern_counts.get(P_ASYNCMOCK_SPEC, 0))
    if nA:
        batches.append((
            "Batch 6: `AsyncMock` → Fake async client",
            nA,
            "Judgment. Group by client interface.",
        ))
    # Batch 7: @patch external (clock/subprocess) → injection or fixture
    nExt = sum(1 for h in hits
               if h.pattern in {P_PATCH_DECORATOR, P_PATCH_CM}
               and classify_patch_target(h.target) == "external")
    if nExt:
        batches.append((
            "Batch 7: `@patch` on external libs (time/subprocess/etc)",
            nExt,
            "Acceptable to keep; or migrate to ManualClock / fixtures.",
        ))
    # Batch 8: @patch internal → REFACTOR (dep injection in production).
    nRef = sum(1 for h in hits
               if h.pattern in {P_PATCH_DECORATOR, P_PATCH_CM,
                                P_PATCH_OBJECT_DECORATOR, P_PATCH_OBJECT_CM}
               and h.category == "REFACTOR")
    if nRef:
        batches.append((
            "Batch 8: `@patch` internal private fns / Protocols (REFACTOR)",
            nRef,
            "Requires production dep-injection. Smaller per change but more risk.",
        ))
    # Batch 9: patch.object on internal — JUDGMENT
    nPO = (pattern_counts.get(P_PATCH_OBJECT_DECORATOR, 0)
           + pattern_counts.get(P_PATCH_OBJECT_CM, 0))
    if nPO:
        batches.append((
            "Batch 9: `patch.object(...)` cleanup",
            nPO,
            "Judgment. Replace with constructor injection or Fake.",
        ))
    # Batch 10: leftovers (PropertyMock, mock_open, create_autospec, …)
    nMisc = (pattern_counts.get(P_PROPERTYMOCK, 0)
             + pattern_counts.get(P_CREATE_AUTOSPEC, 0)
             + pattern_counts.get(P_MOCK_OPEN, 0)
             + pattern_counts.get(P_NONCALLABLEMOCK, 0)
             + pattern_counts.get(P_PATCH_MULTIPLE_DECORATOR, 0)
             + pattern_counts.get(P_PATCH_MULTIPLE_CM, 0))
    if nMisc:
        batches.append((
            "Batch 10: misc (`PropertyMock`, `create_autospec`, `mock_open`, …)",
            nMisc,
            "Per-case fixes.",
        ))
    return batches


if __name__ == "__main__":
    sys.exit(main())
