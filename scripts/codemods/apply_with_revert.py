"""Apply a libcst codemod per-file with auto-revert on test failure.

For each file in the target scope:

1. Snapshot the file (in memory).
2. Apply the codemod.
3. Run the matching pytest test (the file itself).
4. If tests pass: keep the changes.
5. If tests fail: revert the file from the snapshot, log it.

This converts the codemod into a self-verifying batch operation that
cannot leave the lane red.  Files that revert can be investigated and
either fixed manually or left for a later cleanup pass.

The codemod to apply is selected via ``--codemod <module>`` (default:
``scripts.codemods.magicmock_to_simplenamespace`` to preserve the
Phase 2A invocation).  The module is dynamically imported and we look
for the *single* ``libcst.codemod.Codemod`` subclass it defines.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import subprocess
import sys
from pathlib import Path

import libcst as cst
from libcst.codemod import Codemod, CodemodContext


def _resolve_codemod_class(module_path: str) -> type[Codemod]:
    """Import ``module_path`` and return the single Codemod subclass."""

    mod = importlib.import_module(module_path)
    candidates: list[type[Codemod]] = []
    for _name, obj in inspect.getmembers(mod, inspect.isclass):
        if (
            issubclass(obj, Codemod)
            and obj is not Codemod
            and obj.__module__ == mod.__name__
        ):
            candidates.append(obj)
    if not candidates:
        raise RuntimeError(
            f"No Codemod subclass found in module {module_path!r}.",
        )
    if len(candidates) > 1:
        names = ", ".join(c.__name__ for c in candidates)
        raise RuntimeError(
            f"Multiple Codemod subclasses in {module_path!r}: {names}. "
            "Ambiguous — specify which one to apply.",
        )
    return candidates[0]


def _run_codemod(path: Path, codemod_cls: type[Codemod]) -> int:
    """Return the number of conversions applied, writing in-place.

    ``changed_count`` attribute on the codemod instance is the canonical
    counter; fall back to a simple "did the source change" check for
    codemods that don't expose one.
    """

    src = path.read_text()
    context = CodemodContext(filename=str(path))
    codemod = codemod_cls(context)
    tree = cst.parse_module(src)
    new_tree = codemod.transform_module_impl(tree)
    new_src = new_tree.code
    if new_src == src:
        return 0
    path.write_text(new_src)
    return int(getattr(codemod, "changed_count", 1))


def _run_pytest(file: Path) -> bool:
    """Return True if pytest passes for this file."""

    result = subprocess.run(
        [
            ".venv/bin/python",
            "-m",
            "pytest",
            "-c",
            "tests/pytest.ini",
            str(file),
            "-q",
            "--no-header",
            "--tb=no",
            "-p",
            "no:warnings",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        # Echo for the operator; useful to see WHY it failed.
        sys.stderr.write(f"\n[FAIL] {file}\n{result.stdout[-2000:]}\n")
        return False
    return True


def _iter_python_files(paths: list[str]) -> list[Path]:
    out: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_file() and p.suffix == ".py":
            out.append(p)
        elif p.is_dir():
            out.extend(sorted(p.rglob("test_*.py")))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.codemods.apply_with_revert",
    )
    parser.add_argument(
        "--codemod",
        default="scripts.codemods.magicmock_to_simplenamespace",
        help=(
            "Dotted module path to import for the codemod (the module "
            "must define exactly one Codemod subclass)."
        ),
    )
    parser.add_argument(
        "--paths",
        nargs="*",
        help="Files or directories (alternate way to pass paths).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help=(
            "Required to actually write changes (otherwise dry-run). "
            "Apply means: write codemod output, run pytest, revert on fail."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan without modifying files (default).",
    )
    parser.add_argument(
        "positional_paths",
        nargs="*",
        help="Files or directories.",
    )
    args = parser.parse_args(argv)

    paths_arg: list[str] = list(args.positional_paths)
    if args.paths:
        paths_arg.extend(args.paths)
    if not paths_arg:
        parser.error("at least one path is required")

    apply = args.apply and not args.dry_run

    codemod_cls = _resolve_codemod_class(args.codemod)

    files = _iter_python_files(paths_arg)
    kept_count = 0
    reverted: list[Path] = []
    untouched = 0
    total_conversions = 0

    for f in files:
        original = f.read_text()
        n = _run_codemod(f, codemod_cls)
        if n == 0:
            untouched += 1
            continue
        if not apply:
            f.write_text(original)
            sys.stdout.write(f"[dry-run] {f}: {n} conversions\n")
            total_conversions += n
            continue
        if _run_pytest(f):
            sys.stdout.write(f"[KEEP]   {f}: {n} conversions\n")
            kept_count += 1
            total_conversions += n
        else:
            f.write_text(original)
            sys.stdout.write(f"[REVERT] {f}: {n} conversions reverted\n")
            reverted.append(f)

    sys.stdout.write(
        f"\nSummary: "
        f"{kept_count} kept ({total_conversions} conversions), "
        f"{len(reverted)} reverted, "
        f"{untouched} untouched.\n",
    )
    if reverted:
        sys.stdout.write("Reverted files:\n")
        for f in reverted:
            sys.stdout.write(f"  {f}\n")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    sys.exit(main())
