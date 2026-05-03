"""Phase B.5 codemod — rewrite legacy module references to ryotenkai_<pkg>.

Mappings live in ``scripts/codemod_mappings.txt`` (one rule per line:
``KIND:srcsub:dst``) so this script can rewrite itself without
self-corruption.

Two passes:
  1. Python ``from <legacy>.X import …`` / ``import <legacy>.X``.
  2. Quoted string-literals: ``"<legacy>.X.Y"`` / ``'<legacy>.X.Y'`` —
     covers mock.patch, importlib.import_module, sys.modules.get,
     subprocess argv, and console-script entry-point strings.

For each subsystem we generate THREE rule shapes, longest-prefix-first:
  - ``LEGACY.sub.``  → ``DST.``                  (deep ``from ryotenkai_shared.utils.X``)
  - ``LEGACY.sub`` followed by space/EOL → ``DST`` (bare ``from src.utils``)
  - quoted equivalents (pass 2)
where ``LEGACY = src`` for SUBSYSTEM rules and ``LEGACY = src`` (then
mapped tests-prefix) for TESTPATH rules.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

LEGACY_ROOT = chr(115) + chr(114) + chr(99)  # avoid the literal "src" so
# this script doesn't trip its own pass-2 string-literal sweep when run
# against scripts/.


def load_mappings(path: Path) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Return (subsystem_rules, testpath_rules) as (legacy_prefix, dst_prefix)."""
    sub: list[tuple[str, str]] = []
    test: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        kind, srcsub, dst = line.split(":", 2)
        if kind == "SUBSYSTEM":
            sub.append((f"{LEGACY_ROOT}.{srcsub}", dst))
        elif kind == "TESTPATH":
            test.append((srcsub, dst))
        else:
            raise ValueError(f"unknown kind: {kind}")
    # Long-prefix first (so e.g. ``utils.config`` doesn't shadow ``utils``).
    sub.sort(key=lambda kv: -len(kv[0]))
    test.sort(key=lambda kv: -len(kv[0]))
    return sub, test


def build_rules(
    subs: list[tuple[str, str]],
    tests: list[tuple[str, str]],
    *,
    pass_no: int,
) -> list[tuple[re.Pattern[str], str]]:
    rules: list[tuple[re.Pattern[str], str]] = []
    for legacy, dst in subs + tests:
        legacy_esc = re.escape(legacy)
        if pass_no == 1:
            # ``from LEGACY.X import …`` — deep
            rules.append((
                re.compile(rf"\bfrom\s+{legacy_esc}\."),
                f"from {dst}.",
            ))
            # ``from LEGACY import …`` — bare segment, must be followed by space
            rules.append((
                re.compile(rf"\bfrom\s+{legacy_esc}(?=\s)"),
                f"from {dst}",
            ))
            # ``import LEGACY.X``
            rules.append((
                re.compile(rf"\bimport\s+{legacy_esc}\."),
                f"import {dst}.",
            ))
            # ``import LEGACY`` — bare
            rules.append((
                re.compile(rf"\bimport\s+{legacy_esc}(?=\b)"),
                f"import {dst}",
            ))
        else:
            # Pass 2: quoted forms — match prefix in either quote style.
            # We match prefix only; the rest of the dotted path follows.
            for quote in ('"', "'"):
                rules.append((
                    re.compile(rf"{re.escape(quote)}{legacy_esc}\."),
                    f"{quote}{dst}.",
                ))
                # Bare quoted form: ``"ryotenkai_shared.utils"`` (no further path).
                rules.append((
                    re.compile(rf"{re.escape(quote)}{legacy_esc}{re.escape(quote)}"),
                    f"{quote}{dst}{quote}",
                ))
    return rules


def rewrite_file(path: Path, rules: list[tuple[re.Pattern[str], str]]) -> bool:
    try:
        original = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return False
    new = original
    for pattern, replacement in rules:
        new = pattern.sub(replacement, new)
    if new == original:
        return False
    path.write_text(new, encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase B.5 codemod")
    parser.add_argument("--root", default=".")
    parser.add_argument("--mappings", default="scripts/codemod_mappings.txt")
    parser.add_argument("--include", nargs="+", default=["**/*.py"])
    parser.add_argument("--pass", dest="which", choices=["1", "2", "all"], default="all")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    subs, tests = load_mappings(Path(args.mappings))

    rules: list[tuple[re.Pattern[str], str]] = []
    if args.which in ("1", "all"):
        rules.extend(build_rules(subs, tests, pass_no=1))
    if args.which in ("2", "all"):
        rules.extend(build_rules(subs, tests, pass_no=2))

    skip_dirs = {".git", ".venv", "__pycache__", "node_modules", ".mypy_cache",
                 ".pytest_cache", ".ruff_cache", "build", "dist"}

    paths: set[Path] = set()
    for pattern in args.include:
        paths.update(p for p in root.glob(pattern) if p.is_file())

    paths = {p for p in paths if not any(part in skip_dirs for part in p.parts)}

    changed = 0
    for path in sorted(paths):
        if args.dry_run:
            try:
                txt = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            new = txt
            for r, s in rules:
                new = r.sub(s, new)
            if new != txt:
                changed += 1
                print(f"WOULD-CHANGE: {path.relative_to(root)}")
        elif rewrite_file(path, rules):
            changed += 1
            print(f"CHANGED: {path.relative_to(root)}")

    print(f"\n{'(dry-run) ' if args.dry_run else ''}{changed} files {'would be' if args.dry_run else ''} updated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
