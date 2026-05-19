"""MLflow integration lint rules (Phase M7).

AST-based static checks that complement importlinter contracts. The
importlinter catches *module* imports; this script catches *call sites*
and *attribute access* patterns that importlinter cannot see.

Rules enforced (each rule = one ``Check.<NAME>``):

* **NO_AUTOLOG** — ``mlflow.autolog(...)`` is forbidden anywhere in
  ``packages/``. Per community best-practice (BP #2), mixing
  ``autolog`` with HF Trainer's ``report_to="mlflow"`` produces
  duplicate runs and double-logged metrics. Use ``HFMlflowWiring``
  in pod-trainer for the canonical wiring.

* **NO_SET_TRACKING_URI_GLOBAL** — ``mlflow.set_tracking_uri(...)``
  is forbidden outside an allowlist of composition-root files. The
  call mutates a process-wide singleton; only the construction site
  of :class:`MlflowTransport` is allowed to call it.

* **NO_AD_HOC_MLFLOW_CLIENT** — ``mlflow.tracking.MlflowClient(...)``
  / ``mlflow.MlflowClient(...)`` / ``MlflowClient(...)`` is forbidden
  outside the transport + read-client implementation modules. All
  other code should depend on :class:`ITrackingClient` /
  :class:`IRunQuery` through DI.

* **NO_START_RUN_IN_TRAINER** — ``mlflow.start_run(...)`` is forbidden
  anywhere under ``packages/pod/src/ryotenkai_pod/trainer/``. Pod-side
  training subprocess MUST let HF MLflowCallback adopt
  ``MLFLOW_RUN_ID`` + ``MLFLOW_NESTED_RUN=TRUE`` from env (Pattern A).

Usage:

    python scripts/lint/mlflow_rules.py [path ...]

Exit code 0 if no violations, 1 if any violation found. Designed to be
run from CI and as a pre-commit hook.

The script is intentionally dependency-free (stdlib only) so it can
run in minimal environments.
"""

from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Allowlists — files that are permitted to use the forbidden patterns
# because they ARE the canonical implementations.
# ---------------------------------------------------------------------------

# ``mlflow.set_tracking_uri`` may only appear in these files (relative
# to repository root). Each entry below has a documented reason —
# never expand without a similar rationale.
_SET_TRACKING_URI_ALLOWLIST: frozenset[str] = frozenset(
    {
        # Canonical: one-shot URI stamp at MlflowTransport construction.
        "packages/shared/src/ryotenkai_shared/infrastructure/mlflow/transport.py",
        # Read-client constructor: passes tracking_uri to MlflowClient
        # but does NOT call set_tracking_uri (kept here as a safety
        # belt — the lint flags if a future edit reintroduces it).
        "packages/control/src/ryotenkai_control/pipeline/mlflow/read/client.py",
        # Concrete IPromptRegistry implementation -- calls set_tracking_uri
        # inside a per-request worker so mlflow.genai.load_prompt picks
        # the right server. Each call is hermetic (no leak across requests).
        "packages/shared/src/ryotenkai_shared/infrastructure/mlflow/prompt_registry.py",
        # M7-deletion-backlog: dormant runner-side relay (gated behind
        # RYOTENKAI_RUNNER_MLFLOW_RELAY=1; default off).
        "packages/pod/src/ryotenkai_pod/runner/mlflow_relay.py",
    }
)


# Legacy ``mlflow.autolog`` callsites that are part of the M7 deletion
# backlog. The trainer-side autolog manager is dormant in Pattern A
# (HF MLflowCallback owns logging) but the module still exists until
# the wider cascade — :class:`MLflowManager`, the trainer container's
# lazy MLflowAutologManager init — is unwound in a follow-up phase.
# Remove the entries when the files themselves are gone.
_AUTOLOG_ALLOWLIST: frozenset[str] = frozenset()


# Legacy ``mlflow.start_run`` callsites under ``packages/pod/.../trainer/``
# that are part of the M7 deletion backlog. The trainer no longer needs
# these — Pattern A has HF MLflowCallback adopt the parent run via
# ``MLFLOW_RUN_ID`` — but they are still wired into the orchestrator
# until the wider cascade is unwound.
_START_RUN_IN_TRAINER_LEGACY_PREFIXES: tuple[str, ...] = (
    # M7-deletion-backlog: phase_executor's nested-run helper.
    # TODO(M7-cleanup): delete
    # packages/pod/.../trainer/orchestrator/phase_executor/mlflow_logger.py
    # and rewrite the executor to emit typed events instead.
    "packages/pod/src/ryotenkai_pod/trainer/orchestrator/phase_executor/mlflow_logger.py",
)

# ``MlflowClient`` constructions may only appear in these files.
_MLFLOW_CLIENT_ALLOWLIST: frozenset[str] = frozenset(
    {
        # Canonical write-path transport.
        "packages/shared/src/ryotenkai_shared/infrastructure/mlflow/transport.py",
        # Canonical read-path client.
        "packages/control/src/ryotenkai_control/pipeline/mlflow/read/client.py",
        # Canonical alias-based Model Registry adapter implementing
        # IModelRegistry. Lazy MlflowClient construction is gated by the
        # composition root (publisher / CLI promote).
        "packages/shared/src/ryotenkai_shared/infrastructure/mlflow/registry.py",
        # Summary reporter -- collects per-phase metric bags via the
        # underlying ``MlflowClient`` because :class:`RunHandle` (the
        # narrow read surface) does not expose ``run.data.metrics``.
        "packages/control/src/ryotenkai_control/pipeline/reporting/summary_reporter.py",
        # Journal adapter -- downloads ``events/events.jsonl`` artifact
        # when the workspace copy is missing.
        "packages/control/src/ryotenkai_control/reports/adapters/journal_adapter.py",
        # M7-deletion-backlog: dormant runner-side relay. Activated only
        # via ``RYOTENKAI_RUNNER_MLFLOW_RELAY=1``; default flow has the
        # trainer talk to MLflow directly through ``MlflowTransport``.
        "packages/pod/src/ryotenkai_pod/runner/mlflow_relay.py",
    }
)


@dataclass(frozen=True)
class Violation:
    """One detected lint violation."""

    rule: str
    file: Path
    line: int
    col: int
    message: str

    def format(self, root: Path) -> str:
        rel = self.file.relative_to(root) if self.file.is_absolute() else self.file
        return f"{rel}:{self.line}:{self.col}: [{self.rule}] {self.message}"


# ---------------------------------------------------------------------------
# AST visitor
# ---------------------------------------------------------------------------


def _is_attr_chain(node: ast.expr, names: tuple[str, ...]) -> bool:
    """True if ``node`` is the attribute chain ``names[0].names[1]…``.

    Example: ``_is_attr_chain(node, ("mlflow", "set_tracking_uri")) ==
    True`` when ``node`` represents ``mlflow.set_tracking_uri``.
    """
    parts: list[str] = []
    cur: ast.expr | None = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
    else:
        return False
    parts.reverse()
    return tuple(parts) == names


def _is_call_to(node: ast.Call, candidates: tuple[tuple[str, ...], ...]) -> bool:
    """True if ``node.func`` matches any of the attribute chains in ``candidates``."""
    return any(_is_attr_chain(node.func, chain) for chain in candidates)


def _is_named_call(node: ast.Call, name: str) -> bool:
    """True if ``node.func`` is the bare name ``name`` (e.g. ``MlflowClient(...)``)."""
    return isinstance(node.func, ast.Name) and node.func.id == name


class _MLflowRuleVisitor(ast.NodeVisitor):
    def __init__(self, file_path: Path, rel_path: str) -> None:
        self.file_path = file_path
        self.rel_path = rel_path
        self.violations: list[Violation] = []

    def visit_Call(self, node: ast.Call) -> None:
        # NO_AUTOLOG
        if (
            _is_call_to(node, (("mlflow", "autolog"),))
            or _is_call_to(node, (("mlflow", "transformers", "autolog"),))
        ) and self.rel_path not in _AUTOLOG_ALLOWLIST:
            self.violations.append(
                Violation(
                    rule="NO_AUTOLOG",
                    file=self.file_path,
                    line=node.lineno,
                    col=node.col_offset,
                    message=(
                        "mlflow.autolog/mlflow.transformers.autolog is forbidden. "
                        "Use HFMlflowWiring + report_to=['mlflow'] (Pattern A)."
                    ),
                ),
            )

        # NO_SET_TRACKING_URI_GLOBAL
        if _is_call_to(node, (("mlflow", "set_tracking_uri"),)):
            if self.rel_path not in _SET_TRACKING_URI_ALLOWLIST:
                self.violations.append(
                    Violation(
                        rule="NO_SET_TRACKING_URI_GLOBAL",
                        file=self.file_path,
                        line=node.lineno,
                        col=node.col_offset,
                        message=(
                            "mlflow.set_tracking_uri mutates a process-wide singleton. "
                            "Only MlflowTransport.__init__ may call it; depend on "
                            "ITrackingClient/IRunQuery via DI elsewhere."
                        ),
                    ),
                )

        # NO_AD_HOC_MLFLOW_CLIENT
        is_client_call = (
            _is_call_to(
                node,
                (
                    ("mlflow", "tracking", "MlflowClient"),
                    ("mlflow", "MlflowClient"),
                ),
            )
            or _is_named_call(node, "MlflowClient")
        )
        if is_client_call and self.rel_path not in _MLFLOW_CLIENT_ALLOWLIST:
            self.violations.append(
                Violation(
                    rule="NO_AD_HOC_MLFLOW_CLIENT",
                    file=self.file_path,
                    line=node.lineno,
                    col=node.col_offset,
                    message=(
                        "Construct MlflowClient only inside MlflowTransport or "
                        "MlflowReadClient. All other code must depend on "
                        "ITrackingClient / IRunQuery through DI."
                    ),
                ),
            )

        # NO_START_RUN_IN_TRAINER
        if _is_call_to(node, (("mlflow", "start_run"),)):
            if self.rel_path.startswith("packages/pod/src/ryotenkai_pod/trainer/"):
                # Skip the to-be-deleted legacy modules. M4 / the M7
                # deletion backlog will remove them; until then the
                # lint should not flag.
                if not self.rel_path.startswith(
                    _START_RUN_IN_TRAINER_LEGACY_PREFIXES,
                ):
                    self.violations.append(
                        Violation(
                            rule="NO_START_RUN_IN_TRAINER",
                            file=self.file_path,
                            line=node.lineno,
                            col=node.col_offset,
                            message=(
                                "mlflow.start_run is forbidden in pod-trainer. "
                                "HF MLflowCallback adopts MLFLOW_RUN_ID + "
                                "MLFLOW_NESTED_RUN=TRUE from env (Pattern A)."
                            ),
                        ),
                    )

        self.generic_visit(node)


# ---------------------------------------------------------------------------
# Walking + main
# ---------------------------------------------------------------------------


_SKIP_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        "node_modules",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".import_linter_cache",
        ".hypothesis",
        "build",
        "dist",
        ".claude",
        "tests",  # tests have their own conventions; never lint them here
    }
)


def _iter_python_files(root: Path, paths: list[Path]) -> list[Path]:
    """Yield every ``.py`` file under any of ``paths`` (skipping common cruft)."""
    out: list[Path] = []
    for top in paths:
        if top.is_file() and top.suffix == ".py":
            out.append(top)
            continue
        for path in top.rglob("*.py"):
            if any(part in _SKIP_DIRS for part in path.parts):
                continue
            out.append(path)
    return out


def _scan_file(path: Path, root: Path) -> list[Violation]:
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []

    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return []

    rel_path = str(path.relative_to(root)) if path.is_absolute() else str(path)
    visitor = _MLflowRuleVisitor(path, rel_path)
    visitor.visit(tree)
    return visitor.violations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        default=["packages"],
        help="Files or directories to scan (default: packages).",
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Repository root used for relative-path display (default: cwd).",
    )
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    targets = [Path(p) for p in args.paths]
    targets = [p if p.is_absolute() else root / p for p in targets]

    files = _iter_python_files(root, targets)
    violations: list[Violation] = []
    for f in files:
        violations.extend(_scan_file(f, root))

    if not violations:
        print(f"OK: scanned {len(files)} file(s); no MLflow rule violations.")
        return 0

    print(f"FAIL: {len(violations)} violation(s) across {len({v.file for v in violations})} file(s):")
    for v in sorted(violations, key=lambda x: (str(x.file), x.line, x.rule)):
        print("  " + v.format(root))
    return 1


if __name__ == "__main__":
    sys.exit(main())
