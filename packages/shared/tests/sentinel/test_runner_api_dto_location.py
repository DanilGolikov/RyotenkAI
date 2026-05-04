"""Sentinel — runner HTTP/WS wire DTOs MUST live ONLY in
``ryotenkai_shared.contracts.runner_api`` (transport-unification-v2,
Phase 0 PR-0b).

Why: pre-Phase-0 these models lived in
``ryotenkai_pod.runner.api.schemas``, which made the Mac-side client
(``JobClient`` in ``shared``) reference DTOs from a package it can't
import (``shared`` is the leaf — see :mod:`test_shared_is_leaf`).

This sentinel asserts:

1. **No DTO classes** are defined in ``ryotenkai_pod.runner.api.*`` —
   Pydantic ``BaseModel`` subclasses with the canonical wire-DTO
   names (``Job*``, ``Event*``, ``InternalEvent*``, ``ControlHeartbeat*``).
2. **The canonical names** ARE present in
   ``ryotenkai_shared.contracts.runner_api``.

Two failure modes:

- Drift via inline class: someone adds a new ``class FooResponse(BaseModel)``
  in ``runner.api.foo`` instead of putting it in shared.contracts.
- Stale alias re-exports: PR-0a left re-exports as a transitional
  shim; this sentinel ensures PR-0b actually deleted them.
"""

from __future__ import annotations

import ast
from pathlib import Path

# Class-name patterns that MUST live in shared.contracts.runner_api.
# Suffixed with the family (Spec/Response/Request) to avoid catching
# helper classes that happen to share a prefix.
_DTO_NAMES = frozenset(
    {
        "JobSpec",
        "JobSubmittedResponse",
        "JobSnapshotResponse",
        "JobStopAcceptedResponse",
        "EventResponse",
        "InternalEventRequest",
        "ControlHeartbeatRequest",
        "ControlHeartbeatResponse",
    }
)


def _runner_api_dir() -> Path:
    # packages/shared/tests/sentinel/test_runner_api_dto_location.py
    # → packages/pod/src/ryotenkai_pod/runner/api/
    here = Path(__file__).resolve()
    repo_root = here.parents[4]
    return repo_root / "packages" / "pod" / "src" / "ryotenkai_pod" / "runner" / "api"


def _shared_contracts_runner_api_init() -> Path:
    here = Path(__file__).resolve()
    repo_root = here.parents[4]
    return (
        repo_root
        / "packages"
        / "shared"
        / "src"
        / "ryotenkai_shared"
        / "contracts"
        / "runner_api"
        / "__init__.py"
    )


def test_no_runner_dto_classes_inside_pod_runner_api() -> None:
    """No file in ``pod.runner.api`` may define a class with a
    canonical wire-DTO name."""
    api_dir = _runner_api_dir()
    assert api_dir.exists(), f"sentinel mis-anchored: {api_dir}"

    violations: list[str] = []
    for path in api_dir.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name in _DTO_NAMES:
                violations.append(f"{path.name}: class {node.name}")

    assert not violations, (
        "Wire DTOs must live in ryotenkai_shared.contracts.runner_api, "
        "not inside the pod runner API package. Move these definitions "
        "and import them via "
        "`from ryotenkai_shared.contracts.runner_api import ...`:\n  "
        + "\n  ".join(violations)
    )


def test_canonical_dtos_re_exported_from_shared_contracts() -> None:
    """Every canonical DTO name MUST be re-exported from
    ``ryotenkai_shared.contracts.runner_api.__init__``."""
    init_path = _shared_contracts_runner_api_init()
    assert init_path.exists(), f"sentinel mis-anchored: {init_path}"

    tree = ast.parse(init_path.read_text(encoding="utf-8"))
    exported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                exported.add(alias.asname or alias.name)
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "__all__"
            and isinstance(node.value, ast.List)
        ):
            for elt in node.value.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                    exported.add(elt.value)

    missing = _DTO_NAMES - exported
    assert not missing, (
        "shared.contracts.runner_api.__init__ must re-export every "
        f"canonical wire DTO. Missing: {sorted(missing)}"
    )


def test_no_alias_re_export_of_runner_dtos_in_pod_runner_api() -> None:
    """After PR-0b the alias-re-export shim must be gone — neither
    ``schemas.py`` nor any other file in ``pod.runner.api`` may
    ``from ryotenkai_shared.contracts.runner_api import ...`` AND
    re-publish the symbols via ``__all__``.

    (Pure imports for use inside the file are fine; what we forbid
    is making this package look like the canonical home again.)
    """
    api_dir = _runner_api_dir()
    legacy_schemas = api_dir / "schemas.py"
    assert not legacy_schemas.exists(), (
        f"{legacy_schemas} should have been deleted in PR-0b — "
        "the canonical home is shared.contracts.runner_api."
    )
