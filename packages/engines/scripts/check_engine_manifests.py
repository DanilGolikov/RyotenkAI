#!/usr/bin/env python3
"""Drift detector for ``engine.toml`` manifests.

Cross-checks that every shipped engine's runtime + config classes
match its declared manifest. Failures here mean an author bumped one
side without the other.

Checks:
  1. Every ``packages/engines/src/ryotenkai_engines/<id>/engine.toml``
     parses against ``EngineManifest``.
  2. ``[engine].id`` equals the folder name.
  3. ``[entry_points.runtime].class`` resolves and implements
     ``IInferenceEngine`` (zero-arg ``cls()`` then ``isinstance`` check).
  4. ``[entry_points.config_schema].class`` is a ``BaseEngineConfig``
     subclass and its ``kind: Literal[...]`` matches the engine id.
  5. Runtime ``get_capabilities()`` exactly matches manifest ``[capabilities]``
     block (1:1 parity).
  6. ``[image].default``, if present, is not a floating tag.

Usage:
  uv run python packages/engines/scripts/check_engine_manifests.py

Exit code: 0 = all green, 1 = any drift.
Used by CI; pre-commit hook runs the same script.
"""

from __future__ import annotations

import sys
from pathlib import Path

ENGINES_ROOT = (
    Path(__file__).resolve().parent.parent / "src" / "ryotenkai_engines"
)


def _check_manifest_loads(failures: list[str]) -> tuple[object, list[str]]:
    """Step 1: load registry, surface any LoadFailure."""
    from ryotenkai_engines.registry import EngineRegistry

    registry = EngineRegistry.from_filesystem()
    engine_ids = list(registry.list())
    for f in registry.failures():
        failures.append(
            f"[manifest-load] {f.engine_id}: {f.exc_type}: {f.reason}"
        )
    return registry, engine_ids


def _check_class_resolution(registry, engine_ids: list[str], failures: list[str]) -> None:
    """Step 3 + 4: runtime + config class resolution + ``kind`` parity."""
    for eid in engine_ids:
        try:
            registry.get_runtime(eid)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"[runtime-resolve] {eid}: {type(exc).__name__}: {exc}")
        try:
            registry.get_config_class(eid)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"[config-resolve] {eid}: {type(exc).__name__}: {exc}")


def _check_capability_parity(
    registry, engine_ids: list[str], failures: list[str]
) -> None:
    """Step 5: runtime get_capabilities() == manifest [capabilities]."""
    for eid in engine_ids:
        try:
            manifest_caps = registry.get_manifest(eid).capabilities
            runtime_caps = registry.get_runtime(eid)().get_capabilities()
        except Exception as exc:  # noqa: BLE001
            failures.append(f"[caps-fetch] {eid}: {type(exc).__name__}: {exc}")
            continue
        if manifest_caps != runtime_caps:
            failures.append(
                f"[caps-drift] {eid}: manifest != runtime\n"
                f"  manifest: {manifest_caps.model_dump()}\n"
                f"  runtime:  {runtime_caps.model_dump()}"
            )


def _check_image_tags_pinned(
    registry, engine_ids: list[str], failures: list[str]
) -> None:
    """Step 6: explicit [image].default must be semver-pinned, no floating tags."""
    floating_suffixes = (":latest", ":dev", ":main", ":master", ":nightly", ":edge")
    for eid in engine_ids:
        manifest = registry.get_manifest(eid)
        if manifest.image is None:
            continue  # convention default — no explicit tag to check
        if any(manifest.image.default.endswith(s) for s in floating_suffixes):
            failures.append(
                f"[floating-tag] {eid}: image.default={manifest.image.default!r} "
                f"uses a floating tag — pin to a specific version."
            )


def main() -> int:
    failures: list[str] = []

    registry, engine_ids = _check_manifest_loads(failures)
    if engine_ids:
        _check_class_resolution(registry, engine_ids, failures)
        _check_capability_parity(registry, engine_ids, failures)
        _check_image_tags_pinned(registry, engine_ids, failures)

    if failures:
        print(f"[check_engine_manifests] {len(failures)} drift issue(s):", file=sys.stderr)
        for line in failures:
            print(f"  - {line}", file=sys.stderr)
        return 1

    print(
        f"[check_engine_manifests] OK — {len(engine_ids)} engine(s) checked: "
        f"{', '.join(engine_ids) or '(none)'}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
