#!/usr/bin/env python3
"""Drift detector for ``engine.toml`` manifests.

PR-1 stub — full implementation lands in PR-10. Checks (when complete):

  1. Every ``packages/engines/src/ryotenkai_engines/<id>/engine.toml`` parses
     against ``EngineManifest``.
  2. ``[engine].id`` equals the folder name.
  3. ``[entry_points.runtime].class`` resolves and implements
     ``IInferenceEngine`` (runtime-checkable Protocol).
  4. ``[entry_points.config_schema].class`` subclasses ``BaseEngineConfig``
     and has ``kind: Literal["<id>"]``.
  5. Runtime ``get_capabilities()`` exactly matches manifest ``[capabilities]``
     block (1:1 parity).
  6. ``[image].default``, if present, is not a floating tag (``:latest``,
     ``:dev``, etc.) — drift checker enforces semver-pinned tags.
  7. The convention image name (``f"{prefix}/inference-{id}:{version}"``)
     would not collide with another engine's manifest.

Exits 0 on clean run, 1 on any failure. Used by CI (``.github/workflows/...``)
and pre-commit (``pre-commit-config.yaml``).
"""

from __future__ import annotations

import sys


def main() -> int:
    print("[check_engine_manifests] PR-1 stub — full checks land in PR-10.")
    # PR-1 success criterion: the script is invocable. PR-2/PR-10 fill it in.
    return 0


if __name__ == "__main__":
    sys.exit(main())
