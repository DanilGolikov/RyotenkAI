"""Backward-compat shim — canonical home moved to ``src.constants.RUNTIME_IMAGE``.

The shim re-evaluates :func:`_resolve_runtime_image` from
:mod:`src.constants` at import time so an ``importlib.reload(...)`` of
this module still picks up env-override changes (kept as a defensive
seam for the existing reload-based override test until Phase B removes
the shim).

Removed at the start of Phase B (monorepo packagization, see
``docs/plans/2026-05-03-monorepo-uv-workspace-packagization.md`` §B.4).
"""

from __future__ import annotations

from typing import Final

from src.constants import _resolve_runtime_image

RUNTIME_IMAGE: Final[str] = _resolve_runtime_image()


__all__ = ["RUNTIME_IMAGE"]
