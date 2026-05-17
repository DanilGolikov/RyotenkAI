"""Upcaster registry and chain runner.

Schema evolution policy (full text in ``README.md`` alongside this file):

1. Backward-compatible additions (new optional field with default) do NOT
   need an upcaster — older payloads validate against the new schema.
2. Renames / removals / semantic shifts require a NEW dotted ``type``
   (e.g. ``ryotenkai.pod.training.started`` v1 → a NEW type for v2). The
   upcaster chain rewrites the raw envelope's ``type`` field if needed.
3. An upcaster handles exactly one hop ``(N, N+1)``. Multi-hop is
   composed at chain-application time.

Phase 1 ships an empty registry. The infrastructure is here so journal
readers in later phases (Phase 3 control-side, Phase 7 report adapter)
can call :func:`apply_chain` without short-circuiting based on whether
upcasters exist.
"""

from __future__ import annotations

from typing import Any

from ryotenkai_shared.events.upcasters._types import Upcaster

# Module-private mutable state. The registry is intentionally a plain
# dict — Phase 1 is single-process by design and registration happens at
# import time. A future phase that needs hot-reload semantics will swap
# this for an explicit registrar object.
_REGISTRY: dict[str, list[Upcaster]] = {}


def register(event_type: str, upcaster: Upcaster) -> None:
    """Append ``upcaster`` to the chain for ``event_type``.

    Order matters: upcasters are applied in registration order. The
    expectation is that each upcaster bumps the schema version by one and
    declares its ``(from, to)`` hop via the version arguments it receives.
    """
    _REGISTRY.setdefault(event_type, []).append(upcaster)


def clear() -> None:
    """Empty the registry. Test-only helper to keep cases isolated."""
    _REGISTRY.clear()


def latest_version_for(event_type: str) -> int:
    """Return the latest schema version known for ``event_type``.

    With zero upcasters the answer is 1 — the baseline. Each registered
    upcaster bumps the latest version by one (under the one-hop policy).
    """
    return 1 + len(_REGISTRY.get(event_type, ()))


def apply_chain(
    raw: dict[str, Any],
    event_type: str,
    current_version: int,
    target_version: int,
) -> dict[str, Any]:
    """Run the upcaster chain ``current → target`` for ``event_type``.

    Returns the raw envelope dict at ``target_version``. Identity when
    ``current_version >= target_version``. Raises :class:`KeyError` if a
    required hop is missing — that signals a bug, not a runtime fault.
    The caller (codec) decides whether to surface the error to the user.

    The raw envelope is treated as opaque structure: the chain may
    rewrite ``payload``, rename fields, or even change ``type`` (for the
    "new dotted type on semantic break" pattern).
    """
    if current_version >= target_version:
        return raw
    chain = _REGISTRY.get(event_type, ())
    # Each registered upcaster handles versions (1, 2), (2, 3), ...
    # so the index for hop (N, N+1) is N-1 against the registered list.
    out = raw
    for hop_from in range(current_version, target_version):
        hop_to = hop_from + 1
        idx = hop_from - 1  # baseline is version 1, first upcaster index = 0
        if idx >= len(chain):
            raise KeyError(
                f"missing upcaster for {event_type} hop {hop_from} -> {hop_to}",
            )
        out = chain[idx](out, hop_from, hop_to)
    # Pin the resulting envelope's declared schema_version. Upcasters may
    # leave the field stale (especially if they only touched payload).
    out = {**out, "schema_version": target_version}
    return out


__all__ = [
    "Upcaster",
    "apply_chain",
    "clear",
    "latest_version_for",
    "register",
]
