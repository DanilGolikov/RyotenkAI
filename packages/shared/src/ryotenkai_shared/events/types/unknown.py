"""Catch-all variant for forward-compatibility (Risk R-11 in the plan).

When a consumer reads a journal entry whose ``type`` is not in the local
union (e.g. a newer producer emitted a type unknown to this version of
the code), the codec wraps it in :class:`UnknownEvent` instead of
crashing the read path. The original type and payload are preserved so
downstream tooling (debugger, reports, UI) can still render something.

This is deliberately NOT a default — producers must declare a concrete
event class. The codec only emits :class:`UnknownEvent` when
``strict=False``.
"""

from __future__ import annotations

import copy
from typing import Any, Literal

from pydantic import ConfigDict, model_validator

from ryotenkai_shared.events.envelope import BaseEvent
from ryotenkai_shared.events.severity import Severity  # noqa: TC001 — Pydantic field type, runtime needed

#: Sentinel offset used when a malformed or forward-compat event's raw
#: envelope did not include an ``offset`` field (or it was unusable).
#:
#: Journal readers MUST treat this value as "do not include in
#: monotonic-offset checks" — it is not a real position in any
#: ``(run_id, source)`` sequence. ``-1`` is chosen because the canonical
#: offset domain is the non-negative integers; a negative value cannot
#: collide with a real offset.
UNKNOWN_OFFSET: int = -1


class UnknownEvent(BaseEvent):
    """Forward-compat envelope for unknown event types.

    ``severity`` is intentionally NOT pinned via Literal — an unknown
    event may carry any severity from its origin producer. We keep it as
    a regular field with a default so the codec can preserve whatever
    the source declared.

    ``raw_payload`` is a plain ``dict`` whose contents are
    **deep-copied at construction** — the BaseEvent ``frozen=True``
    guarantee blocks re-binding the field, but it does not prevent
    in-place mutation of the dict's contents. Deep-copying at construction
    means mutations to the dict passed in by the caller cannot affect the
    stored value, and two ``UnknownEvent`` instances built from the same
    literal dict do not silently alias each other. Note: the stored dict
    can still be mutated by hostile code after the fact, but mutations
    won't leak back to future constructions.
    """

    # extra=forbid still applies, but we relax frozen via parent. Unknown
    # events keep the same immutability guarantee as known ones.
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["ryotenkai.unknown"] = "ryotenkai.unknown"
    severity: Severity = "info"
    original_type: str
    raw_payload: dict[str, Any]

    @model_validator(mode="after")
    def _isolate_raw_payload(self) -> UnknownEvent:
        """Snapshot ``raw_payload`` at construction.

        Pydantic's ``frozen=True`` prevents rebinding the attribute but
        not mutation of the nested ``dict``. We deep-copy here so the
        captured value is the snapshot at construction time. Uses
        ``object.__setattr__`` to bypass frozen — this is the canonical
        Pydantic pattern for post-validation normalization.
        """
        object.__setattr__(self, "raw_payload", copy.deepcopy(self.raw_payload))
        return self


__all__ = ["UNKNOWN_OFFSET", "UnknownEvent"]
