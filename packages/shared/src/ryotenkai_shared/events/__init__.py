"""Unified event system foundations (Phase 1).

This package exposes the typed envelope, the discriminated union, the
length-prefixed JSONL codec, the :class:`IEventEmitter` Protocol, and
the upcaster registry — everything downstream packages need to *produce*
or *consume* events without depending on transport-layer code.

Concrete emitter / journal / bus implementations live in
``ryotenkai_pod.runner`` and ``ryotenkai_control.events`` (Phases 2 & 3).

Re-exports here are intentionally flat: most call sites do
``from ryotenkai_shared.events import to_jsonl, from_jsonl, BaseEvent``.
For per-event types, import from
``ryotenkai_shared.events.types.<domain>``.
"""

from __future__ import annotations

from ryotenkai_shared.events.codec import (
    MalformedEventError,
    from_jsonl,
    parse_length_prefix,
    to_jsonl,
)
from ryotenkai_shared.events.discriminator import EVENT_ADAPTER, Event
from ryotenkai_shared.events.envelope import BaseEvent, new_uuid7, utc_now
from ryotenkai_shared.events.protocol import IEventEmitter
from ryotenkai_shared.events.severity import SEVERITY_ORDER, Severity
from ryotenkai_shared.events.types.unknown import UNKNOWN_OFFSET, UnknownEvent

__all__ = [
    "EVENT_ADAPTER",
    "SEVERITY_ORDER",
    "UNKNOWN_OFFSET",
    "BaseEvent",
    "Event",
    "IEventEmitter",
    "MalformedEventError",
    "Severity",
    "UnknownEvent",
    "from_jsonl",
    "new_uuid7",
    "parse_length_prefix",
    "to_jsonl",
    "utc_now",
]
