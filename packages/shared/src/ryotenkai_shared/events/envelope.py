"""Common envelope for the unified event system.

Every event variant in :mod:`ryotenkai_shared.events.types` extends
:class:`BaseEvent` and pins ``type`` / ``severity`` via ``Literal`` defaults
to participate in the closed discriminated union.

Design invariants:

* ``frozen=True``     — events are immutable once constructed. Re-emitting
  the same event MUST produce a new instance; in-place mutation is a bug.
* ``extra="forbid"``  — unknown fields in raw JSON dicts are rejected at
  the boundary. Forward-compat is handled explicitly via
  :class:`ryotenkai_shared.events.types.unknown.UnknownEvent`.
* ``event_id``        — UUIDv7 (RFC 9562). Time-ordered, monotonic at the
  millisecond level on most platforms; used for journal natural ordering
  and SSE ``Last-Event-ID`` resumption.
* ``time``            — UTC-aware ``datetime`` with microsecond resolution.
  Producers fill this at emit time; never trust a caller-supplied value
  unless ``emit_remote`` semantics apply.
* ``offset``          — monotonic counter per ``(run_id, source)`` pair.
  Assigned by the emitter, never by the caller. Required (no default) so
  schemas that forget to set it fail Pydantic validation rather than
  silently emitting ``offset=0``.

Subclass shape (concrete events use this pattern):

    class TrainingStartedEvent(BaseEvent):
        kind: Literal["ryotenkai.pod.training.started"] = "ryotenkai.pod.training.started"
        severity: Literal["info"] = "info"
        payload: TrainingStartedPayload

The ``kind`` field is the discriminator (per the project-wide AD-6
convention enforced by ``tests/_lint/test_discriminator_uniformity.py``);
``severity`` is pinned by Literal default so it can never be lifted from
the per-event policy.
"""

from __future__ import annotations

import os
import time
from datetime import UTC, datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from ryotenkai_shared.events.severity import Severity  # noqa: TC001 — Pydantic field type, needs runtime

# Optional fast Rust-backed generator. The pod image ships a slimmer dep
# set so we cannot rely on it being installed there; fall back to a pure-
# Python RFC 9562 implementation when absent. Both code paths return a
# stdlib :class:`uuid.UUID`, so callers never need to know which is in
# use.
try:
    import uuid_utils as _uuid_utils  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover — pod-side fallback path
    _uuid_utils = None


def _uuid7_pure_python() -> UUID:
    """RFC 9562 UUIDv7 generated without external deps.

    Layout (128 bits, big-endian):
      | 48-bit unix_ts_ms | 4-bit ver=0x7 | 12-bit rand_a |
      | 2-bit  var=0b10   | 62-bit rand_b                  |

    Throughput is roughly ~150k uuids/sec on CPython 3.12 (vs ~700k for
    the Rust binding). Both far above pipeline emit rates (~30 ev/sec
    typical), so the fallback is functionally equivalent.
    """
    ts_ms = int(time.time() * 1000) & 0xFFFFFFFFFFFF  # 48 bits
    rand_a = int.from_bytes(os.urandom(2), "big") & 0x0FFF  # 12 bits
    rand_b = int.from_bytes(os.urandom(8), "big") & 0x3FFFFFFFFFFFFFFF  # 62 bits
    val = (
        (ts_ms << 80)
        | (0x7 << 76)
        | (rand_a << 64)
        | (0b10 << 62)
        | rand_b
    )
    return UUID(int=val)


def new_uuid7() -> UUID:
    """Mint a fresh UUIDv7 (RFC 9562).

    Uses :mod:`uuid_utils` for a Rust-native generator (~700k uuids/sec
    on M-series Mac per the Phase 0 smoke benchmark) when available;
    falls back to a pure-Python implementation otherwise so pod images
    that don't pin the optional dep still work. Returns a stdlib
    :class:`uuid.UUID` so consumers don't need to import ``uuid_utils``
    transitively.
    """
    if _uuid_utils is not None:
        return UUID(str(_uuid_utils.uuid7()))
    return _uuid7_pure_python()


def utc_now() -> datetime:
    """Return the current UTC instant as a timezone-aware ``datetime``.

    Centralising the source of truth here makes it cheap to fake in tests
    by patching ``ryotenkai_shared.events.envelope.utc_now``.
    """
    return datetime.now(UTC)


class BaseEvent(BaseModel):
    """Frozen, ``extra=forbid`` envelope shared by every event variant.

    Concrete subclasses pin ``type`` and ``severity`` via ``Literal``
    defaults, and add a ``payload`` field with their own typed payload
    schema. The discriminated union assembly lives in
    :mod:`ryotenkai_shared.events.discriminator`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    event_id: UUID = Field(default_factory=new_uuid7)
    kind: str
    source: str
    time: datetime = Field(default_factory=utc_now)
    run_id: str
    stage_id: str | None = None
    offset: int
    schema_version: int = 1
    severity: Severity


__all__ = [
    "BaseEvent",
    "new_uuid7",
    "utc_now",
]
