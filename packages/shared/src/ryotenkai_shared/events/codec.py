"""JSONL codec with length-prefixed framing for crash-safe journals.

Wire format:

    <length>\\t<envelope_json>\\n

Where ``<length>`` is the byte length (UTF-8 encoded) of
``<envelope_json>`` — NOT including the leading ``length``, the tab, or
the trailing newline. The prefix lets readers detect torn writes from
``kill -9`` mid-``write()``: if the bytes after the tab don't match the
declared length, the line is partial and the reader truncates it.

UTF-8 byte length (not ``len(str)``) is the safe measurement because
emoji and non-ASCII characters in payloads (e.g. error messages with
non-Latin glyphs, code points outside the BMP) take >1 byte each. Using
character length would let multi-byte content slip past the integrity
check.

Reader semantics:

* :func:`parse_length_prefix` strictly parses the wire format. Raises
  :class:`ValueError` on any deviation. Used by the journal reader for
  torn-write detection.
* :func:`from_jsonl` is the general-purpose decoder. It accepts BOTH
  length-prefixed lines (canonical) and raw JSON lines (back-compat for
  tests / smoke fixtures that emit envelope JSON directly). The format
  is autodetected via the presence of a ``\\t`` after a leading digit
  run.
* On Pydantic :class:`ValidationError`, ``strict=False`` (default for
  read paths) wraps the raw envelope in :class:`UnknownEvent`. ``strict=True``
  re-raises — producer-side guard.
* On JSON decode / framing failure (empty string, partial JSON, length
  mismatch), ``strict=False`` returns an :class:`UnknownEvent` whose
  ``raw_payload`` carries diagnostic crumbs (``_raw_line``,
  ``_decode_error``) and whose ``original_type`` is ``"<malformed>"``.
  ``strict=True`` raises :class:`MalformedEventError` with the original
  exception attached as ``cause`` — journal readers MUST use the strict
  variant when sweeping the tail for torn writes so they can distinguish
  malformed framing from forward-compat unknown types.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import ValidationError

from ryotenkai_shared.events.discriminator import EVENT_ADAPTER, Event
from ryotenkai_shared.events.envelope import BaseEvent, new_uuid7, utc_now
from ryotenkai_shared.events.types.unknown import UNKNOWN_OFFSET, UnknownEvent
from ryotenkai_shared.events.upcasters import apply_chain, latest_version_for


class MalformedEventError(Exception):
    """Raised by :func:`from_jsonl` (``strict=True``) on JSON/framing failure.

    Wraps the original exception (``cause``) and the offending line
    (truncated to 200 chars to keep logs / metrics bounded). The journal
    reader uses this to decide between "torn last line — truncate" and
    "actual logic bug — re-raise".

    Attributes:
        raw_line: the offending line, truncated to 200 characters.
        cause:    the original :class:`json.JSONDecodeError` or
                  :class:`ValueError` raised below the codec surface.
    """

    def __init__(self, raw_line: str, cause: Exception) -> None:
        self.raw_line = raw_line[:200]
        self.cause = cause
        super().__init__(f"malformed event line: {cause}; raw_line={self.raw_line!r}")


def to_jsonl(event: BaseEvent) -> str:
    """Serialize ``event`` as a length-prefixed JSONL line.

    Output ends with ``\\n``; callers write the string as-is to the
    journal file (no extra newline). The invariant: the line is
    self-delimiting because ``<length>`` allows readers to detect torn
    writes regardless of buffering boundaries.
    """
    body = event.model_dump_json()
    # UTF-8 byte length — multi-byte safe (see module docstring).
    n = len(body.encode("utf-8"))
    return f"{n}\t{body}\n"


def parse_length_prefix(line: str) -> tuple[int, str]:
    """Parse ``<length>\\t<json>\\n`` and verify the byte count matches.

    Returns ``(declared_length, json_body)``. Raises :class:`ValueError` if:

    * The line does not contain a tab after a digit run.
    * The declared length cannot be parsed as a non-negative integer.
    * The body's UTF-8 byte count differs from the declared length.
    * The line does not terminate with ``\\n`` (partial write).
    """
    if not line.endswith("\n"):
        raise ValueError("length-prefixed line missing trailing newline")
    tab_idx = line.find("\t")
    if tab_idx <= 0:
        raise ValueError("length-prefixed line missing tab after length")
    prefix = line[:tab_idx]
    if not prefix.isdigit():
        raise ValueError(f"length prefix is not a non-negative integer: {prefix!r}")
    declared = int(prefix)
    body = line[tab_idx + 1 : -1]  # strip tab and trailing newline
    actual = len(body.encode("utf-8"))
    if actual != declared:
        raise ValueError(
            f"length mismatch: declared {declared} bytes, got {actual}",
        )
    return declared, body


def _looks_length_prefixed(line: str) -> bool:
    """Heuristic: does ``line`` begin with ``<digit run>\\t``?

    Used by :func:`from_jsonl` to distinguish the canonical wire format
    from raw JSON (which always starts with ``{``). The heuristic is
    cheap and unambiguous because no valid JSON envelope starts with a
    digit followed by a tab.
    """
    tab_idx = line.find("\t")
    if tab_idx <= 0:
        return False
    prefix = line[:tab_idx]
    return prefix.isdigit()


def _build_unknown(raw: dict[str, Any]) -> UnknownEvent:
    """Wrap a raw envelope dict in :class:`UnknownEvent` for forward-compat.

    Pulls out enough fields to satisfy the envelope contract (filling
    defaults where the source omitted them) and stuffs the rest into
    ``raw_payload``. We intentionally swallow per-field parse errors so
    a malformed unknown event still produces a best-effort wrapper rather
    than crashing the read path.
    """
    original_type = raw.get("kind", "<missing>")

    def _safe_uuid(value: Any) -> UUID:
        if isinstance(value, str):
            try:
                return UUID(value)
            except ValueError:
                pass
        return new_uuid7()

    def _safe_time(value: Any) -> datetime:
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                pass
        return utc_now()

    def _safe_int(value: Any, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    severity = raw.get("severity", "info")
    if severity not in {"debug", "info", "warning", "error", "critical"}:
        severity = "info"

    return UnknownEvent(
        event_id=_safe_uuid(raw.get("event_id")),
        source=str(raw.get("source", "unknown")),
        time=_safe_time(raw.get("time")),
        run_id=str(raw.get("run_id", "unknown")),
        stage_id=raw.get("stage_id") if isinstance(raw.get("stage_id"), str) else None,
        offset=_safe_int(raw.get("offset"), UNKNOWN_OFFSET),
        schema_version=_safe_int(raw.get("schema_version"), 1),
        severity=severity,  # type: ignore[arg-type]  # validated against the Literal set above
        original_type=str(original_type),
        raw_payload=raw.get("payload", {}) if isinstance(raw.get("payload"), dict) else {},
    )


def _build_malformed_unknown(line: str, cause: Exception) -> UnknownEvent:
    """Wrap an undecodable JSONL line in :class:`UnknownEvent`.

    Used when the JSON parser or the length-prefix framing rejects the
    line before any envelope dict exists. We still want a returnable
    value in non-strict mode so callers (journal reader, SSE backfill)
    can render *something* and keep advancing past the torn write.

    The diagnostic crumbs (``_raw_line``, ``_decode_error``) are bounded
    at 200 chars apiece — enough for an operator to grep the bad line
    out of a journal without blowing up metrics labels.
    """
    return UnknownEvent(
        event_id=new_uuid7(),
        source="unknown",
        time=utc_now(),
        run_id="unknown",
        stage_id=None,
        offset=UNKNOWN_OFFSET,
        schema_version=1,
        severity="info",
        original_type="<malformed>",
        raw_payload={
            "_raw_line": line[:200],
            "_decode_error": str(cause)[:200],
        },
    )


def from_jsonl(line: str, *, strict: bool = False) -> Event:
    """Decode a JSONL line into a concrete :class:`Event` variant.

    Accepts BOTH wire formats:

    * Length-prefixed: ``<length>\\t<envelope_json>\\n`` (canonical).
    * Raw JSON:        ``<envelope_json>`` or ``<envelope_json>\\n``
      (back-compat for tests / non-journal sources).

    Failure modes and ``strict``:

    * Unknown ``kind`` (envelope decoded, type not in the union):

      * ``strict=True``  → re-raise :class:`pydantic.ValidationError`.
      * ``strict=False`` → wrap in :class:`UnknownEvent`.

    * JSON / framing failure (empty string, partial JSON, length
      prefix mismatch, non-object root):

      * ``strict=True``  → raise :class:`MalformedEventError`.
      * ``strict=False`` → wrap in :class:`UnknownEvent` with
        ``original_type='<malformed>'`` and diagnostic crumbs.

    The codec consults the upcaster registry before validation, so older
    payloads are migrated to the current schema if a hop is registered.
    """
    try:
        if _looks_length_prefixed(line):
            _, body = parse_length_prefix(line)
        else:
            body = line.rstrip("\n")
        raw = json.loads(body)
        if not isinstance(raw, dict):
            raise ValueError(
                f"event JSON must decode to an object, got {type(raw).__name__}",
            )
    except (json.JSONDecodeError, ValueError) as exc:
        if strict:
            raise MalformedEventError(line, exc) from exc
        return _build_malformed_unknown(line, exc)

    event_kind = raw.get("kind")
    current_version = raw.get("schema_version", 1)
    if isinstance(event_kind, str) and isinstance(current_version, int):
        target_version = latest_version_for(event_kind)
        if current_version < target_version:
            raw = apply_chain(raw, event_kind, current_version, target_version)

    try:
        return EVENT_ADAPTER.validate_python(raw)
    except ValidationError:
        if strict:
            raise
        return _build_unknown(raw)


__all__ = [
    "MalformedEventError",
    "from_jsonl",
    "parse_length_prefix",
    "to_jsonl",
]
