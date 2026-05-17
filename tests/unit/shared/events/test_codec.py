"""Unit tests: :mod:`ryotenkai_shared.events.codec`.

The codec is the choke point that defines durability: every bug here
becomes a torn-write / data-loss bug in the journal. Tests pin:

* The wire format (length-prefix using UTF-8 byte count).
* Round-trip identity for known events.
* Unknown-type forward-compat behaviour under ``strict=False`` vs True.
* Tolerance of multi-byte payloads (R-04 mitigation).
* Behaviour on malformed lines.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import ValidationError

from ryotenkai_shared.events import (
    EVENT_ADAPTER,
    UNKNOWN_OFFSET,
    BaseEvent,
    Event,
    MalformedEventError,
    UnknownEvent,
    from_jsonl,
    parse_length_prefix,
    to_jsonl,
)
from ryotenkai_shared.events.types.pod_memory import (
    MemoryCacheClearedEvent,
    MemoryCacheClearedPayload,
)
from ryotenkai_shared.events.types.pod_training import (
    TrainingFailedEvent,
    TrainingFailedPayload,
    TrainingStartedEvent,
    TrainingStartedPayload,
)


def _training_started(**overrides) -> TrainingStartedEvent:
    payload = TrainingStartedPayload(
        max_steps=1000,
        num_train_epochs=3,
        per_device_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=2e-5,
        algorithm="sft",
    )
    defaults: dict[str, Any] = {
        "source": "pod://run-1/trainer",
        "run_id": "run-1",
        "offset": 0,
        "payload": payload,
    }
    defaults.update(overrides)
    return TrainingStartedEvent(**defaults)


# ===========================================================================
# 1. Positive
# ===========================================================================


class TestPositive:
    def test_round_trip_preserves_identity(self) -> None:
        original = _training_started()
        line = to_jsonl(original)
        restored = from_jsonl(line, strict=True)
        assert isinstance(restored, TrainingStartedEvent)
        assert restored == original

    def test_round_trip_for_two_distinct_types_keeps_them_distinct(self) -> None:
        started = _training_started()
        memory = MemoryCacheClearedEvent(
            source="pod://run-1/trainer",
            run_id="run-1",
            offset=1,
            payload=MemoryCacheClearedPayload(
                device="cuda:0",
                before_bytes=10_000_000_000,
                after_bytes=2_000_000_000,
                trigger="threshold",
            ),
        )
        lines = [to_jsonl(started), to_jsonl(memory)]
        results = [from_jsonl(line, strict=True) for line in lines]
        assert isinstance(results[0], TrainingStartedEvent)
        assert isinstance(results[1], MemoryCacheClearedEvent)

    def test_raw_json_line_without_prefix_still_decodes(self) -> None:
        # Back-compat: tests / smoke fixtures may emit raw JSON without
        # the length prefix. The codec should still decode them.
        original = _training_started()
        raw = original.model_dump_json()  # no prefix
        restored = from_jsonl(raw, strict=True)
        assert isinstance(restored, TrainingStartedEvent)
        assert restored == original


# ===========================================================================
# 2. Negative
# ===========================================================================


class TestNegative:
    def test_unknown_type_in_strict_mode_raises(self) -> None:
        raw = {
            "kind": "ryotenkai.pod.future.gradient_anomaly",
            "source": "pod://run/trainer",
            "time": "2026-05-16T12:00:00+00:00",
            "run_id": "run",
            "offset": 0,
            "severity": "info",
            "payload": {"spike": 4.2},
        }
        with pytest.raises(ValidationError):
            from_jsonl(json.dumps(raw), strict=True)

    def test_unknown_type_in_lax_mode_wraps_in_unknown_event(self) -> None:
        raw = {
            "kind": "ryotenkai.pod.future.gradient_anomaly",
            "source": "pod://run/trainer",
            "time": "2026-05-16T12:00:00+00:00",
            "run_id": "run",
            "offset": 42,
            "schema_version": 7,
            "severity": "warning",
            "payload": {"spike": 4.2, "step": 100},
        }
        event = from_jsonl(json.dumps(raw), strict=False)
        assert isinstance(event, UnknownEvent)
        assert event.original_type == "ryotenkai.pod.future.gradient_anomaly"
        assert event.raw_payload == {"spike": 4.2, "step": 100}
        assert event.severity == "warning"
        assert event.offset == 42

    def test_missing_payload_field_in_strict_mode_raises(self) -> None:
        line = to_jsonl(_training_started())
        _prefix, body = line.split("\t", 1)
        data = json.loads(body[:-1])  # strip newline
        del data["payload"]["learning_rate"]
        body_new = json.dumps(data)
        # Recompute prefix because byte length changed.
        new_prefix = len(body_new.encode("utf-8"))
        line_new = f"{new_prefix}\t{body_new}\n"
        with pytest.raises(ValidationError):
            from_jsonl(line_new, strict=True)

    def test_empty_string_strict_raises_malformed(self) -> None:
        # An empty string has no JSON to parse; journal readers will
        # encounter this during file truncation, so the strict mode MUST
        # surface a typed MalformedEventError rather than letting
        # json.JSONDecodeError leak through.
        with pytest.raises(MalformedEventError) as exc:
            from_jsonl("", strict=True)
        assert isinstance(exc.value.cause, (json.JSONDecodeError, ValueError))
        assert exc.value.raw_line == ""

    def test_empty_string_lax_returns_unknown(self) -> None:
        event = from_jsonl("", strict=False)
        assert isinstance(event, UnknownEvent)
        assert event.original_type == "<malformed>"
        assert event.offset == UNKNOWN_OFFSET
        assert "_decode_error" in event.raw_payload

    def test_partial_json_strict_raises_malformed(self) -> None:
        with pytest.raises(MalformedEventError) as exc:
            from_jsonl("{bad", strict=True)
        assert isinstance(exc.value.cause, json.JSONDecodeError)
        assert "{bad" in exc.value.raw_line

    def test_partial_json_lax_returns_unknown(self) -> None:
        event = from_jsonl("{bad", strict=False)
        assert isinstance(event, UnknownEvent)
        assert event.original_type == "<malformed>"
        assert event.raw_payload["_raw_line"] == "{bad"

    def test_only_newline_lax_returns_unknown(self) -> None:
        # A bare newline simulates a flushed-but-empty record at the
        # tail of the journal — must NOT crash the reader.
        event = from_jsonl("\n", strict=False)
        assert isinstance(event, UnknownEvent)
        assert event.original_type == "<malformed>"

    def test_length_mismatch_strict_raises_malformed(self) -> None:
        # Declared 9 bytes, body is 4 — parse_length_prefix raises
        # ValueError. from_jsonl(strict=True) must rewrap as
        # MalformedEventError so callers can branch on type.
        with pytest.raises(MalformedEventError) as exc:
            from_jsonl("9\tabcd\n", strict=True)
        assert isinstance(exc.value.cause, ValueError)
        assert "length mismatch" in str(exc.value.cause)


# ===========================================================================
# 3. Boundary
# ===========================================================================


class TestBoundary:
    def test_empty_dict_payload_field_still_validates(self) -> None:
        # ``TrainingLogEvent`` carries metrics as ``dict[str, float]``;
        # an empty dict is valid per the schema.
        from ryotenkai_shared.events.types.pod_training import (
            TrainingLogEvent,
            TrainingLogPayload,
        )

        event = TrainingLogEvent(
            source="pod://r/trainer",
            run_id="r",
            offset=0,
            payload=TrainingLogPayload(step=10, metrics={}),
        )
        restored = from_jsonl(to_jsonl(event), strict=True)
        assert isinstance(restored, TrainingLogEvent)
        assert restored.payload.metrics == {}

    def test_large_metrics_dict_round_trips(self) -> None:
        from ryotenkai_shared.events.types.pod_training import (
            TrainingLogEvent,
            TrainingLogPayload,
        )

        metrics = {f"metric_{i}": float(i) for i in range(200)}
        event = TrainingLogEvent(
            source="pod://r/trainer",
            run_id="r",
            offset=0,
            payload=TrainingLogPayload(step=10, metrics=metrics),
        )
        restored = from_jsonl(to_jsonl(event), strict=True)
        assert isinstance(restored, TrainingLogEvent)
        assert restored.payload.metrics == metrics

    def test_multibyte_payload_passes_length_check(self) -> None:
        # An emoji is 4 UTF-8 bytes — using ``len(str)`` instead of
        # byte length would mis-count. The codec must use bytes.
        event = TrainingFailedEvent(
            source="pod://r/trainer",
            run_id="r",
            offset=0,
            payload=TrainingFailedPayload(
                error_type="RuntimeError",
                message="GPU melted",
                traceback_excerpt="Traceback...",
                step=99,
            ),
        )
        # Mutate the payload through model_copy to inject multibyte.
        event = event.model_copy(
            update={
                "payload": TrainingFailedPayload(
                    error_type="RuntimeError",
                    message="loss explodiendo ",
                    traceback_excerpt="Traceback ...",
                    step=99,
                ),
            },
        )
        line = to_jsonl(event)
        # parse_length_prefix verifies the byte count — no exception.
        declared, body = parse_length_prefix(line)
        assert declared == len(body.encode("utf-8"))
        restored = from_jsonl(line, strict=True)
        assert isinstance(restored, TrainingFailedEvent)
        assert restored.payload.message == event.payload.message


# ===========================================================================
# 4. Invariants
# ===========================================================================


class TestInvariants:
    def test_to_jsonl_line_ends_with_newline(self) -> None:
        line = to_jsonl(_training_started())
        assert line.endswith("\n")

    def test_length_prefix_matches_utf8_byte_length(self) -> None:
        line = to_jsonl(_training_started())
        prefix_str, body_with_nl = line.split("\t", 1)
        body = body_with_nl[:-1]
        assert int(prefix_str) == len(body.encode("utf-8"))

    def test_parse_length_prefix_reports_same_pair_as_emitted(self) -> None:
        line = to_jsonl(_training_started())
        declared, body = parse_length_prefix(line)
        # Re-encode and check we can re-parse and compare.
        assert declared == len(body.encode("utf-8"))

    def test_round_trip_through_event_adapter_preserves_concrete_type(self) -> None:
        original = _training_started()
        raw = original.model_dump()
        validated = EVENT_ADAPTER.validate_python(raw)
        assert type(validated) is TrainingStartedEvent


# ===========================================================================
# 5. Dependency errors
# ===========================================================================


class TestDependencyErrors:
    def test_parse_length_prefix_rejects_missing_tab(self) -> None:
        with pytest.raises(ValueError):
            parse_length_prefix("123abc\n")

    def test_parse_length_prefix_rejects_non_digit_prefix(self) -> None:
        with pytest.raises(ValueError):
            parse_length_prefix("abc\tbody\n")

    def test_parse_length_prefix_rejects_missing_newline(self) -> None:
        # Torn write: trailing newline missing → partial line.
        with pytest.raises(ValueError):
            parse_length_prefix("4\tbody")

    def test_parse_length_prefix_rejects_byte_count_mismatch(self) -> None:
        # Declared 9 bytes but body is 4 — torn write detected.
        with pytest.raises(ValueError):
            parse_length_prefix("9\tabcd\n")

    def test_from_jsonl_rejects_non_object_json(self) -> None:
        # Strict mode wraps any JSON / framing failure in MalformedEventError,
        # exposing the raw line and underlying cause for journal readers.
        with pytest.raises(MalformedEventError) as exc:
            from_jsonl("42\n", strict=True)
        assert isinstance(exc.value.cause, ValueError)


# ===========================================================================
# 6. Regressions
# ===========================================================================


class TestRegressions:
    def test_unknown_event_lax_with_missing_optional_envelope_fields(self) -> None:
        # A producer-from-the-future might omit fields we don't recognise;
        # the lax wrapper must still return *something*.
        raw = {"kind": "ryotenkai.unseen.brand_new", "severity": "info"}
        event = from_jsonl(json.dumps(raw), strict=False)
        assert isinstance(event, UnknownEvent)
        assert event.original_type == "ryotenkai.unseen.brand_new"
        assert event.offset == UNKNOWN_OFFSET  # sentinel for "we don't know"
        assert event.raw_payload == {}

    def test_unknown_event_lax_preserves_unknown_severity_as_info(self) -> None:
        # If the future producer used a severity outside our literal,
        # the lax decoder should clamp to a safe default rather than
        # crash.
        raw = {
            "kind": "ryotenkai.unseen.new",
            "severity": "extra-loud",
            "source": "x",
            "run_id": "x",
            "offset": 0,
            "payload": {},
        }
        event = from_jsonl(json.dumps(raw), strict=False)
        assert isinstance(event, UnknownEvent)
        assert event.severity == "info"

    def test_journal_reader_can_skip_torn_writes(self) -> None:
        # Simulates the Phase 3 journal-reader hot path: a JSONL stream
        # of one well-formed line followed by a torn write at the tail.
        # In lax mode the reader must not crash — it must yield the
        # valid event and a wrapped UnknownEvent for the torn line so
        # iteration can complete.
        valid_event = _training_started()
        valid_line = to_jsonl(valid_event)
        # Simulate `kill -9` mid-write: declared length 9999 but body 4 chars.
        torn_line = "9999\tabcd\n"

        results = [from_jsonl(line, strict=False) for line in (valid_line, torn_line)]

        assert isinstance(results[0], TrainingStartedEvent)
        assert results[0] == valid_event
        assert isinstance(results[1], UnknownEvent)
        assert results[1].original_type == "<malformed>"
        assert results[1].offset == UNKNOWN_OFFSET


# ===========================================================================
# 7. Logic-specific
# ===========================================================================


class TestLogicSpecific:
    def test_strict_mode_does_not_consult_upcasters_for_unknown_type(self) -> None:
        # Even with strict=True the codec consults upcasters first;
        # because no upcaster is registered for an unseen type, the
        # call falls through to validation and raises.
        raw = {
            "kind": "ryotenkai.unseen.new",
            "schema_version": 1,
            "source": "x",
            "run_id": "x",
            "offset": 0,
            "severity": "info",
            "payload": {},
        }
        with pytest.raises(ValidationError):
            from_jsonl(json.dumps(raw), strict=True)

    def test_to_jsonl_is_idempotent_via_serialize_deserialize(self) -> None:
        original = _training_started()
        first = from_jsonl(to_jsonl(original), strict=True)
        second = from_jsonl(to_jsonl(first), strict=True)
        assert first == second == original

    def test_to_jsonl_accepts_any_event_base_subclass(self) -> None:
        # to_jsonl is typed against BaseEvent — any concrete subclass
        # produces a parseable line. Use a different family to widen
        # coverage.
        event = MemoryCacheClearedEvent(
            source="pod://r/trainer",
            run_id="r",
            offset=0,
            payload=MemoryCacheClearedPayload(
                device="cuda:0",
                before_bytes=1,
                after_bytes=0,
                trigger="manual",
            ),
        )
        line = to_jsonl(event)
        restored: Event = from_jsonl(line, strict=True)
        # The discriminator dispatched back to the same concrete type.
        assert type(restored) is MemoryCacheClearedEvent
        # And `BaseEvent` is its ancestor — sanity check on the runtime
        # type hierarchy used by emitters.
        assert isinstance(restored, BaseEvent)
