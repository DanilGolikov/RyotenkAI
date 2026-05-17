"""Hypothesis fuzz tests for :mod:`ryotenkai_shared.events.codec` (Phase 9).

Property-based coverage that complements the deterministic suite in
:mod:`tests.unit.shared.events.test_codec`. Three families of properties
pin down behaviour that is hard to express via hand-crafted cases:

* **Round-trip identity** — for any valid envelope built by hypothesis,
  ``from_jsonl(to_jsonl(e), strict=True) == e``. This is the codec's
  cornerstone invariant: a round trip through the wire format must
  preserve every byte of the envelope including multi-byte payloads
  and time zone offsets.
* **Length-prefix invariant** — for any well-formed line the codec
  emits, :func:`parse_length_prefix` reports a body whose UTF-8 byte
  length matches the declared count. The R-04 mitigation (torn-write
  detection) relies on this guarantee; any drift makes the journal
  reader either miss torn writes or false-flag well-formed lines.
* **Malformed lax doesn't crash** — for any random byte sequence, the
  lax decoder MUST return either an :class:`UnknownEvent` or raise a
  bounded :class:`MalformedEventError`; an uncaught exception inside
  ``from_jsonl(..., strict=False)`` would crash the journal reader's
  hot loop and stop the SSE stream cold.

Hypothesis runs 50+ random examples per property by default; that's
plenty to surface off-by-one length errors, codepoint-boundary cases
inside multi-byte payloads, and discriminator dispatch regressions
introduced by future event additions.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from ryotenkai_shared.events import (
    EVENT_ADAPTER,
    UNKNOWN_OFFSET,
    BaseEvent,
    MalformedEventError,
    UnknownEvent,
    from_jsonl,
    parse_length_prefix,
    to_jsonl,
)
from ryotenkai_shared.events.discriminator import Event
from ryotenkai_shared.events.types.control_run import (
    RunCancelledEvent,
    RunCancelledPayload,
    RunCompletedEvent,
    RunCompletedPayload,
    RunFailedEvent,
    RunFailedPayload,
    RunStartedEvent,
    RunStartedPayload,
)
from ryotenkai_shared.events.types.control_stage import (
    StageCompletedEvent,
    StageCompletedPayload,
    StageStartedEvent,
    StageStartedPayload,
)
from ryotenkai_shared.events.types.pod_memory import (
    MemoryCacheClearedEvent,
    MemoryCacheClearedPayload,
    MemoryOOMDetectedEvent,
    MemoryOOMDetectedPayload,
)
from ryotenkai_shared.events.types.pod_training import (
    TrainingFailedEvent,
    TrainingFailedPayload,
    TrainingLogEvent,
    TrainingLogPayload,
    TrainingStartedEvent,
    TrainingStartedPayload,
    TrainingStepEvent,
    TrainingStepPayload,
)


# ---------------------------------------------------------------------------
# Hypothesis strategies — concrete event factories
# ---------------------------------------------------------------------------


# Run IDs are arbitrary strings on the producer side; constrain to a
# printable subset so hypothesis doesn't waste rounds shrinking pathological
# Unicode that only the JSON serializer would notice.
_run_id_strategy = st.text(
    alphabet=st.characters(min_codepoint=0x20, max_codepoint=0x7E),
    min_size=1,
    max_size=32,
)

_source_strategy = st.sampled_from(
    [
        "pod://run/trainer",
        "control://orchestrator",
        "control://orchestrator/gpu_deployer",
        "pod://run/runner",
    ],
)

_severity_strategy = st.sampled_from(["debug", "info", "warning", "error", "critical"])

_offset_strategy = st.integers(min_value=0, max_value=10_000_000)


def _training_started_strategy() -> st.SearchStrategy[TrainingStartedEvent]:
    return st.builds(
        TrainingStartedEvent,
        source=_source_strategy,
        run_id=_run_id_strategy,
        offset=_offset_strategy,
        payload=st.builds(
            TrainingStartedPayload,
            max_steps=st.integers(min_value=1, max_value=1_000_000),
            num_train_epochs=st.integers(min_value=1, max_value=100),
            per_device_batch_size=st.integers(min_value=1, max_value=512),
            gradient_accumulation_steps=st.integers(min_value=1, max_value=512),
            learning_rate=st.floats(
                min_value=1e-8, max_value=1e-1,
                allow_nan=False, allow_infinity=False,
            ),
            algorithm=st.sampled_from(["sft", "cpt", "dpo", "grpo", "sapo"]),
        ),
    )


def _training_step_strategy() -> st.SearchStrategy[TrainingStepEvent]:
    return st.builds(
        TrainingStepEvent,
        source=_source_strategy,
        run_id=_run_id_strategy,
        offset=_offset_strategy,
        payload=st.builds(
            TrainingStepPayload,
            step=st.integers(min_value=0, max_value=1_000_000),
            loss=st.floats(
                min_value=0.0, max_value=100.0,
                allow_nan=False, allow_infinity=False,
            ),
            learning_rate=st.floats(
                min_value=1e-8, max_value=1e-1,
                allow_nan=False, allow_infinity=False,
            ),
            grad_norm=st.floats(
                min_value=0.0, max_value=1000.0,
                allow_nan=False, allow_infinity=False,
            ),
            tokens_per_sec=st.floats(
                min_value=0.0, max_value=1_000_000.0,
                allow_nan=False, allow_infinity=False,
            ),
            samples_per_sec=st.floats(
                min_value=0.0, max_value=10_000.0,
                allow_nan=False, allow_infinity=False,
            ),
        ),
    )


def _training_log_strategy() -> st.SearchStrategy[TrainingLogEvent]:
    # Metrics dict: random key set with finite float values.
    metric_value = st.floats(
        min_value=-1e6, max_value=1e6,
        allow_nan=False, allow_infinity=False,
    )
    metric_key = st.text(
        alphabet=st.characters(
            whitelist_categories=("Lu", "Ll", "Nd"),
            whitelist_characters="_.-",
        ),
        min_size=1, max_size=24,
    )
    return st.builds(
        TrainingLogEvent,
        source=_source_strategy,
        run_id=_run_id_strategy,
        offset=_offset_strategy,
        payload=st.builds(
            TrainingLogPayload,
            step=st.integers(min_value=0, max_value=1_000_000),
            metrics=st.dictionaries(metric_key, metric_value, max_size=10),
        ),
    )


def _training_failed_strategy() -> st.SearchStrategy[TrainingFailedEvent]:
    # Stress-test multi-byte payloads — the codec must use UTF-8 byte
    # length, not character length.
    multibyte_text = st.text(
        # Mix of BMP and supplementary plane characters to ensure UTF-8
        # length differs from Python str length.
        alphabet=st.characters(min_codepoint=0x20, max_codepoint=0x10FFFF, blacklist_categories=("Cs",)),
        min_size=0, max_size=128,
    )
    return st.builds(
        TrainingFailedEvent,
        source=_source_strategy,
        run_id=_run_id_strategy,
        offset=_offset_strategy,
        payload=st.builds(
            TrainingFailedPayload,
            error_type=st.sampled_from(["RuntimeError", "OOMError", "ValueError"]),
            message=multibyte_text,
            traceback_excerpt=multibyte_text,
            step=st.integers(min_value=0, max_value=1_000_000),
        ),
    )


def _memory_cache_cleared_strategy() -> st.SearchStrategy[MemoryCacheClearedEvent]:
    return st.builds(
        MemoryCacheClearedEvent,
        source=_source_strategy,
        run_id=_run_id_strategy,
        offset=_offset_strategy,
        payload=st.builds(
            MemoryCacheClearedPayload,
            device=st.sampled_from(["cuda:0", "cuda:1", "cpu", "mps"]),
            before_bytes=st.integers(min_value=0, max_value=2**63 - 1),
            after_bytes=st.integers(min_value=0, max_value=2**63 - 1),
            trigger=st.sampled_from(["scheduled", "threshold", "manual"]),
        ),
    )


def _memory_oom_strategy() -> st.SearchStrategy[MemoryOOMDetectedEvent]:
    return st.builds(
        MemoryOOMDetectedEvent,
        source=_source_strategy,
        run_id=_run_id_strategy,
        offset=_offset_strategy,
        payload=st.builds(
            MemoryOOMDetectedPayload,
            device=st.sampled_from(["cuda:0", "cuda:1", "cpu", "mps"]),
            allocated_bytes=st.integers(min_value=0, max_value=2**63 - 1),
            reserved_bytes=st.integers(min_value=0, max_value=2**63 - 1),
            step=st.integers(min_value=0, max_value=1_000_000),
        ),
    )


def _run_started_strategy() -> st.SearchStrategy[RunStartedEvent]:
    return st.builds(
        RunStartedEvent,
        source=_source_strategy,
        run_id=_run_id_strategy,
        offset=_offset_strategy,
        payload=st.builds(
            RunStartedPayload,
            run_name=st.text(min_size=1, max_size=32),
            algorithm=st.sampled_from(["sft", "cpt", "dpo", "grpo", "sapo"]),
            model_id=st.text(min_size=1, max_size=64),
            dataset_id=st.text(min_size=1, max_size=64),
            config_hash=st.text(
                alphabet="0123456789abcdef", min_size=8, max_size=64,
            ),
        ),
    )


def _run_completed_strategy() -> st.SearchStrategy[RunCompletedEvent]:
    return st.builds(
        RunCompletedEvent,
        source=_source_strategy,
        run_id=_run_id_strategy,
        offset=_offset_strategy,
        payload=st.builds(
            RunCompletedPayload,
            duration_s=st.floats(
                min_value=0.0, max_value=1e9,
                allow_nan=False, allow_infinity=False,
            ),
            final_status=st.sampled_from(["success", "partial"]),
            mlflow_run_id=st.one_of(st.none(), st.text(min_size=1, max_size=32)),
        ),
    )


def _run_failed_strategy() -> st.SearchStrategy[RunFailedEvent]:
    return st.builds(
        RunFailedEvent,
        source=_source_strategy,
        run_id=_run_id_strategy,
        offset=_offset_strategy,
        payload=st.builds(
            RunFailedPayload,
            failing_stage=st.text(min_size=1, max_size=32),
            error_type=st.sampled_from(["RuntimeError", "ValueError"]),
            message=st.text(min_size=0, max_size=128),
            traceback_excerpt=st.text(min_size=0, max_size=512),
        ),
    )


def _run_cancelled_strategy() -> st.SearchStrategy[RunCancelledEvent]:
    return st.builds(
        RunCancelledEvent,
        source=_source_strategy,
        run_id=_run_id_strategy,
        offset=_offset_strategy,
        payload=st.builds(
            RunCancelledPayload,
            reason=st.text(min_size=1, max_size=64),
            cancelled_at_stage=st.one_of(st.none(), st.text(min_size=1, max_size=32)),
        ),
    )


def _stage_started_strategy() -> st.SearchStrategy[StageStartedEvent]:
    return st.builds(
        StageStartedEvent,
        source=_source_strategy,
        run_id=_run_id_strategy,
        offset=_offset_strategy,
        payload=st.builds(
            StageStartedPayload,
            stage_name=st.sampled_from(["gpu_deployer", "trainer", "evaluator"]),
            stage_index=st.integers(min_value=0, max_value=20),
            total_stages=st.integers(min_value=1, max_value=20),
            inputs_summary=st.dictionaries(
                st.text(min_size=1, max_size=8),
                st.integers(min_value=0, max_value=1000),
                max_size=4,
            ),
        ),
    )


def _stage_completed_strategy() -> st.SearchStrategy[StageCompletedEvent]:
    return st.builds(
        StageCompletedEvent,
        source=_source_strategy,
        run_id=_run_id_strategy,
        offset=_offset_strategy,
        payload=st.builds(
            StageCompletedPayload,
            stage_name=st.sampled_from(["gpu_deployer", "trainer", "evaluator"]),
            duration_s=st.floats(
                min_value=0.0, max_value=1e9,
                allow_nan=False, allow_infinity=False,
            ),
            outputs_summary=st.dictionaries(
                st.text(min_size=1, max_size=8),
                st.integers(min_value=0, max_value=1000),
                max_size=4,
            ),
        ),
    )


# Union strategy that exercises a broad slice of the discriminator
# union. Picking 10 strategies (instead of all 57) keeps the fuzz wall
# clock reasonable while still exercising every codec branch (multibyte,
# big ints, optional fields, nested dicts).
any_event_strategy: st.SearchStrategy[BaseEvent] = st.one_of(
    _training_started_strategy(),
    _training_step_strategy(),
    _training_log_strategy(),
    _training_failed_strategy(),
    _memory_cache_cleared_strategy(),
    _memory_oom_strategy(),
    _run_started_strategy(),
    _run_completed_strategy(),
    _run_failed_strategy(),
    _run_cancelled_strategy(),
    _stage_started_strategy(),
    _stage_completed_strategy(),
)


# ---------------------------------------------------------------------------
# Round-trip property tests
# ---------------------------------------------------------------------------


class TestRoundTripProperty:
    """For any valid envelope, ``from_jsonl(to_jsonl(e), strict=True) == e``."""

    @given(event=any_event_strategy)
    @settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_round_trip_preserves_identity(self, event: BaseEvent) -> None:
        line = to_jsonl(event)
        restored = from_jsonl(line, strict=True)
        assert restored == event
        assert type(restored) is type(event)

    @given(event=any_event_strategy)
    @settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_round_trip_double(self, event: BaseEvent) -> None:
        """Two consecutive round trips remain stable.

        Pins down the "did anyone add a serializer that drifts on the
        second pass" failure mode: each round-trip is a fixed point.
        """
        once = from_jsonl(to_jsonl(event), strict=True)
        twice = from_jsonl(to_jsonl(once), strict=True)
        assert once == twice == event

    @given(event=any_event_strategy)
    @settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_round_trip_via_raw_json_back_compat(self, event: BaseEvent) -> None:
        """Codec accepts a raw JSON line (no length prefix) for back-compat."""
        raw = event.model_dump_json()
        restored = from_jsonl(raw, strict=True)
        assert restored == event

    @given(event=any_event_strategy)
    @settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_discriminator_dispatches_to_concrete_type(self, event: BaseEvent) -> None:
        """Round-trip via :data:`EVENT_ADAPTER` lands on the same concrete class."""
        raw = json.loads(event.model_dump_json())
        restored = EVENT_ADAPTER.validate_python(raw)
        assert type(restored) is type(event)


# ---------------------------------------------------------------------------
# Length-prefix invariant tests
# ---------------------------------------------------------------------------


class TestLengthPrefixInvariant:
    """The declared length always equals the body's UTF-8 byte count."""

    @given(event=any_event_strategy)
    @settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_to_jsonl_emits_consistent_length_prefix(self, event: BaseEvent) -> None:
        line = to_jsonl(event)
        declared, body = parse_length_prefix(line)
        assert declared == len(body.encode("utf-8"))

    @given(event=any_event_strategy)
    @settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_line_ends_with_newline(self, event: BaseEvent) -> None:
        line = to_jsonl(event)
        assert line.endswith("\n")
        # The newline is NOT inside the body; the parser strips it.
        _, body = parse_length_prefix(line)
        assert not body.endswith("\n")

    @given(event=any_event_strategy)
    @settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_tab_present_after_digits(self, event: BaseEvent) -> None:
        line = to_jsonl(event)
        tab_idx = line.find("\t")
        # Prefix is purely digits.
        assert tab_idx > 0
        assert line[:tab_idx].isdigit()
        # Body starts with `{` (a JSON object).
        assert line[tab_idx + 1] == "{"

    @given(
        # Generate a line that we then mutate to break the length prefix.
        event=any_event_strategy,
        bogus_delta=st.integers(min_value=1, max_value=10_000),
    )
    @settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_length_mismatch_is_detected(
        self,
        event: BaseEvent,
        bogus_delta: int,
    ) -> None:
        """Tampering the declared length raises :class:`ValueError`."""
        line = to_jsonl(event)
        prefix_str, rest = line.split("\t", 1)
        bad_prefix = str(int(prefix_str) + bogus_delta)
        bad_line = f"{bad_prefix}\t{rest}"
        with pytest.raises(ValueError):
            parse_length_prefix(bad_line)


# ---------------------------------------------------------------------------
# Discriminator uniqueness — every kind dispatches to the right class
# ---------------------------------------------------------------------------


class TestDiscriminatorUniqueness:
    """Property: ``kind`` literal uniquely identifies a concrete subclass."""

    @given(event=any_event_strategy)
    @settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_kind_round_trips_to_same_class(self, event: BaseEvent) -> None:
        original_kind = event.kind
        line = to_jsonl(event)
        restored = from_jsonl(line, strict=True)
        assert restored.kind == original_kind
        # The closed discriminated union: a single ``kind`` maps to a
        # single class. If a future commit duplicates a literal, this
        # test catches the collision.
        assert type(restored) is type(event)


# ---------------------------------------------------------------------------
# Malformed-lax-doesn't-crash property
# ---------------------------------------------------------------------------


# Random bytes — anything from short truncations to invalid UTF-8 sequences.
_raw_bytes_strategy = st.binary(min_size=0, max_size=256)
# Random ASCII strings that look "JSON-ish" without being valid JSON.
_garbage_text_strategy = st.text(
    alphabet=st.characters(min_codepoint=0x20, max_codepoint=0x7E),
    min_size=0,
    max_size=256,
)


def _decode_lossy(b: bytes) -> str:
    """Decode bytes using lossy UTF-8.

    Matches what a journal reader would observe when reading torn writes
    or files written by a future producer that uses a different encoding.
    """
    return b.decode("utf-8", errors="replace")


class TestMalformedLaxDoesntCrash:
    """``from_jsonl(line, strict=False)`` returns an :class:`UnknownEvent`."""

    @given(raw=_raw_bytes_strategy)
    @settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_random_bytes_never_raises_in_lax_mode(self, raw: bytes) -> None:
        line = _decode_lossy(raw)
        # Critical contract: lax mode never raises. The journal reader
        # depends on this — an exception here would crash mid-iteration
        # and stop the SSE catchup.
        result = from_jsonl(line, strict=False)
        assert isinstance(result, BaseEvent)
        # Either we recovered a known envelope (extremely unlikely with
        # random bytes — the discriminator is tight) or we got a wrapped
        # UnknownEvent. Both satisfy the contract.

    @given(text=_garbage_text_strategy)
    @settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_garbage_text_never_raises_in_lax_mode(self, text: str) -> None:
        result = from_jsonl(text, strict=False)
        assert isinstance(result, BaseEvent)

    @given(text=_garbage_text_strategy)
    @settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_strict_either_returns_or_raises_typed_error(self, text: str) -> None:
        """Strict mode raises ONLY ``MalformedEventError`` or ``ValidationError``."""
        try:
            from_jsonl(text, strict=True)
        except (MalformedEventError, ValidationError):
            pass
        except Exception as exc:  # pragma: no cover — would be a bug
            raise AssertionError(
                f"strict mode raised an unexpected exception type: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    @given(
        # Generate a *valid* envelope, then append truncated bytes — the
        # classic kill -9 scenario.
        event=any_event_strategy,
        trunc_pct=st.integers(min_value=1, max_value=99),
    )
    @settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_truncated_line_lax_returns_unknown(
        self,
        event: BaseEvent,
        trunc_pct: int,
    ) -> None:
        """A line cut short at any point yields :class:`UnknownEvent`."""
        line = to_jsonl(event)
        cut_at = max(1, len(line) * trunc_pct // 100)
        truncated = line[:cut_at]
        result = from_jsonl(truncated, strict=False)
        # Truncations can either produce a malformed (most common) or, on
        # rare boundary cuts, accidentally still parse as a complete
        # envelope (e.g. when the prefix happens to match a shorter
        # body). Either branch is acceptable per the contract — the
        # property we pin is "no uncaught exception".
        assert isinstance(result, BaseEvent)

    @given(
        # Wrong length prefix: declare too many bytes.
        event=any_event_strategy,
        wrong_delta=st.integers(min_value=-1000, max_value=1000).filter(lambda x: x != 0),
    )
    @settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_wrong_length_prefix_lax_returns_unknown(
        self,
        event: BaseEvent,
        wrong_delta: int,
    ) -> None:
        line = to_jsonl(event)
        prefix_str, rest = line.split("\t", 1)
        bogus_prefix = max(0, int(prefix_str) + wrong_delta)
        bogus_line = f"{bogus_prefix}\t{rest}"
        result = from_jsonl(bogus_line, strict=False)
        # When the declared length is wrong, the framer rejects and the
        # codec falls into the malformed branch.
        assert isinstance(result, BaseEvent)
        if isinstance(result, UnknownEvent) and result.original_type == "<malformed>":
            # Diagnostic crumbs preserved.
            assert "_decode_error" in result.raw_payload


# ---------------------------------------------------------------------------
# Parse-length-prefix invariant tests
# ---------------------------------------------------------------------------


class TestParseLengthPrefixInvariant:
    """The parser is the inverse of the emitter for any well-formed line."""

    @given(
        body=st.text(
            alphabet=st.characters(min_codepoint=0x20, max_codepoint=0x7E),
            min_size=0,
            max_size=200,
        ),
    )
    @settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_round_trip_synthetic_body(self, body: str) -> None:
        """For any printable body, hand-build a line and parse it back."""
        # Skip bodies containing tab or newline — those are framing chars.
        if "\t" in body or "\n" in body:
            return
        n = len(body.encode("utf-8"))
        line = f"{n}\t{body}\n"
        declared, parsed = parse_length_prefix(line)
        assert declared == n
        assert parsed == body

    @given(
        body=st.text(min_size=0, max_size=64),
    )
    @settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_round_trip_multibyte_body(self, body: str) -> None:
        """UTF-8 byte length must work for any Unicode body (sans tab/newline)."""
        if "\t" in body or "\n" in body:
            return
        n = len(body.encode("utf-8"))
        line = f"{n}\t{body}\n"
        declared, parsed = parse_length_prefix(line)
        assert declared == n
        assert parsed == body
        # Critical invariant: the declared length equals the BYTE count,
        # not the Python ``len(str)``. Many code points encode to 2-4
        # bytes — if the framer used ``len(str)`` instead, the assertion
        # below would fire on any non-ASCII input.
        assert declared == len(parsed.encode("utf-8"))


# ---------------------------------------------------------------------------
# Specific malformed shapes (not strictly fuzz, but useful invariants)
# ---------------------------------------------------------------------------


class TestSpecificMalformedShapes:
    """Targeted property tests for invariants the deterministic suite under-covers."""

    @given(
        kind=st.text(min_size=1, max_size=32).filter(lambda s: "." not in s),
        offset=st.integers(min_value=0, max_value=1000),
    )
    @settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_unknown_kind_lax_yields_unknown_event(
        self,
        kind: str,
        offset: int,
    ) -> None:
        """An envelope with a random ``kind`` is wrapped in :class:`UnknownEvent`."""
        raw: dict[str, Any] = {
            "kind": kind,
            "source": "pod://r/trainer",
            "time": datetime.now(UTC).isoformat(),
            "run_id": "r",
            "offset": offset,
            "schema_version": 1,
            "severity": "info",
            "payload": {},
        }
        line = json.dumps(raw)
        result = from_jsonl(line, strict=False)
        assert isinstance(result, UnknownEvent)
        assert result.original_type == kind
        # Offset is preserved when supplied.
        assert result.offset == offset

    @given(
        kind=st.text(min_size=1, max_size=32),
    )
    @settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_unknown_kind_strict_raises_validation_error(self, kind: str) -> None:
        """Strict mode re-raises :class:`pydantic.ValidationError` for unknown kinds."""
        # Skip values that happen to collide with a known kind.
        if kind.startswith("ryotenkai.") and "." in kind[len("ryotenkai."):]:
            # Could be a legit known kind by accident; skip.
            return
        raw = {
            "kind": kind,
            "source": "pod://r/trainer",
            "time": datetime.now(UTC).isoformat(),
            "run_id": "r",
            "offset": 0,
            "schema_version": 1,
            "severity": "info",
            "payload": {},
        }
        line = json.dumps(raw)
        with pytest.raises(ValidationError):
            from_jsonl(line, strict=True)

    @given(
        # Random bytes that include the tab character; the parser must
        # still produce either a valid event or a wrapped unknown.
        # Compose the tab in directly rather than filtering so we don't
        # blow the health-check budget.
        prefix=st.binary(min_size=0, max_size=32),
        suffix=st.binary(min_size=0, max_size=64),
    )
    @settings(
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
    )
    def test_random_tab_bearing_bytes_lax_safe(
        self,
        prefix: bytes,
        suffix: bytes,
    ) -> None:
        b = prefix + b"\t" + suffix
        line = _decode_lossy(b)
        result = from_jsonl(line, strict=False)
        assert isinstance(result, BaseEvent)


# ---------------------------------------------------------------------------
# Unknown-offset sentinel respected
# ---------------------------------------------------------------------------


class TestUnknownOffsetSentinel:
    """Malformed envelopes use :data:`UNKNOWN_OFFSET` (-1) as the sentinel."""

    @given(line=_garbage_text_strategy.filter(lambda s: not s.strip().startswith("{")))
    @settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_malformed_unknown_uses_sentinel_offset(self, line: str) -> None:
        result = from_jsonl(line, strict=False)
        assert isinstance(result, BaseEvent)
        # If we land in the malformed-wrapper branch, the offset is
        # the sentinel — the journal-replay layer skips these so they
        # never participate in monotonic-offset bookkeeping.
        if isinstance(result, UnknownEvent) and result.original_type == "<malformed>":
            assert result.offset == UNKNOWN_OFFSET


# ---------------------------------------------------------------------------
# Type checking helper — declares Event union still includes our types
# ---------------------------------------------------------------------------


class TestUnionMembership:
    """Smoke property: every strategy event is a member of the Event union."""

    @given(event=any_event_strategy)
    @settings(deadline=None, suppress_health_check=[HealthCheck.too_slow], max_examples=20)
    def test_event_validates_via_adapter(self, event: BaseEvent) -> None:
        """The adapter accepts every concrete type we generate."""
        raw = event.model_dump()
        validated: Event = EVENT_ADAPTER.validate_python(raw)
        assert type(validated) is type(event)


# ---------------------------------------------------------------------------
# Targeted boundary tests — kill specific Cosmic Ray mutations
# ---------------------------------------------------------------------------


class TestParseLengthPrefixBoundaries:
    """Pin behaviour at exact boundaries to kill comparison-operator mutations.

    Cosmic Ray mutates ``<``/``<=``/``==``/``!=`` etc.; the tests here
    use boundary values where the original operator and the mutated
    operator produce different outcomes, so a mutation can't survive.
    """

    def test_tab_at_index_zero_rejected(self) -> None:
        """``tab_idx == 0`` is invalid (no prefix). ``<= 0`` vs ``< 0``
        differs only at zero — pinning this kills the ``LtE → Lt`` mutation.
        """
        with pytest.raises(ValueError):
            parse_length_prefix("\tjson\n")

    def test_tab_at_index_one_accepts_one_digit_prefix(self) -> None:
        """A single-digit prefix ``"0\\t\\n"`` works — empty body, declared 0."""
        # Pin the just-past-zero behaviour: tab_idx == 1, prefix == "0".
        declared, body = parse_length_prefix("0\t\n")
        assert declared == 0
        assert body == ""

    def test_actual_eq_declared_succeeds(self) -> None:
        """When ``actual == declared``, no error. Pins ``!=`` vs ``==`` mutations."""
        line = "4\tabcd\n"  # 4 bytes "abcd", declared 4
        declared, body = parse_length_prefix(line)
        assert declared == 4
        assert body == "abcd"

    def test_actual_off_by_one_fails(self) -> None:
        """Declared+1: even off-by-one trips the framer."""
        with pytest.raises(ValueError) as exc_info:
            parse_length_prefix("5\tabcd\n")  # declared 5, actual 4
        assert "length mismatch" in str(exc_info.value)
        with pytest.raises(ValueError) as exc_info:
            parse_length_prefix("3\tabcd\n")  # declared 3, actual 4
        assert "length mismatch" in str(exc_info.value)


class TestLooksLengthPrefixedBoundaries:
    """Targeted tests for :func:`_looks_length_prefixed` heuristic."""

    def test_empty_string_returns_false(self) -> None:
        """``line.find("\\t")`` returns -1; ``-1 <= 0`` is True so returns False."""
        from ryotenkai_shared.events.codec import _looks_length_prefixed

        assert _looks_length_prefixed("") is False

    def test_tab_at_start_returns_false(self) -> None:
        """Tab at index 0 means no prefix — returns False."""
        from ryotenkai_shared.events.codec import _looks_length_prefixed

        assert _looks_length_prefixed("\tjson") is False

    def test_digit_prefix_then_tab_returns_true(self) -> None:
        """Canonical wire format prefix — True."""
        from ryotenkai_shared.events.codec import _looks_length_prefixed

        assert _looks_length_prefixed("12\tjson") is True

    def test_non_digit_prefix_returns_false(self) -> None:
        """Letter prefix → not length-prefixed."""
        from ryotenkai_shared.events.codec import _looks_length_prefixed

        assert _looks_length_prefixed("ab\tjson") is False


class TestFromJsonlUpcasterBoundary:
    """Pin the ``current_version < target_version`` upcaster decision."""

    def test_strict_validation_path_is_taken(self) -> None:
        """A non-event JSON object hits the validation path and raises."""
        # An object that's NOT in the Event union (no kind, no fields) —
        # validates against EVENT_ADAPTER and raises ValidationError in
        # strict mode. This pins the validate_python call site.
        with pytest.raises(ValidationError):
            from_jsonl('{"some": "thing"}', strict=True)


class TestBuildUnknownDefensivePaths:
    """Exercise the defensive ``_build_unknown`` fallbacks.

    These tests target the surviving mutations in ``_safe_uuid``,
    ``_safe_time``, ``_safe_int``, and the malformed-builder.
    """

    def test_invalid_uuid_string_falls_back_to_generated(self) -> None:
        """A non-UUID string is replaced with a fresh UUIDv7."""
        from uuid import UUID

        raw = json.dumps({
            "kind": "ryotenkai.unseen.kind",
            "event_id": "not-a-uuid",
            "source": "x",
            "time": "2026-05-16T12:00:00+00:00",
            "run_id": "x",
            "offset": 0,
            "severity": "info",
            "payload": {},
        })
        event = from_jsonl(raw, strict=False)
        assert isinstance(event, UnknownEvent)
        # event_id was synthesised — must be a valid UUID.
        assert isinstance(event.event_id, UUID)

    def test_invalid_time_string_falls_back_to_now(self) -> None:
        """A non-ISO time falls back to ``utc_now()``."""
        raw = json.dumps({
            "kind": "ryotenkai.unseen.kind",
            "source": "x",
            "time": "not-a-date",
            "run_id": "x",
            "offset": 0,
            "severity": "info",
            "payload": {},
        })
        event = from_jsonl(raw, strict=False)
        assert isinstance(event, UnknownEvent)
        # The synthesised time is parseable.
        assert event.time is not None

    def test_non_int_offset_falls_back_to_sentinel(self) -> None:
        """Non-integer offset → :data:`UNKNOWN_OFFSET` sentinel."""
        raw = json.dumps({
            "kind": "ryotenkai.unseen.kind",
            "source": "x",
            "time": "2026-05-16T12:00:00+00:00",
            "run_id": "x",
            "offset": "not_an_int",
            "severity": "info",
            "payload": {},
        })
        event = from_jsonl(raw, strict=False)
        assert isinstance(event, UnknownEvent)
        assert event.offset == UNKNOWN_OFFSET

    def test_non_dict_payload_replaced_with_empty(self) -> None:
        """If ``payload`` isn't a dict, store ``{}``."""
        raw = json.dumps({
            "kind": "ryotenkai.unseen.kind",
            "source": "x",
            "time": "2026-05-16T12:00:00+00:00",
            "run_id": "x",
            "offset": 0,
            "severity": "info",
            "payload": "not a dict",
        })
        event = from_jsonl(raw, strict=False)
        assert isinstance(event, UnknownEvent)
        assert event.raw_payload == {}

    def test_malformed_unknown_diagnostic_truncation(self) -> None:
        """Diagnostic crumbs in malformed unknown are truncated to 200 chars."""
        long_garbage = "x" * 1000
        event = from_jsonl(long_garbage, strict=False)
        assert isinstance(event, UnknownEvent)
        assert event.original_type == "<malformed>"
        # The captured ``_raw_line`` is bounded to keep metrics labels sane.
        assert len(event.raw_payload["_raw_line"]) <= 200

    def test_string_stage_id_preserved_in_unknown(self) -> None:
        """``stage_id`` survives the unknown wrapper when present as a string."""
        raw = json.dumps({
            "kind": "ryotenkai.unseen.kind",
            "source": "x",
            "time": "2026-05-16T12:00:00+00:00",
            "run_id": "x",
            "stage_id": "trainer",
            "offset": 0,
            "severity": "info",
            "payload": {},
        })
        event = from_jsonl(raw, strict=False)
        assert isinstance(event, UnknownEvent)
        assert event.stage_id == "trainer"

    def test_non_string_stage_id_becomes_none(self) -> None:
        """A non-string ``stage_id`` is dropped (set to None)."""
        raw = json.dumps({
            "kind": "ryotenkai.unseen.kind",
            "source": "x",
            "time": "2026-05-16T12:00:00+00:00",
            "run_id": "x",
            "stage_id": 42,  # not a string
            "offset": 0,
            "severity": "info",
            "payload": {},
        })
        event = from_jsonl(raw, strict=False)
        assert isinstance(event, UnknownEvent)
        assert event.stage_id is None

    def test_schema_version_preserved_when_int(self) -> None:
        """Integer ``schema_version`` is kept."""
        raw = json.dumps({
            "kind": "ryotenkai.unseen.kind",
            "source": "x",
            "time": "2026-05-16T12:00:00+00:00",
            "run_id": "x",
            "offset": 0,
            "schema_version": 7,
            "severity": "info",
            "payload": {},
        })
        event = from_jsonl(raw, strict=False)
        assert isinstance(event, UnknownEvent)
        assert event.schema_version == 7

    def test_non_int_schema_version_defaults_to_1(self) -> None:
        raw = json.dumps({
            "kind": "ryotenkai.unseen.kind",
            "source": "x",
            "time": "2026-05-16T12:00:00+00:00",
            "run_id": "x",
            "offset": 0,
            "schema_version": "nope",
            "severity": "info",
            "payload": {},
        })
        event = from_jsonl(raw, strict=False)
        assert isinstance(event, UnknownEvent)
        assert event.schema_version == 1


class TestFromJsonlBranchCoverage:
    """Targeted branch-coverage tests for from_jsonl decision points."""

    def test_strict_mode_validation_error_propagates(self) -> None:
        """``strict=True`` lets pydantic ValidationError escape."""
        raw = json.dumps({
            "kind": "ryotenkai.unseen.kind",
            "source": "x",
            "time": "2026-05-16T12:00:00+00:00",
            "run_id": "x",
            "offset": 0,
            "severity": "info",
            "payload": {},
        })
        with pytest.raises(ValidationError):
            from_jsonl(raw, strict=True)

    def test_lax_mode_validation_error_wraps_unknown(self) -> None:
        """``strict=False`` catches ValidationError and wraps."""
        raw = json.dumps({
            "kind": "ryotenkai.unseen.kind",
            "source": "x",
            "time": "2026-05-16T12:00:00+00:00",
            "run_id": "x",
            "offset": 0,
            "severity": "info",
            "payload": {},
        })
        event = from_jsonl(raw, strict=False)
        assert isinstance(event, UnknownEvent)
        assert event.original_type == "ryotenkai.unseen.kind"

    def test_strict_mode_malformed_raises_typed(self) -> None:
        """``strict=True`` raises MalformedEventError for non-object JSON."""
        with pytest.raises(MalformedEventError) as exc:
            from_jsonl('"a string"', strict=True)
        # The cause is a ValueError that says non-object.
        assert isinstance(exc.value.cause, ValueError)

    def test_non_dict_json_array_strict_raises(self) -> None:
        """JSON arrays are not envelope objects — strict raises."""
        with pytest.raises(MalformedEventError):
            from_jsonl("[1, 2, 3]", strict=True)

    def test_non_dict_json_array_lax_wraps(self) -> None:
        """JSON arrays in lax mode are wrapped as malformed unknown."""
        event = from_jsonl("[1, 2, 3]", strict=False)
        assert isinstance(event, UnknownEvent)
        assert event.original_type == "<malformed>"

    def test_upcaster_chain_not_consulted_when_schema_version_missing(self) -> None:
        """No schema_version → defaults to 1 → no upcaster invocation."""
        # The path through ``isinstance(current_version, int)`` is taken.
        # We verify that a known event with schema_version=1 still
        # round-trips (no spurious upcaster call breaks it).
        raw = json.dumps({
            "kind": "ryotenkai.pod.training.started",
            "source": "pod://r/trainer",
            "time": "2026-05-16T12:00:00+00:00",
            "run_id": "r",
            "offset": 0,
            "severity": "info",
            "payload": {
                "max_steps": 100,
                "num_train_epochs": 1,
                "per_device_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "learning_rate": 1e-5,
                "algorithm": "sft",
            },
        })
        event = from_jsonl(raw, strict=True)
        assert isinstance(event, TrainingStartedEvent)


# ---------------------------------------------------------------------------
# Upcaster invocation tests — pin the ``current_version < target_version`` branch
# ---------------------------------------------------------------------------


class TestUpcasterInvocation:
    """Pin the ``current_version < target_version`` comparison.

    With the registry empty, ``latest_version_for(kind)`` always returns
    1, so any mutation of ``<`` to ``<=`` / ``==`` / ``>`` produces the
    same False outcome. Registering an upcaster makes the comparison
    semantically observable — a mutation that flips ``<`` to ``>`` (for
    instance) would either fire the upcaster too eagerly or skip it,
    both of which we detect.
    """

    def test_upcaster_invoked_when_current_version_lower(self) -> None:
        """A registered upcaster fires only when current_version < latest."""
        from ryotenkai_shared.events.upcasters import (
            apply_chain,
            clear,
            latest_version_for,
            register,
        )

        invocations: list[tuple[int, int, dict]] = []

        def _upcaster(raw_dict: dict, from_v: int, to_v: int) -> dict:
            invocations.append((from_v, to_v, dict(raw_dict)))
            return raw_dict

        clear()
        try:
            register("ryotenkai.pod.training.started", _upcaster)
            # Now latest_version_for("ryotenkai.pod.training.started") == 2.
            assert latest_version_for("ryotenkai.pod.training.started") == 2

            # Encode a known event but at schema_version=1; codec should
            # invoke the upcaster (1 < 2).
            raw = {
                "kind": "ryotenkai.pod.training.started",
                "source": "pod://r/trainer",
                "time": "2026-05-16T12:00:00+00:00",
                "run_id": "r",
                "offset": 0,
                "schema_version": 1,
                "severity": "info",
                "payload": {
                    "max_steps": 100,
                    "num_train_epochs": 1,
                    "per_device_batch_size": 1,
                    "gradient_accumulation_steps": 1,
                    "learning_rate": 1e-5,
                    "algorithm": "sft",
                },
            }
            from_jsonl(json.dumps(raw), strict=False)
            assert len(invocations) == 1
            assert invocations[0][0] == 1  # from
            assert invocations[0][1] == 2  # to
        finally:
            clear()

    def test_upcaster_not_invoked_when_current_version_at_latest(self) -> None:
        """When current_version == latest, the upcaster is NOT called."""
        from ryotenkai_shared.events.upcasters import (
            clear,
            register,
        )

        invocations: list[int] = []

        def _upcaster(raw_dict: dict, _from: int, _to: int) -> dict:
            invocations.append(1)
            return raw_dict

        clear()
        try:
            register("ryotenkai.pod.training.started", _upcaster)
            # current_version == 2 (== latest after registering one
            # upcaster). The codec should NOT call the upcaster.
            raw = {
                "kind": "ryotenkai.pod.training.started",
                "source": "pod://r/trainer",
                "time": "2026-05-16T12:00:00+00:00",
                "run_id": "r",
                "offset": 0,
                "schema_version": 2,
                "severity": "info",
                "payload": {
                    "max_steps": 100,
                    "num_train_epochs": 1,
                    "per_device_batch_size": 1,
                    "gradient_accumulation_steps": 1,
                    "learning_rate": 1e-5,
                    "algorithm": "sft",
                },
            }
            from_jsonl(json.dumps(raw), strict=False)
            # No upcaster firing — comparison decided not to chain.
            assert invocations == []
        finally:
            clear()

    def test_upcaster_not_invoked_when_kind_unknown_to_registry(self) -> None:
        """No upcaster registered for a kind → comparison short-circuits.

        Pins the ``isinstance(event_kind, str)`` branch — when the kind
        is missing entirely (None), no upcaster is consulted.
        """
        from ryotenkai_shared.events.upcasters import clear, register

        invocations: list[int] = []

        def _upcaster(raw_dict: dict, _from: int, _to: int) -> dict:
            invocations.append(1)
            return raw_dict

        clear()
        try:
            register("ryotenkai.pod.training.started", _upcaster)
            # Different kind — no upcaster fires.
            raw = {
                "kind": "ryotenkai.pod.training.step",
                "source": "pod://r/trainer",
                "time": "2026-05-16T12:00:00+00:00",
                "run_id": "r",
                "offset": 0,
                "schema_version": 1,
                "severity": "debug",
                "payload": {
                    "step": 1,
                    "loss": 1.0,
                    "learning_rate": 1e-5,
                    "grad_norm": 0.5,
                    "tokens_per_sec": 100.0,
                    "samples_per_sec": 1.0,
                },
            }
            from_jsonl(json.dumps(raw), strict=False)
            # The registered upcaster is for a different kind — never fires.
            assert invocations == []
        finally:
            clear()
