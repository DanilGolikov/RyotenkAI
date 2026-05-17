"""Phase 0 smoke test for the unified event system foundations.

Verifies that the chosen primitives (Pydantic v2 closed discriminated union,
UUIDv7 from uuid_utils, JSON round-trip, extra=forbid, UnknownEvent fallback)
work as expected before committing to the full schema in Phase 1.

Run: .venv/bin/python scripts/phase0_smoke_event_envelope.py
This is a one-off probe, not a permanent module.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Annotated, Any, ClassVar, Literal
from uuid import UUID

import uuid_utils
from pydantic import (
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    TypeAdapter,
    ValidationError,
)


# ---------------------------------------------------------------------------
# Envelope
# ---------------------------------------------------------------------------
def new_uuid7() -> UUID:
    return UUID(str(uuid_utils.uuid7()))


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class BaseEvent(BaseModel):
    """Common envelope for every event variant.

    Fields are populated by the emitter when None at construction time so call-sites
    can stay terse; serialized representation is always fully populated.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    event_id: UUID = Field(default_factory=new_uuid7)
    type: str
    source: str
    time: datetime = Field(default_factory=utc_now)
    run_id: str
    stage_id: str | None = None
    offset: int
    schema_version: int = 1
    severity: Literal["debug", "info", "warning", "error", "critical"]


# ---------------------------------------------------------------------------
# Concrete event types
# ---------------------------------------------------------------------------
class TrainingStartedPayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    max_steps: int
    num_train_epochs: int
    per_device_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    algorithm: Literal["sft", "cpt", "dpo", "grpo", "sapo"]


class TrainingStartedEvent(BaseEvent):
    SCHEMA_VERSION: ClassVar[int] = 1
    type: Literal["ryotenkai.pod.training.started"] = "ryotenkai.pod.training.started"
    severity: Literal["info"] = "info"
    payload: TrainingStartedPayload


class CheckpointSavedPayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    step: int
    local_path: str
    size_bytes: int
    is_best: bool


class CheckpointSavedEvent(BaseEvent):
    SCHEMA_VERSION: ClassVar[int] = 1
    type: Literal["ryotenkai.pod.training.checkpoint_saved"] = "ryotenkai.pod.training.checkpoint_saved"
    severity: Literal["info"] = "info"
    payload: CheckpointSavedPayload


class MemoryCacheClearedPayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    device: str
    before_bytes: int
    after_bytes: int
    trigger: Literal["scheduled", "threshold", "manual"]


class MemoryCacheClearedEvent(BaseEvent):
    SCHEMA_VERSION: ClassVar[int] = 1
    type: Literal["ryotenkai.pod.memory.cache_cleared"] = "ryotenkai.pod.memory.cache_cleared"
    severity: Literal["info"] = "info"
    payload: MemoryCacheClearedPayload


class UnknownEvent(BaseEvent):
    """Catch-all for forward-compat: events whose type is unknown to the current code.

    The codec wraps unknown variants here (in strict=False mode) so consumers can
    still see them in the journal stream without crashing.
    """

    SCHEMA_VERSION: ClassVar[int] = 1
    type: Literal["ryotenkai.unknown"] = "ryotenkai.unknown"
    severity: Literal["info"] = "info"
    original_type: str
    raw_payload: dict[str, Any]


Event = Annotated[
    TrainingStartedEvent | CheckpointSavedEvent | MemoryCacheClearedEvent | UnknownEvent,
    Discriminator("type"),
]

EVENT_ADAPTER: TypeAdapter[Event] = TypeAdapter(Event)


# ---------------------------------------------------------------------------
# Codec
# ---------------------------------------------------------------------------
def to_jsonl(event: BaseEvent) -> str:
    return event.model_dump_json()


def from_jsonl(line: str, *, strict: bool = False) -> Event:
    raw = json.loads(line)
    try:
        return EVENT_ADAPTER.validate_python(raw)
    except ValidationError:
        if strict:
            raise
        original_type = raw.get("type", "<missing>")
        return UnknownEvent(
            event_id=UUID(raw["event_id"]) if "event_id" in raw else new_uuid7(),
            source=raw.get("source", "unknown"),
            time=datetime.fromisoformat(raw["time"]) if "time" in raw else utc_now(),
            run_id=raw.get("run_id", "unknown"),
            stage_id=raw.get("stage_id"),
            offset=int(raw.get("offset", -1)),
            schema_version=int(raw.get("schema_version", 1)),
            original_type=original_type,
            raw_payload=raw.get("payload", {}),
        )


# ---------------------------------------------------------------------------
# Smoke checks
# ---------------------------------------------------------------------------
def check_basic_round_trip() -> None:
    event = TrainingStartedEvent(
        source="pod://run-001/trainer",
        run_id="run-001",
        offset=0,
        payload=TrainingStartedPayload(
            max_steps=1000,
            num_train_epochs=3,
            per_device_batch_size=4,
            gradient_accumulation_steps=2,
            learning_rate=2e-5,
            algorithm="sft",
        ),
    )
    line = to_jsonl(event)
    restored = from_jsonl(line, strict=True)
    assert isinstance(restored, TrainingStartedEvent)
    assert restored == event, f"round-trip mismatch: {restored!r} != {event!r}"
    assert restored.event_id == event.event_id
    print("[ok] basic round-trip preserves identity")


def check_discriminator_dispatch() -> None:
    payloads = [
        TrainingStartedEvent(
            source="pod://run-002/trainer",
            run_id="run-002",
            offset=0,
            payload=TrainingStartedPayload(
                max_steps=100,
                num_train_epochs=1,
                per_device_batch_size=1,
                gradient_accumulation_steps=1,
                learning_rate=1e-4,
                algorithm="cpt",
            ),
        ),
        CheckpointSavedEvent(
            source="pod://run-002/trainer",
            run_id="run-002",
            offset=1,
            payload=CheckpointSavedPayload(
                step=50, local_path="/ckpt/50", size_bytes=12345, is_best=False
            ),
        ),
        MemoryCacheClearedEvent(
            source="pod://run-002/trainer",
            run_id="run-002",
            offset=2,
            payload=MemoryCacheClearedPayload(
                device="cuda:0", before_bytes=10_000_000_000, after_bytes=2_000_000_000, trigger="threshold"
            ),
        ),
    ]
    expected_types = {type(p) for p in payloads}
    seen_types = {type(from_jsonl(to_jsonl(p), strict=True)) for p in payloads}
    assert expected_types == seen_types, f"discriminator dispatch lost types: {seen_types}"
    print(f"[ok] discriminator dispatch covers {len(payloads)} concrete types")


def check_extra_forbid() -> None:
    # Smuggle an extra field into payload — should be rejected by extra="forbid".
    event = TrainingStartedEvent(
        source="pod://run-003/trainer",
        run_id="run-003",
        offset=0,
        payload=TrainingStartedPayload(
            max_steps=1, num_train_epochs=1, per_device_batch_size=1,
            gradient_accumulation_steps=1, learning_rate=1e-4, algorithm="sft",
        ),
    )
    raw = json.loads(to_jsonl(event))
    raw["payload"]["sneaky_field"] = "should_be_rejected"
    try:
        EVENT_ADAPTER.validate_python(raw)
    except ValidationError as e:
        assert "extra_forbidden" in str(e) or "Extra inputs" in str(e), e
        print("[ok] extra=forbid blocks unknown payload fields")
        return
    raise AssertionError("extra=forbid did not reject the unknown payload field")


def check_unknown_event_fallback() -> None:
    raw = {
        "event_id": str(new_uuid7()),
        "type": "ryotenkai.pod.future.gradient_anomaly",  # not in the union
        "source": "pod://run-004/trainer",
        "time": utc_now().isoformat(),
        "run_id": "run-004",
        "stage_id": None,
        "offset": 42,
        "schema_version": 2,
        "severity": "warning",
        "payload": {"step": 100, "spike_ratio": 4.2},
    }
    line = json.dumps(raw)
    restored = from_jsonl(line, strict=False)
    assert isinstance(restored, UnknownEvent), f"expected UnknownEvent, got {type(restored).__name__}"
    assert restored.original_type == "ryotenkai.pod.future.gradient_anomaly"
    assert restored.raw_payload == {"step": 100, "spike_ratio": 4.2}
    print("[ok] unknown type wraps into UnknownEvent (forward-compat)")


def check_strict_mode_raises() -> None:
    raw = {
        "type": "ryotenkai.pod.future.gradient_anomaly",
        "source": "pod://run-005/trainer",
        "time": utc_now().isoformat(),
        "run_id": "run-005",
        "offset": 0,
        "severity": "info",
        "payload": {},
    }
    line = json.dumps(raw)
    try:
        from_jsonl(line, strict=True)
    except ValidationError:
        print("[ok] strict=True raises on unknown type (producer-side guard)")
        return
    raise AssertionError("strict=True did not raise on unknown type")


def check_frozen_immutability() -> None:
    event = TrainingStartedEvent(
        source="pod://run-006/trainer",
        run_id="run-006",
        offset=0,
        payload=TrainingStartedPayload(
            max_steps=1, num_train_epochs=1, per_device_batch_size=1,
            gradient_accumulation_steps=1, learning_rate=1e-4, algorithm="sft",
        ),
    )
    try:
        event.offset = 999  # type: ignore[misc]
    except ValidationError:
        print("[ok] frozen=True blocks mutation")
        return
    raise AssertionError("frozen=True did not block mutation")


def check_uuid7_monotonicity() -> None:
    ids = [new_uuid7() for _ in range(1000)]
    sorted_ids = sorted(ids)
    monotonic = sum(1 for a, b in zip(ids, ids[1:]) if a < b)
    # UUIDv7 timestamps have ms precision so back-to-back generations within the
    # same millisecond can tie. We expect strong (not strict) monotonicity.
    ratio = monotonic / (len(ids) - 1)
    assert ratio > 0.95, f"uuid7 monotonicity ratio too low: {ratio:.2%}"
    assert ids == sorted_ids or ratio > 0.95
    print(f"[ok] uuid7 monotonicity ratio = {ratio:.2%} over 1000 samples")


def bench_throughput(n: int = 50_000) -> None:
    print(f"[bench] preparing {n} events...")
    template_payload = TrainingStartedPayload(
        max_steps=1000, num_train_epochs=3, per_device_batch_size=4,
        gradient_accumulation_steps=2, learning_rate=2e-5, algorithm="sft",
    )
    events = [
        TrainingStartedEvent(
            source="pod://bench/trainer", run_id="bench", offset=i, payload=template_payload
        )
        for i in range(n)
    ]

    t0 = time.perf_counter()
    serialized = [to_jsonl(e) for e in events]
    t_ser = time.perf_counter() - t0

    t0 = time.perf_counter()
    parsed = [from_jsonl(line, strict=True) for line in serialized]
    t_de = time.perf_counter() - t0

    assert len(parsed) == n

    print(
        f"[bench] serialize:   {t_ser*1000:7.1f} ms total | "
        f"{n/t_ser:>9.0f} ev/s | {t_ser/n*1e6:6.1f} us/event"
    )
    print(
        f"[bench] deserialize: {t_de*1000:7.1f} ms total | "
        f"{n/t_de:>9.0f} ev/s | {t_de/n*1e6:6.1f} us/event"
    )

    # Plan target: 50k events/sec sustained on M-series Mac (R-09).
    # Smoke threshold is loose — we just want to confirm we are in the right
    # order of magnitude before committing to ~50 types.
    threshold_eps = 20_000
    actual_eps = min(n / t_ser, n / t_de)
    if actual_eps < threshold_eps:
        print(
            f"[warn] throughput {actual_eps:.0f} ev/s below smoke threshold {threshold_eps} ev/s "
            "— Phase 1 will need tag-prefix dispatch (R-09 fallback)."
        )
    else:
        print(f"[ok] throughput {actual_eps:.0f} ev/s above smoke threshold {threshold_eps} ev/s")


if __name__ == "__main__":
    print("=== Phase 0 smoke: event envelope foundations ===")
    check_basic_round_trip()
    check_discriminator_dispatch()
    check_extra_forbid()
    check_unknown_event_fallback()
    check_strict_mode_raises()
    check_frozen_immutability()
    check_uuid7_monotonicity()
    bench_throughput()
    print("=== all smoke checks passed ===")
