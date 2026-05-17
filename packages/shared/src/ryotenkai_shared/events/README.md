# ryotenkai_shared.events

Typed event system for RyotenkAI: Pydantic v2 closed discriminated union
+ length-prefixed JSONL codec + `IEventEmitter` Protocol + upcaster
registry. This package is the *boundary* — it ships the envelope and
codec but no transport. Concrete emitter / journal / bus implementations
live in `ryotenkai_pod.runner` and `ryotenkai_control.events`.

See also: [ADR-0009](../../../../../docs/adrs/2026-05-17-unified-event-system.md)
for the architectural decision, and the
[plan file](../../../../../docs/plans/ethereal-tumbling-patterson.md)
for the full design rationale.

## Quick start

```python
from ryotenkai_shared.events import to_jsonl, from_jsonl
from ryotenkai_shared.events.types import (
    TrainingStartedEvent,
    TrainingStartedPayload,
)

event = TrainingStartedEvent(
    source="pod://run-abc/trainer",
    run_id="run-abc",
    offset=0,
    payload=TrainingStartedPayload(
        max_steps=1000,
        num_train_epochs=3,
        per_device_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,
        algorithm="sft",
    ),
)

# Serialise to length-prefixed JSONL line (canonical wire format).
line = to_jsonl(event)
# "<bytes>\t{\"event_id\":\"...\",\"kind\":\"ryotenkai.pod.training.started\",...}\n"

# Decode (round-trip safe).
decoded = from_jsonl(line)
assert decoded == event
```

For the `IEventEmitter` Protocol surface, see the control-side guide:
[packages/control/src/ryotenkai_control/events/README.md](../../../../control/src/ryotenkai_control/events/README.md).

## Event taxonomy

Naming convention: `ryotenkai.<area>.<domain>.<verb_past_or_noun>` where
area ∈ {`pod`, `control`}. Verb past tense for facts (`started`,
`completed`, `failed`); noun for snapshots (`snapshot`, `step`).

The full list (55 types + `UnknownEvent` catch-all) is grouped by domain:

| Area | Domain | Count | Module |
|---|---|---|---|
| pod | lifecycle | 8 | `types/pod_lifecycle.py` |
| pod | training | 9 | `types/pod_training.py` |
| pod | memory | 4 | `types/pod_memory.py` |
| pod | health | 3 | `types/pod_health.py` |
| pod | io | 2 | `types/pod_io.py` |
| pod | journal | 2 | `types/pod_journal.py` |
| control | run | 4 | `types/control_run.py` |
| control | stage | 5 | `types/control_stage.py` |
| control | dataset | 3 | `types/control_dataset.py` |
| control | gpu | 6 | `types/control_gpu.py` |
| control | training (monitor) | 2 | `types/control_training.py` |
| control | model | 3 | `types/control_model.py` |
| control | evaluation | 5 | `types/control_evaluation.py` |
| control | inference | 6 | `types/control_inference.py` |
| forward-compat | unknown | 1 | `types/unknown.py` |

Severity defaults are pinned per-class via `Literal[...]`:

* `started` / `completed` / `snapshot` → `info` (or `debug` for hot-path
  `step` / `log`)
* `failed` / `oom_detected` / `preempted` → `error`
* `pressure_warning` / `cancelled` / `interrupted` / `timeout` → `warning`

## Envelope shape

Every event extends `BaseEvent` and pins `kind` + `severity` via
`Literal` defaults:

```python
class BaseEvent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    event_id: UUID                 # UUIDv7, default-factory
    kind: str                      # Literal in subclasses (the discriminator)
    source: str                    # "pod://{run_id}/trainer" | "control://orchestrator/{stage}"
    time: datetime                 # UTC microseconds, default-factory
    run_id: str
    stage_id: str | None = None    # auto-filled from ContextVar in stage_scope
    offset: int                    # monotonic per (run_id, source); required
    schema_version: int = 1
    severity: Literal["debug","info","warning","error","critical"]
```

Subclass shape:

```python
class TrainingStartedEvent(BaseEvent):
    kind: Literal["ryotenkai.pod.training.started"] = "ryotenkai.pod.training.started"
    severity: Literal["info"] = "info"
    payload: TrainingStartedPayload
```

Invariants enforced by the model config:

* `frozen=True` — events are immutable. Re-emit produces a new instance.
* `extra="forbid"` — unknown fields are rejected. Forward-compat goes
  through `UnknownEvent`, not loose JSON.
* `event_id` defaults to `new_uuid7()` — RFC 9562 UUIDv7, time-ordered.
* `offset` has no default — schemas that forget to set it fail at
  Pydantic validation rather than silently emitting `offset=0`.

## How to add a new event

1. **Pick a kind name** following the naming convention
   (`ryotenkai.<area>.<domain>.<verb>`).
2. **Define the payload** as a `BaseModel` with `frozen=True`,
   `extra="forbid"`. Put it next to the event class in the relevant
   `types/<domain>.py`.
3. **Define the event class** extending `BaseEvent`, with `kind` and
   `severity` pinned via `Literal` defaults and a typed `payload`
   field.
4. **Register in the union** — add the class to the `Union[...]`
   block in `discriminator.py` (alphabetical-by-domain, one class per
   line so the diff is a single insertion).
5. **Re-export** from `types/__init__.py` (both the event class and
   its payload class, grouped under the appropriate domain comment).
6. **Write tests** in `tests/unit/shared/events/types/test_<domain>_<kind>.py`
   following the 7-class template from `CLAUDE.md`. Minimum: a
   `TestPositive` round-trip, `TestInvariants` for the `kind` /
   `severity` pins, and `TestNegative` for `extra="forbid"` rejection.

```python
# types/pod_training.py
class TrainingPausedPayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    step: int
    reason: Literal["operator", "auto"]


class TrainingPausedEvent(BaseEvent):
    kind: Literal["ryotenkai.pod.training.paused"] = "ryotenkai.pod.training.paused"
    severity: Literal["warning"] = "warning"
    payload: TrainingPausedPayload


# discriminator.py — add to the Union[] block under the "Pod training" comment:
#     TrainingPausedEvent,
#
# types/__init__.py — add both names under "Pod training":
#     "TrainingPausedPayload",
#     "TrainingPausedEvent",
```

## Schema evolution

| Change | Action |
|---|---|
| Add optional payload field with default | No upcaster — backward-compat |
| Add required payload field | Upcaster: fill default |
| Rename a payload field | Upcaster: rename in `(N, N+1)` hop |
| Change a field's semantic meaning | **New dotted `kind`** — never reuse |
| Remove a load-bearing field | **New dotted `kind`** + upcaster that rewrites `kind` |
| Bump severity for an existing kind | Allowed; no upcaster needed |

See [upcasters/README.md](upcasters/README.md) for the registration
contract (pure, idempotent, one-hop, never mutate inputs).

## Length-prefixed JSONL invariant

Wire format:

```
<utf8_byte_length>\t<envelope_json>\n
```

UTF-8 *byte* length, not character length — multi-byte payloads (non-ASCII
error messages, emoji in log lines) take >1 byte each and a character
count lets corruption slip past the integrity check.

* `to_jsonl(event)` — emits the canonical wire format. Output ends with
  `\n`; callers write the string as-is to the journal file.
* `from_jsonl(line, strict=False)` — decodes BOTH length-prefixed lines
  and raw JSON lines (back-compat for tests / fixtures). Auto-detects
  the format via the leading-digit-then-tab heuristic.
* `parse_length_prefix(line)` — strict parser; raises `ValueError` on
  any deviation. Journal reader uses this for torn-write detection.

Failure modes:

| Failure | `strict=True` | `strict=False` (default for reads) |
|---|---|---|
| Unknown `kind` | re-raise `ValidationError` | wrap in `UnknownEvent` |
| JSON decode / framing | raise `MalformedEventError` | wrap in `UnknownEvent` with `original_type="<malformed>"` and diagnostic crumbs |

Journal readers MUST use `strict=True` when sweeping the tail for
torn writes — that lets them distinguish "torn line, truncate" from
"actual logic bug, re-raise".

## Counter observability

Phase 8 introduced an aggregator in
`ryotenkai_control.events.metrics`. Each collaborator
(`ControlEventEmitter`, `InMemoryBus`, `JournalWriter`, `EventDedup`)
exposes raw counters; `collect_metrics(...)` snapshots them into a
single `EventSubsystemMetrics` dataclass for the
`GET /api/v1/health/events` endpoint. See the
[control/events README](../../../../control/src/ryotenkai_control/events/README.md)
for the surface and the motivating Risk Ledger entry R-06.

## Module layout

```
events/
├── __init__.py        # re-exports: BaseEvent, Event, EVENT_ADAPTER,
│                      #   IEventEmitter, UnknownEvent, codec funcs
├── envelope.py        # BaseEvent + new_uuid7 + utc_now
├── severity.py        # Severity Literal alias + SEVERITY_ORDER
├── codec.py           # to_jsonl, from_jsonl, parse_length_prefix,
│                      #   MalformedEventError, UnknownEvent wrappers
├── discriminator.py   # Event union + cached EVENT_ADAPTER TypeAdapter
├── protocol.py        # IEventEmitter Protocol (interface only)
├── upcasters/         # registry + chain runner + conventions README
└── types/
    ├── __init__.py    # re-exports every concrete event + payload
    ├── pod_lifecycle.py     pod_training.py     pod_memory.py
    ├── pod_health.py        pod_io.py           pod_journal.py
    ├── control_run.py       control_stage.py    control_dataset.py
    ├── control_gpu.py       control_training.py control_model.py
    ├── control_evaluation.py  control_inference.py
    └── unknown.py     # catch-all variant
```

The discriminated union is assembled in `discriminator.py` (not
`types/__init__.py`) to avoid an import cycle between the union and the
`UnknownEvent` catch-all.
