# Upcaster registry — conventions

The unified event system uses a registry of pure-function upcasters to
migrate raw envelope dicts across schema versions. This README documents
the rules a new upcaster must satisfy.

## When you need an upcaster

| Change                                            | Action                       |
|---------------------------------------------------|------------------------------|
| Add an optional payload field with a default      | No upcaster — backward compat |
| Add a required payload field                      | Upcaster: fill default        |
| Rename a payload field                            | Upcaster: rename              |
| Change a field's semantic meaning                 | **New dotted `type`** — never reuse |
| Remove a load-bearing field                       | **New dotted `type`** + upcaster that rewrites `type` |
| Bump severity for an existing type                | Allowed; no upcaster needed   |

## Authoring rules

1. **Pure**: no IO, no `logger`, no `datetime.now()`, no `random`. The
   function MUST be a deterministic mapping `(dict, int, int) -> dict`.
2. **Idempotent**: applying the same hop twice produces the same output.
3. **One hop**: handle exactly `(N, N+1)`. The chain runner composes hops.
4. **Don't mutate inputs**: return a new dict (a shallow copy plus the
   tweaks is fine). Tests rely on input stability.
5. **Don't touch `schema_version`**: the registry pins it after the
   chain completes.

## Registering

In a phase where you add a hop, register it once at import time:

```python
from ryotenkai_shared.events.upcasters import register

def v1_to_v2_training_started(raw: dict, _from: int, _to: int) -> dict:
    payload = {**raw["payload"]}
    payload["total_steps"] = payload.pop("max_steps")  # rename
    return {**raw, "payload": payload}

register("ryotenkai.pod.training.started", v1_to_v2_training_started)
```

## Testing

A new hop deserves at least:

* A positive case asserting the rewritten dict shape.
* A negative case ensuring the function tolerates fields it does NOT
  touch (passthrough invariant).
* Composition: `apply_chain` from 1 → latest produces a dict that
  validates against the current Pydantic schema.

The chain runner is tested in `tests/unit/shared/events/test_upcasters.py`
and does not need to be re-tested per hop.
