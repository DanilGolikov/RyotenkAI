"""Generate ``marker_cancelled_schema.json`` and
``marker_completion_schema.json``.

The production writers in
``packages/pod/src/ryotenkai_pod/trainer/callbacks/{cancellation,completion}_callback.py``
emit two distinct payloads. There is no single Pydantic model — the
payloads are constructed inline as ``json.dumps({...})``. The schema
is therefore *generated from a pinned dictionary literal* in this
script that mirrors the writer one-for-one. The generator script is
the single source of truth: when the writer changes, this script must
change too — the gen-and-diff CI check will fail if they drift.

Why not a Pydantic model? The writers are intentionally allocation-
free in the cancellation hot path (the trainer is already aborting).
Adding a Pydantic round-trip would change observable behaviour. We
keep the literal close to the writer with a direct
``grep cancelled.marker`` annotation in the comments.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

_CONTRACTS_DIR = Path(__file__).resolve().parent.parent
_CANCELLED_OUT = _CONTRACTS_DIR / "marker_cancelled_schema.json"
_COMPLETION_OUT = _CONTRACTS_DIR / "marker_completion_schema.json"


def _verify_writer_alive() -> None:
    """Best-effort import the production marker writers.

    If the import fails the contract is dead — exit non-zero so the
    next regen run surfaces it as a CI failure rather than silently
    keeping a stale schema.
    """
    try:
        # Smoke-imports — the actual writers are private methods,
        # so we just confirm their containing modules load cleanly.
        from ryotenkai_pod.trainer.callbacks import (  # noqa: F401
            cancellation_callback,
            completion_callback,
        )
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            f"FATAL: cannot import pod trainer callback modules ({exc!r}). "
            f"Marker contracts are dead — fix the import first."
        ) from exc


def build_cancelled_schema() -> dict[str, Any]:
    return {
        "$id": "https://ryotenkai.local/schemas/marker_cancelled.schema.json",
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "RyotenkAI Cancelled Marker",
        "version": "1.0.0",
        "description": (
            "Schema for <workspace>/cancelled.marker written by "
            "CancellationCallback._write_cancelled_marker. Mirrors the "
            "literal payload at packages/pod/src/ryotenkai_pod/trainer/"
            "callbacks/cancellation_callback.py:_write_cancelled_marker."
        ),
        "type": "object",
        "properties": {
            "run_id": {"type": ["string", "null"]},
            "flushed_count": {"type": "integer", "minimum": 0},
            "ts_ms": {"type": "integer", "minimum": 0},
            "reason": {
                "type": "string",
                "enum": ["flush_budget_exceeded"],
            },
        },
        "required": ["flushed_count", "ts_ms", "reason"],
        "additionalProperties": False,
    }


def build_completion_schema() -> dict[str, Any]:
    return {
        "$id": "https://ryotenkai.local/schemas/marker_completion.schema.json",
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "RyotenkAI Completion Marker",
        "version": "1.0.0",
        "description": (
            "Schema for <workspace>/completion.marker written by "
            "CompletionCallback._write_completion_marker. Mirrors the "
            "literal payload at packages/pod/src/ryotenkai_pod/trainer/"
            "callbacks/completion_callback.py:_write_completion_marker."
        ),
        "type": "object",
        "properties": {
            "run_id": {"type": ["string", "null"]},
            "flushed_count": {"type": "integer", "minimum": 0},
            "flush_timed_out": {"type": "boolean"},
            "ts_ms": {"type": "integer", "minimum": 0},
            "reason": {
                "type": "string",
                "enum": ["natural_completion", "flush_budget_exceeded"],
            },
        },
        "required": ["flushed_count", "flush_timed_out", "ts_ms", "reason"],
        "additionalProperties": False,
    }


def main() -> int:
    _verify_writer_alive()

    _CANCELLED_OUT.write_text(
        json.dumps(build_cancelled_schema(), indent=2, sort_keys=True) + "\n",
    )
    _COMPLETION_OUT.write_text(
        json.dumps(build_completion_schema(), indent=2, sort_keys=True) + "\n",
    )

    repo_root = _CONTRACTS_DIR.parent.parent
    print(f"wrote {_CANCELLED_OUT.relative_to(repo_root)}")
    print(f"wrote {_COMPLETION_OUT.relative_to(repo_root)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
