"""Generate ``runner_events_schema.json`` for ``events.NNN.jsonl`` records.

The shape is fixed by :class:`ryotenkai_pod.runner.event_journal.JournalRecord`
and the writer at ``EventJournal.append`` (which lays the record down
as ``{"v":SCHEMA_VERSION,"offset":N,"ts":...,"kind":"...","payload":{...}}``).
The schema is generated from those production constants — when
``SCHEMA_VERSION`` bumps, the generated artifact bumps too.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

_CONTRACTS_DIR = Path(__file__).resolve().parent.parent
_OUT_PATH = _CONTRACTS_DIR / "runner_events_schema.json"


def build_schema() -> dict[str, Any]:
    try:
        from ryotenkai_pod.runner.event_journal import (
            MAX_SUPPORTED_SCHEMA_VERSION,
            SCHEMA_VERSION,
        )
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            f"FATAL: cannot import EventJournal constants ({exc!r}). "
            f"runner-events contract is dead."
        ) from exc

    return {
        "$id": "https://ryotenkai.local/schemas/runner_events.schema.json",
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "RyotenkAI Runner Event Journal Record",
        "version": f"{SCHEMA_VERSION}.0.0",
        "description": (
            "JSONL record schema for <workspace>/events/events.NNN.jsonl. "
            "Generated from ryotenkai_pod.runner.event_journal.append + "
            "JournalRecord. SCHEMA_VERSION="
            f"{SCHEMA_VERSION}, MAX_SUPPORTED={MAX_SUPPORTED_SCHEMA_VERSION}."
        ),
        "type": "object",
        "properties": {
            "v": {
                "type": "integer",
                "minimum": 1,
                "maximum": MAX_SUPPORTED_SCHEMA_VERSION,
            },
            "offset": {"type": "integer", "minimum": 0},
            "ts": {"type": "string"},
            "kind": {"type": "string", "minLength": 1, "maxLength": 64},
            "payload": {"type": "object"},
        },
        "required": ["v", "offset", "ts", "kind", "payload"],
        "additionalProperties": False,
    }


def main() -> int:
    schema = build_schema()
    _OUT_PATH.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n")
    print(f"wrote {_OUT_PATH.relative_to(_CONTRACTS_DIR.parent.parent)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
