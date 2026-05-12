"""Generate ``runner_internal_events_schema.json`` from
:class:`ryotenkai_shared.contracts.runner_api.InternalEventRequest`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

_CONTRACTS_DIR = Path(__file__).resolve().parent.parent
_OUT_PATH = _CONTRACTS_DIR / "runner_internal_events_schema.json"


def build_schema() -> dict[str, Any]:
    try:
        from ryotenkai_shared.contracts.runner_api import InternalEventRequest
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            f"FATAL: cannot import InternalEventRequest ({exc!r}). The runner "
            f"internal-events contract is dead."
        ) from exc

    schema = InternalEventRequest.model_json_schema()
    schema["$id"] = (
        "https://ryotenkai.local/schemas/runner_internal_events.schema.json"
    )
    schema["title"] = "Runner Internal Events Request"
    schema["version"] = "1.0.0"
    schema.setdefault("$schema", "https://json-schema.org/draft/2020-12/schema")
    schema["description"] = (
        "Body of POST /api/v1/internal/events. Generated from "
        "ryotenkai_shared.contracts.runner_api.InternalEventRequest."
    )
    return schema


def main() -> int:
    schema = build_schema()
    _OUT_PATH.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n")
    print(f"wrote {_OUT_PATH.relative_to(_CONTRACTS_DIR.parent.parent)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
