"""Print the FastAPI app's OpenAPI spec to stdout.

Used by ``make web-openapi-dump`` to produce a checked-in snapshot at
``web/src/api/openapi.json``. Running this way avoids needing a live
backend for codegen (CI is offline-safe).
"""

from __future__ import annotations

import json
import sys

from src.api.main import create_app


def main() -> None:
    app = create_app()
    spec = app.openapi()
    json.dump(spec, sys.stdout, ensure_ascii=False, indent=2, sort_keys=False)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
