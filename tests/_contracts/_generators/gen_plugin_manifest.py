"""Generate ``plugin_manifest_v5_schema.json`` from
:class:`ryotenkai_community.manifest.PluginManifest`.

Failure mode: if the production model can't be imported the script
exits non-zero with a clear message — that's the contract telling
us "this gate is dead, fix the production import first".
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

_CONTRACTS_DIR = Path(__file__).resolve().parent.parent
_OUT_PATH = _CONTRACTS_DIR / "plugin_manifest_v5_schema.json"


def build_schema() -> dict[str, Any]:
    try:
        from ryotenkai_community.manifest import LATEST_SCHEMA_VERSION, PluginManifest
    except ImportError as exc:  # pragma: no cover -- contract-dead signal
        raise SystemExit(
            f"FATAL: cannot import ryotenkai_community.manifest.PluginManifest "
            f"({exc!r}). The plugin-manifest contract is dead — fix the import "
            f"before regenerating schemas."
        ) from exc

    schema = PluginManifest.model_json_schema()

    # Top-level metadata required for every contract artifact.
    schema["$id"] = (
        "https://ryotenkai.local/schemas/plugin_manifest_v5.schema.json"
    )
    schema["title"] = "RyotenkAI Plugin Manifest"
    # Version must be a string per Phase 3 D4 rules; bumped whenever
    # the production model gains/loses a field. The numeric pydantic
    # ``schema_version`` lives inside ``properties`` and is unrelated
    # to this artifact's own versioning.
    schema["version"] = f"{LATEST_SCHEMA_VERSION}.0.0"
    schema.setdefault("$schema", "https://json-schema.org/draft/2020-12/schema")
    schema["description"] = (
        "Generated from ryotenkai_community.manifest.PluginManifest. "
        "Do NOT edit by hand — regenerate via "
        "`python -m tests._contracts._generators.gen_plugin_manifest`."
    )
    return schema


def main() -> int:
    schema = build_schema()
    _OUT_PATH.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n")
    print(f"wrote {_OUT_PATH.relative_to(_CONTRACTS_DIR.parent.parent)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
