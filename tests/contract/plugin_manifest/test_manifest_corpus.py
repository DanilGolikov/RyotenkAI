"""Corpus test: every committed ``community/**/manifest.toml`` validates.

If a plugin author lands a manifest that the production model rejects
the corpus test fires here with a precise file path + parsed reason —
better than discovering it at pipeline-launch time.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import jsonschema
import pytest

from ryotenkai_community.manifest import LibManifest, PluginManifest, PresetManifest
from tests._contracts import CONTRACTS_DIR

pytestmark = [pytest.mark.contract, pytest.mark.compliance]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_COMMUNITY_DIR = _REPO_ROOT / "community"


def _discover_manifests() -> list[Path]:
    if not _COMMUNITY_DIR.is_dir():
        return []
    return sorted(_COMMUNITY_DIR.rglob("manifest.toml"))


def _classify(payload: dict) -> str:
    """Return 'plugin' / 'preset' / 'lib' based on top-level shape."""
    if "plugin" in payload:
        return "plugin"
    if "preset" in payload:
        return "preset"
    if "lib" in payload:
        return "lib"
    return "unknown"


@pytest.mark.parametrize(
    "manifest_path",
    _discover_manifests(),
    ids=lambda p: str(p.relative_to(_REPO_ROOT)) if isinstance(p, Path) else str(p),
)
def test_manifest_validates_against_pydantic(manifest_path: Path) -> None:
    """Every TOML in ``community/**`` parses + validates."""
    raw = manifest_path.read_bytes()
    payload = tomllib.loads(raw.decode("utf-8"))
    kind = _classify(payload)
    rel = manifest_path.relative_to(_REPO_ROOT)

    if kind == "plugin":
        try:
            PluginManifest.model_validate(payload)
        except Exception as exc:
            pytest.fail(f"{rel}: PluginManifest validation failed: {exc}")
    elif kind == "preset":
        try:
            PresetManifest.model_validate(payload)
        except Exception as exc:
            pytest.fail(f"{rel}: PresetManifest validation failed: {exc}")
    elif kind == "lib":
        try:
            LibManifest.model_validate(payload)
        except Exception as exc:
            pytest.fail(f"{rel}: LibManifest validation failed: {exc}")
    else:
        pytest.fail(
            f"{rel}: cannot classify manifest — top-level keys: "
            f"{sorted(payload.keys())}",
        )


def test_plugin_manifest_schema_matches_real_corpus() -> None:
    """The committed JSON-schema accepts every real plugin manifest.

    A drift between the generator and the model would show up here:
    if the schema is older than the model, real manifests would
    fail JSON-schema validation even though the Pydantic model
    accepts them.
    """
    schema_path = CONTRACTS_DIR / "plugin_manifest_v5_schema.json"
    schema = json.loads(schema_path.read_text())
    validator = jsonschema.Draft202012Validator(schema)

    failures: list[str] = []
    for manifest_path in _discover_manifests():
        payload = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
        if _classify(payload) != "plugin":
            continue
        rel = manifest_path.relative_to(_REPO_ROOT)
        # PluginManifest validates first → only schema-validate things
        # the model would accept; that way schema bugs surface, not
        # malformed user data.
        try:
            normalised = PluginManifest.model_validate(payload).model_dump(by_alias=True)
        except Exception:
            continue
        errors = sorted(validator.iter_errors(normalised), key=lambda e: e.path)
        if errors:
            joined = "; ".join(f"{list(e.path)}: {e.message}" for e in errors)
            failures.append(f"{rel}: {joined}")
    assert not failures, "Schema rejected real manifests:\n" + "\n".join(failures)
