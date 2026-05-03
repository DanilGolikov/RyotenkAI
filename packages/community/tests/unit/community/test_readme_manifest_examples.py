"""Guardrail: every full ``manifest.toml`` example embedded in a community
README must parse against the current pydantic models.

This catches README rot — if ``PluginManifest`` / ``PresetManifest`` change,
the examples are caught here rather than misleading plugin authors.

Partial fragments (e.g. the v2 ``[preset.scope]`` snippet that doesn't
include ``[preset]`` itself) are explicitly skipped: full examples are
identified by the presence of ``plugin.id`` or ``preset.id`` + their
``entry_point`` block.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

from src.community.constants import COMMUNITY_ROOT
from src.community.manifest import PluginManifest, PresetManifest

_TOML_BLOCK = re.compile(r"```toml\n(.*?)```", re.DOTALL)
_EXPECTED_READMES = (
    COMMUNITY_ROOT / "validation" / "README.md",
    COMMUNITY_ROOT / "evaluation" / "README.md",
    COMMUNITY_ROOT / "reward" / "README.md",
    COMMUNITY_ROOT / "reports" / "README.md",
    COMMUNITY_ROOT / "presets" / "README.md",
)


def _extract_toml_blocks(readme: Path) -> list[str]:
    return _TOML_BLOCK.findall(readme.read_text(encoding="utf-8"))


def _classify(doc: dict) -> str:
    """Return 'plugin' | 'preset' | 'partial' based on required structure."""
    plugin = doc.get("plugin") or {}
    if plugin.get("id") and plugin.get("entry_point"):
        return "plugin"

    preset = doc.get("preset") or {}
    if preset.get("id") and preset.get("entry_point"):
        return "preset"

    return "partial"


@pytest.mark.parametrize("readme", _EXPECTED_READMES, ids=lambda p: p.parent.name)
def test_readme_exists(readme: Path) -> None:
    assert readme.is_file(), f"missing {readme.relative_to(COMMUNITY_ROOT)}"


@pytest.mark.parametrize("readme", _EXPECTED_READMES, ids=lambda p: p.parent.name)
def test_readme_toml_examples_validate(readme: Path) -> None:
    blocks = _extract_toml_blocks(readme)
    assert blocks, f"{readme.relative_to(COMMUNITY_ROOT)} has no ```toml``` blocks"

    validated_any = False
    for i, raw in enumerate(blocks):
        try:
            doc = tomllib.loads(raw)
        except tomllib.TOMLDecodeError as exc:
            pytest.fail(
                f"{readme.relative_to(COMMUNITY_ROOT)} block #{i} is not valid TOML: {exc}\n"
                f"---\n{raw}"
            )

        kind = _classify(doc)
        if kind == "plugin":
            PluginManifest.model_validate(doc)
            validated_any = True
        elif kind == "preset":
            PresetManifest.model_validate(doc)
            validated_any = True
        # 'partial' — e.g. a `[preset.scope]` snippet without the full
        # `[preset]` header — is documentation, not a loadable manifest.
        # We leave it alone.

    assert validated_any, (
        f"{readme.relative_to(COMMUNITY_ROOT)} has no FULL manifest example — "
        "add at least one complete manifest.toml block so authors have "
        "something copy-paste-ready."
    )
