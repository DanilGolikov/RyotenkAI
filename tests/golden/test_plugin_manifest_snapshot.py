"""Snapshot нормализованного PluginManifest.

Покрывает критическое свойство загрузки манифестов: дефолты,
заполнение поля ``name`` через ``_fill_name``, форма JSON Schema
после ``model_dump`` — все они должны быть стабильны между
релизами. Любое изменение схемы манифеста v5 → v6 заметит этот
snapshot.
"""

from __future__ import annotations

import pytest

from ryotenkai_community.manifest import (
    EntryPoint,
    PluginManifest,
    PluginSpec,
)

pytestmark = pytest.mark.golden


def _make_manifest() -> PluginManifest:
    """Канонический минимальный манифест плагина — фиксированные
    значения для воспроизводимости snapshot'а."""
    return PluginManifest(
        plugin=PluginSpec(
            id="hello_world",
            kind="validation",
            # name пустое — validator должен заполнить из id
            version="1.0.0",
            description="Test plugin",
            author="Test <test@example.com>",
            entry_point=EntryPoint(module="hello_world", **{"class": "HelloPlugin"}),
        ),
    )


def test_minimal_manifest_canonical_dump(snapshot) -> None:
    """Дамп минимального манифеста v5 — baseline схемы."""
    manifest = _make_manifest()
    payload = manifest.model_dump(by_alias=True)
    assert payload == snapshot
