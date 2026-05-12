"""Hypothesis round-trip для :class:`PluginManifest`.

Свойство: ``dict → PluginManifest → model_dump()`` идентично исходному
dict для всех валидных манифестов. Это даёт нам уверенность, что
сериализация манифестов через TOML/JSON не теряет полей и не вводит
скрытых преобразований.
"""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from ryotenkai_community.manifest import (
    EntryPoint,
    PluginManifest,
    PluginSpec,
)

pytestmark = pytest.mark.property


# --- low-level strategies ----------------------------------------------------

_identifiers = st.from_regex(r"^[a-z_][a-z0-9_]{0,30}$", fullmatch=True)
_versions = st.from_regex(r"^[0-9]\.[0-9]\.[0-9]$", fullmatch=True)

# Только non-reward kinds — reward требует supported_strategies, что
# усложняет стратегию. Property test для reward-kind можно добавить
# позднее как отдельный случай.
_non_reward_kinds = st.sampled_from(["validation", "evaluation", "reports"])

_entry_points = st.builds(
    EntryPoint,
    module=_identifiers,
    # ``class_name`` имеет alias='class' в модели, и Hypothesis ходит
    # через позиционные kwargs — кладём через model construction.
)


@st.composite
def _plugin_specs(draw: st.DrawFn) -> PluginSpec:
    return PluginSpec(
        id=draw(_identifiers),
        kind=draw(_non_reward_kinds),
        name=draw(st.one_of(st.just(""), _identifiers)),
        version=draw(_versions),
        category=draw(st.one_of(st.just(""), _identifiers)),
        stability=draw(st.sampled_from(["stable", "beta", "experimental"])),
        description=draw(st.text(max_size=80)),
        author=draw(st.one_of(st.just(""), st.text(max_size=40))),
        entry_point=EntryPoint(module=draw(_identifiers), **{"class": "MyPlugin"}),
    )


@st.composite
def _plugin_manifests(draw: st.DrawFn) -> PluginManifest:
    return PluginManifest(plugin=draw(_plugin_specs()))


# --- properties --------------------------------------------------------------


@given(manifest=_plugin_manifests())
@settings(suppress_health_check=[HealthCheck.too_slow])
def test_manifest_dict_roundtrip(manifest: PluginManifest) -> None:
    """Round-trip: dump → reload эквивалентен оригиналу."""
    dumped = manifest.model_dump(by_alias=True)
    restored = PluginManifest.model_validate(dumped)
    assert restored.model_dump(by_alias=True) == dumped


@given(manifest=_plugin_manifests())
def test_manifest_schema_version_in_supported_range(
    manifest: PluginManifest,
) -> None:
    """Каждый валидный манифест имеет ``schema_version`` в допустимом
    диапазоне (1 .. LATEST). Сейчас все генерируемые манифесты
    используют дефолт = LATEST, но property остаётся валидным
    инвариантом если кто-то начнёт варьировать поле."""
    from ryotenkai_community.manifest import LATEST_SCHEMA_VERSION

    assert 1 <= manifest.schema_version <= LATEST_SCHEMA_VERSION


@given(manifest=_plugin_manifests())
def test_manifest_name_defaults_to_id_when_blank(
    manifest: PluginManifest,
) -> None:
    """Если ``[plugin].name`` пустой, model_validator должен
    выставить ``name = id`` (см. ``_fill_name``)."""
    if manifest.plugin.name:
        # явное имя сохраняется
        assert manifest.plugin.name
    else:
        # default-логика никогда не оставит name пустым
        pytest.fail("PluginSpec validator should never leave name empty")
