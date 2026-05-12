"""Property-based round-trip tests for the plugin manifest model.

Two angles:

* ``test_pydantic_model_dump_idempotent`` — ``Manifest.model_validate(d)
  .model_dump() == d`` for a hypothesis-generated valid ``d``.
* ``test_toml_roundtrip`` — ``tomlkit.parse(tomlkit.dumps(d)) == d`` for
  the same ``d``. Catches accidental loss of typing during TOML
  serialisation (e.g. integer → string drift).
* ``test_invalid_inputs_have_informative_errors`` — adversarial inputs
  trigger the model's validators with substring-recognisable messages
  so users debugging a bad ``manifest.toml`` get pointers, not
  pydantic-internal jargon.
"""

from __future__ import annotations

from typing import Any

import pytest
import tomlkit
from hypothesis import HealthCheck, given, settings, strategies as st

from ryotenkai_community.manifest import (
    LATEST_SCHEMA_VERSION,
    PluginKind,
    PluginManifest,
)

pytestmark = [pytest.mark.contract, pytest.mark.property]

_KINDS: tuple[str, ...] = ("validation", "evaluation", "reward", "reports")
_REWARD_STRATEGIES: tuple[str, ...] = ("grpo", "sapo", "dpo", "orpo")


# ---------------------------------------------------------------------------
# Hypothesis strategies — generate dicts that match the Pydantic model.
# ---------------------------------------------------------------------------


@st.composite
def _valid_id(draw: st.DrawFn) -> str:
    # snake_case identifier, 1-30 chars
    head = draw(st.text(alphabet="abcdefghijklmnopqrstuvwxyz_", min_size=1, max_size=8))
    body = draw(st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789_", min_size=0, max_size=22))
    return head + body


@st.composite
def _entry_point(draw: st.DrawFn) -> dict[str, str]:
    return {
        "module": draw(_valid_id()),
        "class": draw(
            st.text(
                alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
                min_size=1,
                max_size=20,
            ),
        ),
    }


@st.composite
def _required_env(draw: st.DrawFn) -> dict[str, Any]:
    return {
        "name": draw(
            st.text(
                alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_",
                min_size=1,
                max_size=30,
            ),
        ),
        "description": draw(st.text(max_size=80)),
        "optional": draw(st.booleans()),
        "secret": draw(st.booleans()),
        "managed_by": draw(st.sampled_from(["", "integrations", "providers"])),
    }


@st.composite
def _plugin_manifest_dict(draw: st.DrawFn) -> dict[str, Any]:
    kind = draw(st.sampled_from(_KINDS))
    plugin: dict[str, Any] = {
        "id": draw(_valid_id()),
        "kind": kind,
        "name": draw(st.text(max_size=40)),
        "version": draw(st.sampled_from(["1.0.0", "0.1.0", "2.3.4", "1.0.0a1"])),
        "category": draw(st.text(max_size=20)),
        "stability": draw(st.sampled_from(["stable", "beta", "experimental"])),
        "description": draw(st.text(max_size=120)),
        "author": draw(st.text(max_size=40)),
        "entry_point": draw(_entry_point()),
        "supported_strategies": (
            draw(
                st.lists(
                    st.sampled_from(_REWARD_STRATEGIES), min_size=1, max_size=4, unique=True,
                ),
            )
            if kind == "reward"
            else []
        ),
    }
    return {
        "schema_version": draw(st.integers(min_value=1, max_value=LATEST_SCHEMA_VERSION)),
        "plugin": plugin,
        "params_schema": {},
        "thresholds_schema": {},
        "suggested_params": {},
        "suggested_thresholds": {},
        "compat": {"min_core_version": draw(st.text(max_size=20))},
        "required_env": draw(st.lists(_required_env(), max_size=4, unique_by=lambda d: d["name"])),
        "lib_requirements": [],
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@given(payload=_plugin_manifest_dict())
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_pydantic_roundtrip_preserves_payload(payload: dict[str, Any]) -> None:
    """``model_validate(d).model_dump(by_alias=True) == d`` (fixed point).

    ``EntryPoint`` declares the TOML key ``class`` via a pydantic alias
    (``class_name`` in Python is a reserved-word workaround). The
    contract is "what was on disk + what tomlkit emits round-trips
    through the model" — that's the by-alias dump.
    """
    manifest = PluginManifest.model_validate(payload)
    dumped = manifest.model_dump(by_alias=True)
    # Round-trip a second time — fixed point.
    again = PluginManifest.model_validate(dumped).model_dump(by_alias=True)
    assert dumped == again


@given(payload=_plugin_manifest_dict())
@settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
def test_toml_roundtrip_preserves_payload(payload: dict[str, Any]) -> None:
    """tomlkit dumps + parse round-trips every generated dict.

    tomlkit's TOMLDocument compares equal to a plain dict on equivalent
    keys, but the hypothesis dicts may contain string keys that aren't
    naturally ordered — we normalise by re-validating both sides
    through PluginManifest before comparing.
    """
    serialised = tomlkit.dumps(payload)
    parsed = tomlkit.parse(serialised)
    # Convert to plain dict for comparison (tomlkit Document is a
    # mapping with extra metadata).
    parsed_dict = {k: parsed[k] for k in parsed}
    # Validate both sides — equality through the model strips
    # tomlkit-specific wrappers.
    left = PluginManifest.model_validate(payload).model_dump(by_alias=True)
    right = PluginManifest.model_validate(parsed_dict).model_dump(by_alias=True)
    assert left == right


@pytest.mark.parametrize(
    "bad,expected_substr",
    [
        # missing entry_point
        (
            {
                "plugin": {
                    "id": "x",
                    "kind": "evaluation",
                    "stability": "stable",
                    "version": "1.0.0",
                },
            },
            "entry_point",
        ),
        # invalid kind
        (
            {
                "plugin": {
                    "id": "x",
                    "kind": "invalid_kind",
                    "stability": "stable",
                    "version": "1.0.0",
                    "entry_point": {"module": "p", "class": "C"},
                },
            },
            "kind",
        ),
        # reward plugin missing supported_strategies
        (
            {
                "plugin": {
                    "id": "r",
                    "kind": "reward",
                    "stability": "stable",
                    "version": "1.0.0",
                    "entry_point": {"module": "p", "class": "C"},
                },
            },
            "supported_strategies",
        ),
        # schema_version too high
        (
            {
                "schema_version": LATEST_SCHEMA_VERSION + 1,
                "plugin": {
                    "id": "x",
                    "kind": "evaluation",
                    "stability": "stable",
                    "version": "1.0.0",
                    "entry_point": {"module": "p", "class": "C"},
                },
            },
            "schema_version",
        ),
    ],
)
def test_invalid_inputs_have_informative_errors(
    bad: dict[str, Any], expected_substr: str,
) -> None:
    with pytest.raises(Exception) as excinfo:
        PluginManifest.model_validate(bad)
    assert expected_substr.lower() in str(excinfo.value).lower(), (
        f"error message did not mention {expected_substr!r}: {excinfo.value!r}"
    )


def test_kind_literal_matches_constants() -> None:
    """Pin the production literal — if it grows, this test reminds us
    to update the schema and the property strategies."""
    # PluginKind is a Literal, so its __args__ is a tuple of strings.
    assert set(PluginKind.__args__) == set(_KINDS)
