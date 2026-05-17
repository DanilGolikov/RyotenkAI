"""Unit tests: :mod:`ryotenkai_shared.events.upcasters`.

Phase 1 ships with an empty registry — the tests verify the chain runner
behaves correctly as a no-op now AND that registering a synthetic hop
composes properly so future phases can extend without worry.

Tests register / clear at module-private state, so we ``clear()`` in
fixtures to keep cases isolated.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from ryotenkai_shared.events.upcasters import (
    apply_chain,
    clear,
    latest_version_for,
    register,
)


@pytest.fixture(autouse=True)
def _reset_registry() -> Iterator[None]:
    clear()
    yield
    clear()


def _make_envelope(schema_version: int = 1, **payload_overrides: Any) -> dict[str, Any]:
    payload = {"learning_rate": 1e-4}
    payload.update(payload_overrides)
    return {
        "kind": "ryotenkai.test.upcast",
        "schema_version": schema_version,
        "payload": payload,
    }


# ===========================================================================
# 1. Positive
# ===========================================================================


class TestPositive:
    def test_empty_registry_returns_input_unchanged(self) -> None:
        raw = _make_envelope()
        result = apply_chain(raw, "ryotenkai.test.upcast", 1, 1)
        assert result == raw

    def test_single_hop_invoked_when_target_greater_than_current(self) -> None:
        calls: list[tuple[int, int]] = []

        def hop_1_to_2(raw: dict[str, Any], frm: int, to: int) -> dict[str, Any]:
            calls.append((frm, to))
            payload = dict(raw["payload"])
            payload["new_field"] = "added"
            return {**raw, "payload": payload}

        register("ryotenkai.test.upcast", hop_1_to_2)
        out = apply_chain(_make_envelope(), "ryotenkai.test.upcast", 1, 2)
        assert calls == [(1, 2)]
        assert out["payload"]["new_field"] == "added"
        assert out["schema_version"] == 2

    def test_two_hops_compose_in_registration_order(self) -> None:
        order: list[str] = []

        def hop_1_to_2(raw: dict[str, Any], _frm: int, _to: int) -> dict[str, Any]:
            order.append("first")
            return {**raw, "stage": "after-1"}

        def hop_2_to_3(raw: dict[str, Any], _frm: int, _to: int) -> dict[str, Any]:
            order.append("second")
            assert raw["stage"] == "after-1"
            return {**raw, "stage": "after-2"}

        register("ryotenkai.test.upcast", hop_1_to_2)
        register("ryotenkai.test.upcast", hop_2_to_3)
        out = apply_chain(_make_envelope(), "ryotenkai.test.upcast", 1, 3)
        assert order == ["first", "second"]
        assert out["stage"] == "after-2"
        assert out["schema_version"] == 3


# ===========================================================================
# 2. Negative
# ===========================================================================


class TestNegative:
    def test_missing_hop_raises_key_error(self) -> None:
        # No hops registered but caller asks to migrate to v2 — bug.
        with pytest.raises(KeyError):
            apply_chain(_make_envelope(), "ryotenkai.test.upcast", 1, 2)


# ===========================================================================
# 3. Invariants
# ===========================================================================


class TestInvariants:
    def test_apply_chain_with_current_equal_target_is_identity(self) -> None:
        raw = _make_envelope()
        result = apply_chain(raw, "ryotenkai.test.upcast", 5, 5)
        assert result is raw  # exact reference equality — no copy

    def test_apply_chain_with_current_greater_than_target_is_identity(self) -> None:
        raw = _make_envelope()
        result = apply_chain(raw, "ryotenkai.test.upcast", 7, 3)
        assert result is raw

    def test_latest_version_for_unregistered_is_1(self) -> None:
        assert latest_version_for("ryotenkai.something.brand_new") == 1

    def test_latest_version_for_registered_is_1_plus_hop_count(self) -> None:
        register(
            "ryotenkai.test.upcast",
            lambda raw, _f, _t: raw,
        )
        register(
            "ryotenkai.test.upcast",
            lambda raw, _f, _t: raw,
        )
        assert latest_version_for("ryotenkai.test.upcast") == 3


# ===========================================================================
# 4. Logic-specific
# ===========================================================================


class TestLogicSpecific:
    def test_chain_pins_schema_version_after_completion(self) -> None:
        def hop(raw: dict[str, Any], _frm: int, _to: int) -> dict[str, Any]:
            # Deliberately leave schema_version stale.
            return raw

        register("ryotenkai.test.upcast", hop)
        out = apply_chain(_make_envelope(schema_version=1), "ryotenkai.test.upcast", 1, 2)
        assert out["schema_version"] == 2

    def test_clear_empties_registry(self) -> None:
        register(
            "ryotenkai.test.upcast",
            lambda raw, _f, _t: raw,
        )
        assert latest_version_for("ryotenkai.test.upcast") == 2
        clear()
        assert latest_version_for("ryotenkai.test.upcast") == 1
