"""Unit tests: :class:`ryotenkai_shared.events.types.unknown.UnknownEvent`.

The catch-all variant must accept any severity (no Literal pinning) and
preserve the original type / raw payload verbatim so consumers can render
forward-compat events.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ryotenkai_shared.events import from_jsonl, to_jsonl
from ryotenkai_shared.events.types.unknown import UNKNOWN_OFFSET, UnknownEvent


def _unknown(**overrides) -> UnknownEvent:
    defaults = {
        "source": "pod://r/x",
        "run_id": "r",
        "offset": 0,
        "original_type": "ryotenkai.future.thing",
        "raw_payload": {"a": 1},
    }
    defaults.update(overrides)
    return UnknownEvent(**defaults)


class TestPositive:
    def test_round_trip(self) -> None:
        original = _unknown(severity="warning")
        restored = from_jsonl(to_jsonl(original), strict=True)
        assert isinstance(restored, UnknownEvent)
        assert restored == original

    def test_default_severity_is_info(self) -> None:
        event = _unknown()
        assert event.severity == "info"


class TestNegative:
    def test_severity_outside_literal_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _unknown(severity="extreme")  # type: ignore[arg-type]

    def test_extra_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            UnknownEvent(  # type: ignore[call-arg]
                source="x",
                run_id="x",
                offset=0,
                original_type="x",
                raw_payload={},
                unexpected="y",
            )


class TestInvariants:
    def test_kind_pinned_to_unknown_constant(self) -> None:
        assert _unknown().kind == "ryotenkai.unknown"

    def test_severity_not_pinned_via_literal(self) -> None:
        # Unlike concrete events, UnknownEvent supports the full set.
        for s in ("debug", "info", "warning", "error", "critical"):
            event = _unknown(severity=s)
            assert event.severity == s

    def test_unknown_offset_sentinel_value(self) -> None:
        # Pin the sentinel literal — journal readers and downstream
        # consumers depend on this exact value to skip torn-write /
        # forward-compat events in monotonic-offset checks. A change
        # here is a wire-contract change and MUST be intentional.
        assert UNKNOWN_OFFSET == -1


class TestRegressions:
    def test_raw_payload_isolated_from_input_mutation(self) -> None:
        # ``frozen=True`` blocks rebinding the attribute but does not
        # prevent in-place mutation of the dict. The model_validator
        # deep-copies at construction so a caller that mutates the
        # dict they passed in cannot leak into the stored value.
        external_payload = {"step": 5, "metrics": {"loss": 0.5}}
        event = _unknown(raw_payload=external_payload)

        external_payload["step"] = 999
        external_payload["metrics"]["loss"] = 1.0
        external_payload["evil"] = "injected"

        assert event.raw_payload["step"] == 5
        assert event.raw_payload["metrics"]["loss"] == 0.5
        assert "evil" not in event.raw_payload

    def test_two_unknowns_sharing_input_dict_isolated(self) -> None:
        # Two UnknownEvent instances built from the same literal dict
        # must not silently alias each other through the shared dict.
        shared = {"counter": 0}
        first = _unknown(offset=0, raw_payload=shared)
        second = _unknown(offset=1, raw_payload=shared)

        shared["counter"] = 42

        # Both events captured snapshot=0 and are isolated from each
        # other and from the input dict.
        assert first.raw_payload["counter"] == 0
        assert second.raw_payload["counter"] == 0
        assert first.raw_payload is not second.raw_payload
