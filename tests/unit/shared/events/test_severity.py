"""Unit tests: :mod:`ryotenkai_shared.events.severity`.

Severity is a Literal alias plus an ordering map. The tests pin the
canonical low → high order so threshold comparisons in journal /
metrics code (Phase 3 / 8) cannot silently flip.
"""

from __future__ import annotations

from ryotenkai_shared.events.severity import SEVERITY_ORDER, Severity


class TestPositive:
    def test_order_keys_match_literal_members(self) -> None:
        expected = {"debug", "info", "warning", "error", "critical"}
        assert set(SEVERITY_ORDER.keys()) == expected

    def test_order_is_strictly_ascending(self) -> None:
        ranks = list(SEVERITY_ORDER.values())
        assert ranks == sorted(ranks)
        assert len(ranks) == len(set(ranks))


class TestInvariants:
    def test_debug_is_lowest_critical_is_highest(self) -> None:
        assert SEVERITY_ORDER["debug"] < SEVERITY_ORDER["critical"]
        assert SEVERITY_ORDER["debug"] == min(SEVERITY_ORDER.values())
        assert SEVERITY_ORDER["critical"] == max(SEVERITY_ORDER.values())

    def test_error_outranks_warning_outranks_info(self) -> None:
        assert SEVERITY_ORDER["info"] < SEVERITY_ORDER["warning"]
        assert SEVERITY_ORDER["warning"] < SEVERITY_ORDER["error"]

    def test_severity_alias_is_usable_in_runtime_type_check(self) -> None:
        # ``Severity`` is a Literal — runtime checks aren't supported
        # directly, but membership in SEVERITY_ORDER is a proxy.
        for value in ("debug", "info", "warning", "error", "critical"):
            sev: Severity = value  # type: ignore[assignment]
            assert sev in SEVERITY_ORDER
