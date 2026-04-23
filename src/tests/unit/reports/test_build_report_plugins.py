"""Tests for config-driven report plugin selection."""

from __future__ import annotations

import pytest

from src.reports.plugins.defaults import DEFAULT_REPORT_SECTIONS
from src.reports.plugins.registry import build_report_plugins


def test_default_returns_built_in_order() -> None:
    plugins = build_report_plugins()
    assert [p.plugin_id for p in plugins] == list(DEFAULT_REPORT_SECTIONS)
    # Position-derived orders, unique and monotonically increasing.
    orders = [p.order for p in plugins]
    assert orders == sorted(orders)
    assert len(set(orders)) == len(orders)


def test_custom_sections_filter_and_reorder() -> None:
    plugins = build_report_plugins(["footer", "header"])
    assert [p.plugin_id for p in plugins] == ["footer", "header"]
    # `order` reflects config position.
    assert plugins[0].order < plugins[1].order


def test_subset_omits_everything_not_listed() -> None:
    plugins = build_report_plugins(["header", "footer"])
    assert {p.plugin_id for p in plugins} == {"header", "footer"}


def test_unknown_section_raises_with_listing() -> None:
    with pytest.raises(ValueError, match=r"Unknown report plugin ids"):
        build_report_plugins(["header", "does_not_exist"])


def test_duplicate_section_id_raises() -> None:
    with pytest.raises(ValueError, match=r"Duplicate report plugin ids"):
        build_report_plugins(["header", "summary", "header"])


def test_empty_sections_returns_empty_list() -> None:
    plugins = build_report_plugins([])
    assert plugins == []


def test_default_sections_match_registry_keys() -> None:
    """Guardrail: DEFAULT_REPORT_SECTIONS shouldn't drift away from shipped plugins."""
    from src.community.catalog import catalog
    from src.reports.plugins.registry import ReportPluginRegistry

    catalog.ensure_loaded()
    registered = set(ReportPluginRegistry.get_all())
    defaults = set(DEFAULT_REPORT_SECTIONS)
    assert defaults.issubset(registered), (
        f"DEFAULT_REPORT_SECTIONS references missing plugins: {defaults - registered}"
    )
