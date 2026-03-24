from __future__ import annotations

import importlib
import logging
import uuid
from pathlib import Path

from src.data.validation.discovery import ensure_validation_plugins_discovered
from src.data.validation.registry import ValidationPluginRegistry
from src.reports.plugins.discovery import ensure_report_plugins_discovered
from src.reports.plugins.registry import ReportPluginRegistry, build_report_plugins
from src.utils.plugin_discovery import discover_and_import_modules, discover_modules


def _create_package(tmp_path: Path, files: dict[str, str]) -> str:
    package_name = f"tmp_plugin_pkg_{uuid.uuid4().hex}"
    package_root = tmp_path / package_name
    package_root.mkdir(parents=True, exist_ok=True)
    (package_root / "__init__.py").write_text("", encoding="utf-8")

    for relative_path, content in files.items():
        file_path = package_root / relative_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")

    return package_name


def test_discover_modules_skips_test_files(monkeypatch, tmp_path: Path) -> None:
    package_name = _create_package(
        tmp_path,
        {
            "plugin_a.py": "VALUE = 1\n",
            "nested/plugin_b.py": "VALUE = 2\n",
            "test_plugin.py": "VALUE = 3\n",
            "nested/test_nested.py": "VALUE = 4\n",
        },
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()

    modules = discover_modules(package_name)

    assert f"{package_name}.plugin_a" in modules
    assert f"{package_name}.nested.plugin_b" in modules
    assert all(".test_" not in module for module in modules)


def test_discovery_collects_failures_but_keeps_successes(monkeypatch, tmp_path: Path) -> None:
    package_name = _create_package(
        tmp_path,
        {
            "good_plugin.py": "VALUE = 'ok'\n",
            "broken_plugin.py": "raise RuntimeError('boom')\n",
        },
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()

    diagnostics = discover_and_import_modules(package_name, logger=logging.getLogger(__name__))

    assert f"{package_name}.good_plugin" in diagnostics.imported_modules
    assert any(item.module_name == f"{package_name}.broken_plugin" for item in diagnostics.failed_modules)
    assert len(diagnostics.candidate_modules) == 2


def test_force_discovery_restores_validation_registry() -> None:
    ValidationPluginRegistry.clear()
    assert ValidationPluginRegistry.list_plugins() == []

    diagnostics = ensure_validation_plugins_discovered(force=True)

    assert diagnostics.imported_modules
    assert "min_samples" in ValidationPluginRegistry.list_plugins()


def test_report_discovery_builds_sorted_runtime_plugins_only() -> None:
    ReportPluginRegistry.clear()

    diagnostics = ensure_report_plugins_discovered(force=True)
    plugins = build_report_plugins()
    plugin_ids = [plugin.plugin_id for plugin in plugins]
    orders = [plugin.order for plugin in plugins]

    assert diagnostics.imported_modules
    assert plugin_ids == [plugin.plugin_id for plugin in sorted(plugins, key=lambda plugin: plugin.order)]
    assert orders == sorted(orders)
    assert len(set(plugin_ids)) == len(plugin_ids)
    assert len(set(orders)) == len(orders)
    assert "event_timeline" not in plugin_ids
