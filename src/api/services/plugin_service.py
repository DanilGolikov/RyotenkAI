"""Plugin catalogue service.

Exposes the plugin manifests loaded by ``CommunityCatalog`` to the web UI
in the UI-friendly shape defined by ``PluginManifest`` / ``PluginListResponse``.
"""

from __future__ import annotations

from src.api.schemas.plugin import PluginKind, PluginListResponse, PluginManifest
from src.community.catalog import catalog


def list_plugins(kind: PluginKind) -> PluginListResponse:
    if kind not in ("reward", "validation", "evaluation"):
        raise ValueError(f"unknown plugin kind: {kind!r}")

    catalog.ensure_loaded()
    raw = [loaded.manifest.ui_manifest() for loaded in catalog.plugins(kind)]
    plugins = [PluginManifest(**item) for item in raw]
    plugins.sort(key=lambda m: (m.category or "~", m.id))
    return PluginListResponse(kind=kind, plugins=plugins)


__all__ = ["list_plugins"]
