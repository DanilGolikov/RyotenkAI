"""Plugin catalogue service.

Exposes the plugin manifests loaded by ``CommunityCatalog`` to the web UI
in the UI-friendly shape defined by ``PluginManifest`` / ``PluginListResponse``.
Per-entry load failures (broken manifest, import error, etc.) are
surfaced alongside the successful manifests so the UI can render an
amber error banner without bricking the whole catalog page.
"""

from __future__ import annotations

from src.api.schemas.plugin import (
    PluginKind,
    PluginListResponse,
    PluginLoadError,
    PluginManifest,
)
from src.community.catalog import catalog


def list_plugins(kind: PluginKind) -> PluginListResponse:
    if kind not in ("reward", "validation", "evaluation", "reports"):
        raise ValueError(f"unknown plugin kind: {kind!r}")

    catalog.ensure_loaded()
    raw = [loaded.manifest.ui_manifest() for loaded in catalog.plugins(kind)]
    plugins = [PluginManifest(**item) for item in raw]
    plugins.sort(key=lambda m: (m.category or "~", m.id))

    errors = [
        PluginLoadError(
            entry_name=f.entry_name,
            plugin_id=f.plugin_id,
            error_type=f.error_type,
            message=f.message,
            traceback=f.traceback,
        )
        for f in catalog.failures(kind)
    ]
    return PluginListResponse(kind=kind, plugins=plugins, errors=errors)


__all__ = ["list_plugins"]
