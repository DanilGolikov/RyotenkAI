"""Plugin catalogue service.

Exposes the plugin manifests loaded by ``CommunityCatalog`` to the web UI
in the UI-friendly shape defined by ``PluginManifest`` / ``PluginListResponse``.
Per-entry load failures (broken manifest, import error, etc.) are
surfaced alongside the successful manifests so the UI can render an
amber error banner without bricking the whole catalog page.

Also hosts the preflight env gate (:func:`preflight`) — the Launch
modal calls it before enabling the launch button so the user sees
"set up before launch" chips for any non-optional ``[[required_env]]``
that's still unset.
"""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from src.api.schemas.plugin import (
    InstanceErrorSchema,
    MissingEnvSchema,
    PluginKind,
    PluginListResponse,
    PluginLoadError,
    PluginManifest,
    PreflightResponse,
)
from src.community.catalog import catalog
from src.community.preflight import run_preflight


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


def preflight(
    config_payload: dict[str, Any], project_env: dict[str, str] | None = None
) -> PreflightResponse:
    """Run env-availability and instance-shape gates against ``config_payload``.

    The config dict is parsed into the canonical :class:`PipelineConfig`
    so we run against the same shape the orchestrator sees — a malformed
    config raises :class:`pydantic.ValidationError` upstream as a 422.

    No secrets are loaded here: the Launch modal sends ``project_env``
    explicitly (it's the same dict the launcher merges on fork), and
    process env is read inside the preflight helpers. Plugin secrets
    that come from ``secrets.env`` are deliberately *not* consulted —
    ``secrets.env`` is operator-local and may diverge from the
    deployed environment, so the user must declare via project env.
    """
    from src.utils.config import PipelineConfig

    try:
        config = PipelineConfig.model_validate(config_payload)
    except ValidationError:
        # Re-raise so the FastAPI handler turns it into a 422 with the
        # full per-field error list.
        raise

    report = run_preflight(config, project_env=project_env or {})
    return PreflightResponse(
        ok=report.ok,
        missing=[
            MissingEnvSchema(
                plugin_kind=m.plugin_kind,
                plugin_name=m.plugin_name,
                plugin_instance_id=m.plugin_instance_id,
                name=m.name,
                description=m.description,
                secret=m.secret,
                managed_by=m.managed_by,
            )
            for m in report.missing_envs
        ],
        instance_errors=[
            InstanceErrorSchema(
                plugin_kind=e.plugin_kind,
                plugin_name=e.plugin_name,
                plugin_instance_id=e.plugin_instance_id,
                location=e.location,
                message=e.message,
            )
            for e in report.instance_errors
        ],
    )


__all__ = ["list_plugins", "preflight"]
