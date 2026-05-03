from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from src.api.dependencies import get_runs_dir
from src.api.schemas.config_preset import (
    ConfigPreset,
    ConfigPresetsResponse,
    PresetDiffEntry,
    PresetPlaceholderHint,
    PresetPreviewRequest,
    PresetPreviewResponse,
    PresetRequirementCheck,
    PresetRequirementsOut,
    PresetScopeOut,
)
from src.api.schemas.config_validate import ConfigValidationResult
from src.api.services import config_service

router = APIRouter(prefix="/config", tags=["config"])


class ConfigValidateRequest(BaseModel):
    config_path: str


class ConfigTemplate(BaseModel):
    name: str
    path: str


class DefaultConfigResponse(BaseModel):
    runs_dir: str
    config_templates: list[ConfigTemplate] = Field(default_factory=list)


@router.post("/validate", response_model=ConfigValidationResult)
def validate(body: ConfigValidateRequest) -> ConfigValidationResult:
    config_path = Path(body.config_path).expanduser().resolve()
    if not config_path.exists():
        raise HTTPException(status_code=404, detail=f"config file not found: {config_path}")
    return config_service.validate_config(config_path)


@router.get("/default", response_model=DefaultConfigResponse)
def default(runs_dir: Path = Depends(get_runs_dir)) -> DefaultConfigResponse:
    examples_dir = Path("examples").expanduser().resolve()
    templates: list[ConfigTemplate] = []
    if examples_dir.is_dir():
        for path in sorted(examples_dir.glob("*.yaml")):
            templates.append(ConfigTemplate(name=path.name, path=str(path)))
    return DefaultConfigResponse(runs_dir=str(runs_dir), config_templates=templates)


@router.get("/schema", response_model=dict)
def schema() -> dict:
    """Return the full PipelineConfig JSON schema for the UI builder."""
    from src.config.pipeline.schema import PipelineConfig

    return PipelineConfig.model_json_schema()


@router.get("/presets", response_model=ConfigPresetsResponse)
def presets() -> ConfigPresetsResponse:
    """Return curated starter configs from ``community/presets/``.

    Each preset lives in its own folder with a ``manifest.toml`` (id,
    display name, description, size tier, optional v2 scope/requirements/
    placeholders) and the actual config YAML referenced via
    ``[preset.entry_point]``.
    """
    from src.community.catalog import catalog

    items: list[ConfigPreset] = []
    for loaded in catalog.presets():
        spec = loaded.manifest.preset
        scope_out = (
            PresetScopeOut(replaces=list(spec.scope.replaces),
                           preserves=list(spec.scope.preserves))
            if spec.scope is not None else None
        )
        req_out = (
            PresetRequirementsOut(
                hub_models=list(spec.requirements.hub_models),
                provider_kind=list(spec.requirements.provider_kind),
                required_plugins=list(spec.requirements.required_plugins),
                min_vram_gb=spec.requirements.min_vram_gb,
            )
            if spec.requirements is not None else None
        )
        items.append(
            ConfigPreset(
                name=spec.id,
                display_name=spec.name or spec.id,
                description=spec.description,
                yaml=loaded.yaml_text,
                size_tier=spec.size_tier,
                scope=scope_out,
                requirements=req_out,
                placeholders=dict(spec.placeholders),
            )
        )
    return ConfigPresetsResponse(presets=items)


@router.post("/presets/{preset_id}/preview", response_model=PresetPreviewResponse)
def preview_preset(preset_id: str, body: PresetPreviewRequest) -> PresetPreviewResponse:
    """Dry-run: apply ``preset_id`` to ``current_config`` and return the
    resulting config plus a structured diff, requirements check, and
    placeholder hints.

    The endpoint never writes anything — it's what the frontend calls to
    populate the Apply-preset modal (three sections: what changes / what's
    preserved / what the user still needs to fill).
    """
    from src.community.catalog import catalog
    from src.community.preset_apply import apply_preset
    from src.config.secrets.loader import load_secrets

    try:
        loaded = next(p for p in catalog.presets() if p.manifest.preset.id == preset_id)
    except StopIteration as exc:
        raise HTTPException(status_code=404, detail=f"preset not found: {preset_id}") from exc

    # Gather environment facts that the pure apply function doesn't discover on its own.
    try:
        secrets = load_secrets()
        secrets_extra = dict(secrets.model_extra or {})
    except Exception:
        secrets_extra = {}

    available_plugins: dict[str, set[str]] = {}
    for kind in ("validation", "evaluation", "reward", "reports"):
        available_plugins[kind] = {
            lp.manifest.plugin.id for lp in catalog.plugins(kind)  # type: ignore[arg-type]
        }

    preview = apply_preset(
        body.current_config,
        loaded,
        secrets_model_extra=secrets_extra,
        available_plugin_ids_by_kind=available_plugins,
    )

    return PresetPreviewResponse(
        resulting_config=preview.resulting_config,
        diff=[PresetDiffEntry(
            key=d.key, kind=d.kind, reason=d.reason, before=d.before, after=d.after,
        ) for d in preview.diff],
        requirements=[PresetRequirementCheck(
            label=r.label, status=r.status, detail=r.detail,
        ) for r in preview.requirements],
        placeholders=[PresetPlaceholderHint(path=p.path, hint=p.hint)
                      for p in preview.placeholders],
        warnings=list(preview.warnings),
    )
