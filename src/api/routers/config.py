from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from src.api.dependencies import get_runs_dir
from src.api.schemas.config_preset import ConfigPreset, ConfigPresetsResponse
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
    """Return curated starter configs from ``configs/presets/*.yaml``.

    Preset YAMLs follow a light convention in the leading comment
    block::

        # Preset: <display_name>
        # <first line of description>
        # <more description…>

    The ``# Preset:`` line becomes ``display_name`` (dropdown label);
    remaining ``#`` lines join into ``description`` (secondary text).
    ``name`` is always the file stem — so prefixing filenames with
    ``01-``, ``02-`` etc. forces alphabetical order to match the
    intended display order without leaking digits into the UI.
    """
    presets_dir = Path("configs/presets").expanduser().resolve()
    items: list[ConfigPreset] = []
    if presets_dir.is_dir():
        for path in sorted(presets_dir.glob("*.yaml")):
            text = path.read_text(encoding="utf-8")
            display_name = ""
            description_lines: list[str] = []
            for line in text.splitlines():
                stripped = line.strip()
                if stripped.startswith("#"):
                    content = stripped.lstrip("# ").rstrip()
                    if not display_name and content.lower().startswith("preset:"):
                        display_name = content.split(":", 1)[1].strip()
                    elif content:
                        description_lines.append(content)
                elif stripped:
                    break
            description = " ".join(description_lines).strip()
            items.append(
                ConfigPreset(
                    name=path.stem,
                    display_name=display_name or path.stem,
                    description=description,
                    yaml=text,
                )
            )
    return ConfigPresetsResponse(presets=items)
