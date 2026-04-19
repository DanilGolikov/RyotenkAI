from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from src.api.dependencies import get_runs_dir
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
