from __future__ import annotations

from fastapi import APIRouter, HTTPException

from src.api.schemas.plugin import PluginKind, PluginListResponse
from src.api.services import plugin_service

router = APIRouter(prefix="/plugins", tags=["plugins"])

_KINDS: tuple[PluginKind, ...] = ("reward", "validation", "evaluation")


@router.get("/{kind}", response_model=PluginListResponse)
def list_plugins(kind: PluginKind) -> PluginListResponse:
    if kind not in _KINDS:
        raise HTTPException(
            status_code=404,
            detail=f"unknown plugin kind {kind!r}; expected one of {list(_KINDS)}",
        )
    return plugin_service.list_plugins(kind)
