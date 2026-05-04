"""Shared Pydantic base for runner wire DTOs."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class _StrictModel(BaseModel):
    """Base — forbid extras so contract drift surfaces at parse time."""

    model_config = ConfigDict(extra="forbid")


__all__ = ["_StrictModel"]
