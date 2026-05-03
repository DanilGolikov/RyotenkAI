from __future__ import annotations

from pydantic import BaseModel, Field


class LogFileInfo(BaseModel):
    name: str
    path: str
    size_bytes: int
    exists: bool


class LogChunk(BaseModel):
    file: str
    offset: int = Field(ge=0)
    next_offset: int = Field(ge=0)
    eof: bool
    content: str
