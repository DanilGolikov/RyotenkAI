from __future__ import annotations

from pydantic import BaseModel


class ReportResponse(BaseModel):
    path: str
    markdown: str
    generated_at: str
    regenerated: bool = False
