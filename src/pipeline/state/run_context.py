from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.utils.run_naming import generate_run_name

if TYPE_CHECKING:
    from datetime import datetime


@dataclass(frozen=True, slots=True)
class RunContext:
    """
    Canonical pipeline run context.

    This is the single source of truth for run naming across:
    - control plane logs dir
    - provider training workspace
    - provider inference workspace
    """

    name: str
    created_at_utc: datetime

    @classmethod
    def create(cls, *, now_utc: datetime | None = None) -> RunContext:
        name, created_at = generate_run_name(now_utc=now_utc)
        return cls(name=name, created_at_utc=created_at)
