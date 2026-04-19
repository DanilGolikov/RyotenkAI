from __future__ import annotations

from pathlib import Path
from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class ApiSettings(BaseSettings):
    """Runtime settings for the web backend. Env prefix: RYOTENKAI_API_."""

    model_config = SettingsConfigDict(env_prefix="RYOTENKAI_API_", extra="ignore")

    host: str = "127.0.0.1"
    port: int = 8000
    runs_dir: Path = Field(default_factory=lambda: Path("runs"))
    # NoDecode tells pydantic-settings not to try JSON-decode the raw env value,
    # so our validator can turn a comma-separated string into list[str].
    cors_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["http://localhost:5173"],
    )

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_cors_origins(cls, value: object) -> object:
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value
    mlflow_ui_url: str | None = None
    log_stream_poll_interval_ms: int = 500
    max_log_chunk_bytes: int = 1_048_576  # 1 MiB
    serve_spa: bool = True  # mount web/dist if it exists
    web_dist_dir: Path = Field(default_factory=lambda: Path("web/dist"))

    @property
    def runs_dir_resolved(self) -> Path:
        return self.runs_dir.expanduser().resolve()
