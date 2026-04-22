from __future__ import annotations

from typing import Any

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Secrets(BaseSettings):
    """Secrets loaded from environment variables / secrets.env.

    Two layers:
    - Typed system fields (hf_token, runpod_api_key): validated at startup,
      accessible via attribute access.
    - Arbitrary plugin secrets (EVAL_*, DTST_* keys): stored in model_extra after extra="allow".
      Accessible via SecretsResolver — plugins declare what they need in their
      community ``manifest.toml`` under ``[secrets] required = [...]``.
    """

    # Optional: required only when active provider is type=runpod
    runpod_api_key: str | None = Field(None, alias="RUNPOD_API_KEY")
    hf_token: str = Field(..., alias="HF_TOKEN")

    @field_validator("hf_token", mode="before")
    @classmethod
    def _normalize_hf_token(cls, v: Any) -> str:
        """
        Normalize HF_TOKEN coming from env/.env file.

        Common real-world failure mode: trailing whitespace/newlines or accidental empty value.
        Both lead to confusing 401 errors ("Invalid username or password") from the Hub.
        """
        if v is None:
            raise ValueError("HF_TOKEN is required")
        s = str(v).strip()
        if not s:
            raise ValueError("HF_TOKEN is empty")
        return s

    @field_validator("runpod_api_key", mode="before")
    @classmethod
    def _normalize_optional_secrets(cls, v: Any) -> str | None:
        if v is None:
            return None
        s = str(v).strip()
        return s or None

    @model_validator(mode="after")
    def _normalize_extra(self) -> Secrets:
        """Trim and remove empty strings from arbitrary plugin secrets in model_extra."""
        if not self.model_extra:
            return self
        cleaned: dict[str, Any] = {}
        for k, v in self.model_extra.items():
            if isinstance(v, str):
                v = v.strip()
                if v:
                    cleaned[k] = v
            else:
                cleaned[k] = v
        object.__setattr__(self, "__pydantic_extra__", cleaned)
        return self

    model_config = SettingsConfigDict(
        env_file="secrets.env",
        env_file_encoding="utf-8",
        # extra="allow" lets arbitrary EVAL_* keys from secrets.env flow into model_extra.
        # Only keys from env_file are captured — system env vars (PATH, HOME, etc.) are NOT included.
        extra="allow",
    )


__all__ = [
    "Secrets",
]
