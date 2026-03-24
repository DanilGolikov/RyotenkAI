from __future__ import annotations

from pydantic import Field, model_validator

from ..base import StrictBaseModel
from .constants import (
    SSH_MAX_RETRIES_DEFAULT,
    SSH_MAX_RETRIES_MAX,
    SSH_MAX_RETRIES_MIN,
    SSH_PORT_DEFAULT,
    SSH_PORT_MAX,
    SSH_PORT_MIN,
    SSH_RETRY_DELAY_DEFAULT,
    SSH_RETRY_DELAY_MAX,
    SSH_RETRY_DELAY_MIN,
    SSH_TIMEOUT_DEFAULT,
    SSH_TIMEOUT_MAX,
    SSH_TIMEOUT_MIN,
)


class SSHConnectSettings(StrictBaseModel):
    """SSH connect settings (unified)."""

    max_retries: int = Field(
        SSH_MAX_RETRIES_DEFAULT,
        ge=SSH_MAX_RETRIES_MIN,
        le=SSH_MAX_RETRIES_MAX,
    )
    retry_delay_seconds: int = Field(
        SSH_RETRY_DELAY_DEFAULT,
        ge=SSH_RETRY_DELAY_MIN,
        le=SSH_RETRY_DELAY_MAX,
    )
    timeout_seconds: int = Field(
        SSH_TIMEOUT_DEFAULT,
        ge=SSH_TIMEOUT_MIN,
        le=SSH_TIMEOUT_MAX,
    )


class SSHConfig(StrictBaseModel):
    """
    Unified SSH config (used by providers).

    Connection modes:
    - alias mode: `alias` is set (preferred)
    - explicit mode: `host` + `user` required

    If both are provided, runtime should try alias first, then explicit fallback.
    """

    alias: str | None = None
    host: str | None = None
    port: int = Field(
        SSH_PORT_DEFAULT,
        ge=SSH_PORT_MIN,
        le=SSH_PORT_MAX,
    )
    user: str | None = None
    key_path: str | None = None
    key_env: str | None = None
    connect_settings: SSHConnectSettings = Field(default_factory=SSHConnectSettings)  # pyright: ignore[reportArgumentType]

    @model_validator(mode="after")
    def _run_model_validators(self) -> SSHConfig:
        """
        Centralized cross-field validators for this config.

        Convention:
        - Keep ONE `@model_validator(mode="after")` method per config model.
        - Delegate validation logic to `src.config.validators.*`.
        """
        # Local import to avoid circular imports.
        from ..validators.providers import validate_ssh

        validate_ssh(self)
        return self


__all__ = [
    "SSHConfig",
    "SSHConnectSettings",
]
