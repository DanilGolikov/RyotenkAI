"""
Dataset validation plugin secrets infrastructure.

Provides:
- @requires_secrets(*keys) decorator — declares which DTST_* secrets a plugin needs.
- SecretsResolver               — resolves declared keys from Secrets.model_extra.

Design:
    Plugin secrets are stored in secrets.env under the DTST_* namespace:
        DTST_SCHEMA_VALIDATOR_TOKEN=tok-...

    The DTST_* prefix enforces a hard boundary: validation plugins can only
    request secrets from their own namespace, never system secrets
    (HF_TOKEN, RUNPOD_API_KEY, etc.) which live in typed Secrets fields.

    SecretsResolver reads from Secrets.model_extra (populated by extra="allow" in
    Secrets pydantic-settings). It does NOT fall back to os.environ — secrets.env
    is the single source of truth.

Usage:
    # In secrets.env:
    #   DTST_SCHEMA_VALIDATOR_TOKEN=tok-...

    # In plugin code:
    @requires_secrets("DTST_SCHEMA_VALIDATOR_TOKEN")
    class MySchemaValidator(ValidationPlugin):
        def validate(self, dataset):
            token = self._secrets["DTST_SCHEMA_VALIDATOR_TOKEN"]
            ...
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.utils.plugin_secrets import PluginSecretsResolver, requires_secrets

if TYPE_CHECKING:
    from src.config.secrets.model import Secrets

DTST_SECRET_PREFIX = "DTST_"


class SecretsResolver(PluginSecretsResolver):
    """
    Resolves DTST_* validation plugin secrets from Secrets.model_extra.

    Thin wrapper over PluginSecretsResolver with prefix="DTST_".
    """

    def __init__(self, secrets: Secrets) -> None:
        super().__init__(secrets, prefix=DTST_SECRET_PREFIX)


__all__ = [
    "SecretsResolver",
    "requires_secrets",
]
