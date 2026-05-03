"""Dataset validation plugin secrets resolver (DTST_* namespace)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.utils.plugin_secrets import PluginSecretsResolver

if TYPE_CHECKING:
    from src.config.secrets.model import Secrets

DTST_SECRET_PREFIX = "DTST_"


class SecretsResolver(PluginSecretsResolver):
    """Resolves DTST_* validation plugin secrets from ``Secrets.model_extra``."""

    def __init__(self, secrets: Secrets) -> None:
        super().__init__(secrets, prefix=DTST_SECRET_PREFIX)


__all__ = ["DTST_SECRET_PREFIX", "SecretsResolver"]
