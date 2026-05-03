"""Evaluation plugin secrets resolver (EVAL_* namespace)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.utils.plugin_secrets import PluginSecretsResolver

if TYPE_CHECKING:
    from src.config.secrets.model import Secrets

EVAL_SECRET_PREFIX = "EVAL_"


class SecretsResolver(PluginSecretsResolver):
    """Resolves EVAL_* evaluator plugin secrets from ``Secrets.model_extra``."""

    def __init__(self, secrets: Secrets) -> None:
        super().__init__(secrets, prefix=EVAL_SECRET_PREFIX)


__all__ = ["EVAL_SECRET_PREFIX", "SecretsResolver"]
