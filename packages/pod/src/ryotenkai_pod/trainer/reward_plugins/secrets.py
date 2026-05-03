"""Reward plugin secrets resolver (RWRD_* namespace).

Mirrors the validation (``DTST_*``) and evaluation (``EVAL_*``)
resolvers — same :class:`PluginSecretsResolver` contract, different
prefix. The community loader stamps ``_required_secrets`` onto each
plugin class from the manifest's ``[[required_env]]`` block (entries
with ``secret=true, optional=false``); the registry then asks this
resolver for the values at instantiate time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.utils.plugin_secrets import PluginSecretsResolver

if TYPE_CHECKING:
    from src.config.secrets.model import Secrets

RWRD_SECRET_PREFIX = "RWRD_"


class SecretsResolver(PluginSecretsResolver):
    """Resolves RWRD_* reward plugin secrets from ``Secrets.model_extra``."""

    def __init__(self, secrets: Secrets) -> None:
        super().__init__(secrets, prefix=RWRD_SECRET_PREFIX)


__all__ = ["RWRD_SECRET_PREFIX", "SecretsResolver"]
