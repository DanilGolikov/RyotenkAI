"""Report plugin secrets resolver (RPRT_* namespace).

Mirrors the validation / evaluation / reward resolvers. Currently no
shipped report plugin requires secrets, but the resolver is wired up
through :func:`build_report_plugins` so authors who add API-backed
report blocks (Slack notifier, Linear issue exporter, etc.) get the
same auto-injection contract as the other kinds.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.utils.plugin_secrets import PluginSecretsResolver

if TYPE_CHECKING:
    from src.config.secrets.model import Secrets

RPRT_SECRET_PREFIX = "RPRT_"


class SecretsResolver(PluginSecretsResolver):
    """Resolves RPRT_* report plugin secrets from ``Secrets.model_extra``."""

    def __init__(self, secrets: Secrets) -> None:
        super().__init__(secrets, prefix=RPRT_SECRET_PREFIX)


__all__ = ["RPRT_SECRET_PREFIX", "SecretsResolver"]
