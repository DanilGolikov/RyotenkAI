"""
Evaluation plugin secrets infrastructure.

Provides:
- @requires_secrets(*keys) decorator — declares which EVAL_* secrets a plugin needs.
- SecretsResolver               — resolves declared keys from Secrets.model_extra.

Design:
    Plugin secrets are stored in secrets.env under the EVAL_* namespace:
        EVAL_CEREBRAS_API_KEY=csk-...

    The EVAL_* prefix enforces a hard boundary: plugins can only request secrets
    from their own namespace, never system secrets (HF_TOKEN, RUNPOD_API_KEY, etc.)
    which live in typed Secrets fields.

    SecretsResolver reads from Secrets.model_extra (populated by extra="allow" in
    Secrets pydantic-settings). It does NOT fall back to os.environ — secrets.env
    is the single source of truth.

Usage:
    # In secrets.env:
    #   EVAL_CEREBRAS_API_KEY=csk-...

    # In plugin code:
    @requires_secrets("EVAL_CEREBRAS_API_KEY")
    class MyCerebrasPlugin(EvaluatorPlugin):
        def evaluate(self, samples):
            api_key = self._secrets["EVAL_CEREBRAS_API_KEY"]
            ...
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.utils.plugin_secrets import PluginSecretsResolver, requires_secrets

if TYPE_CHECKING:
    from src.config.secrets.model import Secrets

EVAL_SECRET_PREFIX = "EVAL_"


class SecretsResolver(PluginSecretsResolver):
    """
    Resolves EVAL_* plugin secrets from Secrets.model_extra.

    Thin wrapper over PluginSecretsResolver with prefix="EVAL_".
    """

    def __init__(self, secrets: Secrets) -> None:
        super().__init__(secrets, prefix=EVAL_SECRET_PREFIX)


__all__ = [
    "SecretsResolver",
    "requires_secrets",
]
