"""
Generic plugin secrets infrastructure.

Plugins declare required secrets via ``manifest.toml`` (``[secrets] required = [...]``).
The community loader attaches them to the plugin class as ``_required_secrets``;
runners resolve values through ``PluginSecretsResolver`` below.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.config.secrets.model import Secrets


class PluginSecretsResolver:
    """
    Resolves plugin secrets from Secrets.model_extra within a namespace prefix.

    Only reads keys that start with the configured prefix. Raises ValueError if
    a plugin requests a key outside the namespace. Raises RuntimeError if a
    required key is not present in secrets.env.

    Args:
        secrets: Pydantic Secrets model (with extra="allow").
        prefix:  Required namespace prefix (e.g. "EVAL_", "DTST_").
    """

    def __init__(self, secrets: Secrets, *, prefix: str) -> None:
        self._secrets = secrets
        self._prefix = prefix

    @property
    def prefix(self) -> str:
        return self._prefix

    def resolve(self, keys: tuple[str, ...]) -> dict[str, str]:
        """
        Resolve plugin secret keys from Secrets.model_extra.

        Args:
            keys: Tuple of secret key names (must all start with self._prefix).

        Returns:
            dict mapping each key to its resolved string value.

        Raises:
            ValueError:   if any key does not start with the configured prefix.
            RuntimeError: if a required key is not present in secrets.env.
        """
        for key in keys:
            if not key.startswith(self._prefix):
                raise ValueError(
                    f"Plugin requested secret '{key}' which is outside the allowed "
                    f"'{self._prefix}*' namespace. "
                    f"Plugin secrets must be prefixed with {self._prefix!r}. "
                    "System secrets (HF_TOKEN, RUNPOD_API_KEY, etc.) are not accessible to plugins."
                )

        extra: dict[str, str] = dict(self._secrets.model_extra or {})
        result: dict[str, str] = {}

        for key in keys:
            val = extra.get(key.lower())
            if not val:
                raise RuntimeError(
                    f"Plugin secret '{key}' is required but not found in secrets.env. "
                    f"Add '{key}=<value>' to secrets.env."
                )
            result[key] = val

        return result


__all__ = [
    "PluginSecretsResolver",
]
