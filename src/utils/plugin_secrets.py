"""
Generic plugin secrets infrastructure.

Provides:
- @requires_secrets(*keys) decorator — declares which secrets a plugin needs.
- PluginSecretsResolver              — resolves declared keys from Secrets.model_extra
                                       within a required namespace prefix.

Both evaluation (EVAL_*) and dataset validation (DTST_*) plugin systems
import from this module and bind their own namespace prefix.

Design:
    Plugin secrets are stored in secrets.env under a namespace prefix
    (e.g. EVAL_CEREBRAS_API_KEY, DTST_SCHEMA_VALIDATOR_TOKEN).

    The prefix enforces a hard boundary: plugins can only request secrets
    from their own namespace, never system secrets (HF_TOKEN, RUNPOD_API_KEY, etc.)
    which live in typed Secrets fields.

    PluginSecretsResolver reads from Secrets.model_extra (populated by extra="allow"
    in Secrets pydantic-settings). It does NOT fall back to os.environ — secrets.env
    is the single source of truth.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.config.secrets.model import Secrets


def requires_secrets(*keys: str):
    """
    Class decorator: declare which secrets a plugin needs.

    Stores key names as cls._required_secrets (tuple of str).
    The runner (EvaluationRunner / DatasetValidator) reads this attribute
    and injects resolved values as plugin._secrets (dict[str, str])
    before calling evaluate() / validate().

    Prefix validation is NOT enforced at decoration time — only at resolve time
    by PluginSecretsResolver. This keeps the decorator lightweight and reusable.

    Usage:
        @requires_secrets("EVAL_CEREBRAS_API_KEY")
        class MyPlugin(EvaluatorPlugin): ...

        @requires_secrets("DTST_SCHEMA_VALIDATOR_TOKEN")
        class MyValidationPlugin(ValidationPlugin): ...
    """

    def decorator(cls: type) -> type:
        cls._required_secrets = keys  # type: ignore[attr-defined]
        return cls

    return decorator


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
    "requires_secrets",
]
