from __future__ import annotations

from typing import Any

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Secrets(BaseSettings):
    """Secrets loaded from environment variables / secrets.env.

    Three layers (PR4 contract):

    - **Per-resource tokens** (encrypted) live in a provider/integration's
      workspace under ``token.enc``. They are the preferred source: set
      once in the Web UI, shared across projects, never hit disk in
      plaintext. Access them via ``get_hf_token`` /
      ``get_provider_token`` below.
    - **Typed env fallbacks** (``HF_TOKEN`` / ``RUNPOD_API_KEY``): kept
      for back-compat during the transition. When the workspace token is
      absent, we fall back to the env attribute and log a deprecation
      warning so users migrate in their own time.
    - **Arbitrary plugin secrets** (``EVAL_*``, ``DTST_*`` keys): stored
      in ``model_extra`` after ``extra="allow"``. Resolved by plugin
      code via ``SecretsResolver``; plugins declare what they need in
      their ``community/**/manifest.toml`` under
      ``[secrets] required = [...]``.

    Historical note: ``hf_token`` used to be *required* at top level
    (construction raised when missing). That contract changes with PR4 —
    a project that drives HF exclusively through a Settings integration
    no longer needs ``HF_TOKEN`` in ``secrets.env``.
    """

    runpod_api_key: str | None = Field(None, alias="RUNPOD_API_KEY")
    # Historically required. Now optional — resolved per-integration at
    # the point of use, with the env value kept as a fallback.
    hf_token: str | None = Field(None, alias="HF_TOKEN")

    @field_validator("hf_token", mode="before")
    @classmethod
    def _normalize_hf_token(cls, v: Any) -> str | None:
        """Normalize ``HF_TOKEN`` from env/.env file.

        Real-world failure mode: trailing whitespace/newlines. Both lead
        to confusing 401 errors from the Hub. We trim and drop empty
        strings; ``None`` / absent is allowed.
        """
        if v is None:
            return None
        s = str(v).strip()
        return s or None

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

    # ------------------------------------------------------------------
    # Per-resource token resolution (PR4)
    # ------------------------------------------------------------------

    def get_hf_token(self, integration_id: str | None = None) -> str | None:
        """Resolve the HF token for the given integration.

        Lookup order:

        1. ``token.enc`` in the integration's workspace
           (``~/.ryotenkai/integrations/<id>/token.enc``), decrypted on
           the fly.
        2. ``self.hf_token`` (populated from ``HF_TOKEN`` env/dotenv).
        3. ``None`` when neither is available — callers decide whether
           that's fatal (HF Hub upload) or fine (anonymous read).
        """
        return _resolve_token("integrations", integration_id, fallback=self.hf_token)

    def get_provider_token(self, provider_id: str | None = None) -> str | None:
        """Resolve the token stored under a provider workspace.

        Used for credentials like ``RUNPOD_API_KEY`` that are now set
        per-provider in ``~/.ryotenkai/providers/<id>/token.enc``. Falls
        back to ``self.runpod_api_key`` for back-compat during the
        migration window.
        """
        return _resolve_token("providers", provider_id, fallback=self.runpod_api_key)

    model_config = SettingsConfigDict(
        env_file="secrets.env",
        env_file_encoding="utf-8",
        extra="allow",
    )


def _resolve_token(
    workspace_kind: str, resource_id: str | None, *, fallback: str | None
) -> str | None:
    """Look up a ``token.enc`` for the given resource, else return fallback.

    Split from ``Secrets`` into a free function so it stays easy to stub
    in tests and doesn't pull Pydantic cycles.
    """
    if not resource_id:
        return _fallback_with_warning(workspace_kind, resource_id, fallback)

    from pathlib import Path

    # Local imports keep Secrets construction free of crypto cost until a
    # caller actually asks for a token.
    from src.api.services.token_crypto import TokenCrypto, TokenCryptoError, read_token_file

    path = Path.home() / ".ryotenkai" / workspace_kind / resource_id / "token.enc"
    if not path.is_file():
        return _fallback_with_warning(workspace_kind, resource_id, fallback)

    try:
        token = read_token_file(path, TokenCrypto())
    except TokenCryptoError:
        # Master-key rotated since this token was written — fall back so
        # we don't brick pipelines, but emit a loud warning.
        from src.utils.logger import logger

        logger.error(
            "[SECRETS] Failed to decrypt %s — master key mismatch? Falling back to env.",
            path,
        )
        return fallback
    if token is not None:
        return token
    return _fallback_with_warning(workspace_kind, resource_id, fallback)


def _fallback_with_warning(
    workspace_kind: str, resource_id: str | None, fallback: str | None
) -> str | None:
    if fallback is not None and resource_id:
        # Only warn when a workspace token was *expected* but missing —
        # pure env usage stays quiet for back-compat.
        try:
            from src.utils.logger import logger

            logger.warning(
                "[SECRETS] No token.enc for %s/%s — using env fallback. "
                "Configure it in Settings → %s to avoid this warning.",
                workspace_kind,
                resource_id,
                "Integrations" if workspace_kind == "integrations" else "Providers",
            )
        except Exception:  # noqa: BLE001 — warnings are best-effort
            pass
    return fallback


__all__ = [
    "Secrets",
]
