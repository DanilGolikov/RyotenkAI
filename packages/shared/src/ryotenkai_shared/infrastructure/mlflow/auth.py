"""MLflow authentication config (discriminated union) + header adapter.

Replaces the implicit-by-omission auth surface of the previous
``MLflowConfig`` (no fields, secrets piggybacked on URI userinfo or
process env). The new model uses a tagged ``kind`` discriminator
and references secrets *by env-var name* — never inline values.

Three kinds:

* ``"none"`` — no auth header (default; loopback, dev).
* ``"basic"`` — HTTP Basic. Username inline; password resolved from
  the env var named in ``password_env_var``.
* ``"bearer"`` — single bearer token resolved from ``token_env_var``.

The orchestrator (control-plane) is responsible for decrypting on-disk
secrets (e.g. ``workspace/integrations/store.py:36`` token.enc) and
exporting them into the named env vars before spawning the pod-trainer
subprocess. ``MlflowAuthAdapter`` resolves at request-time and is the
ONLY component that reads the secret env vars — keeping the blast
radius for accidental leakage minimal.

Per ``docs/plans/vectorized-fluttering-mist.md`` §Configuration.
"""

from __future__ import annotations

import os
from base64 import b64encode
from typing import Annotated, Literal

from pydantic import Field

from ryotenkai_shared.config.base import StrictBaseModel


class _AuthNone(StrictBaseModel):
    """No authentication. Default for loopback / dev / Tailscale mesh."""

    kind: Literal["none"] = "none"


class _AuthBasic(StrictBaseModel):
    """HTTP Basic. Secret resolved from env var, never inline."""

    kind: Literal["basic"]
    username: str = Field(..., min_length=1)
    password_env_var: str = Field(
        ...,
        min_length=1,
        description=(
            "Name of the env var holding the password. Never put the "
            "password value here directly — the orchestrator exports "
            "the env at spawn time."
        ),
    )


class _AuthBearer(StrictBaseModel):
    """Bearer token. Secret resolved from env var, never inline."""

    kind: Literal["bearer"]
    token_env_var: str = Field(
        ...,
        min_length=1,
        description="Name of the env var holding the bearer token.",
    )


MLflowAuthConfig = Annotated[
    _AuthNone | _AuthBasic | _AuthBearer,
    Field(discriminator="kind"),
]
"""Discriminated union of supported MLflow auth kinds.

Use ``Field(default_factory=_AuthNone)`` on consumer configs to make
``none`` the default. Validate via standard Pydantic — the
``discriminator="kind"`` field tells Pydantic which variant to pick.
"""


class MlflowAuthAdapter:
    """Resolves an :data:`MLflowAuthConfig` to an HTTP ``Authorization``
    header value at request-time.

    Returns ``None`` for the ``"none"`` kind. For ``"basic"`` and
    ``"bearer"`` kinds, reads the secret from the named env var.
    A missing env var is treated as a config error (``KeyError``) —
    fail-fast rather than silently sending a header with an empty
    secret.

    Stateless; safe to share across threads.
    """

    def __init__(self, cfg: object) -> None:
        # ``object`` rather than ``MLflowAuthConfig`` to avoid a runtime
        # ``isinstance`` against an Annotated union (Pydantic-discriminated
        # types are validated structurally; the adapter just dispatches
        # on the ``kind`` attribute).
        self._cfg = cfg

    def authorization_header(self) -> str | None:
        """Return the value for the ``Authorization`` HTTP header, or
        ``None`` if no auth is configured.

        Raises ``KeyError`` if the named env var is unset (caller-error
        surfaceable at request boundary; never silently degrade).
        """
        kind = getattr(self._cfg, "kind", "none")
        if kind == "none":
            return None
        if kind == "basic":
            username: str = self._cfg.username  # type: ignore[attr-defined]
            env_name: str = self._cfg.password_env_var  # type: ignore[attr-defined]
            password = self._read_secret(env_name)
            token = b64encode(f"{username}:{password}".encode()).decode("ascii")
            return f"Basic {token}"
        if kind == "bearer":
            env_name = self._cfg.token_env_var  # type: ignore[attr-defined]
            token = self._read_secret(env_name)
            return f"Bearer {token}"
        msg = f"Unknown MLflow auth kind: {kind!r}"
        raise ValueError(msg)

    @staticmethod
    def _read_secret(env_var_name: str) -> str:
        try:
            value = os.environ[env_var_name]
        except KeyError as exc:
            msg = (
                f"MLflow auth secret env var {env_var_name!r} is unset. "
                "Orchestrator must export it before constructing the transport."
            )
            raise KeyError(msg) from exc
        if not value:
            msg = f"MLflow auth secret env var {env_var_name!r} is empty."
            raise KeyError(msg)
        return value


__all__ = [
    "MLflowAuthConfig",
    "MlflowAuthAdapter",
]
