"""Runtime URI resolver — single source of truth for "which URI for which role".

Replaces the loose ``uri_resolver.resolve_mlflow_uris`` + manual
re-stamping at 3 sites (``MLflowGateway.load_prompt``,
``MLflowEnvironment.activate``, ``training_launcher._build_job_env``).

Policy preserved from the audit:

* ``control_plane`` role → ``local_tracking_uri`` (loopback / docker bridge),
  falling back to ``tracking_uri`` if local is unset. Env var
  ``MLFLOW_TRACKING_URI`` is **ignored** on this role.
* ``training`` role → ``MLFLOW_TRACKING_URI`` env override first;
  then ``tracking_uri`` (funnel), falling back to ``local_tracking_uri``.

The output :class:`RuntimeUri` is a frozen value object. Use it once
to construct :class:`~.transport.MlflowTransport`; do NOT keep calling
the resolver to re-stamp ``mlflow.set_tracking_uri`` mid-process.

Per ``docs/plans/vectorized-fluttering-mist.md`` §URI resolution.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

from ryotenkai_shared.infrastructure.mlflow.config import MLflowConnectionConfig

RuntimeRole = Literal["control_plane", "training"]
"""Where the resolver is being called from.

``control_plane`` resolves to the local-loop URI; ``training``
allows env override + falls back to the funnel-exposed URI.
"""


@dataclass(frozen=True, slots=True)
class RuntimeUri:
    """Resolved URI for one role on one host. Immutable."""

    uri: str
    role: RuntimeRole

    def __post_init__(self) -> None:
        if not self.uri or not self.uri.strip():
            msg = f"RuntimeUri.uri must be non-empty (role={self.role!r})"
            raise ValueError(msg)


class RuntimeUriResolver:
    """Single source of truth for MLflow URI selection.

    Stateless. All callers (control composition root, training_launcher,
    pod-trainer wiring, SystemPromptLoader) go through these two
    static methods rather than reading config fields directly.
    """

    @staticmethod
    def for_control_plane(cfg: MLflowConnectionConfig) -> RuntimeUri:
        """Resolve the URI for control-plane (Mac orchestrator).

        Priority: ``local_tracking_uri`` → ``tracking_uri``.
        Env vars are **not** consulted on this role.
        """
        uri = (cfg.local_tracking_uri or cfg.tracking_uri or "").strip()
        if not uri:
            msg = (
                "MLflowConnectionConfig has neither tracking_uri nor "
                "local_tracking_uri set. Validator should have rejected "
                "this earlier — check construction path."
            )
            raise ValueError(msg)
        return RuntimeUri(uri=uri, role="control_plane")

    @staticmethod
    def for_training(
        cfg: MLflowConnectionConfig,
        env_override: str | None = None,
    ) -> RuntimeUri:
        """Resolve the URI for the pod-trainer subprocess.

        Priority: explicit ``env_override`` (passed by caller) →
        ``MLFLOW_TRACKING_URI`` env var → ``tracking_uri`` (funnel) →
        ``local_tracking_uri`` (fallback for same-host dev).
        """
        candidates = [
            (env_override or "").strip(),
            (os.getenv("MLFLOW_TRACKING_URI") or "").strip(),
            (cfg.tracking_uri or "").strip(),
            (cfg.local_tracking_uri or "").strip(),
        ]
        for candidate in candidates:
            if candidate:
                return RuntimeUri(uri=candidate, role="training")
        msg = (
            "Cannot resolve training URI: env override, MLFLOW_TRACKING_URI, "
            "tracking_uri, and local_tracking_uri are all empty."
        )
        raise ValueError(msg)


__all__ = [
    "RuntimeRole",
    "RuntimeUri",
    "RuntimeUriResolver",
]
