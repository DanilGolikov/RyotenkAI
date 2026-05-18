"""Fail-fast MLflow connectivity check.

Probes the tracking server BEFORE any run is opened. The legacy
preflight (``MLflowAttemptManager._ensure_preflight``) opened a
disposable probe run to test connectivity which left orphan rows in
the experiment whenever credentials were wrong. The redesigned check
talks only to :meth:`ITrackingClient.ping` -- the transport
implementation decides whether to issue an HTTP HEAD, a search-runs
call, or some other side-effect-free probe.

Failure modes
-------------

* Network / DNS / TCP failures -> :class:`ProviderUnavailableError`.
* Auth failures (e.g. 401/403 surfaced by the transport) -> the
  transport-level exception is wrapped in
  :class:`ProviderUnavailableError`. The check never tries to
  *interpret* the cause; it only enforces that "could not reach
  tracking server" is the single typed surface presented to callers.

Why we do NOT open a probe run here
-----------------------------------
R-02 mitigation per ``docs/plans/vectorized-fluttering-mist.md``.
Opening a disposable run for the probe leaks an artifact on the
server and races with the orchestrator's run-counter. The narrow
ping pattern is enough for fail-fast; the first real
:meth:`ITrackingClient.start_run` call will surface late failures.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ryotenkai_shared.errors import ProviderUnavailableError
from ryotenkai_shared.utils.logger import get_logger

if TYPE_CHECKING:
    from ryotenkai_shared.infrastructure.mlflow.protocols import ITrackingClient


logger = get_logger(__name__)


__all__ = ["PreflightConnectivityCheck"]


class PreflightConnectivityCheck:
    """Fail-fast probe against an :class:`ITrackingClient`.

    The check is stateless apart from its injected dependencies; one
    instance per pipeline attempt is fine.

    :param client: Concrete :class:`ITrackingClient` (typically a
        :class:`MlflowTransport`).
    :param timeout_s: Per-call timeout passed to
        :meth:`ITrackingClient.ping`. The default mirrors the legacy
        ``MLflowManager.ping_timeout_s = 5.0``.
    """

    def __init__(
        self,
        client: ITrackingClient,
        *,
        timeout_s: float = 5.0,
    ) -> None:
        if timeout_s <= 0:
            msg = f"timeout_s must be positive, got {timeout_s!r}"
            raise ValueError(msg)
        self._client = client
        self._timeout_s = float(timeout_s)

    def run(self) -> None:
        """Probe the tracking server.

        :raises ProviderUnavailableError: If the underlying ping fails
            for any reason. The original cause is chained via
            ``__cause__`` so callers can introspect via
            :pep:`3134` traceback chaining.
        """
        try:
            self._client.ping(self._timeout_s)
        except Exception as exc:  # noqa: BLE001 -- single typed surface
            cause_class = type(exc).__name__
            transport_uri = getattr(self._client, "tracking_uri", "<unknown>")
            logger.warning(
                "[MLFLOW_PREFLIGHT] ping failed: timeout_s=%.3f cause=%s "
                "transport_uri=%s",
                self._timeout_s,
                cause_class,
                transport_uri,
            )
            raise ProviderUnavailableError(
                detail=(
                    "MLflow tracking server is unreachable "
                    f"({cause_class}: {exc})"
                ),
                context={
                    "transport_uri": str(transport_uri),
                    "timeout_s": self._timeout_s,
                    "cause_class": cause_class,
                },
                cause=exc,
            ) from exc
        logger.debug(
            "[MLFLOW_PREFLIGHT] OK (timeout_s=%.3f)",
            self._timeout_s,
        )
