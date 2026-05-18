"""Central MLflow HTTP transport with tenacity retry.

Replaces the wide ``IMLflowManager`` (20+ methods, ``Any``-typed) and the
parallel ``MLflowGateway`` (independent timeout + retry budget). Acts as
the SINGLE call-site for ``mlflow.set_tracking_uri`` in the codebase:

* exactly one stamping at :meth:`__init__` (per :class:`RuntimeUri`
  value object);
* never re-stamped — read-paths and other producers attach to runs via
  env vars / explicit run-id arguments.

Retry policy
------------
Every server-touching method is wrapped by a tenacity ``Retrying``
controller with::

    stop = stop_after_delay(retry_total_budget_s)
    wait = wait_exponential(multiplier=0.5, max=8)
    retry = retry_if_exception_type((ConnectionError, TimeoutError))

— transient transport faults only. 4xx (auth, bad request) propagate
immediately; the caller decides whether to bubble or wrap.

Per ``docs/plans/vectorized-fluttering-mist.md`` §Target architecture.
"""

from __future__ import annotations

import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from ryotenkai_shared.errors import ProviderUnavailableError
from ryotenkai_shared.infrastructure.mlflow.auth import (
    MlflowAuthAdapter,
)
from ryotenkai_shared.infrastructure.mlflow.protocols import RunStatus
from ryotenkai_shared.infrastructure.mlflow.run_handle import RunHandle
from ryotenkai_shared.infrastructure.mlflow.taxonomy import ReservedPrefixGuard
from ryotenkai_shared.utils.logger import get_logger

if TYPE_CHECKING:
    from ryotenkai_shared.infrastructure.mlflow.uri import RuntimeUri

logger = get_logger(__name__)

_T = TypeVar("_T")


_DEFAULT_PING_PATH = "api/2.0/mlflow/experiments/list?max_results=1"
"""Endpoint used by :meth:`MlflowTransport.ping`. Chosen because it
exists on every MLflow >=2.x server, requires auth (so we exercise the
auth header path), and is cheap (returns at most 1 row)."""

_MLFLOW_PARENT_TAG = "mlflow.parentRunId"
"""MLflow's reserved tag for nested-run parentage. Whitelisted in
:class:`~.taxonomy.ReservedPrefixGuard`."""


class MlflowTransport:
    """Concrete :class:`~.protocols.ITrackingClient` implementation.

    Holds a single :class:`mlflow.tracking.MlflowClient` constructed
    with the resolved :class:`RuntimeUri`. The ``mlflow.set_tracking_uri``
    call is performed once at construction time so that the fluent API
    (used by HF Trainer's MLflowCallback in the pod-trainer subprocess)
    also targets the correct server without a re-stamping race.

    :param runtime_uri: Resolved URI value object (control or training role).
    :param auth: Discriminated-union auth config; pass
        ``_AuthNone()`` for loopback / dev.
    :param connect_timeout_s: Per-attempt TCP connect timeout.
    :param request_timeout_s: Per-attempt request-completion timeout.
    :param retry_total_budget_s: Total tenacity stop-after-delay budget
        for any single transport-level operation. The exponential
        backoff caps at 8 s between attempts.
    :param ca_bundle_path: Optional CA-bundle path passed to the
        underlying HTTP layer via ``REQUESTS_CA_BUNDLE``-equivalent
        per-call ``verify`` argument. We do NOT mutate
        ``os.environ`` — see the audit (control-plane vs pod-trainer
        env collision).

    All methods raise :class:`ProviderUnavailableError` on terminal
    transport failure (retry budget exhausted, DNS error,
    connection-refused).
    """

    def __init__(
        self,
        runtime_uri: RuntimeUri,
        auth: object,
        *,
        connect_timeout_s: float = 5.0,
        request_timeout_s: float = 30.0,
        retry_total_budget_s: float = 30.0,
        ca_bundle_path: str | None = None,
    ) -> None:
        if connect_timeout_s <= 0 or request_timeout_s <= 0:
            msg = (
                "MlflowTransport timeouts must be positive: "
                f"connect={connect_timeout_s!r}, request={request_timeout_s!r}"
            )
            raise ValueError(msg)
        self._runtime_uri = runtime_uri
        self._auth_adapter = MlflowAuthAdapter(auth)
        self._connect_timeout_s = float(connect_timeout_s)
        self._request_timeout_s = float(request_timeout_s)
        self._retry_total_budget_s = float(retry_total_budget_s)
        self._ca_bundle_path = ca_bundle_path
        # Lazily-constructed; first server-touching call materialises.
        self._client: Any | None = None
        # One-shot URI stamping. We must do this BEFORE constructing the
        # MlflowClient because some entrypoints (e.g. the HF Trainer
        # MLflowCallback in the pod-trainer subprocess) read the URI
        # from the fluent module surface, not from a constructed client.
        self._stamp_tracking_uri()

    # -- properties -------------------------------------------------

    @property
    def tracking_uri(self) -> str:
        """The URI this transport was configured against (frozen)."""
        return self._runtime_uri.uri

    @property
    def client(self) -> Any:
        """The underlying ``mlflow.tracking.MlflowClient`` (lazy).

        Exposed for the small set of operations that need direct
        access to MLflow's wide API surface
        (:class:`~.metric_sink.MetricSink`,
        :class:`~.journal_uploader.JournalUploader`). New consumers
        should add a typed method here instead of reaching through.
        """
        if self._client is None:
            mlflow_client_cls = self._load_mlflow_client_class()
            self._client = mlflow_client_cls(tracking_uri=self._runtime_uri.uri)
        return self._client

    # -- ITrackingClient methods ------------------------------------

    def ping(self, timeout_s: float) -> None:
        """Verify reachability + auth via a cheap GET.

        Implementation uses ``urllib.request`` (stdlib) so we don't
        depend on ``requests`` being importable in trimmed CI venvs.
        The request carries the ``Authorization`` header from the
        configured :data:`MLflowAuthConfig` (if any).

        :param timeout_s: per-attempt socket timeout (caller decides
            the budget; ``MlflowTransport`` does NOT retry the ping).
        :raises ProviderUnavailableError: on any transport-level
            failure (DNS, refused, timeout, TLS, 5xx).
        """
        url = self._runtime_uri.uri.rstrip("/") + "/" + _DEFAULT_PING_PATH
        request = urllib.request.Request(url, method="GET")
        auth_header = self._auth_adapter.authorization_header()
        if auth_header is not None:
            request.add_header("Authorization", auth_header)
        try:
            with urllib.request.urlopen(  # noqa: S310 — URI vetted by config
                request,
                timeout=timeout_s,
                **self._urlopen_kwargs(),
            ) as response:
                # MLflow returns 200 on success; we don't parse the body.
                status = getattr(response, "status", 200)
                if status >= 500:
                    raise ProviderUnavailableError(
                        f"MLflow ping returned {status}",
                        context={"uri": self._runtime_uri.uri},
                    )
        except urllib.error.HTTPError as exc:
            # 4xx is auth/config error — surface as ProviderUnavailable
            # so the orchestrator can route to the user-facing failure.
            raise ProviderUnavailableError(
                f"MLflow ping HTTP {exc.code}: {exc.reason}",
                context={"uri": self._runtime_uri.uri, "status": exc.code},
                cause=exc,
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ProviderUnavailableError(
                f"MLflow ping unreachable: {exc}",
                context={"uri": self._runtime_uri.uri},
                cause=exc,
            ) from exc

    def start_run(
        self,
        experiment: str,
        name: str,
        tags: Mapping[str, str],
        params: Mapping[str, str],
    ) -> RunHandle:
        """Open a new top-level run on the given experiment.

        All ``tags`` and ``params`` keys are validated through
        :meth:`~.taxonomy.ReservedPrefixGuard.assert_safe` BEFORE any
        server call — the guard is cheap, and a typo would otherwise
        be discovered as a 4xx in the middle of the lifecycle.

        :returns: Frozen :class:`RunHandle` with ``parent_run_id=None``
            and ``status=RunStatus.RUNNING``.
        """
        for key in (*tags.keys(), *params.keys()):
            ReservedPrefixGuard.assert_safe(key)

        def _do() -> RunHandle:
            client = self.client
            experiment_id = self._ensure_experiment(experiment)
            run = client.create_run(
                experiment_id=experiment_id,
                tags={**dict(tags), "mlflow.runName": name},
            )
            for pkey, pvalue in params.items():
                client.log_param(run.info.run_id, pkey, pvalue)
            return RunHandle(
                run_id=run.info.run_id,
                experiment_id=experiment_id,
                parent_run_id=None,
                tracking_uri=self._runtime_uri.uri,
                status=RunStatus.RUNNING,
            )

        return self._with_retry("start_run", _do)

    def start_nested_run(
        self,
        parent_run_id: str,
        name: str,
        tags: Mapping[str, str],
    ) -> RunHandle:
        """Open a nested child of ``parent_run_id``.

        The MLflow convention for nested runs is the reserved
        ``mlflow.parentRunId`` tag on the child; we set it explicitly
        here (the key is on the
        :class:`~.taxonomy.ReservedPrefixGuard` whitelist) so the
        relationship is visible in the UI without depending on the
        fluent ``mlflow.start_run(nested=True)`` context manager.
        """
        for key in tags.keys():
            ReservedPrefixGuard.assert_safe(key)

        def _do() -> RunHandle:
            client = self.client
            parent_run = client.get_run(parent_run_id)
            experiment_id = parent_run.info.experiment_id
            merged_tags = {
                **dict(tags),
                _MLFLOW_PARENT_TAG: parent_run_id,
                "mlflow.runName": name,
            }
            run = client.create_run(experiment_id=experiment_id, tags=merged_tags)
            return RunHandle(
                run_id=run.info.run_id,
                experiment_id=experiment_id,
                parent_run_id=parent_run_id,
                tracking_uri=self._runtime_uri.uri,
                status=RunStatus.RUNNING,
            )

        return self._with_retry("start_nested_run", _do)

    def adopt_run(self, run_id: str) -> RunHandle:
        """Re-open an existing run by id (resume / takeover).

        Reads the parent-run-id from the run's tags (``mlflow.parentRunId``)
        so callers downstream can reason about lineage.
        """

        def _do() -> RunHandle:
            client = self.client
            run = client.get_run(run_id)
            parent = run.data.tags.get(_MLFLOW_PARENT_TAG)
            status_raw = getattr(run.info, "status", "RUNNING") or "RUNNING"
            try:
                status = RunStatus(status_raw)
            except ValueError:
                # MLflow may surface non-enum statuses for active runs;
                # default to RUNNING is the conservative choice.
                status = RunStatus.RUNNING
            return RunHandle(
                run_id=run.info.run_id,
                experiment_id=run.info.experiment_id,
                parent_run_id=parent,
                tracking_uri=self._runtime_uri.uri,
                status=status,
            )

        return self._with_retry("adopt_run", _do)

    def set_terminated(self, run_id: str, status: RunStatus) -> None:
        """Mark a run terminated with the given :class:`RunStatus`.

        Idempotent on the MLflow side — sending the same status twice
        is a no-op.
        """

        def _do() -> None:
            self.client.set_terminated(run_id, status.value)

        self._with_retry("set_terminated", _do)

    def set_tags(self, run_id: str, tags: Mapping[str, str]) -> None:
        """Batch-set tags on an existing run.

        Every key is guard-checked before the wire call. We iterate
        rather than calling MLflow's batched ``log_batch`` because
        the batched API requires :class:`mlflow.entities.RunTag` proto
        objects and the cost saving is irrelevant for the typical
        ``<10`` lifecycle tags.
        """
        for key in tags.keys():
            ReservedPrefixGuard.assert_safe(key)

        def _do() -> None:
            client = self.client
            for key, value in tags.items():
                client.set_tag(run_id, key, value)

        self._with_retry("set_tags", _do)

    # -- internal helpers -------------------------------------------

    def _stamp_tracking_uri(self) -> None:
        """One-shot ``mlflow.set_tracking_uri`` at construction time."""
        mlflow = self._load_mlflow_module()
        mlflow.set_tracking_uri(self._runtime_uri.uri)
        logger.info(
            "MlflowTransport stamped tracking_uri=%s role=%s",
            self._runtime_uri.uri,
            self._runtime_uri.role,
        )

    def _ensure_experiment(self, experiment: str) -> str:
        """Resolve experiment name → id; create on miss."""
        client = self.client
        existing = client.get_experiment_by_name(experiment)
        if existing is not None:
            return existing.experiment_id
        return client.create_experiment(experiment)

    def _with_retry(self, op_name: str, fn: Callable[[], _T]) -> _T:
        """Wrap ``fn`` in tenacity retry on transient transport errors.

        Lazy-imports tenacity so this module remains importable in
        trimmed CI venvs (the production wheels carry it; smoke tests
        that don't touch the network skip this code path entirely).
        """
        from tenacity import (
            RetryError,
            Retrying,
            retry_if_exception_type,
            stop_after_delay,
            wait_exponential,
        )

        retryer = Retrying(
            stop=stop_after_delay(self._retry_total_budget_s),
            wait=wait_exponential(multiplier=0.5, max=8.0),
            retry=retry_if_exception_type((ConnectionError, TimeoutError)),
            reraise=True,
        )
        try:
            for attempt in retryer:
                with attempt:
                    return fn()
        except RetryError as exc:
            raise ProviderUnavailableError(
                f"MLflow {op_name} exhausted retry budget "
                f"({self._retry_total_budget_s}s)",
                context={"op": op_name, "uri": self._runtime_uri.uri},
                cause=exc,
            ) from exc
        except (ConnectionError, TimeoutError) as exc:
            raise ProviderUnavailableError(
                f"MLflow {op_name} transport failure: {exc}",
                context={"op": op_name, "uri": self._runtime_uri.uri},
                cause=exc,
            ) from exc
        # Unreachable — Retrying always either returns or raises.
        msg = "MlflowTransport._with_retry exited without return"
        raise RuntimeError(msg)

    def _urlopen_kwargs(self) -> dict[str, Any]:
        """Build kwargs for ``urllib.request.urlopen`` honouring CA bundle.

        ``ssl.create_default_context(cafile=...)`` is the stdlib way to
        verify against an explicit CA bundle without mutating
        ``os.environ`` (the design-doc invariant — see audit R-07).
        """
        if self._ca_bundle_path is None:
            return {}
        import ssl

        ctx = ssl.create_default_context(cafile=self._ca_bundle_path)
        return {"context": ctx}

    @staticmethod
    def _load_mlflow_module() -> Any:
        """Lazy import of ``mlflow`` (kept out of module-level import
        to avoid Network/heavy side-effects in CI tests)."""
        import mlflow  # noqa: PLC0415 — intentional lazy import

        return mlflow

    @staticmethod
    def _load_mlflow_client_class() -> Any:
        """Lazy import of the ``MlflowClient`` class."""
        from mlflow.tracking import MlflowClient  # noqa: PLC0415

        return MlflowClient


__all__ = ["MlflowTransport"]
