"""Concrete :class:`IRunQuery` implementation for the control plane.

Replaces the six ad-hoc ``MlflowClient()`` constructions scattered
across ``packages/control`` (``deletion.py``, ``training_monitor.py``,
``model_retriever/retriever.py``, ``report_generator.py``,
``reports/adapters/mlflow_adapter.py``, and the legacy 884-LOC
``reports/core/builder.py``) with a single DI'd surface that satisfies
:class:`~ryotenkai_shared.infrastructure.mlflow.protocols.IRunQuery`.

Phase M3.A is additive — this client lives next to the legacy
callsites; M3.B re-wires the six call-sites to depend on it.

Design notes
------------
* The underlying ``MlflowClient`` is constructed ONCE in
  :meth:`__init__` with the resolved ``tracking_uri``. There is no
  per-call ``mlflow.set_tracking_uri`` mutation: the URI is frozen
  for the lifetime of the client.
* ``mlflow`` and ``mlflow.tracking`` are imported lazily inside
  :meth:`__init__` so importing this module does NOT pay the heavy
  ``mlflow`` import cost (matters for CI smoke tests).
* :meth:`get_run` has a small per-process LRU cache (``size=256``) so
  repeated lookups for terminal runs avoid the wire round-trip. Only
  runs whose status is FINISHED / FAILED / KILLED are cached; RUNNING
  runs are never cached because they may still mutate (status,
  tags).
* All shared state is guarded by a :class:`threading.Lock`; the
  control plane is largely single-threaded today but tests exercise
  concurrent reads.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

from ryotenkai_shared.infrastructure.mlflow.protocols import RunStatus
from ryotenkai_shared.infrastructure.mlflow.run_handle import RunHandle
from ryotenkai_shared.utils.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = get_logger(__name__)

_MLFLOW_PARENT_TAG = "mlflow.parentRunId"
"""MLflow's reserved tag used to mark a run as nested under another."""

_TERMINAL_STATUSES: frozenset[RunStatus] = frozenset(
    {RunStatus.FINISHED, RunStatus.FAILED, RunStatus.KILLED}
)
"""Statuses that are guaranteed immutable once observed."""

_LIST_CHILDREN_PAGE_SIZE = 200
"""Per-page max-results used when paginating ``search_runs`` for children."""

_LIST_CHILDREN_HARD_CAP = 1000
"""Absolute upper bound on children returned by :meth:`list_children`."""

_SEARCH_PAGE_SIZE = 200
"""Per-call max-results cap for :meth:`search`."""

_CACHE_MAXSIZE = 256
"""LRU cache size for :meth:`get_run` (terminal runs only)."""


class MlflowReadClient:
    """Concrete :class:`IRunQuery` implementation for the control plane.

    :param tracking_uri: Resolved MLflow tracking URI; passed verbatim
        to ``MlflowClient(tracking_uri=...)``. Frozen for the lifetime
        of this instance — re-stamping the global URI is explicitly
        avoided (see module docstring).
    :param request_timeout_s: Reserved for future use; passed through
        for parity with :class:`MlflowTransport`. MLflow itself does
        not expose a per-call timeout knob on the read endpoints, so
        this parameter is currently informational.
    """

    def __init__(
        self,
        tracking_uri: str,
        *,
        request_timeout_s: float = 30.0,
    ) -> None:
        if not tracking_uri:
            raise ValueError("tracking_uri must be a non-empty string")
        if request_timeout_s <= 0:
            raise ValueError(
                f"request_timeout_s must be positive, got {request_timeout_s!r}"
            )
        self._tracking_uri = tracking_uri
        self._request_timeout_s = float(request_timeout_s)
        # Lazy mlflow import keeps module import cheap for CI.
        from mlflow.tracking import MlflowClient  # noqa: PLC0415

        self._client: Any = MlflowClient(tracking_uri=tracking_uri)
        self._lock = threading.Lock()
        self._cache: OrderedDict[str, RunHandle] = OrderedDict()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def tracking_uri(self) -> str:
        """The URI this client was configured against (frozen)."""
        return self._tracking_uri

    @property
    def underlying_client(self) -> Any:
        """Escape hatch: the raw ``mlflow.tracking.MlflowClient``.

        Phase M3.B exposes this so the legacy reads/writes that fall
        outside :class:`IRunQuery` (``download_artifacts``,
        ``get_metric_history``, ``log_text``, ``set_terminated``,
        ``delete_run``, etc.) can still be performed via the single
        DI'd construction. Phase M7 lints forbid constructing
        ``MlflowClient`` ad-hoc outside this module + ``transport.py``;
        consumers reach into ``underlying_client`` for the rich surface
        instead.

        Treat as opaque ``Any``: callers should never store the value
        long-term — the read client owns its lifetime and may swap the
        implementation in a later phase.
        """
        return self._client

    # ------------------------------------------------------------------
    # IRunQuery surface
    # ------------------------------------------------------------------

    def get_run(self, run_id: str) -> RunHandle:
        """Look up a single run by id.

        Consults the per-process LRU cache first; on miss, calls
        ``MlflowClient.get_run`` and converts the result via
        :meth:`_to_handle`. Only terminal runs are cached.

        :param run_id: MLflow-assigned run id.
        :returns: Converted :class:`RunHandle`.
        :raises KeyError: If the underlying client raises (e.g. the run
            does not exist). Other transport-level errors propagate
            untouched.
        """
        if not run_id:
            raise ValueError("run_id must be a non-empty string")
        with self._lock:
            cached = self._cache.get(run_id)
            if cached is not None:
                # Move to end (most-recently-used).
                self._cache.move_to_end(run_id)
                return cached
        try:
            raw_run = self._client.get_run(run_id)
        except Exception as exc:  # noqa: BLE001 — wrap any client error
            raise KeyError(f"MLflow run not found: {run_id}") from exc
        handle = self._to_handle(raw_run)
        if handle.status in _TERMINAL_STATUSES:
            with self._lock:
                self._cache[run_id] = handle
                self._cache.move_to_end(run_id)
                while len(self._cache) > _CACHE_MAXSIZE:
                    self._cache.popitem(last=False)
        return handle

    def list_children(self, parent_run_id: str) -> Sequence[RunHandle]:
        """List all child runs of ``parent_run_id``.

        Resolves the parent's ``experiment_id`` via :meth:`get_run`,
        then issues paginated ``search_runs`` calls scoped to that
        experiment with ``tags.mlflow.parentRunId = '<parent>'``.
        Hard-capped at :data:`_LIST_CHILDREN_HARD_CAP` rows total to
        protect against pathological hierarchies.

        :param parent_run_id: Parent run id.
        :returns: Tuple of children (possibly empty). Order is whatever
            MLflow returns — typically newest-first by ``start_time``.
        """
        if not parent_run_id:
            raise ValueError("parent_run_id must be a non-empty string")
        parent = self.get_run(parent_run_id)
        experiment_ids = [parent.experiment_id]
        filter_string = f"tags.{_MLFLOW_PARENT_TAG} = '{parent_run_id}'"

        collected: list[RunHandle] = []
        page_token: str | None = None
        while True:
            remaining = _LIST_CHILDREN_HARD_CAP - len(collected)
            if remaining <= 0:
                break
            batch_max = min(_LIST_CHILDREN_PAGE_SIZE, remaining)
            kwargs: dict[str, Any] = {
                "experiment_ids": experiment_ids,
                "filter_string": filter_string,
                "max_results": batch_max,
            }
            if page_token is not None:
                kwargs["page_token"] = page_token
            page = self._client.search_runs(**kwargs)
            for raw_run in page:
                collected.append(self._to_handle(raw_run))
                if len(collected) >= _LIST_CHILDREN_HARD_CAP:
                    break
            next_token = getattr(page, "token", None)
            if not next_token:
                break
            page_token = next_token
        return tuple(collected)

    def search(
        self,
        experiment: str,
        filter_: str,
        max_results: int,
    ) -> Sequence[RunHandle]:
        """Search runs in an experiment by filter expression.

        ``experiment`` is treated as an experiment NAME (not id) and
        resolved via ``MlflowClient.get_experiment_by_name``. Unknown
        experiment names yield an empty result.

        :param experiment: Experiment name.
        :param filter_: MLflow filter string (caller's responsibility
            to escape).
        :param max_results: Upper bound; capped at
            :data:`_SEARCH_PAGE_SIZE` rows per call.
        :returns: Tuple of matching handles (truncated to
            ``min(max_results, _SEARCH_PAGE_SIZE)``).
        """
        if max_results <= 0:
            raise ValueError(f"max_results must be positive, got {max_results!r}")
        experiment_obj = self._client.get_experiment_by_name(experiment)
        if experiment_obj is None:
            return ()
        cap = min(max_results, _SEARCH_PAGE_SIZE)
        page = self._client.search_runs(
            experiment_ids=[experiment_obj.experiment_id],
            filter_string=filter_,
            max_results=cap,
        )
        return tuple(self._to_handle(raw) for raw in page)

    # ------------------------------------------------------------------
    # Cache hygiene
    # ------------------------------------------------------------------

    def invalidate_cache(self, run_id: str | None = None) -> None:
        """Drop cached entries.

        :param run_id: If provided, drop only that entry; otherwise
            clear the whole cache.
        """
        with self._lock:
            if run_id is None:
                self._cache.clear()
            else:
                self._cache.pop(run_id, None)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _to_handle(self, raw_run: Any) -> RunHandle:
        """Convert an ``mlflow.entities.Run`` to :class:`RunHandle`.

        ``parent_run_id`` is sourced from ``run.data.tags`` so we treat
        it the same way the rest of the codebase does (MLflow itself
        models nested runs purely via the ``mlflow.parentRunId`` tag).
        """
        info = raw_run.info
        data = raw_run.data
        tags = getattr(data, "tags", {}) or {}
        parent_run_id = tags.get(_MLFLOW_PARENT_TAG)
        status_raw = getattr(info, "status", None) or "RUNNING"
        try:
            status = RunStatus(status_raw)
        except ValueError:
            # MLflow may surface non-enum statuses (UNKNOWN, SCHEDULED).
            # Conservative default keeps the run out of the LRU cache.
            status = RunStatus.RUNNING
        return RunHandle(
            run_id=info.run_id,
            experiment_id=info.experiment_id,
            parent_run_id=parent_run_id,
            tracking_uri=self._tracking_uri,
            status=status,
        )


__all__ = ["MlflowReadClient"]
