"""Canonical fake for :class:`IRunQuery` Protocol.

Use this in tests instead of ``unittest.mock.Mock(spec=IRunQuery)``
— the sentinel :mod:`tests._lint.test_no_protocol_mocking` forbids that.

The fake is the read-path counterpart to :class:`FakeTrackingClient`:
tests seed runs via :meth:`add_run` (or by sharing the dict with a
FakeTrackingClient) and the query methods return matching handles.
A pluggable :class:`SearchPredicate` lets tests model server-side
``filter_`` expressions without re-implementing the MLflow grammar.

Example::

    query = FakeRunQuery()
    query.add_run(RunHandle(run_id="r1", experiment_id="e1",
                            parent_run_id=None,
                            tracking_uri="fake://",
                            status=RunStatus.RUNNING))
    handle = query.get_run("r1")
    assert handle.experiment_id == "e1"
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from ryotenkai_shared.infrastructure.mlflow.protocols import RunStatus
from ryotenkai_shared.infrastructure.mlflow.run_handle import RunHandle


class TransientQueryError(Exception):
    """Default exception raised by :meth:`FakeRunQuery.fail_next_n_calls`."""


# A search predicate: given a RunHandle and the literal filter_ string,
# returns True iff the handle should be included. Tests provide their own
# matcher; default accepts everything.
SearchPredicate = Callable[[RunHandle, str], bool]


def _accept_all(_handle: RunHandle, _filter: str) -> bool:
    """Default predicate: every run matches every filter."""
    return True


@dataclass(frozen=True)
class SearchCall:
    """Captured invocation of :meth:`FakeRunQuery.search`.

    :param experiment: Experiment scope passed to the call.
    :param filter_: Literal MLflow-style filter expression.
    :param max_results: Upper bound on returned runs.
    """

    experiment: str
    filter_: str
    max_results: int


class FakeRunQuery:
    """In-memory fake for :class:`IRunQuery`.

    :param runs: Optional seed iterable of :class:`RunHandle`. Each is
        stored by ``run_id`` and may be looked up by :meth:`get_run`.
    :param predicate: Search matcher (default accepts all). Receives the
        handle and the literal ``filter_`` string for each candidate.
    :param experiment_of: Optional override that maps a :class:`RunHandle`
        to the experiment name used by :meth:`search` (defaults to the
        handle's ``experiment_id``).
    """

    def __init__(
        self,
        runs: Sequence[RunHandle] | None = None,
        *,
        predicate: SearchPredicate | None = None,
        experiment_of: Callable[[RunHandle], str] | None = None,
    ) -> None:
        self._runs: dict[str, RunHandle] = {}
        if runs is not None:
            for handle in runs:
                self._runs[handle.run_id] = handle
        self._predicate: SearchPredicate = predicate or _accept_all
        self._experiment_of: Callable[[RunHandle], str] = (
            experiment_of if experiment_of is not None else (lambda h: h.experiment_id)
        )
        # Call logs.
        self.get_run_calls: list[str] = []
        self.list_children_calls: list[str] = []
        self.search_calls: list[SearchCall] = []
        # Chaos state.
        self._fail_remaining: int = 0
        self._fail_kind: type[Exception] = TransientQueryError

    # ------------------------------------------------------------------
    # Chaos surface
    # ------------------------------------------------------------------

    def fail_next_n_calls(
        self,
        n: int,
        kind: type[Exception] = TransientQueryError,
    ) -> None:
        """Program the next ``n`` query calls to raise.

        :param n: Non-negative count of failures.
        :param kind: Exception class to raise.
        :raises ValueError: If ``n`` is negative.
        """
        if n < 0:
            raise ValueError("fail_next_n_calls requires non-negative count")
        self._fail_remaining = n
        self._fail_kind = kind

    def reset_chaos(self) -> None:
        """Clear chaos state."""
        self._fail_remaining = 0

    # ------------------------------------------------------------------
    # Seed / mutation helpers
    # ------------------------------------------------------------------

    def add_run(self, handle: RunHandle) -> None:
        """Insert (or overwrite) a run.

        :param handle: :class:`RunHandle` to register under its ``run_id``.
        """
        self._runs[handle.run_id] = handle

    def update_status(self, run_id: str, status: RunStatus) -> None:
        """Replace the stored handle for ``run_id`` with a new status snapshot.

        :param run_id: Target run id.
        :param status: New :class:`RunStatus`.
        :raises KeyError: If ``run_id`` is unknown.
        """
        if run_id not in self._runs:
            raise KeyError(f"unknown run_id: {run_id}")
        old = self._runs[run_id]
        self._runs[run_id] = RunHandle(
            run_id=old.run_id,
            experiment_id=old.experiment_id,
            parent_run_id=old.parent_run_id,
            tracking_uri=old.tracking_uri,
            status=status,
        )

    def set_predicate(self, predicate: SearchPredicate) -> None:
        """Swap the matcher used by :meth:`search`."""
        self._predicate = predicate

    # ------------------------------------------------------------------
    # IRunQuery surface
    # ------------------------------------------------------------------

    def get_run(self, run_id: str) -> RunHandle:
        """Look up a run by id.

        :param run_id: Identifier.
        :returns: The stored :class:`RunHandle`.
        :raises KeyError: If ``run_id`` is unknown.
        """
        self._guard()
        self.get_run_calls.append(run_id)
        if run_id not in self._runs:
            raise KeyError(f"unknown run_id: {run_id}")
        return self._runs[run_id]

    def list_children(self, parent_run_id: str) -> Sequence[RunHandle]:
        """Return all stored runs whose ``parent_run_id`` matches.

        :param parent_run_id: Parent identifier.
        :returns: Tuple of matching handles (empty if none).
        """
        self._guard()
        self.list_children_calls.append(parent_run_id)
        return tuple(
            handle
            for handle in self._runs.values()
            if handle.parent_run_id == parent_run_id
        )

    def search(
        self,
        experiment: str,
        filter_: str,
        max_results: int,
    ) -> Sequence[RunHandle]:
        """Search runs under ``experiment`` matching ``filter_``.

        :param experiment: Experiment name.
        :param filter_: MLflow-style filter expression (interpreted by the
            configured predicate).
        :param max_results: Upper bound on returned runs.
        :returns: Tuple of matching handles, truncated to ``max_results``.
        """
        self._guard()
        self.search_calls.append(
            SearchCall(experiment=experiment, filter_=filter_, max_results=max_results)
        )
        matches: list[RunHandle] = []
        for handle in self._runs.values():
            if self._experiment_of(handle) != experiment:
                continue
            if not self._predicate(handle, filter_):
                continue
            matches.append(handle)
            if len(matches) >= max_results:
                break
        return tuple(matches)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _guard(self) -> None:
        if self._fail_remaining > 0:
            self._fail_remaining -= 1
            raise self._fail_kind("fake_injected_failure")


__all__ = [
    "FakeRunQuery",
    "SearchCall",
    "SearchPredicate",
    "TransientQueryError",
]
