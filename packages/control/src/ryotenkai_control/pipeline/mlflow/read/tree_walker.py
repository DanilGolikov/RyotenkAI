"""Single BFS walker over an MLflow run hierarchy.

Subsumes the three independent traversals scattered across the legacy
codebase:

* ``reports/adapters/mlflow_adapter.py::_get_sorted_children``
* ``pipeline/deletion.py::_search_child_run_ids``
* ``reports/core/builder.py`` (descendant metric collection)

Phase M3.A is additive — the legacy callsites continue to traverse
manually until M3.B rewires them.

Design notes
------------
* The walker depends only on :class:`IRunQuery` so it works against
  the production :class:`MlflowReadClient` and the canonical
  :class:`tests._fakes.mlflow_run_query.FakeRunQuery` alike.
* :meth:`walk` returns an immutable :class:`RunNode` tree. The tree
  is bounded by ``max_depth`` so we cannot blow the stack on a
  pathological hierarchy.
* Per-process LRU cache keyed by root ``parent_run_id``. The walker
  caches ONLY trees whose root run is in a terminal status — under
  that condition the entire subtree is observationally immutable, so
  the cached tree never goes stale.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ryotenkai_shared.infrastructure.mlflow.protocols import RunStatus

if TYPE_CHECKING:
    from ryotenkai_shared.infrastructure.mlflow.protocols import IRunQuery
    from ryotenkai_shared.infrastructure.mlflow.run_handle import RunHandle

_DEFAULT_MAX_DEPTH = 10
_DEFAULT_CACHE_MAXSIZE = 64

_TERMINAL_STATUSES: frozenset[RunStatus] = frozenset(
    {RunStatus.FINISHED, RunStatus.FAILED, RunStatus.KILLED}
)


@dataclass(frozen=True)
class RunNode:
    """Immutable node in a run tree.

    :param handle: The run this node represents.
    :param children: Tuple of child nodes (possibly empty).
    """

    handle: RunHandle
    children: tuple[RunNode, ...]


class RunTreeWalker:
    """BFS walker over an :class:`IRunQuery`.

    :param run_query: Read-path query implementation. Either the
        production :class:`MlflowReadClient` or a fake in tests.
    :param cache_maxsize: Upper bound on the number of root trees
        retained in the per-process cache. Trees are only cached when
        the root run is in a terminal status.
    """

    def __init__(
        self,
        run_query: IRunQuery,
        *,
        cache_maxsize: int = _DEFAULT_CACHE_MAXSIZE,
    ) -> None:
        if cache_maxsize <= 0:
            raise ValueError(
                f"cache_maxsize must be positive, got {cache_maxsize!r}"
            )
        self._run_query = run_query
        self._cache_maxsize = cache_maxsize
        self._lock = threading.Lock()
        self._cache: OrderedDict[str, tuple[RunNode, float]] = OrderedDict()

    # ------------------------------------------------------------------
    # Public surface
    # ------------------------------------------------------------------

    def walk(self, parent_run_id: str, *, max_depth: int = _DEFAULT_MAX_DEPTH) -> RunNode:
        """Build a tree rooted at ``parent_run_id`` via BFS.

        :param parent_run_id: Root run id.
        :param max_depth: Maximum depth to descend. ``0`` returns only
            the root with no children; defaults to
            :data:`_DEFAULT_MAX_DEPTH`. Must be non-negative.
        :returns: Immutable :class:`RunNode` tree.
        :raises ValueError: If ``max_depth`` is negative.
        :raises KeyError: If ``parent_run_id`` does not exist (passed
            through from :class:`IRunQuery.get_run`).
        """
        if max_depth < 0:
            raise ValueError(f"max_depth must be non-negative, got {max_depth!r}")
        if not parent_run_id:
            raise ValueError("parent_run_id must be a non-empty string")

        with self._lock:
            cached = self._cache.get(parent_run_id)
            if cached is not None:
                self._cache.move_to_end(parent_run_id)
                return cached[0]

        root_handle = self._run_query.get_run(parent_run_id)
        node = self._build(root_handle, depth=0, max_depth=max_depth)

        if root_handle.status in _TERMINAL_STATUSES:
            with self._lock:
                self._cache[parent_run_id] = (node, time.monotonic())
                self._cache.move_to_end(parent_run_id)
                while len(self._cache) > self._cache_maxsize:
                    self._cache.popitem(last=False)
        return node

    def flat_descendants(self, parent_run_id: str) -> list[RunHandle]:
        """DFS-preorder flat list of every descendant under ``parent_run_id``.

        The root is included as the first element. Useful when a caller
        only wants to iterate over handles without caring about tree
        shape.

        :param parent_run_id: Root run id.
        :returns: List of :class:`RunHandle` in DFS pre-order.
        """
        root = self.walk(parent_run_id)
        out: list[RunHandle] = []
        self._flatten(root, out)
        return out

    def invalidate(self, run_id: str | None = None) -> None:
        """Drop cached trees.

        :param run_id: If provided, drop only the tree rooted at that
            id. Otherwise clear the entire cache.
        """
        with self._lock:
            if run_id is None:
                self._cache.clear()
            else:
                self._cache.pop(run_id, None)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build(
        self,
        handle: RunHandle,
        *,
        depth: int,
        max_depth: int,
    ) -> RunNode:
        """Recursive helper that descends one level per call."""
        if depth >= max_depth:
            return RunNode(handle=handle, children=())
        raw_children = self._run_query.list_children(handle.run_id)
        children = tuple(
            self._build(child, depth=depth + 1, max_depth=max_depth)
            for child in raw_children
        )
        return RunNode(handle=handle, children=children)

    def _flatten(self, node: RunNode, out: list[RunHandle]) -> None:
        out.append(node.handle)
        for child in node.children:
            self._flatten(child, out)


__all__ = ["RunNode", "RunTreeWalker"]
