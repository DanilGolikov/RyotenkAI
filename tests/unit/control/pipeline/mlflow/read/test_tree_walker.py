"""Tests for :class:`RunTreeWalker`.

Uses the canonical :class:`tests._fakes.mlflow_run_query.FakeRunQuery`
to seed run hierarchies and exercise BFS traversal, max-depth bounds,
cache hit/miss, and invalidation.
"""

from __future__ import annotations

import pytest

from ryotenkai_shared.infrastructure.mlflow.protocols import RunStatus
from ryotenkai_shared.infrastructure.mlflow.run_handle import RunHandle
from tests._fakes.mlflow_run_query import FakeRunQuery

from ryotenkai_control.pipeline.mlflow.read.tree_walker import (
    RunNode,
    RunTreeWalker,
)


def _h(
    run_id: str,
    *,
    parent: str | None = None,
    status: RunStatus = RunStatus.FINISHED,
    experiment_id: str = "e1",
) -> RunHandle:
    return RunHandle(
        run_id=run_id,
        experiment_id=experiment_id,
        parent_run_id=parent,
        tracking_uri="fake://",
        status=status,
    )


class TestConstruction:
    def test_rejects_zero_cache_maxsize(self) -> None:
        with pytest.raises(ValueError, match="cache_maxsize"):
            RunTreeWalker(FakeRunQuery(), cache_maxsize=0)


class TestBfsCorrectness:
    def test_single_node_no_children(self) -> None:
        q = FakeRunQuery([_h("root")])
        w = RunTreeWalker(q)
        tree = w.walk("root")
        assert tree.handle.run_id == "root"
        assert tree.children == ()

    def test_two_level_tree(self) -> None:
        q = FakeRunQuery(
            [
                _h("root"),
                _h("c1", parent="root"),
                _h("c2", parent="root"),
            ]
        )
        w = RunTreeWalker(q)
        tree = w.walk("root")
        assert {c.handle.run_id for c in tree.children} == {"c1", "c2"}
        for c in tree.children:
            assert c.children == ()

    def test_three_level_tree(self) -> None:
        q = FakeRunQuery(
            [
                _h("root"),
                _h("a", parent="root"),
                _h("b", parent="root"),
                _h("a1", parent="a"),
                _h("a2", parent="a"),
                _h("b1", parent="b"),
            ]
        )
        w = RunTreeWalker(q)
        tree = w.walk("root")
        assert len(tree.children) == 2
        a_node = next(c for c in tree.children if c.handle.run_id == "a")
        b_node = next(c for c in tree.children if c.handle.run_id == "b")
        assert {c.handle.run_id for c in a_node.children} == {"a1", "a2"}
        assert {c.handle.run_id for c in b_node.children} == {"b1"}


class TestMaxDepth:
    def test_max_depth_zero_returns_root_only(self) -> None:
        q = FakeRunQuery(
            [
                _h("root"),
                _h("c1", parent="root"),
            ]
        )
        w = RunTreeWalker(q)
        tree = w.walk("root", max_depth=0)
        assert tree.children == ()

    def test_max_depth_one_includes_only_first_level(self) -> None:
        q = FakeRunQuery(
            [
                _h("root"),
                _h("a", parent="root"),
                _h("a1", parent="a"),
            ]
        )
        w = RunTreeWalker(q)
        tree = w.walk("root", max_depth=1)
        assert len(tree.children) == 1
        assert tree.children[0].children == ()

    def test_negative_max_depth_rejected(self) -> None:
        w = RunTreeWalker(FakeRunQuery([_h("root")]))
        with pytest.raises(ValueError, match="max_depth"):
            w.walk("root", max_depth=-1)

    def test_rejects_empty_parent(self) -> None:
        w = RunTreeWalker(FakeRunQuery())
        with pytest.raises(ValueError, match="parent_run_id"):
            w.walk("")


class TestFlatDescendants:
    def test_includes_root_and_descendants(self) -> None:
        q = FakeRunQuery(
            [
                _h("root"),
                _h("a", parent="root"),
                _h("b", parent="root"),
                _h("a1", parent="a"),
            ]
        )
        w = RunTreeWalker(q)
        flat = w.flat_descendants("root")
        assert [h.run_id for h in flat][0] == "root"
        assert {h.run_id for h in flat} == {"root", "a", "a1", "b"}

    def test_returns_list(self) -> None:
        q = FakeRunQuery([_h("root")])
        w = RunTreeWalker(q)
        flat = w.flat_descendants("root")
        assert isinstance(flat, list)


class TestCache:
    def test_terminal_root_caches_tree(self) -> None:
        q = FakeRunQuery([_h("root", status=RunStatus.FINISHED)])
        w = RunTreeWalker(q)
        w.walk("root")
        w.walk("root")
        # First call: 1 get_run + 1 list_children. Second call: cache hit.
        assert q.get_run_calls == ["root"]
        assert q.list_children_calls == ["root"]

    def test_running_root_is_not_cached(self) -> None:
        q = FakeRunQuery([_h("root", status=RunStatus.RUNNING)])
        w = RunTreeWalker(q)
        w.walk("root")
        w.walk("root")
        # Both calls fetch.
        assert q.get_run_calls == ["root", "root"]
        assert q.list_children_calls == ["root", "root"]

    def test_invalidate_specific(self) -> None:
        q = FakeRunQuery([_h("root", status=RunStatus.FINISHED)])
        w = RunTreeWalker(q)
        w.walk("root")
        w.invalidate("root")
        w.walk("root")
        assert q.get_run_calls == ["root", "root"]

    def test_invalidate_all(self) -> None:
        q = FakeRunQuery(
            [
                _h("a", status=RunStatus.FINISHED),
                _h("b", status=RunStatus.FINISHED),
            ]
        )
        w = RunTreeWalker(q)
        w.walk("a")
        w.walk("b")
        w.invalidate()
        w.walk("a")
        w.walk("b")
        assert q.get_run_calls == ["a", "b", "a", "b"]

    def test_cache_maxsize_eviction(self) -> None:
        q = FakeRunQuery(
            [
                _h("a", status=RunStatus.FINISHED),
                _h("b", status=RunStatus.FINISHED),
                _h("c", status=RunStatus.FINISHED),
            ]
        )
        w = RunTreeWalker(q, cache_maxsize=2)
        w.walk("a")
        w.walk("b")
        w.walk("c")  # evicts 'a'
        w.walk("a")  # cache miss again
        # 'a' fetched twice; 'b' and 'c' once each.
        assert q.get_run_calls.count("a") == 2
        assert q.get_run_calls.count("b") == 1
        assert q.get_run_calls.count("c") == 1


class TestImmutability:
    def test_run_node_is_frozen(self) -> None:
        node = RunNode(handle=_h("r"), children=())
        with pytest.raises(Exception):  # noqa: BLE001 — FrozenInstanceError or TypeError
            node.handle = _h("other")  # type: ignore[misc]

    def test_children_is_tuple(self) -> None:
        q = FakeRunQuery(
            [
                _h("root"),
                _h("c", parent="root"),
            ]
        )
        w = RunTreeWalker(q)
        tree = w.walk("root")
        assert isinstance(tree.children, tuple)
