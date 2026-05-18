"""Tests for :class:`MlflowReadClient`.

The client wraps ``mlflow.tracking.MlflowClient`` so the tests
substitute a hand-rolled stub via ``monkeypatch``. The stub mimics
the small subset of methods we depend on:

* ``get_run(run_id) -> _Run``
* ``search_runs(experiment_ids, filter_string, max_results, page_token)``
* ``get_experiment_by_name(name)``

Each test class focuses on one behaviour (7-class structure mandated
by the agent testing workflow).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from ryotenkai_shared.infrastructure.mlflow.protocols import RunStatus


# ---------------------------------------------------------------------------
# Stub MlflowClient
# ---------------------------------------------------------------------------


@dataclass
class _RunInfo:
    run_id: str
    experiment_id: str
    status: str = "RUNNING"


@dataclass
class _RunData:
    tags: dict[str, str] = field(default_factory=dict)


@dataclass
class _Run:
    info: _RunInfo
    data: _RunData = field(default_factory=_RunData)


class _PagedList(list):
    """A list with a ``.token`` attribute, mirroring MLflow paged results."""

    token: str | None = None


@dataclass
class _Experiment:
    experiment_id: str


class _StubMlflowClient:
    """In-memory stand-in for ``mlflow.tracking.MlflowClient``."""

    def __init__(self, tracking_uri: str) -> None:
        self.tracking_uri = tracking_uri
        self.runs: dict[str, _Run] = {}
        self.experiments: dict[str, _Experiment] = {}
        self.get_run_calls: list[str] = []
        self.search_runs_calls: list[dict[str, Any]] = []

    def add_run(
        self,
        run_id: str,
        experiment_id: str,
        *,
        status: str = "RUNNING",
        parent_run_id: str | None = None,
        extra_tags: dict[str, str] | None = None,
    ) -> None:
        tags = dict(extra_tags or {})
        if parent_run_id is not None:
            tags["mlflow.parentRunId"] = parent_run_id
        self.runs[run_id] = _Run(
            info=_RunInfo(run_id=run_id, experiment_id=experiment_id, status=status),
            data=_RunData(tags=tags),
        )

    def get_run(self, run_id: str) -> _Run:
        self.get_run_calls.append(run_id)
        if run_id not in self.runs:
            raise RuntimeError(f"run {run_id!r} not found")
        return self.runs[run_id]

    def search_runs(
        self,
        *,
        experiment_ids: list[str],
        filter_string: str,
        max_results: int,
        page_token: str | None = None,
    ) -> _PagedList:
        self.search_runs_calls.append(
            {
                "experiment_ids": tuple(experiment_ids),
                "filter_string": filter_string,
                "max_results": max_results,
                "page_token": page_token,
            }
        )
        matches: list[_Run] = []
        for run in self.runs.values():
            if run.info.experiment_id not in experiment_ids:
                continue
            if filter_string:
                # very small parser — only ``tags.mlflow.parentRunId = '...'``.
                if "tags.mlflow.parentRunId =" in filter_string:
                    wanted = filter_string.split("'")[1]
                    if run.data.tags.get("mlflow.parentRunId") != wanted:
                        continue
            matches.append(run)
            if len(matches) >= max_results:
                break
        out = _PagedList(matches)
        return out

    def get_experiment_by_name(self, name: str) -> _Experiment | None:
        return self.experiments.get(name)


@pytest.fixture
def stub_client_class(monkeypatch: pytest.MonkeyPatch) -> type[_StubMlflowClient]:
    """Patch ``mlflow.tracking.MlflowClient`` to return the stub."""
    import mlflow.tracking  # noqa: PLC0415

    monkeypatch.setattr(mlflow.tracking, "MlflowClient", _StubMlflowClient)
    return _StubMlflowClient


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_constructs_with_uri(self, stub_client_class: type[_StubMlflowClient]) -> None:
        from ryotenkai_control.pipeline.mlflow.read.client import MlflowReadClient

        c = MlflowReadClient("http://mlflow.example/")
        assert c.tracking_uri == "http://mlflow.example/"

    def test_rejects_empty_uri(self, stub_client_class: type[_StubMlflowClient]) -> None:
        from ryotenkai_control.pipeline.mlflow.read.client import MlflowReadClient

        with pytest.raises(ValueError, match="tracking_uri"):
            MlflowReadClient("")

    def test_rejects_non_positive_timeout(self, stub_client_class: type[_StubMlflowClient]) -> None:
        from ryotenkai_control.pipeline.mlflow.read.client import MlflowReadClient

        with pytest.raises(ValueError, match="request_timeout_s"):
            MlflowReadClient("http://x", request_timeout_s=0.0)
        with pytest.raises(ValueError, match="request_timeout_s"):
            MlflowReadClient("http://x", request_timeout_s=-1.0)


class TestGetRun:
    def test_returns_run_handle(self, stub_client_class: type[_StubMlflowClient]) -> None:
        from ryotenkai_control.pipeline.mlflow.read.client import MlflowReadClient

        c = MlflowReadClient("http://x")
        c._client.add_run("r1", "e1", status="FINISHED")
        h = c.get_run("r1")
        assert h.run_id == "r1"
        assert h.experiment_id == "e1"
        assert h.status == RunStatus.FINISHED
        assert h.parent_run_id is None
        assert h.tracking_uri == "http://x"

    def test_extracts_parent_run_id_from_tags(
        self, stub_client_class: type[_StubMlflowClient]
    ) -> None:
        from ryotenkai_control.pipeline.mlflow.read.client import MlflowReadClient

        c = MlflowReadClient("http://x")
        c._client.add_run("child", "e1", parent_run_id="parent")
        h = c.get_run("child")
        assert h.parent_run_id == "parent"

    def test_unknown_status_defaults_to_running(
        self, stub_client_class: type[_StubMlflowClient]
    ) -> None:
        from ryotenkai_control.pipeline.mlflow.read.client import MlflowReadClient

        c = MlflowReadClient("http://x")
        c._client.add_run("r1", "e1", status="SCHEDULED")
        h = c.get_run("r1")
        assert h.status == RunStatus.RUNNING

    def test_missing_run_raises_keyerror(
        self, stub_client_class: type[_StubMlflowClient]
    ) -> None:
        from ryotenkai_control.pipeline.mlflow.read.client import MlflowReadClient

        c = MlflowReadClient("http://x")
        with pytest.raises(KeyError):
            c.get_run("does-not-exist")

    def test_rejects_empty_run_id(self, stub_client_class: type[_StubMlflowClient]) -> None:
        from ryotenkai_control.pipeline.mlflow.read.client import MlflowReadClient

        c = MlflowReadClient("http://x")
        with pytest.raises(ValueError, match="run_id"):
            c.get_run("")


class TestLruCache:
    def test_terminal_run_is_cached(
        self, stub_client_class: type[_StubMlflowClient]
    ) -> None:
        from ryotenkai_control.pipeline.mlflow.read.client import MlflowReadClient

        c = MlflowReadClient("http://x")
        c._client.add_run("r1", "e1", status="FINISHED")
        c.get_run("r1")
        c.get_run("r1")
        c.get_run("r1")
        # First call hits the stub; subsequent calls served from cache.
        assert c._client.get_run_calls == ["r1"]

    def test_running_run_is_not_cached(
        self, stub_client_class: type[_StubMlflowClient]
    ) -> None:
        from ryotenkai_control.pipeline.mlflow.read.client import MlflowReadClient

        c = MlflowReadClient("http://x")
        c._client.add_run("r1", "e1", status="RUNNING")
        c.get_run("r1")
        c.get_run("r1")
        # No caching — both calls hit the stub.
        assert c._client.get_run_calls == ["r1", "r1"]

    def test_invalidate_cache_drops_entry(
        self, stub_client_class: type[_StubMlflowClient]
    ) -> None:
        from ryotenkai_control.pipeline.mlflow.read.client import MlflowReadClient

        c = MlflowReadClient("http://x")
        c._client.add_run("r1", "e1", status="FINISHED")
        c.get_run("r1")
        c.invalidate_cache("r1")
        c.get_run("r1")
        # Cache evicted -> second call refetches.
        assert c._client.get_run_calls == ["r1", "r1"]

    def test_invalidate_cache_clears_all(
        self, stub_client_class: type[_StubMlflowClient]
    ) -> None:
        from ryotenkai_control.pipeline.mlflow.read.client import MlflowReadClient

        c = MlflowReadClient("http://x")
        c._client.add_run("r1", "e1", status="FAILED")
        c._client.add_run("r2", "e1", status="FAILED")
        c.get_run("r1")
        c.get_run("r2")
        c.invalidate_cache()
        c.get_run("r1")
        c.get_run("r2")
        assert c._client.get_run_calls == ["r1", "r2", "r1", "r2"]


class TestListChildren:
    def test_returns_only_children_of_parent(
        self, stub_client_class: type[_StubMlflowClient]
    ) -> None:
        from ryotenkai_control.pipeline.mlflow.read.client import MlflowReadClient

        c = MlflowReadClient("http://x")
        c._client.add_run("parent", "e1", status="FINISHED")
        c._client.add_run("child1", "e1", parent_run_id="parent")
        c._client.add_run("child2", "e1", parent_run_id="parent")
        c._client.add_run("unrelated", "e1", parent_run_id="other")
        children = c.list_children("parent")
        assert {h.run_id for h in children} == {"child1", "child2"}

    def test_returns_empty_when_no_children(
        self, stub_client_class: type[_StubMlflowClient]
    ) -> None:
        from ryotenkai_control.pipeline.mlflow.read.client import MlflowReadClient

        c = MlflowReadClient("http://x")
        c._client.add_run("parent", "e1", status="FINISHED")
        assert c.list_children("parent") == ()

    def test_uses_parent_experiment_id(
        self, stub_client_class: type[_StubMlflowClient]
    ) -> None:
        from ryotenkai_control.pipeline.mlflow.read.client import MlflowReadClient

        c = MlflowReadClient("http://x")
        c._client.add_run("parent", "exp-77", status="FINISHED")
        c._client.add_run("child", "exp-77", parent_run_id="parent")
        c.list_children("parent")
        # The search call must scope to the parent's experiment id.
        last = c._client.search_runs_calls[-1]
        assert last["experiment_ids"] == ("exp-77",)
        assert "parent" in last["filter_string"]

    def test_rejects_empty_parent(
        self, stub_client_class: type[_StubMlflowClient]
    ) -> None:
        from ryotenkai_control.pipeline.mlflow.read.client import MlflowReadClient

        c = MlflowReadClient("http://x")
        with pytest.raises(ValueError, match="parent_run_id"):
            c.list_children("")


class TestSearch:
    def test_resolves_experiment_name(
        self, stub_client_class: type[_StubMlflowClient]
    ) -> None:
        from ryotenkai_control.pipeline.mlflow.read.client import MlflowReadClient

        c = MlflowReadClient("http://x")
        c._client.experiments["my-exp"] = _Experiment(experiment_id="e1")
        c._client.add_run("r1", "e1", status="FINISHED")
        results = c.search("my-exp", "", max_results=10)
        assert len(results) == 1
        assert results[0].run_id == "r1"

    def test_unknown_experiment_returns_empty(
        self, stub_client_class: type[_StubMlflowClient]
    ) -> None:
        from ryotenkai_control.pipeline.mlflow.read.client import MlflowReadClient

        c = MlflowReadClient("http://x")
        results = c.search("does-not-exist", "", max_results=10)
        assert results == ()

    def test_max_results_is_capped_at_page_size(
        self, stub_client_class: type[_StubMlflowClient]
    ) -> None:
        from ryotenkai_control.pipeline.mlflow.read.client import MlflowReadClient
        from ryotenkai_control.pipeline.mlflow.read.client import _SEARCH_PAGE_SIZE

        c = MlflowReadClient("http://x")
        c._client.experiments["e"] = _Experiment(experiment_id="e1")
        c.search("e", "", max_results=10_000)
        last = c._client.search_runs_calls[-1]
        assert last["max_results"] == _SEARCH_PAGE_SIZE

    def test_rejects_zero_max_results(
        self, stub_client_class: type[_StubMlflowClient]
    ) -> None:
        from ryotenkai_control.pipeline.mlflow.read.client import MlflowReadClient

        c = MlflowReadClient("http://x")
        with pytest.raises(ValueError, match="max_results"):
            c.search("any", "", max_results=0)


class TestImplementsIRunQuery:
    def test_is_runtime_checkable_irunquery(
        self, stub_client_class: type[_StubMlflowClient]
    ) -> None:
        from ryotenkai_shared.infrastructure.mlflow.protocols import IRunQuery

        from ryotenkai_control.pipeline.mlflow.read.client import MlflowReadClient

        c = MlflowReadClient("http://x")
        assert isinstance(c, IRunQuery)


class TestRegressions:
    def test_to_handle_with_missing_tags(
        self, stub_client_class: type[_StubMlflowClient]
    ) -> None:
        """Regression: ``data.tags=None`` should not crash ``_to_handle``."""
        from ryotenkai_control.pipeline.mlflow.read.client import MlflowReadClient

        c = MlflowReadClient("http://x")
        # Construct a synthetic raw_run with tags=None to exercise the
        # ``tags or {}`` fallback in ``_to_handle``.
        raw = _Run(
            info=_RunInfo(run_id="r1", experiment_id="e1", status="FINISHED"),
            data=_RunData(tags={}),
        )
        raw.data.tags = None  # type: ignore[assignment]
        h = c._to_handle(raw)
        assert h.parent_run_id is None

    def test_status_none_defaults_to_running(
        self, stub_client_class: type[_StubMlflowClient]
    ) -> None:
        from ryotenkai_control.pipeline.mlflow.read.client import MlflowReadClient

        c = MlflowReadClient("http://x")
        raw = _Run(
            info=_RunInfo(run_id="r1", experiment_id="e1", status=""),
            data=_RunData(tags={}),
        )
        # status='' → falsy → default 'RUNNING'
        h = c._to_handle(raw)
        assert h.status == RunStatus.RUNNING
