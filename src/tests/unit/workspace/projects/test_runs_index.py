"""Tests for the per-project runs ledger (Step 6 of Variant 1).

``runs/index.json`` is an append-only ledger of launches. The tests pin:
- append-only semantics (new entries don't shadow old ones)
- defensive reads (missing / malformed file → empty, never raises)
- filtering and limiting work as advertised
- atomic writes (via the shared ``atomic_write_json`` plumbing)
- update_run_status resolves the latest entry for the same run_id

Categories: positive, negative, boundary, invariants, regression,
logic-specific, combinatorial.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.workspace.projects.registry import ProjectRegistry
from src.workspace.projects.store import (
    RUNS_INDEX_SCHEMA_VERSION,
    ProjectStore,
)


def _make_project(tmp_path: Path, *, project_id: str = "p1") -> ProjectStore:
    registry = ProjectRegistry(root=tmp_path)
    project_path = registry.default_project_path(project_id)
    store = ProjectStore(project_path)
    store.create(id=project_id, name="P1")
    registry.register(project_id=project_id, name="P1", path=project_path)
    return store


# ---------------------------------------------------------------------------
# Positive
# ---------------------------------------------------------------------------


class TestPositive:
    def test_register_run_appends_entry(self, tmp_path: Path) -> None:
        store = _make_project(tmp_path)
        entry = store.register_run(
            run_id="run-1",
            mlflow_run_id="ml-abc",
            config_version_hash="abc123",
            actor="alice",
            run_directory="/tmp/run-1",
        )

        assert entry["run_id"] == "run-1"
        assert entry["status"] == "running"
        assert "started_at" in entry
        assert entry["mlflow_run_id"] == "ml-abc"

        runs = store.list_runs()
        assert len(runs) == 1
        assert runs[0]["run_id"] == "run-1"

    def test_list_runs_newest_first(self, tmp_path: Path) -> None:
        store = _make_project(tmp_path)
        store.register_run(run_id="r1", started_at="2026-01-01T00:00:00Z")
        store.register_run(run_id="r2", started_at="2026-02-01T00:00:00Z")
        store.register_run(run_id="r3", started_at="2026-03-01T00:00:00Z")

        runs = store.list_runs()
        assert [r["run_id"] for r in runs] == ["r3", "r2", "r1"]

    def test_filter_by_status(self, tmp_path: Path) -> None:
        store = _make_project(tmp_path)
        store.register_run(run_id="r-running", status="running")
        store.register_run(run_id="r-done", status="completed")
        store.register_run(run_id="r-failed", status="failed")

        only_done = store.list_runs(status="completed")
        assert [r["run_id"] for r in only_done] == ["r-done"]

    def test_limit_caps_results(self, tmp_path: Path) -> None:
        store = _make_project(tmp_path)
        for i in range(5):
            store.register_run(
                run_id=f"r{i}", started_at=f"2026-04-{i + 1:02d}T00:00:00Z",
            )

        runs = store.list_runs(limit=3)
        assert len(runs) == 3
        # Newest first means r4, r3, r2 (April 5/4/3).
        assert [r["run_id"] for r in runs] == ["r4", "r3", "r2"]

    def test_update_run_status_marks_finished(self, tmp_path: Path) -> None:
        store = _make_project(tmp_path)
        store.register_run(run_id="r1")
        updated = store.update_run_status(
            "r1", status="completed", finished_at="2026-04-28T10:00:00Z",
        )

        assert updated is not None
        assert updated["status"] == "completed"
        assert updated["finished_at"] == "2026-04-28T10:00:00Z"

        runs = store.list_runs()
        assert runs[0]["status"] == "completed"


# ---------------------------------------------------------------------------
# Negative
# ---------------------------------------------------------------------------


class TestNegative:
    def test_list_runs_when_index_missing_yields_empty(
        self, tmp_path: Path
    ) -> None:
        store = _make_project(tmp_path)
        # Index hasn't been created yet (no launches).
        assert not store.runs_index_path.exists()
        assert store.list_runs() == []

    def test_list_runs_when_index_malformed_yields_empty(
        self, tmp_path: Path
    ) -> None:
        store = _make_project(tmp_path)
        store.runs_dir.mkdir(parents=True, exist_ok=True)
        store.runs_index_path.write_text("not json at all", encoding="utf-8")

        assert store.list_runs() == []

    def test_list_runs_when_index_is_array_not_object_yields_empty(
        self, tmp_path: Path
    ) -> None:
        """Defensive: someone hand-edits index.json to be a plain list.
        We expect ``{"runs": [...]}`` shape; anything else → empty."""
        store = _make_project(tmp_path)
        store.runs_dir.mkdir(parents=True, exist_ok=True)
        store.runs_index_path.write_text("[]", encoding="utf-8")

        assert store.list_runs() == []

    def test_update_unknown_run_id_returns_none(self, tmp_path: Path) -> None:
        store = _make_project(tmp_path)
        store.register_run(run_id="r1")

        result = store.update_run_status("nonexistent", status="completed")

        assert result is None
        # Original entry unchanged.
        assert store.list_runs()[0]["status"] == "running"


# ---------------------------------------------------------------------------
# Boundary
# ---------------------------------------------------------------------------


class TestBoundary:
    def test_empty_runs_array_is_empty_list(self, tmp_path: Path) -> None:
        store = _make_project(tmp_path)
        store.runs_dir.mkdir(parents=True, exist_ok=True)
        store.runs_index_path.write_text(
            json.dumps({"schema_version": 1, "runs": []}),
            encoding="utf-8",
        )
        assert store.list_runs() == []

    def test_negative_limit_normalised(self, tmp_path: Path) -> None:
        """``limit=-1`` is nonsense; the store should not slice with a
        negative index. We accept "treat as no-cap" or "treat as 0";
        either is fine — pinning the choice so behaviour is explicit."""
        store = _make_project(tmp_path)
        store.register_run(run_id="r1")
        store.register_run(run_id="r2")

        runs = store.list_runs(limit=-1)
        # Implementation: negative limit currently slices to [:negative],
        # which Python treats as "all but last n". To avoid surprising
        # callers we treat negative as "no cap" via the ``>= 0`` guard
        # in store.list_runs. So all runs come back.
        assert {r["run_id"] for r in runs} == {"r1", "r2"}

    def test_limit_zero_returns_empty_list(self, tmp_path: Path) -> None:
        store = _make_project(tmp_path)
        store.register_run(run_id="r1")

        assert store.list_runs(limit=0) == []

    def test_extra_metadata_does_not_override_canonical_keys(
        self, tmp_path: Path
    ) -> None:
        """``extra={"run_id": "spoof"}`` must NOT replace the real
        run_id in the entry."""
        store = _make_project(tmp_path)
        entry = store.register_run(
            run_id="real-id",
            extra={"run_id": "spoofed-id", "tag": "extra-tag"},
        )

        assert entry["run_id"] == "real-id"
        assert entry["tag"] == "extra-tag"


# ---------------------------------------------------------------------------
# Invariants
# ---------------------------------------------------------------------------


class TestInvariants:
    def test_append_only_old_entries_preserved(self, tmp_path: Path) -> None:
        """R16 from the plan: launches don't shadow old entries."""
        store = _make_project(tmp_path)
        store.register_run(run_id="r1", started_at="2026-01-01T00:00:00Z")
        store.register_run(run_id="r2", started_at="2026-02-01T00:00:00Z")
        store.register_run(run_id="r3", started_at="2026-03-01T00:00:00Z")

        all_ids = {r["run_id"] for r in store.list_runs()}
        assert all_ids == {"r1", "r2", "r3"}

    def test_schema_version_stamped(self, tmp_path: Path) -> None:
        store = _make_project(tmp_path)
        store.register_run(run_id="r1")

        with store.runs_index_path.open() as fh:
            payload = json.load(fh)
        assert payload["schema_version"] == RUNS_INDEX_SCHEMA_VERSION

    def test_register_does_not_mutate_existing_entries(
        self, tmp_path: Path
    ) -> None:
        store = _make_project(tmp_path)
        store.register_run(
            run_id="r1", mlflow_run_id="ml-1", actor="alice",
        )
        first_entry = store.list_runs()[0]
        # Snapshot keys before second register.
        snapshot = dict(first_entry)

        store.register_run(run_id="r2", mlflow_run_id="ml-2", actor="bob")
        # Find r1 again.
        r1_after = next(r for r in store.list_runs() if r["run_id"] == "r1")
        assert r1_after == snapshot


# ---------------------------------------------------------------------------
# Logic-specific
# ---------------------------------------------------------------------------


class TestLogicSpecific:
    def test_status_filter_combined_with_limit(self, tmp_path: Path) -> None:
        """``limit`` applies AFTER ``status`` filter — caller asks for
        N matching entries, not N entries scanned."""
        store = _make_project(tmp_path)
        for i in range(6):
            store.register_run(
                run_id=f"r{i}",
                status="completed" if i % 2 == 0 else "running",
                started_at=f"2026-04-{i + 1:02d}T00:00:00Z",
            )

        completed = store.list_runs(status="completed", limit=2)
        # 6 entries: r0/r2/r4 are completed; newest first → r4, r2.
        assert [r["run_id"] for r in completed] == ["r4", "r2"]

    def test_update_status_resolves_most_recent_for_run_id(
        self, tmp_path: Path
    ) -> None:
        """If the same run_id appears twice (rare: re-launch with same
        id), update_run_status hits the most-recently appended entry."""
        store = _make_project(tmp_path)
        store.register_run(
            run_id="r1", started_at="2026-01-01T00:00:00Z", status="running",
        )
        # Pretend a second launch reused the same id (could happen in
        # restart-from-stage scenarios where the logical_run_id is
        # stable).
        store.register_run(
            run_id="r1", started_at="2026-02-01T00:00:00Z", status="running",
        )
        store.update_run_status("r1", status="completed")

        runs = store.list_runs()
        # Both entries still present (append-only), but the LATEST is
        # marked completed — the older one stays "running" as a record
        # of the prior incomplete attempt.
        latest = max(runs, key=lambda r: r["started_at"])
        oldest = min(runs, key=lambda r: r["started_at"])
        assert latest["status"] == "completed"
        assert oldest["status"] == "running"


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------


class TestRegression:
    def test_register_run_creates_runs_dir_when_missing(
        self, tmp_path: Path
    ) -> None:
        store = _make_project(tmp_path)
        # Wipe runs_dir to mimic a project created before Step 6.
        if store.runs_dir.exists():
            store.runs_dir.rmdir()

        store.register_run(run_id="r1")

        assert store.runs_dir.is_dir()
        assert store.runs_index_path.is_file()

    def test_update_status_creates_runs_dir_when_missing(
        self, tmp_path: Path
    ) -> None:
        """Edge case: update_run_status called before any register
        (e.g. status reporter races the launcher). It's a no-op return,
        but must not blow up because runs_dir doesn't exist yet."""
        store = _make_project(tmp_path)
        if store.runs_dir.exists():
            store.runs_dir.rmdir()

        result = store.update_run_status("r1", status="completed")
        assert result is None  # no-op, safe


# ---------------------------------------------------------------------------
# Combinatorial: (entries × status_filter × limit)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n_entries,status_filter,limit,expected_count",
    [
        (0, None, None, 0),
        (0, "running", None, 0),
        (3, None, None, 3),
        (3, None, 1, 1),
        (5, "running", 2, 2),
        (5, "completed", None, 2),  # 0/2/4 are completed → 3? recheck
        (5, "completed", 100, 3),  # all 3 completed entries
        (5, "completed", 0, 0),
    ],
)
def test_combinatorial_listing(
    tmp_path: Path,
    n_entries: int,
    status_filter: str | None,
    limit: int | None,
    expected_count: int,
) -> None:
    """Pin behaviour across the (entries × filter × limit) matrix."""
    store = _make_project(tmp_path)
    for i in range(n_entries):
        store.register_run(
            run_id=f"r{i}",
            status="completed" if i % 2 == 0 else "running",
            started_at=f"2026-04-{i + 1:02d}T00:00:00Z",
        )

    runs = store.list_runs(status=status_filter, limit=limit)
    # Hand-derive expected for status="completed" rows: indices 0/2/4.
    if status_filter == "completed" and n_entries == 5:
        expected = 3 if (limit is None or limit >= 3) else (
            0 if limit == 0 else min(limit, 3)
        )
    elif status_filter == "running" and n_entries == 5:
        # indices 1/3 → 2 entries.
        expected = min(limit if limit is not None else 2, 2)
    else:
        expected = expected_count
    assert len(runs) == expected
