"""REST endpoint /api/v1/runs/<id>/attempts/<n>/events.

Covers cold-replay reading of ``events_mirror.jsonl`` written by the
pipeline's :class:`TrainingMonitor`. Focused on file-reading
correctness; live-runner subscription is tested in the WS endpoint
suite.

Categories (project policy):
* Positive — basic read, since filter, kind filter, limit
* Negative — run not found, attempt not found, invalid query params
* Boundary — empty file, missing file, since beyond max offset,
  large limit clamp, single-event file
* Invariant — events length ≤ limit; next_since strictly > seen offsets
* Dependency-error — corrupted JSONL line skipped, malformed file
* Regression — read-only when run is past terminal (no SSH triggered)
* Logic-specific — kind filter combined with since
* Combinatorial — (since × limit × kind) matrix
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.config import ApiSettings
from src.api.dependencies import get_settings
from src.api.routers import run_events as run_events_router
from src.pipeline.stages.managers.event_mirror import EventMirrorWriter

# ---------------------------------------------------------------------------
# Fixtures + helpers
# ---------------------------------------------------------------------------


def _make_event(offset: int, kind: str = "trainer_log", **payload: object) -> dict:
    return {
        "v": 1,
        "offset": offset,
        "ts": "2026-04-30T00:00:00Z",
        "kind": kind,
        "payload": payload,
    }


def _seed_mirror(
    settings: ApiSettings,
    run_id: str,
    attempt_no: int,
    events: list[dict],
) -> Path:
    """Create attempt_dir, write events to events_mirror.jsonl. Returns path."""
    run_dir = settings.runs_dir / run_id
    attempt_dir = run_dir / "attempts" / f"attempt_{attempt_no}"
    attempt_dir.mkdir(parents=True)
    if events:
        with EventMirrorWriter(attempt_dir) as mirror:
            for ev in events:
                mirror.write(ev)
    return attempt_dir


@pytest.fixture
def settings(tmp_path: Path) -> ApiSettings:
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    projects_root = tmp_path / "projects"
    projects_root.mkdir()
    return ApiSettings(
        runs_dir=runs_dir,
        projects_root=projects_root,
        serve_spa=False,
        cors_origins=["http://localhost:5173"],
    )


@pytest.fixture
def client(settings: ApiSettings) -> Iterator[TestClient]:
    app = FastAPI()
    app.include_router(run_events_router.router, prefix="/api/v1")
    app.dependency_overrides[get_settings] = lambda: settings
    with TestClient(app) as tc:
        yield tc


# ---------------------------------------------------------------------------
# Positive
# ---------------------------------------------------------------------------


def test_get_events_returns_all_when_no_since(
    client: TestClient, settings: ApiSettings,
) -> None:
    run_id = "run-x"
    events = [_make_event(i, line=f"l{i}") for i in range(5)]
    _seed_mirror(settings, run_id, 1, events)

    response = client.get(f"/api/v1/runs/{run_id}/attempts/1/events")
    assert response.status_code == 200, response.text
    body = response.json()
    assert len(body["events"]) == 5
    assert body["next_since"] == 5  # max offset 4 + 1


def test_get_events_since_filter(
    client: TestClient, settings: ApiSettings,
) -> None:
    run_id = "run-x"
    events = [_make_event(i) for i in range(10)]
    _seed_mirror(settings, run_id, 1, events)

    response = client.get(
        f"/api/v1/runs/{run_id}/attempts/1/events?since=5",
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body["events"]) == 5
    assert [e["offset"] for e in body["events"]] == [5, 6, 7, 8, 9]


def test_get_events_kind_filter(
    client: TestClient, settings: ApiSettings,
) -> None:
    run_id = "run-x"
    events = [
        _make_event(0, kind="trainer_log"),
        _make_event(1, kind="health_snapshot"),
        _make_event(2, kind="trainer_log"),
        _make_event(3, kind="trainer_exited"),
    ]
    _seed_mirror(settings, run_id, 1, events)

    response = client.get(
        f"/api/v1/runs/{run_id}/attempts/1/events?kind=trainer_log",
    )
    assert response.status_code == 200
    body = response.json()
    kinds = [e["kind"] for e in body["events"]]
    assert kinds == ["trainer_log", "trainer_log"]


def test_get_events_limit_pagination(
    client: TestClient, settings: ApiSettings,
) -> None:
    run_id = "run-x"
    events = [_make_event(i) for i in range(20)]
    _seed_mirror(settings, run_id, 1, events)

    page1 = client.get(
        f"/api/v1/runs/{run_id}/attempts/1/events?since=0&limit=10",
    ).json()
    assert len(page1["events"]) == 10
    assert page1["next_since"] == 10

    page2 = client.get(
        f"/api/v1/runs/{run_id}/attempts/1/events"
        f"?since={page1['next_since']}&limit=10",
    ).json()
    assert len(page2["events"]) == 10
    assert [e["offset"] for e in page2["events"]] == list(range(10, 20))


# ---------------------------------------------------------------------------
# Negative
# ---------------------------------------------------------------------------


def test_run_not_found_returns_404(client: TestClient) -> None:
    response = client.get("/api/v1/runs/nope/attempts/1/events")
    assert response.status_code == 404


def test_attempt_not_found_returns_404(
    client: TestClient, settings: ApiSettings,
) -> None:
    run_id = "run-x"
    (settings.runs_dir / run_id).mkdir()
    response = client.get(f"/api/v1/runs/{run_id}/attempts/99/events")
    assert response.status_code == 404


def test_invalid_since_returns_422(
    client: TestClient, settings: ApiSettings,
) -> None:
    run_id = "run-x"
    _seed_mirror(settings, run_id, 1, [])
    response = client.get(
        f"/api/v1/runs/{run_id}/attempts/1/events?since=-1",
    )
    assert response.status_code == 422


def test_invalid_limit_returns_422(
    client: TestClient, settings: ApiSettings,
) -> None:
    run_id = "run-x"
    _seed_mirror(settings, run_id, 1, [])
    response = client.get(
        f"/api/v1/runs/{run_id}/attempts/1/events?limit=0",
    )
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Boundary
# ---------------------------------------------------------------------------


def test_empty_mirror_returns_empty_list(
    client: TestClient, settings: ApiSettings,
) -> None:
    """Attempt dir exists, no events yet — endpoint returns 200 with
    empty events list. UI polls before any event arrives."""
    run_id = "run-x"
    _seed_mirror(settings, run_id, 1, [])  # empty events
    response = client.get(f"/api/v1/runs/{run_id}/attempts/1/events")
    assert response.status_code == 200
    body = response.json()
    assert body["events"] == []
    assert body["next_since"] == 0


def test_missing_mirror_file_returns_empty_list(
    client: TestClient, settings: ApiSettings,
) -> None:
    """Attempt exists but events/ subdir was never created — same
    behaviour as empty mirror, no exception."""
    run_id = "run-x"
    run_dir = settings.runs_dir / run_id
    (run_dir / "attempts" / "attempt_1").mkdir(parents=True)
    response = client.get(f"/api/v1/runs/{run_id}/attempts/1/events")
    assert response.status_code == 200
    assert response.json()["events"] == []


def test_since_beyond_max_offset_returns_empty(
    client: TestClient, settings: ApiSettings,
) -> None:
    run_id = "run-x"
    _seed_mirror(settings, run_id, 1, [_make_event(i) for i in range(5)])
    response = client.get(
        f"/api/v1/runs/{run_id}/attempts/1/events?since=100",
    )
    assert response.status_code == 200
    body = response.json()
    assert body["events"] == []
    # next_since is monotonic — never goes backwards even if no match.
    assert body["next_since"] == 100


# ---------------------------------------------------------------------------
# Invariant
# ---------------------------------------------------------------------------


def test_events_length_does_not_exceed_limit(
    client: TestClient, settings: ApiSettings,
) -> None:
    run_id = "run-x"
    _seed_mirror(settings, run_id, 1, [_make_event(i) for i in range(50)])
    response = client.get(
        f"/api/v1/runs/{run_id}/attempts/1/events?limit=7",
    )
    assert len(response.json()["events"]) == 7


def test_next_since_is_monotonic(
    client: TestClient, settings: ApiSettings,
) -> None:
    run_id = "run-x"
    _seed_mirror(settings, run_id, 1, [_make_event(i) for i in range(10)])
    response = client.get(
        f"/api/v1/runs/{run_id}/attempts/1/events?since=3",
    )
    body = response.json()
    # next_since must be > since when events returned
    assert body["next_since"] > 3
    # next_since must equal the max offset + 1
    assert body["next_since"] == max(e["offset"] for e in body["events"]) + 1


# ---------------------------------------------------------------------------
# Dependency-error — malformed mirror handling
# ---------------------------------------------------------------------------


def test_corrupted_jsonl_line_skipped(
    client: TestClient, settings: ApiSettings,
) -> None:
    """A single broken line in the mirror must NOT 500 the endpoint."""
    run_id = "run-x"
    attempt_dir = _seed_mirror(
        settings, run_id, 1, [_make_event(0), _make_event(1)],
    )
    mirror_path = (
        attempt_dir
        / EventMirrorWriter.EVENTS_DIR_NAME
        / EventMirrorWriter.MIRROR_FILE_NAME
    )
    # Inject an obviously broken line in the middle.
    with mirror_path.open("a", encoding="utf-8") as fp:
        fp.write("{not json\n")
        json.dump(_make_event(2), fp)
        fp.write("\n")

    response = client.get(f"/api/v1/runs/{run_id}/attempts/1/events")
    assert response.status_code == 200
    offsets = [e["offset"] for e in response.json()["events"]]
    assert offsets == [0, 1, 2]


def test_event_missing_offset_skipped(
    client: TestClient, settings: ApiSettings,
) -> None:
    """An event without an integer offset is unusable for cursor
    arithmetic; skip it rather than break pagination."""
    run_id = "run-x"
    attempt_dir = _seed_mirror(settings, run_id, 1, [_make_event(0)])
    mirror_path = (
        attempt_dir
        / EventMirrorWriter.EVENTS_DIR_NAME
        / EventMirrorWriter.MIRROR_FILE_NAME
    )
    with mirror_path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps({"v": 1, "kind": "x", "payload": {}}) + "\n")
    response = client.get(f"/api/v1/runs/{run_id}/attempts/1/events")
    assert response.status_code == 200
    assert len(response.json()["events"]) == 1


# ---------------------------------------------------------------------------
# Logic-specific
# ---------------------------------------------------------------------------


def test_kind_filter_combined_with_since(
    client: TestClient, settings: ApiSettings,
) -> None:
    run_id = "run-x"
    events = [
        _make_event(0, kind="trainer_log"),
        _make_event(1, kind="trainer_log"),
        _make_event(2, kind="health_snapshot"),
        _make_event(3, kind="trainer_log"),
    ]
    _seed_mirror(settings, run_id, 1, events)

    response = client.get(
        f"/api/v1/runs/{run_id}/attempts/1/events?since=2&kind=trainer_log",
    )
    assert response.status_code == 200
    body = response.json()
    assert [e["offset"] for e in body["events"]] == [3]


# ---------------------------------------------------------------------------
# Combinatorial
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("since", [0, 5, 10])
@pytest.mark.parametrize("limit", [1, 3, 100])
@pytest.mark.parametrize("kind", [None, "trainer_log", "missing"])
def test_combinatorial_query(
    client: TestClient,
    settings: ApiSettings,
    since: int,
    limit: int,
    kind: str | None,
) -> None:
    run_id = f"run-x-{since}-{limit}-{kind}"
    events = [
        _make_event(i, kind="trainer_log" if i % 2 == 0 else "health_snapshot")
        for i in range(10)
    ]
    _seed_mirror(settings, run_id, 1, events)

    url = f"/api/v1/runs/{run_id}/attempts/1/events?since={since}&limit={limit}"
    if kind is not None:
        url += f"&kind={kind}"
    response = client.get(url)
    assert response.status_code == 200
    body = response.json()
    assert len(body["events"]) <= limit
    # All returned events satisfy since + kind constraints
    for ev in body["events"]:
        assert ev["offset"] >= since
        if kind is not None:
            assert ev["kind"] == kind
