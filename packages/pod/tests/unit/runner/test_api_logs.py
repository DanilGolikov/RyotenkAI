"""Tests for ``GET /api/v1/logs/{name}`` (Phase 2 PR-2.3)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("RYOTENKAI_RUNTIME_PROVIDER", "single_node")

from ryotenkai_pod.runner.main import create_app  # noqa: E402


class _StubSupervisor:
    is_running = False

    async def shutdown(self) -> None:
        pass


def _factory(fsm, bus, *, terminal_hook=None, stdio_log_path=None):  # type: ignore[no-untyped-def]
    return _StubSupervisor()


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    """TestClient with the pod_layout pointed at ``tmp_path`` so we
    can write fake log files and observe the endpoint reading them.
    Uses ``with`` so the lifespan runs and ``app.state.pod_layout``
    is wired before any request fires."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RYOTENKAI_WORKSPACE", str(tmp_path))
    with TestClient(create_app(supervisor_factory=_factory)) as c:
        yield c


def _trainer_log_path(tmp_path: Path) -> Path:
    """Resolve the ``trainer.stdio.log`` path inside the test's
    workspace — same source-of-truth as the runner uses."""
    from pathlib import PurePosixPath
    from ryotenkai_shared.utils.pod_layout import PodLayout

    layout = PodLayout.from_root(PurePosixPath(str(tmp_path)))
    return Path(str(layout.trainer_stdio_log))


# ---------------------------------------------------------------------------
# Positive
# ---------------------------------------------------------------------------


class TestPositive:
    def test_first_chunk_returns_full_file(
        self, client: TestClient, tmp_path: Path,
    ) -> None:
        path = _trainer_log_path(tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("line1\nline2\nline3\n")
        r = client.get("/api/v1/logs/trainer_stdio")
        assert r.status_code == 200
        body = r.json()
        assert body["content"] == "line1\nline2\nline3\n"
        assert body["total_size"] == 18
        assert body["next_offset"] == 18
        assert body["truncated"] is False

    def test_offset_returns_tail_only(
        self, client: TestClient, tmp_path: Path,
    ) -> None:
        path = _trainer_log_path(tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("aaaaabbbbb")
        r = client.get("/api/v1/logs/trainer_stdio?offset=5")
        assert r.status_code == 200
        body = r.json()
        assert body["content"] == "bbbbb"
        assert body["next_offset"] == 10

    def test_size_endpoint(self, client: TestClient, tmp_path: Path) -> None:
        path = _trainer_log_path(tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("abc")
        r = client.get("/api/v1/logs/trainer_stdio/size")
        assert r.status_code == 200
        assert r.json() == {"size_bytes": 3}


# ---------------------------------------------------------------------------
# Negative
# ---------------------------------------------------------------------------


class TestNegative:
    def test_missing_file_returns_404(self, client: TestClient) -> None:
        r = client.get("/api/v1/logs/trainer_stdio")
        assert r.status_code == 404
        body = r.json()
        assert body["code"] == "LOG_NOT_AVAILABLE"

    def test_offset_out_of_range_returns_416(
        self, client: TestClient, tmp_path: Path,
    ) -> None:
        path = _trainer_log_path(tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("xy")
        r = client.get("/api/v1/logs/trainer_stdio?offset=100")
        assert r.status_code == 416
        body = r.json()
        assert body["code"] == "LOG_OFFSET_OUT_OF_RANGE"

    def test_invalid_name_returns_422(self, client: TestClient) -> None:
        # FastAPI validates StrEnum; universal handler maps to
        # JOB_SPEC_INVALID with errors[].
        r = client.get("/api/v1/logs/../etc/passwd")
        assert r.status_code in (404, 422)  # router 404s before validation


# ---------------------------------------------------------------------------
# Boundary
# ---------------------------------------------------------------------------


class TestBoundary:
    def test_empty_file(self, client: TestClient, tmp_path: Path) -> None:
        path = _trainer_log_path(tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")
        r = client.get("/api/v1/logs/trainer_stdio")
        assert r.status_code == 200
        body = r.json()
        assert body["content"] == ""
        assert body["total_size"] == 0
        assert body["next_offset"] == 0

    def test_offset_equals_total_size_returns_empty_chunk(
        self, client: TestClient, tmp_path: Path,
    ) -> None:
        path = _trainer_log_path(tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("abc")
        r = client.get("/api/v1/logs/trainer_stdio?offset=3")
        assert r.status_code == 200
        body = r.json()
        assert body["content"] == ""
        assert body["next_offset"] == 3

    def test_truncated_flag_set_when_more_available(
        self, client: TestClient, tmp_path: Path,
    ) -> None:
        path = _trainer_log_path(tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("a" * 10_000)
        r = client.get("/api/v1/logs/trainer_stdio?limit_bytes=100")
        assert r.status_code == 200
        body = r.json()
        assert body["truncated"] is True
        assert len(body["content"]) == 100
        assert body["next_offset"] == 100


# ---------------------------------------------------------------------------
# Logic-specific
# ---------------------------------------------------------------------------


class TestLogicSpecific:
    def test_invalid_utf8_does_not_crash(
        self, client: TestClient, tmp_path: Path,
    ) -> None:
        path = _trainer_log_path(tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\xff\xfe\xfd")
        r = client.get("/api/v1/logs/trainer_stdio")
        assert r.status_code == 200
        # Replaced bytes → U+FFFD per ``errors='replace'``.
        assert "�" in r.json()["content"]


# ---------------------------------------------------------------------------
# Combinatorial
# ---------------------------------------------------------------------------


class TestCombinatorial:
    @pytest.mark.parametrize("name", ["trainer_stdio", "runner"])
    @pytest.mark.parametrize("offset_state", ["zero", "mid", "exact"])
    def test_matrix(
        self,
        client: TestClient,
        tmp_path: Path,
        name: str,
        offset_state: str,
    ) -> None:
        from pathlib import PurePosixPath
        from ryotenkai_shared.utils.pod_layout import PodLayout

        layout = PodLayout.from_root(PurePosixPath(str(tmp_path)))
        path = (
            Path(str(layout.trainer_stdio_log))
            if name == "trainer_stdio"
            else Path(str(layout.runner_log))
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("hello world")

        offset = {"zero": 0, "mid": 5, "exact": 11}[offset_state]
        r = client.get(f"/api/v1/logs/{name}?offset={offset}")
        assert r.status_code == 200
