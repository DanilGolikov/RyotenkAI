"""Tests for ``POST /api/v1/files/upload`` (Phase 2 PR-2.4)."""

from __future__ import annotations

import hashlib
import io
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("RYOTENKAI_RUNTIME_PROVIDER", "single_node")

from ryotenkai_pod.runner.main import create_app


class _StubSupervisor:
    is_running = False

    async def shutdown(self) -> None:
        pass


def _factory(fsm, bus, *, terminal_hook=None, stdio_log_path=None):  # type: ignore[no-untyped-def]
    return _StubSupervisor()


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RYOTENKAI_WORKSPACE", str(tmp_path))
    with TestClient(create_app(supervisor_factory=_factory)) as c:
        yield c, tmp_path


# ---------------------------------------------------------------------------
# Positive
# ---------------------------------------------------------------------------


class TestPositive:
    def test_upload_config_round_trips_sha256(self, client) -> None:  # type: ignore[no-untyped-def]
        c, tmp_path = client
        body = b"some yaml content here\n" * 100
        sha = hashlib.sha256(body).hexdigest()
        r = c.post(
            "/api/v1/files/upload",
            data={"target": "config"},
            files={"file": ("pipeline_config.yaml", io.BytesIO(body), "application/yaml")},
        )
        assert r.status_code == 200, r.text
        body_resp = r.json()
        assert body_resp["target"] == "config"
        assert body_resp["bytes_written"] == len(body)
        assert body_resp["sha256"] == sha
        # Server-side file landed at the canonical path
        assert (tmp_path / "config" / "pipeline_config.yaml").read_bytes() == body
        # No leftover .partial
        assert not (tmp_path / "config" / "pipeline_config.yaml.partial").exists()

    def test_upload_dataset_uses_basename(self, client) -> None:  # type: ignore[no-untyped-def]
        c, tmp_path = client
        body = b'{"x": 1}\n'
        r = c.post(
            "/api/v1/files/upload",
            data={"target": "dataset"},
            files={"file": ("train.jsonl", io.BytesIO(body), "application/json")},
        )
        assert r.status_code == 200
        assert (tmp_path / "data" / "train.jsonl").read_bytes() == body


# ---------------------------------------------------------------------------
# Negative
# ---------------------------------------------------------------------------


class TestNegative:
    def test_unknown_target_returns_422(self, client) -> None:  # type: ignore[no-untyped-def]
        c, _ = client
        r = c.post(
            "/api/v1/files/upload",
            data={"target": "evil"},
            files={"file": ("x", io.BytesIO(b"x"), "application/octet-stream")},
        )
        assert r.status_code == 422
        body = r.json()
        # FastAPI's enum validator → universal handler → JOB_SPEC_INVALID.
        assert body["code"] in {"JOB_SPEC_INVALID", "FILE_TARGET_INVALID"}

    def test_path_traversal_filename_rejected(self, client) -> None:  # type: ignore[no-untyped-def]
        c, tmp_path = client
        r = c.post(
            "/api/v1/files/upload",
            data={"target": "dataset"},
            files={"file": ("../etc/passwd", io.BytesIO(b"x"), "text/plain")},
        )
        assert r.status_code == 422
        assert r.json()["code"] == "FILE_TARGET_INVALID"
        # Nothing written outside data/
        assert not any(p.name == "passwd" for p in tmp_path.rglob("*"))

    def test_oversize_returns_413_and_cleans_partial(self, client) -> None:  # type: ignore[no-untyped-def]
        c, tmp_path = client
        # Patch MAX_FILE_SIZE small for the test
        import ryotenkai_pod.runner.api.files as files_mod
        original = files_mod.MAX_FILE_SIZE
        files_mod.MAX_FILE_SIZE = 1024  # 1 KB
        try:
            body = b"a" * (10 * 1024)
            r = c.post(
                "/api/v1/files/upload",
                data={"target": "config"},
                files={"file": ("config.yaml", io.BytesIO(body), "application/yaml")},
            )
            assert r.status_code == 413
            assert r.json()["code"] == "FILE_TOO_LARGE"
            # Partial cleanup verified
            assert not (tmp_path / "config" / "pipeline_config.yaml.partial").exists()
            assert not (tmp_path / "config" / "pipeline_config.yaml").exists()
        finally:
            files_mod.MAX_FILE_SIZE = original


# ---------------------------------------------------------------------------
# Boundary
# ---------------------------------------------------------------------------


class TestBoundary:
    def test_zero_byte_upload(self, client) -> None:  # type: ignore[no-untyped-def]
        c, tmp_path = client
        r = c.post(
            "/api/v1/files/upload",
            data={"target": "config"},
            files={"file": ("config.yaml", io.BytesIO(b""), "application/yaml")},
        )
        assert r.status_code == 200
        assert r.json()["bytes_written"] == 0
        assert r.json()["sha256"] == hashlib.sha256(b"").hexdigest()
        assert (tmp_path / "config" / "pipeline_config.yaml").read_bytes() == b""

    def test_exact_max_size_succeeds(self, client) -> None:  # type: ignore[no-untyped-def]
        c, tmp_path = client
        import ryotenkai_pod.runner.api.files as files_mod
        original = files_mod.MAX_FILE_SIZE
        files_mod.MAX_FILE_SIZE = 1024
        try:
            body = b"x" * 1024
            r = c.post(
                "/api/v1/files/upload",
                data={"target": "config"},
                files={"file": ("config.yaml", io.BytesIO(body), "application/yaml")},
            )
            assert r.status_code == 200
            assert r.json()["bytes_written"] == 1024
        finally:
            files_mod.MAX_FILE_SIZE = original


# ---------------------------------------------------------------------------
# Logic-specific
# ---------------------------------------------------------------------------


class TestLogicSpecific:
    def test_atomic_rename_no_partial_on_success(self, client) -> None:  # type: ignore[no-untyped-def]
        c, tmp_path = client
        c.post(
            "/api/v1/files/upload",
            data={"target": "community-plugins-zip"},
            files={"file": ("plugins.zip", io.BytesIO(b"\x50\x4b\x03\x04"), "application/zip")},
        )
        community_dir = tmp_path / "community"
        # Final exists + .partial absent ⇒ atomic rename worked.
        assert (community_dir / "plugins.zip").exists()
        assert not list(community_dir.glob("*.partial"))


# ---------------------------------------------------------------------------
# Combinatorial: target × size × content-type
# ---------------------------------------------------------------------------


class TestCombinatorial:
    @pytest.mark.parametrize("target,filename", [
        ("config", "pipeline_config.yaml"),
        ("dataset", "train.jsonl"),
        ("community-plugins-zip", "plugins.zip"),
    ])
    @pytest.mark.parametrize("size", [0, 1, 1024])
    def test_matrix(
        self, client, target: str, filename: str, size: int,  # type: ignore[no-untyped-def]
    ) -> None:
        c, _ = client
        body = b"x" * size
        r = c.post(
            "/api/v1/files/upload",
            data={"target": target},
            files={"file": (filename, io.BytesIO(body), "application/octet-stream")},
        )
        assert r.status_code == 200, (target, size, r.text)
        assert r.json()["bytes_written"] == size
