"""Container-aware RAM reading for HealthReporter.

``psutil.virtual_memory()`` reports HOST RAM inside a Docker container,
which on a shared GPU host (e.g. RunPod) gives us nonsense values like
"504 GB total" while our pod actually has a 32-GB cgroup limit. These
tests pin the cgroup-fallback behaviour: prefer container limit when
present, fall back to psutil only when no usable cgroup limit exists.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from ryotenkai_pod.runner import health_reporter as hr


def _patch_cgroup_paths(
    monkeypatch: pytest.MonkeyPatch,
    *,
    v2_max: Path | None = None,
    v2_current: Path | None = None,
    v1_limit: Path | None = None,
    v1_usage: Path | None = None,
) -> None:
    # Each pair must come together; missing paths point at /nonexistent so
    # OSError fires when read_text runs and the function falls through.
    nx = Path("/nonexistent/cgroup/path")
    monkeypatch.setattr(hr, "_CGROUP_V2_MEM_MAX", v2_max or nx)
    monkeypatch.setattr(hr, "_CGROUP_V2_MEM_CURRENT", v2_current or nx)
    monkeypatch.setattr(hr, "_CGROUP_V1_MEM_LIMIT", v1_limit or nx)
    monkeypatch.setattr(hr, "_CGROUP_V1_MEM_USAGE", v1_usage or nx)


class TestCgroupV2:
    def test_returns_used_and_limit_in_bytes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        v2_max = tmp_path / "memory.max"
        v2_current = tmp_path / "memory.current"
        # 32 GiB limit, 4 GiB used — typical RunPod-pod numbers.
        v2_max.write_text(str(32 * 1024**3))
        v2_current.write_text(str(4 * 1024**3))
        _patch_cgroup_paths(monkeypatch, v2_max=v2_max, v2_current=v2_current)

        result = hr._read_cgroup_memory()
        assert result == (4 * 1024**3, 32 * 1024**3)

    def test_max_sentinel_falls_through(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # cgroup v2 spells "no limit" as the literal string "max".
        v2_max = tmp_path / "memory.max"
        v2_max.write_text("max\n")
        _patch_cgroup_paths(monkeypatch, v2_max=v2_max)

        # No v1 fallback configured → returns None, caller falls back to psutil.
        assert hr._read_cgroup_memory() is None

    def test_huge_limit_treated_as_no_limit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Some kernels report a near-LLONG_MAX value rather than "max".
        v2_max = tmp_path / "memory.max"
        v2_current = tmp_path / "memory.current"
        v2_max.write_text(str(9223372036854771712))  # ≈ 8 EiB
        v2_current.write_text(str(1024**3))
        _patch_cgroup_paths(monkeypatch, v2_max=v2_max, v2_current=v2_current)

        assert hr._read_cgroup_memory() is None


class TestCgroupV1Fallback:
    def test_v1_used_when_v2_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        v1_limit = tmp_path / "memory.limit_in_bytes"
        v1_usage = tmp_path / "memory.usage_in_bytes"
        v1_limit.write_text(str(16 * 1024**3))
        v1_usage.write_text(str(2 * 1024**3))
        _patch_cgroup_paths(monkeypatch, v1_limit=v1_limit, v1_usage=v1_usage)

        result = hr._read_cgroup_memory()
        assert result == (2 * 1024**3, 16 * 1024**3)

    def test_v1_no_limit_sentinel_falls_through(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        v1_limit = tmp_path / "memory.limit_in_bytes"
        # Linux often prints this 8-EiB sentinel when no cgroup limit set.
        v1_limit.write_text("9223372036854771712")
        _patch_cgroup_paths(monkeypatch, v1_limit=v1_limit)

        assert hr._read_cgroup_memory() is None


class TestNoCgroupAvailable:
    def test_returns_none_when_no_files(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Mac dev environment: no /sys/fs/cgroup/* at all.
        _patch_cgroup_paths(monkeypatch)
        assert hr._read_cgroup_memory() is None

    def test_corrupt_file_falls_through(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        v2_max = tmp_path / "memory.max"
        v2_max.write_text("not-a-number\n")
        _patch_cgroup_paths(monkeypatch, v2_max=v2_max)
        assert hr._read_cgroup_memory() is None


class TestReadPsutilIntegration:
    def test_psutil_uses_cgroup_when_available(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import asyncio

        v2_max = tmp_path / "memory.max"
        v2_current = tmp_path / "memory.current"
        v2_max.write_text(str(32 * 1024**3))   # 32 GiB
        v2_current.write_text(str(8 * 1024**3))  # 8 GiB
        _patch_cgroup_paths(monkeypatch, v2_max=v2_max, v2_current=v2_current)

        cpu, used_gb, total_gb = asyncio.run(hr._read_psutil())
        assert total_gb == pytest.approx(32.0, rel=1e-9)
        assert used_gb == pytest.approx(8.0, rel=1e-9)

    def test_psutil_falls_back_to_host_when_no_cgroup(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import asyncio
        _patch_cgroup_paths(monkeypatch)  # no cgroup paths

        cpu, used_gb, total_gb = asyncio.run(hr._read_psutil())
        # Either real psutil values (running on Linux/Mac with psutil)
        # or None triple (no psutil). In either case it MUST NOT raise.
        assert (used_gb is None and total_gb is None) or (
            isinstance(used_gb, float) and isinstance(total_gb, float)
        )
