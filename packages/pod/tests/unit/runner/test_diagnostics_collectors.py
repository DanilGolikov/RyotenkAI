"""Unit tests for :mod:`ryotenkai_pod.runner.diagnostics.collectors`.

Test categories:

* positive       — happy path: dmesg / nvidia-smi return data, parsed.
* negative       — non-zero rc, malformed CSV.
* boundary       — empty kernel buffer, single GPU, max-line tail.
* invariant      — block error never crashes the call (returns sentinel).
* dependency-err — tool missing (``which`` returns None), permission denied.
* regression     — kernel-signal regex matches OOM/NVRM/XID/nvidia.
* logic-specific — truncation flag set when output exceeds cap.
* combinatorial  — (subprocess outcome) × (tool present/absent).
"""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock

import pytest

from ryotenkai_pod.runner.diagnostics.collectors import (
    DEFAULT_KERNEL_SIGNALS_TAIL_LINES,
    KERNEL_SIGNAL_PATTERN,
    _parse_nvidia_smi_csv,
    collect_dmesg,
    collect_kernel_signals,
    collect_nvidia_smi,
)
from ryotenkai_shared.contracts.runner_api.diagnostics import (
    DiagnosticsBlockError,
    DmesgReport,
    GpuReport,
    KernelSignalsReport,
)


def _completed(rc: int = 0, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=rc, stdout=stdout, stderr=stderr)


# ---------------------------------------------------------------------------
# dmesg
# ---------------------------------------------------------------------------


class TestDmesgPositive:
    def test_returns_lines_on_zero_rc(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import shutil
        monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/dmesg")
        runner = MagicMock(return_value=_completed(stdout="line1\nline2\nline3"))
        report = collect_dmesg(runner=runner)
        assert isinstance(report, DmesgReport)
        assert report.error is None
        assert report.lines == ["line1", "line2", "line3"]
        assert report.truncated is False


class TestDmesgNegative:
    def test_permission_denied_classified_via_stderr(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import shutil
        monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/dmesg")
        runner = MagicMock(return_value=_completed(
            rc=1, stderr="dmesg: read kernel buffer failed: Operation not permitted",
        ))
        report = collect_dmesg(runner=runner)
        assert report.error == DiagnosticsBlockError.PERMISSION_DENIED
        assert report.lines == []

    def test_nonzero_rc_without_perm_marker_falls_through_to_unknown(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import shutil
        monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/dmesg")
        runner = MagicMock(return_value=_completed(rc=1, stderr="some other failure"))
        assert collect_dmesg(runner=runner).error == DiagnosticsBlockError.UNKNOWN


class TestDmesgDependencyError:
    def test_tool_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import shutil
        monkeypatch.setattr(shutil, "which", lambda name: None)
        report = collect_dmesg(runner=MagicMock())
        assert report.error == DiagnosticsBlockError.TOOL_MISSING

    def test_subprocess_timeout_classified(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import shutil
        monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/dmesg")
        runner = MagicMock(side_effect=subprocess.TimeoutExpired(cmd="dmesg", timeout=10))
        assert collect_dmesg(runner=runner).error == DiagnosticsBlockError.TIMEOUT

    def test_permission_error_classified(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import shutil
        monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/dmesg")
        runner = MagicMock(side_effect=PermissionError("nope"))
        assert collect_dmesg(runner=runner).error == DiagnosticsBlockError.PERMISSION_DENIED


class TestDmesgBoundary:
    def test_empty_buffer(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import shutil
        monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/dmesg")
        runner = MagicMock(return_value=_completed(stdout=""))
        report = collect_dmesg(runner=runner)
        assert report.lines == []
        assert report.truncated is False

    def test_truncation_flag_set_when_over_cap(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import shutil
        monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/dmesg")
        big = "\n".join(f"line{i}" for i in range(500))
        runner = MagicMock(return_value=_completed(stdout=big))
        report = collect_dmesg(runner=runner, response_cap_lines=100)
        assert len(report.lines) == 100
        assert report.truncated is True


class TestDmesgInvariant:
    def test_always_returns_dmesgreport(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import shutil
        # Even with an arbitrary subprocess crash, return type holds.
        monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/dmesg")
        runner = MagicMock(side_effect=OSError("disk gone"))
        result = collect_dmesg(runner=runner)
        assert isinstance(result, DmesgReport)


# ---------------------------------------------------------------------------
# kernel signals
# ---------------------------------------------------------------------------


class TestKernelSignalsRegression:
    @pytest.mark.parametrize("line", [
        "Out of memory: Killed process 1234 (python)",       # OOM
        "NVRM: Xid (PCI:0000:00:1e): 31",                    # NVRM/XID
        "nvidia: probe of 0000:00:1e.0 failed",              # nvidia
        "Memory cgroup out of memory",                        # memory
    ])
    def test_pattern_matches_known_signals(self, line: str) -> None:
        assert KERNEL_SIGNAL_PATTERN.search(line) is not None

    def test_non_matching_lines_filtered(self) -> None:
        line = "kernel: random non-matching line"
        assert KERNEL_SIGNAL_PATTERN.search(line) is None


class TestKernelSignalsLogic:
    def test_filters_dmesg_in_process(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import shutil
        monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/dmesg")
        big = "\n".join([
            "boring line",
            "kernel: NVRM: Xid: 31",          # match
            "boring line 2",
            "Out of memory: Killed process",  # match
        ])
        runner = MagicMock(return_value=_completed(stdout=big))
        report = collect_kernel_signals(runner=runner)
        assert len(report.matches) == 2
        assert any("NVRM" in line for line in report.matches)
        assert any("Out of memory" in line for line in report.matches)


class TestKernelSignalsTruncation:
    def test_truncation_flag_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import shutil
        monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/dmesg")
        # Generate 100 OOM matches; tail to 30 → truncated.
        big = "\n".join(f"OOM: process {i}" for i in range(100))
        runner = MagicMock(return_value=_completed(stdout=big))
        report = collect_kernel_signals(runner=runner, tail_lines=30)
        assert len(report.matches) == 30
        assert report.truncated is True


class TestKernelSignalsErrorPropagation:
    def test_error_from_dmesg_propagates(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # If dmesg block has PERMISSION_DENIED, kernel_signals must
        # surface the same sentinel — same root cause.
        import shutil
        monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/dmesg")
        runner = MagicMock(return_value=_completed(
            rc=1, stderr="Operation not permitted",
        ))
        report = collect_kernel_signals(runner=runner)
        assert report.error == DiagnosticsBlockError.PERMISSION_DENIED


# ---------------------------------------------------------------------------
# nvidia-smi
# ---------------------------------------------------------------------------


class TestNvidiaSmiPositive:
    def test_single_gpu_parsed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import shutil
        monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/nvidia-smi")
        csv = "NVIDIA RTX 4090, 78 %, 12345 MiB, 24576 MiB"
        runner = MagicMock(return_value=_completed(stdout=csv))
        report = collect_nvidia_smi(runner=runner)
        assert report.error is None
        assert len(report.rows) == 1
        row = report.rows[0]
        assert row.name == "NVIDIA RTX 4090"
        assert row.utilization_gpu_percent == 78
        assert row.memory_used_mib == 12345
        assert row.memory_total_mib == 24576

    def test_multi_gpu_parsed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import shutil
        monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/nvidia-smi")
        csv = (
            "RTX 4090, 0 %, 0 MiB, 24576 MiB\n"
            "RTX 4090, 100 %, 24000 MiB, 24576 MiB"
        )
        runner = MagicMock(return_value=_completed(stdout=csv))
        report = collect_nvidia_smi(runner=runner)
        assert len(report.rows) == 2


class TestNvidiaSmiNegative:
    def test_malformed_lines_are_skipped(self) -> None:
        # Pure-function test of parser
        rows = _parse_nvidia_smi_csv("garbage\nRTX, 50 %, 10 MiB, 20 MiB")
        assert len(rows) == 1
        assert rows[0].name == "RTX"

    def test_negative_utilization_skipped(self) -> None:
        # Pydantic ge=0 rejects negative util — row dropped.
        rows = _parse_nvidia_smi_csv("RTX, -10 %, 10 MiB, 20 MiB")
        assert rows == []


class TestNvidiaSmiDependencyError:
    def test_tool_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import shutil
        monkeypatch.setattr(shutil, "which", lambda name: None)
        report = collect_nvidia_smi(runner=MagicMock())
        assert report.error == DiagnosticsBlockError.TOOL_MISSING


class TestNvidiaSmiBoundary:
    def test_empty_output_no_rows(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import shutil
        monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/nvidia-smi")
        runner = MagicMock(return_value=_completed(stdout=""))
        report = collect_nvidia_smi(runner=runner)
        assert report.error is None
        assert report.rows == []


# ---------------------------------------------------------------------------
# Combinatorial — tool × outcome matrix
# ---------------------------------------------------------------------------


class TestCombinatorial:
    @pytest.mark.parametrize("tool_present,subprocess_outcome,expected_error", [
        (False, "n/a", DiagnosticsBlockError.TOOL_MISSING),
        (True, "rc=0", None),
        (True, "rc=1", DiagnosticsBlockError.UNKNOWN),
        (True, "timeout", DiagnosticsBlockError.TIMEOUT),
        (True, "permission_error", DiagnosticsBlockError.PERMISSION_DENIED),
    ])
    def test_dmesg_outcomes(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tool_present: bool,
        subprocess_outcome: str,
        expected_error: DiagnosticsBlockError | None,
    ) -> None:
        import shutil
        monkeypatch.setattr(
            shutil, "which",
            lambda name: "/usr/bin/dmesg" if tool_present else None,
        )
        if not tool_present:
            runner = MagicMock()
        elif subprocess_outcome == "rc=0":
            runner = MagicMock(return_value=_completed(rc=0, stdout="line"))
        elif subprocess_outcome == "rc=1":
            runner = MagicMock(return_value=_completed(rc=1, stderr="other failure"))
        elif subprocess_outcome == "timeout":
            runner = MagicMock(side_effect=subprocess.TimeoutExpired(cmd="dmesg", timeout=10))
        elif subprocess_outcome == "permission_error":
            runner = MagicMock(side_effect=PermissionError("CAP_SYSLOG missing"))
        else:
            raise AssertionError(f"unhandled outcome: {subprocess_outcome}")
        report = collect_dmesg(runner=runner)
        assert report.error == expected_error
