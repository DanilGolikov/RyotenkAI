"""Unit tests for telemetry plugin via pytester sub-session."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest

pytest_plugins = ["pytester"]


def test_telemetry_writes_jsonl_per_test(pytester: pytest.Pytester) -> None:
    repo_root = str(__import__("pathlib").Path(__file__).resolve().parents[2])
    pytester.makepyfile(
        conftest=f"""
import sys
sys.path.insert(0, {repo_root!r})
from pathlib import Path
from tests._harness import telemetry


def pytest_configure(config):
    out = Path(config.rootpath) / "tests" / ".telemetry"
    telemetry.register(config, output_dir=out)


def pytest_runtest_makereport(item, call):
    telemetry.pytest_runtest_makereport(item, call)
"""
    )
    pytester.makepyfile(
        test_demo="""
import pytest


@pytest.mark.uses_fake("FakeMLflowManager")
@pytest.mark.exercises_protocol("IMLflowManager")
def test_pass():
    assert True


def test_fail():
    assert False
"""
    )
    pytester.makefile(
        ".ini",
        pytest=(
            "[pytest]\n"
            "markers =\n"
            "    uses_fake(name): canonical fake exercised\n"
            "    exercises_protocol(name): protocol exercised\n"
        ),
    )
    result = pytester.runpytest("-p", "no:cacheprovider", "--import-mode=importlib")
    assert result.ret != 0, result.outlines  # one test fails on purpose

    telemetry_dir = pytester.path / "tests" / ".telemetry"
    assert telemetry_dir.exists(), "telemetry dir not created"
    files = list(telemetry_dir.glob("run-*.jsonl"))
    assert len(files) == 1, files
    lines = [json.loads(line) for line in files[0].read_text().splitlines()]
    by_id = {row["test_id"]: row for row in lines}
    assert any("test_pass" in k for k in by_id), by_id
    pass_row = next(v for k, v in by_id.items() if "test_pass" in k)
    fail_row = next(v for k, v in by_id.items() if "test_fail" in k)
    assert pass_row["outcome"] == "passed"
    assert pass_row["fakes_used"] == ["FakeMLflowManager"]
    assert pass_row["protocols_exercised"] == ["IMLflowManager"]
    assert fail_row["outcome"] == "failed"
    assert "error" in fail_row
