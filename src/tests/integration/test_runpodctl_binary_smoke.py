from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from src.providers.runpod.runpodctl_client import RunPodCtlClient, resolve_runpodctl_binary

pytestmark = [pytest.mark.integration]


def _local_binary() -> Path:
    return Path(__file__).resolve().parents[3] / "runpodctl"


def test_local_runpodctl_binary_exists_and_is_executable() -> None:
    binary = _local_binary()
    assert binary.exists(), f"Expected local runpodctl binary at {binary}"
    assert binary.is_file()
    assert binary.stat().st_size > 0


def test_runpodctl_client_resolves_repo_local_binary() -> None:
    resolved = Path(resolve_runpodctl_binary())
    assert resolved == _local_binary()
    client = RunPodCtlClient(api_key="rk-test")
    assert client.is_available() is True


def test_local_runpodctl_version_runs() -> None:
    result = subprocess.run(
        [str(_local_binary()), "version"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert "runpodctl" in (result.stdout or "").lower()


@pytest.mark.parametrize(
    "args,expected",
    [
        (["--help"], "runpodctl"),
        (["pod", "create", "--help"], "create a new pod"),
        (["pod", "get", "--help"], "get"),
        (["pod", "start", "--help"], "start"),
        (["pod", "stop", "--help"], "stop"),
        (["pod", "delete", "--help"], "delete"),
        (["send", "--help"], "send"),
        (["receive", "--help"], "receive"),
    ],
)
def test_local_runpodctl_help_contract(args: list[str], expected: str) -> None:
    result = subprocess.run(
        [str(_local_binary()), *args],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    combined = f"{result.stdout}\n{result.stderr}"
    assert expected.lower() in combined.lower()
