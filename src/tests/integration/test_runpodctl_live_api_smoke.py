from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.requires_internet]


def _local_binary() -> Path:
    return Path(__file__).resolve().parents[3] / "runpodctl"


@pytest.mark.skipif(not os.environ.get("RUNPOD_API_KEY"), reason="RUNPOD_API_KEY is required for live RunPod smoke")
def test_runpodctl_live_get_pod_returns_json() -> None:
    result = subprocess.run(
        [str(_local_binary()), "get", "pod", "--output", "json"],
        capture_output=True,
        text=True,
        timeout=60,
        env=os.environ.copy(),
    )
    assert result.returncode == 0, result.stderr or result.stdout
    payload = json.loads(result.stdout or "[]")
    assert isinstance(payload, (list, dict))
