from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.providers.runpod.file_transport import RunPodCtlFileTransport
from src.utils.result import Err, Ok, ProviderError

pytestmark = pytest.mark.unit


class _FakeClient:
    def __init__(self, send_result=None):
        self.send_result = send_result if send_result is not None else Ok(None)  # type: ignore[call-arg]
        self.calls: list[dict[str, Any]] = []

    def send(self, *, local_path: str, code: str):
        self.calls.append({"local_path": local_path, "code": code})
        return self.send_result


class _SSH:
    def __init__(self, responses: list[tuple[bool, str, str]]):
        self.responses = list(responses)
        self.commands: list[str] = []

    def exec_command(self, command: str, background: bool = False, timeout: int = 30, silent: bool = False):
        self.commands.append(command)
        if self.responses:
            return self.responses.pop(0)
        return True, "", ""


def test_upload_batch_happy_path(tmp_path: Path) -> None:
    local_file = tmp_path / "data.jsonl"
    local_file.write_text("{}\n")
    client = _FakeClient()
    transport = RunPodCtlFileTransport(client=client)
    ssh = _SSH(
        [
            (True, "READY\n", ""),
            (True, "", ""),
            (True, "STATUS=RUNNING\n", ""),
            (True, "", ""),
        ]
    )

    res = transport.upload_batch(
        ssh_client=ssh,
        workspace="/workspace/run1",
        pod_id="pod-123",
        files_to_upload=[(str(local_file), "data/sft/data.jsonl")],
        verify_timeout=30,
    )
    assert res.is_success()
    assert client.calls
    assert "runpodctl receive" in ssh.commands[1]


def test_upload_batch_rejects_absolute_remote_path(tmp_path: Path) -> None:
    local_file = tmp_path / "data.jsonl"
    local_file.write_text("{}\n")
    transport = RunPodCtlFileTransport(client=_FakeClient())
    ssh = _SSH([])

    res = transport.upload_batch(
        ssh_client=ssh,
        workspace="/workspace/run1",
        pod_id="pod-123",
        files_to_upload=[(str(local_file), "/abs/path.jsonl")],
        verify_timeout=30,
    )
    assert res.is_failure()
    assert res.unwrap_err().code == "INVALID_REMOTE_PATH"


def test_upload_batch_returns_no_files_to_upload_when_all_missing(tmp_path: Path) -> None:
    transport = RunPodCtlFileTransport(client=_FakeClient())
    ssh = _SSH([])
    res = transport.upload_batch(
        ssh_client=ssh,
        workspace="/workspace/run1",
        pod_id="pod-123",
        files_to_upload=[(str(tmp_path / "missing.jsonl"), "data/x.jsonl")],
        verify_timeout=30,
    )
    assert res.is_failure()
    assert res.unwrap_err().code == "NO_FILES_TO_UPLOAD"


def test_upload_batch_fails_when_remote_runpodctl_missing(tmp_path: Path) -> None:
    local_file = tmp_path / "data.jsonl"
    local_file.write_text("{}\n")
    transport = RunPodCtlFileTransport(client=_FakeClient())
    ssh = _SSH([(True, "MISSING\n", "")])
    res = transport.upload_batch(
        ssh_client=ssh,
        workspace="/workspace/run1",
        pod_id="pod-123",
        files_to_upload=[(str(local_file), "data/x.jsonl")],
        verify_timeout=30,
    )
    assert res.is_failure()
    assert res.unwrap_err().code == "RUNPODCTL_REMOTE_UNAVAILABLE"


def test_upload_batch_fails_when_prepare_receive_fails(tmp_path: Path) -> None:
    local_file = tmp_path / "data.jsonl"
    local_file.write_text("{}\n")
    transport = RunPodCtlFileTransport(client=_FakeClient())
    ssh = _SSH([(True, "READY\n", ""), (False, "", "mkdir failed")])
    res = transport.upload_batch(
        ssh_client=ssh,
        workspace="/workspace/run1",
        pod_id="pod-123",
        files_to_upload=[(str(local_file), "data/x.jsonl")],
        verify_timeout=30,
    )
    assert res.is_failure()
    assert res.unwrap_err().code == "RUNPODCTL_RECEIVE_PREPARE_FAILED"


def test_upload_batch_fails_when_send_fails(tmp_path: Path) -> None:
    local_file = tmp_path / "data.jsonl"
    local_file.write_text("{}\n")
    transport = RunPodCtlFileTransport(
        client=_FakeClient(send_result=Err(ProviderError(message="send failed", code="RUNPODCTL_COMMAND_FAILED")))
    )
    ssh = _SSH([(True, "READY\n", ""), (True, "", ""), (True, "STATUS=RUNNING\n", ""), (True, "remote log", "")])
    res = transport.upload_batch(
        ssh_client=ssh,
        workspace="/workspace/run1",
        pod_id="pod-123",
        files_to_upload=[(str(local_file), "data/x.jsonl")],
        verify_timeout=30,
    )
    assert res.is_failure()
    assert res.unwrap_err().code == "RUNPODCTL_COMMAND_FAILED"


def test_upload_batch_fails_when_extract_fails(tmp_path: Path) -> None:
    local_file = tmp_path / "data.jsonl"
    local_file.write_text("{}\n")
    transport = RunPodCtlFileTransport(client=_FakeClient())
    ssh = _SSH([(True, "READY\n", ""), (True, "", ""), (True, "STATUS=RUNNING\n", ""), (False, "", "tar failed")])
    res = transport.upload_batch(
        ssh_client=ssh,
        workspace="/workspace/run1",
        pod_id="pod-123",
        files_to_upload=[(str(local_file), "data/x.jsonl")],
        verify_timeout=30,
    )
    assert res.is_failure()
    assert res.unwrap_err().code == "RUNPODCTL_EXTRACT_FAILED"


def test_upload_batch_ignores_missing_files_but_sends_existing(tmp_path: Path) -> None:
    existing = tmp_path / "data.jsonl"
    existing.write_text("{}\n")
    missing = tmp_path / "missing.jsonl"
    client = _FakeClient()
    transport = RunPodCtlFileTransport(client=client)
    ssh = _SSH([(True, "READY\n", ""), (True, "", ""), (True, "STATUS=RUNNING\n", ""), (True, "", "")])

    res = transport.upload_batch(
        ssh_client=ssh,
        workspace="/workspace/run1",
        pod_id="pod-123",
        files_to_upload=[
            (str(missing), "data/missing.jsonl"),
            (str(existing), "data/existing.jsonl"),
        ],
        verify_timeout=30,
    )
    assert res.is_success()
    assert len(client.calls) == 1


def test_upload_batch_fails_when_remote_receive_exits_before_send(tmp_path: Path) -> None:
    local_file = tmp_path / "data.jsonl"
    local_file.write_text("{}\n")
    transport = RunPodCtlFileTransport(client=_FakeClient())
    ssh = _SSH([(True, "READY\n", ""), (True, "", ""), (True, "STATUS=EXITED\nbind failed\n", "")])

    res = transport.upload_batch(
        ssh_client=ssh,
        workspace="/workspace/run1",
        pod_id="pod-123",
        files_to_upload=[(str(local_file), "data/x.jsonl")],
        verify_timeout=30,
    )

    assert res.is_failure()
    assert res.unwrap_err().code == "RUNPODCTL_RECEIVE_NOT_READY"
