from __future__ import annotations

import io
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import src.utils.ssh_client as ssh_mod
from src.utils.ssh_client import SSHClient, _mask_secrets


class _RunResult(SimpleNamespace):
    returncode: int
    stdout: str
    stderr: str


class _FakeProc:
    """Minimal subprocess.Popen stand-in for download_directory tests."""

    def __init__(self, returncode: int = 0, always_timeout: bool = False, stderr_text: str = "") -> None:
        self.returncode = returncode
        self._always_timeout = always_timeout
        self.stderr = io.StringIO(stderr_text)

    def wait(self, timeout: float | None = None) -> int:
        if self._always_timeout and timeout is not None:
            raise subprocess.TimeoutExpired(cmd="x", timeout=timeout)
        return self.returncode

    def kill(self) -> None:
        pass


def test_mask_secrets_redacts_tokens() -> None:
    text = 'HF_TOKEN="hf_abcdef" and also hf_12345 and API_KEY="secret"'
    masked = _mask_secrets(text)
    assert "hf_abcdef" not in masked
    assert "secret" not in masked
    assert "hf_***" in masked


def test_build_ssh_cmd_alias_mode() -> None:
    c = SSHClient(host="pc", port=22, username=None)
    cmd = c._build_ssh_cmd("echo hi")
    assert "ssh " in cmd
    assert " pc " in cmd or cmd.endswith(" pc 'echo hi'")
    assert "-p 22" not in cmd
    assert " -i " not in cmd


def test_build_ssh_cmd_explicit_mode() -> None:
    c = SSHClient(host="1.2.3.4", port=2222, username="root", key_path="/k")
    cmd = c._build_ssh_cmd("echo hi")
    assert "-p 2222" in cmd
    assert '-i "/k"' in cmd
    assert "root@1.2.3.4" in cmd


def test_test_connection_success_first_try(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_run(cmd, **kwargs):
        calls.append(str(cmd))
        return _RunResult(returncode=0, stdout="SSH OK\n", stderr="")

    monkeypatch.setattr(ssh_mod.subprocess, "run", fake_run)

    c = SSHClient(host="pc", username=None)
    ok, err = c.test_connection(max_retries=3, retry_delay=0)
    assert ok is True
    assert err == ""
    assert calls


def test_test_connection_retries_and_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = {"n": 0}

    def fake_run(cmd, **kwargs):
        attempts["n"] += 1
        return _RunResult(returncode=1, stdout="", stderr="no")

    monkeypatch.setattr(ssh_mod.subprocess, "run", fake_run)
    monkeypatch.setattr(ssh_mod.time, "sleep", lambda s: None)

    c = SSHClient(host="pc", username=None)
    ok, err = c.test_connection(max_retries=3, retry_delay=1)
    assert ok is False
    assert "after 3 attempts" in err
    assert attempts["n"] == 3


def test_exec_command_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*a, **k):
        raise ssh_mod.subprocess.TimeoutExpired(cmd="x", timeout=1)

    monkeypatch.setattr(ssh_mod.subprocess, "run", fake_run)

    c = SSHClient(host="pc", username=None)
    ok, out, err = c.exec_command("HF_TOKEN=hf_abc", timeout=1)
    assert ok is False
    assert out == ""
    assert "Timeout" in err


def test_get_file_content_file_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_exec(cmd, **kwargs):
        return True, "FILE_NOT_FOUND\n", ""

    c = SSHClient(host="pc", username=None)
    monkeypatch.setattr(c, "exec_command", fake_exec)

    ok, content = c.get_file_content("/missing.txt")
    assert ok is False
    assert "File not found" in content


def test_upload_file_success_without_verify(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    local = tmp_path / "x.txt"
    local.write_text("hi", encoding="utf-8")

    def fake_run(cmd, **kwargs):
        return _RunResult(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(ssh_mod.subprocess, "run", fake_run)

    c = SSHClient(host="pc", username=None)
    ok, err = c.upload_file(str(local), "/remote/x.txt", verify=False)
    assert ok is True
    assert err == ""


def test_download_directory_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(ssh_mod.subprocess, "Popen", lambda *a, **k: _FakeProc(returncode=0))

    c = SSHClient(host="pc", username=None)
    res = c.download_directory("/remote/dir", tmp_path / "out")
    assert res.is_success()


def test_ssh_target_property() -> None:
    alias = SSHClient(host="pc", username=None)
    assert alias.ssh_target == "pc"
    explicit = SSHClient(host="1.2.3.4", port=22, username="root", key_path="/k")
    assert explicit.ssh_target == "root@1.2.3.4"


def test_build_ssh_cmd_background_adds_nohup() -> None:
    c = SSHClient(host="pc", username=None)
    cmd = c._build_ssh_cmd("echo hi", background=True)
    assert "nohup" in cmd
    assert cmd.endswith("&'")


def test_find_ssh_key_autodetects_when_explicit_mode(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Create fake ~/.ssh/id_ed25519
    ssh_dir = tmp_path / ".ssh"
    ssh_dir.mkdir()
    key = ssh_dir / "id_ed25519"
    key.write_text("k", encoding="utf-8")

    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    c = SSHClient(host="1.2.3.4", port=22, username="root", key_path=None)
    assert c.key_path == str(key)


def test_find_ssh_key_raises_when_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / ".ssh").mkdir()
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    with pytest.raises(FileNotFoundError):
        SSHClient(host="1.2.3.4", port=22, username="root", key_path=None)


def test_build_scp_cmd_explicit_mode_includes_port_and_key() -> None:
    c = SSHClient(host="1.2.3.4", port=2222, username="root", key_path="/k")
    cmd = c._build_scp_cmd("a.txt", "/remote/a.txt")
    assert cmd[:3] == ["scp", "-P", "2222"]
    assert "-i" in cmd and "/k" in cmd
    assert cmd[-1] == "root@1.2.3.4:/remote/a.txt"


def test_test_connection_timeout_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*a, **k):
        raise ssh_mod.subprocess.TimeoutExpired(cmd="x", timeout=1)

    monkeypatch.setattr(ssh_mod.subprocess, "run", fake_run)
    monkeypatch.setattr(ssh_mod.time, "sleep", lambda s: None)

    c = SSHClient(host="pc", username=None)
    ok, err = c.test_connection(max_retries=2, retry_delay=0)
    assert ok is False
    assert "after 2 attempts" in err


def test_test_connection_exception_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(ssh_mod.subprocess, "run", fake_run)
    monkeypatch.setattr(ssh_mod.time, "sleep", lambda s: None)

    c = SSHClient(host="pc", username=None)
    ok, err = c.test_connection(max_retries=1, retry_delay=0)
    assert ok is False
    assert "after 1 attempts" in err


def test_upload_file_missing_local_returns_false(tmp_path: Path) -> None:
    c = SSHClient(host="pc", username=None)
    ok, err = c.upload_file(str(tmp_path / "missing.txt"), "/remote/x.txt", verify=False)
    assert ok is False
    assert "Local file not found" in err


def test_upload_file_scp_failure_returncode(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    local = tmp_path / "x.txt"
    local.write_text("hi", encoding="utf-8")

    def fake_run(cmd, **kwargs):
        return _RunResult(returncode=1, stdout="", stderr="no")

    monkeypatch.setattr(ssh_mod.subprocess, "run", fake_run)
    c = SSHClient(host="pc", username=None)
    ok, err = c.upload_file(str(local), "/remote/x.txt", verify=False)
    assert ok is False
    assert "SCP error" in err


def test_upload_file_verify_fails_when_remote_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    local = tmp_path / "x.txt"
    local.write_text("hi", encoding="utf-8")

    def fake_run(cmd, **kwargs):
        return _RunResult(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(ssh_mod.subprocess, "run", fake_run)
    c = SSHClient(host="pc", username=None)
    monkeypatch.setattr(c, "file_exists", lambda p: False)  # noqa: ARG005
    ok, err = c.upload_file(str(local), "/remote/x.txt", verify=True)
    assert ok is False
    assert "after upload" in err


def test_upload_file_timeout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    local = tmp_path / "x.txt"
    local.write_text("hi", encoding="utf-8")

    def fake_run(cmd, **kwargs):
        raise ssh_mod.subprocess.TimeoutExpired(cmd="x", timeout=1)

    monkeypatch.setattr(ssh_mod.subprocess, "run", fake_run)
    c = SSHClient(host="pc", username=None)
    ok, err = c.upload_file(str(local), "/remote/x.txt", verify=False)
    assert ok is False
    assert "timeout" in err.lower()


def test_upload_file_exception(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    local = tmp_path / "x.txt"
    local.write_text("hi", encoding="utf-8")

    def fake_run(cmd, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(ssh_mod.subprocess, "run", fake_run)
    c = SSHClient(host="pc", username=None)
    ok, err = c.upload_file(str(local), "/remote/x.txt", verify=False)
    assert ok is False
    assert "Upload error" in err


def test_exec_command_success_and_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    # success branch
    monkeypatch.setattr(
        ssh_mod.subprocess,
        "run",
        lambda *a, **k: _RunResult(returncode=0, stdout="ok\n", stderr=""),
    )
    c = SSHClient(host="pc", username=None)
    ok, out, err = c.exec_command('HF_TOKEN="hf_abc"', timeout=1, silent=False)
    assert ok is True
    assert out == "ok\n"
    assert err == ""

    # failure branch
    monkeypatch.setattr(
        ssh_mod.subprocess,
        "run",
        lambda *a, **k: _RunResult(returncode=2, stdout="", stderr="no"),
    )
    ok, out, err = c.exec_command("echo hi", timeout=1, silent=False)
    assert ok is False
    assert err == "no"


def test_exec_command_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ssh_mod.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    c = SSHClient(host="pc", username=None)
    ok, out, err = c.exec_command("echo hi", timeout=1, silent=False)
    assert ok is False
    assert out == ""
    assert "boom" in err


def test_file_and_directory_exists_and_create_directory(monkeypatch: pytest.MonkeyPatch) -> None:
    c = SSHClient(host="pc", username=None)
    monkeypatch.setattr(c, "exec_command", lambda *a, **k: (True, "EXISTS\n", ""))
    assert c.file_exists("/x") is True
    assert c.directory_exists("/d") is True

    monkeypatch.setattr(c, "exec_command", lambda *a, **k: (False, "", "err"))
    assert c.file_exists("/x") is False
    assert c.directory_exists("/d") is False
    ok, err = c.create_directory("/d")
    assert ok is False
    assert "Failed to create directory" in err

    # success branch
    monkeypatch.setattr(c, "exec_command", lambda *a, **k: (True, "", ""))
    ok, err = c.create_directory("/d")
    assert ok is True
    assert err == ""


def test_download_directory_failure_and_exceptions(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    c = SSHClient(host="pc", username=None)

    # returncode != 0 → SSH_DOWNLOAD_FAILED
    monkeypatch.setattr(ssh_mod.subprocess, "Popen", lambda *a, **k: _FakeProc(returncode=1, stderr_text="no"))
    res = c.download_directory("/remote/dir", tmp_path / "out")
    assert res.is_failure()
    assert res.unwrap_err().code == "SSH_DOWNLOAD_FAILED"

    # Stall detected across all retries → SSH_DOWNLOAD_STALLED
    monkeypatch.setattr(ssh_mod.subprocess, "Popen", lambda *a, **k: _FakeProc(always_timeout=True))
    res2 = c.download_directory(
        "/remote/dir", tmp_path / "out2",
        stall_timeout=1,
        max_retries=2,
    )
    assert res2.is_failure()
    assert res2.unwrap_err().code == "SSH_DOWNLOAD_STALLED"

    # OSError when spawning Popen → SSH_DOWNLOAD_IO_ERROR
    monkeypatch.setattr(ssh_mod.subprocess, "Popen", lambda *a, **k: (_ for _ in ()).throw(OSError("boom")))
    assert c.download_directory("/remote/dir", tmp_path / "out3").is_failure()


def test_get_file_content_tail_lines(monkeypatch: pytest.MonkeyPatch) -> None:
    c = SSHClient(host="pc", username=None)
    called: dict[str, str] = {}

    def fake_exec(cmd, **kwargs):
        called["cmd"] = cmd
        return True, "hi\n", ""

    monkeypatch.setattr(c, "exec_command", fake_exec)
    ok, content = c.get_file_content("/x.txt", tail_lines=10)
    assert ok is True
    assert "tail -n 10" in called["cmd"]
    assert content == "hi\n"


def test_get_process_list_filters_and_handles_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    c = SSHClient(host="pc", username=None)
    monkeypatch.setattr(c, "exec_command", lambda *a, **k: (True, "p1\n\np2\n", ""))
    assert c.get_process_list() == ["p1", "p2"]

    # filter path
    called: dict[str, str] = {}

    def fake_exec(cmd, **kwargs):
        called["cmd"] = cmd
        return True, "p\n", ""

    monkeypatch.setattr(c, "exec_command", fake_exec)
    _ = c.get_process_list(filter_pattern="train")
    assert "grep 'train'" in called["cmd"]

    # empty path
    monkeypatch.setattr(c, "exec_command", lambda *a, **k: (False, "", ""))
    assert c.get_process_list() == []


# ---------------------------------------------------------------------------
# is_alias_mode property (line 175)
# ---------------------------------------------------------------------------


def test_is_alias_mode_property() -> None:
    alias = SSHClient(host="pc", username=None)
    assert alias.is_alias_mode is True

    explicit = SSHClient(host="1.2.3.4", port=22, username="root", key_path="/k")
    assert explicit.is_alias_mode is False


# ---------------------------------------------------------------------------
# Control socket dir OSError → continues without ControlMaster (lines 127-129)
# ---------------------------------------------------------------------------


def test_control_socket_dir_oserror_disables_control_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """When ~/.ssh/sockets creation raises OSError, _control_path must be None."""

    real_mkdir = Path.mkdir

    def patched_mkdir(self, **kwargs):
        if "sockets" in str(self):
            raise OSError("permission denied")
        real_mkdir(self, **kwargs)

    monkeypatch.setattr(Path, "mkdir", patched_mkdir)
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

    c = SSHClient(host="pc", username=None)
    assert c._control_path is None
    # ControlMaster options must NOT appear in ssh_base_opts
    assert "ControlMaster=auto" not in " ".join(c.ssh_base_opts)


# ---------------------------------------------------------------------------
# close_master() branches (lines 248-271)
# ---------------------------------------------------------------------------


def test_close_master_no_control_path_returns_immediately(monkeypatch: pytest.MonkeyPatch) -> None:
    """close_master() must be a no-op when _control_path is None."""
    calls: list[object] = []
    monkeypatch.setattr(ssh_mod.subprocess, "run", lambda *a, **k: calls.append(a))

    c = SSHClient(host="pc", username=None)
    c._control_path = None
    c.close_master()
    assert calls == []


def test_close_master_nonzero_with_real_error_logs_debug(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-zero returncode with unfiltered stderr should reach the debug log branch."""

    def fake_run(cmd, **kwargs):
        return _RunResult(returncode=1, stdout="", stderr="some unexpected error")

    monkeypatch.setattr(ssh_mod.subprocess, "run", fake_run)

    c = SSHClient(host="pc", username=None)
    c._control_path = "/tmp/helix_%C"
    # Should not raise; debug branch is hit silently.
    c.close_master()


def test_close_master_nonzero_filtered_stderr_no_log(monkeypatch: pytest.MonkeyPatch) -> None:
    """'No such file or directory' in stderr must NOT trigger debug log (filtered out)."""

    def fake_run(cmd, **kwargs):
        return _RunResult(returncode=1, stdout="", stderr="No such file or directory: /tmp/helix_abc")

    monkeypatch.setattr(ssh_mod.subprocess, "run", fake_run)

    c = SSHClient(host="pc", username=None)
    c._control_path = "/tmp/helix_%C"
    c.close_master()  # Must not raise


def test_close_master_control_socket_connect_filtered(monkeypatch: pytest.MonkeyPatch) -> None:
    """'Control socket connect' in stderr must be silently ignored."""

    def fake_run(cmd, **kwargs):
        return _RunResult(returncode=255, stdout="", stderr="Control socket connect(/tmp/helix): No such file or directory")

    monkeypatch.setattr(ssh_mod.subprocess, "run", fake_run)

    c = SSHClient(host="pc", username=None)
    c._control_path = "/tmp/helix_%C"
    c.close_master()  # Must not raise


def test_close_master_exception_is_swallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Any Exception raised by subprocess.run must be swallowed (line 271)."""

    def fake_run(*a, **k):
        raise RuntimeError("connection lost")

    monkeypatch.setattr(ssh_mod.subprocess, "run", fake_run)

    c = SSHClient(host="pc", username=None)
    c._control_path = "/tmp/helix_%C"
    c.close_master()  # Must not propagate


def test_close_master_explicit_mode_includes_port_and_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """In explicit mode, close_master must pass -p and -i to the ssh command."""
    captured: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        captured.append(cmd)
        return _RunResult(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(ssh_mod.subprocess, "run", fake_run)

    c = SSHClient(host="1.2.3.4", port=2222, username="root", key_path="/mykey")
    c._control_path = "/tmp/helix_%C"
    c.close_master()

    assert captured, "subprocess.run should have been called"
    cmd = captured[0]
    assert "-p" in cmd and "2222" in cmd
    assert "-i" in cmd and "/mykey" in cmd
    assert "-O" in cmd and "exit" in cmd


# ---------------------------------------------------------------------------
# download_directory — local mkdir OSError (lines 476-477)
# ---------------------------------------------------------------------------


def test_download_directory_local_mkdir_oserror(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """When local_path.mkdir raises OSError, returns Err with SSH_DOWNLOAD_LOCAL_DIR_FAILED."""

    c = SSHClient(host="pc", username=None)

    bad_local = tmp_path / "no_perms"

    real_mkdir = Path.mkdir

    def patched_mkdir(self, **kwargs):
        if self == bad_local:
            raise OSError("read-only filesystem")
        real_mkdir(self, **kwargs)

    monkeypatch.setattr(Path, "mkdir", patched_mkdir)

    res = c.download_directory("/remote/dir", bad_local)
    assert res.is_failure()
    err = res.unwrap_err()
    assert err.code == "SSH_DOWNLOAD_LOCAL_DIR_FAILED"
    assert "read-only filesystem" in err.message


# ---------------------------------------------------------------------------
# upload_directory — all branches (lines 548-603)
# ---------------------------------------------------------------------------


def test_upload_directory_local_not_found(tmp_path: Path) -> None:
    """Err with SSH_UPLOAD_LOCAL_NOT_FOUND when local path doesn't exist."""
    c = SSHClient(host="pc", username=None)
    res = c.upload_directory(tmp_path / "nonexistent", "/remote")
    assert res.is_failure()
    assert res.unwrap_err().code == "SSH_UPLOAD_LOCAL_NOT_FOUND"


def test_upload_directory_not_a_directory(tmp_path: Path) -> None:
    """Err with SSH_UPLOAD_NOT_A_DIRECTORY when local path is a file."""
    local = tmp_path / "file.txt"
    local.write_text("data", encoding="utf-8")

    c = SSHClient(host="pc", username=None)
    res = c.upload_directory(local, "/remote")
    assert res.is_failure()
    assert res.unwrap_err().code == "SSH_UPLOAD_NOT_A_DIRECTORY"


def test_upload_directory_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Ok(None) when subprocess returns zero exit code."""
    local = tmp_path / "mydir"
    local.mkdir()

    def fake_run(cmd, **kwargs):
        return _RunResult(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(ssh_mod.subprocess, "run", fake_run)

    c = SSHClient(host="pc", username=None)
    res = c.upload_directory(local, "/remote")
    assert res.is_success()


def test_upload_directory_failure_returncode(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Err with SSH_UPLOAD_FAILED when subprocess returns non-zero."""
    local = tmp_path / "mydir"
    local.mkdir()

    def fake_run(cmd, **kwargs):
        return _RunResult(returncode=1, stdout="", stderr="tar: error")

    monkeypatch.setattr(ssh_mod.subprocess, "run", fake_run)

    c = SSHClient(host="pc", username=None)
    res = c.upload_directory(local, "/remote")
    assert res.is_failure()
    err = res.unwrap_err()
    assert err.code == "SSH_UPLOAD_FAILED"
    assert "tar: error" in err.message


def test_upload_directory_timeout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Err with SSH_UPLOAD_TIMEOUT when subprocess times out."""
    local = tmp_path / "mydir"
    local.mkdir()

    def fake_run(cmd, **kwargs):
        raise ssh_mod.subprocess.TimeoutExpired(cmd="x", timeout=1)

    monkeypatch.setattr(ssh_mod.subprocess, "run", fake_run)

    c = SSHClient(host="pc", username=None)
    res = c.upload_directory(local, "/remote", timeout=1)
    assert res.is_failure()
    assert res.unwrap_err().code == "SSH_UPLOAD_TIMEOUT"


def test_upload_directory_oserror(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Err with SSH_UPLOAD_IO_ERROR when subprocess raises OSError."""
    local = tmp_path / "mydir"
    local.mkdir()

    def fake_run(cmd, **kwargs):
        raise OSError("broken pipe")

    monkeypatch.setattr(ssh_mod.subprocess, "run", fake_run)

    c = SSHClient(host="pc", username=None)
    res = c.upload_directory(local, "/remote")
    assert res.is_failure()
    assert res.unwrap_err().code == "SSH_UPLOAD_IO_ERROR"
