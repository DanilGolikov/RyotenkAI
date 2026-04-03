"""
Runpodctl-first file transport for RunPod training deployments.
"""

from __future__ import annotations

import logging
import shutil
import tarfile
import tempfile
import time
from pathlib import Path
from uuid import uuid4

from src.providers.runpod.runpodctl_client import RunPodCtlClient
from src.utils.result import ConfigError, Err, Ok, ProviderError, Result

_REMOTE_READY_DELAY_SECONDS = 2
_REMOTE_RECEIVE_START_TIMEOUT_SECONDS = 10
_REMOTE_RECEIVE_POLL_SECONDS = 1

logger = logging.getLogger("ryotenkai")


class RunPodCtlFileTransport:
    """Transfer files with runpodctl send/receive, fallback handled by caller."""

    def __init__(self, *, client: RunPodCtlClient):
        self._client = client

    def upload_batch(
        self,
        *,
        ssh_client,
        workspace: str,
        pod_id: str,
        files_to_upload: list[tuple[str, str]],
        verify_timeout: int,
    ) -> Result[None, ProviderError]:
        existing_files: list[tuple[str, str]] = []
        for local_path, remote_name in files_to_upload:
            if not Path(local_path).exists():
                continue
            if Path(remote_name).is_absolute():
                return Err(
                    ConfigError(
                        message=f"Invalid remote path (must be relative): {remote_name}",
                        code="INVALID_REMOTE_PATH",
                    )
                )
            existing_files.append((local_path, remote_name))

        if not existing_files:
            return Err(ProviderError(message="No files to upload", code="NO_FILES_TO_UPLOAD"))

        logger.info("📦 [runpodctl] Verifying remote runpodctl availability...")
        remote_check = ssh_client.exec_command(
            "command -v runpodctl >/dev/null 2>&1 && echo READY || echo MISSING",
            background=False,
            timeout=verify_timeout,
            silent=True,
        )
        if not remote_check[0] or "READY" not in (remote_check[1] or ""):
            return Err(
                ProviderError(
                    message="runpodctl is not available inside the RunPod pod",
                    code="RUNPODCTL_REMOTE_UNAVAILABLE",
                )
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            stage_root = Path(tmpdir) / "payload"
            stage_root.mkdir(parents=True, exist_ok=True)

            for local_path, remote_name in existing_files:
                dest_path = stage_root / remote_name
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(local_path, dest_path)

            archive_name = f"helix-transfer-{pod_id[:12]}-{uuid4().hex[:8]}.tar.gz"
            archive_path = Path(tmpdir) / archive_name
            with tarfile.open(archive_path, "w:gz") as tar:
                tar.add(stage_root, arcname=".")
            archive_size_bytes = archive_path.stat().st_size

            transfer_dir = f"{workspace}/.runpodctl-transfer"
            code = f"helix-{uuid4().hex[:20]}"
            remote_archive = f"{transfer_dir}/{archive_name}"
            receive_log = f"{transfer_dir}/receive.log"
            receive_pid = f"{transfer_dir}/receive.pid"

            logger.info(
                "📦 [runpodctl] Prepared archive: %s files, %.1f KB",
                len(existing_files),
                archive_size_bytes / 1024,
            )
            prepare_cmd = (
                f"mkdir -p {transfer_dir} && "
                f"rm -f {remote_archive} {receive_log} {receive_pid} && "
                f"cd {transfer_dir} && "
                f"nohup sh -c 'runpodctl receive {code} > {receive_log} 2>&1' </dev/null >/dev/null 2>&1 & "
                f"echo $! > {receive_pid}"
            )
            logger.info("📦 [runpodctl] Starting remote receive session...")
            success, _, stderr = ssh_client.exec_command(
                prepare_cmd,
                background=False,
                timeout=verify_timeout,
                silent=True,
            )
            if not success:
                return Err(
                    ProviderError(
                        message=f"Failed to prepare remote runpodctl receive: {stderr}",
                        code="RUNPODCTL_RECEIVE_PREPARE_FAILED",
                    )
                )

            ready_result = self._wait_for_remote_receive(
                ssh_client=ssh_client,
                receive_pid=receive_pid,
                receive_log=receive_log,
                verify_timeout=verify_timeout,
            )
            if ready_result.is_failure():
                return ready_result

            logger.info("📦 [runpodctl] Sending archive to remote receive session...")

            send_result = self._client.send(local_path=str(archive_path), code=code)
            if send_result.is_failure():
                details = self._read_receive_diagnostics(
                    ssh_client=ssh_client,
                    receive_log=receive_log,
                    verify_timeout=verify_timeout,
                )
                err = send_result.unwrap_err()  # type: ignore[union-attr]
                detail_suffix = f" Remote receive log:\n{details}" if details else ""
                return Err(
                    ProviderError(
                        message=f"{err.message}{detail_suffix}",
                        code=err.code,
                        details=err.details,
                    )
                )

            logger.info("📦 [runpodctl] Extracting received archive on remote...")
            extract_cmd = (
                f"test -f {remote_archive} && "
                f"cd {workspace} && "
                f"tar xzf {remote_archive} && "
                f"rm -f {remote_archive}"
            )
            success, stdout, stderr = ssh_client.exec_command(
                extract_cmd,
                background=False,
                timeout=max(verify_timeout, 120),
                silent=True,
            )
            if not success:
                return Err(
                    ProviderError(
                        message=f"Failed to extract runpodctl archive: {stderr or stdout}",
                        code="RUNPODCTL_EXTRACT_FAILED",
                    )
                )

        return Ok(None)

    def _wait_for_remote_receive(
        self,
        *,
        ssh_client,
        receive_pid: str,
        receive_log: str,
        verify_timeout: int,
    ) -> Result[None, ProviderError]:
        deadline = time.time() + _REMOTE_RECEIVE_START_TIMEOUT_SECONDS
        while time.time() < deadline:
            success, stdout, stderr = ssh_client.exec_command(
                (
                    f"if [ -s {receive_pid} ] && kill -0 \"$(cat {receive_pid})\" 2>/dev/null; then "
                    "echo STATUS=RUNNING; "
                    f"elif [ -f {receive_log} ]; then echo STATUS=EXITED; tail -n 20 {receive_log} 2>/dev/null || true; "
                    "else echo STATUS=PENDING; fi"
                ),
                background=False,
                timeout=verify_timeout,
                silent=True,
            )
            if success:
                lines = (stdout or "").splitlines()
                status_line = lines[0].strip() if lines else ""
                status = status_line.split("=", 1)[1].strip() if status_line.startswith("STATUS=") else ""
                if status == "RUNNING":
                    time.sleep(_REMOTE_READY_DELAY_SECONDS)
                    return Ok(None)
                if status == "EXITED":
                    details = "\n".join(line.strip() for line in lines[1:] if line.strip())
                    return Err(
                        ProviderError(
                            message=f"Remote runpodctl receive exited before send started: {details or 'no receive log output'}",
                            code="RUNPODCTL_RECEIVE_NOT_READY",
                        )
                    )
            elif stderr:
                logger.debug("runpodctl receive readiness probe failed: %s", stderr)

            time.sleep(_REMOTE_RECEIVE_POLL_SECONDS)

        details = self._read_receive_diagnostics(
            ssh_client=ssh_client,
            receive_log=receive_log,
            verify_timeout=verify_timeout,
        )
        return Err(
            ProviderError(
                message=(
                    "Timed out waiting for remote runpodctl receive to become ready"
                    + (f". Remote receive log:\n{details}" if details else "")
                ),
                code="RUNPODCTL_RECEIVE_NOT_READY",
            )
        )

    def _read_receive_diagnostics(self, *, ssh_client, receive_log: str, verify_timeout: int) -> str:
        success, stdout, _stderr = ssh_client.exec_command(
            command=f"tail -n 40 {receive_log} 2>/dev/null || true",
            background=False,
            timeout=verify_timeout,
            silent=True,
        )
        return (stdout or "").strip() if success else ""


__all__ = ["RunPodCtlFileTransport"]
