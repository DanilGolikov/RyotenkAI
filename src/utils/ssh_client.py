"""
SSH Client - unified interface for all SSH operations.

Provider-agnostic SSH client that works with any SSH-accessible server.
Handles Direct TCP SSH connection (ip:port).

This is a refactored version of RunPodSSHClient, made generic for all providers.
"""

from __future__ import annotations

import logging
import re
import shlex
import shutil
import subprocess
import time
from pathlib import Path
from typing import ClassVar

from src.utils.constants import LOG_OUTPUT_LONG_CHARS, LOG_OUTPUT_SHORT_CHARS
from src.utils.result import Err, Ok, ProviderError, Result

logger = logging.getLogger("ryotenkai")

# Patterns to mask in logs (secrets, tokens, keys)

_SSH_PROBE_MIN_TIMEOUT = 15  # seconds — floor for connection probes
_LOCAL_PATH_KEY = "local_path"
_REMOTE_PATH_KEY = "remote_path"
_SSH_CMD_PREVIEW_LEN = 55  # chars — command preview in error log lines
_SSH_CONNECT_CHECK_TIMEOUT = 300  # seconds — scp/sftp connection timeout
_SSH_QUICK_CMD_TIMEOUT = 30  # seconds — fast one-liners (healthcheck, rm)
_SSH_CONTROL_PERSIST_SECONDS = 120
_SSH_CONTROL_CLOSE_TIMEOUT = 5
_SSH_SOCKET_DIR_MODE = 0o700  # Restrictive permissions for control socket directory

_SSH_DOWNLOAD_STALL_S: int = 120       # seconds without new bytes → stall
_SSH_DOWNLOAD_MAX_RETRIES: int = 3     # restart attempts before giving up
_SSH_DOWNLOAD_CHECK_INTERVAL: int = 30 # progress-poll interval


class _StallDetected(Exception):
    """Raised internally when download makes no progress for too long."""

_MASK_REPL = r"\1***\3"
_SECRET_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r'(\bHF_TOKEN\b\s*=\s*["\']?)([^"\s\n]+)(["\']?)'), _MASK_REPL),
    (re.compile(r'(\bAPI_KEY\b\s*=\s*["\']?)([^"\s\n]+)(["\']?)'), _MASK_REPL),
    (re.compile(r'(\bSECRET\b\s*=\s*["\']?)([^"\s\n]+)(["\']?)'), _MASK_REPL),
    (re.compile(r'(\bPASSWORD\b\s*=\s*["\']?)([^"\s\n]+)(["\']?)'), _MASK_REPL),
    (re.compile(r"(hf_)[A-Za-z0-9]+"), r"\1***"),
)


def _mask_secrets(text: str) -> str:
    """Mask sensitive data in text for safe logging."""
    for pattern, replacement in _SECRET_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


class SSHClient:
    """
    Unified SSH client for GPU servers.

    Two connection modes:
        1. Alias mode: host is SSH alias from ~/.ssh/config, username=None
        2. Explicit mode: host is IP/hostname, username required

    Features:
        - SSH alias support (uses ~/.ssh/config)
        - Auto-detect SSH key
        - Retry logic for connection tests
        - File upload/download via SCP
        - Command execution with timeout
        - Background process support

    Example (alias mode):
        client = SSHClient(host="pc", port=22)  # Uses ~/.ssh/config

    Example (explicit mode):
        client = SSHClient(
            host="<your-gpu-host>",
            port=22,
            username="<your-user>",
            key_path="/home/user/.ssh/id_ed25519"
        )
    """

    DEFAULT_KEY_NAMES: ClassVar[list[str]] = [
        "id_ed25519_runpod",
        "id_ed25519",
        "id_rsa",
    ]

    def __init__(
        self,
        host: str,
        port: int = 22,
        username: str | None = None,
        key_path: str | None = None,
        connect_timeout: int = 10,
    ):
        """
        Initialize SSH client.

        Args:
            host: SSH host — can be IP, hostname, or SSH alias from ~/.ssh/config
            port: SSH port number (default: 22)
            username: SSH username. If None, uses alias mode (from ~/.ssh/config)
            key_path: Path to SSH private key (auto-detected if None)
        """
        self.host = host
        self.port = port
        self.username = username
        self._is_alias_mode = username is None
        self._connect_timeout = int(connect_timeout)
        self._control_path: str | None = None

        # Only auto-detect key if not using alias mode
        if key_path:
            self.key_path = key_path
        elif self._is_alias_mode:
            self.key_path = ""  # SSH will use config
        else:
            self.key_path = self._find_ssh_key()

        socket_dir = Path.home() / ".ssh" / "sockets"
        try:
            socket_dir.mkdir(parents=True, exist_ok=True, mode=_SSH_SOCKET_DIR_MODE)
            socket_dir.chmod(_SSH_SOCKET_DIR_MODE)
            self._control_path = str(socket_dir / "helix_%C")
        except OSError as e:
            logger.warning(f"⚠️ Failed to initialize SSH control socket dir: {e}. Continuing without ControlMaster.")
            self._control_path = None

        _o = "-o"
        self.ssh_base_opts = [
            _o,
            "StrictHostKeyChecking=no",
            _o,
            f"ConnectTimeout={self._connect_timeout}",
            _o,
            "BatchMode=yes",
            _o,
            "PasswordAuthentication=no",
            _o,
            "KbdInteractiveAuthentication=no",
            _o,
            "ServerAliveInterval=30",
            _o,
            "ServerAliveCountMax=3",
            _o,
            "LogLevel=ERROR",
            _o,
            "UserKnownHostsFile=/dev/null",
        ]
        if self._control_path:
            self.ssh_base_opts.extend(
                [
                    _o,
                    "ControlMaster=auto",
                    _o,
                    f"ControlPath={self._control_path}",
                    _o,
                    f"ControlPersist={_SSH_CONTROL_PERSIST_SECONDS}",
                ]
            )

        if self._is_alias_mode:
            logger.info(f"🔗 SSH Client initialized (alias mode): {host}")
        else:
            logger.info(f"🔗 SSH Client initialized: {username}@{host}:{port}")

        if self.key_path:
            logger.debug(f"🔑 Using key: {self.key_path}")

    @property
    def is_alias_mode(self) -> bool:
        """Whether the client uses SSH alias mode (reads ~/.ssh/config)."""
        return self._is_alias_mode

    @property
    def ssh_target(self) -> str:
        """Get SSH target string (user@host or just alias)."""
        if self._is_alias_mode:
            return self.host
        return f"{self.username}@{self.host}"

    def _find_ssh_key(self) -> str:
        """
        Auto-detect SSH key from ~/.ssh directory.

        Searches for keys in order of preference defined in DEFAULT_KEY_NAMES.

        Returns:
            Path to SSH key

        Raises:
            FileNotFoundError: If no SSH key found
        """
        ssh_dir = Path.home() / ".ssh"

        for key_name in self.DEFAULT_KEY_NAMES:
            key_path = ssh_dir / key_name
            if key_path.exists():
                logger.debug(f"🔑 Found SSH key: {key_path}")
                return str(key_path)

        raise FileNotFoundError(f"SSH key not found in {ssh_dir}. Searched for: {', '.join(self.DEFAULT_KEY_NAMES)}")

    def _build_ssh_cmd(self, command: str, background: bool = False) -> str:
        """Build SSH command string."""
        ssh_opts = " ".join(self.ssh_base_opts)

        if background:
            command = f"nohup {command} &"

        # Quote remote command safely. This supports:
        # - Multiline commands (heredocs)
        # - Commands that contain single quotes (e.g., `<< 'EOF'`)
        remote_cmd = shlex.quote(command)

        # Build command based on mode
        if self._is_alias_mode:
            # Alias mode: ssh [opts] alias 'command'
            # Port and key are from ~/.ssh/config
            return f"ssh {ssh_opts} {self.host} {remote_cmd}"
        else:
            # Explicit mode: ssh -p port -i key [opts] user@host 'command'
            key_opt = f'-i "{self.key_path}"' if self.key_path else ""
            return f"ssh -p {self.port} {ssh_opts} {key_opt} {self.ssh_target} {remote_cmd}"

    def _build_scp_cmd(self, local_path: str, remote_path: str) -> list[str]:
        """Build SCP command for file upload."""
        if self._is_alias_mode:
            # Alias mode: scp [opts] local alias:remote
            return [
                "scp",
                *self.ssh_base_opts,
                local_path,
                f"{self.host}:{remote_path}",
            ]
        else:
            # Explicit mode: scp -P port -i key [opts] local user@host:remote
            cmd = ["scp", "-P", str(self.port)]
            if self.key_path:
                cmd.extend(["-i", self.key_path])
            cmd.extend([*self.ssh_base_opts, local_path, f"{self.ssh_target}:{remote_path}"])
            return cmd

    def close_master(self) -> None:
        """Close persistent SSH ControlMaster connection (best-effort)."""
        if not self._control_path:
            return

        cmd = ["ssh"]
        if not self._is_alias_mode:
            cmd.extend(["-p", str(self.port)])
            if self.key_path:
                cmd.extend(["-i", self.key_path])
        cmd.extend(["-o", f"ControlPath={self._control_path}", "-O", "exit", self.ssh_target])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=max(_SSH_CONTROL_CLOSE_TIMEOUT, self._connect_timeout),
            )
            if result.returncode != 0:
                stderr = (result.stderr or "").strip()
                # Normal when no control socket was created yet.
                if stderr and "No such file or directory" not in stderr and "Control socket connect" not in stderr:
                    logger.debug(f"[SSH] close_master returned non-zero: {stderr}")
        except Exception as e:
            logger.debug(f"[SSH] close_master failed: {e}")

    def test_connection(
        self,
        max_retries: int = 12,
        retry_delay: int = 10,
    ) -> tuple[bool, str]:
        """
        Test SSH connection with retries.

        Args:
            max_retries: Maximum number of connection attempts
            retry_delay: Seconds between retries

        Returns:
            (success: bool, error_message: str)
        """
        for attempt in range(1, max_retries + 1):
            logger.debug(f"🔍 Testing SSH connection (attempt {attempt}/{max_retries})...")

            try:
                cmd = self._build_ssh_cmd("echo 'SSH OK'")
                result = subprocess.run(
                    cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=max(_SSH_PROBE_MIN_TIMEOUT, self._connect_timeout + 5),
                )

                if result.returncode == 0 and "SSH OK" in result.stdout:
                    logger.info(f"✅ SSH connection successful on attempt {attempt}")
                    return True, ""

                logger.debug(f"❌ SSH test failed (attempt {attempt}/{max_retries})")
                logger.debug(f"   Stdout: {result.stdout[:LOG_OUTPUT_SHORT_CHARS]}")
                logger.debug(f"   Stderr: {result.stderr[:LOG_OUTPUT_SHORT_CHARS]}")

            except subprocess.TimeoutExpired:
                logger.debug(f"❌ SSH test timeout (attempt {attempt}/{max_retries})")
            except Exception as e:
                logger.debug(f"❌ SSH test error: {e}")

            if attempt < max_retries:
                logger.debug(f"⏳ Waiting {retry_delay}s before retry...")
                time.sleep(retry_delay)

        return False, f"SSH connection failed after {max_retries} attempts"

    def upload_file(
        self,
        local_path: str,
        remote_path: str,
        verify: bool = True,
    ) -> tuple[bool, str]:
        """
        Upload file via SCP.

        Args:
            local_path: Local file path
            remote_path: Remote file path
            verify: Verify file exists after upload

        Returns:
            (success: bool, error_message: str)
        """
        if not Path(local_path).exists():
            return False, f"Local file not found: {local_path}"

        logger.info(f"📤 Uploading: {local_path} → {remote_path}")

        scp_cmd = self._build_scp_cmd(local_path, remote_path)

        try:
            result = subprocess.run(
                scp_cmd,
                capture_output=True,
                text=True,
                timeout=_SSH_CONNECT_CHECK_TIMEOUT,
            )

            if result.returncode != 0:
                logger.error(f"❌ SCP failed: {result.stderr}")
                return False, f"SCP error: {result.stderr}"

            # Verify upload
            if verify:
                logger.debug("🔍 Verifying upload...")
                exists = self.file_exists(remote_path)
                if not exists:
                    return False, f"File not found on server after upload: {remote_path}"

            logger.info(f"✅ Upload successful: {remote_path}")
            return True, ""

        except subprocess.TimeoutExpired:
            return False, "Upload timeout (>5 min)"
        except Exception as e:
            return False, f"Upload error: {e!s}"

    def exec_command(
        self,
        command: str,
        background: bool = False,
        timeout: int = 30,
        silent: bool = False,
    ) -> tuple[bool, str, str]:
        """
        Execute command on server.

        Args:
            command: Shell command to execute
            background: Run in background with nohup
            timeout: Command timeout in seconds
            silent: Suppress INFO/ERROR logs (for frequent polling commands)

        Returns:
            (success: bool, stdout: str, stderr: str)
        """
        ssh_cmd = self._build_ssh_cmd(command, background=background)

        try:
            result = subprocess.run(
                ssh_cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            success = result.returncode == 0

            # Log command result with masked secrets
            if not silent:
                status_text = "OK" if success else f"FAIL (exit {result.returncode})"
                safe_cmd = _mask_secrets(command)
                logger.info(f"SH -> {safe_cmd} -> {status_text}")

                # Show stderr only on failure
                if not success and result.stderr:
                    logger.error(f"   Stderr: {result.stderr[:LOG_OUTPUT_LONG_CHARS]}")

            return success, result.stdout, result.stderr

        except subprocess.TimeoutExpired:
            if not silent:
                logger.error(f"SH -> {command[:_SSH_CMD_PREVIEW_LEN]}... -> TIMEOUT (>{timeout}s)")
            return False, "", f"Timeout after {timeout}s"
        except Exception as e:
            if not silent:
                logger.error(f"SH -> {command[:_SSH_CMD_PREVIEW_LEN]}... -> ERROR: {e}")
            return False, "", str(e)

    def file_exists(self, remote_path: str) -> bool:
        """Check if file exists on server."""
        success, stdout, _ = self.exec_command(
            f"test -f {remote_path} && echo 'EXISTS'",
            timeout=10,
            silent=True,
        )
        return success and "EXISTS" in stdout

    def directory_exists(self, remote_path: str) -> bool:
        """Check if directory exists on server."""
        success, stdout, _ = self.exec_command(
            f"test -d {remote_path} && echo 'EXISTS'",
            timeout=10,
            silent=True,
        )
        return success and "EXISTS" in stdout

    def create_directory(self, remote_path: str) -> tuple[bool, str]:
        """Create directory on server (with parents)."""
        # In cloud/container environments an SSH command can occasionally take longer
        # than a minimal 10s budget due to handshake overhead. Keep this generous.
        timeout_seconds = max(_SSH_QUICK_CMD_TIMEOUT, int(self._connect_timeout) + 10)
        success, _, stderr = self.exec_command(
            f"mkdir -p {remote_path}",
            timeout=timeout_seconds,
        )
        if not success:
            return False, f"Failed to create directory: {stderr}"
        return True, ""

    @staticmethod
    def _dir_size_mb(path: Path) -> float:
        """Return current size of a directory tree in MB (best-effort, non-blocking)."""
        try:
            return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / (1024 * 1024)
        except OSError:
            return 0.0

    def _monitor_download(
        self,
        proc: subprocess.Popen,  # type: ignore[type-arg]
        local_path: Path,
        stall_timeout: int,
        interval: int = _SSH_DOWNLOAD_CHECK_INTERVAL,
    ) -> None:
        """
        Block until *proc* finishes, logging progress every *interval* seconds.
        If no new bytes arrive for *stall_timeout* seconds the process is killed
        and _StallDetected is raised so the caller can restart.
        """
        last_size_mb = 0.0
        last_progress_t = time.monotonic()
        elapsed = 0

        while True:
            try:
                proc.wait(timeout=interval)
                return  # finished normally
            except subprocess.TimeoutExpired:
                elapsed += interval
                size_mb = self._dir_size_mb(local_path)
                stalled_for = int(time.monotonic() - last_progress_t)

                if size_mb > last_size_mb:
                    last_size_mb = size_mb
                    last_progress_t = time.monotonic()
                    stalled_for = 0

                logger.info(
                    f"⏳ Downloading... {elapsed}s elapsed, "
                    f"{size_mb:.1f} MB received"
                    + (f", no progress for {stalled_for}s/{stall_timeout}s" if stalled_for else "")
                )

                if stalled_for >= stall_timeout:
                    proc.kill()
                    proc.wait()
                    raise _StallDetected(
                        f"No new bytes for {stalled_for}s (stall_timeout={stall_timeout}s)"
                    )

    def download_directory(
        self,
        remote_path: str,
        local_path: Path,
        stall_timeout: int = _SSH_DOWNLOAD_STALL_S,
        max_retries: int = _SSH_DOWNLOAD_MAX_RETRIES,
    ) -> Result[None, ProviderError]:
        """
        Download a directory from server using tar+ssh streaming.

        Instead of a fixed timeout the download is monitored for stalls: if no
        new bytes land in *stall_timeout* seconds the transfer is killed and
        restarted (up to *max_retries* times).

        Args:
            remote_path:   Remote directory path (e.g. /workspace/output)
            local_path:    Local directory path to save to
            stall_timeout: Seconds without new bytes before restart (default 120s)
            max_retries:   Max restart attempts before giving up (default 3)

        Returns:
            Result[None, ProviderError]
        """
        logger.info(f"📥 Downloading directory: {remote_path} -> {local_path}")

        tar_create_cmd = f"cd {remote_path} && tar czf - ."
        ssh_cmd = self._build_ssh_cmd(tar_create_cmd, background=False)
        full_cmd = f"{ssh_cmd} | tar xzf - -C {local_path}"

        for attempt in range(1, max_retries + 1):
            # Clear partial files from any previous attempt
            if local_path.exists():
                try:
                    shutil.rmtree(local_path)
                except OSError as e:
                    return Err(
                        ProviderError(
                            message=f"Failed to clear local directory '{local_path}': {e}",
                            code="SSH_DOWNLOAD_LOCAL_DIR_FAILED",
                            details={_LOCAL_PATH_KEY: str(local_path)},
                        )
                    )
            try:
                local_path.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                return Err(
                    ProviderError(
                        message=f"Failed to create local directory '{local_path}': {e}",
                        code="SSH_DOWNLOAD_LOCAL_DIR_FAILED",
                        details={_LOCAL_PATH_KEY: str(local_path)},
                    )
                )

            logger.info(
                f"🚀 Starting download"
                + (f" (attempt {attempt}/{max_retries})" if attempt > 1 else "")
                + f" (stall_timeout: {stall_timeout}s)..."
            )

            try:
                proc = subprocess.Popen(
                    full_cmd,
                    shell=True,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            except OSError as e:
                return Err(
                    ProviderError(
                        message=f"Download I/O error: {e}",
                        code="SSH_DOWNLOAD_IO_ERROR",
                        details={_REMOTE_PATH_KEY: remote_path, _LOCAL_PATH_KEY: str(local_path)},
                    )
                )

            try:
                self._monitor_download(proc, local_path, stall_timeout)
            except _StallDetected as exc:
                if attempt < max_retries:
                    logger.warning(
                        f"⚠️ Download stalled (attempt {attempt}/{max_retries}): {exc}. Restarting..."
                    )
                    continue
                return Err(
                    ProviderError(
                        message=f"Download stalled after {max_retries} attempts: {exc}",
                        code="SSH_DOWNLOAD_STALLED",
                        details={_REMOTE_PATH_KEY: remote_path, "stall_timeout": stall_timeout},
                    )
                )

            stderr_output = proc.stderr.read() if proc.stderr else ""
            if proc.returncode != 0:
                return Err(
                    ProviderError(
                        message=f"Download failed: {stderr_output}",
                        code="SSH_DOWNLOAD_FAILED",
                        details={_REMOTE_PATH_KEY: remote_path, _LOCAL_PATH_KEY: str(local_path)},
                    )
                )

            logger.info("✅ Directory downloaded successfully")
            return Ok(None)

        # Unreachable — loop always returns, but keeps type-checker happy
        return Err(  # pragma: no cover
            ProviderError(message="Download failed", code="SSH_DOWNLOAD_FAILED", details={})
        )

    def upload_directory(
        self,
        local_path: Path,
        remote_path: str,
        timeout: int = 1800,
    ) -> Result[None, ProviderError]:
        """
        Upload a directory to server using tar+ssh streaming.

        This is symmetrical to download_directory() and avoids SCP recursion issues.

        Args:
            local_path: Local directory path to upload
            remote_path: Remote directory path to extract into
            timeout: Upload timeout in seconds (default 30 min)

        Returns:
            Result[None, ProviderError]: Success or structured provider error
        """
        if not local_path.exists():
            return Err(
                ProviderError(
                    message=f"Local directory not found: {local_path}",
                    code="SSH_UPLOAD_LOCAL_NOT_FOUND",
                    details={_LOCAL_PATH_KEY: str(local_path)},
                )
            )
        if not local_path.is_dir():
            return Err(
                ProviderError(
                    message=f"Local path is not a directory: {local_path}",
                    code="SSH_UPLOAD_NOT_A_DIRECTORY",
                    details={_LOCAL_PATH_KEY: str(local_path)},
                )
            )

        logger.info(f"📤 Uploading directory: {local_path} -> {remote_path}")

        try:
            remote_cmd = f"mkdir -p {remote_path} && tar xzf - -C {remote_path}"
            ssh_cmd = self._build_ssh_cmd(remote_cmd, background=False)
            tar_cmd = f"tar czf - -C {local_path} ."
            full_cmd = f"{tar_cmd} | {ssh_cmd}"

            logger.info(f"🚀 Starting upload (timeout: {timeout}s)...")
            result = subprocess.run(
                full_cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            if result.returncode != 0:
                return Err(
                    ProviderError(
                        message=f"Upload failed: {result.stderr}",
                        code="SSH_UPLOAD_FAILED",
                        details={_LOCAL_PATH_KEY: str(local_path), _REMOTE_PATH_KEY: remote_path},
                    )
                )

            logger.info("✅ Directory uploaded successfully")
            return Ok(None)

        except subprocess.TimeoutExpired:
            return Err(
                ProviderError(
                    message=f"Upload timeout (>{timeout}s)",
                    code="SSH_UPLOAD_TIMEOUT",
                    details={_LOCAL_PATH_KEY: str(local_path), _REMOTE_PATH_KEY: remote_path, "timeout": timeout},
                )
            )
        except OSError as e:
            return Err(
                ProviderError(
                    message=f"Upload I/O error: {e}",
                    code="SSH_UPLOAD_IO_ERROR",
                    details={_LOCAL_PATH_KEY: str(local_path), _REMOTE_PATH_KEY: remote_path},
                )
            )

    def get_file_content(
        self,
        remote_path: str,
        tail_lines: int | None = None,
        silent: bool = False,
    ) -> tuple[bool, str]:
        """
        Read file content from server.

        Args:
            remote_path: Remote file path
            tail_lines: If set, read only last N lines
            silent: Suppress command execution logs (for frequent polling)

        Returns:
            (success: bool, content: str)
        """
        if tail_lines:
            cmd = f"tail -n {tail_lines} {remote_path} 2>/dev/null || echo 'FILE_NOT_FOUND'"
        else:
            cmd = f"cat {remote_path} 2>/dev/null || echo 'FILE_NOT_FOUND'"

        success, stdout, _ = self.exec_command(cmd, timeout=_SSH_QUICK_CMD_TIMEOUT, silent=silent)

        if "FILE_NOT_FOUND" in stdout:
            return False, f"File not found: {remote_path}"

        return success, stdout

    def get_process_list(self, filter_pattern: str | None = None) -> list[str]:
        """
        Get list of running processes.

        Args:
            filter_pattern: Optional grep pattern to filter processes

        Returns:
            List of process lines
        """
        if filter_pattern:
            cmd = f"ps aux | grep '{filter_pattern}' | grep -v grep"
        else:
            cmd = "ps aux"

        success, stdout, _ = self.exec_command(cmd, timeout=10, silent=True)

        if not success or not stdout.strip():
            return []

        return [line for line in stdout.split("\n") if line.strip()]

    def __repr__(self) -> str:
        if self._is_alias_mode:
            return f"SSHClient(alias={self.host})"
        return f"SSHClient({self.ssh_target}:{self.port})"
