"""
Training Monitor Stage - SIMPLIFIED VERSION

Monitors training progress:
- Training process alive/dead
- Resource usage (vCPU, vRAM)
- Training duration
- Final marker files
- Periodic log downloads

No complex parsing, no JSON status, no W&B tracking - just simple checks every 10 seconds.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, Any

from src.pipeline.constants import (
    TRAINING_MONITOR_LINE_WIDTH,
    TRAINING_MONITOR_LOG_DOWNLOAD_INTERVAL,
    TRAINING_MONITOR_LOG_STATUS_INTERVAL,
    TRAINING_MONITOR_MOCK_LOSS_DECAY,
    TRAINING_MONITOR_MOCK_LOSS_INIT,
    TRAINING_MONITOR_SSH_PORT,
    TRAINING_MONITOR_START_TIMEOUT_DEFAULT,
)
from src.pipeline.stages.base import PipelineStage
from src.pipeline.stages.constants import StageNames
from src.pipeline.stages.managers import LogManager
from src.utils.docker import docker_is_container_running
from src.utils.logger import logger
from src.utils.result import AppError, Err, Ok, Result, TrainingError
from src.utils.ssh_client import SSHClient

_TRAINING_FAILED_MARKER = "TRAINING_FAILED"

if TYPE_CHECKING:
    from collections.abc import Callable

    from src.utils.config import PipelineConfig


# =============================================================================
# EVENT CALLBACKS
# =============================================================================


@dataclass
class TrainingMonitorEventCallbacks:
    """
    Callbacks for TrainingMonitor events (SOLID-compliant event collection).

    Used to integrate TrainingMonitor with MLflow or other logging systems.
    """

    # Training started event
    on_training_started: Callable[[], None] | None = None

    # Training completed event
    on_training_completed: Callable[[float], None] | None = None
    # Args: duration_seconds

    # Training failed event
    on_training_failed: Callable[[str, float], None] | None = None
    # Args: error_message, duration_seconds

    # Training timeout event (DEPRECATED - training runs without time limits)
    on_training_timeout: Callable[[int, float], None] | None = None
    # Args: max_hours, duration_seconds

    # Process died without marker event
    on_process_died: Callable[[float], None] | None = None
    # Args: duration_seconds

    # Resource check event (periodic)
    on_resource_check: Callable[[dict], None] | None = None
    # Args: resources dict {gpu_util, vram_used_gb, vram_total_gb, vram_pct, gpu_temp, ram_used_gb, ram_total_gb}


class TrainingMonitor(PipelineStage):
    """
    Training Monitor Stage - watches training process.

    SIMPLIFIED: Only checks if process is alive, shows resources, and tracks time.
    Works with any provider (RunPod, SingleNode, etc.)
    """

    def __init__(
        self,
        config: PipelineConfig,
        callbacks: TrainingMonitorEventCallbacks | None = None,
    ):
        super().__init__(config, StageNames.TRAINING_MONITOR)

        # Get provider config for monitoring settings
        # Prefer training-scoped provider view (new schemas), fallback to legacy flat provider dict.
        # NOTE: MagicMock returns "hasattr=True" for everything, so we check for dict result.
        provider_cfg_obj = None
        get_train_cfg = getattr(config, "get_provider_training_config", None)
        if callable(get_train_cfg):
            try:
                provider_cfg_obj = get_train_cfg()
            except Exception:
                provider_cfg_obj = None
        if not isinstance(provider_cfg_obj, dict):
            provider_cfg_obj = config.get_provider_config()
        provider_cfg = provider_cfg_obj if isinstance(provider_cfg_obj, dict) else {}
        self.check_interval = 5  # Training status check interval (seconds)
        self.training_start_timeout = TRAINING_MONITOR_START_TIMEOUT_DEFAULT
        self.log_download_interval = TRAINING_MONITOR_LOG_DOWNLOAD_INTERVAL
        _ = provider_cfg  # retained for future use

        self._log_manager: LogManager | None = None
        self._workspace_path: str = "/workspace"  # Default, updated in execute()
        self._callbacks = callbacks or TrainingMonitorEventCallbacks()
        self._training_start_time: float = 0.0  # Track training start for duration

    def execute(self, context: dict[str, Any]) -> Result[dict[str, Any], AppError]:
        """Monitor training on any provider (RunPod, SingleNode, etc.)."""
        # Get connection info from GPU Deployer stage context
        deployer_context = context.get(StageNames.GPU_DEPLOYER, {})

        resource_id = deployer_context.get("resource_id")  # pod_id for RunPod, run_dir for SingleNode
        ssh_host = deployer_context.get("ssh_host")
        ssh_port = deployer_context.get("ssh_port")
        ssh_key_path = deployer_context.get("ssh_key_path")
        ssh_user = deployer_context.get("ssh_user", "root")
        is_alias_mode = deployer_context.get("is_alias_mode", False)
        workspace_path = deployer_context.get("workspace_path")
        provider_info = deployer_context.get("provider_info", {})

        # Check for mock mode
        if provider_info.get("mock"):
            logger.info("[MOCK] Running in MOCK MODE - simulating training monitoring")
            return self._execute_mock(context, resource_id)

        if not all([ssh_host, ssh_port, workspace_path]):
            return Err(
                TrainingError(
                    message="Missing SSH/workspace connection info from GPU Deployer",
                    code="MISSING_SSH_INFO",
                )
            )

        logger.info(f"[MONITOR] Monitoring training: {resource_id or 'unknown'}")
        logger.info(f"Checking every {self.check_interval}s...")

        # Initialize SSH client
        if not isinstance(ssh_port, int):
            ssh_port = int(ssh_port) if ssh_port else TRAINING_MONITOR_SSH_PORT

        # For alias mode, username should be None (SSH will use ~/.ssh/config)
        effective_username = None if is_alias_mode else ssh_user
        ssh_client = SSHClient(
            host=str(ssh_host),
            port=ssh_port,
            username=effective_username,
            key_path=str(ssh_key_path) if ssh_key_path else "",
        )

        try:
            # Store workspace path for marker checks
            self._workspace_path = workspace_path

            # Initialize log manager
            self._log_manager = LogManager(ssh_client, remote_path=f"{workspace_path}/training.log")

            # Wait for training to start (configurable timeout)
            logger.info(f"Waiting for training to start (timeout: {self.training_start_timeout}s)...")
            training_started = self._wait_for_training_start(ssh_client, timeout=self.training_start_timeout)

            if not training_started:
                # Still collect and download logs even on timeout
                logger.error("Training failed to start within timeout")
                if self._log_manager:
                    self._log_manager.download_on_error("Training failed to start")
                self._display_last_log_lines()
                return Err(
                    TrainingError(
                        message=f"Training process failed to start within {self.training_start_timeout}s",
                        code="TRAINING_START_TIMEOUT",
                    )
                )

            logger.info("Training start detected (or marker present). Switching to monitoring.")
            self._training_start_time = time.time()

            # Fire callback
            if self._callbacks.on_training_started:
                self._callbacks.on_training_started()

            # Monitor training
            return self._monitor_training(ssh_client, context)
        finally:
            try:
                ssh_client.close_master()
            except Exception as e:
                logger.debug(f"[MONITOR] Failed to close SSH ControlMaster: {e}")

    def _wait_for_training_start(self, ssh_client: SSHClient, timeout: int = 120) -> bool:
        """Wait for training process to start (or already complete)."""
        start_time = time.time()
        workspace = self._workspace_path

        while time.time() - start_time < timeout:
            poll_cmd_timeout_seconds = min(10, max(3, int(self.check_interval)))

            # First check if training already completed (fast training scenario)
            if self._check_marker(ssh_client, "TRAINING_COMPLETE", timeout_seconds=poll_cmd_timeout_seconds):
                logger.info("Training already completed (fast training scenario)")
                return True

            if self._check_marker(ssh_client, _TRAINING_FAILED_MARKER, timeout_seconds=poll_cmd_timeout_seconds):
                logger.warning("Training already failed")
                return True  # Return True to let _monitor_training handle the error

            # Check for training.log
            success, content, _ = ssh_client.exec_command(
                command=f"test -f {workspace}/training.log && tail -n 1 {workspace}/training.log",
                silent=True,
                timeout=poll_cmd_timeout_seconds,
            )

            if success and content.strip() and "FILE_NOT_FOUND" not in content:
                alive = self._is_training_alive(ssh_client, timeout_seconds=poll_cmd_timeout_seconds)
                if alive:
                    # Training process is alive (supports both Docker and direct execution)
                    return True

                # training.log exists but the process is not alive and no markers exist yet.
                # This usually indicates an early crash before notifiers were initialized.
                # Return True so that _monitor_training can surface the error promptly.
                logger.warning("training.log exists but training process is not alive (no markers).")
                return True

            logger.debug(f"Waiting for training start... ({int(time.time() - start_time)}s/{timeout}s)")
            time.sleep(self.check_interval)

        return False

    def _monitor_training(self, ssh_client: SSHClient, context: dict[str, Any]) -> Result[dict[str, Any], AppError]:
        """
        Monitor training - SIMPLIFIED.

        Every N seconds:
        - Check if process is alive
        - Show resources (vCPU, vRAM)
        - Show training duration
        - Download logs periodically
        """
        start_time = time.time()
        last_log_time = 0.0  # Rate limiting for status logs
        last_log_download = 0.0  # Rate limiting for log downloads
        log_download_interval = self.log_download_interval

        def format_status_line(status: str, elapsed_str: str, resources: dict[str, Any]) -> str:
            return (
                f"[MONITOR] {status} | {elapsed_str} | "
                f"GPU: {resources['gpu_util']:.0f}% | "
                f"VRAM: {resources['vram_used_gb']:.1f}/{resources['vram_total_gb']:.0f}GB ({resources['vram_pct']:.0f}%) | "
                f"Temp: {resources['gpu_temp']:.0f}C | "
                f"RAM: {resources['ram_used_gb']:.0f}/{resources['ram_total_gb']:.0f}GB"
            )

        def build_success_result(elapsed_seconds: float) -> Result[dict[str, Any], AppError]:
            logger.info("Training completed successfully!")
            if self._log_manager:
                self._log_manager.download(silent=False)
            self._display_last_log_lines()
            # Fire callback
            if self._callbacks.on_training_completed:
                self._callbacks.on_training_completed(elapsed_seconds)
            return Ok(
                self.update_context(
                    context,
                    {"status": "completed", "training_duration_seconds": elapsed_seconds},
                )
            )

        def build_failed_result(elapsed_seconds: float, error_msg: str) -> Result[dict[str, Any], AppError]:
            logger.error(f"Training failed: {error_msg}")
            if self._log_manager:
                self._log_manager.download_on_error(f"Training failed: {error_msg}")
            self._display_last_log_lines()
            # Fire callback
            if self._callbacks.on_training_failed:
                self._callbacks.on_training_failed(error_msg, elapsed_seconds)
            return Err(TrainingError(message=f"Training failed: {error_msg}", code=_TRAINING_FAILED_MARKER))

        def check_markers(elapsed_seconds: float) -> Result[dict[str, Any], AppError] | None:
            # Check for completion markers
            if self._check_marker(ssh_client, "TRAINING_COMPLETE"):
                return build_success_result(elapsed_seconds)
            if self._check_marker(ssh_client, _TRAINING_FAILED_MARKER):
                error_msg = self._read_marker_content(ssh_client, _TRAINING_FAILED_MARKER)
                return build_failed_result(elapsed_seconds, error_msg)
            return None

        def log_status(is_alive: bool, elapsed_seconds: float) -> None:
            nonlocal last_log_time
            pod_resources = self._get_resources(ssh_client)

            # Rate limiting: log status only every N seconds
            if time.time() - last_log_time >= TRAINING_MONITOR_LOG_STATUS_INTERVAL:
                elapsed_str = str(timedelta(seconds=int(elapsed_seconds)))
                status = "ALIVE" if is_alive else "DEAD"

                logger.info(format_status_line(status, elapsed_str, pod_resources))
                last_log_time = time.time()

                # Fire resource check callback
                if self._callbacks.on_resource_check:
                    self._callbacks.on_resource_check(pod_resources)
            else:
                logger.debug(f"{int(elapsed_seconds)}s | alive={is_alive} | GPU={pod_resources['gpu_util']:.0f}%")

        while True:
            elapsed = time.time() - start_time

            marker_result = check_markers(elapsed)
            if marker_result is not None:
                return marker_result

            # Check if process is alive
            alive = self._is_training_alive(ssh_client)

            if not alive:
                # Process ended - check markers one more time (race condition fix)
                time.sleep(2)  # Brief wait for file system sync

                marker_result = check_markers(elapsed)
                if marker_result is not None:
                    return marker_result

                # No marker found - wait for container to finish writing files
                # Docker containers may take a few seconds to flush and exit
                logger.warning("Process ended, waiting for marker file...")

                # Try multiple times with short delays (total ~15s)
                for attempt in range(5):
                    time.sleep(3)

                    marker_result = check_markers(elapsed)
                    if marker_result is not None:
                        return marker_result

                    logger.debug(f"   Attempt {attempt + 1}/5: marker not found yet...")

                # Still no marker after retries - this is a real failure
                logger.error("Training process died without completion marker")
                if self._log_manager:
                    self._log_manager.download_on_error("Process died without marker")
                self._display_last_log_lines()
                # Fire callback
                if self._callbacks.on_process_died:
                    self._callbacks.on_process_died(elapsed)
                return Err(
                    TrainingError(
                        message="Training process died without completion marker", code="TRAINING_PROCESS_DIED"
                    )
                )

            log_status(alive, elapsed)

            # Periodic log download
            if time.time() - last_log_download >= log_download_interval:
                if self._log_manager:
                    self._log_manager.download(silent=True)
                last_log_download = time.time()

            # Wait before next check
            time.sleep(self.check_interval)

    @staticmethod
    def _is_training_alive(ssh_client: SSHClient, timeout_seconds: int = 10) -> bool:
        """Check if training process is running (supports both Docker and direct execution)."""
        # First check for Docker container (ryotenkai_training_*)
        if docker_is_container_running(
            ssh_client, name_filter="ryotenkai_training", timeout_seconds=min(5, timeout_seconds)
        ):
            return True

        # Fallback: check for direct Python process (cloud mode)
        success, stdout, _ = ssh_client.exec_command(
            command="ps aux | grep -E 'python.*train' | grep -v grep",
            silent=True,
            timeout=timeout_seconds,
        )
        return success and bool(stdout.strip())

    @staticmethod
    def _parse_cgroup_ram(cgroup_out: str) -> tuple[float, float] | None:
        """
        Parse cgroup memory output into (total_gb, used_gb).

        Supports cgroup v2 (memory.max / memory.current) and
        cgroup v1 (memory.limit_in_bytes / memory.usage_in_bytes).
        Returns None if data is absent, incomplete, or reports an unlimited limit.

        Expected input format — one "key=value" pair per line, e.g.:
            memory.max=33285996544
            memory.current=1073741824
        """
        # Values above this threshold are treated as "unlimited" (not a real container limit).
        # cgroup v1 commonly uses 9223372036854771712 (~8 EiB) to signal "no limit".
        max_reasonable_bytes = 1 << 60  # 1 EiB

        def _parse_int(value: str | None) -> int | None:
            if not value:
                return None
            s = value.strip()
            if not s or s == "max" or not s.isdigit():
                return None
            return int(s)

        kv: dict[str, str] = {}
        for line in cgroup_out.splitlines():
            line = line.strip()
            if "=" in line:
                k, _, v = line.partition("=")
                kv[k.strip()] = v.strip()

        total_bytes: int | None = None
        used_bytes: int | None = None

        # cgroup v2
        v2_max = _parse_int(kv.get("memory.max"))
        v2_cur = _parse_int(kv.get("memory.current"))
        if v2_max is not None and v2_cur is not None and 0 < v2_max < max_reasonable_bytes:
            total_bytes, used_bytes = v2_max, max(0, v2_cur)

        # cgroup v1 (only if v2 data was absent / unlimited)
        if total_bytes is None:
            v1_lim = _parse_int(kv.get("memory.limit_in_bytes"))
            v1_use = _parse_int(kv.get("memory.usage_in_bytes"))
            if v1_lim is not None and v1_use is not None and 0 < v1_lim < max_reasonable_bytes:
                total_bytes, used_bytes = v1_lim, max(0, v1_use)

        if total_bytes is None or used_bytes is None:
            return None

        used_bytes = min(used_bytes, total_bytes)
        return total_bytes / (1024**3), used_bytes / (1024**3)

    @staticmethod
    def _parse_meminfo_ram(meminfo_out: str) -> tuple[float, float] | None:
        """
        Parse /proc/meminfo output into (total_gb, used_gb).

        Returns None if MemTotal is missing or the output is malformed.
        Note: inside containers this may reflect *host* memory, not the container limit.
        """
        try:
            mem: dict[str, int] = {}
            for line in meminfo_out.splitlines():
                if ":" in line:
                    key, _, rest = line.partition(":")
                    tokens = rest.strip().split()
                    if tokens and tokens[0].isdigit():
                        mem[key.strip()] = int(tokens[0])

            if "MemTotal" not in mem:
                return None

            total_kb = mem["MemTotal"]
            available_kb = mem.get("MemAvailable", mem.get("MemFree", 0))
            used_kb = total_kb - available_kb
            return total_kb / (1024 * 1024), used_kb / (1024 * 1024)  # kB → GiB-ish
        except (IndexError, TypeError, ValueError):
            return None

    @staticmethod
    def _get_resources(ssh_client: SSHClient) -> dict:
        """Get pod resource usage (GPU + RAM)."""
        # GPU stats
        gpu_success, gpu_out, _ = ssh_client.exec_command(
            command="nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits",
            silent=True,
        )

        gpu_util = 0.0
        vram_used_mb = 0.0
        vram_total_mb = 0.0
        vram_pct = 0.0
        gpu_temp = 0.0

        if gpu_success and gpu_out.strip():
            parts = gpu_out.strip().split(",")
            if len(parts) >= 4:
                gpu_util = float(parts[0].strip())
                vram_used_mb = float(parts[1].strip())
                vram_total_mb = float(parts[2].strip())
                vram_pct = (vram_used_mb / vram_total_mb * 100) if vram_total_mb > 0 else 0.0
                gpu_temp = float(parts[3].strip())

        ram_total_gb = 0.0
        ram_used_gb = 0.0

        # 1. cgroup: container-aware limit (Linux).
        # /proc/meminfo reflects host memory inside many container runtimes, so cgroup is preferred.
        cgroup_success, cgroup_out, _ = ssh_client.exec_command(
            command=(
                "cat /sys/fs/cgroup/memory.max 2>/dev/null && "
                "echo memory.max=$(cat /sys/fs/cgroup/memory.max 2>/dev/null); "
                "echo memory.current=$(cat /sys/fs/cgroup/memory.current 2>/dev/null); "
                "echo memory.limit_in_bytes=$(cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null); "
                "echo memory.usage_in_bytes=$(cat /sys/fs/cgroup/memory/memory.usage_in_bytes 2>/dev/null)"
            ),
            silent=True,
        )
        if cgroup_success and cgroup_out.strip():
            result = TrainingMonitor._parse_cgroup_ram(cgroup_out)
            if result is not None:
                ram_total_gb, ram_used_gb = result

        # 2. /proc/meminfo — Linux fallback (may show host memory in containers).
        if ram_total_gb == 0.0:
            ram_success, ram_out, _ = ssh_client.exec_command(command="cat /proc/meminfo", silent=True)
            if ram_success and ram_out.strip():
                result = TrainingMonitor._parse_meminfo_ram(ram_out)
                if result is not None:
                    ram_total_gb, ram_used_gb = result

        return {
            "gpu_util": gpu_util,
            "vram_used_gb": vram_used_mb / 1024,  # MB → GB
            "vram_total_gb": vram_total_mb / 1024,
            "vram_pct": vram_pct,
            "gpu_temp": gpu_temp,
            "ram_used_gb": ram_used_gb,
            "ram_total_gb": ram_total_gb,
        }

    def _check_marker(self, ssh_client: SSHClient, marker_name: str, *, timeout_seconds: int = 10) -> bool:
        """Check if marker file exists."""
        workspace = self._workspace_path
        success, content, _ = ssh_client.exec_command(
            command=f"test -f {workspace}/{marker_name} && echo EXISTS",
            silent=True,
            timeout=timeout_seconds,
        )
        return success and "EXISTS" in content

    def _read_marker_content(self, ssh_client: SSHClient, marker_name: str, *, timeout_seconds: int = 10) -> str:
        """Read marker file content."""
        workspace = self._workspace_path
        success, content, _ = ssh_client.exec_command(
            command=f"cat {workspace}/{marker_name}",
            silent=True,
            timeout=timeout_seconds,
        )
        return content.strip() if success else "Unknown error"

    def _display_last_log_lines(self, n: int = 30) -> None:
        """Display last N lines from remote training log."""
        if not self._log_manager:
            return

        lines = self._log_manager.get_last_lines(n)
        if not lines:
            logger.warning("No training log content available")
            return

        logger.info("┌" + "─" * TRAINING_MONITOR_LINE_WIDTH)
        logger.info(f"| Training Log (last {len(lines)} lines)")
        logger.info("├" + "─" * TRAINING_MONITOR_LINE_WIDTH)
        for line in lines:
            # Strip ANSI codes for cleaner output
            clean_line = (
                line.replace("\x1b[0m", "")
                .replace("\x1b[32m", "")
                .replace("\x1b[31m", "")
                .replace("\x1b[33m", "")
                .replace("\x1b[36m", "")
            )
            logger.info(f"│ {clean_line}")
        logger.info("└" + "─" * TRAINING_MONITOR_LINE_WIDTH)

        if self._log_manager.local_path.exists():
            logger.info(f"Full log: {self._log_manager.local_path}")

    def _execute_mock(self, context: dict[str, Any], pod_id: str | None) -> Result[dict[str, Any], AppError]:
        """
        Mock execution for testing without real RunPod training monitoring.
        Simulates training progress with realistic timing.
        """
        strategies = self.config.training.get_strategy_chain()
        if strategies:
            total_epochs = sum((s.hyperparams.epochs or 0) for s in strategies)
            if total_epochs <= 0:
                total_epochs = self.config.training.hyperparams.epochs or 3
        else:
            total_epochs = self.config.training.hyperparams.epochs or 3
        logger.info(f"[MOCK] Monitoring training for pod: {pod_id or 'mock-pod'}")
        logger.info(f"[MOCK] Simulating training for {total_epochs} epoch(s)...")

        # Simulate training phases
        total_steps = 10  # Simulate 10 training steps

        for step in range(1, total_steps + 1):
            # Simulate step progress
            time.sleep(0.5)  # 500ms per step for quick mock

            # Log progress every few steps
            if step % 3 == 0 or step == total_steps:
                progress = (step / total_steps) * 100
                mock_loss = TRAINING_MONITOR_MOCK_LOSS_INIT - (step * TRAINING_MONITOR_MOCK_LOSS_DECAY)
                logger.info(
                    f"[MOCK] Step {step}/{total_steps} | "
                    f"Progress: {progress:.0f}% | "
                    f"Loss: {mock_loss:.3f} | "
                    f"GPU: 85% | VRAM: 12.5/16GB"
                )

        logger.info("[MOCK] Training completed successfully!")
        logger.info("[MOCK] Final metrics: loss=1.0, accuracy=0.85")

        # Return mock training info
        return Ok(
            self.update_context(
                context,
                {
                    "status": "completed",
                    "training_info": {
                        "runtime_seconds": total_steps * 0.5,
                        "final_loss": 1.0,
                        "final_accuracy": 0.85,
                        "total_steps": total_steps,
                        "mock": True,
                    },
                },
            )
        )
