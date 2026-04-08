"""
RunPod Pod Session — inline lifecycle management for the evaluation stage.

Responsibilities:
- Start a stopped pod and wait for SSH readiness
- Open an SSH tunnel (localhost:{port} → 127.0.0.1:{port} inside pod)
- Run LoRA merge (idempotent via config-hash check)
- Launch vLLM in background (nohup), wait for /v1/models health
- Clean up: kill vLLM, close tunnel, delete pod

Used exclusively by RunPodPodInferenceProvider.activate_for_eval() /
deactivate_after_eval(). Not intended for interactive sessions (see chat_inference.py
for that use case).

Design:
- All blocking operations use explicit timeouts to avoid hanging the pipeline.
- SSH subprocess errors are propagated as Err(ProviderError) — never raise.
- Timeouts are hardcoded constants (not user config) as per project policy.
"""

from __future__ import annotations

import shlex
import socket
import subprocess
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path  # noqa: TC003
from typing import TYPE_CHECKING, Any

from src.providers.runpod.inference.pods.constants import POD_MERGE_SCRIPT
from src.utils.logger import logger
from src.utils.result import Err, Ok, ProviderError, Result

if TYPE_CHECKING:
    from src.providers.runpod.inference.pods.api_client import RunPodPodsRESTClient

# ---------------------------------------------------------------------------
# Timeouts (hardcoded: not exposed to user config)
# ---------------------------------------------------------------------------

# How long to wait for the pod to become RUNNING + SSH port mapped + TCP ready
_POD_SSH_READY_TIMEOUT_SEC = 600

# How long to wait for vLLM /v1/models health endpoint
_VLLM_READY_TIMEOUT_SEC = 600

# Merge LoRA timeout (can take several minutes for large models)
_MERGE_TIMEOUT_SEC = 3600

# Poll interval for SSH / health-check waits
_POLL_INTERVAL_SEC = 5

# SSH exec timeout for short commands
_SSH_SHORT_TIMEOUT_SEC = 30

# SSH exec timeout for merge
_SSH_MERGE_TIMEOUT_SEC = _MERGE_TIMEOUT_SEC

# SSH port inside pod containers
_SSH_CONTAINER_PORT = 22  # noqa: WPS432

# Truncation limits for error messages in logs
_STDERR_SNIPPET_LEN = 500  # noqa: WPS432
_STDOUT_SNIPPET_LEN = 200  # noqa: WPS432
_LOG_TAIL_LEN = 4000  # noqa: WPS432

# vLLM defaults (used when config omits the key)
_VLLM_DEFAULT_MAX_MODEL_LEN = 4096  # noqa: WPS432
_VLLM_DEFAULT_GPU_MEM_UTIL = 0.90  # noqa: WPS432


_HTTP_OK_STATUS = 200  # noqa: WPS432
_HTTP_SERVER_ERROR_THRESHOLD = 500  # noqa: WPS432


_SSH_OPT = "-o"  # noqa: WPS226


def _ssh_opts() -> list[str]:
    """Return common SSH option flags used in all subprocess calls."""
    return [  # noqa: WPS226
        _SSH_OPT,
        "StrictHostKeyChecking=no",
        _SSH_OPT,
        "UserKnownHostsFile=/dev/null",
        _SSH_OPT,
        "BatchMode=yes",
        _SSH_OPT,
        "PasswordAuthentication=no",
        _SSH_OPT,
        "LogLevel=ERROR",
    ]


@dataclass
class PodSessionState:
    """
    Mutable state captured during activate_for_eval().
    Passed back to deactivate_after_eval() so it can clean up correctly.
    """

    pod_id: str
    public_ip: str = ""
    ssh_port: int = 0
    tunnel_proc: subprocess.Popen[bytes] | None = None
    vllm_pid_file: str = ""
    endpoint_url: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def activate(
    *,
    api: RunPodPodsRESTClient,
    pod_id: str,
    key_path: Path,
    serve_port: int,
    merged_dir: str,
    run_dir: str,
    hf_cache_dir: str,
    config_hash: str,
    base_model_id: str,
    adapter_ref: str,
    trust_remote_code: bool,
    hf_token: str,
    vllm_cfg: dict[str, Any],
    pid_file: str,
    log_file: str,
    hash_file: str,
    lock_dir: str,
) -> Result[PodSessionState, ProviderError]:
    """
    Bring up a RunPod pod for evaluation:

    1. Start pod (if stopped)
    2. Wait for SSH readiness
    3. Open SSH tunnel
    4. Run LoRA merge (idempotent)
    5. Start vLLM
    6. Wait for /v1/models health
    7. Return PodSessionState

    All steps return Err on failure — no exceptions propagate.
    """
    state = PodSessionState(pod_id=pod_id)

    # --- Step 1: start pod --------------------------------------------------
    logger.info("[EVAL] Starting RunPod Pod %s for evaluation ...", pod_id)
    start_res = _start_pod(api=api, pod_id=pod_id)
    if start_res.is_failure():
        return Err(start_res.unwrap_err())  # type: ignore[union-attr]

    # --- Step 2: wait for SSH -----------------------------------------------
    logger.info("[EVAL] Waiting for SSH readiness (timeout=%ds) ...", _POD_SSH_READY_TIMEOUT_SEC)
    ssh_res = _wait_for_ssh(api=api, pod_id=pod_id, timeout_sec=_POD_SSH_READY_TIMEOUT_SEC)
    if ssh_res.is_failure():
        return Err(ssh_res.unwrap_err())  # type: ignore[union-attr]
    public_ip, ssh_port = ssh_res.unwrap()
    state.public_ip = public_ip
    state.ssh_port = ssh_port
    logger.info("[EVAL] Pod SSH ready: %s:%d", public_ip, ssh_port)

    # --- Step 3: open SSH tunnel --------------------------------------------
    tunnel_res = _open_tunnel(
        host=public_ip,
        ssh_port=ssh_port,
        key_path=key_path,
        local_port=serve_port,
        remote_port=serve_port,
    )
    if tunnel_res.is_failure():
        return Err(tunnel_res.unwrap_err())  # type: ignore[union-attr]
    state.tunnel_proc = tunnel_res.unwrap()
    logger.info("[EVAL] SSH tunnel open: localhost:%d → 127.0.0.1:%d (inside pod)", serve_port, serve_port)

    # --- Step 4: ensure dirs + merge LoRA ----------------------------------
    dirs_res = _ssh_exec(
        host=public_ip,
        ssh_port=ssh_port,
        key_path=key_path,
        command=f"mkdir -p {shlex.quote(hf_cache_dir)} {shlex.quote(run_dir)}",
        timeout_sec=_SSH_SHORT_TIMEOUT_SEC,
    )
    if dirs_res.is_failure():
        return Err(  # type: ignore[union-attr]
            ProviderError(
                message=f"[EVAL] mkdir failed: {dirs_res.unwrap_err()}",
                code="POD_MKDIR_FAILED",
            )
        )

    merge_res = _ensure_merge(
        host=public_ip,
        ssh_port=ssh_port,
        key_path=key_path,
        config_hash=config_hash,
        hash_file=hash_file,
        merged_dir=merged_dir,
        hf_cache_dir=hf_cache_dir,
        base_model_id=base_model_id,
        adapter_ref=adapter_ref,
        trust_remote_code=trust_remote_code,
        hf_token=hf_token,
    )
    if merge_res.is_failure():
        return Err(merge_res.unwrap_err())  # type: ignore[union-attr]

    # --- Step 5: acquire lock + start vLLM ---------------------------------
    lock_res = _acquire_lock(
        host=public_ip,
        ssh_port=ssh_port,
        key_path=key_path,
        lock_dir=lock_dir,
        pid_file=pid_file,
    )
    if lock_res.is_failure():
        return Err(lock_res.unwrap_err())  # type: ignore[union-attr]

    vllm_res = _start_vllm(
        host=public_ip,
        ssh_port=ssh_port,
        key_path=key_path,
        serve_port=serve_port,
        merged_dir=merged_dir,
        pid_file=pid_file,
        log_file=log_file,
        trust_remote_code=trust_remote_code,
        vllm_cfg=vllm_cfg,
    )
    if vllm_res.is_failure():
        return Err(vllm_res.unwrap_err())  # type: ignore[union-attr]
    state.vllm_pid_file = pid_file

    # --- Step 6: wait for vLLM health ---------------------------------------
    endpoint_url = f"http://127.0.0.1:{serve_port}/v1"
    health_url = f"{endpoint_url}/models"
    logger.info("[EVAL] Waiting for vLLM health: %s (timeout=%ds)", health_url, _VLLM_READY_TIMEOUT_SEC)
    health_res = _wait_http_ok(url=health_url, timeout_sec=_VLLM_READY_TIMEOUT_SEC, interval_sec=_POLL_INTERVAL_SEC)
    if health_res.is_failure():
        # Dump vLLM log tail for diagnostics
        _dump_vllm_log(
            host=public_ip,
            ssh_port=ssh_port,
            key_path=key_path,
            log_file=log_file,
        )
        return Err(health_res.unwrap_err())  # type: ignore[union-attr]

    state.endpoint_url = endpoint_url
    logger.info("[EVAL] vLLM ready: %s", endpoint_url)
    return Ok(state)


def deactivate(
    *,
    api: RunPodPodsRESTClient,
    state: PodSessionState,
    key_path: Path,
) -> Result[None, ProviderError]:
    """
    Shut down the evaluation session:
    1. Kill vLLM process inside the pod (best-effort)
    2. Close SSH tunnel
    3. Delete pod (preserves Network Volume)
    """
    errors: list[str] = []

    # --- Kill vLLM ----------------------------------------------------------
    if state.vllm_pid_file and state.public_ip and state.ssh_port:
        logger.info("[EVAL] Killing vLLM process inside pod %s ...", state.pod_id)
        kill_cmd = (
            f"set +e; "
            f"if test -f {shlex.quote(state.vllm_pid_file)}; then "
            f"  pid=$(cat {shlex.quote(state.vllm_pid_file)} 2>/dev/null || true); "
            f'  if test -n "$pid"; then kill "$pid" 2>/dev/null || true; sleep 2; kill -9 "$pid" 2>/dev/null || true; fi; '
            f"  rm -f {shlex.quote(state.vllm_pid_file)}; "
            f"fi"
        )
        kill_res = _ssh_exec(
            host=state.public_ip,
            ssh_port=state.ssh_port,
            key_path=key_path,
            command=kill_cmd,
            timeout_sec=_SSH_SHORT_TIMEOUT_SEC,
        )
        if kill_res.is_failure():
            err = f"vLLM kill warning (non-fatal): {kill_res.unwrap_err()}"
            logger.warning("[EVAL] %s", err)
            errors.append(err)
        else:
            logger.info("[EVAL] vLLM process terminated")

    # --- Close SSH tunnel ---------------------------------------------------
    if state.tunnel_proc is not None:
        logger.info("[EVAL] Closing SSH tunnel ...")
        try:
            state.tunnel_proc.terminate()
            try:
                state.tunnel_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                state.tunnel_proc.kill()
        except Exception as e:
            err = f"SSH tunnel close warning (non-fatal): {e}"
            logger.warning("[EVAL] %s", err)
            errors.append(err)

    # --- Delete pod (preserves Network Volume) ------------------------------
    logger.info("[EVAL] Deleting RunPod Pod %s (volume preserved) ...", state.pod_id)
    del_res = api.delete_pod(pod_id=state.pod_id)
    if del_res.is_failure():
        pod_err = del_res.unwrap_err()  # type: ignore[union-attr]
        logger.warning("[EVAL] Pod delete failed (non-fatal): %s", pod_err)
        errors.append(f"delete_pod: {pod_err}")
    else:
        logger.info("[EVAL] Pod %s deleted successfully", state.pod_id)

    if errors:
        # Non-fatal: return Err so caller can log, but pipeline continues
        return Err(ProviderError(message="; ".join(errors), code="POD_DEACTIVATE_PARTIAL_FAILURE"))
    return Ok(None)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _start_pod(*, api: RunPodPodsRESTClient, pod_id: str) -> Result[None, ProviderError]:
    """Start a stopped pod. No-op if already running."""
    get_res = api.get_pod(pod_id=pod_id)
    if get_res.is_failure():
        return Err(get_res.unwrap_err())  # type: ignore[union-attr]
    pod = get_res.unwrap()
    status = str(pod.get("desiredStatus") or "")

    if status == "RUNNING":
        logger.debug("[EVAL] Pod %s is already RUNNING", pod_id)
        return Ok(None)

    start_res = api.start_pod(pod_id=pod_id)
    if start_res.is_failure():
        return Err(start_res.unwrap_err())  # type: ignore[union-attr]
    return Ok(None)


def _wait_for_ssh(
    *,
    api: RunPodPodsRESTClient,
    pod_id: str,
    timeout_sec: int,
) -> Result[tuple[str, int], ProviderError]:
    """
    Poll RunPod API until pod is RUNNING with a public IP + SSH port mapping
    and the SSH port is accepting TCP connections.
    """
    deadline = time.time() + timeout_sec
    last_preview = ""
    _INFO_INTERVAL_SEC = 30
    last_info_ts = 0.0

    while time.time() < deadline:
        get_res = api.get_pod(pod_id=pod_id)
        if get_res.is_failure():
            time.sleep(_POLL_INTERVAL_SEC)
            continue

        pod = get_res.unwrap()
        status = str(pod.get("desiredStatus") or "")
        public_ip = str(pod.get("publicIp") or "").strip()
        ssh_port: int | None = _parse_ssh_port(pod.get("portMappings") or {})

        tcp_ok: bool | None = None
        if status == "RUNNING" and public_ip and ssh_port:
            try:
                with socket.create_connection((public_ip, int(ssh_port)), timeout=3):
                    tcp_ok = True
            except Exception:
                tcp_ok = False

            if tcp_ok:
                return Ok((public_ip, int(ssh_port)))

        remaining = int(max(0, deadline - time.time()))
        elapsed = int(timeout_sec - remaining)
        preview = (
            f"status={status or '∅'} ip={public_ip or '∅'} "
            f"ssh_port={ssh_port or '∅'} tcp={'OK' if tcp_ok else 'NO' if tcp_ok is False else '∅'}"
        )

        now = time.time()
        if preview != last_preview:
            logger.info("[EVAL] SSH wait: %s (elapsed %ds/%ds)", preview, elapsed, timeout_sec)
            last_preview = preview
            last_info_ts = now
        elif now - last_info_ts >= _INFO_INTERVAL_SEC:
            logger.info("[EVAL] SSH wait: %s (elapsed %ds/%ds)", preview, elapsed, timeout_sec)
            last_info_ts = now

        time.sleep(_POLL_INTERVAL_SEC)

    return Err(
        ProviderError(
            message=f"runpod: pod {pod_id} not SSH-ready within {timeout_sec}s. Last state: {last_preview}",
            code="POD_SSH_READY_TIMEOUT",
        )
    )


def _parse_ssh_port(mappings: Any) -> int | None:
    """Extract host SSH port from RunPod portMappings (dict or list shape)."""
    if isinstance(mappings, dict):
        for key in ("22", 22, "22/tcp", "tcp/22"):
            if key not in mappings:
                continue
            v = mappings[key]
            if isinstance(v, dict):
                v = v.get("hostPort") or v.get("publicPort") or v.get("port")
            try:
                return int(v)  # type: ignore[arg-type]
            except Exception:
                continue
    elif isinstance(mappings, list):
        for item in mappings:
            if not isinstance(item, dict):
                continue
            cport = item.get("containerPort") or item.get("internalPort") or item.get("port")
            hport = item.get("hostPort") or item.get("externalPort") or item.get("publicPort")
            if cport is None or hport is None:
                continue
            try:
                if int(cport) == _SSH_CONTAINER_PORT:
                    return int(hport)
            except Exception:
                continue
    return None


def _open_tunnel(
    *,
    host: str,
    ssh_port: int,
    key_path: Path,
    local_port: int,
    remote_port: int,
) -> Result[subprocess.Popen[bytes], ProviderError]:
    """
    Open a persistent SSH tunnel in background.
    Returns the Popen handle so caller can terminate it in deactivate().
    """
    cmd = [
        "ssh",
        "-N",
        "-i",
        str(key_path),
        "-p",
        str(ssh_port),
        *_ssh_opts(),
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "ServerAliveCountMax=3",
        "-L",
        f"{local_port}:127.0.0.1:{remote_port}",
        f"root@{host}",
    ]
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
        )
        # Brief wait to catch immediate failures (e.g. port already in use)
        time.sleep(2)
        if proc.poll() is not None:
            return Err(
                ProviderError(
                    message=(
                        f"SSH tunnel process exited immediately (returncode={proc.returncode}). "
                        f"Check that port {local_port} is free and SSH key is valid."
                    ),
                    code="POD_SSH_TUNNEL_FAILED",
                )
            )
        return Ok(proc)
    except Exception as e:
        return Err(ProviderError(message=f"Failed to open SSH tunnel: {e}", code="POD_SSH_TUNNEL_FAILED"))


def _ssh_exec(
    *,
    host: str,
    ssh_port: int,
    key_path: Path,
    command: str,
    timeout_sec: int,
) -> Result[str, ProviderError]:
    """
    Execute a shell command remotely via SSH.
    Returns Ok(stdout) on returncode==0, Err(ProviderError) otherwise.
    """
    remote_cmd = f"bash -lc {shlex.quote(command)}"
    full_cmd = [
        "ssh",
        "-i",
        str(key_path),
        "-p",
        str(ssh_port),
        *_ssh_opts(),
        f"root@{host}",
        remote_cmd,
    ]
    try:
        result = subprocess.run(
            full_cmd,
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=timeout_sec,
        )
        if result.returncode != 0:
            stderr = (result.stderr or "").strip()[:_STDERR_SNIPPET_LEN]
            stdout = (result.stdout or "").strip()[:_STDOUT_SNIPPET_LEN]
            return Err(
                ProviderError(
                    message=f"SSH command failed (rc={result.returncode}): stderr={stderr!r} stdout={stdout!r}",
                    code="POD_SSH_EXEC_FAILED",
                )
            )
        return Ok((result.stdout or "").strip())
    except subprocess.TimeoutExpired:
        return Err(
            ProviderError(
                message=f"SSH command timed out after {timeout_sec}s: {command[:100]!r}",
                code="POD_SSH_EXEC_TIMEOUT",
            )
        )
    except Exception as e:
        return Err(ProviderError(message=f"SSH exec error: {e}", code="POD_SSH_EXEC_ERROR"))


def _ensure_merge(
    *,
    host: str,
    ssh_port: int,
    key_path: Path,
    config_hash: str,
    hash_file: str,
    merged_dir: str,
    hf_cache_dir: str,
    base_model_id: str,
    adapter_ref: str,
    trust_remote_code: bool,
    hf_token: str,
) -> Result[None, ProviderError]:
    """Run LoRA merge on the pod (idempotent: skip if config_hash matches)."""
    check_cmd = (
        f"test -f {shlex.quote(hash_file)} && "
        f"test -f {shlex.quote(merged_dir)}/config.json && "
        f'test "$(cat {shlex.quote(hash_file)} 2>/dev/null || true)" = {shlex.quote(config_hash)} '
        f"&& echo OK || echo NO"
    )
    chk_res = _ssh_exec(
        host=host,
        ssh_port=ssh_port,
        key_path=key_path,
        command=check_cmd,
        timeout_sec=_SSH_SHORT_TIMEOUT_SEC,
    )
    if chk_res.is_success() and chk_res.unwrap().strip() == "OK":
        logger.info("[EVAL] LoRA merge skipped (config_hash matches)")
        return Ok(None)

    logger.info("[EVAL] Running LoRA merge (base=%r adapter=%r) ...", base_model_id, adapter_ref)
    trust_arg = "--trust-remote-code" if trust_remote_code else ""
    merge_cmd = (
        f"HF_TOKEN={shlex.quote(hf_token)} "
        f"python3 {POD_MERGE_SCRIPT} "
        f"--base-model {shlex.quote(base_model_id)} "
        f"--adapter {shlex.quote(adapter_ref)} "
        f"--output {shlex.quote(merged_dir)} "
        f"--cache-dir {shlex.quote(hf_cache_dir)} "
        f"{trust_arg}"
    ).strip()

    merge_res = _ssh_exec(
        host=host,
        ssh_port=ssh_port,
        key_path=key_path,
        command=merge_cmd,
        timeout_sec=_SSH_MERGE_TIMEOUT_SEC,
    )
    if merge_res.is_failure():
        return Err(  # type: ignore[union-attr]
            ProviderError(
                message=f"LoRA merge failed: {merge_res.unwrap_err()}",
                code="POD_MERGE_FAILED",
            )
        )

    # Write hash file
    write_hash_res = _ssh_exec(
        host=host,
        ssh_port=ssh_port,
        key_path=key_path,
        command=f"echo {shlex.quote(config_hash)} > {shlex.quote(hash_file)}",
        timeout_sec=_SSH_SHORT_TIMEOUT_SEC,
    )
    if write_hash_res.is_failure():
        logger.warning("[EVAL] Failed to write config hash file (non-fatal): %s", write_hash_res.unwrap_err())

    logger.info("[EVAL] LoRA merge completed")
    return Ok(None)


def _acquire_lock(
    *,
    host: str,
    ssh_port: int,
    key_path: Path,
    lock_dir: str,
    pid_file: str,
) -> Result[None, ProviderError]:
    """Acquire exclusive volume lock on the pod (mkdir-based)."""
    lock_cmd = (
        f"set +e; "
        f"mkdir {shlex.quote(lock_dir)} 2>/dev/null; rc=$?; "
        f"if [ $rc -eq 0 ]; then echo ACQUIRED; exit 0; fi; "
        f"if test -f {shlex.quote(pid_file)}; then "
        f"  pid=$(cat {shlex.quote(pid_file)} 2>/dev/null || true); "
        f'  if test -n "$pid" && kill -0 "$pid" 2>/dev/null; then echo BUSY; exit 2; fi; '
        f"fi; "
        f"rm -rf {shlex.quote(lock_dir)}; "
        f"mkdir {shlex.quote(lock_dir)} 2>/dev/null; exit $?"
    )
    res = _ssh_exec(
        host=host,
        ssh_port=ssh_port,
        key_path=key_path,
        command=lock_cmd,
        timeout_sec=_SSH_SHORT_TIMEOUT_SEC,
    )
    if res.is_failure():
        out = str(res.unwrap_err())  # type: ignore[union-attr]
        if "BUSY" in out or "exit 2" in out:
            return Err(
                ProviderError(
                    message="Volume lock is held by another session (vLLM is alive). Cannot run evaluation concurrently.",
                    code="POD_LOCK_BUSY",
                )
            )
        return Err(ProviderError(message=f"Failed to acquire volume lock: {out}", code="POD_LOCK_FAILED"))
    return Ok(None)


def _start_vllm(
    *,
    host: str,
    ssh_port: int,
    key_path: Path,
    serve_port: int,
    merged_dir: str,
    pid_file: str,
    log_file: str,
    trust_remote_code: bool,
    vllm_cfg: dict[str, Any],
) -> Result[None, ProviderError]:
    """Start vLLM server inside the pod (nohup, background)."""
    tp = int(vllm_cfg.get("tensor_parallel_size") or 1)
    max_len = int(vllm_cfg.get("max_model_len") or _VLLM_DEFAULT_MAX_MODEL_LEN)
    gpu_mem = float(vllm_cfg.get("gpu_memory_utilization") or _VLLM_DEFAULT_GPU_MEM_UTIL)
    quant = vllm_cfg.get("quantization")
    eager = bool(vllm_cfg.get("enforce_eager") or False)

    args = [
        "vllm serve",
        shlex.quote(merged_dir),
        "--host",
        "127.0.0.1",
        "--port",
        str(serve_port),
        "--tensor-parallel-size",
        str(tp),
        "--max-model-len",
        str(max_len),
        "--gpu-memory-utilization",
        str(gpu_mem),
    ]
    if quant:
        args += ["--quantization", shlex.quote(str(quant))]
    if eager:
        args += ["--enforce-eager"]
    if trust_remote_code:
        args += ["--trust-remote-code"]

    start_cmd = (
        f"set +e; "
        f"python3 -m pip install -q \"setuptools<70.0.0\"; "
        f"if test -f {shlex.quote(pid_file)}; then "
        f"  pid=$(cat {shlex.quote(pid_file)} 2>/dev/null || true); "
        f"  if test -n \"$pid\" && kill -0 \"$pid\" 2>/dev/null; then "
        f"    kill \"$pid\" || true; sleep 2; kill -9 \"$pid\" 2>/dev/null || true; "
        f"  fi; "
        f"fi; "
        f"rm -f {shlex.quote(pid_file)}; "
        f"nohup {' '.join(args)} > {shlex.quote(log_file)} 2>&1 & echo $! > {shlex.quote(pid_file)}; "
        f"set -e"
    )

    res = _ssh_exec(
        host=host,
        ssh_port=ssh_port,
        key_path=key_path,
        command=start_cmd,
        timeout_sec=_SSH_SHORT_TIMEOUT_SEC,
    )
    if res.is_failure():
        return Err(  # type: ignore[union-attr]
            ProviderError(
                message=f"Failed to start vLLM: {res.unwrap_err()}",
                code="POD_VLLM_START_FAILED",
            )
        )
    logger.info("[EVAL] vLLM process launched (nohup), waiting for health ...")
    return Ok(None)


def _wait_http_ok(*, url: str, timeout_sec: int, interval_sec: float) -> Result[None, ProviderError]:
    """Poll url until HTTP 200. Returns Err on timeout."""
    deadline = time.time() + timeout_sec
    last_err = ""
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == _HTTP_OK_STATUS:
                    return Ok(None)
        except Exception as e:
            last_err = str(e)
        time.sleep(interval_sec)
    return Err(
        ProviderError(
            message=f"Timed out waiting for {url} to become healthy (timeout={timeout_sec}s, last_err={last_err!r})",
            code="POD_VLLM_HEALTH_TIMEOUT",
        )
    )


def _dump_vllm_log(
    *,
    host: str,
    ssh_port: int,
    key_path: Path,
    log_file: str,
) -> None:
    """Fetch and log the last 60 lines of vLLM log for diagnostics (best-effort)."""
    res = _ssh_exec(
        host=host,
        ssh_port=ssh_port,
        key_path=key_path,
        command=f"tail -n 60 {shlex.quote(log_file)}",
        timeout_sec=_SSH_SHORT_TIMEOUT_SEC,
    )
    if res.is_success():
        tail = res.unwrap().strip()
        if tail:
            logger.error("[EVAL] vLLM log tail:\n%s", tail[-_LOG_TAIL_LEN:])
        else:
            logger.warning("[EVAL] vLLM log is empty or unreadable")
    else:
        logger.warning("[EVAL] Could not fetch vLLM log: %s", res.unwrap_err())


__all__ = [
    "PodSessionState",
    "activate",
    "deactivate",
]
