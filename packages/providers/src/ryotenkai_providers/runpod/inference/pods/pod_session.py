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
- SSH subprocess errors propagate as typed exceptions
  (Phase A2 Batch 11 — was ``Err(ProviderError)``).
- Timeouts are hardcoded constants (not user config) as per project policy.
"""

from __future__ import annotations

import shlex
import subprocess
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path  # noqa: TC003
from typing import TYPE_CHECKING, Any

from ryotenkai_providers.runpod.inference.pods.constants import POD_MERGE_SCRIPT
from ryotenkai_shared.errors import (
    InferenceUnavailableError,
    ProviderUnavailableError,
    RyotenkAIError,
    SSHExecFailedError,
)
from ryotenkai_shared.utils.logger import logger

if TYPE_CHECKING:
    from ryotenkai_providers.runpod.inference.pods.api_client import RunPodPodsRESTClient

# ---------------------------------------------------------------------------
# Timeouts (hardcoded: not exposed to user config)
# ---------------------------------------------------------------------------

# How long to wait for the pod to become RUNNING + SSH port mapped + TCP ready
_POD_SSH_READY_TIMEOUT_SEC = 600
# Early-bailout windows (RUNNING-without-SSH-ports, RUNNING-no-ports-allocated)
# are tuned in the lifecycle policy module, not here.

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
) -> PodSessionState:
    """
    Bring up a RunPod pod for evaluation:

    1. Start pod (if stopped)
    2. Wait for SSH readiness
    3. Open SSH tunnel
    4. Run LoRA merge (idempotent)
    5. Start vLLM
    6. Wait for /v1/models health
    7. Return PodSessionState

    Phase A2 Batch 11: raise-based contract. On failure raises a typed
    exception (subclass of :class:`RyotenkAIError`). The inference
    provider facade catches and translates to ``Result``.
    """
    state = PodSessionState(pod_id=pod_id)

    # --- Step 1: start pod --------------------------------------------------
    logger.info("[EVAL] Starting RunPod Pod %s for evaluation ...", pod_id)
    _start_pod(api=api, pod_id=pod_id)

    # --- Step 2: wait for SSH -----------------------------------------------
    logger.info("[EVAL] Waiting for SSH readiness (timeout=%ds) ...", _POD_SSH_READY_TIMEOUT_SEC)
    public_ip, ssh_port = _wait_for_ssh(api=api, pod_id=pod_id, timeout_sec=_POD_SSH_READY_TIMEOUT_SEC)
    state.public_ip = public_ip
    state.ssh_port = ssh_port
    logger.info("[EVAL] Pod SSH ready: %s:%d", public_ip, ssh_port)

    # --- Step 3: open SSH tunnel --------------------------------------------
    state.tunnel_proc = _open_tunnel(
        host=public_ip,
        ssh_port=ssh_port,
        key_path=key_path,
        local_port=serve_port,
        remote_port=serve_port,
    )
    logger.info("[EVAL] SSH tunnel open: localhost:%d → 127.0.0.1:%d (inside pod)", serve_port, serve_port)

    # --- Step 4: ensure dirs + merge LoRA ----------------------------------
    try:
        _ssh_exec(
            host=public_ip,
            ssh_port=ssh_port,
            key_path=key_path,
            command=f"mkdir -p {shlex.quote(hf_cache_dir)} {shlex.quote(run_dir)}",
            timeout_sec=_SSH_SHORT_TIMEOUT_SEC,
        )
    except RyotenkAIError as exc:
        raise SSHExecFailedError(
            detail=f"[EVAL] mkdir failed: {exc.detail or exc}",
            context={"code": "POD_MKDIR_FAILED"},
            cause=exc,
        ) from exc

    _ensure_merge(
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

    # --- Step 5: acquire lock + start vLLM ---------------------------------
    _acquire_lock(
        host=public_ip,
        ssh_port=ssh_port,
        key_path=key_path,
        lock_dir=lock_dir,
        pid_file=pid_file,
    )

    _start_vllm(
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
    state.vllm_pid_file = pid_file

    # --- Step 6: wait for vLLM health ---------------------------------------
    endpoint_url = f"http://127.0.0.1:{serve_port}/v1"
    health_url = f"{endpoint_url}/models"
    logger.info("[EVAL] Waiting for vLLM health: %s (timeout=%ds)", health_url, _VLLM_READY_TIMEOUT_SEC)
    try:
        _wait_http_ok(url=health_url, timeout_sec=_VLLM_READY_TIMEOUT_SEC, interval_sec=_POLL_INTERVAL_SEC)
    except RyotenkAIError:
        # Dump vLLM log tail for diagnostics
        _dump_vllm_log(
            host=public_ip,
            ssh_port=ssh_port,
            key_path=key_path,
            log_file=log_file,
        )
        raise

    state.endpoint_url = endpoint_url
    logger.info("[EVAL] vLLM ready: %s", endpoint_url)
    return state


def deactivate(
    *,
    api: RunPodPodsRESTClient,
    state: PodSessionState,
    key_path: Path,
) -> None:
    """
    Shut down the evaluation session:
    1. Kill vLLM process inside the pod (best-effort)
    2. Close SSH tunnel
    3. Delete pod (preserves Network Volume)

    Phase A2 Batch 11: raises :class:`ProviderUnavailableError` with
    ``context['code'] == 'POD_DEACTIVATE_PARTIAL_FAILURE'`` when at
    least one cleanup step fails. Best-effort: continues through all
    cleanup steps even if earlier ones fail.
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
        try:
            _ssh_exec(
                host=state.public_ip,
                ssh_port=state.ssh_port,
                key_path=key_path,
                command=kill_cmd,
                timeout_sec=_SSH_SHORT_TIMEOUT_SEC,
            )
            logger.info("[EVAL] vLLM process terminated")
        except RyotenkAIError as exc:
            err = f"vLLM kill warning (non-fatal): {exc.detail or exc}"
            logger.warning("[EVAL] %s", err)
            errors.append(err)

    # --- Close SSH tunnel ---------------------------------------------------
    if state.tunnel_proc is not None:
        logger.info("[EVAL] Closing SSH tunnel ...")
        try:
            state.tunnel_proc.terminate()
            try:
                state.tunnel_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                state.tunnel_proc.kill()
        except Exception as exc:
            err = f"SSH tunnel close warning (non-fatal): {exc}"
            logger.warning("[EVAL] %s", err)
            errors.append(err)

    # --- Delete pod (preserves Network Volume) ------------------------------
    logger.info("[EVAL] Deleting RunPod Pod %s (volume preserved) ...", state.pod_id)
    try:
        api.delete_pod(pod_id=state.pod_id)
        logger.info("[EVAL] Pod %s deleted successfully", state.pod_id)
    except RyotenkAIError as exc:
        logger.warning("[EVAL] Pod delete failed (non-fatal): %s", exc)
        errors.append(f"delete_pod: {exc.detail or exc}")

    if errors:
        raise ProviderUnavailableError(
            detail="; ".join(errors),
            context={"code": "POD_DEACTIVATE_PARTIAL_FAILURE"},
        )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _start_pod(*, api: RunPodPodsRESTClient, pod_id: str) -> None:
    """Start a stopped pod. No-op if already running."""
    pod = api.get_pod(pod_id=pod_id)
    status = str(pod.get("desiredStatus") or "")

    if status == "RUNNING":
        logger.debug("[EVAL] Pod %s is already RUNNING", pod_id)
        return

    api.start_pod(pod_id=pod_id)


def _wait_for_ssh(
    *,
    api: RunPodPodsRESTClient,
    pod_id: str,
    timeout_sec: int,
) -> tuple[str, int]:
    """Wait for the eval pod to be SSH-ready (RUNNING + endpoint + TCP).

    Thin wrapper over the canonical :class:`PodSshWaiter`. Uses
    :data:`INFERENCE_PROFILE` thresholds, overriding only
    ``total_timeout_s`` to honour the caller's ``timeout_sec``.

    Reshapes the result back to the historical ``(host, port)`` tuple
    so the rest of ``activate()`` stays untouched. Cancel propagation
    (``PipelineCancelled`` from ``sleep_cancellable``) is intentionally
    not caught — the inference provider's cleanup hook will handle it.

    Phase A2 Batch 11: raises typed exceptions from the waiter.
    """
    from dataclasses import replace

    from ryotenkai_providers.runpod.lifecycle import (
        INFERENCE_PROFILE,
        PodSshWaiter,
    )
    from ryotenkai_providers.runpod.pod_control import RunPodInferencePodControl

    policy = (
        INFERENCE_PROFILE
        if timeout_sec == INFERENCE_PROFILE.total_timeout_s
        else replace(INFERENCE_PROFILE, total_timeout_s=int(timeout_sec))
    )
    waiter = PodSshWaiter(
        query=RunPodInferencePodControl(api=api),
        policy=policy,
        log=lambda level, msg: getattr(logger, level, logger.info)(f"[EVAL] {msg}"),
    )
    snapshot = waiter.wait(pod_id)
    ssh = snapshot.ssh_endpoint
    # Defensive: ``is_ready`` (which the waiter pinned before returning) already
    # guarantees this — assert for static analysis only.
    assert ssh is not None
    return ssh.host, ssh.port


def _open_tunnel(
    *,
    host: str,
    ssh_port: int,
    key_path: Path,
    local_port: int,
    remote_port: int,
) -> subprocess.Popen[bytes]:
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
            raise SSHExecFailedError(
                detail=(
                    f"SSH tunnel process exited immediately (returncode={proc.returncode}). "
                    f"Check that port {local_port} is free and SSH key is valid."
                ),
                context={"code": "POD_SSH_TUNNEL_FAILED"},
            )
        return proc
    except SSHExecFailedError:
        raise
    except Exception as exc:
        raise SSHExecFailedError(
            detail=f"Failed to open SSH tunnel: {exc}",
            context={"code": "POD_SSH_TUNNEL_FAILED"},
            cause=exc,
        ) from exc


def _ssh_exec(
    *,
    host: str,
    ssh_port: int,
    key_path: Path,
    command: str,
    timeout_sec: int,
) -> str:
    """
    Execute a shell command remotely via SSH.
    Returns stdout on returncode==0, raises :class:`SSHExecFailedError`
    otherwise.
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
            raise SSHExecFailedError(
                detail=f"SSH command failed (rc={result.returncode}): stderr={stderr!r} stdout={stdout!r}",
                context={"code": "POD_SSH_EXEC_FAILED"},
            )
        return (result.stdout or "").strip()
    except subprocess.TimeoutExpired as exc:
        raise SSHExecFailedError(
            detail=f"SSH command timed out after {timeout_sec}s: {command[:100]!r}",
            context={"code": "POD_SSH_EXEC_TIMEOUT"},
            cause=exc,
        ) from exc
    except SSHExecFailedError:
        raise
    except Exception as exc:
        raise SSHExecFailedError(
            detail=f"SSH exec error: {exc}",
            context={"code": "POD_SSH_EXEC_ERROR"},
            cause=exc,
        ) from exc


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
) -> None:
    """Run LoRA merge on the pod (idempotent: skip if config_hash matches)."""
    check_cmd = (
        f"test -f {shlex.quote(hash_file)} && "
        f"test -f {shlex.quote(merged_dir)}/config.json && "
        f'test "$(cat {shlex.quote(hash_file)} 2>/dev/null || true)" = {shlex.quote(config_hash)} '
        f"&& echo OK || echo NO"
    )
    try:
        chk = _ssh_exec(
            host=host,
            ssh_port=ssh_port,
            key_path=key_path,
            command=check_cmd,
            timeout_sec=_SSH_SHORT_TIMEOUT_SEC,
        )
        if chk.strip() == "OK":
            logger.info("[EVAL] LoRA merge skipped (config_hash matches)")
            return
    except RyotenkAIError:
        # If the check itself failed, fall through and try the merge.
        pass

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

    try:
        _ssh_exec(
            host=host,
            ssh_port=ssh_port,
            key_path=key_path,
            command=merge_cmd,
            timeout_sec=_SSH_MERGE_TIMEOUT_SEC,
        )
    except RyotenkAIError as exc:
        raise SSHExecFailedError(
            detail=f"LoRA merge failed: {exc.detail or exc}",
            context={"code": "POD_MERGE_FAILED"},
            cause=exc,
        ) from exc

    # Write hash file (best-effort)
    try:
        _ssh_exec(
            host=host,
            ssh_port=ssh_port,
            key_path=key_path,
            command=f"echo {shlex.quote(config_hash)} > {shlex.quote(hash_file)}",
            timeout_sec=_SSH_SHORT_TIMEOUT_SEC,
        )
    except RyotenkAIError as exc:
        logger.warning("[EVAL] Failed to write config hash file (non-fatal): %s", exc)

    logger.info("[EVAL] LoRA merge completed")


def _acquire_lock(
    *,
    host: str,
    ssh_port: int,
    key_path: Path,
    lock_dir: str,
    pid_file: str,
) -> None:
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
    try:
        _ssh_exec(
            host=host,
            ssh_port=ssh_port,
            key_path=key_path,
            command=lock_cmd,
            timeout_sec=_SSH_SHORT_TIMEOUT_SEC,
        )
    except RyotenkAIError as exc:
        out = exc.detail or str(exc)
        if "BUSY" in out or "exit 2" in out:
            raise SSHExecFailedError(
                detail="Volume lock is held by another session (vLLM is alive). Cannot run evaluation concurrently.",
                context={"code": "POD_LOCK_BUSY"},
                cause=exc,
            ) from exc
        raise SSHExecFailedError(
            detail=f"Failed to acquire volume lock: {out}",
            context={"code": "POD_LOCK_FAILED"},
            cause=exc,
        ) from exc


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
) -> None:
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

    try:
        _ssh_exec(
            host=host,
            ssh_port=ssh_port,
            key_path=key_path,
            command=start_cmd,
            timeout_sec=_SSH_SHORT_TIMEOUT_SEC,
        )
    except RyotenkAIError as exc:
        raise SSHExecFailedError(
            detail=f"Failed to start vLLM: {exc.detail or exc}",
            context={"code": "POD_VLLM_START_FAILED"},
            cause=exc,
        ) from exc
    logger.info("[EVAL] vLLM process launched (nohup), waiting for health ...")


def _wait_http_ok(*, url: str, timeout_sec: int, interval_sec: float) -> None:
    """Poll url until HTTP 200. Raises on timeout.

    Uses ``sleep_cancellable`` between probes so Ctrl+C during a vLLM
    cold-start (which can take 5+ minutes for big models) wakes the
    waiter immediately instead of blocking out the full interval.
    The exception propagates up to ``activate()``'s caller, where the
    inference-provider's cleanup hook synchronously tears down the pod.
    """
    from ryotenkai_shared.utils.cancellation import sleep_cancellable

    deadline = time.monotonic() + timeout_sec
    last_err = ""
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == _HTTP_OK_STATUS:
                    return
        except Exception as e:
            last_err = str(e)
        sleep_cancellable(interval_sec)
    raise InferenceUnavailableError(
        detail=f"Timed out waiting for {url} to become healthy (timeout={timeout_sec}s, last_err={last_err!r})",
        context={"code": "POD_VLLM_HEALTH_TIMEOUT"},
    )


def _dump_vllm_log(
    *,
    host: str,
    ssh_port: int,
    key_path: Path,
    log_file: str,
) -> None:
    """Fetch and log the last 60 lines of vLLM log for diagnostics (best-effort)."""
    try:
        tail = _ssh_exec(
            host=host,
            ssh_port=ssh_port,
            key_path=key_path,
            command=f"tail -n 60 {shlex.quote(log_file)}",
            timeout_sec=_SSH_SHORT_TIMEOUT_SEC,
        ).strip()
        if tail:
            logger.error("[EVAL] vLLM log tail:\n%s", tail[-_LOG_TAIL_LEN:])
        else:
            logger.warning("[EVAL] vLLM log is empty or unreadable")
    except RyotenkAIError as exc:
        logger.warning("[EVAL] Could not fetch vLLM log: %s", exc)


__all__ = [
    "PodSessionState",
    "activate",
    "deactivate",
]
