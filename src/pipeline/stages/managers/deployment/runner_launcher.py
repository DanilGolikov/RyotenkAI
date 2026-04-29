"""
Runner launcher — SSH-execs the in-pod uvicorn server.

The training pod is intentionally INERT at boot: its entrypoint.sh
sets up sshd + the Mac's PUBLIC_KEY and then ``sleep infinity``s.
The Mac (this module) drives uvicorn the same way it drives the
trainer subprocess — via SSH exec — for three reasons:

* **Symmetry.** Trainer is already SSH-launched. Runner the same way
  collapses two patterns into one.
* **Restartability.** A runner crash can be recovered by re-execing
  uvicorn over the same pod. With auto-launch in entrypoint.sh, a
  crash forces a full pod cycle.
* **Provider compatibility.** RunPod's ``docker_args`` (CMD-override)
  used to silently shadow the image's auto-launch path. The inert
  pod removes that ambiguity for any provider.

The launched uvicorn writes its stdout/stderr to
``/workspace/runner.log`` from the very first byte (including
ImportError / SyntaxError that fire before Python's logging is
initialized). That file is rsync'd to the Mac via the existing
``LogManager`` chain — see ``docs/architecture/log-collection.md``.
"""

from __future__ import annotations

import shlex
from typing import TYPE_CHECKING

from src.utils.logger import logger
from src.utils.result import Err, Ok, ProviderError, Result

if TYPE_CHECKING:
    from collections.abc import Mapping

    from src.utils.ssh_client import SSHClient


# How long to wait for uvicorn to bind 127.0.0.1:RUNNER_PORT inside
# the pod. We probe once per second via ``curl /healthz``. 30 s
# matches the Mac-side ``RUNNER_READY_MAX_ATTEMPTS`` so a slow runner
# fails on this preflight check (with diagnostics) instead of in the
# subsequent SSH-tunnel /healthz probe (which has no diagnostics).
RUNNER_READY_TIMEOUT_SECONDS: int = 30
RUNNER_HOST: str = "127.0.0.1"
RUNNER_PORT: int = 8080
RUNNER_LOG_PATH: str = "/workspace/runner.log"

# Path where the image bakes its baseline ``src/`` (Dockerfile.runtime
# does ``COPY src /opt/ryotenkai/src`` and sets PYTHONPATH). Mac's
# pipeline rsyncs run-scoped code into ``/workspace/runs/<run>/...``
# which gets prepended to PYTHONPATH per-run; this baseline ensures
# the runner can import even on a fresh pod with no rsync done yet
# (or for crash-recovery before the next rsync).
IMAGE_BASELINE_PYTHONPATH: str = "/opt/ryotenkai"


def _build_launch_command(env: Mapping[str, str] | None = None) -> str:
    """Return the shell command that ssh-execs uvicorn in the pod.

    Four concerns the script handles:

    1. **Idempotency.** ``pgrep -f 'uvicorn src.runner'`` short-circuits
       if uvicorn is already running on the pod (e.g. a retry of
       this stage after a transient SSH glitch). Re-execing would
       race for port 8080.
    2. **Runtime env vars.** The runner needs at least
       ``RYOTENKAI_RUNTIME_PROVIDER`` at startup (the lifespan
       hook ``resolve_lifecycle_client_from_env`` rejects an unset
       value with ``BootstrapConfigError`` and uvicorn dies). The
       caller passes the full set the provider declares via
       :meth:`IGPUProvider.required_runtime_env_vars` (provider,
       API keys, pod_id when known) and we forward it through
       ``env KEY=VALUE ...`` between ``nohup`` and ``stdbuf``.
       Values are shell-escaped to survive secrets containing
       quotes / spaces / ``$``.
    3. **Detachment.** ``nohup ... < /dev/null & disown`` so the
       process survives the SSH session closing and won't get a
       SIGHUP.
    4. **Readiness probe.** ``curl /healthz`` polled in a 30 s loop.
       On failure we dump ``ls -la`` and ``tail`` of runner.log to
       stderr so the Mac sees the cause without a separate fetch.

    Returns the full command as a single string suitable for
    ``ssh_client.exec_command(command=..., timeout=N)``.

    Why a string of ``;``-separated commands instead of a heredoc /
    script-file: keeps the deployment side stateless — no pre-
    deployed runner-launch.sh on the pod to drift out of sync with
    the Mac's expectations. The image already has everything
    (Python, uvicorn, ``src.runner``) baked in.

    Args:
        env: provider-supplied env vars to inject into the runner
            process. ``None`` is equivalent to ``{}``: only PYTHONPATH
            is set, and the runner will fail at startup if it requires
            anything else (e.g. RYOTENKAI_RUNTIME_PROVIDER on a real
            provider).
    """
    env_assignments = ""
    if env:
        # ``shlex.quote`` makes the value safe for any POSIX shell —
        # tokens with quotes, spaces, ``$`` or backticks can't break
        # out of the env arg.
        parts = [f"{key}={shlex.quote(str(value))}" for key, value in env.items()]
        env_assignments = " ".join(parts) + " "

    return (
        "set -e; "
        # Make sure the workspace directory exists before redirect.
        # Without this, if /workspace happens to be unmounted the
        # ``> runner.log`` redirect fails silently in the async &
        # branch and we get "file not found" diagnostics with no
        # trace of WHY the redirect didn't take.
        "mkdir -p /workspace; "
        # Probe writability EAGERLY so a read-only /workspace fails
        # here with a clear ``cannot create`` error instead of in the
        # async ``> runner.log`` redirect (where the failure would
        # be silenced and surface as "file not found" 30 s later).
        f"touch {RUNNER_LOG_PATH}; "
        # Idempotency: skip launching if a runner is ALREADY answering
        # /healthz on the loopback port.
        #
        # Why curl-probe and NOT ``pgrep -f 'uvicorn src.runner'``:
        # ``pgrep -f`` matches on the full command line of every
        # process — including the very bash subshell that's evaluating
        # THIS string, which trivially contains the pattern as an
        # argument. The pgrep version always returned "match" and
        # short-circuited the launch. Curl can't false-positive: if
        # nothing's bound to 8080 it returns non-zero.
        f"if curl -sf http://{RUNNER_HOST}:{RUNNER_PORT}/healthz "
        "  >/dev/null 2>&1; then "
        "  echo 'runner already running'; "
        "  exit 0; "
        "else "
        # Launch detached. nohup + disown survives SSH disconnect.
        # stdbuf -oL forces line-buffering so the log file gets
        # populated in real time and a fast crash still flushes its
        # last line. Use ``>>`` (append) so a runner-crash retry
        # doesn't truncate the previous log — successive boots
        # accumulate in the file with their own timestamps for
        # forensics.
        "  nohup env "
        f"PYTHONPATH={IMAGE_BASELINE_PYTHONPATH}:${{PYTHONPATH:-}} "
        f"    {env_assignments}"
        "    stdbuf -oL -eL "
        "    /usr/local/bin/python3 -m uvicorn src.runner.main:app "
        f"      --host {RUNNER_HOST} --port {RUNNER_PORT} --no-access-log "
        f"    >> {RUNNER_LOG_PATH} 2>&1 < /dev/null & "
        "  disown; "
        "fi; "
        # Readiness probe — poll /healthz until uvicorn is bound.
        f"for i in $(seq 1 {RUNNER_READY_TIMEOUT_SECONDS}); do "
        f"  if curl -sf http://{RUNNER_HOST}:{RUNNER_PORT}/healthz "
        "    >/dev/null 2>&1; then "
        "    echo 'runner ready'; exit 0; "
        "  fi; "
        "  sleep 1; "
        "done; "
        # Failure path — diagnostic dump.
        # NOTE: we deliberately do NOT do ``tail ... 2>/dev/null`` —
        # masking tail's stderr means a missing/empty runner.log
        # looks identical to a real traceback and the user can't
        # tell whether the redirect broke or uvicorn just didn't
        # write. Echoing ``ls -la`` of the file makes the existence
        # / size visible regardless of content.
        f"echo 'runner did not become ready within {RUNNER_READY_TIMEOUT_SECONDS}s' >&2; "
        f"echo '--- ls -la {RUNNER_LOG_PATH} ---' >&2; "
        f"ls -la {RUNNER_LOG_PATH} >&2 || echo '(file does not exist)' >&2; "
        f"echo '--- tail of {RUNNER_LOG_PATH} ---' >&2; "
        f"tail -100 {RUNNER_LOG_PATH} >&2 || true; "
        "echo '--- end of diagnostic dump ---' >&2; "
        "exit 1"
    )


def launch_runner(
    ssh_client: SSHClient,
    *,
    env: Mapping[str, str] | None = None,
) -> Result[None, ProviderError]:
    """Start the uvicorn runner inside the pod and wait for /healthz.

    On success, the pod has a uvicorn listening on
    ``127.0.0.1:8080`` ready to accept job submissions through the
    SSH tunnel that the caller will open next.

    On failure, the runner is either not running or not responding
    on /healthz; the returned ``ProviderError`` carries the tail of
    ``runner.log`` (or whatever stderr the SSH command produced) so
    callers can surface a useful error message without a separate
    log fetch.

    Args:
        ssh_client: alive SSH connection to the pod.
        env: provider-supplied env vars to inject into the runner
            process. Should at minimum include
            ``RYOTENKAI_RUNTIME_PROVIDER``; missing it makes the
            runner's lifespan hook raise ``BootstrapConfigError`` and
            uvicorn dies before binding the port. Typically obtained
            from ``provider.required_runtime_env_vars(resource_id)``.

    Returns:
        ``Ok(None)`` if uvicorn is running and /healthz returns 200
        within :data:`RUNNER_READY_TIMEOUT_SECONDS`. ``Err`` with code
        ``RUNNER_LAUNCH_FAILED`` otherwise.
    """
    cmd = _build_launch_command(env=env)
    # Allow the SSH command a bit longer than the readiness loop so
    # tail-of-log can run even if uvicorn never started.
    timeout_s = RUNNER_READY_TIMEOUT_SECONDS + 15
    logger.info("[RUNNER_LAUNCHER] Starting uvicorn in pod (timeout %ds)...", timeout_s)
    success, stdout, stderr = ssh_client.exec_command(
        command=cmd, silent=True, timeout=timeout_s,
    )

    out_clean = (stdout or "").strip()
    err_clean = (stderr or "").strip()

    if success and ("runner ready" in out_clean or "already running" in out_clean):
        logger.info("[RUNNER_LAUNCHER] ✅ Runner ready on %s:%d", RUNNER_HOST, RUNNER_PORT)
        if "already running" in out_clean:
            logger.debug("[RUNNER_LAUNCHER] (idempotent — was already running)")
        return Ok(None)

    # Failure path. Surface stderr (which includes runner.log tail).
    detail = err_clean or out_clean or "no diagnostic output"
    logger.error("[RUNNER_LAUNCHER] ❌ Runner failed to become ready:\n%s", detail)
    return Err(
        ProviderError(
            message=(
                f"runner did not become ready within "
                f"{RUNNER_READY_TIMEOUT_SECONDS}s — see runner.log on pod "
                f"({RUNNER_LOG_PATH})"
            ),
            code="RUNNER_LAUNCH_FAILED",
            details={"stderr_tail": detail[:4000]},
        ),
    )


__all__ = [
    "RUNNER_HOST",
    "RUNNER_LOG_PATH",
    "RUNNER_PORT",
    "RUNNER_READY_TIMEOUT_SECONDS",
    "launch_runner",
]
