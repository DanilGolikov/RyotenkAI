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
``<pod_layout.runner_log>`` (per-run, e.g.
``/workspace/runs/<run_id>/logs/runner.log``) from the very first
byte (including ImportError / SyntaxError that fire before Python's
logging is initialized). That file is rsync'd to the Mac via the
existing ``LogManager`` chain — see ``docs/architecture/log-collection.md``.

PYTHONPATH points at the run-scoped workspace (the rsync target,
``pod_layout.root``) — that is now the SOLE source of ``src/runner``.
The thin-image migration removed the baked-in ``/opt/ryotenkai/src``
baseline; if the rsync didn't run, uvicorn fails with
``ModuleNotFoundError: No module named 'ryotenkai_pod.runner'`` and the
diagnostic dump surfaces it. See ``docs/architecture/thin-image.md``.

Per-run isolation is owned by :class:`PodLayout` — sequential runs
on the same pod each get their own ``logs/runner.log`` so previous
runs' diagnostics survive even after a re-launch. This is the
resume-collision fix: pre-PodLayout the path was the global
``/workspace/runner.log`` and a new run silently overwrote the
previous one.
"""

from __future__ import annotations

import shlex
from typing import TYPE_CHECKING

from ryotenkai_shared.errors import SSHExecFailedError
from ryotenkai_shared.utils.logger import logger

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ryotenkai_shared.utils.pod_layout import PodLayout
    from ryotenkai_shared.utils.ssh_client import SSHClient


# How long to wait for uvicorn to bind 127.0.0.1:RUNNER_PORT inside
# the pod. We probe once per second via ``curl /healthz``. 30 s
# matches the Mac-side ``RUNNER_READY_MAX_ATTEMPTS`` so a slow runner
# fails on this preflight check (with diagnostics) instead of in the
# subsequent SSH-tunnel /healthz probe (which has no diagnostics).
RUNNER_READY_TIMEOUT_SECONDS: int = 30
RUNNER_HOST: str = "127.0.0.1"
RUNNER_PORT: int = 8080

def _build_launch_command(
    *,
    pod_layout: PodLayout,
    env: Mapping[str, str] | None = None,
) -> str:
    """Return the shell command that ssh-execs uvicorn in the pod.

    Four concerns the script handles:

    1. **Idempotency.** A curl probe of /healthz short-circuits if
       uvicorn is already running on the pod. Re-execing would race
       for port 8080.
    2. **Runtime env vars.** The runner needs at least
       ``RYOTENKAI_RUNTIME_PROVIDER`` at startup (the lifespan
       hook ``resolve_lifecycle_client_from_env`` rejects an unset
       value with ``BootstrapConfigError`` and uvicorn dies). The
       caller passes the full set the provider declares via
       :meth:`IGPUProvider.required_runtime_env_vars` (provider,
       API keys, pod_id when known) and we forward it through
       ``env KEY=VALUE ...`` between ``nohup`` and ``stdbuf``.
       Values are shell-escaped to survive secrets containing
       quotes / spaces / ``$``. ``RYOTENKAI_WORKSPACE`` is added
       implicitly so the runner's ``_resolve_workspace`` finds the
       per-run root regardless of cwd.
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
    the Mac's expectations. The image carries Python + uvicorn; the
    application code (``src.runner`` and its deps) arrives via
    ``CodeSyncer.rsync`` into ``pod_layout.src_dir``.

    Args:
        pod_layout: per-run filesystem layout for the pod. Provides:
            - ``pod_layout.root`` — PYTHONPATH (where ``src/...`` was
              rsync'd by ``CodeSyncer`` for this run);
            - ``pod_layout.runner_log`` — per-run runner.log path
              under ``logs/``;
            - ``pod_layout.logs_dir`` — directory the bash script
              creates eagerly so the redirect succeeds even on a
              freshly-mounted pod with nothing else under workspace.
        env: provider-supplied env vars to inject into the runner
            process. ``None`` is equivalent to ``{}``: only
            ``RYOTENKAI_WORKSPACE`` + PYTHONPATH are set, and the
            runner will fail at startup if it requires anything
            else (e.g. RYOTENKAI_RUNTIME_PROVIDER on a real provider).
    """
    workspace = str(pod_layout.root)
    runner_log = str(pod_layout.runner_log)
    logs_dir = str(pod_layout.logs_dir)

    # Always inject RYOTENKAI_WORKSPACE so the runner's lifespan
    # finds its per-run root regardless of cwd. Caller-supplied env
    # values take precedence (intentional: a test harness might
    # override).
    merged_env: dict[str, str] = {"RYOTENKAI_WORKSPACE": workspace}
    if env:
        merged_env.update(env)

    # ``shlex.quote`` makes each value safe for any POSIX shell —
    # tokens with quotes, spaces, ``$`` or backticks can't break
    # out of the env arg.
    env_parts = [f"{key}={shlex.quote(str(value))}" for key, value in merged_env.items()]
    env_assignments = " ".join(env_parts) + " "

    quoted_workspace = shlex.quote(workspace)
    quoted_runner_log = shlex.quote(runner_log)
    quoted_logs_dir = shlex.quote(logs_dir)

    return (
        "set -e; "
        # Create the per-run logs directory eagerly. Without this, if
        # the parent directory tree is missing the ``> runner.log``
        # redirect fails silently in the async & branch and we get
        # "file not found" diagnostics with no trace of WHY the
        # redirect didn't take.
        f"mkdir -p {quoted_logs_dir}; "
        # Probe writability EAGERLY so a read-only filesystem fails
        # here with a clear ``cannot create`` error instead of in the
        # async ``> runner.log`` redirect (where the failure would
        # be silenced and surface as "file not found" 30 s later).
        f"touch {quoted_runner_log}; "
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
        f"PYTHONPATH={quoted_workspace}:${{PYTHONPATH:-}} "
        f"    {env_assignments}"
        "    stdbuf -oL -eL "
        "    /usr/local/bin/python3 -m uvicorn ryotenkai_pod.runner.main:app "
        f"      --host {RUNNER_HOST} --port {RUNNER_PORT} --no-access-log "
        f"    >> {quoted_runner_log} 2>&1 < /dev/null & "
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
        f"echo '--- ls -la {runner_log} ---' >&2; "
        f"ls -la {quoted_runner_log} >&2 || echo '(file does not exist)' >&2; "
        f"echo '--- tail of {runner_log} ---' >&2; "
        f"tail -100 {quoted_runner_log} >&2 || true; "
        "echo '--- end of diagnostic dump ---' >&2; "
        "exit 1"
    )


def launch_runner(
    ssh_client: SSHClient,
    *,
    pod_layout: PodLayout,
    env: Mapping[str, str] | None = None,
) -> None:
    """Start the uvicorn runner inside the pod and wait for /healthz.

    On success, the pod has a uvicorn listening on
    ``127.0.0.1:8080`` ready to accept job submissions through the
    SSH tunnel that the caller will open next.

    On failure, the runner is either not running or not responding
    on /healthz; the raised :class:`SSHExecFailedError` carries the
    tail of ``pod_layout.runner_log`` (or whatever stderr the SSH
    command produced) so callers can surface a useful error message
    without a separate log fetch.

    Args:
        ssh_client: alive SSH connection to the pod.
        pod_layout: per-run filesystem layout for the pod. Built by
            the caller via :meth:`IGPUProvider.pod_layout_for_run`
            so the layout root matches the per-run workspace where
            ``CodeSyncer`` rsync'd ``src/...`` for this run.
        env: provider-supplied env vars to inject into the runner
            process. Should at minimum include
            ``RYOTENKAI_RUNTIME_PROVIDER``; missing it makes the
            runner's lifespan hook raise ``BootstrapConfigError`` and
            uvicorn dies before binding the port. Typically obtained
            from ``provider.required_runtime_env_vars(resource_id)``.
            ``RYOTENKAI_WORKSPACE`` is automatically injected by the
            launcher (caller does not need to pre-populate it).

    Returns:
        ``None`` if uvicorn is running and /healthz returns 200
        within :data:`RUNNER_READY_TIMEOUT_SECONDS`.

    Raises:
        SSHExecFailedError: the runner did not become ready within
            :data:`RUNNER_READY_TIMEOUT_SECONDS`. ``context`` carries
            the tail of stderr (which includes runner.log content) and
            the path to the on-pod runner.log for offline diagnosis.
            Phase A2 Batch 9 (2026-05-15): migrated from
            ``Result[None, ProviderError(code="RUNNER_LAUNCH_FAILED")]``.
    """
    cmd = _build_launch_command(pod_layout=pod_layout, env=env)
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
        return None

    # Failure path. Surface stderr (which includes runner.log tail).
    detail = err_clean or out_clean or "no diagnostic output"
    logger.error("[RUNNER_LAUNCHER] ❌ Runner failed to become ready:\n%s", detail)
    raise SSHExecFailedError(
        detail=(
            f"runner did not become ready within "
            f"{RUNNER_READY_TIMEOUT_SECONDS}s — see runner.log on pod "
            f"({pod_layout.runner_log})"
        ),
        context={
            "stderr_tail": detail[:4000],
            "runner_log_path": str(pod_layout.runner_log),
            "reason": "RUNNER_LAUNCH_FAILED",
        },
    )


__all__ = [
    "RUNNER_HOST",
    "RUNNER_PORT",
    "RUNNER_READY_TIMEOUT_SECONDS",
    "launch_runner",
]
