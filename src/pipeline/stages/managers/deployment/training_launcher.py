"""Submit a training job to the in-pod runner — Phase 6.3 rewrite.

The legacy implementation generated a ``start_training.sh`` shell
script, ``nohup``'d it through SSH, and polled marker files on disk
to detect "is training alive?". That layout had three structural
problems closed by Phase 1-5 of the runner project:

1. **No push events** — every status check needed a fresh SSH round
   trip; Mac sleeping on the laptop killed the SSH ControlMaster
   and left the pod alone. Now we open a tunnel once, subscribe to
   a WebSocket event stream, and reconnect on a Mac wake-up
   transparently.
2. **Marker file IPC** — three independent processes (Python
   trainer, bash watchdog, sshd) coordinated through 9 marker
   files on a shared filesystem. Now there is one supervisor
   process owning the trainer subprocess, with explicit FSM
   transitions (:mod:`src.runner.state`).
3. **No plugin delivery** — :class:`CodeSyncer` deliberately skips
   ``community/``, so reward plugins never reached the pod. Now
   :class:`PluginPacker` packs the needed plugins and ships them
   in the multipart ``POST /jobs`` body (Phase 6.1 / 6.2).

Public API is preserved 1:1 so :class:`TrainingDeploymentManager`
and the upstream :mod:`src.pipeline.stages.gpu_deployer` don't have
to change. Internally, ``start_training`` builds a job spec, opens
an SSH tunnel to the runner, submits the job, and stashes the open
tunnel + :class:`JobClient` on the pipeline context so
:class:`TrainingMonitor` can subscribe to the WebSocket event
stream.

Async strategy:
The class is sync (`def`) to match the upstream pipeline. The
small async island around ``SSHTunnelManager``/``JobClient`` is
driven via ``asyncio.run`` inside :meth:`_submit_via_tunnel`. The
launcher MUST NOT be invoked from inside an existing event loop —
``asyncio.run`` would raise ``RuntimeError`` in that case. Today
:mod:`src.pipeline.stages.gpu_deployer` is plain sync; if we ever
re-host the pipeline inside FastAPI we'll need to revisit this.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from src.api.clients.job_client import JobClient, JobClientError
from src.api.services.tunnel_service import (
    SSHTunnelEndpoint,
    SSHTunnelError,
    SSHTunnelManager,
)
from src.infrastructure.mlflow.uri_resolver import resolve_mlflow_uris
from src.pipeline.stages.constants import PipelineContextKeys
from src.pipeline.stages.managers.deployment.plugin_packer import (
    PluginPacker,
    PluginPackError,
)
from src.pipeline.stages.managers.deployment.provider_config import (
    is_single_node_provider,
)
from src.pipeline.stages.managers.deployment_constants import (
    DEPLOYMENT_CONFIG_PATH,
)
from src.providers.training.interfaces import TrainingScriptHooks
from src.utils.logger import logger
from src.utils.result import AppError, Err, Ok, ProviderError, Result

if TYPE_CHECKING:
    from src.pipeline.stages.managers.deployment.dependency_installer import (
        DependencyInstaller,
    )
    from src.providers.training.interfaces import IGPUProvider
    from src.utils.config import PipelineConfig, Secrets
    from src.utils.ssh_client import SSHClient


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


DEFAULT_WORKSPACE = "/workspace"

# How long we wait for the runner's /healthz to start returning 200
# after the SSH tunnel comes up. The runner boots with sshd in
# parallel so the tunnel may be open before uvicorn is. Each probe
# is one HTTP GET; we cap total attempts at 30 (≈ 30 s with our
# 1 s sleep step), enough for slow pod cold-starts.
RUNNER_READY_MAX_ATTEMPTS = 30
RUNNER_READY_POLL_SECONDS = 1.0


# ---------------------------------------------------------------------------
# Launcher
# ---------------------------------------------------------------------------


class TrainingLauncher:
    """Submit the training job to the in-pod runner over an SSH tunnel.

    Constructed once per pipeline run by :class:`TrainingDeploymentManager`.
    The public surface (:meth:`start_training`, :attr:`workspace`,
    :meth:`set_workspace`) is identical to the legacy implementation
    so callers don't have to change.

    Args:
        config: pipeline config — read for strategy chain (plugin
            discovery), MLflow URIs, and provider type.
        secrets: HF token + (future) other secrets to forward as
            env vars to the trainer subprocess inside the pod.
        deps_installer: kept for parity with the legacy constructor;
            no longer used internally now that we don't manage the
            runtime image install on the host (the runner image
            ships with everything pre-installed). Reserved for
            ``single_node`` provider future hooks.
    """

    def __init__(
        self,
        config: PipelineConfig,
        secrets: Secrets,
        *,
        deps_installer: DependencyInstaller,
    ) -> None:
        self.config = config
        self.secrets = secrets
        self._deps_installer = deps_installer  # reserved
        self._workspace = DEFAULT_WORKSPACE

    # --- workspace --------------------------------------------------------

    @property
    def workspace(self) -> str:
        return self._workspace

    def set_workspace(self, workspace_path: str) -> None:
        self._workspace = workspace_path

    # --- public entry point ----------------------------------------------

    def start_training(
        self,
        ssh_client: SSHClient,
        context: dict[str, Any],
        provider: IGPUProvider | None = None,
    ) -> Result[dict[str, Any], AppError]:
        """Submit the training job and return once the runner has accepted it.

        Flow (sync facade — async work is encapsulated in
        :meth:`_submit_via_tunnel`):

        1. Collect provider env hooks (RunPod auto-stop credentials,
           etc.). ``provider=None`` means no provider customisation
           (single_node, tests).
        2. Pack any reward plugins the strategy chain references
           into a ZIP via :class:`PluginPacker`. SFT-only configs
           produce ``b""`` — :class:`JobClient` happily ships an
           empty placeholder part.
        3. Build the :class:`JobSpec` dict (command + env).
        4. ``asyncio.run`` the small async island: open SSH tunnel,
           probe runner ``/healthz``, ``POST /jobs``.
        5. Stash the open tunnel + :class:`JobClient` + ``job_id`` on
           ``context`` so :class:`TrainingMonitor` can re-use them.

        Returns ``Ok({"mode": "job_server", "job_id": ..., "tunnel_port": N})``.
        On any error the SSH tunnel is closed before returning ``Err``
        — no leaked port forwards.
        """
        logger.info("[LAUNCHER] Submitting training job to in-pod runner...")

        hooks = self._collect_provider_hooks(ssh_client, context, provider)
        if hooks is None:
            return Err(
                ProviderError(
                    message="provider hooks preparation failed",
                    code="PROVIDER_HOOKS_FAILED",
                ),
            )

        try:
            packer = PluginPacker(self.config)
            plugins_payload = packer.pack_required()
        except PluginPackError as exc:
            logger.exception("[LAUNCHER] PluginPacker failed")
            return Err(
                ProviderError(
                    message=f"plugin payload pack failed: {exc}",
                    code="PLUGIN_PACK_FAILED",
                ),
            )

        job_id = self._resolve_job_id(context)
        job_spec = {
            "job_id": job_id,
            "command": [
                "python",
                "-m",
                "src.training.run_training",
                "--config",
                DEPLOYMENT_CONFIG_PATH,
            ],
            "env": self._build_job_env(context, hooks.env_vars),
        }

        try:
            tunnel, client = asyncio.run(
                self._submit_via_tunnel(ssh_client, job_spec, plugins_payload),
            )
        except SSHTunnelError as exc:
            logger.exception("[LAUNCHER] SSH tunnel open failed")
            return Err(
                ProviderError(
                    message=f"ssh tunnel to runner failed: {exc}",
                    code="RUNNER_TUNNEL_FAILED",
                ),
            )
        except JobClientError as exc:
            logger.exception("[LAUNCHER] Job submission failed")
            return Err(
                ProviderError(
                    message=f"runner rejected job submission: {exc}",
                    code="JOB_SUBMIT_FAILED",
                ),
            )

        # Stash everything the monitor needs. We pass the live tunnel
        # so the monitor closes it at end-of-run instead of leaking
        # the port forward — owning the tunnel lifecycle here would
        # mean shutting it down before the monitor can subscribe.
        context["job_client"] = client
        context["ssh_tunnel"] = tunnel
        context["job_id"] = job_id

        logger.info(
            "[LAUNCHER] Job %s submitted; tunnel localhost:%s → pod:8080",
            job_id, tunnel.local_port,
        )
        return Ok({
            "mode": "job_server",
            "job_id": job_id,
            "tunnel_port": tunnel.local_port,
        })

    # --- helpers ----------------------------------------------------------

    def _collect_provider_hooks(
        self,
        ssh_client: SSHClient,
        context: dict[str, Any],
        provider: IGPUProvider | None,
    ) -> TrainingScriptHooks | None:
        """Run the provider's hook preparation, returning hooks or
        ``None`` on failure (caller turns that into ``Err``).

        Provider-supplied hooks contribute ``env_vars`` only — the
        legacy ``pre_python`` / ``post_python`` bash injection points
        are dead after Phase 6.5 (the runner replaces watchdog.sh
        and runpod_stop_pod.sh with :class:`IdleDetector` and
        :class:`PodStopper`). The dataclass still carries those
        fields for compatibility but we ignore them here.
        """
        if provider is None:
            return TrainingScriptHooks.empty()
        result = provider.prepare_training_script_hooks(ssh_client, context)
        if result.is_err():
            return None
        return result.unwrap()

    def _resolve_job_id(self, context: dict[str, Any]) -> str:
        """Pick the ``job_id`` we'll send to the runner.

        Priority: ``context["logical_run_id"]`` (canonical attempt
        identifier set by the orchestrator). Falls back to
        ``run.name`` for tests / scripts that don't go through the
        full pipeline bootstrap. Always returns a non-empty string —
        the runner's :class:`JobSpec` validator requires it.
        """
        logical = context.get(PipelineContextKeys.LOGICAL_RUN_ID)
        if isinstance(logical, str) and logical:
            return logical
        run = context.get(PipelineContextKeys.RUN)
        name = getattr(run, "name", None)
        if isinstance(name, str) and name:
            return name
        # Last-resort fallback — generate a stable string. The runner
        # only requires non-empty / ≤ 128 chars so anything goes.
        return "unnamed-job"

    def _build_job_env(
        self,
        context: dict[str, Any],
        extra_env_vars: dict[str, str],
    ) -> dict[str, str]:
        """Compose the ``env`` block of the :class:`JobSpec`.

        Mirrors what the legacy ``_create_env_file`` wrote into the
        pod's ``.env`` file — same keys, same priority order — but
        returns a dict ready for the multipart submit instead of a
        file on disk. The runner's supervisor merges this dict over
        ``os.environ`` before exec-ing the trainer subprocess.

        Priority (last wins):
        1. Built-in keys (``LOG_LEVEL``, ``HELIX_WORKSPACE``,
           ``PYTHONPATH``).
        2. Optional secrets (``HF_TOKEN`` if set).
        3. MLflow vars (resolved through
           :func:`resolve_mlflow_uris`).
        4. Provider hook env_vars (``RUNPOD_API_KEY``,
           ``RUNPOD_AUTO_STOP``, etc.) — these intentionally win
           over our defaults so a provider can override e.g.
           ``LOG_LEVEL`` if it has a better idea.
        """
        # ``single_node`` runs the trainer in a docker container with
        # the run dir mounted as ``/workspace``. Cloud providers run
        # in the pod where the workspace IS the run dir. This is the
        # only path-aware setting; everything else is workspace-agnostic.
        workspace_env = (
            "/workspace" if is_single_node_provider(self.config) else self._workspace
        )
        env: dict[str, str] = {
            "LOG_LEVEL": "DEBUG",
            "HELIX_WORKSPACE": workspace_env,
            "PYTHONPATH": workspace_env,
        }

        if self.secrets.hf_token:
            env["HF_TOKEN"] = self.secrets.hf_token

        mlflow_config = self.config.experiment_tracking.mlflow
        if mlflow_config:
            uris = resolve_mlflow_uris(mlflow_config, runtime_role="training")
            if uris.effective_remote_tracking_uri:
                env["MLFLOW_TRACKING_URI"] = uris.effective_remote_tracking_uri
            parent_run_id = context.get(PipelineContextKeys.MLFLOW_PARENT_RUN_ID)
            if isinstance(parent_run_id, str) and parent_run_id:
                env["MLFLOW_PARENT_RUN_ID"] = parent_run_id
            env["MLFLOW_HTTP_REQUEST_TIMEOUT"] = "15"
            env["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] = "2"
            if mlflow_config.ca_bundle_path:
                env["REQUESTS_CA_BUNDLE"] = mlflow_config.ca_bundle_path
                env["SSL_CERT_FILE"] = mlflow_config.ca_bundle_path

        # Crash-observability env vars previously injected by the
        # generated ``start_training.sh``. The runner inherits
        # ``os.environ`` but the supervisor's :func:`subprocess.exec`
        # path doesn't get our own process env, so we add them
        # explicitly here.
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONFAULTHANDLER"] = "1"
        env["PYTHONFAULTHANDLER_PATH"] = f"{workspace_env}/training.faulthandler.log"

        # Tell the trainer where the runner is so RunnerEventCallback
        # (Phase 3) starts pushing events. Loopback URL — the runner
        # binds 127.0.0.1:8080 inside the pod, and the trainer
        # subprocess is in the same network namespace.
        env["RYOTENKAI_RUNNER_URL"] = "http://127.0.0.1:8080"

        # Provider hooks merged LAST so e.g. RUNPOD_AUTO_STOP=true
        # propagates to PodStopper without our defaults shadowing it.
        if extra_env_vars:
            env.update(extra_env_vars)

        return env

    # --- async island -----------------------------------------------------

    async def _submit_via_tunnel(
        self,
        ssh_client: SSHClient,
        job_spec: dict[str, Any],
        plugins_payload: bytes,
    ) -> tuple[SSHTunnelManager, JobClient]:
        """Open the SSH tunnel, probe ``/healthz``, ``POST /jobs``.

        Returns the live tunnel + client so the monitor can re-use
        them. On ANY failure mid-flight the tunnel is closed before
        the exception propagates — leaving an open ``ssh -L`` after
        a failed launch would leak a local port forever.
        """
        endpoint = SSHTunnelEndpoint(
            host=ssh_client.host,
            port=int(ssh_client.port),
            username=ssh_client.username or "root",
            key_path=ssh_client.key_path or None,
        )
        tunnel = SSHTunnelManager(endpoint)
        await tunnel.open()

        client: JobClient | None = None
        try:
            client = JobClient(tunnel.base_url)
            await self._wait_for_runner_ready(client)
            await client.submit_job(job_spec, plugins_payload=plugins_payload or None)
            return tunnel, client
        except BaseException:
            # Roll the tunnel back. The client's own connection pool
            # is closed by ``aclose`` if we got that far.
            if client is not None:
                await client.aclose()
            await tunnel.close()
            raise

    async def _wait_for_runner_ready(self, client: JobClient) -> None:
        """Poll ``/healthz`` until the runner returns 200 or we
        exhaust :data:`RUNNER_READY_MAX_ATTEMPTS`.

        The runner's uvicorn boot is in parallel with sshd inside
        the pod entrypoint, so the tunnel can come up before HTTP
        is serving. Without this probe the very first ``submit_job``
        could hit a connection refused.
        """
        for attempt in range(1, RUNNER_READY_MAX_ATTEMPTS + 1):
            if await client.health_check():
                return
            await asyncio.sleep(RUNNER_READY_POLL_SECONDS)
            logger.debug(
                "[LAUNCHER] runner /healthz still failing (attempt %d/%d)",
                attempt, RUNNER_READY_MAX_ATTEMPTS,
            )
        raise SSHTunnelError(
            "runner /healthz did not return 200 within "
            f"{RUNNER_READY_MAX_ATTEMPTS}s — image not started?",
        )
