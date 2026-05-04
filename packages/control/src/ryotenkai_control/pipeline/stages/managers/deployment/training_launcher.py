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

from ryotenkai_control.pipeline.heartbeat.heartbeat import ControlPlaneHeartbeat
from ryotenkai_shared.utils.clients.job_client import JobClient, JobClientError
from ryotenkai_shared.utils.clients.ssh_tunnel import (
    SSHTunnelEndpoint,
    SSHTunnelError,
    SSHTunnelManager,
)
from ryotenkai_shared.infrastructure.mlflow.uri_resolver import resolve_mlflow_uris
from ryotenkai_control.pipeline.stages.constants import PipelineContextKeys
from ryotenkai_control.pipeline.stages.managers.deployment.plugin_packer import (
    PluginPacker,
    PluginPackError,
)
from ryotenkai_control.pipeline.stages.managers.deployment.provider_config import (
    is_single_node_provider,
)
from ryotenkai_control.pipeline.stages.managers.deployment.runner_launcher import launch_runner
from ryotenkai_control.pipeline.stages.managers.deployment_constants import (
    DEPLOYMENT_CONFIG_PATH,
)
from ryotenkai_control.pipeline.state.job_submission import JobSubmission, save_job_submission
from ryotenkai_providers.training.interfaces import TrainingScriptHooks
from ryotenkai_shared.utils.logger import logger
from ryotenkai_shared.utils.result import AppError, Err, Ok, ProviderError, Result

if TYPE_CHECKING:
    from ryotenkai_control.pipeline.stages.managers.deployment.dependency_installer import (
        DependencyInstaller,
    )
    from ryotenkai_control.pipeline.stages.managers.deployment.file_uploader import (
        FileUploader,
    )
    from ryotenkai_providers.training.interfaces import IGPUProvider
    from ryotenkai_shared.config import PipelineConfig, Secrets
    from ryotenkai_shared.utils.ssh_client import SSHClient


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


DEFAULT_WORKSPACE = "/workspace"


class _FileUploadFailed(Exception):
    """Internal — propagates an HTTP file-upload :class:`AppError`
    out of the async island in :meth:`_submit_via_tunnel` so the
    sync caller can map it back to a typed ``Err`` without losing
    the original error code/details.
    """

    def __init__(self, app_error: "AppError") -> None:
        super().__init__(app_error.message)
        self.app_error = app_error


class _ImportCheckFailed(Exception):
    """Internal — same role as :class:`_FileUploadFailed` for the
    HTTP import-check gate (PR-2.5 endpoint, PR-3.3 migration).
    """

    def __init__(self, app_error: "AppError") -> None:
        super().__init__(app_error.message)
        self.app_error = app_error

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
        file_uploader: "FileUploader | None" = None,
    ) -> None:
        self.config = config
        self.secrets = secrets
        self._deps_installer = deps_installer  # reserved
        # Phase 3 PR-3.3: HTTP file upload moved into start_training,
        # between /healthz wait and POST /jobs. The deployment-manager
        # facade injects the FileUploader here. Optional for tests
        # that exercise the launcher in isolation.
        self._file_uploader = file_uploader
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

        # Step 0 — start the in-pod uvicorn runner.
        #
        # The pod boot-time entrypoint is intentionally inert (sleep
        # infinity); we SSH-exec uvicorn here so the runner's stdout/
        # stderr are captured to /workspace/runner.log from the very
        # first byte (LogManager picks that file up via the existing
        # rsync chain). This is a precondition for the SSH tunnel
        # below — without uvicorn listening on 127.0.0.1:8080, the
        # tunnel would open but /healthz would forever timeout.
        #
        # The runner needs at least ``RYOTENKAI_RUNTIME_PROVIDER`` at
        # startup (its lifespan hook ``resolve_lifecycle_client_from_env``
        # bails with BootstrapConfigError otherwise), plus any
        # provider-specific vars (RUNPOD_API_KEY, RUNPOD_POD_ID, …).
        # The provider's :meth:`required_runtime_env_vars` is the
        # single source of truth for that set — same call we already
        # make below for the trainer's env.
        #
        # ``launch_runner`` is idempotent: a retry of this stage that
        # finds uvicorn already running short-circuits and returns
        # Ok immediately.
        runner_env: dict[str, str] = {}
        run_id_for_layout = ""
        if provider is not None:
            resource_id = (
                context.get("resource_id") or context.get("pod_id")
            )
            runner_env = dict(
                provider.required_runtime_env_vars(
                    resource_id=resource_id if resource_id else None,
                ),
            )
            # Provider's per-run layout drives every pod-side path
            # the runner uses (logs/, events/, state/). The layout's
            # root MUST equal the workspace_path the CodeSyncer used
            # — provider.connect() set both from ``run.name``.
            run_id_for_layout = context.get("run_id") or context.get("run_name") or ""
            if not run_id_for_layout:
                # Fall back to deriving run_id from the workspace path.
                # ``self.workspace`` is ``<base>/runs/<run_id>``.
                run_id_for_layout = self.workspace.rstrip("/").rsplit("/", 1)[-1]
            pod_layout_for_runner = provider.pod_layout_for_run(run_id_for_layout)
        else:
            # Should not happen in production paths but the test path
            # builds the launcher without a provider for unit-coverage.
            from ryotenkai_shared.utils.pod_layout import PodLayout
            from pathlib import PurePosixPath
            pod_layout_for_runner = PodLayout.from_root(
                PurePosixPath(self.workspace),
            )
        # The layout's root is the rsync target the CodeSyncer dropped
        # ``src/...`` into for this run; the thin image (v2.0.0+) ships
        # only Python + libs, so this directory is the SOLE source of
        # ``src.runner`` on the pod. Without it, uvicorn fails with
        # ``ModuleNotFoundError: No module named 'ryotenkai_pod.runner'``.
        runner_ready = launch_runner(
            ssh_client,
            pod_layout=pod_layout_for_runner,
            env=runner_env,
        )
        if runner_ready.is_err():
            err = runner_ready.unwrap_err()  # type: ignore[union-attr]
            logger.error("[LAUNCHER] Runner failed to launch: %s", err.message)
            return Err(err)

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
        # Phase 14.D+F — provider supplies its own runtime env via
        # ``required_runtime_env_vars``. The hooks dataclass'
        # ``env_vars`` is preserved for back-compat (any provider
        # that still uses ``prepare_training_script_hooks`` will
        # see its values merged on top).
        #
        # ``workdir`` is the absolute pod-side directory the in-pod
        # supervisor will spawn the trainer in (``cwd=`` for
        # ``asyncio.create_subprocess_exec``). Without it the trainer
        # inherits uvicorn's cwd (typically ``/root`` after
        # SSH-launch) and any relative path — including the
        # ``--config config/pipeline_config.yaml`` argv below — fails
        # with FileNotFoundError. ``self.workspace`` is set by
        # ``set_workspace`` (called by ``GPUDeployer`` from
        # ``ssh_info.workspace_path``) to ``/workspace/runs/<run_id>``.
        job_spec = {
            "job_id": job_id,
            "command": [
                "python",
                "-m",
                "ryotenkai_pod.trainer.run_training",
                "--config",
                DEPLOYMENT_CONFIG_PATH,
            ],
            "env": self._build_job_env(
                context, provider, hooks.env_vars,
            ),
            "workdir": self.workspace,
        }

        # Phase 3 PR-3.3: HTTP file upload context — passed through
        # to ``_submit_via_tunnel`` so it can call ``upload_via_http``
        # between /healthz and POST /jobs.
        upload_context = dict(context)

        try:
            tunnel, client = asyncio.run(
                self._submit_via_tunnel(
                    ssh_client, job_spec, plugins_payload,
                    upload_context=upload_context,
                ),
            )
        except _ImportCheckFailed as exc:
            logger.error("[LAUNCHER] HTTP import-check failed: %s", exc.app_error.message)
            return Err(exc.app_error)
        except _FileUploadFailed as exc:
            logger.error("[LAUNCHER] HTTP file upload failed: %s", exc.app_error.message)
            return Err(exc.app_error)
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

        # Phase 11.E — start the control-plane heartbeat. While the
        # orchestrator process is alive, this service POSTs to
        # ``/api/v1/control/heartbeat`` every 30 s so the in-pod
        # :class:`MacHeartbeat` stays fresh (TTL 120 s) regardless
        # of WS / REST traffic. This is critical for ModelRetriever:
        # its tar+ssh adapter download bypasses the runner's
        # FastAPI entirely, and without explicit pings the heartbeat
        # would stale out and trigger ``podStop`` mid-download.
        # TrainingMonitor's :meth:`cleanup` (pipeline-level, runs
        # AFTER ModelRetriever) stops this service.
        try:
            heartbeat_service = ControlPlaneHeartbeat(client)
            asyncio.run(heartbeat_service.start())
            context["control_plane_heartbeat"] = heartbeat_service
            logger.info(
                "[LAUNCHER] Control-plane heartbeat started "
                "(ping every 30 s, TTL 120 s)"
            )
        except Exception as exc:
            # Heartbeat startup failure is NOT fatal: the implicit
            # WS / REST heartbeat still works for the duration of
            # TrainingMonitor's WS subscription. ModelRetriever may
            # see stale heartbeat → podStop mid-download in the
            # worst case, but that's a degradation, not a failure.
            logger.warning(
                "[LAUNCHER] control-plane heartbeat failed to start: %s. "
                "Continuing without explicit pings; ModelRetriever may "
                "see stale heartbeat on long downloads.", exc,
            )

        # Persist the SSH endpoint so out-of-process CLI commands
        # (``ryotenkai job status``, ``... events``, ``... stop``)
        # can rebuild the connection later. Best-effort — failure to
        # write the file is logged but doesn't abort the job, since
        # the monitor in this same process already has live handles.
        self._persist_job_submission(
            context, ssh_client, job_id, provider,
        )

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
        :class:`PodTerminator`). The dataclass still carries those
        fields for compatibility but we ignore them here.
        """
        if provider is None:
            return TrainingScriptHooks.empty()
        result = provider.prepare_training_script_hooks(ssh_client, context)
        if result.is_err():
            return None
        return result.unwrap()

    def _persist_job_submission(
        self,
        context: dict[str, Any],
        ssh_client: SSHClient,
        job_id: str,
        provider: IGPUProvider | None,
    ) -> None:
        """Write ``attempts/<n>/job_submission.json`` for out-of-process
        CLI tooling. Best-effort: errors are logged, never raised.

        The attempt directory is read from ``context[ATTEMPT_DIRECTORY]``
        — the orchestrator sets this before calling start_training.
        Tests that don't go through the full pipeline bootstrap won't
        have it set, in which case we silently skip.
        """
        from pathlib import Path

        attempt_dir_str = context.get(PipelineContextKeys.ATTEMPT_DIRECTORY)
        if not isinstance(attempt_dir_str, str) or not attempt_dir_str:
            logger.debug(
                "[LAUNCHER] no attempt_directory in context; "
                "skipping job_submission.json persistence",
            )
            return

        provider_name = (
            getattr(provider, "provider_name", None)
            if provider is not None
            else "single_node"
        )
        pod_id = context.get("resource_id") or context.get("pod_id")

        submission = JobSubmission.now(
            job_id=job_id,
            provider_name=str(provider_name or "unknown"),
            pod_id=str(pod_id) if pod_id else None,
            ssh_host=ssh_client.host,
            ssh_port=int(ssh_client.port),
            ssh_username=ssh_client.username or "root",
            ssh_key_path=ssh_client.key_path or None,
        )
        try:
            target = save_job_submission(Path(attempt_dir_str), submission)
            logger.info("[LAUNCHER] persisted job submission to %s", target)
        except OSError as exc:
            logger.warning(
                "[LAUNCHER] failed to persist job_submission.json: %s",
                exc,
            )

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
        provider: IGPUProvider | None,
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
           ``RUNPOD_KEEP_ON_ERROR``, etc.) — these intentionally win
           over our defaults so a provider can override e.g.
           ``LOG_LEVEL`` if it has a better idea. Phase 11.B removed
           ``RUNPOD_AUTO_STOP`` (no toggle: PodTerminator's decision
           matrix runs unconditionally on terminal hooks).
        """
        # Phase 14.D+F — capability-driven dispatch (was
        # ``is_single_node_provider(self.config)`` string check).
        # Local providers (single_node) run the trainer in a docker
        # container with the run dir mounted as ``/workspace``.
        # Cloud providers run in the pod where the workspace IS the
        # run dir.
        is_local = (
            provider.get_capabilities().is_local
            if provider is not None
            else is_single_node_provider(self.config)
        )
        workspace_env = "/workspace" if is_local else self._workspace
        env: dict[str, str] = {
            "LOG_LEVEL": "DEBUG",
            "HELIX_WORKSPACE": workspace_env,
            "PYTHONPATH": workspace_env,
        }

        if self.secrets.hf_token:
            env["HF_TOKEN"] = self.secrets.hf_token

        mlflow_config = self.config.integrations.mlflow
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
        # Note: ``PYTHONFAULTHANDLER_PATH`` removed — faulthandler
        # writes to stderr by default, which the runner's Supervisor
        # captures into ``trainer.stdio.log``. One ground-truth path
        # for both regular Python tracebacks AND native crash dumps.
        #
        # Note: ``RYOTENKAI_TRAINING_LOG_PATH`` removed — the trainer
        # no longer attaches its own FileHandler. Pod-side
        # ``trainer.stdio.log`` (written by Supervisor pump) is the
        # source of truth for trainer output, pulled by LogManager scp.

        # Tell the trainer where the runner is so RunnerEventCallback
        # (Phase 3) starts pushing events. Loopback URL — the runner
        # binds 127.0.0.1:8080 inside the pod, and the trainer
        # subprocess is in the same network namespace.
        env["RYOTENKAI_RUNNER_URL"] = "http://127.0.0.1:8080"

        # Phase 14.D+F — provider supplies its own runtime env vars.
        # Replaces the pre-14.D hardcoded
        # ``env["RUNPOD_VOLUME_KIND"] = "persistent"`` (which fired
        # for ALL providers, including single_node where the var is
        # meaningless). The provider's :meth:`required_runtime_env_vars`
        # returns the full set keyed to the resource_id from the
        # current run context — RunPod includes ``RUNPOD_*`` vars,
        # single_node returns just ``RYOTENKAI_RUNTIME_PROVIDER``.
        if provider is not None:
            resource_id = (
                context.get("resource_id") or context.get("pod_id")
            )
            env.update(
                provider.required_runtime_env_vars(
                    resource_id=resource_id if resource_id else None,
                ),
            )

        # Provider hooks merged LAST so any back-compat values
        # (e.g. legacy ``prepare_training_script_hooks.env_vars``)
        # propagate without the launcher's defaults shadowing them.
        # Note: ``RUNPOD_AUTO_STOP`` is NO LONGER honoured (Phase
        # 11.B removed the toggle — PodTerminator always runs the
        # decision matrix).
        if extra_env_vars:
            env.update(extra_env_vars)

        return env

    # --- async island -----------------------------------------------------

    async def _submit_via_tunnel(
        self,
        ssh_client: SSHClient,
        job_spec: dict[str, Any],
        plugins_payload: bytes,
        upload_context: dict[str, Any] | None = None,
    ) -> tuple[SSHTunnelManager, JobClient]:
        """Open the SSH tunnel, probe ``/healthz``, upload files via
        HTTP, ``POST /jobs``.

        Phase 3 PR-3.3 (transport-unification-v2): config + dataset
        upload now happens via ``JobClient.upload_file`` between the
        ``/healthz`` wait and the ``POST /jobs``. The tar-pipe + SCP
        path the legacy ``deploy_files`` ran pre-launch is gone.

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
            # Phase 3 PR-3.3 (transport-unification-v2): post-sync
            # importability gate moved off SSH. Validates the SAME
            # synced source the trainer will exercise; failure halts
            # before files upload.
            await self._verify_imports_async(client)
            # HTTP file upload between /healthz and submit.
            if self._file_uploader is not None and upload_context is not None:
                upload_result = await self._upload_files_async(
                    client, upload_context,
                )
                if upload_result.is_failure():
                    raise _FileUploadFailed(upload_result.unwrap_err())
            await client.submit_job(job_spec, plugins_payload=plugins_payload or None)
            return tunnel, client
        except BaseException:
            # Roll the tunnel back. The client's own connection pool
            # is closed by ``aclose`` if we got that far.
            if client is not None:
                await client.aclose()
            await tunnel.close()
            raise

    async def _verify_imports_async(self, client: JobClient) -> None:
        """Phase 3 PR-3.3 — call ``POST /api/v1/runtime/import-check``
        for the canonical pod-relevant module set.

        Replaces the SSH ``runtime_check.py --check-source`` gate that
        used to live at the tail of ``CodeSyncer.sync``. Same intent
        — validate the synced source is importable on the runner's
        PYTHONPATH; new mechanism — HTTP, structured per-module
        report, no shell-output parsing.

        Raises :class:`_ImportCheckFailed` carrying an :class:`AppError`
        when one or more modules fail. The sync caller maps it back
        to ``Err`` outside the async island.
        """
        from ryotenkai_control.pipeline.stages.managers.deployment.code_syncer import (
            REQUIRED_SRC_MODULES,
        )

        try:
            report = await client.check_imports(REQUIRED_SRC_MODULES)
        except Exception as exc:  # noqa: BLE001 — propagate upstream
            raise _ImportCheckFailed(
                AppError(
                    message=f"runner import-check call failed: {exc}",
                    code="IMPORT_GATE_ERROR",
                )
            ) from exc
        if report.all_importable:
            logger.info(
                f"✅ Post-sync HTTP import-check passed "
                f"({len(report.results)} modules)",
            )
            return
        failed = report.failed
        details = {r.module: r.error for r in report.results if not r.importable}
        raise _ImportCheckFailed(
            AppError(
                message=(
                    f"Post-sync import check failed. Modules not importable "
                    f"on pod: {failed}. Action: ensure these modules exist "
                    f"in your local checkout before deploy."
                ),
                code="IMPORT_GATE_FAILED",
                details={"failed_modules": failed, "errors": details},
            ),
        )

    async def _upload_files_async(
        self,
        client: JobClient,
        upload_context: dict[str, Any],
    ) -> Result[None, AppError]:
        """Async wrapper around the FileUploader's HTTP upload core.

        Called from inside ``_submit_via_tunnel`` — already on an
        event loop, so we cannot call the sync facade
        ``upload_via_http`` (which itself does ``asyncio.run``).
        Instead, replicate the validation step here and delegate to
        the async core ``_upload_all`` directly so the same JobClient
        (and its loop-bound httpx pool) is used end-to-end.
        """
        assert self._file_uploader is not None
        from pathlib import Path
        from ryotenkai_control.pipeline.stages.managers.deployment_constants import (
            DEPLOYMENT_CONFIG_PATH,
        )

        config_path = Path(upload_context.get("config_path", DEPLOYMENT_CONFIG_PATH))
        if not config_path.exists():
            return Err(
                ProviderError(
                    message=f"Config file not found: {config_path}",
                    code="CONFIG_FILE_NOT_FOUND",
                ),
            )

        dataset_files, missing = self._file_uploader._collect_local_datasets()
        if missing and not dataset_files:
            return Err(
                ProviderError(
                    message=(
                        "Dataset files referenced but not found: "
                        + ", ".join(missing)
                    ),
                    code="DATASET_FILE_NOT_FOUND",
                    details={"missing": missing},
                ),
            )

        logger.info(
            f"📤 HTTP upload (post-healthz): 1 config + "
            f"{len(dataset_files)} dataset(s)",
        )
        try:
            await self._file_uploader._upload_all(
                client, config_path, dataset_files,
            )
        except Exception as exc:  # noqa: BLE001
            return Err(
                ProviderError(
                    message=f"HTTP upload failed: {exc}",
                    code="HTTP_FILE_UPLOAD_FAILED",
                ),
            )
        logger.info("✅ HTTP file upload complete")
        return Ok(None)

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
