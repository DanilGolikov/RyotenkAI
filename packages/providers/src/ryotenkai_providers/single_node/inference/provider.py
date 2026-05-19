"""
Single-node inference provider (SSH + Docker(vLLM)).

MVP assumptions:
- inference node is reachable via SSH (alias or explicit)
- Docker is installed and has NVIDIA runtime configured
- vLLM is run in a Docker container
- LoRA adapters are merged into base model before serving (merge_before_deploy=true)
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from ryotenkai_shared.constants import INFERENCE_MANIFEST_FILENAME, PROVIDER_SINGLE_NODE, VLLM_INFERENCE_CONTAINER_NAME


def _resolve_engine_image(engine_kind: str) -> str:
    """Resolve container image for ``engine_kind`` via the engine registry.

    Wraps ``ryotenkai_engines.get_registry().get_image(...)`` — uses the
    convention default + override chain (env / provider / manifest).
    Lazy local import to avoid the shared→engines→shared circular at
    package import time.
    """
    from ryotenkai_engines import get_registry

    return get_registry().get_image(engine_kind)
from ryotenkai_providers.constants import ENCODING_UTF8 as _ENCODING_UTF8
from ryotenkai_providers.constants import SHA12_LEN
from ryotenkai_providers.inference.interfaces import (
    EndpointInfo,
    IInferenceProvider,
    InferenceArtifacts,
    InferenceArtifactsContext,
    InferenceCapabilities,
    PipelineReadinessMode,
)
from ryotenkai_providers.single_node.training.health_check import SingleNodeHealthCheck
from ryotenkai_shared.errors import (
    ConfigInvalidError,
    EngineConfigInvalidError,
    InferenceUnavailableError,
    ProviderUnavailableError,
    RyotenkAIError,
    SSHTransferFailedError,
)
from ryotenkai_shared.infrastructure.docker import IDockerClient, LocalDockerClient
from ryotenkai_shared.utils.constants import LOG_OUTPUT_LONG_CHARS, LOG_OUTPUT_SHORT_CHARS
from ryotenkai_shared.utils.logger import get_run_log_dir, logger
from ryotenkai_shared.utils.ssh_client import SSHClient

from .artifacts import CHAT_SCRIPT as _CHAT_SCRIPT
from .artifacts import render_readme as _render_readme

if TYPE_CHECKING:
    from ryotenkai_engines.interfaces import PreparePlan
    from ryotenkai_engines.vllm.config import VLLMEngineConfig


PULL_TIMEOUT = 1200
SOURCE_SINGLE_NODE_INFERENCE = "SingleNodeInferenceProvider"

# Polling cadence for ephemeral prepare containers (seconds). Engine-agnostic
# fabric concern; engine declares step-level timeout via PrepareStep.
_PREPARE_POLL_INTERVAL_S = 2

# JSON dict keys used in >3 places (WPS226)
_KEY_ENGINE = "engine"
_KEY_BASE_MODEL_ID = "base_model_id"
_KEY_ADAPTER_REF = "adapter_ref"

# SSH command timeouts (seconds)
_SETUP_CMD_TIMEOUT_S = 120  # mkdir / rm -rf skeleton commands
_QUICK_CMD_TIMEOUT_S = 30  # fast one-liners (container start probe, cleanup)

# ls output truncation in diagnostics
_DIR_LISTING_MAX_CHARS = 800


from ryotenkai_providers.training.interfaces import ProviderBase


class SingleNodeInferenceProvider(ProviderBase, IInferenceProvider):
    """Deploy vLLM inference endpoint on a single SSH-accessible node.

    Identity comes from ``provider.toml`` via :class:`ProviderBase`.

    ``SUPPORTED_ENGINES`` declares which engine kinds this provider can
    launch. The provider.toml ``[capabilities.inference]`` block is the
    user-facing source of truth; this ClassVar is a defensive
    runtime-side mirror that catches direct callers who bypass the
    PipelineConfig cross-validator.
    """

    _CONTAINER_NAME = VLLM_INFERENCE_CONTAINER_NAME
    SUPPORTED_ENGINES: ClassVar[frozenset[str]] = frozenset({"vllm"})

    def __init__(
        self,
        ctx: ProviderContext,
        *,
        docker: IDockerClient | None = None,
    ) -> None:
        """Initialize from a :class:`ProviderContext`.

        ``docker`` is the injection point for the Docker client used
        by all container operations (logs / rm / inspect / image pull).
        Defaults to :class:`LocalDockerClient` which delegates to the
        legacy SSH-backed functions in
        :mod:`ryotenkai_shared.utils.docker` — production behaviour
        is unchanged. Tests inject
        :class:`tests._fakes.docker.FakeDockerClient`.
        """
        config = ctx.pipeline_config
        secrets = ctx.secrets
        self._cfg = config
        self._secrets = secrets
        self._docker: IDockerClient = docker if docker is not None else LocalDockerClient()

        self._inf_cfg = config.inference

        # NEW (v3): Get providers.single_node config
        provider_cfg_raw = self._cfg.get_provider_config(PROVIDER_SINGLE_NODE)
        from ryotenkai_shared.config.providers.single_node import SingleNodeProviderConfig

        # PipelineConfig validator promotes the YAML block to a typed
        # SingleNodeProviderConfig; modular runtimes / tests may still
        # pass raw dicts. Accept both.
        if isinstance(provider_cfg_raw, SingleNodeProviderConfig):
            self._provider_cfg = provider_cfg_raw
        else:
            self._provider_cfg = SingleNodeProviderConfig(**provider_cfg_raw)

        # Access inference-specific settings via .inference
        self._serve_cfg = self._provider_cfg.inference.serve

        # SSH config from connect.ssh (single source of truth)
        self._ssh_cfg = self._provider_cfg.connect.ssh

        # Post-discriminated-union: engine config is typed directly on
        # cfg.inference.engine (Pydantic discriminator narrowed it).
        # The provider declares which engine kinds it supports via its
        # provider.toml [capabilities.inference] block — cross-validated
        # at PipelineConfig load. Defensive belt-and-suspenders check
        # here in case of direct callers.
        self._engine_cfg = self._inf_cfg.engine
        if self._engine_cfg.kind not in self.SUPPORTED_ENGINES:
            from ryotenkai_providers.registry import ProviderRegistryError

            raise ProviderRegistryError(
                message=(
                    f"engine {self._engine_cfg.kind!r} not supported by "
                    f"{type(self).__name__}; supported: {sorted(self.SUPPORTED_ENGINES)}"
                ),
                code="PROVIDER_ENGINE_NOT_SUPPORTED",
            )
        # Engine runtime resolved via the registry — no concrete-class
        # hardcoding. Returns an IInferenceEngine implementation that
        # builds structured LaunchSpec values; this provider then formats
        # the spec into a docker shell command (k8s providers would map
        # the same spec to a ContainerSpec instead).
        from ryotenkai_engines import get_registry

        runtime_cls = get_registry().get_runtime(self._engine_cfg.kind)
        self._engine = runtime_cls()

        # Engine validates its own invariants (replaces inline gating in
        # the old __init__: e.g. merge_before_deploy=True for vLLM).
        # As of Phase A2 Batch 2, ``validate_config`` returns None on
        # success and raises ``EngineConfigInvalidError`` on failure;
        # we translate to the provider's legacy ``ProviderRegistryError``
        # at the boundary (provider-side migration is Batch 11/12).
        # The engine's ``context["reason"]`` subcode is preserved as the
        # ProviderRegistryError ``code`` (upper-cased) so legacy tests
        # asserting on ``VLLM_LIVE_LORA_NOT_SUPPORTED`` etc. keep matching.
        try:
            self._engine.validate_config(self._engine_cfg)
        except EngineConfigInvalidError as engine_err:
            from ryotenkai_providers.registry import ProviderRegistryError

            subcode = engine_err.context.get("reason", "")
            translated_code = (
                str(subcode).upper() if subcode else "PROVIDER_ENGINE_CONFIG_INVALID"
            )
            raise ProviderRegistryError(
                message=engine_err.detail or str(engine_err),
                code=translated_code,
                details=dict(engine_err.context),
            ) from engine_err

        self._ssh_client: SSHClient | None = None
        self._endpoint_info: EndpointInfo | None = None
        self._run_id: str | None = None
        self._mlflow_manager: Any | None = None  # Optional MLflow integration

    # ---------------------------------------------------------------------
    # Interface properties
    # ---------------------------------------------------------------------

    # provider_name / provider_type / provider_id default impls live on
    # ProviderBase — manifest-derived.

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def deploy(
        self,
        model_source: str,
        *,
        run_id: str,
        base_model_id: str,
        trust_remote_code: bool = False,
        lora_path: str | None = None,
        keep_running: bool = False,  # noqa: ARG002
    ) -> EndpointInfo:
        """Provision the vLLM endpoint on the single-node host.

        Raises:
            ConfigInvalidError: invariants violated (merge_before_deploy
                false, adapter not a directory).
            InferenceUnavailableError: SSH connect / health check /
                docker / merge / vLLM launch failure.
        """
        self._run_id = run_id

        # MVP safety: we do not support native LoRA in vLLM yet.
        if not self._engine_cfg.merge_before_deploy:
            raise ConfigInvalidError(
                detail=(
                    "inference.engines.vllm.merge_before_deploy=false is not supported in MVP. "
                    "Set it to true (merge) to serve the trained adapter reliably."
                ),
                context={"reason": "SINGLENODE_LORA_MERGE_REQUIRED"},
            )

        self._connect_ssh()

        assert self._ssh_client is not None
        ssh = self._ssh_client

        # Pre-flight checks: docker + nvidia runtime + disk + GPU visibility
        health = SingleNodeHealthCheck(ssh).run_all_checks(
            workspace_path=self._serve_cfg.workspace,
        )
        if not health.passed:
            errs = "; ".join(health.errors or ["Unknown error"])
            raise InferenceUnavailableError(
                detail=f"Inference node health checks failed: {errs}",
                context={"reason": "SINGLENODE_HEALTH_CHECK_FAILED"},
            )

        workspace = self._serve_cfg.workspace.rstrip("/")
        hf_cache = f"{workspace}/hf_cache"

        # Canonical run directory name (single source of truth): run_YYYYMMDD_HHMMSS_id5
        run_dir_name = run_id
        run_dir = f"{workspace}/runs/{run_dir_name}"
        adapter_dir = f"{run_dir}/adapter"
        merged_dir = f"{run_dir}/model"

        # Prepare workspace dirs
        for d in (workspace, hf_cache, f"{workspace}/runs", run_dir):
            ok, dir_err = ssh.create_directory(d)
            if not ok:
                raise InferenceUnavailableError(
                    detail=f"Failed to create remote directory '{d}': {dir_err}",
                    context={"reason": "SINGLENODE_DIR_CREATE_FAILED"},
                )

        # Resolve adapter reference
        adapter_ref = lora_path or model_source
        adapter_ref_for_merge = adapter_ref

        # If adapter is local on the orchestrator machine, upload to remote run dir.
        local_adapter = Path(adapter_ref).expanduser()
        if local_adapter.exists():
            if not local_adapter.is_dir():
                raise ConfigInvalidError(
                    detail=f"Adapter local path must be a directory, got: {local_adapter}",
                    context={"reason": "SINGLENODE_ADAPTER_NOT_DIR"},
                )

            ok, dir_err = ssh.create_directory(adapter_dir)
            if not ok:
                raise InferenceUnavailableError(
                    detail=f"Failed to create remote adapter dir '{adapter_dir}': {dir_err}",
                    context={"reason": "SINGLENODE_DIR_CREATE_FAILED"},
                )

            try:
                ssh.upload_directory(local_path=local_adapter, remote_path=adapter_dir)
            except SSHTransferFailedError as exc:
                raise InferenceUnavailableError(
                    detail=f"Failed to upload adapter to inference node: {exc.detail or exc}",
                    context={"reason": "SINGLENODE_ADAPTER_UPLOAD_FAILED"},
                    cause=exc,
                ) from exc
            adapter_ref_for_merge = adapter_dir

        # PR-16: ask the engine what preparation work it needs (LoRA merge,
        # GGUF conversion, TensorRT compile, …). The engine returns a
        # structured PreparePlan; the provider executes it via the generic
        # _run_prepare_plan runner. Empty plan ⇒ skip prep entirely.
        adapter_path_in_container: str | None = None
        if adapter_ref_for_merge != adapter_ref or local_adapter.exists():
            # We have a real on-disk adapter (uploaded above). Map host → container.
            adapter_path_in_container = self._host_to_container(
                adapter_ref_for_merge, workspace_host_path=workspace
            )
        elif adapter_ref_for_merge:
            # Adapter ref is an HF repo id (or other engine-resolvable string);
            # pass through unchanged. Engine decides whether to merge.
            adapter_path_in_container = adapter_ref_for_merge

        # Engine.prepare_model returns PreparePlan directly and raises
        # EngineConfigInvalidError on engine-boundary errors (Phase A2
        # Batch 2). Batch 12 surfaces the typed error directly through
        # the provider boundary (no translation to legacy InferenceError).
        try:
            plan = self._engine.prepare_model(
                cfg=self._engine_cfg,
                base_model=base_model_id,
                adapter_path_in_container=adapter_path_in_container,
                workspace_host_path=workspace,
                run_id=run_id,
                trust_remote_code=trust_remote_code,
            )
        except EngineConfigInvalidError as engine_err:
            raise InferenceUnavailableError(
                detail=engine_err.detail or str(engine_err),
                context={"reason": "SINGLENODE_PREPARE_PLAN_BUILD_FAILED"},
                cause=engine_err,
            ) from engine_err

        # _run_prepare_plan raises InferenceUnavailableError on failure.
        self._run_prepare_plan(
            ssh=ssh, plan=plan, run_id=run_id, workspace_host_path=workspace,
        )

        # Engine config for this deployment (avoid mutating global PipelineConfig).
        engine_cfg = self._engine_cfg.model_copy()

        # Choose the model path the engine should serve. When the prep plan
        # produced artifacts, use plan.final_model_path; otherwise fall back
        # to the original model_source mapped into the container (engines
        # that need no prep — SGLang, future LiveLoRA-vLLM).
        if plan.final_model_path is not None:
            container_model_path = plan.final_model_path
        else:
            container_model_path = self._host_to_container(
                model_source, workspace_host_path=workspace
            )

        # _start_engine_container raises InferenceUnavailableError on failure.
        self._start_engine_container(
            ssh=ssh,
            engine_cfg=engine_cfg,
            workspace_host_path=workspace,
            model_path_in_container=container_model_path,
        )

        # Endpoint is bound on node to serve.host:serve.port, but MVP expects local access via SSH tunnel.
        port = self._serve_cfg.port
        endpoint_url = f"http://127.0.0.1:{port}/v1"
        health_url = f"http://127.0.0.1:{port}/v1/models"

        self._endpoint_info = EndpointInfo(
            endpoint_url=endpoint_url,
            api_type="openai_compatible",
            provider_type=self.provider_type,
            engine="vllm",
            model_id=base_model_id,
            health_url=health_url,
            resource_id=self._CONTAINER_NAME,
        )

        # Save remote runtime metadata (best-effort, no secrets)
        _ = self._write_runtime_json_best_effort(
            ssh=ssh,
            run_dir=run_dir,
            runtime={
                "run_id": run_id,
                "provider": self.provider_type,
                _KEY_ENGINE: "vllm",
                "container_name": self._CONTAINER_NAME,
                # Same unified image is used for both merge & serve;
                # see packages/engines/src/ryotenkai_engines/vllm/IMAGE_README.md.
                "image": _resolve_engine_image(self._engine_cfg.kind),
                "host_bind": self._serve_cfg.host,
                "port": port,
                "workspace": workspace,
                "merged_model_dir": merged_dir,
                "config_hash": self._sha12(
                    json.dumps(
                        {
                            _KEY_BASE_MODEL_ID: base_model_id,
                            _KEY_ADAPTER_REF: adapter_ref,
                            _KEY_ENGINE: self._engine_cfg.model_dump(mode="python", exclude_none=True),
                        },
                        sort_keys=True,
                    )
                ),
            },
        )

        logger.info(f"✅ Inference deployed (single_node/vLLM): {endpoint_url} (use SSH tunnel)")
        return self._endpoint_info

    def get_pipeline_readiness_mode(self) -> PipelineReadinessMode:
        return PipelineReadinessMode.WAIT_FOR_HEALTHY

    def collect_startup_logs(self, *, local_path: Path) -> None:
        """Collect logs from inference container (best-effort)."""
        try:
            if not self._ssh_client:
                return
            # Docker logs surface raises on failure (Phase A2 Batch 4);
            # this method is best-effort — swallow the exception in the
            # outer ``except`` rather than propagating.
            stdout = self._docker.logs(
                self._ssh_client, container_name=self._CONTAINER_NAME, timeout_seconds=10
            )
            if stdout:
                local_path.parent.mkdir(parents=True, exist_ok=True)
                local_path.write_text(stdout, encoding=_ENCODING_UTF8)
        except Exception as e:
            logger.debug(f"Failed to collect inference logs: {e}")

    def build_inference_artifacts(
        self, *, ctx: InferenceArtifactsContext
    ) -> InferenceArtifacts:
        ssh_cfg = self._ssh_cfg
        ssh_block = {
            "alias": ssh_cfg.alias,
            "host": ssh_cfg.host,
            "port": ssh_cfg.port,
            "user": ssh_cfg.user,
            "key_path": ssh_cfg.key_path,
            "key_env": ssh_cfg.key_env,
        }

        alias = ssh_block.get("alias")
        if isinstance(alias, str) and alias:
            target = alias
        else:
            host = ssh_block.get("host") or "<SSH_HOST>"
            user = ssh_block.get("user") or "<SSH_USER>"
            target = f"{user}@{host}"
        tunnel_hint = f"ssh -L {int(self._serve_cfg.port)}:127.0.0.1:{int(self._serve_cfg.port)} {target}"

        workspace = self._serve_cfg.workspace.rstrip("/")
        port = int(self._serve_cfg.port)

        manifest: dict[str, Any] = {
            "run_name": ctx.run_name,
            "mlflow_run_id": ctx.mlflow_run_id,
            "provider": self.provider_type,
            _KEY_ENGINE: self._inf_cfg.engine,
            "ssh": ssh_block,
            "docker": {
                # Same unified image is used for both merge & serve;
                # see packages/engines/src/ryotenkai_engines/vllm/IMAGE_README.md.
                "image": _resolve_engine_image(self._engine_cfg.kind),
                "container_name": VLLM_INFERENCE_CONTAINER_NAME,
                "host_bind": self._serve_cfg.host,
                "port": port,
                "workspace": workspace,
                "hf_cache_path": f"{workspace}/hf_cache",
            },
            "model": {
                _KEY_BASE_MODEL_ID: self._cfg.model.name,
                _KEY_ADAPTER_REF: ctx.model_source,
                "merged_model_path": f"{workspace}/runs/{ctx.run_name}/model",
            },
            "llm": self._resolve_llm_manifest_block(),
            "config_hash": self._sha12(
                json.dumps(
                    {
                        _KEY_BASE_MODEL_ID: self._cfg.model.name,
                        _KEY_ADAPTER_REF: ctx.model_source,
                        _KEY_ENGINE: self._engine_cfg.model_dump(mode="python", exclude_none=True),
                        "serve": self._serve_cfg.model_dump(mode="python", exclude_none=True),
                    },
                    sort_keys=True,
                )
            ),
            "endpoint": {
                "client_base_url": ctx.endpoint.endpoint_url,
                "health_url": ctx.endpoint.health_url,
                "tunnel_hint": tunnel_hint,
            },
        }

        return InferenceArtifacts(
            manifest=manifest,
            chat_script=_CHAT_SCRIPT,
            readme=_render_readme(
                manifest_filename=INFERENCE_MANIFEST_FILENAME, endpoint_url=ctx.endpoint.endpoint_url
            ),
        )

    def undeploy(self) -> None:
        """Tear down the inference container (best-effort).

        Single-node's contract is "always succeed" — docker-daemon
        failures are swallowed (logged at DEBUG), endpoint state is
        always cleared.
        """
        if not self._ssh_client:
            return

        try:
            self._docker.rm_force(
                self._ssh_client, container_name=self._CONTAINER_NAME, timeout_seconds=60
            )
        except (ProviderUnavailableError, ConfigInvalidError) as exc:
            logger.debug(f"Best-effort container teardown failed: {exc}")
        self._endpoint_info = None

    def health_check(self) -> bool:
        """Probe whether the vLLM endpoint is ready.

        Returns:
            True iff ``/v1/models`` responds 200.

        Raises:
            InferenceUnavailableError: SSH client missing (deploy not
                called), or healthcheck command failed at transport
                level. ``context["legacy_code"]`` distinguishes
                ``SINGLENODE_NOT_DEPLOYED`` vs
                ``SINGLENODE_HEALTH_CHECK_COMMAND_FAILED``.
        """
        if not self._ssh_client:
            raise InferenceUnavailableError(
                detail="SSH client not initialized (deploy was not called)",
                context={"reason": "SINGLENODE_NOT_DEPLOYED"},
            )

        # IMPORTANT: Check on REMOTE host, not localhost
        # Container binds to 0.0.0.0 inside, but we access via host's 127.0.0.1
        host = self._serve_cfg.host  # Should be 127.0.0.1 on remote
        port = self._serve_cfg.port
        cmd = self._engine.build_healthcheck_command(host=host, port=port)
        ok, stdout, stderr = self._ssh_client.exec_command(cmd, timeout=10, silent=True)

        if not ok:
            raise InferenceUnavailableError(
                detail=f"Health check command failed: {stderr[:LOG_OUTPUT_SHORT_CHARS]}",
                context={"reason": "SINGLENODE_HEALTH_CHECK_COMMAND_FAILED"},
            )

        # Check output: "1" = ready, "0" = not ready
        return stdout.strip() == "1"

    def get_capabilities(self) -> InferenceCapabilities:
        return InferenceCapabilities(
            provider_type=self.provider_type,
            supported_engines=["vllm"],
            supports_lora=True,  # via merge_before_deploy
            supports_activate_for_eval=True,  # endpoint live after deploy(); activate is a no-op
        )

    def get_endpoint_info(self) -> EndpointInfo | None:
        return self._endpoint_info

    def activate_for_eval(self) -> str:
        """
        Single-node: endpoint is already live after deploy() — return existing URL.

        No additional startup needed; SSH tunnel and vLLM container were started
        during deploy(). Evaluation can immediately use the endpoint.

        Raises:
            InferenceUnavailableError: deploy() was not called before
                activate (no endpoint to return).
        """
        endpoint = self._endpoint_info
        if endpoint is None or endpoint.endpoint_url is None:
            raise InferenceUnavailableError(
                detail="single_node: activate_for_eval called but endpoint is not deployed. Call deploy() first.",
                context={"reason": "SINGLENODE_NOT_DEPLOYED"},
            )
        logger.info(f"[EVAL] single_node endpoint ready for evaluation: {endpoint.endpoint_url}")
        return endpoint.endpoint_url

    def deactivate_after_eval(self) -> None:
        """
        Single-node: no-op — endpoint lifecycle is managed by the user (via generated scripts).

        We do NOT stop the endpoint here because:
        - The user may want to interact with it after the pipeline.
        - The generated stop-script handles teardown on user request.
        """
        logger.info("[EVAL] single_node: deactivate_after_eval is a no-op (endpoint managed externally)")

    # ---------------------------------------------------------------------
    # Internals
    # ---------------------------------------------------------------------

    @staticmethod
    def _sha12(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:SHA12_LEN]

    def _resolve_llm_manifest_block(self) -> dict[str, Any]:
        """
        Resolve the system prompt once and return the llm manifest block.

        The resolved prompt text (not path) is stored in the manifest so that
        the chat script on the remote machine does not need MLflow access.
        system_prompt_source carries audit metadata (origin type + name/path + version).
        """
        from ryotenkai_shared.inference.prompts import SystemPromptLoader
        from ryotenkai_shared.infrastructure.mlflow.prompt_registry import (
            MlflowPromptRegistry,
        )

        llm_cfg = self._provider_cfg.inference.llm
        mlflow_cfg = getattr(getattr(self._cfg, "integrations", None), "mlflow", None)
        tracking_uri = (
            getattr(mlflow_cfg, "tracking_uri", None)
            or getattr(mlflow_cfg, "local_tracking_uri", None)
            if mlflow_cfg is not None
            else None
        )

        if not tracking_uri or not llm_cfg.system_prompt_mlflow_name:
            registry = None
        else:
            try:
                registry = MlflowPromptRegistry(tracking_uri=tracking_uri)
            except ValueError as exc:
                logger.error(f"[INFERENCE] MlflowPromptRegistry init failed: {exc}")
                registry = None

        loader = SystemPromptLoader(registry=registry)

        try:
            result = loader.load(llm_cfg, on_mlflow_failure="warn")
        except ValueError as exc:
            logger.error(f"[INFERENCE] System prompt configuration error: {exc}")
            result = None

        return {
            "system_prompt": result.text if result else None,
            "system_prompt_source": result.source if result else None,
        }

    @staticmethod
    def _looks_like_unresolved_env(value: str) -> bool:
        s = value.strip()
        return s.startswith("${") and s.endswith("}")

    def _connect_ssh(self) -> None:
        """
        Connect to inference node via SSH.

        NEW (v3): SSH config comes from providers.single_node.connect.ssh
        No fallback logic needed - single source of truth.

        Raises:
            InferenceUnavailableError: configuration / transport
                failure. Subcodes carried in
                ``context["reason"]``.
        """
        if self._ssh_client is not None:
            return

        try:
            ssh_cfg = self._ssh_cfg

            # Resolve key path via key_env if present
            key_path = ssh_cfg.key_path
            if not key_path and ssh_cfg.key_env:
                key_path = os.environ.get(ssh_cfg.key_env)

            host = ssh_cfg.alias or ssh_cfg.host
            if not host:
                raise InferenceUnavailableError(
                    detail="SSH host is not configured in providers.single_node.connect.ssh",
                    context={"reason": "SINGLENODE_SSH_HOST_NOT_CONFIGURED"},
                )

            if self._looks_like_unresolved_env(host):
                raise InferenceUnavailableError(
                    detail=f"SSH host looks like unresolved env placeholder: {host}",
                    context={"reason": "SINGLENODE_SSH_HOST_UNRESOLVED_ENV"},
                )

            username = None if ssh_cfg.alias else ssh_cfg.user
            if username and self._looks_like_unresolved_env(username):
                raise InferenceUnavailableError(
                    detail=f"SSH user looks like unresolved env placeholder: {username}",
                    context={"reason": "SINGLENODE_SSH_USER_UNRESOLVED_ENV"},
                )

            self._ssh_client = SSHClient(
                host=host,
                port=ssh_cfg.port,
                username=username,
                key_path=key_path if key_path else None,
                connect_timeout=int(ssh_cfg.connect_settings.timeout_seconds),
            )

            ok, err = self._ssh_client.test_connection(
                max_retries=ssh_cfg.connect_settings.max_retries,
                retry_delay=ssh_cfg.connect_settings.retry_delay_seconds,
            )
            if not ok:
                self._ssh_client = None
                raise InferenceUnavailableError(
                    detail=f"SSH connection failed: {err}",
                    context={"reason": "SINGLENODE_SSH_CONNECT_FAILED"},
                )
        except RyotenkAIError:
            self._ssh_client = None
            raise
        except Exception as e:
            self._ssh_client = None
            raise InferenceUnavailableError(
                detail=f"SSH initialization failed: {e!s}",
                context={"reason": "SINGLENODE_SSH_INIT_FAILED"},
                cause=e,
            ) from e

    def _ensure_docker_image(
        self,
        *,
        ssh: SSHClient,
        image: str,
    ) -> None:
        """Ensure the docker image is available on the host.

        Raises:
            InferenceUnavailableError: pull failed (translated from
                the docker client's ``ProviderUnavailableError``).
        """
        try:
            self._docker.ensure_image(
                ssh=ssh, image=image, pull_timeout_seconds=PULL_TIMEOUT,
            )
        except ProviderUnavailableError as exc:
            raise InferenceUnavailableError(
                detail=str(exc.detail or exc),
                context={"reason": "SINGLENODE_DOCKER_IMAGE_FAILED"},
                cause=exc,
            ) from exc

    @staticmethod
    def _host_to_container(
        host_or_container_path: str,
        *,
        workspace_host_path: str,
        workspace_container_path: str = "/workspace",
    ) -> str:
        """Map a host path inside the workspace mount to its in-container path.

        If the path already looks container-side (starts with
        ``workspace_container_path``), returns it unchanged. If it's a host
        path inside ``workspace_host_path``, rewrites the prefix. Otherwise
        returns the path unchanged (for HF repo IDs, etc.).

        Provider concern — engines never see host paths, only container ones.
        """
        p = host_or_container_path.rstrip("/")
        if p == workspace_container_path or p.startswith(workspace_container_path + "/"):
            return p
        if p == workspace_host_path or p.startswith(workspace_host_path + "/"):
            return workspace_container_path + p[len(workspace_host_path) :]
        return host_or_container_path

    def _run_prepare_plan(
        self,
        *,
        ssh: SSHClient,
        plan: PreparePlan,
        run_id: str,
        workspace_host_path: str,
    ) -> None:
        """Execute every :class:`PrepareStep` in ``plan`` sequentially.

        For each step:
          1. Resolve image (``step.image`` or fall through to engine serve image)
          2. Ensure image is pulled (``_ensure_docker_image``)
          3. Clean output dirs on host (``rm -rf`` then ``mkdir -p``) so reruns
             are idempotent — same behavior as the legacy ``_run_merge_container``.
          4. Format the step into a docker shell command and start it detached.
          5. Poll logs every ``_PREPARE_POLL_INTERVAL_S`` seconds until the
             container stops or ``step.timeout_seconds`` elapses.
          6. Validate: exit code 0; ``success_marker`` substring present (when
             declared); ``success_artifact`` exists on host (when declared).
          7. ``docker rm -f`` cleanup regardless of success/failure.

        Fails fast: a step failure aborts subsequent steps. Partial outputs
        are LEFT on disk for operator inspection (no rollback).

        All preparation events flow through the optional MLflow event logger
        with stable kwargs (``step_name``, ``step_index``, ``step_count``,
        ``duration_seconds``) — operator dashboards filter on these.
        """
        from ryotenkai_engines.interfaces import PrepareStep

        from ryotenkai_providers.inference.launch import format_prepare_step

        if not plan.steps:
            return

        if plan.spec_version != 1:
            raise InferenceUnavailableError(
                detail=(
                    f"Provider does not support PreparePlan spec_version="
                    f"{plan.spec_version}. Upgrade ryotenkai_providers."
                ),
                context={"reason": "SINGLENODE_PREPARE_SPEC_VERSION_UNSUPPORTED"},
            )

        engine_serve_image = _resolve_engine_image(self._engine_cfg.kind)
        hf_token = str(self._secrets.hf_token)

        # Determine log path once (provider concern — operators expect a
        # stable file). Same fallback as legacy: tests run without
        # PipelineOrchestrator.init_run_logging() and we don't want to
        # crash. ``prepare.log`` (renamed from ``merge.log``).
        try:
            log_dir = get_run_log_dir()
        except RuntimeError:
            log_dir = Path("/tmp/helix_logs")
        prepare_log_path = log_dir / "inference" / "prepare.log"
        prepare_log_path.parent.mkdir(parents=True, exist_ok=True)

        # Phase 7: legacy MLflow ``log_event_*`` removed; lifecycle is
        # captured via typed events on the journal. ``plan_started_at``
        # is no longer recorded as a separate timestamp.

        step: PrepareStep
        for _index, step in enumerate(plan.steps):
            step_started_at = time.time()
            container_name = f"helix-prepare-{run_id}-{step.name}"
            image = step.image if step.image else engine_serve_image

            # Phase 7: legacy MLflow ``log_event_*`` removed; lifecycle is
            # captured via typed events on the journal.

            # 1. Pull image (idempotent — docker layer cache makes this cheap).
            try:
                self._ensure_docker_image(ssh=ssh, image=image)
            except RyotenkAIError as exc:
                err_msg = exc.detail or str(exc)
                # Phase 7: legacy MLflow ``log_event_*`` removed; lifecycle is
                # captured via typed events on the journal.
                raise InferenceUnavailableError(
                    detail=f"Prepare step {step.name!r} image not available: {err_msg}",
                    context={"reason": "SINGLENODE_PREPARE_IMAGE_PULL_FAILED"},
                    cause=exc,
                ) from exc

            # 2. Clean each declared output (host-side, via volume mapping).
            for container_output in step.outputs:
                host_output = self._container_to_host(
                    container_output, workspace_host_path=workspace_host_path
                )
                ssh.exec_command(
                    f"rm -rf {host_output} && mkdir -p {host_output}",
                    timeout=_SETUP_CMD_TIMEOUT_S,
                    silent=True,
                )

            # 3. Build + start the container.
            cmd = format_prepare_step(
                step,
                image=image,
                container_name=container_name,
                extra_env={"HF_TOKEN": hf_token},
            )
            logger.info(f"Starting prepare step '{step.name}': {image}")
            ok, _stdout, stderr = ssh.exec_command(cmd, timeout=_QUICK_CMD_TIMEOUT_S, silent=False)
            if not ok:
                logger.error(f"Failed to start prepare container: {stderr[:LOG_OUTPUT_LONG_CHARS]}")
                # Phase 7: legacy MLflow ``log_event_*`` removed; lifecycle is
                # captured via typed events on the journal.
                raise InferenceUnavailableError(
                    detail=f"Prepare container start failed for {step.name!r}: {stderr[:LOG_OUTPUT_LONG_CHARS]}",
                    context={"reason": "SINGLENODE_PREPARE_CONTAINER_START_FAILED"},
                )

            # 4. Poll: collect logs, watch exit, enforce timeout.
            marker_seen = step.success_marker is None  # no marker required ⇒ already "seen"
            exit_code: int | None = None
            started_polling = time.time()
            while True:
                if time.time() - started_polling > step.timeout_seconds:
                    # Best-effort cleanup on timeout — swallow Docker errors.
                    try:
                        self._docker.rm_force(
                            ssh, container_name=container_name, timeout_seconds=_QUICK_CMD_TIMEOUT_S
                        )
                    except (ProviderUnavailableError, ConfigInvalidError) as exc:
                        logger.debug(f"Timeout-path cleanup rm_force failed: {exc}")
                    # Phase 7: legacy MLflow ``log_event_*`` removed; lifecycle is
                    # captured via typed events on the journal.
                    raise InferenceUnavailableError(
                        detail=f"Prepare step {step.name!r} timed out after {step.timeout_seconds}s",
                        context={"reason": "SINGLENODE_PREPARE_TIMEOUT"},
                    )
                is_running = self._docker.is_container_running(
                    ssh, name_filter=container_name, timeout_seconds=5
                )
                # logs() now raises on failure (Phase A2 Batch 4) — best-effort
                # in the polling loop: a transient docker-daemon error must not
                # abort the poll, just skip log capture for this iteration.
                logs_content = ""
                try:
                    logs_content = self._docker.logs(
                        ssh, container_name=container_name, timeout_seconds=10
                    )
                except (ProviderUnavailableError, ConfigInvalidError) as exc:
                    logger.debug(f"Polling-loop logs fetch failed: {exc}")
                if logs_content:
                    try:
                        prepare_log_path.write_text(logs_content, encoding=_ENCODING_UTF8)
                    except OSError as e:
                        logger.warning(f"Failed to write prepare logs to {prepare_log_path}: {e}")
                    if step.success_marker and step.success_marker in logs_content:
                        marker_seen = True
                if not is_running:
                    # container_exit_code raises on failure — fall through to
                    # exit_code=None which the next block treats as failure.
                    try:
                        exit_code = self._docker.container_exit_code(
                            ssh, container_name=container_name, timeout_seconds=5
                        )
                    except (ProviderUnavailableError, ConfigInvalidError) as exc:
                        logger.debug(f"container_exit_code lookup failed: {exc}")
                    break
                time.sleep(_PREPARE_POLL_INTERVAL_S)

            # 5. Cleanup container regardless of outcome.
            try:
                self._docker.rm_force(
                    ssh, container_name=container_name, timeout_seconds=_QUICK_CMD_TIMEOUT_S
                )
            except (ProviderUnavailableError, ConfigInvalidError) as exc:
                logger.debug(f"Post-step cleanup rm_force failed: {exc}")

            step_duration = time.time() - step_started_at

            # 6. Validate exit code.
            if exit_code != 0:
                # Phase 7: legacy MLflow ``log_event_*`` removed; lifecycle is
                # captured via typed events on the journal.
                raise InferenceUnavailableError(
                    detail=(
                        f"Prepare step {step.name!r} failed with exit code "
                        f"{exit_code}. Check {prepare_log_path}"
                    ),
                    context={"reason": "SINGLENODE_PREPARE_CONTAINER_FAILED"},
                )

            # 7. Validate success marker (when declared).
            if not marker_seen:
                # Phase 7: legacy MLflow ``log_event_*`` removed; lifecycle is
                # captured via typed events on the journal.
                raise InferenceUnavailableError(
                    detail=(
                        f"Prepare step {step.name!r} did not emit success "
                        f"marker {step.success_marker!r}. Check {prepare_log_path}"
                    ),
                    context={"reason": "SINGLENODE_PREPARE_NO_SUCCESS_MARKER"},
                )

            # 8. Validate success artifact (when declared).
            if step.success_artifact:
                host_artifact = self._container_to_host(
                    step.success_artifact, workspace_host_path=workspace_host_path
                )
                verify_cmd = f"test -f {host_artifact} && echo OK || echo MISSING"
                _ok_v, v_stdout, _v_stderr = ssh.exec_command(
                    verify_cmd, timeout=10, silent=True
                )
                if "OK" not in v_stdout:
                    host_dir = str(Path(host_artifact).parent)
                    _ok_ls, ls_stdout, _ls_stderr = ssh.exec_command(
                        f"ls -lah {host_dir} || true", timeout=10, silent=True
                    )
                    # Phase 7: legacy MLflow ``log_event_*`` removed; lifecycle is
                    # captured via typed events on the journal.
                    raise InferenceUnavailableError(
                        detail=(
                            f"Prepare step {step.name!r} reported success but artifact "
                            f"was not found on host. Expected: {host_artifact}. "
                            f"Directory:\n{ls_stdout[:_DIR_LISTING_MAX_CHARS]}"
                        ),
                        context={"reason": "SINGLENODE_PREPARE_ARTIFACTS_NOT_FOUND"},
                    )

            # Phase 7: legacy MLflow ``log_event_*`` removed; lifecycle is
            # captured via typed events on the journal.
            logger.info(f"✅ Prepare step '{step.name}' completed in {step_duration:.1f}s")

        # Phase 7: legacy MLflow ``log_event_*`` (plan completed) removed;
        # the typed event journal is the SSOT for prepare-plan lifecycle.

    @staticmethod
    def _container_to_host(
        container_path: str,
        *,
        workspace_host_path: str,
        workspace_container_path: str = "/workspace",
    ) -> str:
        """Inverse of :meth:`_host_to_container` — map a container path to host.

        Used by the prepare-plan runner to clean output dirs and verify
        artifacts on the host filesystem.
        """
        p = container_path.rstrip("/")
        if p == workspace_container_path:
            return workspace_host_path
        if p.startswith(workspace_container_path + "/"):
            return workspace_host_path + p[len(workspace_container_path) :]
        return container_path

    def _start_engine_container(
        self,
        *,
        ssh: SSHClient,
        engine_cfg: VLLMEngineConfig,
        workspace_host_path: str,
        model_path_in_container: str,
    ) -> None:
        """Pull image and start the vLLM serving container.

        Raises:
            InferenceUnavailableError: image pull failed or docker run
                returned non-zero.
        """
        host_bind = self._provider_cfg.inference.serve.host
        port = self._provider_cfg.inference.serve.port

        # Same unified inference image as the merge step — pinned in
        # :data:`INFERENCE_IMAGES`.
        serve_image = _resolve_engine_image(self._engine_cfg.kind)

        # Ensure serve image is available
        try:
            self._ensure_docker_image(
                ssh=ssh,
                image=serve_image,
            )
        except RyotenkAIError as exc:
            raise InferenceUnavailableError(
                detail=f"Serve image not available: {exc.detail or exc}",
                context={"reason": "SINGLENODE_SERVE_IMAGE_PULL_FAILED"},
                cause=exc,
            ) from exc

        # Engine returns a structured LaunchSpec; provider formats it into a
        # docker shell command. A k8s provider would translate the same spec
        # into a ContainerSpec instead — engines stay docker-agnostic.
        from ryotenkai_providers.inference.launch import format_docker_run

        spec = self._engine.build_launch_spec(
            cfg=engine_cfg,
            image=serve_image,
            container_name=self._CONTAINER_NAME,
            port=port,
            workspace_host_path=workspace_host_path,
            model_path_in_container=model_path_in_container,
        )
        run_cmd = format_docker_run(spec, host_bind=host_bind)

        # Stop old container (if exists) and start new one — best-effort.
        try:
            self._docker.rm_force(ssh, container_name=self._CONTAINER_NAME, timeout_seconds=60)
        except (ProviderUnavailableError, ConfigInvalidError) as exc:
            logger.debug(f"Pre-start container cleanup failed: {exc}")

        ok, _stdout, stderr = ssh.exec_command(run_cmd, timeout=PULL_TIMEOUT, silent=False)
        if not ok:
            # Best-effort log fetch for diagnostics — discard exceptions.
            try:
                self._docker.logs(
                    ssh,
                    container_name=self._CONTAINER_NAME,
                    tail=LOG_OUTPUT_SHORT_CHARS,
                    timeout_seconds=60,
                )
            except (ProviderUnavailableError, ConfigInvalidError) as exc:
                logger.debug(f"Diagnostic logs fetch failed: {exc}")
            raise InferenceUnavailableError(
                detail=f"Failed to start vLLM container: {stderr[:LOG_OUTPUT_LONG_CHARS]}",
                context={"reason": "SINGLENODE_VLLM_START_FAILED"},
            )

    @staticmethod
    def _write_runtime_json_best_effort(
        *,
        ssh: SSHClient,
        run_dir: str,
        runtime: dict[str, Any],
    ) -> bool:
        try:
            content = json.dumps(runtime, ensure_ascii=False, indent=2)
            b64 = base64.b64encode(content.encode("utf-8")).decode("ascii")
            cmd = f"echo '{b64}' | base64 -d > {run_dir}/runtime.json"
            ok, _stdout, _stderr = ssh.exec_command(cmd, timeout=_QUICK_CMD_TIMEOUT_S, silent=True)
            return ok
        except Exception:
            return False
