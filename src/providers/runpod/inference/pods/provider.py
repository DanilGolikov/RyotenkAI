"""
RunPod Pods inference provider (vLLM on a stopped/resumable Pod + Network Volume).

Responsibilities:
- Ensure a RunPod Network Volume exists for persistent HF cache and artifacts.
- Ensure a Pod exists with that Network Volume attached and keep it stopped by default.
- Return a local (SSH-tunnel) OpenAI-compatible base URL for generated scripts.

Non-goals:
- Running merge + vLLM inside the pipeline (handled by generated scripts for interactive use).
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.constants import INFERENCE_MANIFEST_FILENAME, PROVIDER_RUNPOD
from src.providers.constants import KEY_ID, KEY_NAME, RETRY_BACKOFF_FACTOR, SHA12_LEN
from src.providers.inference.interfaces import (
    EndpointInfo,
    IInferenceProvider,
    InferenceArtifacts,
    InferenceArtifactsContext,
    InferenceCapabilities,
    InferenceEventLogger,
    PipelineReadinessMode,
)
from src.utils.logger import logger
from src.utils.result import Err, InferenceError, Ok, Result

from ...pod_control import RunPodInferencePodControl
from .api_client import RunPodPodsRESTClient
from .artifacts import CHAT_SCRIPT as _CHAT_SCRIPT
from .artifacts import render_readme as _render_readme
from .constants import (
    POD_HF_CACHE_DIR,
    POD_LOCK_DIR,
    POD_WORKSPACE,
    RUNPOD_REST_API_BASE_URL,
    SESSION_KEY_ADAPTER_REF,
    SESSION_KEY_BASE_MODEL_ID,
    pod_hash_file,
    pod_log_file,
    pod_merged_dir,
    pod_pid_file,
    pod_run_dir,
)

_POD_NAME_MAX_LEN = 191
_RETRY_SLEEP_CAP_SEC = 12.0
_UNKNOWN_ERROR = "<unknown>"

if TYPE_CHECKING:
    from src.config.providers.runpod import RunPodProviderConfig
    from src.utils.config import InferenceVLLMEngineConfig, PipelineConfig, Secrets


def _sha12(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:SHA12_LEN]


class RunPodPodInferenceProvider(IInferenceProvider):
    """Provision a RunPod Pod + Network Volume and keep it stopped by default."""

    def __init__(self, *, config: PipelineConfig, secrets: Secrets):
        self._cfg = config
        self._secrets = secrets
        self._inf_cfg = config.inference
        self._engine_cfg: InferenceVLLMEngineConfig = self._inf_cfg.engines.vllm

        provider_cfg_raw = self._cfg.get_provider_config(PROVIDER_RUNPOD)
        from src.config.providers.runpod import RunPodProviderConfig

        self._provider_cfg: RunPodProviderConfig = RunPodProviderConfig(**provider_cfg_raw)

        # Inference (Pods) config is stored under providers.runpod.inference.*
        pods_cfg = self._provider_cfg.inference
        if pods_cfg.pod is None:
            raise ValueError(
                "RunPod inference provider requires providers.runpod.inference.pod config block. "
                "Add it under providers.runpod.inference (volume is optional; keep providers.runpod.connect.ssh.key_path set)."
            )

        self._volume_cfg = pods_cfg.volume  # None = no network volume (pod will use volumeInGb)
        self._pod_cfg = pods_cfg.pod
        self._serve_cfg = pods_cfg.serve

        self._api: RunPodPodsRESTClient | None = None
        self._pod_control: RunPodInferencePodControl | None = None
        self._endpoint_info: EndpointInfo | None = None

        self._network_volume_id: str | None = None
        self._network_volume_meta: dict[str, Any] | None = None
        self._pod_id: str | None = None
        self._pod_name: str | None = None
        self._event_logger: InferenceEventLogger | None = None

        # Populated by activate_for_eval(); consumed by deactivate_after_eval()
        self._eval_session: Any | None = None
        # Resolved adapter ref (HF repo or path) from deploy(); used by activate_for_eval for merge
        self._adapter_ref: str = ""

    # ---------------------------------------------------------------------
    # Interface properties
    # ---------------------------------------------------------------------

    @property
    def provider_name(self) -> str:
        return PROVIDER_RUNPOD

    @property
    def provider_type(self) -> str:
        return PROVIDER_RUNPOD

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
        quantization: str | None = None,
        keep_running: bool = False,
    ) -> Result[EndpointInfo, InferenceError]:
        """
        Prepare a stopped Pod for interactive inference.

        Notes:
        - We intentionally do NOT run merge/vLLM here to avoid GPU billing in the pipeline.
        - Readiness and runtime actions are handled by generated scripts (chat/stop).
        - If keep_running=True (e.g. evaluation is about to start), the pod is NOT stopped
          after provisioning to avoid a wasteful stop/start cycle.
        """

        _ = (run_id, trust_remote_code, quantization)  # reserved for deterministic naming later

        adapter_ref = (lora_path or model_source or "").strip()
        if adapter_ref.startswith(("/", "~", ".", "file:")):
            return Err(
                InferenceError(
                    message=(
                        "runpod: adapter_ref looks like a local filesystem path on the orchestrator and cannot be used. "
                        "Upload the adapter to Hugging Face (preferred) or use a provider that can access local paths."
                    ),
                    code="RUNPOD_ADAPTER_LOCAL_PATH_NOT_SUPPORTED",
                )
            )
        self._adapter_ref = adapter_ref  # Resolved model source for activate_for_eval (merge LoRA)

        api_key = getattr(self._secrets, "runpod_api_key", None)
        if not api_key:
            return Err(
                InferenceError(
                    message="RUNPOD_API_KEY is missing (required for runpod inference provider)",
                    code="RUNPOD_API_KEY_MISSING",
                )
            )

        api_key_str = str(api_key)
        self._api = RunPodPodsRESTClient(api_key=api_key_str)
        self._pod_control = RunPodInferencePodControl(api=self._api)

        # Fail-fast: SSH key path must exist locally (used later by scripts).
        key_path = Path(str(self._provider_cfg.connect.ssh.key_path)).expanduser()
        if not key_path.exists():
            return Err(
                InferenceError(
                    message=f"runpod: SSH key not found at: {key_path}",
                    code="RUNPOD_SSH_KEY_NOT_FOUND",
                )
            )

        # Warn if both network volume and pod volume are configured (network volume wins per RunPod API).
        if self._volume_cfg is not None and int(self._pod_cfg.volume_disk_gb) > 0:
            logger.warning(
                "runpod: both providers.runpod.inference.volume and pod.volume_disk_gb are configured. "
                "Network volume takes priority — pod.volume_disk_gb will be ignored by RunPod."
            )

        # 1) Ensure network volume exists (optional)
        network_volume_id: str | None = None
        if self._volume_cfg is not None:
            vol_res = self._ensure_network_volume()
            if vol_res.is_failure():
                vol_err = vol_res.unwrap_err()
                return Err(
                    vol_err
                    if isinstance(vol_err, InferenceError)
                    else InferenceError(message=str(vol_err), code="RUNPOD_VOLUME_ENSURE_FAILED")
                )
            network_volume_id = vol_res.unwrap()
            self._network_volume_id = network_volume_id

        # 2) Ensure pod exists
        pod_res = self._ensure_pod(network_volume_id=network_volume_id, key_path=key_path)
        if pod_res.is_failure():
            pod_err = pod_res.unwrap_err()
            return Err(
                pod_err
                if isinstance(pod_err, InferenceError)
                else InferenceError(message=str(pod_err), code="RUNPOD_POD_ENSURE_FAILED")
            )
        pod_id, pod_name = pod_res.unwrap()
        self._pod_id = pod_id
        self._pod_name = pod_name

        # 3) Stop pod after provisioning to avoid GPU billing — UNLESS keep_running=True
        #    (e.g. evaluation is enabled and activate_for_eval() will be called right after,
        #    so stopping and immediately restarting is a wasteful no-op with zero GPU risk).
        if keep_running:
            logger.info(
                "✅ RunPod Pod provisioned (kept running): pod_id=%s volume_id=%s",
                pod_id,
                network_volume_id or "none (pod volume)",
            )
        else:
            stop_res = self._stop_pod_if_running(pod_id=pod_id)
            if stop_res.is_failure():
                stop_err = stop_res.unwrap_err()
                return Err(
                    stop_err
                    if isinstance(stop_err, InferenceError)
                    else InferenceError(message=str(stop_err), code="RUNPOD_POD_STOP_FAILED")
                )
            logger.info(
                "✅ RunPod Pod prepared (stopped): pod_id=%s volume_id=%s",
                pod_id,
                network_volume_id or "none (pod volume)",
            )

        # Local access via SSH tunnel (scripts)
        port = int(self._serve_cfg.port)
        endpoint_url = f"http://127.0.0.1:{port}/v1"
        health_url = f"http://127.0.0.1:{port}/v1/models"

        self._endpoint_info = EndpointInfo(
            endpoint_url=endpoint_url,
            api_type="openai_compatible",
            provider_type=self.provider_type,
            engine="vllm",
            model_id=base_model_id,
            health_url=health_url,
            resource_id=pod_id,
        )
        return Ok(self._endpoint_info)

    def set_event_logger(self, event_logger: InferenceEventLogger | None) -> None:
        # Currently unused for pods (kept for interface uniformity).
        self._event_logger = event_logger

    def get_pipeline_readiness_mode(self) -> PipelineReadinessMode:
        # Pod is intentionally stopped after provisioning to avoid GPU billing.
        return PipelineReadinessMode.SKIP

    def collect_startup_logs(self, *, local_path: Path) -> None:
        # No-op: pod logs are best inspected via SSH / RunPod console during interactive session.
        _ = local_path
        return

    def build_inference_artifacts(
        self, *, ctx: InferenceArtifactsContext
    ) -> Result[InferenceArtifacts, InferenceError]:
        pod_id = self._pod_id or ctx.endpoint.resource_id
        if not (isinstance(pod_id, str) and pod_id.strip()):
            return Err(
                InferenceError(
                    message=f"runpod: cannot determine pod_id from EndpointInfo: {ctx.endpoint!r}",
                    code="RUNPOD_POD_ID_MISSING",
                )
            )
        pod_id = pod_id.strip()

        network_volume_id: str | None = self._network_volume_id or (
            self._volume_cfg.id if self._volume_cfg is not None else None
        )
        if isinstance(network_volume_id, str):
            network_volume_id = network_volume_id.strip() or None

        pod_name = self._pod_name
        if not (isinstance(pod_name, str) and pod_name.strip()):
            if network_volume_id:
                suffix = _sha12(network_volume_id)
            else:
                suffix = "ephemeral"
            pod_name = f"{self._pod_cfg.name_prefix}-{suffix}".replace('"', "").strip()
            if len(pod_name) > _POD_NAME_MAX_LEN:
                pod_name = pod_name[:_POD_NAME_MAX_LEN]

        serve_port = int(self._serve_cfg.port)
        workspace = "/workspace"
        hf_cache_dir = f"{workspace}/hf_cache"
        run_dir = f"{workspace}/runs/{ctx.run_name}"
        merged_dir = f"{run_dir}/model"

        runpod_manifest: dict[str, Any] = {
            "rest_api_base_url": RUNPOD_REST_API_BASE_URL,
            "pod": {
                KEY_ID: pod_id,
                KEY_NAME: pod_name,
                "image_name": self._pod_cfg.image_name,
                "gpu_type_ids": self._pod_cfg.gpu_type_ids,
                "gpu_count": self._pod_cfg.gpu_count,
                "allowed_cuda_versions": self._pod_cfg.allowed_cuda_versions,
                "container_disk_gb": self._pod_cfg.container_disk_gb,
                "volume_disk_gb": self._pod_cfg.volume_disk_gb,
                "ports": self._pod_cfg.ports,
            },
        }
        if network_volume_id:
            vol_meta = self._network_volume_meta if isinstance(self._network_volume_meta, dict) else {}
            vol_name = str(vol_meta.get(KEY_NAME) or (self._volume_cfg.name if self._volume_cfg else ""))
            vol_size_raw = vol_meta.get("size")
            vol_size_gb = (
                int(vol_size_raw)
                if isinstance(vol_size_raw, int)
                else (int(self._volume_cfg.size_gb) if self._volume_cfg else 0)
            )
            vol_dc = str(
                vol_meta.get("dataCenterId") or (self._volume_cfg.data_center_id if self._volume_cfg else "") or ""
            ).strip()

            network_volume_manifest: dict[str, Any] = {
                KEY_ID: network_volume_id,
                KEY_NAME: vol_name,
                "size_gb": vol_size_gb,
            }
            if vol_dc:
                network_volume_manifest["data_center_id"] = vol_dc
            runpod_manifest["network_volume"] = network_volume_manifest

        manifest: dict[str, Any] = {
            "run_name": ctx.run_name,
            "mlflow_run_id": ctx.mlflow_run_id,
            "provider": self.provider_type,
            "engine": self._inf_cfg.engine,
            "runpod": runpod_manifest,
            "ssh": {
                "key_path": self._provider_cfg.connect.ssh.key_path,
            },
            "serve": {
                "port": serve_port,
                "workspace": workspace,
                "hf_cache_dir": hf_cache_dir,
                "run_dir": run_dir,
                "merged_model_dir": merged_dir,
                "vllm_pid_file": f"{run_dir}/vllm.pid",
                "vllm_log_file": f"{run_dir}/vllm.log",
                "config_hash_file": f"{run_dir}/config_hash.txt",
                "lock_dir": f"{workspace}/.helix_inference_lock",
            },
            "model": {
                SESSION_KEY_BASE_MODEL_ID: self._cfg.model.name,
                SESSION_KEY_ADAPTER_REF: ctx.model_source,
                "trust_remote_code": bool(self._cfg.model.trust_remote_code),
            },
            "llm": self._resolve_llm_manifest_block(),
            "vllm": self._engine_cfg.model_dump(mode="python", exclude_none=True),  # noqa: WPS226
            "endpoint": {
                "client_base_url": ctx.endpoint.endpoint_url,
                "health_url": ctx.endpoint.health_url,
            },
            "config_hash": _sha12(
                json.dumps(
                    {
                        SESSION_KEY_BASE_MODEL_ID: self._cfg.model.name,
                        SESSION_KEY_ADAPTER_REF: ctx.model_source,
                        "provider": self._provider_cfg.model_dump(mode="python", exclude_none=True),
                        "engine": self._engine_cfg.model_dump(mode="python", exclude_none=True),
                        "serve": {
                            "port": serve_port,
                            "workspace": workspace,
                        },
                    },
                    sort_keys=True,
                )
            ),
        }

        return Ok(
            InferenceArtifacts(
                manifest=manifest,
                chat_script=_CHAT_SCRIPT,
                readme=_render_readme(
                    manifest_filename=INFERENCE_MANIFEST_FILENAME,
                    endpoint_url=ctx.endpoint.endpoint_url,
                ),
            )
        )

    def undeploy(self) -> Result[None, InferenceError]:
        # Policy: stop instead of delete (reuse like a personal PC).
        if not self._pod_control or not self._pod_id:
            return Ok(None)
        res = self._pod_control.stop_pod(pod_id=self._pod_id)
        if res.is_failure():
            stop_err = res.unwrap_err()
            return Err(InferenceError(message=str(stop_err), code="RUNPOD_POD_STOP_FAILED"))
        return Ok(None)

    def health_check(self) -> Result[bool, InferenceError]:
        # Pipeline health_check is not meaningful for a stopped pod.
        # We treat "pod exists" as healthy at provisioning time.
        if not self._api or not self._pod_id:
            return Err(
                InferenceError(
                    message="RunPod Pods client not initialized (deploy was not called)",
                    code="RUNPOD_NOT_DEPLOYED",
                )
            )
        res = self._api.get_pod(pod_id=self._pod_id)
        if res.is_failure():
            api_err = res.unwrap_err()
            return Err(InferenceError(message=str(api_err), code="RUNPOD_POD_GET_FAILED"))
        return Ok(True)

    def get_capabilities(self) -> InferenceCapabilities:
        return InferenceCapabilities(
            provider_type=self.provider_type,
            supported_engines=["vllm"],
            supports_lora=True,
            supports_streaming=True,
        )

    def get_endpoint_info(self) -> EndpointInfo | None:
        return self._endpoint_info

    def _build_eval_session_params(self) -> dict[str, Any]:
        """
        Build parameters for pod_session.activate() from stored provider state.

        Centralises all path/hash/token computation so that activate_for_eval()
        remains a thin orchestrator: guards → params → activate → store result.
        Must only be called after deploy() has populated self._pod_id and self._adapter_ref.
        """
        run_key = self._pod_id
        if not run_key:
            raise ValueError("_pod_id is not set; call deploy() before _build_eval_session_params()")
        serve_port = int(self._serve_cfg.port)

        config_hash = _sha12(
            json.dumps(
                {
                    SESSION_KEY_BASE_MODEL_ID: self._cfg.model.name,
                    SESSION_KEY_ADAPTER_REF: self._adapter_ref,
                    "provider": self._provider_cfg.model_dump(mode="python", exclude_none=True),
                    "engine": self._engine_cfg.model_dump(mode="python", exclude_none=True),
                    "serve": {"port": serve_port, "workspace": POD_WORKSPACE},
                },
                sort_keys=True,
            )
        )

        return {
            "key_path": Path(str(self._provider_cfg.connect.ssh.key_path)).expanduser(),
            "serve_port": serve_port,
            "run_dir": pod_run_dir(run_key),
            "merged_dir": pod_merged_dir(run_key),
            "hf_cache_dir": POD_HF_CACHE_DIR,
            "pid_file": pod_pid_file(run_key),
            "log_file": pod_log_file(run_key),
            "hash_file": pod_hash_file(run_key),
            "lock_dir": POD_LOCK_DIR,
            "config_hash": config_hash,
            SESSION_KEY_BASE_MODEL_ID: self._cfg.model.name,
            SESSION_KEY_ADAPTER_REF: self._adapter_ref,
            "trust_remote_code": bool(self._cfg.model.trust_remote_code),
            "hf_token": str(getattr(self._secrets, "hf_token", "") or "").strip(),
            "vllm_cfg": self._engine_cfg.model_dump(mode="python", exclude_none=True),  # noqa: WPS226
        }

    def activate_for_eval(self) -> Result[str, InferenceError]:
        """
        Bring up the pod for evaluation:
        1. Start pod
        2. Wait for SSH
        3. Open SSH tunnel
        4. Merge LoRA (idempotent)
        5. Start vLLM
        6. Wait for /v1/models health
        7. Return live endpoint URL
        """
        pod_control = getattr(self, "_pod_control", None)
        pod_id = getattr(self, "_pod_id", None)
        adapter_ref = getattr(self, "_adapter_ref", None)

        if pod_control is None:
            return Err(
                InferenceError(
                    message="runpod_pods: activate_for_eval called before deploy() — API client not initialized",
                    code="RUNPOD_NOT_DEPLOYED",
                )
            )
        if not pod_id:
            return Err(
                InferenceError(
                    message="runpod_pods: activate_for_eval called before deploy() — pod_id not set",
                    code="RUNPOD_NOT_DEPLOYED",
                )
            )
        if not (isinstance(adapter_ref, str) and adapter_ref.strip()):
            return Err(
                InferenceError(
                    message=(
                        "runpod_pods: activate_for_eval requires resolved model_source (HF repo or path). "
                        "deploy() stores it from InferenceDeployer._resolve_model_source; ensure model_source=auto "
                        "and ModelRetriever provides hf_repo_id or local_model_path."
                    ),
                    code="RUNPOD_ADAPTER_REF_MISSING",
                )
            )

        params = self._build_eval_session_params()

        from src.providers.runpod.inference.pods import pod_session

        session_res = pod_session.activate(
            api=pod_control,
            pod_id=pod_id,
            **params,
        )
        if session_res.is_failure():
            return Err(
                InferenceError(
                    message=str(session_res.unwrap_err()),  # type: ignore[union-attr]
                    code="RUNPOD_EVAL_ACTIVATE_FAILED",
                )
            )

        self._eval_session = session_res.unwrap()
        return Ok(self._eval_session.endpoint_url)

    def deactivate_after_eval(self) -> Result[None, InferenceError]:
        """
        Shut down the evaluation session:
        1. Kill vLLM process inside the pod
        2. Close SSH tunnel
        3. Delete the Pod (Network Volume is preserved)

        Fallback: if activate_for_eval() was never called but a pod was already
        provisioned (e.g. Ctrl+C between deploy() and activate_for_eval()), the
        pod is deleted directly via API to avoid ongoing GPU billing.
        """
        pod_control = getattr(self, "_pod_control", None)
        pod_id = getattr(self, "_pod_id", None)
        eval_session = getattr(self, "_eval_session", None)

        if pod_control is None:
            # deploy() was never called — nothing to clean up.
            return Ok(None)

        if eval_session is None:
            # activate_for_eval() was not called, but a pod may already exist.
            if pod_id:
                logger.info(
                    "[CLEANUP] activate_for_eval was not called but pod %s exists — deleting to avoid billing.",
                    pod_id,
                )
                del_res = pod_control.delete_pod(pod_id=pod_id)
                if del_res.is_failure():
                    del_err = del_res.unwrap_err()
                    return Err(
                        InferenceError(
                            message=f"deactivate_after_eval fallback delete failed: {del_err}",
                            code="RUNPOD_POD_DELETE_FAILED",
                        )
                    )
                self._pod_id = None
            return Ok(None)

        key_path = Path(str(self._provider_cfg.connect.ssh.key_path)).expanduser()

        from src.providers.runpod.inference.pods import pod_session

        result = pod_session.deactivate(
            api=pod_control,
            state=eval_session,
            key_path=key_path,
        )
        self._eval_session = None
        if result.is_failure():
            err = result.unwrap_err()
            return Err(InferenceError(message=str(err), code="RUNPOD_EVAL_DEACTIVATE_FAILED"))
        return Ok(None)

    # ---------------------------------------------------------------------
    # Internals
    # ---------------------------------------------------------------------

    def _ensure_network_volume(self) -> Result[str, InferenceError]:
        assert self._api is not None
        assert self._volume_cfg is not None  # only called when volume is configured
        cfg = self._volume_cfg

        def _select_by_name(
            *,
            volumes: list[dict[str, Any]],
            name: str,
            data_center_id: str | None = None,
        ) -> Result[dict[str, Any] | None, InferenceError]:
            matches_name: list[dict[str, Any]] = []
            for v in volumes:
                if str(v.get("name") or "") != name:
                    continue
                matches_name.append(v)

            if data_center_id:
                matches = [v for v in matches_name if str(v.get("dataCenterId") or "") == str(data_center_id)]
            else:
                matches = matches_name

            if not matches:
                return Ok(None)

            if len(matches) == 1:
                return Ok(matches[0])

            ids = [str(v.get(KEY_ID) or "").strip() for v in matches]
            ids = [x for x in ids if x]
            preview = ", ".join(ids[:5])
            more = "" if len(ids) <= 5 else f" (+{len(ids) - 5} more)"
            return Err(
                InferenceError(
                    message=(
                        "runpod: multiple network volumes found with the same name (and datacenter filter, if set). "
                        "Set providers.runpod.inference.volume.id explicitly. "
                        f"name={name!r} ids=[{preview}{more}]"
                    ),
                    code="RUNPOD_VOLUME_AMBIGUOUS",
                )
            )

        # Prefer explicit id if provided.
        if cfg.id:
            get_res = self._api.get_network_volume(network_volume_id=str(cfg.id))
            if get_res.is_success():
                vol_by_id = get_res.unwrap()
                vol_id = str(vol_by_id.get(KEY_ID) or "").strip()
                if vol_id:
                    self._network_volume_meta = vol_by_id if isinstance(vol_by_id, dict) else None
                    return Ok(vol_id)
            return Err(
                InferenceError(
                    message=f"runpod: network volume id not found or inaccessible: {cfg.id!r}",
                    code="RUNPOD_VOLUME_NOT_FOUND",
                )
            )

        list_res = self._api.list_network_volumes()
        if list_res.is_failure():
            list_err = list_res.unwrap_err()
            return Err(InferenceError(message=str(list_err), code="RUNPOD_VOLUME_LIST_FAILED"))
        volumes = list_res.unwrap()

        sel_res = _select_by_name(volumes=volumes, name=cfg.name, data_center_id=cfg.data_center_id)
        if sel_res.is_failure():
            return Err(sel_res.unwrap_err())  # type: ignore[return-value]
        selected = sel_res.unwrap()
        if selected is not None:
            vol_id = str(selected.get(KEY_ID) or "").strip()
            if not vol_id:
                return Err(
                    InferenceError(
                        message=f"runpod: unexpected network volume object without id: {selected!r}",
                        code="RUNPOD_VOLUME_NO_ID",
                    )
                )
            self._network_volume_meta = selected
            return Ok(vol_id)

        data_center_id = str(cfg.data_center_id or "").strip()
        if not data_center_id:
            return Err(
                InferenceError(
                    message=(
                        "runpod: network volume not found and auto-create requires providers.runpod.inference.volume.data_center_id "
                        "(RunPod REST API requires dataCenterId at creation time). "
                        "Set it to a valid RunPod datacenter id (e.g. US-KS-2) or create the volume manually and set providers.runpod.inference.volume.id. "
                        f"name={cfg.name!r}"
                    ),
                    code="RUNPOD_VOLUME_DATA_CENTER_MISSING",
                )
            )

        logger.info(
            "☁️ Creating RunPod Network Volume: name=%r size_gb=%d data_center_id=%r",
            cfg.name,
            int(cfg.size_gb),
            data_center_id,
        )

        payload: dict[str, Any] = {
            "name": cfg.name,
            "size": int(cfg.size_gb),
            "dataCenterId": data_center_id,
        }

        last_err: str = _UNKNOWN_ERROR
        max_attempts = 4
        for attempt in range(1, max_attempts + 1):
            # Before a new create attempt, re-check if the previous attempt actually succeeded server-side.
            if attempt > 1:
                list_res0 = self._api.list_network_volumes()
                if list_res0.is_success():
                    sel0 = _select_by_name(
                        volumes=list_res0.unwrap(),
                        name=cfg.name,
                        data_center_id=data_center_id,
                    )
                    if sel0.is_success():
                        vol0 = sel0.unwrap()
                        if isinstance(vol0, dict):
                            vol_id0 = str(vol0.get(KEY_ID) or "").strip()
                            if vol_id0:
                                self._network_volume_meta = vol0
                                logger.warning(
                                    "runpod: network volume appears after previous create attempt. Proceeding with discovered id."
                                )
                                return Ok(vol_id0)
                    else:
                        return Err(sel0.unwrap_err())  # type: ignore[return-value]

            create_res = self._api.create_network_volume(payload=payload)
            if create_res.is_success():
                created = create_res.unwrap()
                vol_id = str(created.get(KEY_ID) or "").strip()
                if vol_id:
                    self._network_volume_meta = created if isinstance(created, dict) else None
                    return Ok(vol_id)
                last_err = f"runpod: create_network_volume succeeded but id is missing: {created!r}"
            else:
                last_err = str(create_res.unwrap_err())

            # Re-check by deterministic key: name (+ datacenter as tie-breaker).
            list_res2 = self._api.list_network_volumes()
            if list_res2.is_success():
                sel2 = _select_by_name(
                    volumes=list_res2.unwrap(),
                    name=cfg.name,
                    data_center_id=data_center_id,
                )
                if sel2.is_success():
                    vol2 = sel2.unwrap()
                    if isinstance(vol2, dict):
                        vol_id2 = str(vol2.get(KEY_ID) or "").strip()
                        if vol_id2:
                            self._network_volume_meta = vol2
                            logger.warning(
                                "runpod: create_network_volume did not return a usable response, "
                                "but volume now exists. Proceeding with discovered id."
                            )
                            return Ok(vol_id2)
                else:
                    return Err(sel2.unwrap_err())  # type: ignore[return-value]

            if attempt < max_attempts:
                sleep_s = min(RETRY_BACKOFF_FACTOR * (2 ** (attempt - 1)), _RETRY_SLEEP_CAP_SEC)
                logger.warning(
                    "runpod: failed to create network volume (attempt %d/%d). Retrying in %.1fs. Error: %s",
                    attempt,
                    max_attempts,
                    sleep_s,
                    last_err,
                )
                time.sleep(sleep_s)
            else:
                logger.warning(
                    "runpod: failed to create network volume (attempt %d/%d). No more retries. Error: %s",
                    attempt,
                    max_attempts,
                    last_err,
                )

        return Err(
            InferenceError(
                message=f"runpod: failed to create network volume after {max_attempts} attempts: {last_err}",
                code="RUNPOD_VOLUME_CREATE_FAILED",
            )
        )

    def _ensure_pod(self, *, network_volume_id: str | None, key_path: Path) -> Result[tuple[str, str], InferenceError]:
        assert self._api is not None
        pod_cfg = self._pod_cfg

        # Deterministic name: with volume — bound to volume (one pod per volume); without — fixed ephemeral suffix.
        if network_volume_id:
            suffix = _sha12(network_volume_id)
        else:
            suffix = "ephemeral"
        name_val = f"{pod_cfg.name_prefix}-{suffix}".replace('"', "").strip()
        if len(name_val) > _POD_NAME_MAX_LEN:
            name_val = name_val[:_POD_NAME_MAX_LEN]

        params: dict[str, Any] = {
            "computeType": "GPU",
            KEY_NAME: name_val,
        }
        if network_volume_id:
            params["networkVolumeId"] = network_volume_id
        pods_res = self._api.list_pods(params=params)
        if pods_res.is_failure():
            pods_err = pods_res.unwrap_err()
            return Err(InferenceError(message=str(pods_err), code="RUNPOD_POD_LIST_FAILED"))
        pods = pods_res.unwrap()

        # Prefer an already stopped pod; otherwise reuse running and stop later.
        chosen: dict[str, Any] | None = None
        for p in pods:
            if isinstance(p, dict) and str(p.get("desiredStatus") or "") == "EXITED":
                chosen = p
                break
        if chosen is None:
            for p in pods:
                if isinstance(p, dict) and str(p.get("desiredStatus") or "") == "RUNNING":
                    chosen = p
                    break

        if chosen is not None:
            pod_id = str(chosen.get(KEY_ID) or "").strip()
            if not pod_id:
                return Err(
                    InferenceError(
                        message=f"runpod: unexpected pod object without id: {chosen!r}",
                        code="RUNPOD_POD_NO_ID",
                    )
                )
            return Ok((pod_id, name_val))

        # No pod found → create a new one.
        public_key = ""
        pub_path = Path(str(key_path) + ".pub")
        if pub_path.exists():
            try:
                public_key = pub_path.read_text(encoding="utf-8").strip()
            except Exception:
                public_key = ""
        if not public_key:
            logger.warning(
                "runpod: public key file not found or empty next to ssh key; "
                "continuing without explicit PUBLIC_KEY env (may rely on RunPod injection). "
                f"expected: {pub_path}"
            )

        env: dict[str, str] = {}
        hf_token = str(getattr(self._secrets, "hf_token", "") or "").strip()
        if hf_token:
            env["HF_TOKEN"] = hf_token
        if public_key:
            env["PUBLIC_KEY"] = public_key

        # IMPORTANT: Network volumes for Pods are Secure Cloud only.
        # Without a network volume, any datacenter with available GPU resources can be used.
        payload: dict[str, Any] = {
            KEY_NAME: name_val,
            "cloudType": "SECURE",
            "computeType": "GPU",
            "imageName": pod_cfg.image_name,
            "gpuCount": int(pod_cfg.gpu_count),
            "gpuTypeIds": list(pod_cfg.gpu_type_ids),
            "gpuTypePriority": "availability",
            # NOTE:
            # We intentionally do NOT pass dataCenterIds here.
            # With networkVolumeId — RunPod derives the DC automatically from the volume location.
            # Without networkVolumeId — RunPod searches all available DCs for the requested GPU.
            "allowedCudaVersions": pod_cfg.allowed_cuda_versions,
            "containerDiskInGb": int(pod_cfg.container_disk_gb),
            "volumeMountPath": "/workspace",
            "ports": list(pod_cfg.ports),
            "supportPublicIp": True,
            "interruptible": False,
            "locked": False,
            "env": env,
        }
        if network_volume_id:
            payload["networkVolumeId"] = network_volume_id
        else:
            payload["volumeInGb"] = int(pod_cfg.volume_disk_gb)

        # Strip None values (RunPod API is strict about types).
        payload = {k: v for k, v in payload.items() if v is not None}

        logger.info(f"☁️ Creating RunPod Pod for inference: name={name_val!r} image={pod_cfg.image_name!r}")

        last_err: str = _UNKNOWN_ERROR
        max_attempts = 4
        for attempt in range(1, max_attempts + 1):
            # Before a new create attempt, re-check if the previous attempt actually succeeded server-side.
            if attempt > 1:
                pods_res0 = self._api.list_pods(params=params)
                if pods_res0.is_success():
                    pods0 = pods_res0.unwrap()
                    for p in pods0:
                        pod_id = str(p.get(KEY_ID) or "").strip()
                        if pod_id:
                            logger.warning(
                                "runpod: inference pod appears after previous create attempt (name+volume match). "
                                "Proceeding with discovered id."
                            )
                            return Ok((pod_id, name_val))

            create_res = self._api.create_pod(payload=payload)
            if create_res.is_success():
                pod = create_res.unwrap()
                pod_id = str(pod.get(KEY_ID) or "").strip()
                if pod_id:
                    return Ok((pod_id, name_val))
                last_err = f"runpod: create_pod succeeded but id is missing: {pod!r}"
            else:
                last_err = str(create_res.unwrap_err())

            # Re-check by deterministic name + network volume (create may have succeeded).
            pods_res2 = self._api.list_pods(params=params)
            if pods_res2.is_success():
                pods2 = pods_res2.unwrap()
                for p in pods2:
                    pod_id = str(p.get(KEY_ID) or "").strip()
                    if pod_id:
                        logger.warning(
                            "runpod: pod create did not return a usable response, "
                            "but pod now exists (name+volume match). Proceeding with discovered id."
                        )
                        return Ok((pod_id, name_val))

            if attempt < max_attempts:
                sleep_s = min(RETRY_BACKOFF_FACTOR * (2 ** (attempt - 1)), _RETRY_SLEEP_CAP_SEC)
                logger.warning(
                    "runpod: failed to create inference pod (attempt %d/%d). Retrying in %.1fs. Error: %s",
                    attempt,
                    max_attempts,
                    sleep_s,
                    last_err,
                )
                time.sleep(sleep_s)
            else:
                logger.warning(
                    "runpod: failed to create inference pod (attempt %d/%d). No more retries. Error: %s",
                    attempt,
                    max_attempts,
                    last_err,
                )

        return Err(
            InferenceError(
                message=f"runpod: failed to create inference pod after {max_attempts} attempts: {last_err}",
                code="RUNPOD_POD_CREATE_FAILED",
            )
        )

    def _stop_pod_if_running(self, *, pod_id: str) -> Result[None, InferenceError]:
        assert self._api is not None
        assert self._pod_control is not None
        get_res = self._api.get_pod(pod_id=pod_id)
        if get_res.is_failure():
            get_err = get_res.unwrap_err()
            return Err(InferenceError(message=str(get_err), code="RUNPOD_POD_GET_FAILED"))
        pod = get_res.unwrap()
        status = str(pod.get("desiredStatus") or "")
        if status == "RUNNING":
            logger.info(f"🛑 Stopping RunPod Pod after provisioning: {pod_id}")
            stop_res = self._pod_control.stop_pod(pod_id=pod_id)
            if stop_res.is_failure():
                stop_err = stop_res.unwrap_err()
                return Err(InferenceError(message=str(stop_err), code="RUNPOD_POD_STOP_FAILED"))
        return Ok(None)

    def _resolve_llm_manifest_block(self) -> dict[str, Any]:
        """
        Resolve the system prompt once and return the llm manifest block.

        The resolved prompt text (not path) is stored in the manifest so that
        the chat script on the remote machine does not need MLflow access.
        system_prompt_source carries audit metadata (origin type + name/path + version).
        """
        from src.evaluation.system_prompt import SystemPromptLoader

        llm_cfg = self._provider_cfg.inference.llm
        mlflow_cfg = getattr(getattr(self._cfg, "experiment_tracking", None), "mlflow", None)

        try:
            result = SystemPromptLoader.load(llm_cfg, mlflow_cfg=mlflow_cfg)
        except ValueError as exc:
            logger.error(f"[INFERENCE] System prompt configuration error: {exc}")
            result = None

        return {
            "system_prompt": result.text if result else None,
            "system_prompt_source": result.source if result else None,
        }


__all__ = [
    "RunPodPodInferenceProvider",
]
