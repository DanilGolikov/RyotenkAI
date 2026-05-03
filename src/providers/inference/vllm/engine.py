"""
vLLM engine helper.

This module only builds commands. Execution is done by providers via SSH.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.utils.config import InferenceVLLMEngineConfig


class VLLMEngine:
    """vLLM OpenAI-compatible server (Docker-based)."""

    engine_type: str = "vllm"

    @staticmethod
    def build_healthcheck_command(*, host: str, port: int) -> str:
        """
        Build a remote command that checks readiness.

        Simple curl-based check. Returns exit code 0 on success.
        Readiness criteria (MVP): `GET /v1/models` returns 2xx.
        """
        url = f"http://{host}:{port}/v1/models"
        return f"curl -s -f -m 5 {url} >/dev/null 2>&1 && echo 1 || echo 0"

    @staticmethod
    def build_docker_run_command(
        *,
        cfg: InferenceVLLMEngineConfig,
        image: str,
        container_name: str,
        host_bind: str,
        port: int,
        workspace_host_path: str,
        model_path_in_container: str,
    ) -> str:
        """
        Build a docker command to start vLLM OpenAI-compatible server.

        Notes:
        - binds to 0.0.0.0 in container
        - host port is restricted by docker `-p host:container` mapping
        - cache is persisted in mounted workspace (/workspace/hf_cache)
        """
        vllm_args: list[str] = [
            "serve",
            json.dumps(model_path_in_container),
            "--host",
            "0.0.0.0",
            "--port",
            str(port),
            "--tensor-parallel-size",
            str(cfg.tensor_parallel_size),
            "--max-model-len",
            str(cfg.max_model_len),
            "--gpu-memory-utilization",
            str(cfg.gpu_memory_utilization),
        ]
        if cfg.quantization:
            vllm_args.extend(["--quantization", cfg.quantization])
        if cfg.enforce_eager:
            vllm_args.append("--enforce-eager")

        vllm_cmd = " ".join(vllm_args)

        return (
            "docker run --detach "
            f"--name {container_name} "
            "--gpus all "
            f"-p {host_bind}:{port}:{port} "
            f"-v {workspace_host_path}:/workspace "
            "-e HF_HOME=/workspace/hf_cache "
            "-e HUGGINGFACE_HUB_CACHE=/workspace/hf_cache "
            "-e TRANSFORMERS_CACHE=/workspace/hf_cache "
            f"{image} "
            f"{vllm_cmd}"
        )
