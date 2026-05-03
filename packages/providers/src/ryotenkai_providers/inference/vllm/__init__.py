"""vLLM inference engine helpers.

Builds the docker-run + healthcheck commands a provider executes over
SSH. Engine itself is stateless — only command construction lives here.

Originally hosted under ``src.pipeline.inference.vllm``; moved to the
providers namespace in Phase A.3 of monorepo packagization (plan §A.3) —
vLLM commands belong with the provider that runs them, not with the
pipeline orchestrator.
"""

from __future__ import annotations

from ryotenkai_providers.inference.vllm.engine import VLLMEngine

__all__ = ["VLLMEngine"]
