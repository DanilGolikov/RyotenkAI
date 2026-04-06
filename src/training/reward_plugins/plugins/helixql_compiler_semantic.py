from __future__ import annotations

from difflib import SequenceMatcher
import re
from typing import Any

from src.training.reward_plugins.base import RewardPlugin
from src.training.reward_plugins.registry import RewardPluginRegistry
from src.utils.domains.helixql import extract_query_text, extract_schema_block, semantic_match_details
from src.utils.domains.helixql_cli import HelixCompiler

_DEFAULT_TIMEOUT_SECONDS = 10
_BACKEND_COMPILE = "compile"
_BACKEND_SEMANTIC_ONLY = "semantic_only"
_SUPPORTED_BACKENDS = frozenset({_BACKEND_COMPILE, _BACKEND_SEMANTIC_ONLY})
_QUERY_PREFIX_RE = re.compile(r"^\s*QUERY\b", flags=re.IGNORECASE)
_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


@RewardPluginRegistry.register
class HelixQLCompilerSemanticRewardPlugin(RewardPlugin):
    """Domain plugin for HelixQL GRPO/SAPO reward."""

    name = "helixql_compiler_semantic"

    def __init__(self, params: dict[str, Any]):
        self._compiler: HelixCompiler | None = None
        super().__init__(params)

    def _validate_params(self) -> None:
        backend = str(self.params.get("validation_backend", _BACKEND_COMPILE)).strip().lower()
        if backend not in _SUPPORTED_BACKENDS:
            raise ValueError(
                f"Unsupported validation_backend={backend!r}. Expected one of {sorted(_SUPPORTED_BACKENDS)}"
            )

    def _backend(self) -> str:
        return str(self.params.get("validation_backend", _BACKEND_COMPILE)).strip().lower()

    def _get_compiler(self) -> HelixCompiler:
        if self._compiler is None:
            timeout = int(self.params.get("timeout_seconds", _DEFAULT_TIMEOUT_SECONDS))
            self._compiler = HelixCompiler(timeout_seconds=timeout)
        return self._compiler

    def _semantic_only_score(self, *, output: str, reference: str, prompt: str) -> float:
        """
        Softer reward for smoke RL when compile-time validation is unavailable.

        We still prefer the domain-aware score from `semantic_match_details`, but when it
        collapses to 0.0 (common for early random generations), we fall back to a lexical
        similarity score so GRPO/SAPO can see non-zero variation and actually update.
        """
        details = semantic_match_details(candidate=output, expected=reference, user_text=prompt)
        score = float(details["score"])
        if score > 0.0:
            return score

        output_text = (output or "").strip()
        reference_text = (reference or "").strip()
        if not output_text or not reference_text:
            return 0.0

        output_lower = output_text.lower()
        reference_lower = reference_text.lower()
        seq = SequenceMatcher(a=output_lower, b=reference_lower).ratio()
        output_tokens = set(_TOKEN_RE.findall(output_lower))
        reference_tokens = set(_TOKEN_RE.findall(reference_lower))
        union = output_tokens | reference_tokens
        jaccard = (len(output_tokens & reference_tokens) / len(union)) if union else 0.0
        prefix_bonus = 0.1 if _QUERY_PREFIX_RE.search(output_text) else 0.0

        fallback = (0.65 * seq) + (0.25 * jaccard) + prefix_bonus
        return round(max(0.0, min(1.0, fallback)), 4)

    def build_trainer_kwargs(
        self,
        *,
        train_dataset: Any,
        phase_config: Any,
        pipeline_config: Any,
    ) -> dict[str, Any]:
        del phase_config, pipeline_config
        features = getattr(train_dataset, "features", {}) or {}
        available_fields = set(features.keys()) if hasattr(features, "keys") else set()
        required = {"prompt", "reference_answer"}
        missing = sorted(required - available_fields)
        if missing:
            raise ValueError(
                "Reward plugin 'helixql_compiler_semantic' requires dataset fields "
                f"{sorted(required)}. Missing: {missing}"
            )

        backend = self._backend()
        compiler = self._get_compiler() if backend == _BACKEND_COMPILE else None

        def compiler_reward(completions: Any, **kwargs: Any) -> list[float]:
            outputs = [extract_query_text(item) for item in completions]
            prompts = _coerce_column(kwargs, "prompt", len(outputs))
            schemas = _coerce_column(kwargs, "schema_context", len(outputs))

            scores: list[float] = []
            for idx, output in enumerate(outputs):
                schema_text = schemas[idx] or extract_schema_block(prompts[idx])
                if not schema_text.strip() or not output.strip():
                    scores.append(-1.0)
                    continue
                if compiler is None:
                    scores.append(0.0)
                    continue
                result = compiler.validate(schema=schema_text, query=output)
                scores.append(1.0 if result.ok else -1.0)
            return scores

        def semantic_reward(completions: Any, **kwargs: Any) -> list[float]:
            outputs = [extract_query_text(item) for item in completions]
            prompts = _coerce_column(kwargs, "prompt", len(outputs))
            schemas = _coerce_column(kwargs, "schema_context", len(outputs))
            references = _coerce_column(kwargs, "reference_answer", len(outputs))

            scores: list[float] = []
            for idx, output in enumerate(outputs):
                if not output.strip():
                    scores.append(0.0)
                    continue
                if backend == _BACKEND_COMPILE:
                    schema_text = schemas[idx] or extract_schema_block(prompts[idx])
                    if not schema_text.strip():
                        scores.append(0.0)
                        continue
                    if compiler is None:
                        scores.append(0.0)
                        continue
                    result = compiler.validate(schema=schema_text, query=output)
                    if not result.ok:
                        scores.append(0.0)
                        continue
                if backend == _BACKEND_SEMANTIC_ONLY:
                    scores.append(
                        self._semantic_only_score(output=output, reference=references[idx], prompt=prompts[idx])
                    )
                    continue
                details = semantic_match_details(candidate=output, expected=references[idx], user_text=prompts[idx])
                scores.append(float(details["score"]))
            return scores

        compiler_reward.__name__ = "compiler_reward"
        semantic_reward.__name__ = "semantic_reward"
        reward_funcs = [compiler_reward, semantic_reward] if backend == _BACKEND_COMPILE else [semantic_reward]
        return {"reward_funcs": reward_funcs}

    def build_config_kwargs(
        self,
        *,
        train_dataset: Any,
        phase_config: Any,
        pipeline_config: Any,
    ) -> dict[str, Any]:
        del train_dataset, phase_config, pipeline_config
        backend = self._backend()
        reward_weights = [1.0, 1.0] if backend == _BACKEND_COMPILE else [1.0]
        return {"reward_weights": reward_weights}


def _coerce_column(kwargs: dict[str, Any], key: str, size: int) -> list[str]:
    value = kwargs.get(key)
    if value is None:
        return [""] * size
    if isinstance(value, list):
        coerced = [extract_query_text(item) for item in value]
        if len(coerced) < size:
            coerced.extend([""] * (size - len(coerced)))
        return coerced
    return [extract_query_text(value) for _ in range(size)]


__all__ = ["HelixQLCompilerSemanticRewardPlugin"]
