from __future__ import annotations

from typing import Any

from src.training.reward_plugins.base import RewardPlugin
from src.training.reward_plugins.registry import RewardPluginRegistry
from src.utils.domains.helixql import extract_query_text, extract_schema_block, semantic_match_details
from src.utils.domains.helixql_cli import HelixCompiler

_DEFAULT_TIMEOUT_SECONDS = 10


@RewardPluginRegistry.register
class HelixQLCompilerSemanticRewardPlugin(RewardPlugin):
    """Domain plugin for HelixQL GRPO/SAPO reward."""

    name = "helixql_compiler_semantic"

    def __init__(self, params: dict[str, Any]):
        self._compiler: HelixCompiler | None = None
        super().__init__(params)

    def _validate_params(self) -> None:
        pass

    def _get_compiler(self) -> HelixCompiler:
        if self._compiler is None:
            timeout = int(self.params.get("timeout_seconds", _DEFAULT_TIMEOUT_SECONDS))
            self._compiler = HelixCompiler(timeout_seconds=timeout)
        return self._compiler

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

        compiler = self._get_compiler()

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
                schema_text = schemas[idx] or extract_schema_block(prompts[idx])
                if not schema_text.strip() or not output.strip():
                    scores.append(0.0)
                    continue
                result = compiler.validate(schema=schema_text, query=output)
                if not result.ok:
                    scores.append(0.0)
                    continue
                details = semantic_match_details(candidate=output, expected=references[idx], user_text=prompts[idx])
                scores.append(float(details["score"]))
            return scores

        compiler_reward.__name__ = "compiler_reward"
        semantic_reward.__name__ = "semantic_reward"
        return {
            "reward_funcs": [compiler_reward, semantic_reward],
            "reward_weights": [1.0, 1.0],
        }


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
