"""
EvaluationRunner — orchestrates the evaluation pipeline.

Responsibilities:
1. Load JSONL dataset → list of (question, expected_answer, context)
2. Collect model answers via IModelInference
3. Run enabled plugins (sorted by priority)
4. Aggregate results → RunSummary
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.utils.logger import logger

if TYPE_CHECKING:
    from src.config.evaluation.schema import EvaluationConfig
    from src.evaluation.model_client.interfaces import IModelInference
    from src.evaluation.plugins.base import EvalResult, EvalSample
    from src.pipeline.stages.model_evaluator import EvaluatorEventCallbacks


@dataclass
class RunSummary:
    """
    Aggregated output of EvaluationRunner.run().

    Attributes:
        plugin_results:  Per-plugin EvalResult objects (keyed by plugin instance id).
        plugin_meta:     Per-plugin metadata keyed by plugin instance id.
        overall_passed:  True if ALL enabled plugins passed.
        sample_count:    Number of samples evaluated.
        duration_seconds: Total evaluation time.
        skipped_plugins: Plugin instance ids skipped due to missing expected_answer or other reasons.
        errors:          Fatal errors that prevented evaluation.
    """

    plugin_results: dict[str, EvalResult] = field(default_factory=dict)
    plugin_meta: dict[str, dict[str, Any]] = field(default_factory=dict)
    overall_passed: bool = True
    sample_count: int = 0
    duration_seconds: float = 0.0
    skipped_plugins: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-serializable dict for MLflow artifact logging."""
        return {
            "overall_passed": self.overall_passed,
            "sample_count": self.sample_count,
            "duration_seconds": round(self.duration_seconds, 2),
            "skipped_plugins": self.skipped_plugins,
            "errors": self.errors,
            "plugins": {
                name: {
                    "plugin_name": self.plugin_meta.get(name, {}).get("plugin_name", ""),
                    "passed": r.passed,
                    "description": self.plugin_meta.get(name, {}).get("description", ""),
                    "params": self.plugin_meta.get(name, {}).get("params", {}),
                    "thresholds": self.plugin_meta.get(name, {}).get("thresholds", {}),
                    "metrics": r.metrics,
                    "errors": r.errors,
                    "recommendations": r.recommendations,
                    "sample_count": r.sample_count,
                    "failed_samples": r.failed_samples,
                }
                for name, r in self.plugin_results.items()
            },
        }


class EvaluationRunner:
    """
    Runs the evaluation pipeline for a single ModelEvaluator stage execution.

    Usage:
        runner = EvaluationRunner(eval_config)
        summary = runner.run(inference_client)
    """

    def __init__(
        self,
        eval_config: EvaluationConfig,
        callbacks: EvaluatorEventCallbacks | None = None,
        secrets_resolver: Any | None = None,
    ) -> None:
        """
        Args:
            eval_config:      Populated EvaluationConfig from pipeline config.
            callbacks:        Optional event callbacks (MLflow, logging, etc.).
            secrets_resolver: SecretsResolver instance for plugins that declare
                              ``[secrets] required = [...]`` in their manifest.toml.
                              Pass None if no plugins in the config require secrets.
        """
        self._eval_config = eval_config
        self._callbacks = callbacks
        self._secrets_resolver = secrets_resolver

    def run(self, inference_client: IModelInference) -> RunSummary:
        """
        Execute the full evaluation flow.

        Steps:
          1. Load dataset
          2. Collect model answers
          3. Run plugins in priority order
          4. Return aggregated summary

        Args:
            inference_client: Client to call the trained model endpoint.

        Returns:
            RunSummary with per-plugin results and overall verdict.
        """
        start = time.time()
        summary = RunSummary()

        # Step 1: Load dataset
        dataset_cfg = self._eval_config.dataset
        if dataset_cfg is None:
            summary.errors.append("evaluation.dataset is not configured")
            summary.overall_passed = False
            summary.duration_seconds = time.time() - start
            return summary

        dataset_path = Path(dataset_cfg.path)
        load_result = self._load_dataset(dataset_path)
        if load_result is None:
            summary.errors.append(f"Failed to load eval dataset from {dataset_path}")
            summary.overall_passed = False
            summary.duration_seconds = time.time() - start
            return summary

        raw_samples = load_result
        logger.info(f"[EVAL] Loaded {len(raw_samples)} samples from {dataset_path.name}")

        # Step 2: Collect model answers
        logger.info(f"[EVAL] Sending {len(raw_samples)} samples to inference endpoint for answer collection ...")
        samples = self._collect_model_answers(raw_samples, inference_client)
        summary.sample_count = len(samples)
        logger.info(f"[EVAL] Collected model answers for {len(samples)} samples")

        if self._callbacks and self._callbacks.on_eval_start:
            self._callbacks.on_eval_start(str(dataset_path), len(samples))

        # Step 2a: Persist human-readable answers.md (best-effort)
        if self._eval_config.save_answers_md:
            self._save_answers_md(samples)

        # Step 3: Build plugins from config
        plugins = self._build_plugins()
        if not plugins:
            summary.errors.append("No evaluation plugins enabled or registered")
            summary.overall_passed = False
            summary.duration_seconds = time.time() - start
            return summary

        # Step 4: Run plugins in user-declared order (config YAML list is
        # order-preserving). No more priority-based sort — the manifest
        # field was dead weight for evaluation.
        for plugin_cfg, plugin in plugins:
            plugin_id = plugin_cfg.id
            # Skip plugin if it requires expected_answer but none are available
            if plugin.requires_expected_answer:
                has_expected = any(s.expected_answer is not None for s in samples)
                if not has_expected:
                    logger.warning(
                        f"[EVAL] Skipping plugin '{plugin_id}' ({plugin.name}): "
                        "requires_expected_answer=True but no expected_answer in dataset"
                    )
                    summary.skipped_plugins.append(plugin_id)
                    continue

            logger.info(f"[EVAL] Running plugin '{plugin_id}' ({plugin.name}) ...")
            plugin_start = time.time()

            if self._callbacks and self._callbacks.on_plugin_start:
                self._callbacks.on_plugin_start(plugin_id, plugin.name, plugin.get_description())

            try:
                result = plugin.evaluate(samples)
            except Exception as e:
                logger.error(f"[EVAL] Plugin '{plugin.name}' raised an exception: {e}")
                from src.evaluation.plugins.base import EvalResult

                result = EvalResult(
                    plugin_name=plugin.name,
                    passed=False,
                    errors=[f"Plugin crashed: {e!s}"],
                    sample_count=len(samples),
                )

            plugin_duration = time.time() - plugin_start
            plugin_duration_ms = plugin_duration * 1000
            summary.plugin_results[plugin_id] = result
            summary.plugin_meta[plugin_id] = {
                "plugin_name": plugin.name,
                "description": plugin.get_description(),
                "params": plugin.params,
                "thresholds": plugin.thresholds,
            }

            status = "PASSED" if result.passed else "FAILED"
            logger.info(
                f"[EVAL] Plugin '{plugin_id}' ({plugin.name}): {status} "
                f"({plugin_duration:.1f}s) — metrics: {result.metrics}"
            )

            if result.passed:
                if self._callbacks and self._callbacks.on_plugin_complete:
                    self._callbacks.on_plugin_complete(plugin_id, plugin.name, result.metrics, plugin_duration_ms)
            else:
                summary.overall_passed = False
                for rec in result.recommendations:
                    logger.info(f"[EVAL]   Recommendation: {rec}")
                if self._callbacks and self._callbacks.on_plugin_failed:
                    self._callbacks.on_plugin_failed(
                        plugin_id,
                        plugin.name,
                        result.metrics,
                        result.errors,
                        result.recommendations,
                        plugin_duration_ms,
                    )

        summary.duration_seconds = time.time() - start

        if self._callbacks and self._callbacks.on_eval_complete:
            self._callbacks.on_eval_complete(summary.overall_passed, summary.sample_count, summary.duration_seconds)

        return summary

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load_dataset(self, path: Path) -> list[dict[str, Any]] | None:
        """Load JSONL dataset. Returns None on failure."""
        if not path.exists():
            logger.error(f"[EVAL] Dataset not found: {path}")
            return None

        rows: list[dict[str, Any]] = []
        try:
            with path.open("r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning(f"[EVAL] Skipping malformed JSON at line {line_num}: {e}")
        except OSError as e:
            logger.error(f"[EVAL] Cannot read dataset {path}: {e}")
            return None

        return rows if rows else None

    def _collect_model_answers(
        self,
        raw_samples: list[dict[str, Any]],
        inference_client: IModelInference,
    ) -> list[EvalSample]:
        """
        Iterate over dataset rows, call inference, build EvalSample list.

        Dataset row format (flexible):
            {"question": "...", "expected_answer": "...", "docs": "...", ...}
            {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

        Any fields beyond the reserved keys (question, expected_answer, answer,
        context, messages) are collected into EvalSample.metadata and passed
        through to plugins unchanged.
        """
        from src.evaluation.plugins.base import EvalSample

        samples: list[EvalSample] = []

        for idx, row in enumerate(raw_samples):
            question, expected_answer, metadata = self._extract_row_fields(row)

            if not question:
                logger.debug(f"[EVAL] Skipping row {idx}: no question/user message found")
                continue

            try:
                model_answer = inference_client.generate(question)
            except RuntimeError as e:
                logger.warning(f"[EVAL] Inference failed for sample {idx}: {e}. Using empty answer.")
                model_answer = ""

            samples.append(
                EvalSample(
                    question=question,
                    model_answer=model_answer,
                    expected_answer=expected_answer,
                    metadata=metadata,
                )
            )

        return samples

    # Keys extracted explicitly — everything else goes into metadata.
    _RESERVED_FIELDS: frozenset[str] = frozenset({
        "question", "prompt", "expected_answer", "answer",
        "completion", "reference_answer", "context", "messages",
    })

    # Alias mapping: dataset field → canonical eval field.
    _QUESTION_ALIASES: tuple[str, ...] = ("question", "prompt")
    _ANSWER_ALIASES: tuple[str, ...] = ("expected_answer", "answer", "completion", "reference_answer")

    @staticmethod
    def _extract_row_fields(row: dict[str, Any]) -> tuple[str, str | None, dict[str, Any]]:
        """
        Extract (question, expected_answer, metadata) from a dataset row.

        Supports three formats:
        1. Flat: {"question": "...", "expected_answer": "...", <extra fields>}
        2. Prompt-completion: {"prompt": "...", "completion": "...", <extra fields>}
        3. Messages: {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}

        All keys not in _RESERVED_FIELDS are collected into the metadata dict
        and passed through to plugins via EvalSample.metadata.
        """
        question: str = ""
        expected_answer: str | None = None

        for alias in EvaluationRunner._QUESTION_ALIASES:
            if alias in row:
                question = str(row.get(alias) or "")
                break

        if question:
            for alias in EvaluationRunner._ANSWER_ALIASES:
                val = row.get(alias)
                if val is not None:
                    expected_answer = str(val)
                    break

        elif "messages" in row:
            messages = row.get("messages") or []
            for msg in messages:
                role = msg.get("role", "")
                content = str(msg.get("content") or "")
                if role == "user" and not question:
                    question = content
                elif role == "assistant" and expected_answer is None:
                    expected_answer = content

        metadata = {k: v for k, v in row.items() if k not in EvaluationRunner._RESERVED_FIELDS}

        return question, expected_answer, metadata

    def _save_answers_md(self, samples: list[EvalSample]) -> None:
        """
        Write runs/{run}/evaluation/answers.md with question / model answer / expected answer
        for every collected sample. Best-effort: errors are logged but never propagate.
        """
        try:
            from src.utils.logger import get_run_log_dir

            out_dir = get_run_log_dir() / "evaluation"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "answers.md"

            lines: list[str] = ["# Evaluation Answers\n"]
            for idx, sample in enumerate(samples):
                lines.append(f"\n### Task {idx + 1}\n\n")
                lines.append(f"~~~\n{sample.question}\n~~~\n")
                lines.append("\n### Trained model answer\n\n")
                lines.append(f"~~~\n{sample.model_answer}\n~~~\n")
                lines.append("\n### Expected answer\n\n")
                expected = sample.expected_answer if sample.expected_answer is not None else "_not set_"
                lines.append(f"~~~\n{expected}\n~~~\n")
                lines.append("\n---\n")

            out_path.write_text("".join(lines), encoding="utf-8")
            logger.info(f"[EVAL] Answers saved to {out_path}")
        except Exception as e:
            logger.warning(f"[EVAL] Failed to save answers.md (non-fatal): {e}")

    def _build_plugins(self) -> list[tuple[Any, EvaluatorPlugin]]:
        """
        Instantiate enabled plugins from evaluation config.

        Run explicit plugin discovery first, then instantiate enabled plugin instances.
        """
        from src.community.catalog import catalog
        from src.evaluation.plugins.registry import EvaluatorPluginRegistry

        catalog.ensure_loaded()
        plugins: list[tuple[Any, EvaluatorPlugin]] = []

        for plugin_cfg in self._eval_config.evaluators.plugins:
            plugin_name = plugin_cfg.plugin
            if not plugin_cfg.enabled:
                logger.debug(f"[EVAL] Plugin instance '{plugin_cfg.id}' is disabled, skipping")
                continue

            if not EvaluatorPluginRegistry.is_registered(plugin_name):
                logger.warning(
                    f"[EVAL] Plugin '{plugin_name}' for instance '{plugin_cfg.id}' is not registered. "
                    "Check plugin name in config and ensure the plugin module is imported."
                )
                continue

            try:
                plugin_cls = EvaluatorPluginRegistry.get(plugin_name)
                plugin = plugin_cls(params=plugin_cfg.params, thresholds=plugin_cfg.thresholds)

                required_secrets_keys: tuple[str, ...] | None = getattr(plugin_cls, "_required_secrets", None)
                if required_secrets_keys:
                    if self._secrets_resolver is None:
                        raise RuntimeError(
                            f"Plugin '{plugin_name}' requires secrets {list(required_secrets_keys)} "
                            "but no SecretsResolver was provided to EvaluationRunner. "
                            "Ensure ModelEvaluator passes secrets to EvaluationRunner."
                        )
                    resolved = self._secrets_resolver.resolve(required_secrets_keys)
                    object.__setattr__(plugin, "_secrets", resolved)
                    logger.debug(f"[EVAL] Injected {len(resolved)} secret(s) for plugin instance '{plugin_cfg.id}'")

                object.__setattr__(plugin, "_save_report", bool(plugin_cfg.save_report))

                plugins.append((plugin_cfg, plugin))
                logger.debug(f"[EVAL] Loaded plugin instance '{plugin_cfg.id}' using plugin '{plugin_name}'")
            except Exception as e:
                logger.error(f"[EVAL] Failed to instantiate plugin instance '{plugin_cfg.id}' ({plugin_name}): {e}")

        return plugins


# Alias used in type hints inside _build_plugins (to avoid circular import at module level)
EvaluatorPlugin = Any


__all__ = [
    "EvaluationRunner",
    "RunSummary",
]
