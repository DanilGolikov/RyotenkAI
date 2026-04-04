"""
ModelCardGenerator — HuggingFace model card (README.md) generation.

Single responsibility: knows about Markdown/YAML formatting for HF model cards,
does NOT know about HF API or SSH.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.utils.logger import logger

from src.pipeline.stages.model_retriever.types import (
    ModelCardContext,
    _PHASE_IDX,
    _STRATEGY_TYPE,
    _STATUS,
)

if TYPE_CHECKING:
    from src.utils.config import PipelineConfig


class ModelCardGenerator:
    """
    Generates README.md content for HuggingFace model cards.

    Stateless: all inputs are passed explicitly to generate().
    """

    def generate(
        self,
        ctx: ModelCardContext | None,
        *,
        config: PipelineConfig,
        provider_name: str,
        provider_training_cfg: dict[str, Any],
    ) -> str:
        """Generate README.md content."""
        if ctx is None:
            ctx = ModelCardContext(phase_metrics=[], datasets=[])

        cfg = config
        repo_id = getattr(config, "hf_repo_id", None)
        try:
            hf_cfg = cfg.experiment_tracking.huggingface
            repo_id = hf_cfg.repo_id if hf_cfg else "unknown"
        except Exception:
            pass

        model_name = repo_id.split("/")[-1] if repo_id else "model"
        base_model = cfg.model.name or "Unknown"

        def _pick(row: dict[str, Any], *keys: str) -> Any:
            for k in keys:
                if k in row and row[k] is not None:
                    return row[k]
            return None

        def _fmt(v: Any, *, digits: int | None = None) -> str:
            if v is None:
                return "—"
            if isinstance(v, bool):
                return "true" if v else "false"
            if isinstance(v, int | str):
                return str(v)
            if isinstance(v, float):
                if digits is None:
                    return str(v)
                return f"{v:.{digits}f}"
            return str(v)

        tags: list[str] = []
        for t in ("fine-tuned", "adapter", "peft", "lora", cfg.training.type, "trl", "ryotenkai"):
            if isinstance(t, str) and t and t not in tags:
                tags.append(t)

        yaml_lines: list[str] = [
            "---",
            "license: apache-2.0",
            f"base_model: {base_model}",
            "base_model_relation: adapter",
            "library_name: transformers",
            "pipeline_tag: text-generation",
        ]

        if ctx.datasets:
            yaml_lines.append("datasets:")
            for ds in ctx.datasets:
                if isinstance(ds, str) and ds.strip():
                    yaml_lines.append(f"  - {ds.strip()}")

        yaml_lines.append("tags:")
        for t in tags:
            yaml_lines.append(f"  - {t}")
        yaml_lines.append("---")

        datasets_md = ", ".join(f"`{d}`" for d in ctx.datasets) if ctx.datasets else "—"

        results_lines: list[str] = []
        if ctx.training_started_at or ctx.training_completed_at:
            results_lines.append("### Run timeline")
            if ctx.training_started_at:
                results_lines.append(f"- **Started at**: `{ctx.training_started_at}`")
            if ctx.training_completed_at:
                results_lines.append(f"- **Completed at**: `{ctx.training_completed_at}`")
            results_lines.append("")

        if ctx.phase_metrics:
            results_lines.extend(
                [
                    "| Phase | Strategy | Status | train_loss | eval_loss | global_step | epoch | runtime_s | peak_mem_gb |",
                    "|---:|---|---|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for row in ctx.phase_metrics:
                phase_idx = _fmt(_pick(row, _PHASE_IDX))
                strategy = _fmt(_pick(row, _STRATEGY_TYPE))
                status = _fmt(_pick(row, _STATUS))
                train_loss = _fmt(_pick(row, "train_loss", "loss"), digits=4)
                eval_loss = _fmt(_pick(row, "eval_loss"), digits=4)
                global_step = _fmt(_pick(row, "global_step"))
                epoch = _fmt(_pick(row, "epoch"), digits=2)
                runtime_s = _fmt(_pick(row, "train_runtime"), digits=1)
                peak_mem = _fmt(_pick(row, "peak_memory_gb"), digits=2)
                results_lines.append(
                    f"| {phase_idx} | {strategy} | {status} | {train_loss} | {eval_loss} | "
                    f"{global_step} | {epoch} | {runtime_s} | {peak_mem} |"
                )
        else:
            results_lines.append("No per-phase metrics were found for this run.")

        trust_remote_code = bool(getattr(cfg.model, "trust_remote_code", False))

        hyper = cfg.training.hyperparams
        per_device_bs = getattr(hyper, "per_device_train_batch_size", None)
        grad_acc = getattr(hyper, "gradient_accumulation_steps", None)
        effective_bs: int | None = None
        if isinstance(per_device_bs, int) and isinstance(grad_acc, int):
            effective_bs = per_device_bs * grad_acc

        md: list[str] = []
        md.extend(yaml_lines)
        md.append("")
        md.append(f"# {model_name}")
        md.append("")
        md.append(f"This repository contains a **PEFT LoRA adapter** for `{base_model}`.")
        md.append("")
        md.append("## Model Details")
        md.append("")
        md.extend(
            [
                "| Parameter | Value |",
                "|---|---|",
                f"| **Base model** | `{base_model}` |",
                f"| **Adapter repo** | `{repo_id}` |",
                f"| **Training type** | `{cfg.training.type}` |",
                f"| **Strategy chain** | {self._format_strategies(cfg)} |",
                f"| **Dataset source** | `{_fmt(ctx.dataset_source_type)}` |",
                f"| **Datasets** | {datasets_md} |",
                f"| **Batch Size** | {hyper.per_device_train_batch_size} |",
                f"| **LoRA r** | `{self._get_lora_param(cfg, 'r')}` |",
                f"| **LoRA alpha** | `{self._get_lora_param(cfg, 'lora_alpha')}` |",
                f"| **LoRA dropout** | `{self._get_lora_param(cfg, 'lora_dropout')}` |",
                f"| **LoRA bias** | `{self._get_lora_param(cfg, 'bias')}` |",
                f"| **Target modules** | `{self._get_lora_param(cfg, 'target_modules')}` |",
                f"| **DoRA** | `{self._get_lora_param(cfg, 'use_dora')}` |",
                f"| **rsLoRA** | `{self._get_lora_param(cfg, 'use_rslora')}` |",
                f"| **Init** | `{self._get_lora_param(cfg, 'init_lora_weights')}` |",
            ]
        )

        md.append("")
        md.append("## Training Details")
        md.append("")
        md.extend(
            [
                "| Hyperparameter | Value |",
                "|---|---|",
                f"| **epochs** | `{hyper.epochs}` |",
                f"| **learning_rate** | `{hyper.learning_rate}` |",
                f"| **warmup_ratio** | `{hyper.warmup_ratio}` |",
                f"| **per_device_train_batch_size** | `{hyper.per_device_train_batch_size}` |",
                f"| **gradient_accumulation_steps** | `{hyper.gradient_accumulation_steps}` |",
                f"| **effective_batch_size** | `{_fmt(effective_bs)}` |",
                f"| **optimizer** | `{cfg.training.get_effective_optimizer()}` |",
                f"| **lr_scheduler** | `{hyper.lr_scheduler_type}` |",
            ]
        )

        md.append("")
        md.append("## Training Results")
        md.append("")
        md.extend(results_lines)

        md.append("")
        md.append("## Usage")
        md.append("")
        md.append("### Load as a PEFT adapter (recommended)")
        md.append("")
        md.append("```python")
        md.append("from transformers import AutoModelForCausalLM, AutoTokenizer")
        md.append("from peft import PeftModel")
        md.append("")
        md.append(f'base_model_id = "{base_model}"')
        md.append(f'adapter_id = "{repo_id}"')
        md.append("")
        md.append(
            f"tokenizer = AutoTokenizer.from_pretrained(adapter_id, trust_remote_code={trust_remote_code})"
        )
        md.append(
            "model = AutoModelForCausalLM.from_pretrained("
            'base_model_id, device_map="auto", torch_dtype="auto", trust_remote_code='
            f"{trust_remote_code})"
        )
        md.append("model = PeftModel.from_pretrained(model, adapter_id)")
        md.append("model.eval()")
        md.append("")
        md.append('prompt = "Hello!"')
        md.append('inputs = tokenizer(prompt, return_tensors="pt").to(model.device)')
        md.append("outputs = model.generate(**inputs, max_new_tokens=256)")
        md.append("print(tokenizer.decode(outputs[0], skip_special_tokens=True))")
        md.append("```")

        md.append("")
        md.append("### Merge adapter into base model (optional)")
        md.append("")
        md.append("```python")
        md.append("merged = model.merge_and_unload()")
        md.append('merged.save_pretrained("merged-model")')
        md.append('tokenizer.save_pretrained("merged-model")')
        md.append("```")

        md.append("")
        md.append("## Training Infrastructure")
        md.append("")
        md.append(f"- **Platform**: {provider_name}")
        md.append(f"- **GPU**: {provider_training_cfg.get('gpu_type', 'auto-detect')}")

        return "\n".join(md) + "\n"

    @staticmethod
    def _format_strategies(cfg: PipelineConfig) -> str:
        """Format strategy chain for display."""
        strategies = cfg.training.get_strategy_chain()
        if not strategies:
            return "SFT (default)"
        parts: list[str] = []
        for phase in strategies:
            epochs = phase.hyperparams.epochs
            if epochs is None:
                epochs = cfg.training.hyperparams.epochs
            parts.append(
                f"{phase.strategy_type.upper()} ({epochs}ep)"
                if epochs is not None
                else phase.strategy_type.upper()
            )
        return " → ".join(parts)

    @staticmethod
    def _get_lora_param(cfg: PipelineConfig, param: str) -> str:
        """Get LoRA/QLoRA parameter safely based on training type."""
        try:
            adapter_config = cfg.get_adapter_config()
            return str(getattr(adapter_config, param, "N/A"))
        except (AttributeError, ValueError):
            return "N/A"

    @staticmethod
    def _basename(path_str: str) -> str:
        """Return filename portion of a path string."""
        s = path_str.strip()
        if not s:
            return ""
        name = Path(s).name
        if "/" in name or "\\" in name:
            name = name.split("/")[-1].split("\\")[-1]
        if not name:
            s2 = s.rstrip("/\\")
            name = s2.split("/")[-1].split("\\")[-1] if s2 else ""
        return name


__all__ = ["ModelCardGenerator"]
