"""End-of-pipeline reporting: summary print, MLflow metric aggregation, report.

Extracted from PipelineOrchestrator. All four methods run only after the
pipeline finishes, so they operate purely on the final state + MLflow run
tree — they never touch the main execution loop.

The reporter holds only the immutable config. Context and MLflow manager
are passed in on each call so the orchestrator stays the single owner of
that mutable state.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Any

from src.config.datasets.constants import SOURCE_TYPE_HUGGINGFACE
from src.pipeline.constants import (
    CTX_PROVIDER_NAME_UNKNOWN,
    CTX_PROVIDER_TYPE_UNKNOWN,
    CTX_RUNTIME_SECONDS,
    CTX_TRAINING_DURATION,
    CTX_TRAINING_INFO,
    CTX_UPLOAD_DURATION,
    SECONDS_PER_HOUR,
    SEPARATOR_CHAR,
    SUMMARY_LINE_WIDTH,
)
from src.pipeline.stages import StageNames
from src.reports import ExperimentReportGenerator
from src.utils.config import AdaLoraConfig
from src.utils.logger import console, get_run_log_dir, logger

if TYPE_CHECKING:
    from collections.abc import Callable

    from src.training.managers.mlflow_manager import MLflowManager
    from src.utils.config import PipelineConfig


class ExecutionSummaryReporter:
    """Pipeline-end reporter: console summary + MLflow aggregation + experiment report."""

    def __init__(self, config: PipelineConfig) -> None:
        self._config = config

    # ---- console summary ----------------------------------------------------

    def print_summary(self, *, context: dict[str, Any]) -> None:
        """Print a comprehensive summary of the pipeline execution to the console."""
        console.print("\n" + SEPARATOR_CHAR * SUMMARY_LINE_WIDTH)
        console.print("[bold green]PIPELINE EXECUTION SUMMARY[/bold green]")
        console.print(SEPARATOR_CHAR * SUMMARY_LINE_WIDTH)

        self._print_configuration_section()
        self._print_dataset_section(context)
        deployer_ctx: dict = context.get(StageNames.GPU_DEPLOYER, {})
        self._print_training_section(context, deployer_ctx)
        self._print_gpu_provider_section(context, deployer_ctx)
        self._print_model_output_section(context)
        self._print_evaluation_section(context)
        self._print_launch_command_section(context)

        console.print("\n" + SEPARATOR_CHAR * SUMMARY_LINE_WIDTH)

    def _print_configuration_section(self) -> None:
        console.print("\n[bold cyan]Configuration:[/bold cyan]")
        console.print(f"   Model: {self._config.model.name}")
        console.print(f"   Training Type: {self._config.training.type}")
        console.print(f"   4-bit Quantization: {self._config.training.get_effective_load_in_4bit()}")
        try:
            adapter_cfg = self._config.get_adapter_config()
            if isinstance(adapter_cfg, AdaLoraConfig):
                console.print(f"   AdaLoRA init_r/target_r: {adapter_cfg.init_r}/{adapter_cfg.target_r}")
            else:
                console.print(f"   LoRA r/alpha: {adapter_cfg.r}/{adapter_cfg.lora_alpha}")
        except ValueError:
            pass  # Not qlora/lora training type
        console.print(f"   Batch Size: {self._config.training.hyperparams.per_device_train_batch_size}")
        strategies = self._config.training.get_strategy_chain()
        if strategies:
            console.print(f"   Strategies: {' -> '.join(s.strategy_type.upper() for s in strategies)}")

    def _print_dataset_section(self, context: dict[str, Any]) -> None:
        default_ds = self._config.get_primary_dataset()
        console.print("\n[bold cyan]Dataset:[/bold cyan]")
        if default_ds.get_source_type() == SOURCE_TYPE_HUGGINGFACE and default_ds.source_hf is not None:
            console.print(f"   Train (HF): {default_ds.source_hf.train_id}")
            if default_ds.source_hf.eval_id:
                console.print(f"   Eval  (HF): {default_ds.source_hf.eval_id}")
        elif default_ds.source_local is not None:
            console.print(f"   Train (local): {default_ds.source_local.local_paths.train}")
            if default_ds.source_local.local_paths.eval:
                console.print(f"   Eval  (local): {default_ds.source_local.local_paths.eval}")
        else:
            console.print("   [dim]Dataset source not configured[/dim]")
        if StageNames.DATASET_VALIDATOR in context:
            validator_ctx = context[StageNames.DATASET_VALIDATOR]
            console.print(f"   Samples: {validator_ctx.get('sample_count', 'N/A')}")
            if validator_ctx.get("avg_length"):
                console.print(f"   Avg Length: {validator_ctx.get('avg_length', 0):.0f} chars")
        adapter_type = default_ds.adapter_type or "auto-detect"
        console.print(f"   Adapter: {adapter_type}")

    def _print_training_section(self, context: dict[str, Any], deployer_ctx: dict) -> None:
        console.print("\n[bold cyan]Training:[/bold cyan]")
        if StageNames.TRAINING_MONITOR not in context:
            console.print("   [dim]Training info not available[/dim]")
            return
        monitor_ctx = context[StageNames.TRAINING_MONITOR]
        training_duration = monitor_ctx.get(CTX_TRAINING_DURATION) or monitor_ctx.get(CTX_TRAINING_INFO, {}).get(
            CTX_RUNTIME_SECONDS, 0
        )
        console.print(f"   Duration: {training_duration / 60:.1f} minutes")

        pod_startup = deployer_ctx.get("pod_startup_seconds")
        upload_dur = deployer_ctx.get(CTX_UPLOAD_DURATION)
        if pod_startup is not None:
            console.print(f"   Pod ready: {pod_startup:.0f}s")
        if upload_dur is not None:
            console.print(f"   Files upload: {upload_dur:.0f}s")

        training_info = monitor_ctx.get(CTX_TRAINING_INFO, {})
        if training_info.get("final_loss"):
            console.print(f"   Final Loss: {training_info['final_loss']:.4f}")
        if training_info.get("final_accuracy"):
            console.print(f"   Accuracy: {training_info['final_accuracy']:.2%}")

    def _print_gpu_provider_section(self, context: dict[str, Any], deployer_ctx: dict) -> None:
        console.print("\n[bold cyan]GPU Provider:[/bold cyan]")
        if not deployer_ctx:
            console.print("   [dim]Provider info not available[/dim]")
            return

        provider_name = deployer_ctx.get("provider_name", CTX_PROVIDER_NAME_UNKNOWN)
        provider_type = deployer_ctx.get("provider_type", CTX_PROVIDER_TYPE_UNKNOWN)
        gpu_type = deployer_ctx.get("gpu_type")
        gpu_count = deployer_ctx.get("gpu_count")

        provider_label = provider_name
        if gpu_type:
            gpu_label = f"{gpu_count}x {gpu_type}" if gpu_count and gpu_count > 1 else gpu_type
            provider_label = f"{provider_name} ({gpu_label})"
        console.print(f"   Provider: {provider_label} [{provider_type}]")

        if provider_type != "cloud":
            console.print("   Cost: $0 (local)")
            return

        cost_per_hr = deployer_ctx.get("cost_per_hr") or 0
        if cost_per_hr <= 0:
            console.print("   Rate: [dim]N/A[/dim]")
            return

        console.print(f"   Rate: ${cost_per_hr:.4f}/hr")
        training_sec = 0.0
        if StageNames.TRAINING_MONITOR in context:
            monitor_ctx = context[StageNames.TRAINING_MONITOR]
            training_sec = monitor_ctx.get(CTX_TRAINING_DURATION) or monitor_ctx.get(
                CTX_TRAINING_INFO, {}
            ).get(CTX_RUNTIME_SECONDS, 0)
        training_hours = training_sec / SECONDS_PER_HOUR
        total_cost = cost_per_hr * training_hours
        console.print(f"   Training time: {training_hours:.2f}h")
        console.print(f"   [bold yellow]Training cost: ${total_cost:.4f}[/bold yellow]")

    @staticmethod
    def _print_model_output_section(context: dict[str, Any]) -> None:
        console.print("\n[bold cyan]Model Output:[/bold cyan]")
        if StageNames.MODEL_RETRIEVER not in context:
            console.print("   Output Dir: output/ (hardcoded inside remote run workspace)")
            return
        retriever_ctx = context[StageNames.MODEL_RETRIEVER]
        console.print(f"   Local Path: {retriever_ctx.get('local_model_path', 'N/A')}")
        hf_repo = retriever_ctx.get("hf_repo_id")
        if hf_repo:
            console.print(f"   HuggingFace: {hf_repo}")
        else:
            console.print("   HuggingFace: [dim]Not uploaded[/dim]")
        model_size_mb = retriever_ctx.get("model_size_mb")
        if model_size_mb:
            console.print(f"   Size: {model_size_mb:.0f} MB")

    @staticmethod
    def _print_evaluation_section(context: dict[str, Any]) -> None:
        if StageNames.MODEL_EVALUATOR not in context:
            return
        eval_ctx = context[StageNames.MODEL_EVALUATOR]
        metrics = eval_ctx.get("metrics", {})
        if not metrics:
            return
        console.print("\n[bold cyan]Evaluation:[/bold cyan]")
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, float):
                console.print(f"   {metric_name}: {metric_value:.4f}")
            else:
                console.print(f"   {metric_name}: {metric_value}")

    @staticmethod
    def _print_launch_command_section(context: dict[str, Any]) -> None:
        if StageNames.INFERENCE_DEPLOYER not in context:
            return
        chat_script = context[StageNames.INFERENCE_DEPLOYER].get("inference_scripts", {}).get("chat")
        if chat_script:
            console.print("\n[bold cyan]Launch Command:[/bold cyan]")
            console.print(f"   [yellow]python {chat_script}[/yellow]")

    # ---- MLflow metric aggregation -----------------------------------------

    def aggregate_training_metrics(
        self,
        *,
        mlflow_manager: MLflowManager | None,
        collect_fn: Callable[[], list[dict[str, float]]] | None = None,
    ) -> None:
        """Aggregate training metrics from child/grandchild runs into the parent run.

        MLflow best practice: log aggregated/summary metrics to the parent run,
        keep detailed metrics in child runs.

        ``collect_fn`` lets callers (orchestrator) inject their own collector,
        which keeps the existing test seams that patch the orchestrator method.
        """
        if mlflow_manager is None:
            return
        if collect_fn is None:
            all_phase_metrics = self.collect_descendant_metrics(
                mlflow_manager=mlflow_manager, max_depth=2
            )
        else:
            all_phase_metrics = collect_fn()
        if not all_phase_metrics:
            logger.debug("[METRICS] No metrics found in descendant runs")
            return

        logger.info(f"[METRICS] Aggregating metrics from {len(all_phase_metrics)} phase runs...")

        aggregated_metrics: dict[str, float] = {}
        total_steps = 0
        total_runtime = 0.0
        final_loss: float | None = None

        # NB: explicit ``is not None`` — converged runs can have train_loss=0.0,
        # and total_steps=0 / runtime=0 are legitimate "no-op" phases worth
        # surfacing; the previous ``if train_loss := ...:`` dropped them.
        for phase_metrics in all_phase_metrics:
            train_loss = phase_metrics.get("train_loss")
            if train_loss is not None:
                final_loss = train_loss
            runtime = phase_metrics.get("train_runtime")
            if runtime is not None:
                total_runtime += runtime
            steps = phase_metrics.get("global_step")
            if steps is not None:
                total_steps += int(steps)

        if final_loss is not None:
            aggregated_metrics["final_train_loss"] = final_loss
        if total_steps > 0:
            aggregated_metrics["total_train_steps"] = float(total_steps)
        if total_runtime > 0:
            aggregated_metrics["total_train_runtime"] = total_runtime

        if aggregated_metrics:
            mlflow_manager.log_metrics(aggregated_metrics)
            logger.info(f"[METRICS] Aggregated {len(aggregated_metrics)} metrics to parent run")
            if final_loss:
                logger.info(f"[METRICS] Final train_loss: {final_loss:.4f}")

    @staticmethod
    def collect_descendant_metrics(
        *, mlflow_manager: MLflowManager | None, max_depth: int = 2
    ) -> list[dict[str, float]]:
        """Collect metrics from all descendant runs (BFS, ``phase_*`` children only)."""
        if mlflow_manager is None:
            return []

        try:
            client = mlflow_manager.client
            parent_run_id = _get_run_id(mlflow_manager)
            if not parent_run_id:
                return []

            run = client.get_run(parent_run_id)
            experiment_id = run.info.experiment_id

            phase_metrics: list[dict[str, float]] = []
            visited: set[str] = set()
            # deque.popleft is O(1); list.pop(0) is O(n). Matters for deep run trees.
            bfs_queue: deque[tuple[str, int]] = deque([(parent_run_id, 0)])

            while bfs_queue:
                current_id, depth = bfs_queue.popleft()
                if current_id in visited or depth > max_depth:
                    continue
                visited.add(current_id)

                children = client.search_runs(
                    experiment_ids=[experiment_id],
                    filter_string=f"tags.`mlflow.parentRunId` = '{current_id}'",
                )
                for child in children:
                    child_name = child.info.run_name or ""
                    if child_name.startswith("phase_"):
                        metrics = dict(child.data.metrics)
                        if metrics:
                            phase_metrics.append(metrics)
                            logger.debug(
                                f"[METRICS] Found phase run: {child_name} ({len(metrics)} metrics)"
                            )
                    if depth + 1 <= max_depth:
                        bfs_queue.append((child.info.run_id, depth + 1))
            return phase_metrics
        except Exception as e:
            logger.warning(f"[METRICS] Failed to collect descendant metrics: {e}")
            return []

    # ---- experiment report --------------------------------------------------

    @staticmethod
    def generate_experiment_report(
        *,
        run_id: str | None,
        mlflow_manager: MLflowManager | None,
    ) -> None:
        """Generate a full experiment Markdown report after pipeline completion."""
        if not run_id:
            logger.warning("[REPORT] Cannot generate report: no run_id provided")
            return
        try:
            # Use the public tracking_uri accessor instead of reaching into _gateway.
            tracking_uri = (mlflow_manager.tracking_uri or "") if mlflow_manager else ""
            local_logs_dir = get_run_log_dir()
            logger.info(f"[REPORT] Generating experiment report for run {run_id[:8]}...")
            generator = ExperimentReportGenerator(tracking_uri)
            report = generator.generate(run_id=run_id, local_logs_dir=local_logs_dir)
            logger.info(f"[REPORT] Report generated ({len(report)} chars)")
            logger.info(f"[REPORT] Saved to: {local_logs_dir / 'experiment_report.md'}")
        except Exception as e:
            logger.warning(f"[REPORT] Failed to generate report: {e}")


def _get_run_id(mlflow_manager: MLflowManager) -> str | None:
    run_id = getattr(mlflow_manager, "run_id", None)
    if isinstance(run_id, str) and run_id:
        return run_id
    legacy_run_id = getattr(mlflow_manager, "_run_id", None)
    if isinstance(legacy_run_id, str) and legacy_run_id:
        return legacy_run_id
    return None


__all__ = ["ExecutionSummaryReporter"]
