"""
Trainer Factory for creating TRL Trainers.

Factory Pattern for TRL trainers based on training strategy.
Handles the complexity of different trainer types and their configurations.

Key Responsibilities:
1. Strategy-to-Trainer mapping (SFT→SFTTrainer, DPO→DPOTrainer)
2. Config creation (SFTConfig, DPOConfig, ORPOConfig)
3. Reference model handling for DPO (via PEFT adapters)
4. Hyperparameter merging from strategy defaults

Architecture:
    StrategyFactory  → TrainingStrategy (data prep)
    TrainerFactory   → TRL Trainer (training loop)  ← THIS FILE
    TrainingFactory  → TrainingAdapter (model setup: QLoRA, LoRA)

Example:
    from ryotenkai_pod.trainer.trainers import TrainerFactory

    # Create SFT trainer
    trainer = TrainerFactory().create(
        strategy_type="sft",
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_data,
        eval_dataset=eval_data,
        config=config,
    )
    trainer.train()
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, TypeAlias

from ryotenkai_pod.trainer.constants import DEFAULT_EVAL_SAVE_STEPS
from ryotenkai_pod.trainer.reward_plugins import build_reward_plugin_result
from ryotenkai_pod.trainer.strategies.factory import StrategyFactory
from ryotenkai_shared.utils.logger import logger

if TYPE_CHECKING:
    from datasets import Dataset
    from transformers import PreTrainedModel, PreTrainedTokenizer

    from ryotenkai_shared.config import (
        GlobalHyperparametersConfig,
        PhaseHyperparametersConfig,
        PipelineConfig,
        StrategyPhaseConfig,
    )

# =============================================================================
# TYPE ALIASES
# =============================================================================

# Generic Trainer type (we don't import all specific ones to avoid coupling)
TrainerType: TypeAlias = Any  # Was Union[SFTTrainer, ...]

# Generic Config type
ConfigType: TypeAlias = Any


# =============================================================================
# TRAINER FACTORY
# =============================================================================


class TrainerFactory:
    """
    Factory for creating TRL Trainer instances.

    Generic implementation that delegates logic to TrainingStrategy.
    """

    def __init__(self) -> None:
        """Initialize TrainerFactory."""
        logger.debug("[TF:INIT] TrainerFactory initialized (Generic)")
        self._reward_plugin: Any | None = None

    @property
    def reward_plugin(self) -> Any | None:
        """Last reward plugin created during :meth:`create`. Available for teardown."""
        return self._reward_plugin

    def create(
        self,
        strategy_type: str,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        train_dataset: Dataset,
        config: PipelineConfig,
        *,
        output_dir: str,
        eval_dataset: Dataset | None = None,
        phase_config: StrategyPhaseConfig | None = None,
        ref_model: PreTrainedModel | None = None,
        **kwargs: Any,
    ) -> TrainerType:
        """
        Create a TRL trainer for the given strategy.

        Delegates creation logic to the specific TrainingStrategy implementation.
        """
        strategy_type = strategy_type.lower()
        logger.info(f"Creating trainer for {strategy_type} strategy")

        # 1. Get Strategy Instance
        strategy_factory = StrategyFactory()
        if not strategy_factory.is_registered(strategy_type):
            raise ValueError(f"Unknown strategy type: '{strategy_type}'")

        strategy = strategy_factory.create(strategy_type, config)

        # 2. Get Trainer and Config classes
        trainer_class = strategy.get_trainer_class()
        config_class = strategy.get_config_class()

        # 3. Create Training Config
        # Merge hyperparameters (Global + Phase Override)
        hp = self._merge_hyperparams(config.training.hyperparams, phase_config.hyperparams if phase_config else None)

        # Determine defaults
        learning_rate = hp.learning_rate
        num_epochs = hp.epochs

        # Calculate report_to. The legacy wide manager used to suppress
        # the ``"mlflow"`` reporter when the manager wasn't active; the
        # trainer now joins the parent MLflow run via env vars
        # (``MLFLOW_RUN_ID``) so we always keep whatever the user
        # configured. HF's MLflowCallback no-ops when the env var
        # is absent, so this is safe.
        report_to = config.integrations.get_report_to()

        # Base config kwargs
        config_kwargs = {
            "output_dir": output_dir,
            "num_train_epochs": num_epochs,
            "per_device_train_batch_size": hp.per_device_train_batch_size,
            "gradient_accumulation_steps": hp.gradient_accumulation_steps,
            "learning_rate": learning_rate,
            "lr_scheduler_type": hp.lr_scheduler_type,
            "warmup_ratio": hp.warmup_ratio,
            "bf16": hp.bf16,
            "fp16": hp.fp16,
            "gradient_checkpointing": hp.gradient_checkpointing,
            "logging_steps": hp.logging_steps,
            "save_steps": hp.save_steps,
            "weight_decay": hp.weight_decay,
            "optim": hp.optim,
            "report_to": report_to,
        }
        # Filter None
        config_kwargs = {k: v for k, v in config_kwargs.items() if v is not None}

        # Delegate strategy-specific config params
        strategy_config_kwargs = strategy.build_config_kwargs(hp)
        config_kwargs.update(strategy_config_kwargs)

        # Resolve reward plugin config kwargs before instantiating the TRL config.
        # reward_weights (and any future config-level plugin params) must reach the
        # config constructor directly — not via setattr after the fact.
        reward_result = None
        self._reward_plugin = None
        if strategy.requires_reward_plugin:
            if phase_config is None:
                raise ValueError(
                    f"{strategy_type.upper()} strategy requires explicit phase_config for reward plugin resolution"
                )
            reward_result = build_reward_plugin_result(
                train_dataset=train_dataset,
                phase_config=phase_config,
                pipeline_config=config,
            )
            self._reward_plugin = reward_result.plugin
            config_kwargs.update(reward_result.config_kwargs)

        # Enable evaluation only when eval_dataset is provided
        if eval_dataset is not None:
            # Default eval cadence must be compatible with save cadence when load_best_model_at_end=True.
            # If user overrides save_steps but not eval_steps, using a hard default (500) can crash:
            #   "--load_best_model_at_end requires the saving steps to be a round multiple of the evaluation steps"
            # So we default eval_steps to save_steps (or 500 if both are unset).
            eval_steps = (
                hp.eval_steps
                if hp.eval_steps is not None
                else (hp.save_steps if hp.save_steps is not None else DEFAULT_EVAL_SAVE_STEPS)
            )
            save_steps = hp.save_steps if hp.save_steps is not None else DEFAULT_EVAL_SAVE_STEPS

            # Ensure compatibility: save_steps must be a multiple of eval_steps.
            # If user provided an incompatible combination, auto-fix by rounding save_steps up.
            if eval_steps > 0 and save_steps % eval_steps != 0:
                adjusted_save_steps = ((save_steps // eval_steps) + 1) * eval_steps
                logger.warning(
                    "[TF:EVAL] save_steps=%s is not a multiple of eval_steps=%s while load_best_model_at_end=True. "
                    "Adjusting save_steps -> %s.",
                    save_steps,
                    eval_steps,
                    adjusted_save_steps,
                )
                save_steps = adjusted_save_steps
                config_kwargs["save_steps"] = save_steps
            # transformers/trl API compatibility:
            # - Newer versions renamed `evaluation_strategy` -> `eval_strategy`
            # - Some config classes accept **kwargs (tests/mocks)
            sig = inspect.signature(config_class.__init__)
            params = sig.parameters
            has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())

            if "eval_strategy" in params or ("evaluation_strategy" not in params and has_varkw):
                config_kwargs["eval_strategy"] = "steps"
            elif "evaluation_strategy" in params:
                config_kwargs["evaluation_strategy"] = "steps"
            else:
                # If neither is supported, skip setting strategy to avoid crashing.
                logger.warning(
                    f"[TF:EVAL] Config class {getattr(config_class, '__name__', str(config_class))} "
                    "does not support eval_strategy/evaluation_strategy; skipping evaluation strategy param."
                )

            config_kwargs.update(
                {
                    "eval_steps": eval_steps,
                    "load_best_model_at_end": True,
                    "metric_for_best_model": "eval_loss",
                    "greater_is_better": False,
                }
            )

        # Instantiate Config
        training_config = config_class(**config_kwargs)

        # Phase M5 — Pattern A HF MLflow wiring. When the trainer is
        # launched by the control plane the ``MLFLOW_RUN_ID`` env var
        # is set; in that case force ``report_to=["mlflow"]`` and tag
        # system metrics with the local rank via
        # :meth:`mlflow.set_system_metrics_node_id`. The wiring is a
        # no-op for standalone trainer runs (local dev with no parent
        # run) -- ``configure_training_args`` only flips ``report_to``
        # and the env-detection in HF's MLflowCallback short-circuits
        # without MLFLOW_RUN_ID anyway. See plan §"Phase M4 follow-up:
        # configure_training_args injection".
        import os as _os

        if _os.environ.get("MLFLOW_RUN_ID", "").strip():
            from ryotenkai_pod.trainer.mlflow.hf_wiring import HFMlflowWiring

            local_rank_raw = _os.environ.get("LOCAL_RANK")
            local_rank: int | None
            try:
                local_rank = int(local_rank_raw) if local_rank_raw is not None else None
            except ValueError:
                local_rank = None
            HFMlflowWiring.configure_training_args(
                training_config, local_rank=local_rank,
            )
            logger.info(
                "[TF:HF_WIRING] configure_training_args applied (local_rank=%s)",
                local_rank,
            )

        logger.debug(f"[TF:CONFIG_CREATED] config={config_class.__name__}, lr={learning_rate}, epochs={num_epochs}")

        # Convert string prompts to conversational format so TRL applies chat template.
        # Polymorphic dispatch — each strategy decides whether (and how) to convert.
        # SFT/DPO/ORPO inherit the no-op default; GRPO/SAPO override via BaseRLStrategy.
        train_dataset, eval_dataset = strategy.prepare_prompts_for_chat_template(
            train_dataset, eval_dataset, tokenizer,
        )

        # 4. Build Trainer Kwargs
        trainer_kwargs = {
            "model": model,
            "args": training_config,
            "train_dataset": train_dataset,
            "processing_class": tokenizer,
        }

        # Add PEFT config (Common logic)
        # All shipped adapter kinds (lora/qlora/adalora) need a PEFT config —
        # the discriminated union restricts ``adapter.kind`` to that set, so
        # we can call ``create_peft_config`` unconditionally. If/when a
        # non-PEFT adapter kind is added, ``create_peft_config`` itself
        # MUST raise — the dispatcher stays here generic.
        if hasattr(config, "training") and hasattr(config.training, "adapter"):
            from ryotenkai_pod.trainer.trainer_builder import create_peft_config

            # PEFT double-apply guard. When a model is loaded from a HF checkpoint
            # that was previously published as a PEFT adapter (e.g. SFT checkpoints
            # uploaded with ``adapter_config.json``), the loader returns a
            # ``PeftModel`` already carrying ``peft_config``. Passing another
            # ``peft_config`` to TRL causes ``get_peft_model`` to be called on
            # an already-wrapped model, which corrupts the ``requires_grad``
            # mask and yields ``Trainable parameters: 0 (0.00%)`` — silent no-op.
            #
            # We check ``isinstance(model, PeftModel)`` specifically, NOT
            # ``hasattr(model, "base_model")`` — plain ``PreTrainedModel``
            # instances expose a ``base_model`` property too (e.g. LlamaModel
            # inside ``LlamaForCausalLM``), so an attribute-based check would
            # false-positive on every fresh QLoRA run and silently disable
            # adapter injection.
            try:
                from peft import PeftModel
                model_already_peft = isinstance(model, PeftModel)
            except ImportError:
                # peft not installed → definitely not a PeftModel
                model_already_peft = False

            if model_already_peft:
                logger.warning(
                    "[TF:PEFT_GUARD] Model is already a PeftModel (loaded from "
                    "a checkpoint with adapter_config.json). Skipping "
                    "peft_config in trainer_kwargs to avoid double-apply "
                    "(which would zero out trainable params). If this is a "
                    "fresh run, either merge the adapter at publish time or "
                    "reload the checkpoint as a base model.",
                )
            else:
                peft_config = create_peft_config(config)
                if peft_config is not None:
                    trainer_kwargs["peft_config"] = peft_config

        if eval_dataset is not None:
            trainer_kwargs["eval_dataset"] = eval_dataset

        # Delegate strategy-specific trainer args (e.g. ref_model, reward_funcs)
        # Pass model and ref_model in kwargs context
        strategy_kwargs: dict[str, Any] = {
            "model": model,
            "ref_model": ref_model,
        }
        if strategy.requires_reference_dataset:
            strategy_kwargs["train_dataset"] = train_dataset

        # Apply any post-build config mutations (e.g. DPO PEFT adapter names)
        strategy.post_build_config_hook(training_config, model=model, ref_model=ref_model)

        strategy_trainer_kwargs = strategy.build_trainer_kwargs(
            training_config,
            **strategy_kwargs,
        )
        if reward_result is not None:
            strategy_trainer_kwargs.update(reward_result.trainer_kwargs)
        trainer_kwargs.update(strategy_trainer_kwargs)

        # 5. Add Callbacks (Common Logic)
        mlflow_config = config.integrations.mlflow if config.integrations else None
        callbacks = trainer_kwargs.get("callbacks", []) or []

        if mlflow_config:
            # Phase 2 (ethereal-tumbling-patterson): ``TrainingEventsCallback``
            # was the dual-path MLflow-event sibling to
            # :class:`RunnerEventCallback`. With the unified event system
            # the SSOT lives on the runner bus + journal; the MLflow
            # attribution layer no longer needs its own HF callback.
            # The MLflow ``training_events.json`` artifact will be
            # populated by ``MlflowFinalizer`` (Phase 6) from the journal
            # instead.

            # ``SystemMetricsCallback`` is the single source of truth for
            # system metrics (CPU / GPU / RAM). It logs step-aligned
            # ``gpu/{idx}/*``, ``cpu/*``, ``ram/*`` via ``mlflow.log_metrics``,
            # which is monkey-patched by ``ResilientMLflowTransport`` so
            # payloads survive offline windows through ``MetricsBuffer``.
            #
            # ``GPUMetricsCallback`` was removed in this refactor — it was a
            # near-duplicate that wrote to a parallel ``system/gpu_0_*``
            # namespace via ``nvidia-smi`` subprocess and didn't go through
            # ``ResilientMLflowTransport`` consistently.
            sm_block = getattr(mlflow_config, "system_metrics", None)
            sm_callback_on = bool(getattr(sm_block, "callback_enabled", False))
            if sm_callback_on:
                from ryotenkai_pod.trainer.callbacks import SystemMetricsCallback

                callbacks.append(SystemMetricsCallback())

        # Phase 3.2 — runner-side event push.
        #
        # When the trainer subprocess runs inside the in-pod runner, the
        # supervisor sets ``RYOTENKAI_RUNNER_URL=http://127.0.0.1:8080``
        # in the spawn env (see TrainingLauncher._build_job_env). Picking
        # the env up here threads a :class:`RunnerEventCallback` into the
        # trainer's callback list so HF lifecycle hooks (on_train_begin /
        # on_step_end / on_log / on_evaluate / on_save / on_train_end)
        # fan out as structured ``training_started`` / ``step`` / ``log``
        # / ``eval_metrics`` / ``checkpoint_saved`` / ``training_complete``
        # events on the runner's bus → WebSocket → Mac.
        #
        # The callback no-ops when the env is unset, so local-mode runs
        # (no runner attached) and unit tests stay unaffected. Activation
        # is env-driven rather than config-driven so the same trainer
        # binary works in both contexts without a YAML toggle.
        import os as _os

        from ryotenkai_pod.trainer.callbacks.runner_event_callback import (
            RUNNER_URL_ENV,
            RunnerEventCallback,
        )

        if _os.environ.get(RUNNER_URL_ENV):
            runner_event_callback = RunnerEventCallback()
            callbacks.append(runner_event_callback)

            # Terminal-state finalization (cooperative cancellation +
            # natural completion).
            #
            # The unified ``TerminalCallback`` polls the global
            # :class:`ShutdownHandler` flag on each ``on_step_end`` (when
            # reason="cancel") and turns it into HF's own
            # ``TrainerControl(should_save, should_training_stop)`` so the
            # trainer saves + exits at the next checkpoint boundary
            # instead of being SIGKILLed mid-step. Same activation gate
            # as ``RunnerEventCallback`` -- when the supervisor sets
            # ``RYOTENKAI_RUNNER_URL`` we know we're inside the in-pod
            # runner and stop signals are meaningful.
            #
            # **Inserted at index 0** -- BEFORE HF Trainer's
            # auto-registered MLflow callback. HF MLflow callback owns
            # the final ``end_run()`` on ``on_train_end``; our callback
            # only flips control flags. Order guarantees that on the
            # cancelled step, HF observes ``should_save+should_training_stop``,
            # checkpoints, then HF MLflow callback runs and closes the
            # run with the correct status.
            from ryotenkai_pod.trainer.callbacks.terminal_callback import (
                TerminalCallback,
            )

            # Cooperative cancellation + natural completion are now
            # served by a single :class:`TerminalCallback` parametric on
            # ``reason`` (post-cleanup merge of the historical
            # CancellationCallback + CompletionCallback). Both share
            # the same ``mlflow_manager`` reference (no-op when ``None``)
            # and the same event publisher channel -- the
            # ``cancellation_finalized`` / ``completion_finalized``
            # telemetry kinds are selected internally based on
            # ``reason``.
            #
            # We wrap with ``flush_now=True`` so terminal events land
            # immediately rather than buffering -- by the time
            # ``on_train_end`` fires, the trainer is about to exit and
            # the buffer has no further flush opportunities. Operator
            # visibility wins over micro-batching here.
            def _terminal_event_publisher(
                kind: str, payload: dict[str, Any],
            ) -> None:
                runner_event_callback._publish(kind, payload, flush_now=True)

            callbacks.insert(
                0,
                TerminalCallback(
                    reason="cancel",
                    event_publisher=_terminal_event_publisher,
                ),
            )
            callbacks.insert(
                1,
                TerminalCallback(
                    reason="complete",
                    event_publisher=_terminal_event_publisher,
                ),
            )

        if callbacks:
            trainer_kwargs["callbacks"] = callbacks

        # Merge extra kwargs
        trainer_kwargs.update(kwargs)

        # 6. Instantiate Trainer
        trainer = trainer_class(**trainer_kwargs)
        logger.info(f"✅ {trainer_class.__name__} created successfully")

        return trainer

    def create_from_phase(
        self,
        phase: StrategyPhaseConfig,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        train_dataset: Dataset,
        config: PipelineConfig,
        *,
        output_dir: str,
        **kwargs: Any,
    ) -> TrainerType:
        """Create trainer from a strategy phase configuration.

        :param phase: phase config describing the strategy.
        :param model: pre-trained model to fine-tune.
        :param tokenizer: tokenizer paired with ``model``.
        :param train_dataset: training split.
        :param config: pipeline-level config.
        :param output_dir: directory the trainer should write to.
        :returns: TRL trainer instance.
        """
        return self.create(
            strategy_type=phase.strategy_type,
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            config=config,
            output_dir=output_dir,
            phase_config=phase,
            **kwargs,
        )

    @staticmethod
    def _merge_hyperparams(
        global_params: GlobalHyperparametersConfig, phase_params: PhaseHyperparametersConfig | None
    ) -> GlobalHyperparametersConfig:
        """Merge global hyperparameters with phase-specific overrides."""
        merged = global_params.model_copy()
        if phase_params:
            overrides = phase_params.model_dump(exclude_unset=True, exclude_none=True)
            for k, v in overrides.items():
                setattr(merged, k, v)
        return merged

    def get_trainer_class(self, strategy_type: str) -> type[TrainerType]:
        """Get the trainer class for a strategy type (via StrategyFactory)."""
        # Create dummy config just to get strategy instance
        # This is a bit awkward but Strategy requires config init.
        # Alternatively we could make get_trainer_class static in strategy but that breaks polymorphism if it depends on config?
        # Ideally get_trainer_class should be class method or we instantiate strategy.
        # For now, let's just error or Mock config?
        # Actually this method is rarely used outside tests or info.
        # Let's try to get it from registry directly if possible, or instantiate.
        # But Strategy init requires config.
        # Assuming we can't easily get it without config.
        # Let's raise NotImplementedError or try best effort.
        raise NotImplementedError("get_trainer_class is deprecated. Use StrategyFactory to get strategy instance.")

    def get_config_class(self, strategy_type: str) -> type[ConfigType]:
        raise NotImplementedError("get_config_class is deprecated.")

    @staticmethod
    def list_supported_strategies() -> list[str]:
        """List all supported strategy types."""
        return list(StrategyFactory().list_available().keys())

    @staticmethod
    def get_trainer_info(strategy_type: str) -> dict[str, str | bool]:
        """Get information about trainer for a strategy."""
        # This needs config to instantiate strategy.
        # Maybe we can just list from metadata?
        strategy_factory = StrategyFactory()
        if not strategy_factory.is_registered(strategy_type):
            raise ValueError(f"Unknown strategy: {strategy_type}")

        # We can't get class without instance easily if we strictly follow the new pattern.
        # But for info, we can look at metadata.
        return {"strategy_type": strategy_type, "info": "See StrategyMetadata"}


__all__ = ["TrainerFactory", "TrainerType"]
