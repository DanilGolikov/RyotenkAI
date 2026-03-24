# Training Module

Multi-phase LLM training with strategy orchestration, MLflow tracking, and GPU memory management.

## Layout

```
src/training/
├── run_training.py          # Entry point (CLI + programmatic API)
├── trainer_builder.py       # Build TRL trainer / PEFT config
├── constants.py             # Module constants
│
├── orchestrator/            # Multi-phase training orchestration
│   ├── strategy_orchestrator.py  # Facade — main launch class
│   ├── chain_runner.py           # Iterate over phases
│   ├── phase_executor.py         # Run a single phase
│   ├── dataset_loader.py         # Per-phase dataset loading
│   ├── metrics_collector.py      # Collect metrics after a phase
│   ├── resume_manager.py         # Resume interrupted training
│   └── shutdown_handler.py       # Handle SIGINT / SIGTERM
│
├── strategies/              # Training strategies (CPT/SFT/CoT/DPO/ORPO/SAPO)
│   └── README.md            # Detailed strategy documentation
│
├── trainers/
│   └── factory.py           # TrainerFactory — build TRL trainer via strategy
│
├── managers/                # State management
│   ├── data_buffer.py       # DataBuffer — checkpoints, phase status, resume state
│   ├── data_loader.py       # Dataset loader
│   ├── mlflow_manager.py    # Facade for MLflow (experiment tracking)
│   ├── model_saver.py       # Save final model
│   └── constants.py         # Buffer/checkpoint constants
│
├── mlflow/                  # MLflow subcomponents
│   ├── autolog.py           # MLflowAutologManager
│   ├── dataset_logger.py    # MLflowDatasetLogger
│   ├── domain_logger.py     # MLflowDomainLogger (log_stage_start, etc.)
│   ├── event_log.py         # MLflowEventLog (in-memory event journal)
│   ├── model_registry.py    # MLflowModelRegistry (model registration)
│   ├── primitives.py        # Base primitives
│   └── run_analytics.py     # MLflowRunAnalytics (search/compare runs)
│
├── models/
│   └── loader.py            # Load model and tokenizer
│
├── callbacks/               # HuggingFace Trainer callbacks
│   ├── gpu_metrics_callback.py     # GPU utilization / memory
│   ├── system_metrics_callback.py  # CPU/RAM metrics
│   └── training_events_callback.py # Log training events
│
├── reward_plugins/          # Reward model plugins (GRPO/SAPO)
│   ├── base.py              # RewardPlugin base class
│   ├── registry.py          # Plugin registry
│   ├── factory.py           # Plugin factory
│   ├── discovery.py         # Auto-discovery
│   └── plugins/             # Built-in reward plugins
│
├── templates/
│   └── experiment_description.md   # MLflow experiment description template
│
└── notifiers/               # Completion notifications
    ├── log.py               # Log notification
    └── marker_file.py       # Marker file for RunPod (completion signal)
```

## Entry point

```bash
# Multi-phase training
python -m src.training.run_training --config src/config/pipeline_config.yaml

# With debug logs
LOG_LEVEL=DEBUG python -m src.training.run_training --config src/config/pipeline_config.yaml

# Resume interrupted training
python -m src.training.run_training --config src/config/pipeline_config.yaml --resume --run-id run_xxx
```

Programmatic launch via pipeline: `src/main.py` (`train-local` command) calls `src.training.run_training`.

## Strategies

| Strategy | TRL Trainer | Data format |
|----------|-------------|-------------|
| `cpt` | SFTTrainer | `{"text": "..."}` |
| `sft` | SFTTrainer | `{"messages": [...]}` or `{"instruction", "input", "output"}` |
| `cot` | SFTTrainer | `{"messages": [...]}` with reasoning |
| `dpo` | DPOTrainer | `{"prompt", "chosen", "rejected"}` |
| `orpo` | ORPOTrainer | `{"prompt", "chosen", "rejected"}` |
| `grpo` | GRPOTrainer | `{"prompt", "completion"}` with reward plugins |
| `sapo` | GRPOTrainer | `{"prompt", "completion"}` |

Details: [strategies/README.md](./strategies/README.md)

## Configuration

```yaml
training:
  # PEFT (applied globally to all phases)
  lora:
    r: 16
    lora_alpha: 32
    lora_dropout: 0.05
    target_modules: "all-linear"  # auto-detect all linear layers

  # Global hyperparameters (override per-phase)
  hyperparams:
    per_device_train_batch_size: 4
    gradient_accumulation_steps: 4

  # Phase chain
  strategies:
    - strategy_type: cpt
      dataset: cpt_dataset
      hyperparams:
        epochs: 1
        learning_rate: 1e-5

    - strategy_type: sft
      dataset: sft_dataset
      hyperparams:
        epochs: 3
        learning_rate: 2e-4

    - strategy_type: dpo
      dataset: dpo_dataset
      hyperparams:
        epochs: 1
        learning_rate: 5e-6
```

## Key components

### StrategyOrchestrator

Facade for running the phase chain. Takes model/tokenizer/config and coordinates subcomponents.

```python
from src.training.orchestrator.strategy_orchestrator import StrategyOrchestrator

orchestrator = StrategyOrchestrator(model, tokenizer, config)
result = orchestrator.run_chain()
if result.is_success():
    final_model = result.unwrap()
```

Supports dependency injection for testing:

```python
orchestrator = StrategyOrchestrator(
    model, tokenizer, config,
    strategy_factory=mock_strategy_factory,
    trainer_factory=mock_trainer_factory,
)
```

### trainer_builder.py

Builds the TRL trainer directly without extra adapter layers. TRL supports LoRA natively via `peft_config`.

```python
from src.training.trainer_builder import create_peft_config, create_trainer

peft_config = create_peft_config(config)
trainer = create_trainer(config, strategy_phase, model, tokenizer, train_dataset, peft_config)
trainer.train()
```

### DataBuffer

Manages phase state across runs. Stores each phase status (`PENDING / RUNNING / COMPLETED / FAILED`). Used by `ResumeManager` to resume from the right point.

### MLflowManager

Facade for MLflow operations. Structure: parent run (whole pipeline) → nested runs (each phase).

```python
manager = MLflowManager(config)
manager.setup()

with manager.start_run(run_name="training_sft"):
    manager.log_params({"lr": 2e-4, "epochs": 3})
    manager.log_stage_start("SFT", stage_idx=1, total_stages=3)
    # ... training ...
    manager.log_metrics({"train/loss": 0.42})
```

## PEFT / LoRA

LoRA is configured globally and applied by TRL to all phases:

```yaml
training:
  lora:
    r: 16              # Rank (4–64, higher = more parameters)
    lora_alpha: 32     # Scaling factor (typically 2*r)
    lora_dropout: 0.05
    target_modules: "all-linear"  # or ["q_proj", "v_proj", ...]
    use_4bit: true     # QLoRA
```

For QLoRA additionally:

```yaml
training:
  lora:
    use_4bit: true
    bnb_4bit_compute_dtype: "bfloat16"
    bnb_4bit_quant_type: "nf4"
```

## Related files

| File | Role |
|------|------|
| `src/main.py` | CLI entry point (`train-local` command) |
| `src/utils/config.py` | `PipelineConfig`, `StrategyPhaseConfig`, `GlobalHyperparametersConfig`, `PhaseHyperparametersConfig` |
| `src/utils/container.py` | `TrainingContainer` — DI container |
| `src/utils/memory_manager.py` | `MemoryManager` — OOM protection, auto batch_size |
| `src/config/training/` | Pydantic schemas for training config |
| `src/data/` | Dataset loading and validation |
