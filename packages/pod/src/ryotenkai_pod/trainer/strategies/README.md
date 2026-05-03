# Training Strategies

Training strategies in RyotenkAI are modular components that define **what** to train (objective) and **how** to format data. They work with `trainer_builder.py`, which selects the TRL trainer/config by strategy type.

## Available strategies

| Strategy | Description | TRL Trainer | TRL Config | Default LR |
|----------|-------------|---------------|------------|------------|
| **CPT** | Continual Pre-Training — domain adaptation | SFTTrainer | SFTConfig | 1e-5 |
| **SFT** | Supervised Fine-Tuning — instruction tuning | SFTTrainer | SFTConfig | 2e-4 |
| **CoT** | Chain-of-Thought — reasoning training | SFTTrainer | SFTConfig | 2e-5 |
| **DPO** | Direct Preference Optimization — alignment | DPOTrainer | DPOConfig | 5e-6 |
| **ORPO** | Odds Ratio Preference Optimization — SFT + alignment | ORPOTrainer | ORPOConfig | 1e-5 |
| **GRPO** | Group Relative Policy Optimization — reward-guided RL | GRPOTrainer | GRPOConfig | 2e-4 |
| **SAPO** | Soft Adaptive Policy Optimization — online RL (via GRPO) | GRPOTrainer | GRPOConfig | 2e-4 |

## Architecture

```
strategies/
├── base.py          # TrainingStrategy (ABC) + StrategyMetadata
├── base_rl.py       # BaseRLStrategy — shared logic for RL-based strategies (GRPO/SAPO)
├── factory.py       # StrategyFactory + register_builtin_strategies() (auto-runs on import)
├── cpt.py           # CPTStrategy
├── sft.py           # SFTStrategy
├── cot.py           # CoTStrategy
├── dpo.py           # DPOStrategy
├── orpo.py          # ORPOStrategy
├── grpo.py          # GRPOStrategy
└── sapo.py          # SAPOStrategy
```

### Execution flow

```
StrategyOrchestrator
    → StrategyFactory.create_from_phase(phase, config)   # create strategy
        → strategy.prepare_dataset(dataset, tokenizer)  # format dataset
    → TrainerFactory.create(...)
        → trainer_builder.create_trainer(strategy, ...)  # create TRL trainer
            → trainer_builder: STRATEGY_TRAINERS[type]   # SFTTrainer / DPOTrainer / GRPOTrainer
            → trainer_builder: STRATEGY_CONFIGS[type]    # SFTConfig / DPOConfig / GRPOConfig
            → strategy.build_config_kwargs(hp)           # map hyperparameters
            → strategy.build_trainer_kwargs(config)      # extra trainer kwargs
```

### Base class `TrainingStrategy`

Abstract methods (must implement):

```python
class TrainingStrategy(ABC):
    def prepare_dataset(self, dataset, tokenizer) -> Result[Dataset, StrategyError]: ...
    def validate_dataset(self, dataset) -> Result[bool, StrategyError]: ...
    def get_training_objective(self) -> str: ...          # "language_modeling" / "preference_optimization" / ...
    def get_metadata(self) -> StrategyMetadata: ...
    def get_trainer_type(self) -> str: ...                # "sft" / "dpo" / "orpo" / "grpo" / "sapo"
    def get_trainer_class(self) -> Any: ...               # SFTTrainer / DPOTrainer / ...
    def get_config_class(self) -> Any: ...                # SFTConfig / DPOConfig / ...

    # Optional (default implementations exist):
    def build_config_kwargs(self, hp, **kwargs) -> dict: ...   # map hyperparameters → TRL Config
    def build_trainer_kwargs(self, config, **kwargs) -> dict:... # extra Trainer kwargs
    def get_recommended_hyperparameters(self) -> dict: ...
```

### StrategyFactory

```python
factory = StrategyFactory()

# Create strategy by type
strategy = factory.create("sft", config)

# Create from phase config (used by StrategyOrchestrator)
strategy = factory.create_from_phase(phase, config)

# Default hyperparameters
defaults = factory.get_default_hyperparameters("dpo")
# → {"learning_rate": 5e-6, "epochs": 1, "batch_size": 4}

# List available strategies
factory.list_available()       # dict[str, str] → {name: description}
factory.list_with_metadata()   # dict[str, StrategyMetadata]
```

All strategies register automatically when `factory.py` is imported via `register_builtin_strategies()`.

## Usage examples

### Create a strategy

```python
from src.constants import STRATEGY_SFT
from src.training.strategies.factory import StrategyFactory

factory = StrategyFactory()
strategy = factory.create(STRATEGY_SFT, config)

result = strategy.prepare_dataset(dataset, tokenizer)
if result.is_ok():
    prepared = result.unwrap()
```

### Create trainer via TrainerFactory

```python
from src.training.trainers.factory import TrainerFactory

trainer_factory = TrainerFactory()
trainer = trainer_factory.create(
    strategy_type="dpo",
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_data,
    config=config,
    output_dir="output/phase_0_dpo",
)
trainer.train()
```

## Adding a new strategy

### Step 1: Constants

- `src/constants.py` — add `STRATEGY_MY = "my"` and include it in `ALL_STRATEGIES`
- `src/config/training/strategies/transitions.py` — add to `VALID_STRATEGY_TRANSITIONS`, and `VALID_START_STRATEGIES` if needed; the map is used for diagnostics and warning logs
- `src/constants.py` — add to `DEFAULT_LEARNING_RATES`, `DEFAULT_EPOCHS`, `DEFAULT_BATCH_SIZES`, `STRATEGY_DESCRIPTIONS`

### Step 2: Strategy class

Create `my_strategy.py`:

```python
from trl import SFTTrainer, SFTConfig
from src.training.strategies.base import TrainingStrategy, StrategyMetadata

class MyStrategy(TrainingStrategy):
    def get_trainer_class(self): return SFTTrainer
    def get_config_class(self): return SFTConfig
    def get_trainer_type(self) -> str: return "sft"

    def prepare_dataset(self, dataset, tokenizer): ...
    def validate_dataset(self, dataset): ...
    def get_training_objective(self) -> str: return "my_objective"
    def get_metadata(self) -> StrategyMetadata: ...

    def build_config_kwargs(self, hp, **kwargs) -> dict:
        # Map HyperparametersConfig → TRL Config kwargs
        return {"learning_rate": hp.learning_rate, "num_train_epochs": hp.epochs, ...}
```

### Step 3: Register in factory.py

Add to `register_builtin_strategies()`:

```python
from src.constants import STRATEGY_MY
from src.training.strategies.my_strategy import MyStrategy

StrategyFactory.register(
    STRATEGY_MY,
    MyStrategy,
    metadata=StrategyMetadata(
        name=STRATEGY_MY,
        version=STRATEGY_VERSION_DEFAULT,
        description="My strategy description",
        strategy_type=STRATEGY_MY,
        data_format="...",
        objective="...",
        recommended_use="...",
    ),
)
```

### Step 4: trainer_builder.py

Add to `STRATEGY_TRAINERS` and `STRATEGY_CONFIGS`:

```python
STRATEGY_TRAINERS = MappingProxyType({
    ...,
    STRATEGY_MY: SFTTrainer,  # or DPOTrainer / GRPOTrainer
})
STRATEGY_CONFIGS = MappingProxyType({
    ...,
    STRATEGY_MY: SFTConfig,
})
```

If the strategy needs unique TRL Config parameters (e.g. SAPO → `loss_type`, `num_generations`) — add logic in `create_training_args()`.

### Step 5 (optional): Unique hyperparameters

If the strategy needs extra config fields — add them in `src/utils/config.py` under `HyperparametersConfig` or `PhaseHyperparametersConfig`.

## Related files

| File | Role |
|------|------|
| `base.py` | `TrainingStrategy` ABC + `StrategyMetadata` |
| `factory.py` | `StrategyFactory` — registry and strategy creation |
| `src/constants.py` | `DEFAULT_LEARNING_RATES`, `DEFAULT_EPOCHS`, `DEFAULT_BATCH_SIZES`, `STRATEGY_DESCRIPTIONS` |
| `cpt/sft/cot/dpo/orpo/grpo/sapo.py` | Strategy implementations |
| `base_rl.py` | `BaseRLStrategy` — shared RL logic (GRPO/SAPO) |
| `src/constants.py` | Global `STRATEGY_*` constants and `ALL_STRATEGIES` |
| `src/config/training/strategies/transitions.py` | Recommended transitions between strategies; invalid ordering emits warnings |
| `../../trainer_builder.py` | `STRATEGY_TRAINERS`, `STRATEGY_CONFIGS`, `create_trainer()` |
| `../../trainers/factory.py` | `TrainerFactory` — uses strategies + trainer_builder |
| `../../orchestrator/strategy_orchestrator.py` | Training orchestrator — runs the strategy chain |

## Testing

```bash
pytest src/tests/unit/training/test_strategies.py -v
pytest src/tests/unit/training/test_strategies_preference_and_sapo.py -v
pytest src/tests/unit/training/test_strategy_factory_phase_hyperparams.py -v
```
