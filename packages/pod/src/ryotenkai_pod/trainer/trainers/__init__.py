"""
TRL Trainers Factory

Factory Pattern for creating TRL trainers (SFTTrainer, DPOTrainer, ORPOTrainer).

This module bridges Training Strategies (WHAT to train) with actual TRL trainers
(HOW to execute the training loop).

Mapping:
    Strategy → Trainer
    ├── CPT  → SFTTrainer (language modeling)
    ├── SFT  → SFTTrainer (instruction tuning)
    ├── CoT  → SFTTrainer (reasoning)
    ├── DPO  → DPOTrainer (preference optimization)
    └── ORPO → ORPOTrainer (combined SFT+alignment)

Example:
    from src.training.trainers import TrainerFactory

    # Create trainer for SFT strategy
    trainer = TrainerFactory().create(
        strategy_type="sft",
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        config=config,
    )
    trainer.train()
"""

from src.training.trainers.factory import (
    TrainerFactory,
    TrainerType,
)

__all__ = [
    "TrainerFactory",
    "TrainerType",
]
