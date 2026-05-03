"""
Mock trainer classes for testing without actual training.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock


class MockTrainerState:
    """Mock for transformers.TrainerState."""

    def __init__(self):
        self.global_step = 0
        self.epoch = 0.0
        self.log_history: list[dict] = []
        self.best_metric: float | None = None
        self.best_model_checkpoint: str | None = None


class MockTrainOutput:
    """Mock for trainer.train() return value."""

    def __init__(
        self,
        training_loss: float = 1.5,
        global_step: int = 100,
        metrics: dict | None = None,
    ):
        self.training_loss = training_loss
        self.global_step = global_step
        self.metrics = metrics or {
            "train_loss": training_loss,
            "epoch": 1.0,
            "train_runtime": 60.0,
            "train_samples_per_second": 10.0,
        }


class MockSFTTrainer:
    """
    Mock for trl.SFTTrainer.

    Simulates training without computation.
    """

    def __init__(
        self,
        model: Any = None,
        args: Any = None,
        train_dataset: Any = None,
        eval_dataset: Any = None,
        tokenizer: Any = None,
        **kwargs,
    ):
        self.model = model or MagicMock()
        self.args = args or MagicMock()
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer

        # State
        self.state = MockTrainerState()
        self._is_trained = False
        self._saved_models: list[str] = []

        # Configuration
        self._training_loss = 1.5
        self._should_fail = False
        self._fail_at_step: int | None = None

    def train(self, resume_from_checkpoint: str | None = None) -> MockTrainOutput:
        """
        Mock training.

        Returns training output with configurable loss.
        """
        if self._should_fail:
            if self._fail_at_step:
                self.state.global_step = self._fail_at_step
            raise RuntimeError("Mock training failure")

        # Simulate training progress
        self.state.global_step = 100
        self.state.epoch = 1.0
        self.state.log_history = [
            {"loss": 2.5, "step": 10},
            {"loss": 2.0, "step": 50},
            {"loss": self._training_loss, "step": 100},
        ]

        self._is_trained = True

        return MockTrainOutput(
            training_loss=self._training_loss,
            global_step=self.state.global_step,
        )

    def save_model(self, output_dir: str | None = None) -> None:
        """Mock save model."""
        path = output_dir or self.args.output_dir
        self._saved_models.append(path)

    def save_state(self) -> None:
        """Mock save state."""
        pass

    def evaluate(self, eval_dataset: Any = None) -> dict:
        """Mock evaluation."""
        return {
            "eval_loss": self._training_loss * 1.1,
            "eval_runtime": 10.0,
        }

    def log(self, logs: dict) -> None:
        """Mock logging."""
        self.state.log_history.append(logs)

    def get_train_dataloader(self) -> MagicMock:
        """Mock dataloader."""
        return MagicMock()

    def get_eval_dataloader(self, eval_dataset: Any = None) -> MagicMock:
        """Mock eval dataloader."""
        return MagicMock()

    # Test helpers
    def set_training_loss(self, loss: float) -> None:
        """Set the training loss for testing."""
        self._training_loss = loss

    def set_should_fail(self, fail: bool, at_step: int | None = None) -> None:
        """Configure training to fail for testing error handling."""
        self._should_fail = fail
        self._fail_at_step = at_step


class MockDPOTrainer(MockSFTTrainer):
    """Mock for trl.DPOTrainer."""

    def __init__(
        self,
        model: Any = None,
        ref_model: Any = None,
        beta: float = 0.1,
        **kwargs,
    ):
        super().__init__(model=model, **kwargs)
        self.ref_model = ref_model
        self.beta = beta


class MockORPOTrainer(MockSFTTrainer):
    """Mock for trl.ORPOTrainer."""

    pass


__all__ = [
    "MockDPOTrainer",
    "MockORPOTrainer",
    "MockSFTTrainer",
    "MockTrainOutput",
    "MockTrainerState",
]
