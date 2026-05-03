"""
Mock model classes for testing without loading real model weights.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

if TYPE_CHECKING:
    from collections.abc import Iterator


class MockParameter:
    """Mock for torch.nn.Parameter."""

    def __init__(self, size: int = 1000, requires_grad: bool = True):
        self._size = size
        self.requires_grad = requires_grad
        self.data = MagicMock()

    def numel(self) -> int:
        return self._size


class MockPreTrainedModel:
    """
    Mock for transformers.PreTrainedModel.

    Provides essential interface without loading weights.
    """

    def __init__(
        self,
        model_type: str = "qwen2",
        hidden_size: int = 768,
        num_layers: int = 12,
        vocab_size: int = 32000,
    ):
        self.config = MagicMock()
        self.config.model_type = model_type
        self.config.hidden_size = hidden_size
        self.config.num_hidden_layers = num_layers
        self.config.vocab_size = vocab_size

        # Trainable parameters
        self._parameters = [MockParameter(1000, True) for _ in range(10)]

        # Device
        self.device = "cpu"
        self._is_peft = False

    def parameters(self) -> Iterator[MockParameter]:
        return iter(self._parameters)

    def named_parameters(self) -> Iterator[tuple[str, MockParameter]]:
        for i, p in enumerate(self._parameters):
            yield f"layer.{i}.weight", p

    def print_trainable_parameters(self) -> None:
        """Print trainable params summary."""
        trainable = sum(p.numel() for p in self._parameters if p.requires_grad)
        total = sum(p.numel() for p in self._parameters)
        print(f"Trainable: {trainable:,} / Total: {total:,} ({100 * trainable / total:.2f}%)")

    def save_pretrained(self, path: str, **kwargs) -> None:
        """Mock save (does nothing)."""
        pass

    def to(self, device: str) -> MockPreTrainedModel:
        """Mock device transfer."""
        self.device = device
        return self

    def train(self, mode: bool = True) -> MockPreTrainedModel:
        """Set training mode."""
        return self

    def eval(self) -> MockPreTrainedModel:
        """Set evaluation mode."""
        return self

    def forward(self, *args, **kwargs) -> MagicMock:
        """Mock forward pass."""
        output = MagicMock()
        output.loss = MagicMock()
        output.loss.item.return_value = 1.5
        return output

    def __call__(self, *args, **kwargs) -> MagicMock:
        """Make callable."""
        return self.forward(*args, **kwargs)


class MockTokenizer:
    """
    Mock for transformers.PreTrainedTokenizer.

    Provides essential tokenizer interface.
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        model_max_length: int = 2048,
    ):
        self.vocab_size = vocab_size
        self.model_max_length = model_max_length

        # Special tokens
        self.pad_token = "<pad>"
        self.pad_token_id = 0
        self.eos_token = "</s>"
        self.eos_token_id = 2
        self.bos_token = "<s>"
        self.bos_token_id = 1
        self.unk_token = "<unk>"
        self.unk_token_id = 3

        # Chat template markers
        self.im_start = "<|im_start|>"
        self.im_end = "<|im_end|>"

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ) -> str | list[int]:
        """Apply chat template to messages."""
        result = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            result += f"{self.im_start}{role}\n{content}{self.im_end}\n"

        if add_generation_prompt:
            result += f"{self.im_start}assistant\n"

        if tokenize:
            # Return mock token IDs
            return list(range(len(result.split())))

        return result

    def __call__(
        self,
        text: str | list[str],
        padding: bool = True,
        truncation: bool = True,
        max_length: int | None = None,
        return_tensors: str | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Tokenize text."""
        if isinstance(text, str):
            text = [text]

        max_len = max_length or self.model_max_length

        # Mock tokenization
        input_ids = []
        attention_mask = []

        for t in text:
            tokens = list(range(min(len(t.split()), max_len)))
            mask = [1] * len(tokens)

            if padding:
                pad_len = max_len - len(tokens)
                tokens = tokens + [self.pad_token_id] * pad_len
                mask = mask + [0] * pad_len

            input_ids.append(tokens)
            attention_mask.append(mask)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    def save_pretrained(self, path: str, **kwargs) -> None:
        """Mock save (does nothing)."""
        pass

    def decode(self, token_ids: list[int], **kwargs) -> str:
        """Mock decode."""
        return f"<decoded text with {len(token_ids)} tokens>"

    def encode(self, text: str, **kwargs) -> list[int]:
        """Mock encode."""
        return list(range(len(text.split())))


__all__ = ["MockParameter", "MockPreTrainedModel", "MockTokenizer"]
