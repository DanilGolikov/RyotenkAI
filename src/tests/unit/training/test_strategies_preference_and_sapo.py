from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock

from src.training.strategies.dpo import DPOStrategy
from src.training.strategies.orpo import ORPOStrategy
from src.training.strategies.sapo import SAPOStrategy


@dataclass
class _PairDataset:
    column_names: list[str]
    sample: dict

    def __len__(self) -> int:
        """Dataset length."""
        return 1

    def __getitem__(self, idx: int):
        assert idx == 0
        return self.sample


def test_dpo_validate_dataset_missing_rejected() -> None:
    s = DPOStrategy(MagicMock())
    ds = _PairDataset(column_names=["chosen"], sample={})
    assert s.validate_dataset(ds).is_failure()


def test_dpo_validate_dataset_valid_columns() -> None:
    s = DPOStrategy(MagicMock())
    ds = _PairDataset(column_names=["chosen", "rejected"], sample={})
    assert s.validate_dataset(ds).is_success()


def test_dpo_build_trainer_kwargs_uses_ref_model_when_provided() -> None:
    s = DPOStrategy(MagicMock())
    cfg = SimpleNamespace()
    ref = object()
    kwargs = s.build_trainer_kwargs(cfg, model=object(), ref_model=ref)
    assert kwargs["ref_model"] is ref


def test_dpo_post_build_config_hook_is_noop() -> None:
    """post_build_config_hook is a no-op — TRL handles PeftModel reference natively."""
    cfg = SimpleNamespace()
    model = object()

    s = DPOStrategy(MagicMock())
    s.post_build_config_hook(cfg, model=model, ref_model=None)

    assert not hasattr(cfg, "model_adapter_name")
    assert not hasattr(cfg, "ref_adapter_name")


def test_orpo_validate_dataset() -> None:
    s = ORPOStrategy(MagicMock())
    ds = _PairDataset(
        column_names=["chosen", "rejected"],
        sample={},
    )
    assert s.validate_dataset(ds).is_success()


@dataclass
class _SAPODataset:
    column_names: list[str]


def test_sapo_validate_dataset_requires_prompt() -> None:
    s = SAPOStrategy(MagicMock())
    ds = _SAPODataset(column_names=[])
    result = s.validate_dataset(ds)
    assert result.is_failure()
    assert result.unwrap_err().code == "RL_MISSING_PROMPT_COLUMN"


def test_sapo_validate_dataset_with_prompt() -> None:
    s = SAPOStrategy(MagicMock())
    ds = _SAPODataset(column_names=["prompt", "reference_answer"])
    result = s.validate_dataset(ds)
    assert result.is_success()


def test_sapo_build_trainer_kwargs_returns_empty_without_injected_reward() -> None:
    s = SAPOStrategy(MagicMock())
    out = s.build_trainer_kwargs(config=None)
    assert out == {}
