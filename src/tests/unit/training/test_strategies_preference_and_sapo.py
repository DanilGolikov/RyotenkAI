from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import src.training.strategies.dpo as dpo_mod
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


def test_dpo_validate_dataset_missing_columns() -> None:
    s = DPOStrategy(MagicMock())
    ds = _PairDataset(column_names=["chosen"], sample={})
    assert s.validate_dataset(ds).is_failure()


def test_dpo_validate_dataset_bad_message_shape() -> None:
    s = DPOStrategy(MagicMock())
    ds = _PairDataset(column_names=["chosen", "rejected"], sample={"chosen": "x", "rejected": []})
    res = s.validate_dataset(ds)
    assert res.is_failure()


def test_dpo_prepare_dataset_valid_pass_through() -> None:
    s = DPOStrategy(MagicMock())
    ds = _PairDataset(
        column_names=["chosen", "rejected"],
        sample={
            "chosen": [{"role": "user", "content": "hi"}],
            "rejected": [{"role": "user", "content": "no"}],
        },
    )
    res = s.prepare_dataset(ds, MagicMock())
    assert res.is_success()
    assert res.unwrap() is ds


def test_dpo_build_trainer_kwargs_uses_ref_model_when_provided() -> None:
    s = DPOStrategy(MagicMock())
    cfg = SimpleNamespace()
    ref = object()
    kwargs = s.build_trainer_kwargs(cfg, model=object(), ref_model=ref)
    assert kwargs["ref_model"] is ref


def test_dpo_post_build_config_hook_loads_reference_adapter_for_peft(monkeypatch: pytest.MonkeyPatch) -> None:
    # Patch PeftModel type check
    class DummyPeft:
        pass

    monkeypatch.setattr(dpo_mod, "PeftModel", DummyPeft)

    model = DummyPeft()
    model.active_adapter = "train"
    model.peft_config = {"train": SimpleNamespace(base_model_name_or_path="/adapter")}
    model.load_adapter = MagicMock()

    cfg = SimpleNamespace()

    s = DPOStrategy(MagicMock())
    # build_trainer_kwargs no longer mutates config — mutation lives in post_build_config_hook
    kwargs = s.build_trainer_kwargs(cfg, model=model, ref_model=None)
    assert "ref_model" not in kwargs or kwargs.get("ref_model") is None

    s.post_build_config_hook(cfg, model=model, ref_model=None)

    assert cfg.model_adapter_name == "train"
    assert cfg.ref_adapter_name == "reference"
    model.load_adapter.assert_called_once_with("/adapter", adapter_name="reference")


def test_orpo_validate_and_prepare_dataset() -> None:
    s = ORPOStrategy(MagicMock())
    ds = _PairDataset(
        column_names=["chosen", "rejected"],
        sample={
            "chosen": [{"role": "user", "content": "hi"}],
            "rejected": [{"role": "user", "content": "no"}],
        },
    )
    assert s.validate_dataset(ds).is_success()
    assert s.prepare_dataset(ds, MagicMock()).is_success()


@dataclass
class _SAPODataset:
    features: dict
    mapped: bool = False

    def map(self, fn):
        # exercise mapping once
        fn({"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"}]})
        self.mapped = True
        return self


def test_sapo_validate_dataset_requires_prompt_or_messages() -> None:
    s = SAPOStrategy(MagicMock())
    ds = _SAPODataset(features={})
    result = s.validate_dataset(ds)
    assert result.is_failure()
    assert result.unwrap_err().code == "RL_MISSING_PROMPT_COLUMN"


def test_sapo_prepare_dataset_uses_prompt_column_when_present() -> None:
    s = SAPOStrategy(MagicMock())
    ds = _SAPODataset(features={"prompt": object(), "reference_answer": object(), "schema_context": object()})
    res = s.prepare_dataset(ds, tokenizer=MagicMock())
    assert res.is_success()
    assert res.unwrap() is ds


def test_sapo_prepare_dataset_extracts_prompt_from_messages(monkeypatch: pytest.MonkeyPatch) -> None:
    s = SAPOStrategy(MagicMock())
    ds = _SAPODataset(features={"messages": object()})

    tok = MagicMock()
    tok.chat_template = "x"
    tok.apply_chat_template.return_value = "PROMPT"

    res = s.prepare_dataset(ds, tokenizer=tok)
    assert res.is_success()
    assert ds.mapped is True


def test_sapo_build_trainer_kwargs_returns_empty_without_injected_reward() -> None:
    s = SAPOStrategy(MagicMock())
    out = s.build_trainer_kwargs(config=None)
    assert out == {}
