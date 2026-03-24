from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import src.training.models.loader as loader
from src.utils.config import LoraConfig as LoraConfigType


class _Model:
    def __init__(self):
        self.config = SimpleNamespace(model_type="x", use_cache=True)
        self._grads_enabled = False

    def enable_input_require_grads(self) -> None:
        self._grads_enabled = True

    def num_parameters(self) -> int:
        return 123


class _Tokenizer:
    def __init__(self, *, pad_token: str | None = None):
        self.pad_token = pad_token
        self.pad_token_id = None
        self.eos_token = "<eos>"
        self.eos_token_id = 2
        self.padding_side = None

    def __len__(self) -> int:
        return 100


def _mk_cfg(*, use_4bit: bool) -> MagicMock:
    cfg = MagicMock()
    cfg.model.name = "org/model"
    cfg.model.torch_dtype = "bfloat16"
    cfg.model.device_map = "auto"
    cfg.model.trust_remote_code = False
    cfg.model.flash_attention = True
    cfg.model.tokenizer_name = None

    cfg.training.get_effective_load_in_4bit.return_value = use_4bit
    cfg.training.lora = LoraConfigType(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules="all-linear",
        use_dora=False,
        use_rslora=False,
        init_lora_weights="gaussian",
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="bfloat16",
        bnb_4bit_use_double_quant=True,
    )
    return cfg


def test_get_torch_dtype_mapping() -> None:
    assert loader._get_torch_dtype("float16") == loader.torch.float16
    assert loader._get_torch_dtype("unknown") == loader.torch.bfloat16


def test_load_model_and_tokenizer_4bit_sets_quant_config_and_pad_token(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _mk_cfg(use_4bit=True)

    created = {}

    def fake_bnb(**kw):
        created.update(kw)
        return {"bnb": True}

    monkeypatch.setattr(loader, "BitsAndBytesConfig", fake_bnb)
    monkeypatch.setattr(loader.AutoModelForCausalLM, "from_pretrained", lambda *a, **k: _Model())
    monkeypatch.setattr(loader.AutoTokenizer, "from_pretrained", lambda *a, **k: _Tokenizer(pad_token=None))

    model, tok = loader.load_model_and_tokenizer(cfg)
    assert model.config.use_cache is False
    assert model._grads_enabled is True
    assert tok.pad_token == tok.eos_token
    assert tok.pad_token_id == tok.eos_token_id
    assert tok.padding_side == "right"
    assert created["load_in_4bit"] is True


def test_load_model_and_tokenizer_no_4bit_does_not_create_bnb(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _mk_cfg(use_4bit=False)

    monkeypatch.setattr(loader, "BitsAndBytesConfig", lambda **kw: (_ for _ in ()).throw(AssertionError("no bnb")))
    monkeypatch.setattr(loader.AutoModelForCausalLM, "from_pretrained", lambda *a, **k: _Model())
    monkeypatch.setattr(loader.AutoTokenizer, "from_pretrained", lambda *a, **k: _Tokenizer(pad_token="PAD"))

    _model, tok = loader.load_model_and_tokenizer(cfg)
    assert tok.pad_token == "PAD"
