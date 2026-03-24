from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.providers.training.factory import GPUProviderFactory
from src.utils.config import Secrets
from src.utils.result import ProviderError


@dataclass
class DummyProvider:
    config: dict[str, Any]
    secrets: Secrets


def test_register_and_create_provider_roundtrip() -> None:
    prev = dict(GPUProviderFactory._providers)
    GPUProviderFactory._providers = {}
    try:
        GPUProviderFactory.register("dummy", DummyProvider)  # type: ignore[arg-type]

        secrets = Secrets(HF_TOKEN="hf_test")
        result = GPUProviderFactory.create("dummy", {"x": 1}, secrets)

        assert result.is_success()
        inst = result.unwrap()
        assert isinstance(inst, DummyProvider)
        assert inst.config == {"x": 1}
        assert inst.secrets.hf_token == "hf_test"
    finally:
        GPUProviderFactory._providers = prev


def test_create_unknown_provider_returns_provider_error() -> None:
    prev = dict(GPUProviderFactory._providers)
    GPUProviderFactory._providers = {}
    try:
        result = GPUProviderFactory.create("missing", {}, Secrets(HF_TOKEN="hf_test"))

        assert result.is_failure()
        error = result.unwrap_err()
        assert isinstance(error, ProviderError)
        assert error.code == "PROVIDER_NOT_REGISTERED"
        assert "missing" in error.message
    finally:
        GPUProviderFactory._providers = prev


def test_create_type_error_returns_provider_error() -> None:
    class BadProvider:
        def __init__(self) -> None:
            pass

    prev = dict(GPUProviderFactory._providers)
    GPUProviderFactory._providers = {}
    try:
        GPUProviderFactory.register("bad", BadProvider)  # type: ignore[arg-type]

        result = GPUProviderFactory.create("bad", {}, Secrets(HF_TOKEN="hf_test"))

        assert result.is_failure()
        error = result.unwrap_err()
        assert isinstance(error, ProviderError)
        assert error.code == "PROVIDER_INIT_FAILED"
        assert "bad" in error.message
    finally:
        GPUProviderFactory._providers = prev


def test_create_from_config_missing_key_returns_provider_error() -> None:
    prev = dict(GPUProviderFactory._providers)
    GPUProviderFactory._providers = {}
    try:
        GPUProviderFactory.register("dummy", DummyProvider)  # type: ignore[arg-type]

        result = GPUProviderFactory.create_from_config("missing", {"dummy": {}}, Secrets(HF_TOKEN="hf_test"))

        assert result.is_failure()
        error = result.unwrap_err()
        assert isinstance(error, ProviderError)
        assert error.code == "PROVIDER_CONFIG_MISSING"
        assert "missing" in error.message
    finally:
        GPUProviderFactory._providers = prev
