"""Smoke tests for the new :class:`SystemPromptLoader`.

Covers:
    * file-path success path (legacy parity).
    * mlflow-path success + cache hit on repeated load.
    * cache TTL expiry (re-fetch).
    * cache FIFO eviction on overflow.
    * the three ``on_mlflow_failure`` modes: ``fail`` / ``warn`` /
      ``fallback_to_file``.
    * :meth:`SystemPromptLoader.invalidate` (single key + full clear).
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from ryotenkai_shared.config.inference.common import InferenceLLMConfig
from ryotenkai_shared.inference.prompts import (
    SystemPromptLoader,
    SystemPromptResult,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


@dataclass
class _FakePromptArtifact:
    """Minimal in-memory stand-in for ``PromptArtifact``."""

    name: str
    version: str
    template: str


class _FakePromptRegistry:
    """Counts ``load()`` calls so cache behaviour is observable."""

    def __init__(self, templates: dict[str, str] | None = None) -> None:
        self._templates: dict[str, str] = templates or {}
        self.calls: list[tuple[str, float]] = []
        self.raise_with: Exception | None = None
        self.return_none: bool = False

    def load(self, name_or_uri: str, timeout_s: float):  # noqa: ANN201
        self.calls.append((name_or_uri, timeout_s))
        if self.raise_with is not None:
            raise self.raise_with
        if self.return_none:
            return None
        template = self._templates.get(name_or_uri)
        if template is None:
            return None
        return _FakePromptArtifact(name=name_or_uri, version="1", template=template)


# ---------------------------------------------------------------------------
# File path
# ---------------------------------------------------------------------------


def test_load_from_file_returns_result(tmp_path) -> None:
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("you are a helpful assistant", encoding="utf-8")
    cfg = InferenceLLMConfig(system_prompt_path=str(prompt_file))

    loader = SystemPromptLoader(registry=None)
    result = loader.load(cfg)

    assert isinstance(result, SystemPromptResult)
    assert result.text == "you are a helpful assistant"
    assert result.source == {"type": "file", "path": str(prompt_file)}


def test_load_returns_none_when_no_source_configured() -> None:
    loader = SystemPromptLoader(registry=None)
    assert loader.load(InferenceLLMConfig()) is None


# ---------------------------------------------------------------------------
# MLflow path + cache
# ---------------------------------------------------------------------------


def test_mlflow_path_caches_repeated_loads() -> None:
    registry = _FakePromptRegistry({"my-prompt": "be terse"})
    loader = SystemPromptLoader(registry=registry, cache_ttl_s=300.0)
    cfg = InferenceLLMConfig(system_prompt_mlflow_name="my-prompt")

    first = loader.load(cfg)
    second = loader.load(cfg)

    assert first is not None and second is not None
    assert first.text == second.text == "be terse"
    assert first.source["type"] == "mlflow"
    assert first.source["name"] == "my-prompt"
    # Cache hit on the second call: the fake registry must have been
    # called exactly once.
    assert len(registry.calls) == 1


def test_mlflow_cache_expires_after_ttl() -> None:
    registry = _FakePromptRegistry({"my-prompt": "first"})
    loader = SystemPromptLoader(
        registry=registry,
        cache_ttl_s=0.0,  # everything expires immediately
    )
    cfg = InferenceLLMConfig(system_prompt_mlflow_name="my-prompt")

    loader.load(cfg)
    loader.load(cfg)

    # TTL == 0 ⇒ each call sees a stale entry and re-fetches.
    assert len(registry.calls) == 2


def test_mlflow_cache_fifo_eviction_on_overflow() -> None:
    registry = _FakePromptRegistry({"a": "A", "b": "B", "c": "C"})
    loader = SystemPromptLoader(
        registry=registry,
        cache_ttl_s=300.0,
        cache_maxsize=2,
    )

    loader.load(InferenceLLMConfig(system_prompt_mlflow_name="a"))
    loader.load(InferenceLLMConfig(system_prompt_mlflow_name="b"))
    loader.load(InferenceLLMConfig(system_prompt_mlflow_name="c"))  # evicts "a"

    # "b" and "c" remain — hits ⇒ no new call.
    loader.load(InferenceLLMConfig(system_prompt_mlflow_name="b"))
    loader.load(InferenceLLMConfig(system_prompt_mlflow_name="c"))

    # "a" was evicted ⇒ re-fetch.
    loader.load(InferenceLLMConfig(system_prompt_mlflow_name="a"))

    call_keys = [k for k, _ in registry.calls]
    assert call_keys == ["a", "b", "c", "a"]


def test_invalidate_evicts_single_key() -> None:
    registry = _FakePromptRegistry({"my-prompt": "be terse"})
    loader = SystemPromptLoader(registry=registry)
    cfg = InferenceLLMConfig(system_prompt_mlflow_name="my-prompt")

    loader.load(cfg)
    loader.invalidate("my-prompt")
    loader.load(cfg)

    assert len(registry.calls) == 2


def test_invalidate_all_clears_cache() -> None:
    registry = _FakePromptRegistry({"a": "A", "b": "B"})
    loader = SystemPromptLoader(registry=registry)
    loader.load(InferenceLLMConfig(system_prompt_mlflow_name="a"))
    loader.load(InferenceLLMConfig(system_prompt_mlflow_name="b"))

    loader.invalidate()

    loader.load(InferenceLLMConfig(system_prompt_mlflow_name="a"))
    loader.load(InferenceLLMConfig(system_prompt_mlflow_name="b"))
    assert len(registry.calls) == 4


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


def test_on_mlflow_failure_fail_reraises_on_exception() -> None:
    registry = _FakePromptRegistry()
    registry.raise_with = RuntimeError("mlflow down")
    loader = SystemPromptLoader(registry=registry)
    cfg = InferenceLLMConfig(system_prompt_mlflow_name="missing")

    with pytest.raises(RuntimeError, match="MLflow load failed"):
        loader.load(cfg, on_mlflow_failure="fail")


def test_on_mlflow_failure_warn_returns_none_on_exception(caplog) -> None:
    registry = _FakePromptRegistry()
    registry.raise_with = RuntimeError("boom")
    loader = SystemPromptLoader(registry=registry)
    cfg = InferenceLLMConfig(system_prompt_mlflow_name="missing")

    result = loader.load(cfg, on_mlflow_failure="warn")
    assert result is None


def test_on_mlflow_failure_warn_returns_none_when_registry_returns_none() -> None:
    registry = _FakePromptRegistry()
    registry.return_none = True
    loader = SystemPromptLoader(registry=registry)
    cfg = InferenceLLMConfig(system_prompt_mlflow_name="absent")

    assert loader.load(cfg, on_mlflow_failure="warn") is None


def test_on_mlflow_failure_fallback_to_file_reads_local_when_set(tmp_path) -> None:
    prompt_file = tmp_path / "fallback.txt"
    prompt_file.write_text("fallback prompt", encoding="utf-8")

    registry = _FakePromptRegistry()
    registry.return_none = True  # simulate mlflow miss
    loader = SystemPromptLoader(registry=registry)

    # InferenceLLMConfig's validator forbids setting both fields at
    # the model level. We construct the config field-by-field via
    # ``model_construct`` to bypass validation, since this loader
    # supports fallback in failure scenarios where both happen to be
    # set in deployed configs that bypass the validator.
    cfg = InferenceLLMConfig.model_construct(
        system_prompt_path=str(prompt_file),
        system_prompt_mlflow_name="missing-prompt",
    )

    result = loader.load(cfg, on_mlflow_failure="fallback_to_file")
    assert result is not None
    assert result.text == "fallback prompt"
    assert result.source == {"type": "file", "path": str(prompt_file)}


def test_on_mlflow_failure_fallback_without_file_returns_none() -> None:
    registry = _FakePromptRegistry()
    registry.raise_with = RuntimeError("boom")
    loader = SystemPromptLoader(registry=registry)
    cfg = InferenceLLMConfig(system_prompt_mlflow_name="missing")

    assert loader.load(cfg, on_mlflow_failure="fallback_to_file") is None


def test_missing_registry_raises_when_mlflow_source_requested() -> None:
    loader = SystemPromptLoader(registry=None)
    cfg = InferenceLLMConfig(system_prompt_mlflow_name="missing")

    # No registry → counts as a registry-side failure routed through
    # ``on_mlflow_failure``.
    with pytest.raises(RuntimeError, match="MLflow load failed"):
        loader.load(cfg, on_mlflow_failure="fail")

    # warn mode just returns None.
    assert loader.load(cfg, on_mlflow_failure="warn") is None


# ---------------------------------------------------------------------------
# Constructor argument validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("ttl", "maxsize"),
    [(-1.0, 64), (300.0, 0)],
)
def test_constructor_rejects_invalid_cache_args(ttl: float, maxsize: int) -> None:
    with pytest.raises(ValueError):
        SystemPromptLoader(registry=None, cache_ttl_s=ttl, cache_maxsize=maxsize)
