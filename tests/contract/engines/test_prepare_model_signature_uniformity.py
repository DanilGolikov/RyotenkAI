"""Contract: every engine's ``prepare_model`` exposes the canonical signature.

The provider's plan-runner calls every engine through the same kwargs
contract::

    engine.prepare_model(
        cfg=...,
        base_model=...,
        adapter_path_in_container=...,
        workspace_host_path=...,
        run_id=...,
        trust_remote_code=...,
    )

If an engine drifts (renames a kwarg, adds a positional, drops one), the
call site silently passes the wrong shape. This sentinel pins the
contract by walking every shipped engine and asserting its actual
signature matches the Protocol.

Pairs with :class:`IInferenceEngine` ``runtime_checkable`` Protocol — the
Protocol catches *missing* methods at instantiation; this sentinel
catches *signature drift* on a method that's present.
"""

from __future__ import annotations

import inspect

import pytest

from ryotenkai_engines.interfaces import IInferenceEngine
from ryotenkai_engines.registry import EngineRegistry

pytestmark = pytest.mark.unit


# Canonical signature pulled from the Protocol — single source of truth.
_PROTOCOL_PARAMS: tuple[str, ...] = (
    "cfg",
    "base_model",
    "adapter_path_in_container",
    "workspace_host_path",
    "run_id",
    "trust_remote_code",
)


@pytest.fixture(scope="module")
def registry() -> EngineRegistry:
    return EngineRegistry.from_filesystem()


@pytest.mark.parametrize("engine_id", EngineRegistry.from_filesystem().list())
def test_prepare_model_kwargs_uniform(
    registry: EngineRegistry, engine_id: str
) -> None:
    """Every engine's ``prepare_model`` accepts exactly the canonical kwargs.

    Allows extra keyword-only parameters with defaults (forward-compat
    optional args), but the canonical six MUST be present and
    keyword-only — providers always pass them by name.
    """
    runtime_cls = registry.get_runtime(engine_id)
    method = runtime_cls.prepare_model
    sig = inspect.signature(method)

    # Drop ``self`` if present (bound vs unbound).
    params = [p for p in sig.parameters.values() if p.name != "self"]
    param_names = [p.name for p in params]

    for canonical in _PROTOCOL_PARAMS:
        assert canonical in param_names, (
            f"{engine_id}: prepare_model missing canonical kwarg "
            f"{canonical!r}; got {param_names}"
        )
        p = sig.parameters[canonical]
        assert p.kind == inspect.Parameter.KEYWORD_ONLY, (
            f"{engine_id}: prepare_model.{canonical} must be keyword-only "
            f"(got {p.kind.name}). The provider always calls by name; "
            f"positional params are an accident waiting to happen."
        )


@pytest.mark.parametrize("engine_id", EngineRegistry.from_filesystem().list())
def test_prepare_model_callable(
    registry: EngineRegistry, engine_id: str
) -> None:
    """``prepare_model`` exists and is callable — Protocol membership only
    asserts attribute presence, not callability."""
    runtime = registry.get_runtime(engine_id)()
    assert callable(getattr(runtime, "prepare_model", None))


def test_protocol_method_signature_unchanged() -> None:
    """Pin the canonical six on the Protocol itself — drift in interfaces.py
    would loosen the contract for everyone. Updating ``_PROTOCOL_PARAMS``
    must accompany an intentional Protocol change."""
    sig = inspect.signature(IInferenceEngine.prepare_model)
    proto_params = [p for p in sig.parameters.values() if p.name != "self"]
    proto_names = tuple(p.name for p in proto_params)
    assert proto_names == _PROTOCOL_PARAMS, (
        f"Protocol prepare_model signature changed:\n"
        f"  expected: {_PROTOCOL_PARAMS}\n"
        f"  got:      {proto_names}\n"
        f"Update _PROTOCOL_PARAMS in this sentinel iff the change is intentional."
    )
