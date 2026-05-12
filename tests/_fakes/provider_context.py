"""``make_provider_context`` — factory for the production :class:`ProviderContext`.

Phase 1.6 / 1.7 changed the constructor of every provider class
(``RunPodProvider``, ``SingleNodeProvider``, …) from the legacy
``(config=dict, secrets=Secrets)`` keyword pair to a single
:class:`ProviderContext` argument. The greenfield tests that survived
the migration still call the legacy form; this factory lets them keep
their dict-shaped fixtures while constructing a real
:class:`ProviderContext` that the production constructor accepts.

Why a factory rather than a fake class:

* :class:`ProviderContext` is a small, frozen dataclass — there's no
  surface to fake. Tests want a *real* instance with overridable
  defaults; ``__init__`` simply forwards to the production type.
* Provider constructors accept either a typed Pydantic block
  (`RunPodProviderConfig`) or a raw mapping (via the ``from_dict``
  fallback). The factory takes a dict / mapping under
  ``provider_block`` and lets the provider do its own promotion —
  this matches how the legacy tests built their dicts.

The factory does **not** validate the dict — that's the production
config validator's job — so a malformed ``provider_block`` will raise
inside the provider constructor, exactly as it would in production.

Usage::

    from tests._fakes.provider_context import make_provider_context

    ctx = make_provider_context(
        provider_id="runpod",
        provider_block={"training": {...}, "connect": {...}, ...},
        secrets=Secrets(RUNPOD_API_KEY="rk", HF_TOKEN="hf"),
    )
    provider = RunPodProvider(ctx)
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from ryotenkai_providers.registry import ProviderContext
from ryotenkai_providers.training.interfaces import ProviderCapabilities
from ryotenkai_shared.config import Secrets


def _default_pipeline_config() -> Any:
    """Return a minimal stand-in for :class:`PipelineConfig`.

    Provider ``__init__`` paths under test don't read
    ``pipeline_config`` (training providers consume ``provider_block``
    only). A MagicMock is structurally compatible with the field type
    (`Any`-erased through ``ctx.pipeline_config``) and lets tests
    monkey-patch any attribute they happen to touch in a code path
    they're not actually exercising.
    """
    return MagicMock(name="FakePipelineConfig")


def make_provider_context(
    *,
    provider_id: str = "runpod",
    config: dict[str, Any] | None = None,
    provider_block: Any = None,
    secrets: Secrets | None = None,
    pipeline_config: Any | None = None,
) -> ProviderContext:
    """Build a real :class:`ProviderContext` for tests.

    Args:
        provider_id: Canonical id of the provider; defaults to
            ``"runpod"`` because most callers are RunPod tests.
        config: Legacy compatibility alias for ``provider_block`` —
            tests that already had ``config=...`` keyword can pass it
            unchanged.
        provider_block: Typed Pydantic block or raw mapping; the
            provider constructor handles promotion.
        secrets: :class:`Secrets` bundle. Defaults to an empty
            ``Secrets()`` (sufficient when the path under test doesn't
            require real keys).
        pipeline_config: Full :class:`PipelineConfig` instance.
            Defaults to a MagicMock stand-in (see
            :func:`_default_pipeline_config`).
    """
    block = provider_block if provider_block is not None else config
    if block is None:
        block = {}
    return ProviderContext(
        provider_id=provider_id,
        pipeline_config=pipeline_config if pipeline_config is not None else _default_pipeline_config(),
        provider_block=block,
        secrets=secrets if secrets is not None else Secrets(),
    )


def attach_manifest_capabilities(
    provider_class: type,
    *,
    provider_id: str,
    provider_name: str | None = None,
    provider_type: str = "cloud",
    capabilities: ProviderCapabilities | None = None,
) -> None:
    """Attach manifest-style ClassVars to a provider class.

    Tests that bypass :class:`ProviderRegistry` (which normally sets
    these at load time) and call ``provider.provider_name`` or
    ``provider.get_capabilities()`` need the ClassVar set explicitly.
    This helper does so idempotently — repeated calls overwrite, which
    is fine because the values are derived from the manifest and tests
    pass consistent values.

    See ``ProviderBase`` docstring (interfaces.py): "Test fixtures may
    set the ClassVar directly."
    """
    provider_class._manifest_provider_id = provider_id
    provider_class._manifest_provider_name = provider_name or provider_id
    provider_class._manifest_provider_type = provider_type
    provider_class._manifest_capabilities = capabilities or ProviderCapabilities(
        provider_type=provider_type,
    )


__all__ = ["make_provider_context", "attach_manifest_capabilities"]
