"""Tag-based discriminated union builder for engine configs.

Pydantic v2 design context:
* A plain ``Union[T]`` collapses to ``T`` and rejects ``Field(discriminator=…)``.
* ``Annotated[T, Tag("…")] + Discriminator("kind")`` ALSO requires Pydantic
  to see a ``Union`` type — single-member case raises ``TypeError:
  Discriminator must be used with a Union type``.
* Therefore we shape-shift based on member count:
    - 0 engines: ``_NoEnginesPlaceholder`` (deployment misconfig).
    - 1 engine:  return the raw config class — Pydantic still enforces
      ``kind: Literal["<id>"]`` on the class itself (closed-world for
      one engine is degenerate but type-safe).
    - 2+ engines: real ``Annotated[Union[Tag…], Discriminator("kind")]``.

The "1 engine" branch is a temporary degenerate case — once we add a
second engine the Tag/Discriminator path activates. The contract from the
caller's perspective is identical: the field accepts a config dict and
returns a typed instance.

Public API:
  * :func:`build_engine_config_union` — build the union from a registry.
  * :func:`get_engine_config_union` — module-level lazy singleton, used
    by ``ryotenkai_shared.config.inference.schema``.

Adding a new engine ⇒ drop folder ⇒ registry sees it ⇒ union picks it up.
No edits to ``InferenceConfig`` or any consumer.
"""

from __future__ import annotations

import threading
from typing import Annotated, Any, Union

from pydantic import Discriminator, Tag

from ryotenkai_engines.registry import EngineRegistry, get_registry

#: Discriminator field name. Uniformly ``"kind"`` across all engines.
#: This is part of the public YAML contract — operators see ``engine.kind:
#: vllm`` in their configs.
DISCRIMINATOR_FIELD = "kind"


def build_engine_config_union(registry: EngineRegistry | None = None) -> Any:
    """Build the discriminated union over all registered engine configs.

    Args:
        registry: Optional registry to use. Defaults to the lazy singleton
            from :func:`ryotenkai_engines.registry.get_registry`. Tests
            pass a synthetic registry (``EngineRegistry.from_filesystem(roots=[tmp])``).

    Returns:
        A Pydantic-compatible type:

          * Empty registry: :class:`_NoEnginesPlaceholder` (caller treats as
            deployment misconfig — at least one engine must be registered).

          * One engine: the raw config class. Pydantic's ``kind: Literal["<id>"]``
            on the class still enforces type-safety; ``extra="forbid"`` rejects
            unknown fields. The Tag/Discriminator wrapper is skipped because
            Pydantic v2 requires a ``Union`` type for ``Discriminator``.

          * Multiple engines: ``Annotated[Union[Tag(A) | Tag(B) | …],
            Discriminator("kind")]`` — fully discriminated.
    """
    registry = registry if registry is not None else get_registry()
    engine_ids = registry.list()

    if not engine_ids:
        return _NoEnginesPlaceholder

    if len(engine_ids) == 1:
        # Degenerate one-engine case — return the raw class.
        # The ``kind: Literal[…]`` on the class still enforces the
        # discriminator value at the field level.
        return registry.get_config_class(engine_ids[0])

    members: list[Any] = []
    for engine_id in engine_ids:
        cfg_cls = registry.get_config_class(engine_id)
        members.append(Annotated[cfg_cls, Tag(engine_id)])

    union_type = Union[tuple(members)]  # type: ignore[valid-type]
    return Annotated[union_type, Discriminator(DISCRIMINATOR_FIELD)]


# ---------------------------------------------------------------------------
# Lazy singleton — shared.config imports this; building once at import time
# is fine because the engine registry is itself a lock-protected singleton.
# ---------------------------------------------------------------------------


_lock = threading.Lock()
_cached_union: Any = None


def get_engine_config_union() -> Any:
    """Lazy singleton — build once, return on every call.

    Used by ``ryotenkai_shared.config.inference.schema.InferenceConfig.engine``.
    Tests that need a synthetic union call :func:`build_engine_config_union`
    directly with a custom registry, then use that result inline rather
    than calling this singleton.
    """
    global _cached_union
    if _cached_union is not None:
        return _cached_union
    with _lock:
        if _cached_union is None:
            _cached_union = build_engine_config_union()
        return _cached_union


def reset_engine_config_union() -> None:
    """Clear the cached union — used by test fixtures."""
    global _cached_union
    with _lock:
        _cached_union = None


# ---------------------------------------------------------------------------
# Placeholder for the "no engines registered" case
# ---------------------------------------------------------------------------


class _NoEnginesPlaceholder:
    """Marker type used when the registry is empty.

    Used in place of a real Pydantic union so the union builder always
    returns SOMETHING (consumers can pattern-match on ``is _NoEnginesPlaceholder``
    if they want to surface a clearer error). In practice an empty
    registry indicates a deployment misconfiguration; the caller's
    Pydantic validation will fail with an unintelligible "type not
    callable" error, which is fine for this edge case (we never expect
    to ship without at least one engine).
    """


__all__ = (
    "DISCRIMINATOR_FIELD",
    "build_engine_config_union",
    "get_engine_config_union",
    "reset_engine_config_union",
)
