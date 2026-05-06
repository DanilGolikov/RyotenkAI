"""Tag-based discriminated union builder for engine configs.

PR-1 stub — landed in PR-2. The challenge: Pydantic v2 rejects a single-
member ``Union[T]`` (it collapses to ``T``), but we want type-safety from
day one (today: only vLLM exists; tomorrow: SGLang, MAX, …).

Workaround documented in Pydantic v2.5+ docs:
``Annotated[T, Tag("…")]`` + ``Discriminator("kind")`` works with one OR
many members. We use it from day one so:

  1. The contract is uniform: every engine config IS a discriminated variant.
  2. Adding a second engine = drop folder; the union builder auto-discovers
     it via the registry. No edits to ``InferenceConfig``.

Sketch (lands in PR-2)::

    def build_engine_config_union():
        registry = EngineRegistry.from_filesystem()
        members = [
            Annotated[registry.get_config_class(eid), Tag(eid)]
            for eid in registry.list()
        ]
        if len(members) == 1:
            return Annotated[members[0], Discriminator("kind")]
        return Annotated[Union[tuple(members)], Discriminator("kind")]

    EngineConfigUnion = build_engine_config_union()

Consumed by ``packages/shared/src/ryotenkai_shared/config/inference/schema.py``
in PR-6.
"""

from __future__ import annotations

# TODO(PR-2): build_engine_config_union(); EngineConfigUnion module-level.
__all__: tuple[str, ...] = ()
