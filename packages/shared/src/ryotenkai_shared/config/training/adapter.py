"""Tag-based discriminated union for training adapter configs.

Replaces the legacy flat-namespace shape (``type: str`` selector +
``lora|qlora|adalora`` flat fields) with a single discriminator-driven
field — ``adapter: AdapterConfigUnion``.

YAML before::

    training:
      type: qlora
      qlora:
        r: 16
        lora_alpha: 32

YAML after::

    training:
      adapter:
        kind: qlora
        r: 16
        lora_alpha: 32

Wins:
  * Single source of truth for the adapter selector (``kind``).
  * Pydantic catches typos at YAML load (vs runtime ``KeyError``).
  * Type-narrowed access in code: ``cfg.training.adapter.r`` is typed
    via the union member matching ``kind``.
  * Adding a new adapter (e.g. DoRA) = new class + 1 line in this file.

Discriminator design:
  * ``Tag(name)`` annotation pairs each variant class with its
    ``kind`` Literal value (``Tag("lora")``, etc.). The union is wrapped
    in ``Annotated[…, Discriminator("kind")]``.
  * Why Tag-based instead of bare ``Field(discriminator="kind")``?
    ``QloraConfig`` SUBCLASSES ``LoraConfig``, so a plain Union[…]
    treats QloraConfig as also satisfying LoraConfig's shape.
    Tag-based discriminator dispatches strictly on the ``kind`` value —
    the subclass relationship is irrelevant.
"""

from __future__ import annotations

from typing import Annotated, Union

from pydantic import Discriminator, Tag

from .lora.adalora import AdaLoraConfig
from .lora.lora import LoraConfig, QloraConfig

#: The discriminator field is uniformly named ``kind`` across the codebase
#: (engines, dataset sources, training adapters). Constant exists so callers
#: can reference one symbol if/when the field name ever changes (it won't).
DISCRIMINATOR_FIELD = "kind"


#: Tag-based discriminated union of all training-adapter variants.
#:
#: Pydantic dispatches on ``kind: Literal["lora"|"qlora"|"adalora"]`` —
#: subclass relationships (QloraConfig ⊂ LoraConfig) don't confuse it
#: because Tag pairs class ↔ tag value explicitly.
AdapterConfigUnion = Annotated[
    Union[
        Annotated[LoraConfig, Tag("lora")],
        Annotated[QloraConfig, Tag("qlora")],
        Annotated[AdaLoraConfig, Tag("adalora")],
    ],
    Discriminator(DISCRIMINATOR_FIELD),
]


__all__ = (
    "DISCRIMINATOR_FIELD",
    "AdapterConfigUnion",
)
