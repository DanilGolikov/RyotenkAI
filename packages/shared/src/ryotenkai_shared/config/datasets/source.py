"""Tag-based discriminated union for dataset source configs.

Replaces the legacy flat-namespace shape (``source_type: Literal[…]``
selector + ``source_local|source_hf: ... | None`` flat fields) with a
single discriminator-driven field — ``source: DatasetSourceUnion``.

YAML before::

    datasets:
      default:
        source_type: local
        source_local:
          local_paths:
            train: data/train.jsonl

YAML after::

    datasets:
      default:
        source:
          kind: local
          local_paths:
            train: data/train.jsonl

Wins:
  * One source of truth for the source selector (``kind``).
  * Pydantic catches typos at YAML load (vs runtime ``KeyError``).
  * Type-narrowed access in code: ``cfg.datasets["d"].source.local_paths``
    is typed via the union member matching ``kind``.
  * Adding a new source (e.g. ``s3``) = new class + 1 line in this file.
"""

from __future__ import annotations

from typing import Annotated, Union

from pydantic import Discriminator, Tag

from .sources import DatasetSourceHF, DatasetSourceLocal

#: The discriminator field is uniformly named ``kind`` across the codebase
#: (engines, dataset sources, training adapters).
DISCRIMINATOR_FIELD = "kind"


#: Tag-based discriminated union of all dataset-source variants.
#:
#: Pydantic dispatches on ``kind: Literal["local"|"huggingface"]``.
DatasetSourceUnion = Annotated[
    Union[
        Annotated[DatasetSourceLocal, Tag("local")],
        Annotated[DatasetSourceHF, Tag("huggingface")],
    ],
    Discriminator(DISCRIMINATOR_FIELD),
]


__all__ = (
    "DISCRIMINATOR_FIELD",
    "DatasetSourceUnion",
)
