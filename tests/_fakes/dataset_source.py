"""``make_dataset_*`` — factories for typed :class:`DatasetSource` instances.

After Phase B packagization the production code that consumes
``config.get_primary_dataset()`` now uses :func:`isinstance` against the
typed :class:`DatasetSourceLocal` / :class:`DatasetSourceHF` Pydantic
models (see ``HFModelUploader.extract_datasets_for_readme`` in
``ryotenkai_control.pipeline.stages.model_retriever``).

Legacy greenfield tests built ``types.SimpleNamespace`` stubs with
``get_source_type``/``source_local``/``source_hf`` attributes — that
shape no longer matches the consumer, and ``_extract_datasets_for_readme``
returns ``[]`` because the isinstance check fails. This helper closes
that gap.

There are two shapes:

* :func:`make_dataset_local` — returns a real :class:`DatasetSourceLocal`
  when the inputs validate (strings), and a ``MagicMock(spec=DatasetSourceLocal)``
  when they don't (int, etc.). The mock keeps ``isinstance`` happy so
  production reaches the inner ``getattr`` fallback that handles those
  edge cases.
* :func:`make_dataset_hf` — same shape but for :class:`DatasetSourceHF`.

Both factories return a "dataset" object exposing a ``.source`` attribute
(the typed source) and the legacy convenience accessors
``get_source_type()`` / ``source_local`` / ``source_hf`` for back-compat
with tests that still introspect both shapes.

Consumers:

* ``tests/unit/control/pipeline/test_stages_model_retriever.py`` —
  ``_extract_datasets_for_readme`` + ``_extract_dataset_source_type_for_readme``
  parametrized tests.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

from ryotenkai_shared.config import DatasetSourceHF, DatasetSourceLocal


def _local_source(train: Any, eval_path: Any) -> Any:
    """Return a DatasetSourceLocal (real if valid, MagicMock(spec=...) otherwise)."""
    try:
        return DatasetSourceLocal(local_paths={"train": train, "eval": eval_path})
    except Exception:  # ValidationError, etc.
        m = MagicMock(spec=DatasetSourceLocal)
        m.kind = "local"
        m.local_paths = SimpleNamespace(train=train, eval=eval_path)
        return m


def _hf_source(train_id: Any, eval_id: Any) -> Any:
    """Return a DatasetSourceHF (real if valid, MagicMock(spec=...) otherwise)."""
    try:
        return DatasetSourceHF(train_id=train_id, eval_id=eval_id)
    except Exception:
        m = MagicMock(spec=DatasetSourceHF)
        m.kind = "huggingface"
        m.train_id = train_id
        m.eval_id = eval_id
        return m


def make_dataset_local(train: Any = None, eval_path: Any = None) -> SimpleNamespace:
    """Build a typed-local "dataset" object for ``_extract_datasets_for_readme`` tests.

    The returned namespace exposes ``.source`` (a real DatasetSourceLocal
    or MagicMock(spec=DatasetSourceLocal)) plus legacy aliases.
    """
    source = _local_source(train, eval_path)
    return SimpleNamespace(
        source=source,
        source_local=source,
        source_hf=None,
        get_source_type=lambda: "local",
    )


def make_dataset_hf(train_id: Any = None, eval_id: Any = None) -> SimpleNamespace:
    """Build a typed-HF "dataset" object for ``_extract_datasets_for_readme`` tests."""
    source = _hf_source(train_id, eval_id)
    return SimpleNamespace(
        source=source,
        source_local=None,
        source_hf=source,
        get_source_type=lambda: "huggingface",
    )


def make_dataset_with_kind(kind: str) -> SimpleNamespace:
    """Build a dataset whose ``.source.kind`` returns the given string.

    Used by ``_extract_dataset_source_type_for_readme`` parametrized tests
    where the test exercises the production code's "propagate non-empty
    strings" branch for forward-compat with future source types.
    """
    source = SimpleNamespace(kind=kind)
    return SimpleNamespace(
        source=source,
        source_local=None,
        source_hf=None,
        get_source_type=lambda: kind,
    )


__all__ = [
    "make_dataset_local",
    "make_dataset_hf",
    "make_dataset_with_kind",
]
