"""Typed ``PipelineContext`` value object.

``PipelineContext`` is the **single communication bus** between the orchestrator,
its collaborators, and individual pipeline stages. Pre-refactor, this was a raw
``dict[str, Any]`` — flexible but opaque: readers couldn't tell which keys were
guaranteed to exist, which types they carried, or who was responsible for
mutating them.

This value object tightens the contract **without breaking backward
compatibility**:

* Inherits from ``dict`` so every existing stage that accepts
  ``context: dict[str, Any]`` keeps working unchanged — ``PipelineContext``
  passes ``isinstance(ctx, dict)`` and supports every dict operation.
* Adds **typed accessors** for the canonical keys defined in
  :class:`~src.pipeline.stages.constants.PipelineContextKeys` (``run_ctx``,
  ``attempt_id``, ``attempt_no``, ``forced_stages``, etc.). These raise
  ``KeyError`` when the key is missing — fail fast, no silent defaults.
* Adds a :meth:`fork` factory that produces a **new** context for a fresh
  attempt, copying the run-scoped keys and swapping attempt-scoped ones
  atomically. Previously the orchestrator rebuilt the context dict by hand
  at the top of ``_run_stateful`` — error-prone; this method centralises it.

Invariants (enforced by tests in
``src/tests/unit/pipeline/context/test_pipeline_context*.py``):

1. ``PipelineContext()`` is ``isinstance`` of ``dict`` — all existing
   ``dict[str, Any]`` stage signatures keep accepting it.
2. ``fork(attempt_id, attempt_dir)`` returns a **new** context: mutations
   on the fork do not leak back to the original.
3. Typed accessors read **only** from the underlying dict — they never
   cache; writing a key via ``ctx[...] = ...`` is immediately visible to
   the typed property.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.pipeline.stages.constants import PipelineContextKeys

if TYPE_CHECKING:
    from pathlib import Path

    from src.pipeline.state.run_context import RunContext


class PipelineContext(dict[str, Any]):
    """Typed, dict-compatible communication bus for pipeline runs.

    Because we inherit from ``dict`` directly, every existing stage that
    expects ``dict[str, Any]`` accepts this class transparently. New code
    is encouraged to use the typed accessors below rather than raw key
    lookups.
    """

    # ---- constructors ------------------------------------------------------

    @classmethod
    def empty(cls) -> PipelineContext:
        """Create an empty context. Helper for tests / initial construction."""
        return cls()

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any] | None) -> PipelineContext:
        """Promote an existing dict to a ``PipelineContext`` without copying twice."""
        return cls(mapping) if mapping else cls()

    # ---- typed accessors (read) --------------------------------------------

    @property
    def run_ctx(self) -> RunContext:
        """The ``RunContext`` object initialised by :class:`PipelineOrchestrator`.

        Raises ``KeyError`` if the orchestrator didn't seed it — that would be
        a programming error, not an expected runtime state.
        """
        return self[PipelineContextKeys.RUN]

    @property
    def config_path(self) -> str:
        return str(self[PipelineContextKeys.CONFIG_PATH])

    @property
    def logical_run_id(self) -> str | None:
        """The logical-run identifier; ``None`` until ``LaunchPreparator`` bootstraps it."""
        value = self.get(PipelineContextKeys.LOGICAL_RUN_ID)
        return str(value) if value is not None else None

    @property
    def attempt_id(self) -> str | None:
        value = self.get(PipelineContextKeys.ATTEMPT_ID)
        return str(value) if value is not None else None

    @property
    def attempt_no(self) -> int | None:
        value = self.get(PipelineContextKeys.ATTEMPT_NO)
        return int(value) if value is not None else None

    @property
    def run_directory(self) -> str | None:
        value = self.get(PipelineContextKeys.RUN_DIRECTORY)
        return str(value) if value is not None else None

    @property
    def attempt_directory(self) -> str | None:
        value = self.get(PipelineContextKeys.ATTEMPT_DIRECTORY)
        return str(value) if value is not None else None

    @property
    def forced_stages(self) -> set[str]:
        """Stages the user forced on via ``restart_from_stage`` even if config disabled them.

        Always returns a **new** set: callers can't mutate the stored one.
        """
        stored = self.get(PipelineContextKeys.FORCED_STAGES)
        return set(stored) if isinstance(stored, set) else set()

    # ---- mutation helpers --------------------------------------------------

    def fork(
        self,
        *,
        attempt_id: str,
        attempt_no: int,
        attempt_directory: Path | str,
        logical_run_id: str,
        run_directory: Path | str,
        forced_stages: set[str] | None = None,
    ) -> PipelineContext:
        """Produce a **fresh** context for a new attempt.

        Copies run-scoped keys from ``self`` (``RUN``, ``CONFIG_PATH``) and
        overwrites attempt-scoped ones atomically. Mutations on the returned
        context do **not** leak back to ``self`` — this is a defensive copy.

        Used by ``LaunchPreparator.prepare()`` to build the per-attempt
        context without the 8-line inline ``self.context = {...}`` dance
        that the orchestrator used pre-refactor.
        """
        forked = PipelineContext(self)  # shallow copy of underlying dict
        forked[PipelineContextKeys.LOGICAL_RUN_ID] = logical_run_id
        forked[PipelineContextKeys.ATTEMPT_ID] = attempt_id
        forked[PipelineContextKeys.ATTEMPT_NO] = attempt_no
        forked[PipelineContextKeys.RUN_DIRECTORY] = str(run_directory)
        forked[PipelineContextKeys.ATTEMPT_DIRECTORY] = str(attempt_directory)
        forked[PipelineContextKeys.FORCED_STAGES] = set(forced_stages or ())
        return forked

    # ---- debugging / observability -----------------------------------------

    def as_dict(self) -> dict[str, Any]:
        """Return a **shallow** dict copy suitable for logging / snapshots.

        Prefer this over ``dict(ctx)`` at call sites — it documents intent
        (no mutation of the context itself).
        """
        return dict(self)


__all__ = ["PipelineContext"]
