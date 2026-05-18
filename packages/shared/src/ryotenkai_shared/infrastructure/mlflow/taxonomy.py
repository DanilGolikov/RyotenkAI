"""Single namespace for MLflow params / tags / metrics.

The audit found four parallel naming conventions in the pre-refactor code:

* ``log_training_config`` flat snake_case (``model_name``, ``lora_r``)
* ``log_pipeline_config`` dotted (``config.model.name``,
  ``training.hyperparams.*``)
* ``system_metrics_callback`` dotted (``system.gpu.0.name``)
  AND slash-style (``gpu/0/utilization``) for the same data
* HF Trainer-emitted (``train/loss``, ``eval/*``)

Target taxonomy: **lowercase dotted** under the ``ryotenkai.*`` namespace.
Matches the ``kind`` format used by the unified-event-system (ADR-0009)
and lets MLflow UI group by prefix in filter queries.

Reserved prefixes:

* ``mlflow.*`` — system tags. Blocked except the small whitelist
  (``mlflow.note.content``, ``mlflow.runName``, ``mlflow.parentRunId``,
  ``mlflow.source.*``, ``mlflow.user``) which MLflow itself sets/uses.
* ``ryotenkai.*`` — our namespace. Every required tag is declared in the
  enums below; the lint rule (M7) enforces that runtime code does not
  invent new keys outside this module.
* ``hf.*`` — HuggingFace Trainer auto-emitted (``hf.*``).

Per ``docs/plans/vectorized-fluttering-mist.md`` §Taxonomy.
"""

from __future__ import annotations

from enum import StrEnum


class TagKey(StrEnum):
    """All ``ryotenkai.*`` tag keys we set on MLflow runs.

    Tags are short, queryable, and shown in the UI tree. Use for
    identifiers, lifecycle status, lineage links, and grouping —
    NOT for high-cardinality data (which goes in params or
    artifacts).
    """

    # Lineage — set by ParentRunOpener
    LINEAGE_PIPELINE_ID = "ryotenkai.lineage.pipeline_id"
    LINEAGE_RUN_ID = "ryotenkai.lineage.run_id"
    LINEAGE_CONFIG_SHA256 = "ryotenkai.lineage.config_sha256"
    LINEAGE_CODE_COMMIT = "ryotenkai.lineage.code_commit"
    LINEAGE_RESUMES_FROM = "ryotenkai.lineage.resumes_from"

    # Attempt-level — set by ParentRunOpener.open_attempt
    ATTEMPT_ID = "ryotenkai.attempt.id"
    ATTEMPT_NO = "ryotenkai.attempt.no"

    # Lifecycle — split between Opener (open) and Finalizer (close)
    LIFECYCLE_OPENED_BY = "ryotenkai.lifecycle.opened_by"
    LIFECYCLE_FINALIZED = "ryotenkai.lifecycle.finalized"
    LIFECYCLE_STATUS = "ryotenkai.lifecycle.status"

    # Engine / provider classification
    ENGINE_KIND = "ryotenkai.engine.kind"
    PROVIDER_KIND = "ryotenkai.provider.kind"
    PROVIDER_GPU = "ryotenkai.provider.gpu"

    # Phase identification (set on phase runs)
    PHASE_IDX = "ryotenkai.phase.idx"
    PHASE_STRATEGY = "ryotenkai.phase.strategy"

    # Termination context
    EXIT_REASON = "ryotenkai.exit.reason"

    # Journal artifact integrity (set by JournalUploader)
    JOURNAL_SHA256 = "ryotenkai.journal.sha256"


class ParamKey(StrEnum):
    """Top-level ``ryotenkai.*`` param namespaces.

    Params are immutable once set per run; use the FULL key with
    the nested suffix when calling ``ITrackingClient.log_params``.
    These enum values are *prefixes* / canonical roots; the concrete
    keys are produced by appending nested names (e.g.
    ``CONFIG_MODEL_NAME = ParamKey.CONFIG + ".model.name"``).
    """

    CONFIG = "ryotenkai.config"  # whole config snapshot, dotted nesting
    TRAINING_ACTUAL = "ryotenkai.training.actual"  # effective hyperparams
    DATASET = "ryotenkai.dataset"  # dataset metadata


class MetricKey(StrEnum):
    """Top-level ``ryotenkai.*`` metric namespaces.

    HF Trainer-emitted (``train/loss``, ``eval/*``) and MLflow's
    native system-metric prefixes are NOT under our namespace and are
    accepted as-is.
    """

    METRIC = "ryotenkai.metric"  # custom domain metrics
    SUMMARY = "ryotenkai.metric.summary"  # aggregated metrics on attempt close


_MLFLOW_WHITELIST: frozenset[str] = frozenset(
    {
        "mlflow.note.content",
        "mlflow.runName",
        "mlflow.parentRunId",
        "mlflow.user",
    }
)
"""Tags MLflow itself sets/uses. Allowed in writes; everything else
under ``mlflow.*`` is rejected by :class:`ReservedPrefixGuard`."""

_MLFLOW_WHITELIST_PREFIXES: tuple[str, ...] = ("mlflow.source.",)
"""Prefixes (substring match) for system tags whose individual keys
are too numerous to enumerate (``mlflow.source.name``, ``.git.*``,
``.type``, ``.entryPoint``, etc.)."""


class ReservedPrefixGuard:
    """Block accidental writes to reserved tag/param prefixes.

    Use at every write site that accepts user-supplied keys:

        ReservedPrefixGuard.assert_safe(key)

    Raises ``ValueError`` if the key violates a reserved prefix
    that isn't on the whitelist. Stateless; safe to call inline.

    Behaviour matrix:

    * ``ryotenkai.*`` — accepted (callers should verify the suffix
      against the enums above; this guard only catches namespace
      violations, not unknown sub-keys).
    * ``mlflow.*`` — accepted only if on the whitelist or matches
      a whitelisted prefix; otherwise rejected.
    * ``hf.*`` — accepted (HF Trainer auto-emits these).
    * Anything else — accepted (third-party plugins, custom).
    """

    @staticmethod
    def assert_safe(key: str) -> None:
        if not key:
            msg = "Tag/param key must be non-empty."
            raise ValueError(msg)
        if key.startswith("mlflow."):
            if key in _MLFLOW_WHITELIST:
                return
            for prefix in _MLFLOW_WHITELIST_PREFIXES:
                if key.startswith(prefix):
                    return
            msg = (
                f"Tag/param key {key!r} starts with reserved 'mlflow.' prefix "
                f"but is not on the whitelist. Use 'ryotenkai.*' for our "
                f"namespace; whitelist is {sorted(_MLFLOW_WHITELIST)!r} "
                f"plus prefixes {list(_MLFLOW_WHITELIST_PREFIXES)!r}."
            )
            raise ValueError(msg)


__all__ = [
    "MetricKey",
    "ParamKey",
    "ReservedPrefixGuard",
    "TagKey",
]
