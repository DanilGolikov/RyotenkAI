"""Pipeline stage conditions — Phase G (Layer 10).

Augments the FSM-based stage lifecycle (``StageRunState.status``) with
a k8s/OpenShift-Operator-style ``conditions[]`` array. Each
:class:`Condition` is a typed observation (type + status + reason +
message + last_transition_time) — multiple can be true simultaneously
on the same stage.

The FSM remains the source of truth for lifecycle state ("is this
stage running / failed / completed?"). Conditions add a side-channel
for **warnings and progress hints** that the FSM cannot represent
cleanly:

* ``Available``    — stage is ready to do work.
* ``Progressing``  — operation in progress (mirrors ``status="running"``
  but also true during retries / sub-step transitions).
* ``Degraded``     — issues but operation continues (or has finished
  badly).
* ``OOMRisk``      — GPU memory near the limit; failure imminent.
* ``RateLimited``  — provider throttling our calls.

Convention: ``last_transition_time`` updates ONLY when ``status``
changes (k8s metav1.Condition rule). Repeated ``update_condition``
calls with the same status are idempotent w.r.t. the timestamp; only
``reason`` / ``message`` are refreshed.

Deviation from plan Layer 10
----------------------------
The plan specifies ``reason: ErrorCode``. We instead use
``reason: str`` (CamelCase, validated by a regex) because Operator
convention emits **positive** reasons too (``"AsExpected"``,
``"MinimumReplicasAvailable"``) that don't fit the failure-only
:class:`ErrorCode` enum. The :mod:`tests/_lint/test_condition_reason_format`
sentinel keeps literal ``reason=...`` callsites CamelCase so the
shape stays consistent with k8s tooling that consumes conditions[].

References
----------
- k8s ``meta/v1.Condition``:
  https://pkg.go.dev/k8s.io/apimachinery/pkg/apis/meta/v1#Condition
- Operator SDK status conventions:
  https://sdk.operatorframework.io/docs/building-operators/golang/references/client/#updating-status-subresource
- Project plan:
  ``docs/plans/sharded-stargazing-wigderson.md`` (Phase G, Layer 10).
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = [
    "CAMEL_CASE_RE",
    "STANDARD_CONDITION_TYPES",
    "Condition",
    "ConditionStatus",
    "update_condition",
]


class ConditionStatus(StrEnum):
    """Tri-state condition value, mirroring k8s metav1.ConditionStatus.

    ``UNKNOWN`` is distinct from ``FALSE`` — it signals "we couldn't
    determine, treat as transient" so a dashboard can render a
    different glyph (yellow vs red) than for a definitively-false
    condition.
    """

    TRUE = "True"
    FALSE = "False"
    UNKNOWN = "Unknown"


# CamelCase validator. ``reason`` strings MUST start with an uppercase
# ASCII letter and contain only ASCII letters/digits (no underscore,
# no hyphen, no spaces). Matches k8s convention (``"AsExpected"``,
# ``"MinimumReplicasAvailable"``).
CAMEL_CASE_RE: re.Pattern[str] = re.compile(r"^[A-Z][A-Za-z0-9]*$")


# Standard, well-known condition types seeded by Phase G emitters. New
# types are allowed (the field is ``str``, not an enum) but listing
# the common ones makes the contract self-documenting and gives the
# CLI a stable column order.
STANDARD_CONDITION_TYPES: tuple[str, ...] = (
    "Available",
    "Progressing",
    "Degraded",
    "OOMRisk",
    "RateLimited",
)


class Condition(BaseModel):
    """One observation about a pipeline stage.

    Multiple conditions can coexist on a single stage: e.g. a stage
    can be ``Progressing=True`` AND ``OOMRisk=True`` simultaneously,
    which a single FSM status cannot express.
    """

    # ``frozen=True`` makes :class:`Condition` immutable after
    # construction so the CamelCase ``reason`` validator cannot be
    # bypassed by post-hoc attribute assignment. Updates go through
    # :func:`update_condition`, which substitutes a fresh model
    # instance via ``model_copy``.
    model_config = ConfigDict(extra="forbid", frozen=True)

    type: str = Field(
        description=(
            "CamelCase semantic category (``Available``, ``Progressing``, "
            "``Degraded``, ``OOMRisk``, ``RateLimited`` are seeded; "
            "custom types are permitted but should follow the same "
            "convention)."
        ),
    )
    status: ConditionStatus = Field(
        description="Tri-state value: True / False / Unknown.",
    )
    reason: str = Field(
        description=(
            "Machine-readable CamelCase reason. Free-form string "
            "(NOT :class:`ErrorCode`) so positive states like "
            "``\"AsExpected\"`` are expressible alongside failure "
            "reasons like ``\"GPUMemoryHigh\"``."
        ),
    )
    message: str | None = Field(
        default=None,
        description="Human-readable diagnostic. Optional.",
    )
    last_transition_time: datetime = Field(
        description=(
            "Timestamp of the last status change. Updates ONLY when "
            "``status`` flips (k8s convention)."
        ),
    )

    @field_validator("type")
    @classmethod
    def _validate_type_camel_case(cls, v: str) -> str:
        if not CAMEL_CASE_RE.match(v):
            raise ValueError(
                f"Condition.type must be CamelCase (matching {CAMEL_CASE_RE.pattern!r}); got {v!r}",
            )
        return v

    @field_validator("reason")
    @classmethod
    def _validate_reason_camel_case(cls, v: str) -> str:
        if not CAMEL_CASE_RE.match(v):
            raise ValueError(
                f"Condition.reason must be CamelCase (matching {CAMEL_CASE_RE.pattern!r}); got {v!r}",
            )
        return v


def update_condition(
    conditions: list[Condition],
    *,
    type: str,
    status: ConditionStatus,
    reason: str,
    message: str | None = None,
    now: datetime | None = None,
) -> list[Condition]:
    """Idempotent in-place update of a condition list.

    K8s convention: ``last_transition_time`` is bumped ONLY when the
    new ``status`` differs from the previous one. Repeated calls with
    the same status only refresh ``reason`` and ``message`` (useful
    when the underlying reason narrows over time but the high-level
    True/False didn't flip).

    Mutates the input list in place AND returns it for convenience
    (so callers can write ``state.conditions = update_condition(...)``
    or ignore the return value).

    Args:
        conditions: Existing condition list (typically
            ``stage_run_state.conditions``). Mutated in place.
        type: CamelCase semantic category (see
            :data:`STANDARD_CONDITION_TYPES`).
        status: New :class:`ConditionStatus` value.
        reason: CamelCase machine-readable reason.
        message: Optional human-readable diagnostic.
        now: Override the current time (test seam). Defaults to
            :func:`datetime.now(UTC)`.

    Returns:
        The same list that was passed in (mutation already applied).
    """
    transition_time = now if now is not None else datetime.now(UTC)
    for i, existing in enumerate(conditions):
        if existing.type == type:
            # :class:`Condition` is frozen. Construct a fresh model
            # via the public constructor so the CamelCase validators
            # re-run on the new ``reason``. ``model_copy(update=...)``
            # SKIPS field validators in pydantic v2, which would let
            # callers slip a non-CamelCase reason through this seam.
            if existing.status != status:
                conditions[i] = Condition(
                    type=type,
                    status=status,
                    reason=reason,
                    message=message,
                    last_transition_time=transition_time,
                )
            else:
                # k8s convention: do NOT bump last_transition_time when
                # status is unchanged. Only refresh reason/message.
                conditions[i] = Condition(
                    type=type,
                    status=existing.status,
                    reason=reason,
                    message=message,
                    last_transition_time=existing.last_transition_time,
                )
            return conditions
    conditions.append(
        Condition(
            type=type,
            status=status,
            reason=reason,
            message=message,
            last_transition_time=transition_time,
        ),
    )
    return conditions
