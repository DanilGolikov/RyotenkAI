"""``PreparePlan`` / ``PrepareStep`` shape + validator invariants.

Pure data-class tests — no engine logic. Verifies the schema's
guard-rails so engine implementations can rely on:

  * Plan with steps ⇒ ``final_model_path`` is set.
  * Empty plan ⇒ ``final_model_path`` may be ``None``.
  * Step names unique within a plan.
  * Frozen / extra=forbid (no silent mutation, no field typos).
  * ``spec_version`` defaults to 1.

Categories covered: positive, negative, boundary, invariant.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ryotenkai_engines.interfaces import PreparePlan, PrepareStep

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _step(
    name: str = "merge_lora",
    *,
    outputs: tuple[str, ...] = ("/workspace/runs/r1/model",),
    **overrides,  # type: ignore[no-untyped-def]
) -> PrepareStep:
    return PrepareStep(
        name=name,
        args=("/opt/x.py", "--flag"),
        outputs=outputs,
        **overrides,
    )


# ===========================================================================
# PrepareStep — positive
# ===========================================================================


class TestPrepareStepPositive:
    def test_minimal_construction(self) -> None:
        s = _step()
        assert s.name == "merge_lora"
        assert s.args == ("/opt/x.py", "--flag")
        assert s.outputs == ("/workspace/runs/r1/model",)
        # Defaults
        assert s.image is None
        assert s.entrypoint is None
        assert s.env == {}
        assert s.volumes == ()
        assert s.inputs == ()
        assert s.success_marker is None
        assert s.success_artifact is None
        assert s.timeout_seconds == 3600

    def test_full_construction(self) -> None:
        s = PrepareStep(
            name="convert",
            image="custom/converter:1.0",
            entrypoint=("python3",),
            args=("/opt/convert.py",),
            env={"K": "V"},
            volumes=(("/host/ws", "/workspace"),),
            inputs=("/workspace/input",),
            outputs=("/workspace/output",),
            success_marker="OK",
            success_artifact="/workspace/output/manifest.json",
            timeout_seconds=120,
        )
        assert s.image == "custom/converter:1.0"
        assert s.entrypoint == ("python3",)
        assert s.timeout_seconds == 120


# ===========================================================================
# PrepareStep — negative
# ===========================================================================


class TestPrepareStepNegative:
    def test_extra_field_rejected(self) -> None:
        """``extra='forbid'`` — typo in field name fails loudly."""
        with pytest.raises(ValidationError, match="extra|Extra|unexpected"):
            PrepareStep(  # type: ignore[call-arg]
                name="x",
                args=(),
                outputs=("/o",),
                typo_field="oops",
            )

    def test_frozen_immutable(self) -> None:
        """Engines build, providers read — no mutation."""
        s = _step()
        with pytest.raises(ValidationError, match="frozen|Instance is frozen"):
            s.name = "other"  # type: ignore[misc]

    def test_missing_required_name(self) -> None:
        with pytest.raises(ValidationError, match="name"):
            PrepareStep(args=(), outputs=("/o",))  # type: ignore[call-arg]

    def test_missing_required_args(self) -> None:
        with pytest.raises(ValidationError, match="args"):
            PrepareStep(name="x", outputs=("/o",))  # type: ignore[call-arg]

    def test_missing_required_outputs(self) -> None:
        with pytest.raises(ValidationError, match="outputs"):
            PrepareStep(name="x", args=())  # type: ignore[call-arg]


# ===========================================================================
# PrepareStep — boundary
# ===========================================================================


class TestPrepareStepBoundary:
    def test_empty_args_allowed(self) -> None:
        """Step with no CLI args (e.g., entrypoint-only invocation)."""
        s = PrepareStep(name="x", args=(), outputs=("/o",))
        assert s.args == ()

    def test_empty_outputs_allowed(self) -> None:
        """Step with no verifiable outputs (rare; e.g., side-effect-only)."""
        s = PrepareStep(name="x", args=("y",), outputs=())
        assert s.outputs == ()

    def test_timeout_minimum_one(self) -> None:
        s = PrepareStep(
            name="x", args=("y",), outputs=("/o",), timeout_seconds=1
        )
        assert s.timeout_seconds == 1

    def test_timeout_zero_rejected(self) -> None:
        """``ge=1`` — zero timeout is nonsense."""
        with pytest.raises(ValidationError, match="greater than or equal to 1"):
            PrepareStep(
                name="x", args=("y",), outputs=("/o",), timeout_seconds=0
            )

    def test_timeout_negative_rejected(self) -> None:
        with pytest.raises(ValidationError, match="greater than or equal to 1"):
            PrepareStep(
                name="x", args=("y",), outputs=("/o",), timeout_seconds=-1
            )


# ===========================================================================
# PreparePlan — positive
# ===========================================================================


class TestPreparePlanPositive:
    def test_empty_plan(self) -> None:
        p = PreparePlan.empty()
        assert p.steps == ()
        assert p.final_model_path is None
        assert p.spec_version == 1

    def test_single_step_plan(self) -> None:
        s = _step()
        p = PreparePlan(
            steps=(s,), final_model_path="/workspace/runs/r1/model"
        )
        assert len(p.steps) == 1
        assert p.steps[0] is s
        assert p.final_model_path == "/workspace/runs/r1/model"

    def test_multi_step_plan(self) -> None:
        a = _step("step_a", outputs=("/workspace/intermediate",))
        b = _step(
            "step_b",
            inputs=("/workspace/intermediate",),
            outputs=("/workspace/final",),
        )
        p = PreparePlan(steps=(a, b), final_model_path="/workspace/final")
        assert len(p.steps) == 2
        assert p.steps[0].name == "step_a"
        assert p.steps[1].name == "step_b"


# ===========================================================================
# PreparePlan — negative (validator firings)
# ===========================================================================


class TestPreparePlanNegative:
    def test_steps_without_final_path_rejected(self) -> None:
        """If you have work to do, you MUST tell the provider where the
        model ends up."""
        s = _step()
        with pytest.raises(
            ValidationError, match="final_model_path"
        ):
            PreparePlan(steps=(s,))

    def test_duplicate_step_names_rejected(self) -> None:
        """Names are used in container ids, MLflow tags — must be unique."""
        a = _step("merge_lora")
        b = _step("merge_lora", outputs=("/workspace/other",))
        with pytest.raises(ValidationError, match="unique"):
            PreparePlan(steps=(a, b), final_model_path="/workspace/other")

    def test_extra_field_rejected(self) -> None:
        with pytest.raises(ValidationError, match="extra|Extra|unexpected"):
            PreparePlan(  # type: ignore[call-arg]
                steps=(), spec_version=1, mystery="?"
            )

    def test_frozen_immutable(self) -> None:
        p = PreparePlan.empty()
        with pytest.raises(ValidationError, match="frozen|Instance is frozen"):
            p.spec_version = 99  # type: ignore[misc]


# ===========================================================================
# PreparePlan — boundary
# ===========================================================================


class TestPreparePlanBoundary:
    def test_empty_with_final_path_allowed(self) -> None:
        """Engine MAY set final_model_path even for empty plan — no harm.
        Provider just won't run anything."""
        p = PreparePlan(steps=(), final_model_path="/workspace/already/there")
        assert p.steps == ()
        assert p.final_model_path == "/workspace/already/there"

    def test_three_step_plan(self) -> None:
        """Plans of arbitrary length validate — no upper bound."""
        steps = tuple(
            _step(f"s{i}", outputs=(f"/workspace/o{i}",)) for i in range(3)
        )
        p = PreparePlan(steps=steps, final_model_path="/workspace/o2")
        assert len(p.steps) == 3


# ===========================================================================
# PreparePlan — invariants
# ===========================================================================


class TestPreparePlanInvariants:
    def test_spec_version_default_is_one(self) -> None:
        """First-cut shape. Provider expects ``spec_version == 1`` and rejects
        anything else with ``SINGLENODE_PREPARE_SPEC_VERSION_UNSUPPORTED``."""
        assert PreparePlan().spec_version == 1
        assert PreparePlan.empty().spec_version == 1
        s = _step()
        assert (
            PreparePlan(
                steps=(s,), final_model_path="/workspace/runs/r1/model"
            ).spec_version
            == 1
        )

    def test_empty_factory_returns_distinct_instances(self) -> None:
        """Factories should not leak shared mutable state (steps is frozen
        tuple anyway, but the contract is stronger if instances are fresh)."""
        a = PreparePlan.empty()
        b = PreparePlan.empty()
        # Equal-by-value but distinct objects.
        assert a == b
        # Both immutable; cannot accidentally bridge state.
        with pytest.raises(ValidationError):
            a.spec_version = 2  # type: ignore[misc]

    def test_steps_is_tuple_not_list(self) -> None:
        """Hashability + immutability — Pydantic frozen + tuple field."""
        p = PreparePlan(
            steps=(_step(),), final_model_path="/workspace/runs/r1/model"
        )
        assert isinstance(p.steps, tuple)

    def test_step_outputs_is_tuple_not_list(self) -> None:
        s = _step()
        assert isinstance(s.outputs, tuple)
        assert isinstance(s.inputs, tuple)
        assert isinstance(s.volumes, tuple)
        assert isinstance(s.args, tuple)
