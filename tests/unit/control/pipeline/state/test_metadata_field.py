"""Tests for the additive ``metadata`` field on PipelineState/PipelineAttemptState.

Variant 1 Step 2 — caller-provided tags propagated to MLflow as
``meta.*`` by the orchestrator.

Test categories: positive, negative, boundary, invariants, regression,
logic-specific.
"""

from __future__ import annotations

from ryotenkai_control.pipeline.mlflow.lifecycle._tag_utils import (
    stringify_tag_value as _stringify_tag_value,
)
from ryotenkai_control.pipeline.state.models import PipelineAttemptState, PipelineState, StageRunState

# ---------------------------------------------------------------------------
# 1. Positive — round-trip serialisation
# ---------------------------------------------------------------------------


class TestPositive:
    def test_pipeline_state_metadata_roundtrip(self) -> None:
        state = PipelineState(
            schema_version=1,
            logical_run_id="run-1",
            run_directory="/tmp/run",
            config_path="/tmp/config.yaml",
            active_attempt_id=None,
            pipeline_status=StageRunState.STATUS_PENDING,
            training_critical_config_hash="abc",
            late_stage_config_hash="def",
            metadata={
                "project_id": "helixql-v7",
                "actor": "agent:claude-code",
                "config_version_hash": "sha256:xyz",
            },
        )
        d = state.to_dict()
        assert d["metadata"] == {
            "project_id": "helixql-v7",
            "actor": "agent:claude-code",
            "config_version_hash": "sha256:xyz",
        }
        restored = PipelineState.from_dict(d)
        assert restored.metadata == state.metadata

    def test_pipeline_attempt_state_metadata_roundtrip(self) -> None:
        attempt = PipelineAttemptState(
            attempt_id="att-1",
            attempt_no=1,
            runtime_name="single_node",
            requested_action="fresh",
            effective_action="fresh",
            restart_from_stage=None,
            status=StageRunState.STATUS_PENDING,
            started_at="2026-04-28T05:00:00Z",
            metadata={"attempt_actor": "human"},
        )
        d = attempt.to_dict()
        assert d["metadata"] == {"attempt_actor": "human"}
        restored = PipelineAttemptState.from_dict(d)
        assert restored.metadata == attempt.metadata


# ---------------------------------------------------------------------------
# 2. Negative — invalid types in stored metadata
# ---------------------------------------------------------------------------


class TestNegative:
    def test_non_dict_metadata_in_payload_falls_back_to_empty(self) -> None:
        # Defensive: ``from_dict`` ignores non-dict ``metadata`` values
        # to keep deserialisation robust against hand-edited state files.
        d = {
            "schema_version": 1,
            "logical_run_id": "x",
            "run_directory": "/tmp",
            "config_path": "/tmp/c.yaml",
            "active_attempt_id": None,
            "pipeline_status": StageRunState.STATUS_PENDING,
            "training_critical_config_hash": "",
            "late_stage_config_hash": "",
            "metadata": "not a dict",  # garbage
        }
        state = PipelineState.from_dict(d)
        assert state.metadata == {}

    def test_attempt_non_dict_metadata_falls_back_to_empty(self) -> None:
        d = {
            "attempt_id": "a",
            "attempt_no": 1,
            "runtime_name": "x",
            "requested_action": "fresh",
            "effective_action": "fresh",
            "restart_from_stage": None,
            "status": StageRunState.STATUS_PENDING,
            "started_at": "",
            "metadata": ["a", "b"],  # garbage
        }
        attempt = PipelineAttemptState.from_dict(d)
        assert attempt.metadata == {}


# ---------------------------------------------------------------------------
# 3. Boundary — empty / null / huge values
# ---------------------------------------------------------------------------


class TestBoundary:
    def test_default_metadata_is_empty_dict(self) -> None:
        state = PipelineState(
            schema_version=1,
            logical_run_id="x",
            run_directory="/tmp",
            config_path="/tmp/c.yaml",
            active_attempt_id=None,
            pipeline_status=StageRunState.STATUS_PENDING,
            training_critical_config_hash="",
            late_stage_config_hash="",
        )
        assert state.metadata == {}

    def test_empty_metadata_is_omitted_from_serialised_output(self) -> None:
        # Pin: the JSON omits ``metadata`` entirely when empty so legacy
        # state-file diff tools don't see noise in the output.
        state = PipelineState(
            schema_version=1,
            logical_run_id="x",
            run_directory="/tmp",
            config_path="/tmp/c.yaml",
            active_attempt_id=None,
            pipeline_status=StageRunState.STATUS_PENDING,
            training_critical_config_hash="",
            late_stage_config_hash="",
        )
        d = state.to_dict()
        assert "metadata" not in d

    def test_large_value_truncates_to_under_mlflow_limit(self) -> None:
        # Pin: values longer than the MLflow tag length cap get truncated
        # with an ellipsis suffix, not silently dropped.
        long_value = "x" * 10000
        out = _stringify_tag_value(long_value)
        assert len(out) <= 5000
        assert out.endswith("…")
        assert out.startswith("xxxx")

    def test_non_string_value_coerced_via_str(self) -> None:
        assert _stringify_tag_value(123) == "123"
        assert _stringify_tag_value(None) == "None"
        assert _stringify_tag_value([1, 2, 3]) == "[1, 2, 3]"


# ---------------------------------------------------------------------------
# 4. Invariants
# ---------------------------------------------------------------------------


class TestInvariants:
    def test_metadata_independent_per_instance(self) -> None:
        # Pin: default_factory yields a fresh dict per instance,
        # not a shared mutable default.
        a = PipelineState(
            schema_version=1,
            logical_run_id="A",
            run_directory="/tmp/a",
            config_path="/tmp/c.yaml",
            active_attempt_id=None,
            pipeline_status=StageRunState.STATUS_PENDING,
            training_critical_config_hash="",
            late_stage_config_hash="",
        )
        b = PipelineState(
            schema_version=1,
            logical_run_id="B",
            run_directory="/tmp/b",
            config_path="/tmp/c.yaml",
            active_attempt_id=None,
            pipeline_status=StageRunState.STATUS_PENDING,
            training_critical_config_hash="",
            late_stage_config_hash="",
        )
        a.metadata["k"] = "v"
        assert b.metadata == {}

    def test_serialise_deserialise_idempotent(self) -> None:
        original = PipelineState(
            schema_version=1,
            logical_run_id="x",
            run_directory="/tmp",
            config_path="/tmp/c.yaml",
            active_attempt_id=None,
            pipeline_status=StageRunState.STATUS_PENDING,
            training_critical_config_hash="",
            late_stage_config_hash="",
            metadata={"k": "v"},
        )
        first = PipelineState.from_dict(original.to_dict())
        second = PipelineState.from_dict(first.to_dict())
        assert first.metadata == second.metadata == {"k": "v"}


# ---------------------------------------------------------------------------
# 5. Regression — old state files (no metadata field) keep loading
# ---------------------------------------------------------------------------


class TestRegression:
    def test_legacy_state_file_without_metadata_loads(self) -> None:
        # Old pipeline_state.json files predate the metadata field.
        legacy_dict = {
            "schema_version": 1,
            "logical_run_id": "legacy-run",
            "run_directory": "/tmp/legacy",
            "config_path": "/tmp/c.yaml",
            "active_attempt_id": None,
            "pipeline_status": StageRunState.STATUS_PENDING,
            "training_critical_config_hash": "",
            "late_stage_config_hash": "",
            # No "metadata" key at all.
        }
        state = PipelineState.from_dict(legacy_dict)
        assert state.metadata == {}
        # Round-trip back: still no metadata in output.
        assert "metadata" not in state.to_dict()

    def test_legacy_attempt_without_metadata_loads(self) -> None:
        legacy_dict = {
            "attempt_id": "a",
            "attempt_no": 1,
            "runtime_name": "x",
            "requested_action": "fresh",
            "effective_action": "fresh",
            "restart_from_stage": None,
            "status": StageRunState.STATUS_PENDING,
            "started_at": "",
        }
        attempt = PipelineAttemptState.from_dict(legacy_dict)
        assert attempt.metadata == {}


# ---------------------------------------------------------------------------
# 6. Logic-specific — _stringify_tag_value preserves common values
# ---------------------------------------------------------------------------


class TestLogicSpecific:
    def test_stringify_short_value_passthrough(self) -> None:
        assert _stringify_tag_value("hello") == "hello"

    def test_stringify_value_at_exactly_limit_passthrough(self) -> None:
        # A value exactly at the cap should NOT be truncated.
        from ryotenkai_control.pipeline.mlflow.lifecycle._tag_utils import (
            _MLFLOW_TAG_VALUE_MAX_CHARS,
        )

        v = "x" * _MLFLOW_TAG_VALUE_MAX_CHARS
        out = _stringify_tag_value(v)
        assert out == v
        assert "…" not in out

    def test_stringify_value_one_over_limit_gets_truncated(self) -> None:
        from ryotenkai_control.pipeline.mlflow.lifecycle._tag_utils import (
            _MLFLOW_TAG_VALUE_MAX_CHARS,
        )

        v = "x" * (_MLFLOW_TAG_VALUE_MAX_CHARS + 1)
        out = _stringify_tag_value(v)
        assert len(out) == _MLFLOW_TAG_VALUE_MAX_CHARS + 1  # value + "…"
        assert out.endswith("…")
