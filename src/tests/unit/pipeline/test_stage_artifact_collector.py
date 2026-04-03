"""
Unit tests for src/pipeline/artifacts/base.py

Covers:
  - StageArtifactEnvelope: to_dict / from_dict, invariants
  - StageArtifactCollector: put / append / flush* lifecycle, is_flushed invariant
  - save_stage_artifact: dependency errors, best-effort behaviour
  - utc_now_iso: basic sanity
  - Schemas TypedDict structural checks
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.pipeline.artifacts.base import (
    STATUS_FAILED,
    STATUS_INTERRUPTED,
    STATUS_PASSED,
    STATUS_SKIPPED,
    StageArtifactCollector,
    StageArtifactEnvelope,
    save_stage_artifact,
    utc_now_iso,
)
from src.pipeline.artifacts.schemas import (
    DeploymentArtifactData,
    EvalArtifactData,
    EvalPluginData,
    InferenceArtifactData,
    ModelArtifactData,
    TrainingArtifactData,
    ValidationArtifactData,
    ValidationDatasetData,
    ValidationPluginData,
)

# =============================================================================
# Helpers
# =============================================================================

def _make_envelope(**kwargs: Any) -> StageArtifactEnvelope:
    defaults: dict[str, Any] = {
        "stage": "test_stage",
        "status": STATUS_PASSED,
        "started_at": "2026-01-15T10:00:00",
        "duration_seconds": 42.5,
        "error": None,
        "data": {"key": "value"},
    }
    defaults.update(kwargs)
    return StageArtifactEnvelope(**defaults)


def _make_collector(stage: str = "test_stage") -> StageArtifactCollector:
    return StageArtifactCollector(stage=stage, artifact_name=f"{stage}_results.json")


# =============================================================================
# StageArtifactEnvelope
# =============================================================================

class TestStageArtifactEnvelope:
    # ----- Positive -----

    def test_to_dict_contains_all_fields(self) -> None:
        env = _make_envelope()
        d = env.to_dict()
        assert d["stage"] == "test_stage"
        assert d["status"] == STATUS_PASSED
        assert d["started_at"] == "2026-01-15T10:00:00"
        assert d["duration_seconds"] == 42.5
        assert d["error"] is None
        assert d["data"] == {"key": "value"}

    def test_from_dict_round_trip(self) -> None:
        original = _make_envelope(data={"x": 1, "y": [1, 2, 3]})
        reconstructed = StageArtifactEnvelope.from_dict(original.to_dict())
        assert reconstructed == original

    def test_from_dict_with_error_field(self) -> None:
        env = StageArtifactEnvelope.from_dict(
            {"stage": "s", "status": STATUS_FAILED, "started_at": "t", "duration_seconds": 1.0, "error": "boom", "data": {}}
        )
        assert env.error == "boom"
        assert env.status == STATUS_FAILED

    # ----- Boundary -----

    def test_from_dict_missing_keys_use_defaults(self) -> None:
        env = StageArtifactEnvelope.from_dict({})
        assert env.stage == ""
        assert env.status == STATUS_PASSED
        assert env.started_at == ""
        assert env.duration_seconds == 0.0
        assert env.error is None
        assert env.data == {}

    def test_from_dict_negative_duration(self) -> None:
        env = StageArtifactEnvelope.from_dict({"duration_seconds": -1.0})
        assert env.duration_seconds == -1.0  # preserves value

    def test_to_dict_data_is_deep_copy_via_asdict(self) -> None:
        env = _make_envelope(data={"nested": {"a": 1}})
        d = env.to_dict()
        # mutating the returned dict should not affect the envelope
        d["data"]["nested"]["a"] = 999
        assert env.data["nested"]["a"] == 1  # type: ignore[index]

    def test_all_statuses_round_trip(self) -> None:
        for status in (STATUS_PASSED, STATUS_FAILED, STATUS_SKIPPED, STATUS_INTERRUPTED):
            env = _make_envelope(status=status)
            assert StageArtifactEnvelope.from_dict(env.to_dict()).status == status

    # ----- Invariants -----

    def test_to_dict_is_json_serializable(self) -> None:
        env = _make_envelope(data={"score": 0.95, "tags": ["a", "b"]})
        payload = json.dumps(env.to_dict())
        assert isinstance(payload, str)
        assert "test_stage" in payload

    def test_from_dict_accepts_extra_keys_without_crash(self) -> None:
        d = {"stage": "s", "status": "passed", "started_at": "t", "duration_seconds": 0.0,
             "error": None, "data": {}, "extra_field_unknown": True}
        env = StageArtifactEnvelope.from_dict(d)
        assert env.stage == "s"


# =============================================================================
# StageArtifactCollector
# =============================================================================

class TestStageArtifactCollectorInit:
    def test_initial_state(self) -> None:
        c = _make_collector()
        assert c.is_flushed is False
        assert c.artifact_name == "test_stage_results.json"

    def test_set_started_at(self) -> None:
        c = _make_collector()
        c.set_started_at("2026-01-15T10:00:00")
        assert c._started_at == "2026-01-15T10:00:00"


class TestStageArtifactCollectorPutAppend:
    # ----- Positive -----

    def test_put_adds_keys(self) -> None:
        c = _make_collector()
        c.put(score=0.9, count=5)
        assert c._data["score"] == 0.9
        assert c._data["count"] == 5

    def test_put_overwrites_existing_key(self) -> None:
        c = _make_collector()
        c.put(score=0.1)
        c.put(score=0.9)
        assert c._data["score"] == 0.9

    def test_append_creates_list_on_first_use(self) -> None:
        c = _make_collector()
        c.append("items", {"name": "a"})
        assert c._data["items"] == [{"name": "a"}]

    def test_append_accumulates_items(self) -> None:
        c = _make_collector()
        c.append("items", 1)
        c.append("items", 2)
        c.append("items", 3)
        assert c._data["items"] == [1, 2, 3]

    # ----- Boundary -----

    def test_put_with_none_values(self) -> None:
        c = _make_collector()
        c.put(x=None, y=None)
        assert c._data["x"] is None

    def test_append_mixed_types(self) -> None:
        c = _make_collector()
        c.append("bag", 1)
        c.append("bag", "str")
        c.append("bag", {"k": "v"})
        assert len(c._data["bag"]) == 3


class TestStageArtifactCollectorFlushOk:
    def test_flush_ok_returns_envelope(self) -> None:
        c = _make_collector("gpu_deployer")
        c.put(upload_duration_seconds=33.4)
        env = c.flush_ok(started_at="2026-01-01T00:00:00", duration_seconds=100.0, context={})
        assert env is not None
        assert env.status == STATUS_PASSED
        assert env.stage == "gpu_deployer"
        assert env.duration_seconds == 100.0
        assert env.data["upload_duration_seconds"] == 33.4
        assert env.error is None

    def test_flush_ok_marks_as_flushed(self) -> None:
        c = _make_collector()
        c.flush_ok(started_at="t", duration_seconds=0.0, context={})
        assert c.is_flushed is True

    def test_flush_ok_rounds_duration(self) -> None:
        c = _make_collector()
        env = c.flush_ok(started_at="t", duration_seconds=1.23456789, context={})
        assert env is not None
        assert env.duration_seconds == 1.235


class TestStageArtifactCollectorFlushError:
    def test_flush_error_returns_failed_envelope(self) -> None:
        c = _make_collector()
        env = c.flush_error(error="something broke", started_at="t", duration_seconds=5.0, context={})
        assert env is not None
        assert env.status == STATUS_FAILED
        assert env.error == "something broke"

    def test_flush_error_marks_as_flushed(self) -> None:
        c = _make_collector()
        c.flush_error(error="err", started_at="t", duration_seconds=0.0, context={})
        assert c.is_flushed is True


class TestStageArtifactCollectorFlushInterrupted:
    def test_flush_interrupted_status(self) -> None:
        c = _make_collector()
        env = c.flush_interrupted(started_at="t", duration_seconds=3.0, context={})
        assert env is not None
        assert env.status == STATUS_INTERRUPTED
        assert env.error is None

    def test_flush_interrupted_marks_as_flushed(self) -> None:
        c = _make_collector()
        c.flush_interrupted(started_at="t", duration_seconds=0.0, context={})
        assert c.is_flushed is True


class TestStageArtifactCollectorFlushSkipped:
    def test_flush_skipped_status_and_zero_duration(self) -> None:
        c = _make_collector()
        env = c.flush_skipped(started_at="t", context={})
        assert env is not None
        assert env.status == STATUS_SKIPPED
        assert env.duration_seconds == 0.0
        assert env.data == {}


class TestStageArtifactCollectorIsFlushedInvariant:
    """Invariant: after flush, further flushes return None and keep state."""

    @pytest.mark.parametrize("first_flush,second_flush", [
        ("ok", "ok"),
        ("ok", "error"),
        ("ok", "interrupted"),
        ("ok", "skipped"),
        ("error", "ok"),
        ("error", "error"),
        ("interrupted", "ok"),
        ("skipped", "ok"),
    ])
    def test_double_flush_returns_none(self, first_flush: str, second_flush: str) -> None:
        c = _make_collector()
        self._do_flush(c, first_flush)
        result = self._do_flush(c, second_flush)
        assert result is None

    @staticmethod
    def _do_flush(c: StageArtifactCollector, method: str) -> StageArtifactEnvelope | None:
        if method == "ok":
            return c.flush_ok(started_at="t", duration_seconds=0.0, context={})
        elif method == "error":
            return c.flush_error(error="e", started_at="t", duration_seconds=0.0, context={})
        elif method == "interrupted":
            return c.flush_interrupted(started_at="t", duration_seconds=0.0, context={})
        else:
            return c.flush_skipped(started_at="t", context={})

    def test_is_flushed_false_before_any_flush(self) -> None:
        c = _make_collector()
        assert c.is_flushed is False

    def test_is_flushed_true_after_ok(self) -> None:
        c = _make_collector()
        c.flush_ok(started_at="t", duration_seconds=0.0, context={})
        assert c.is_flushed is True

    def test_put_after_flush_does_not_change_envelope(self) -> None:
        """put() after flush does not affect written artifact."""
        c = _make_collector()
        c.put(score=1.0)
        env = c.flush_ok(started_at="t", duration_seconds=0.0, context={})
        assert env is not None
        assert env.data["score"] == 1.0
        # Put after flush — internal dict changes but envelope was already created
        c.put(extra="extra")
        # is_flushed remains True
        assert c.is_flushed is True


# =============================================================================
# save_stage_artifact
# =============================================================================

class TestSaveStageArtifact:
    def test_no_mlflow_manager_is_silent(self) -> None:
        """No MLflowManager → no crash, no-op."""
        env = _make_envelope()
        save_stage_artifact({}, env, "test.json")  # must not raise

    def test_inactive_mlflow_is_silent(self) -> None:
        """Inactive MLflowManager does not write artifact."""
        from src.pipeline.stages.constants import PipelineContextKeys
        from src.training.managers.mlflow_manager import MLflowManager

        mock_mgr = MagicMock(spec=MLflowManager)
        mock_mgr.is_active = False
        context = {
            PipelineContextKeys.MLFLOW_MANAGER: mock_mgr,
            PipelineContextKeys.MLFLOW_PARENT_RUN_ID: "rid",
        }
        env = _make_envelope()
        save_stage_artifact(context, env, "test.json")
        mock_mgr.log_artifact.assert_not_called()

    def test_no_run_id_is_silent(self) -> None:
        """MLflowManager present but run_id missing → does not write artifact."""
        from src.pipeline.stages.constants import PipelineContextKeys
        from src.training.managers.mlflow_manager import MLflowManager

        mock_mgr = MagicMock(spec=MLflowManager)
        mock_mgr.is_active = True
        context = {PipelineContextKeys.MLFLOW_MANAGER: mock_mgr}  # no run_id
        env = _make_envelope()
        save_stage_artifact(context, env, "test.json")
        mock_mgr.log_artifact.assert_not_called()

    def test_mlflow_write_and_cleanup_temp_file(self, tmp_path: Path) -> None:
        """Happy path: log_artifact called; temp file removed after write."""
        from src.pipeline.stages.constants import PipelineContextKeys
        from src.training.managers.mlflow_manager import MLflowManager

        logged_args: list[tuple[Any, ...]] = []

        mock_mgr = MagicMock(spec=MLflowManager)
        mock_mgr.is_active = True

        def capture_log(path: str, *, artifact_path: str = "", run_id: str = "") -> None:
            logged_args.append((path, artifact_path, run_id))

        mock_mgr.log_artifact.side_effect = capture_log

        context = {
            PipelineContextKeys.MLFLOW_MANAGER: mock_mgr,
            PipelineContextKeys.MLFLOW_PARENT_RUN_ID: "run_abc",
        }
        env = _make_envelope(data={"score": 0.99})
        save_stage_artifact(context, env, "my_artifact.json")

        assert mock_mgr.log_artifact.call_count == 1
        call_path = logged_args[0][0]
        # The file passed to log_artifact MUST have the exact artifact name
        # (no random prefix from NamedTemporaryFile).
        assert Path(call_path).name == "my_artifact.json"
        # temp file should be deleted after log
        assert not Path(call_path).exists()

    def test_exception_in_log_artifact_does_not_propagate(self) -> None:
        """Dependency error: exception in log_artifact is swallowed (best-effort)."""
        from src.pipeline.stages.constants import PipelineContextKeys
        from src.training.managers.mlflow_manager import MLflowManager

        mock_mgr = MagicMock(spec=MLflowManager)
        mock_mgr.is_active = True
        mock_mgr.log_artifact.side_effect = RuntimeError("MLflow server down")

        context = {
            PipelineContextKeys.MLFLOW_MANAGER: mock_mgr,
            PipelineContextKeys.MLFLOW_PARENT_RUN_ID: "run_abc",
        }
        env = _make_envelope()
        # must not raise
        save_stage_artifact(context, env, "test.json")

    def test_artifact_path_passed_through(self) -> None:
        """artifact_path argument forwarded to log_artifact."""
        from src.pipeline.stages.constants import PipelineContextKeys
        from src.training.managers.mlflow_manager import MLflowManager

        logged_kwargs: list[dict[str, Any]] = []

        mock_mgr = MagicMock(spec=MLflowManager)
        mock_mgr.is_active = True
        mock_mgr.log_artifact.side_effect = lambda path, **kw: logged_kwargs.append(kw)

        context = {
            PipelineContextKeys.MLFLOW_MANAGER: mock_mgr,
            PipelineContextKeys.MLFLOW_PARENT_RUN_ID: "rid",
        }
        save_stage_artifact(context, _make_envelope(), "eval.json", artifact_path="evaluation")
        assert logged_kwargs[0]["artifact_path"] == "evaluation"

    def test_empty_run_id_string_is_skipped(self) -> None:
        """Boundary: run_id == '' → does not write artifact."""
        from src.pipeline.stages.constants import PipelineContextKeys
        from src.training.managers.mlflow_manager import MLflowManager

        mock_mgr = MagicMock(spec=MLflowManager)
        mock_mgr.is_active = True
        context = {
            PipelineContextKeys.MLFLOW_MANAGER: mock_mgr,
            PipelineContextKeys.MLFLOW_PARENT_RUN_ID: "",
        }
        save_stage_artifact(context, _make_envelope(), "test.json")
        mock_mgr.log_artifact.assert_not_called()


# =============================================================================
# utc_now_iso
# =============================================================================

class TestUtcNowIso:
    def test_returns_string(self) -> None:
        result = utc_now_iso()
        assert isinstance(result, str)

    def test_is_valid_iso_format(self) -> None:
        from datetime import datetime
        result = utc_now_iso()
        # Should parse without error
        parsed = datetime.fromisoformat(result)
        assert parsed.year >= 2026


# =============================================================================
# Schemas TypedDict
# =============================================================================

class TestSchemasStructure:
    """TypedDict schemas contain expected keys (structural invariants)."""

    def test_validation_plugin_data_keys(self) -> None:
        data: ValidationPluginData = {
            "name": "test",
            "passed": True,
            "duration_ms": 1.0,
            "description": "d",
            "metrics": {},
            "params": {},
            "errors": [],
            "recommendations": [],
        }
        assert data["name"] == "test"
        assert data["passed"] is True

    def test_validation_dataset_data_keys(self) -> None:
        data: ValidationDatasetData = {
            "name": "ds",
            "path": "/tmp/x",
            "sample_count": 10,
            "status": "passed",
            "critical_failures": 0,
            "plugins": [],
        }
        assert data["path"] == "/tmp/x"

    def test_validation_artifact_data_keys(self) -> None:
        data: ValidationArtifactData = {"datasets": []}
        assert "datasets" in data

    def test_eval_plugin_data_keys(self) -> None:
        data: EvalPluginData = {
            "passed": True,
            "metrics": {"score": 0.9},
            "errors": [],
            "recommendations": [],
            "sample_count": 5,
            "failed_samples": 0,
        }
        assert data["metrics"]["score"] == 0.9

    def test_eval_artifact_data_keys(self) -> None:
        data: EvalArtifactData = {
            "overall_passed": True,
            "sample_count": 10,
            "duration_seconds": 3.5,
            "skipped_plugins": [],
            "errors": [],
            "plugins": {},
        }
        assert data["sample_count"] == 10

    def test_deployment_artifact_data_keys(self) -> None:
        data: DeploymentArtifactData = {
            "upload_duration_seconds": 33.4,
            "deps_duration_seconds": 7.1,
            "provider_name": "runpod",
            "provider_type": "cloud",
            "gpu_type": "A100",
            "resource_id": "pod123",
        }
        assert data["provider_name"] == "runpod"

    def test_training_artifact_data_keys(self) -> None:
        data: TrainingArtifactData = {"training_duration_seconds": 3600.0}
        assert data["training_duration_seconds"] == 3600.0

    def test_model_artifact_data_keys(self) -> None:
        data: ModelArtifactData = {
            "model_size_mb": 1500.0,
            "hf_repo_id": "org/model",
            "upload_duration_seconds": 60.0,
        }
        assert data["hf_repo_id"] == "org/model"

    def test_inference_artifact_data_keys(self) -> None:
        data: InferenceArtifactData = {
            "endpoint_url": "http://localhost:8000/v1",
            "model_name": "test-model",
            "provider": "single_node",
        }
        assert data["endpoint_url"] == "http://localhost:8000/v1"

    def test_all_nullable_fields_accept_none(self) -> None:
        dep: DeploymentArtifactData = {
            "upload_duration_seconds": None,
            "deps_duration_seconds": None,
            "provider_name": None,
            "provider_type": None,
            "gpu_type": None,
            "resource_id": None,
        }
        assert dep["upload_duration_seconds"] is None

        inf: InferenceArtifactData = {"endpoint_url": None, "model_name": None, "provider": None}
        assert inf["model_name"] is None


# =============================================================================
# Combinatorial collector tests
# =============================================================================

class TestCollectorCombinatorial:
    @pytest.mark.parametrize("status,flush_method", [
        (STATUS_PASSED, "ok"),
        (STATUS_FAILED, "error"),
        (STATUS_INTERRUPTED, "interrupted"),
        (STATUS_SKIPPED, "skipped"),
    ])
    def test_each_flush_method_produces_correct_status(self, status: str, flush_method: str) -> None:
        c = _make_collector()
        if flush_method == "ok":
            env = c.flush_ok(started_at="t", duration_seconds=1.0, context={})
        elif flush_method == "error":
            env = c.flush_error(error="err", started_at="t", duration_seconds=1.0, context={})
        elif flush_method == "interrupted":
            env = c.flush_interrupted(started_at="t", duration_seconds=1.0, context={})
        else:
            env = c.flush_skipped(started_at="t", context={})

        assert env is not None
        assert env.status == status

    @pytest.mark.parametrize("duration", [0.0, 0.001, 1.0, 3600.0, 86400.0])
    def test_various_durations_are_preserved(self, duration: float) -> None:
        c = _make_collector()
        env = c.flush_ok(started_at="t", duration_seconds=duration, context={})
        assert env is not None
        # Duration is rounded to 3 decimal places
        assert abs(env.duration_seconds - round(duration, 3)) < 1e-9

    def test_put_multiple_then_flush_all_data_preserved(self) -> None:
        c = _make_collector()
        c.put(a=1, b=2, c=3)
        c.put(d=4)  # additional call
        env = c.flush_ok(started_at="t", duration_seconds=0.0, context={})
        assert env is not None
        assert env.data == {"a": 1, "b": 2, "c": 3, "d": 4}

    def test_append_then_put_then_flush(self) -> None:
        c = _make_collector()
        c.append("items", "x")
        c.append("items", "y")
        c.put(total=2)
        env = c.flush_ok(started_at="t", duration_seconds=0.0, context={})
        assert env is not None
        assert env.data["items"] == ["x", "y"]
        assert env.data["total"] == 2


# =============================================================================
# Regressions
# =============================================================================

class TestArtifactRegressions:
    def test_envelope_to_dict_data_contains_nested_list(self) -> None:
        """Regression: lists in data serialize to JSON correctly."""
        env = _make_envelope(data={"plugins": [{"name": "a", "passed": True}]})
        payload = json.dumps(env.to_dict())
        parsed = json.loads(payload)
        assert parsed["data"]["plugins"][0]["name"] == "a"

    def test_collector_stage_name_propagated_to_envelope(self) -> None:
        """Invariant: stage name in collector matches envelope."""
        c = StageArtifactCollector(stage="dataset_validator", artifact_name="x.json")
        env = c.flush_ok(started_at="t", duration_seconds=0.0, context={})
        assert env is not None
        assert env.stage == "dataset_validator"

    def test_collector_artifact_name_readable(self) -> None:
        """Invariant: artifact_name returned unchanged."""
        c = StageArtifactCollector(stage="s", artifact_name="evaluation_results.json")
        assert c.artifact_name == "evaluation_results.json"

    def test_envelope_from_dict_coerces_duration_to_float(self) -> None:
        """Boundary: duration_seconds int in JSON → float on object."""
        env = StageArtifactEnvelope.from_dict(
            {"stage": "s", "status": "passed", "started_at": "t", "duration_seconds": 100, "error": None, "data": {}}
        )
        assert isinstance(env.duration_seconds, float)
        assert env.duration_seconds == 100.0

    def test_save_stage_artifact_json_payload_matches_envelope(self) -> None:
        """Regression: written JSON payload matches envelope.to_dict()."""
        from src.pipeline.stages.constants import PipelineContextKeys
        from src.training.managers.mlflow_manager import MLflowManager

        written_payloads: list[str] = []

        mock_mgr = MagicMock(spec=MLflowManager)
        mock_mgr.is_active = True

        def capture(path: str, **kwargs: Any) -> None:
            written_payloads.append(Path(path).read_text(encoding="utf-8"))

        mock_mgr.log_artifact.side_effect = capture

        context = {
            PipelineContextKeys.MLFLOW_MANAGER: mock_mgr,
            PipelineContextKeys.MLFLOW_PARENT_RUN_ID: "rid",
        }
        env = _make_envelope(data={"score": 0.75, "count": 10})
        save_stage_artifact(context, env, "results.json")

        assert len(written_payloads) == 1
        parsed = json.loads(written_payloads[0])
        assert parsed == env.to_dict()
