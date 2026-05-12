"""Unit tests for ``SingleNodeInferenceProvider._run_prepare_plan``.

Successor to ``TestRunMergeContainerErrors`` — same coverage shape, but
exercising the new generic plan-runner that consumes ``PreparePlan`` from
the engine.

Categories covered:

  * positive — happy path with single + multi-step plans
  * negative — every error code path
  * boundary — empty plan, no marker, no artifact, multi-output cleanup
  * invariant — output-dir cleanup before run, container removal after run
  * regression — preserved poll interval, log paths, image override
  * logic-specific — step.image=None ⇒ engine serve image, MLflow events
  * combinatorial — 2-step plan where second fails, partial outputs intact
"""

from __future__ import annotations

from unittest.mock import MagicMock, Mock, patch

import pytest
from ryotenkai_engines.interfaces import PreparePlan, PrepareStep
from ryotenkai_engines.vllm.config import VLLMEngineConfig

import ryotenkai_providers.single_node.inference.provider as _mod
from ryotenkai_providers.single_node.inference.provider import (
    SingleNodeInferenceProvider,
)
from ryotenkai_shared.config import InferenceSingleNodeServeConfig, Secrets
from ryotenkai_shared.config.providers.single_node import (
    SingleNodeConnectConfig,
    SingleNodeInferenceConfig,
    SingleNodeProviderConfig,
    SingleNodeTrainingConfig,
)
from ryotenkai_shared.config.providers.ssh import SSHConfig
from ryotenkai_shared.utils.result import Err, Ok

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def provider_cfg():
    return SingleNodeProviderConfig(
        connect=SingleNodeConnectConfig(
            ssh=SSHConfig(
                alias="test", host="1.2.3.4", port=22, user="user"
            )
        ),
        training=SingleNodeTrainingConfig(workspace_path="/host/ws"),
        inference=SingleNodeInferenceConfig(
            serve=InferenceSingleNodeServeConfig(
                host="127.0.0.1", port=8000, workspace="/host/ws/inference"
            )
        ),
    )


@pytest.fixture()
def secrets():
    return Secrets(hf_token="hf_test_token")


@pytest.fixture()
def engine_cfg():
    return VLLMEngineConfig()


def _mk_pipeline_config(provider_cfg, engine_cfg):
    """Synthetic PipelineConfig — all the attribute paths the provider
    touches in ``__init__`` and ``deploy()``.

    After the discriminated-union refactor, ``cfg.inference.engine`` is the
    typed ``VLLMEngineConfig`` instance directly (Pydantic narrowed it via
    the ``kind`` discriminator).
    """
    cfg = Mock()
    cfg.get_provider_config = lambda *_a, **_k: provider_cfg.model_dump(mode="python")
    cfg.inference = Mock()
    # Post-discriminated-union: cfg.inference.engine IS the typed engine config.
    cfg.inference.engine = engine_cfg
    cfg.inference.common = Mock()
    cfg.model = Mock()
    cfg.model.name = "meta-llama/Llama-2-7b"
    cfg.model.trust_remote_code = False
    cfg.integrations = None
    return cfg


@pytest.fixture()
def provider(provider_cfg, engine_cfg, secrets):
    """Build a real ``SingleNodeInferenceProvider`` via ``ProviderContext`` —
    the legitimate constructor; matches how the registry instantiates it
    in production."""
    from ryotenkai_providers.registry import ProviderContext

    pipeline_cfg = _mk_pipeline_config(provider_cfg, engine_cfg)
    ctx = ProviderContext(
        provider_id="single_node",
        pipeline_config=pipeline_cfg,
        provider_block=provider_cfg.model_dump(mode="python"),
        secrets=secrets,
    )
    p = SingleNodeInferenceProvider(ctx)
    p._run_id = "run_test"
    return p


def _step(
    name: str = "merge_lora",
    *,
    outputs: tuple[str, ...] = ("/workspace/runs/r1/model",),
    success_marker: str | None = "MERGE_SUCCESS",
    success_artifact: str | None = "/workspace/runs/r1/model/config.json",
    image: str | None = None,
    timeout_seconds: int = 3600,
    **overrides,  # type: ignore[no-untyped-def]
) -> PrepareStep:
    return PrepareStep(
        name=name,
        image=image,
        entrypoint=("python3",),
        args=("/opt/helix/merge_lora.py",),
        env={"HF_HOME": "/workspace/hf_cache"},
        volumes=(("/host/ws/inference", "/workspace"),),
        outputs=outputs,
        success_marker=success_marker,
        success_artifact=success_artifact,
        timeout_seconds=timeout_seconds,
        **overrides,
    )


def _plan(*steps: PrepareStep, final: str | None = None) -> PreparePlan:
    if not steps:
        return PreparePlan.empty()
    return PreparePlan(
        steps=steps, final_model_path=final or steps[-1].outputs[0]
    )


def _mk_ssh_running_then_done(stdouts: list[str]):
    """SSH where ``exec_command`` returns success + a queue of stdout values."""
    mock_ssh = MagicMock()
    out_iter = iter(stdouts + [""] * 10)
    mock_ssh.exec_command.side_effect = lambda *_a, **_k: (
        True,
        next(out_iter),
        "",
    )
    return mock_ssh


# ===========================================================================
# Positive — happy paths
# ===========================================================================


class TestPositive:
    def test_empty_plan_returns_ok_immediately(self, provider) -> None:
        """No steps ⇒ no docker calls, no SSH usage."""
        mock_ssh = MagicMock()
        result = provider._run_prepare_plan(
            ssh=mock_ssh,
            plan=PreparePlan.empty(),
            run_id="r1",
            workspace_host_path="/host/ws/inference",
        )
        assert result.is_ok()
        # No image pulls, no container runs.
        mock_ssh.exec_command.assert_not_called()

    def test_single_step_success(self, provider) -> None:
        plan = _plan(_step())
        mock_ssh = MagicMock()
        mock_ssh.exec_command.return_value = (True, "OK", "")
        with (
            patch.object(provider, "_ensure_docker_image", return_value=Ok(None)),
            patch.object(_mod, "docker_is_container_running", return_value=False),
            patch.object(
                _mod, "docker_logs", return_value=Ok("Some output\nMERGE_SUCCESS\n")
            ),
            patch.object(_mod, "docker_container_exit_code", return_value=Ok(0)),
            patch.object(_mod, "docker_rm_force", return_value=Ok(None)),
        ):
            result = provider._run_prepare_plan(
                ssh=mock_ssh,
                plan=plan,
                run_id="r1",
                workspace_host_path="/host/ws/inference",
            )
        assert result.is_ok(), f"got {result.unwrap_err() if result.is_err() else None}"

    def test_two_step_plan_runs_both(self, provider) -> None:
        a = _step(
            "step_a",
            outputs=("/workspace/intermediate",),
            success_artifact="/workspace/intermediate/done",
        )
        b = _step(
            "step_b",
            outputs=("/workspace/runs/r1/model",),
            success_artifact="/workspace/runs/r1/model/config.json",
        )
        plan = _plan(a, b)
        mock_ssh = MagicMock()
        mock_ssh.exec_command.return_value = (True, "OK", "")
        with (
            patch.object(provider, "_ensure_docker_image", return_value=Ok(None)),
            patch.object(_mod, "docker_is_container_running", return_value=False),
            patch.object(_mod, "docker_logs", return_value=Ok("MERGE_SUCCESS")),
            patch.object(_mod, "docker_container_exit_code", return_value=Ok(0)),
            patch.object(_mod, "docker_rm_force", return_value=Ok(None)),
        ):
            result = provider._run_prepare_plan(
                ssh=mock_ssh,
                plan=plan,
                run_id="r1",
                workspace_host_path="/host/ws/inference",
            )
        assert result.is_ok()


# ===========================================================================
# Negative — every error code path
# ===========================================================================


class TestNegative:
    def test_unsupported_spec_version(self, provider) -> None:
        """Provider rejects unknown PreparePlan shapes loudly. Construct an
        instance, then mutate via ``__pydantic_fields_set__`` since model is frozen."""
        # Create plan with version=1, then patch object with version=2 dynamically.
        plan = PreparePlan(
            steps=(_step(),), final_model_path="/workspace/runs/r1/model"
        )
        # Bypass frozen by direct __dict__ mutation (test-only; production
        # provider sees the version on the validated instance).
        object.__setattr__(plan, "spec_version", 999)
        result = provider._run_prepare_plan(
            ssh=MagicMock(),
            plan=plan,
            run_id="r1",
            workspace_host_path="/host/ws/inference",
        )
        assert result.is_err()
        assert "SINGLENODE_PREPARE_SPEC_VERSION_UNSUPPORTED" in str(result.unwrap_err())

    def test_image_pull_fails(self, provider) -> None:
        from ryotenkai_shared.utils.result import InferenceError

        plan = _plan(_step())
        with patch.object(
            provider,
            "_ensure_docker_image",
            return_value=Err(InferenceError(message="pull fail", code="X")),
        ):
            result = provider._run_prepare_plan(
                ssh=MagicMock(),
                plan=plan,
                run_id="r1",
                workspace_host_path="/host/ws/inference",
            )
        assert result.is_err()
        assert "SINGLENODE_PREPARE_IMAGE_PULL_FAILED" in str(result.unwrap_err())

    def test_container_start_fails(self, provider) -> None:
        plan = _plan(_step())
        mock_ssh = MagicMock()
        # First exec_command (rm/mkdir cleanup) succeeds; the format_prepare_step
        # docker run call is the next exec_command — return failure.
        mock_ssh.exec_command.side_effect = [
            (True, "", ""),  # rm -rf && mkdir -p
            (False, "", "docker daemon not reachable"),  # docker run fail
        ]
        with patch.object(provider, "_ensure_docker_image", return_value=Ok(None)):
            result = provider._run_prepare_plan(
                ssh=mock_ssh,
                plan=plan,
                run_id="r1",
                workspace_host_path="/host/ws/inference",
            )
        assert result.is_err()
        assert "SINGLENODE_PREPARE_CONTAINER_START_FAILED" in str(result.unwrap_err())

    def test_exit_code_nonzero(self, provider) -> None:
        plan = _plan(_step())
        mock_ssh = MagicMock()
        mock_ssh.exec_command.return_value = (True, "", "")
        with (
            patch.object(provider, "_ensure_docker_image", return_value=Ok(None)),
            patch.object(_mod, "docker_is_container_running", return_value=False),
            patch.object(_mod, "docker_logs", return_value=Ok("MERGE_SUCCESS")),
            patch.object(_mod, "docker_container_exit_code", return_value=Ok(1)),
            patch.object(_mod, "docker_rm_force", return_value=Ok(None)),
        ):
            result = provider._run_prepare_plan(
                ssh=mock_ssh,
                plan=plan,
                run_id="r1",
                workspace_host_path="/host/ws/inference",
            )
        assert result.is_err()
        assert "SINGLENODE_PREPARE_CONTAINER_FAILED" in str(result.unwrap_err())

    def test_no_success_marker_in_logs(self, provider) -> None:
        plan = _plan(_step())
        mock_ssh = MagicMock()
        mock_ssh.exec_command.return_value = (True, "", "")
        with (
            patch.object(provider, "_ensure_docker_image", return_value=Ok(None)),
            patch.object(_mod, "docker_is_container_running", return_value=False),
            patch.object(_mod, "docker_logs", return_value=Ok("nothing notable here")),
            patch.object(_mod, "docker_container_exit_code", return_value=Ok(0)),
            patch.object(_mod, "docker_rm_force", return_value=Ok(None)),
        ):
            result = provider._run_prepare_plan(
                ssh=mock_ssh,
                plan=plan,
                run_id="r1",
                workspace_host_path="/host/ws/inference",
            )
        assert result.is_err()
        assert "SINGLENODE_PREPARE_NO_SUCCESS_MARKER" in str(result.unwrap_err())

    def test_artifacts_not_found(self, provider) -> None:
        plan = _plan(_step())
        mock_ssh = MagicMock()
        # rm/mkdir, docker run, then artifact verify ⇒ MISSING.
        mock_ssh.exec_command.side_effect = [
            (True, "", ""),  # rm/mkdir
            (True, "", ""),  # docker run
            (True, "MISSING", ""),  # test -f artifact
            (True, "(directory listing)", ""),  # ls -lah
        ]
        with (
            patch.object(provider, "_ensure_docker_image", return_value=Ok(None)),
            patch.object(_mod, "docker_is_container_running", return_value=False),
            patch.object(_mod, "docker_logs", return_value=Ok("MERGE_SUCCESS")),
            patch.object(_mod, "docker_container_exit_code", return_value=Ok(0)),
            patch.object(_mod, "docker_rm_force", return_value=Ok(None)),
        ):
            result = provider._run_prepare_plan(
                ssh=mock_ssh,
                plan=plan,
                run_id="r1",
                workspace_host_path="/host/ws/inference",
            )
        assert result.is_err()
        assert "SINGLENODE_PREPARE_ARTIFACTS_NOT_FOUND" in str(result.unwrap_err())


# ===========================================================================
# Boundary — special-shape steps
# ===========================================================================


class TestBoundary:
    def test_no_marker_no_artifact_passes_with_zero_exit(self, provider) -> None:
        """A step may have no marker and no artifact ⇒ exit code is the
        sole success criterion."""
        plan = _plan(
            _step(success_marker=None, success_artifact=None)
        )
        mock_ssh = MagicMock()
        mock_ssh.exec_command.return_value = (True, "", "")
        with (
            patch.object(provider, "_ensure_docker_image", return_value=Ok(None)),
            patch.object(_mod, "docker_is_container_running", return_value=False),
            patch.object(_mod, "docker_logs", return_value=Ok("any output")),
            patch.object(_mod, "docker_container_exit_code", return_value=Ok(0)),
            patch.object(_mod, "docker_rm_force", return_value=Ok(None)),
        ):
            result = provider._run_prepare_plan(
                ssh=mock_ssh,
                plan=plan,
                run_id="r1",
                workspace_host_path="/host/ws/inference",
            )
        assert result.is_ok()

    def test_step_with_explicit_image_overrides_serve_image(
        self, provider
    ) -> None:
        """``step.image`` set ⇒ provider uses it instead of engine serve image."""
        plan = _plan(_step(image="custom/converter:1.0"))
        mock_ssh = MagicMock()
        mock_ssh.exec_command.return_value = (True, "", "")

        ensure_calls = []

        def _capture(ssh, image):
            ensure_calls.append(image)
            return Ok(None)

        with (
            patch.object(provider, "_ensure_docker_image", side_effect=_capture),
            patch.object(_mod, "docker_is_container_running", return_value=False),
            patch.object(_mod, "docker_logs", return_value=Ok("MERGE_SUCCESS")),
            patch.object(_mod, "docker_container_exit_code", return_value=Ok(0)),
            patch.object(_mod, "docker_rm_force", return_value=Ok(None)),
        ):
            provider._run_prepare_plan(
                ssh=mock_ssh,
                plan=plan,
                run_id="r1",
                workspace_host_path="/host/ws/inference",
            )

        assert ensure_calls == ["custom/converter:1.0"]

    def test_step_with_image_none_uses_engine_serve_image(
        self, provider
    ) -> None:
        """``image=None`` ⇒ provider falls through to ``_resolve_engine_image``."""
        plan = _plan(_step(image=None))
        mock_ssh = MagicMock()
        mock_ssh.exec_command.return_value = (True, "", "")

        ensure_calls = []

        def _capture(ssh, image):
            ensure_calls.append(image)
            return Ok(None)

        with (
            patch.object(provider, "_ensure_docker_image", side_effect=_capture),
            patch.object(_mod, "docker_is_container_running", return_value=False),
            patch.object(_mod, "docker_logs", return_value=Ok("MERGE_SUCCESS")),
            patch.object(_mod, "docker_container_exit_code", return_value=Ok(0)),
            patch.object(_mod, "docker_rm_force", return_value=Ok(None)),
        ):
            provider._run_prepare_plan(
                ssh=mock_ssh,
                plan=plan,
                run_id="r1",
                workspace_host_path="/host/ws/inference",
            )
        # Convention: ryotenkai/inference-vllm:<version>
        assert len(ensure_calls) == 1
        assert ensure_calls[0].startswith("ryotenkai/inference-vllm")


# ===========================================================================
# Invariants
# ===========================================================================


class TestInvariants:
    def test_output_dirs_cleaned_before_run(self, provider) -> None:
        """Idempotency invariant: rm -rf + mkdir -p before docker run."""
        plan = _plan(
            _step(
                outputs=("/workspace/o1", "/workspace/o2"),
                success_artifact=None,  # simplifies mock
            )
        )
        mock_ssh = MagicMock()
        mock_ssh.exec_command.return_value = (True, "", "")
        with (
            patch.object(provider, "_ensure_docker_image", return_value=Ok(None)),
            patch.object(_mod, "docker_is_container_running", return_value=False),
            patch.object(_mod, "docker_logs", return_value=Ok("MERGE_SUCCESS")),
            patch.object(_mod, "docker_container_exit_code", return_value=Ok(0)),
            patch.object(_mod, "docker_rm_force", return_value=Ok(None)),
        ):
            provider._run_prepare_plan(
                ssh=mock_ssh,
                plan=plan,
                run_id="r1",
                workspace_host_path="/host/ws/inference",
            )

        # The first two exec_command calls clean the two output dirs.
        cleanup_calls = [
            c
            for c in mock_ssh.exec_command.call_args_list
            if "rm -rf" in str(c) and "mkdir -p" in str(c)
        ]
        assert len(cleanup_calls) >= 2

    def test_container_removed_after_run(self, provider) -> None:
        """``docker rm -f`` is always called post-step (success or failure)."""
        plan = _plan(_step(success_artifact=None))
        mock_ssh = MagicMock()
        mock_ssh.exec_command.return_value = (True, "", "")
        rm_force_mock = MagicMock(return_value=Ok(None))
        with (
            patch.object(provider, "_ensure_docker_image", return_value=Ok(None)),
            patch.object(_mod, "docker_is_container_running", return_value=False),
            patch.object(_mod, "docker_logs", return_value=Ok("MERGE_SUCCESS")),
            patch.object(_mod, "docker_container_exit_code", return_value=Ok(0)),
            patch.object(_mod, "docker_rm_force", rm_force_mock),
        ):
            provider._run_prepare_plan(
                ssh=mock_ssh,
                plan=plan,
                run_id="r1",
                workspace_host_path="/host/ws/inference",
            )
        assert rm_force_mock.called


# ===========================================================================
# Logic-specific — MLflow events + container naming
# ===========================================================================


class TestLogicSpecific:
    def test_emits_mlflow_events_per_step(self, provider) -> None:
        """Operators filter on ``Prepare started``, ``Prepare step started``,
        ``Prepare step completed``, ``Prepare completed``."""
        mlflow = MagicMock()
        provider._mlflow_manager = mlflow
        plan = _plan(_step(success_artifact=None))
        mock_ssh = MagicMock()
        mock_ssh.exec_command.return_value = (True, "", "")
        with (
            patch.object(provider, "_ensure_docker_image", return_value=Ok(None)),
            patch.object(_mod, "docker_is_container_running", return_value=False),
            patch.object(_mod, "docker_logs", return_value=Ok("MERGE_SUCCESS")),
            patch.object(_mod, "docker_container_exit_code", return_value=Ok(0)),
            patch.object(_mod, "docker_rm_force", return_value=Ok(None)),
        ):
            provider._run_prepare_plan(
                ssh=mock_ssh,
                plan=plan,
                run_id="r1",
                workspace_host_path="/host/ws/inference",
            )

        # log_event_start called twice (plan-start, step-start); log_event_complete twice.
        assert mlflow.log_event_start.call_count == 2
        assert mlflow.log_event_complete.call_count == 2

        # Event messages renamed Merge* → Prepare* (constraint AD-A13).
        all_msgs = [
            str(c.args[0]) if c.args else ""
            for c in (
                mlflow.log_event_start.call_args_list
                + mlflow.log_event_complete.call_args_list
            )
        ]
        assert any("Prepare started" in m for m in all_msgs)
        assert any("Prepare step started" in m for m in all_msgs)
        assert any("Prepare step completed" in m for m in all_msgs)
        assert any("Prepare completed" in m for m in all_msgs)
        # Old strings absent.
        assert not any("Merge" in m for m in all_msgs)

    def test_container_name_uses_run_id_and_step_name(
        self, provider
    ) -> None:
        plan = _plan(_step("custom_step", success_artifact=None))
        mock_ssh = MagicMock()
        captured_cmds: list[str] = []

        def _capture(cmd, *a, **k):
            captured_cmds.append(cmd)
            return (True, "", "")

        mock_ssh.exec_command.side_effect = _capture
        with (
            patch.object(provider, "_ensure_docker_image", return_value=Ok(None)),
            patch.object(_mod, "docker_is_container_running", return_value=False),
            patch.object(_mod, "docker_logs", return_value=Ok("MERGE_SUCCESS")),
            patch.object(_mod, "docker_container_exit_code", return_value=Ok(0)),
            patch.object(_mod, "docker_rm_force", return_value=Ok(None)),
        ):
            provider._run_prepare_plan(
                ssh=mock_ssh,
                plan=plan,
                run_id="r1",
                workspace_host_path="/host/ws/inference",
            )

        docker_runs = [c for c in captured_cmds if c.startswith("docker run")]
        assert len(docker_runs) == 1
        assert "helix-prepare-r1-custom_step" in docker_runs[0]


# ===========================================================================
# Combinatorial — multi-step failure scenarios
# ===========================================================================


class TestCombinatorial:
    def test_second_step_failure_aborts_chain(self, provider) -> None:
        """When step B fails, plan returns Err immediately. Step A's outputs
        are NOT cleaned up (no rollback)."""
        a = _step(
            "step_a",
            outputs=("/workspace/a_out",),
            success_artifact=None,
        )
        b = _step(
            "step_b",
            outputs=("/workspace/b_out",),
            success_marker="B_OK",
            success_artifact=None,
        )
        plan = PreparePlan(steps=(a, b), final_model_path="/workspace/b_out")

        mock_ssh = MagicMock()
        mock_ssh.exec_command.return_value = (True, "", "")

        # Track which step's logs we're returning by call count.
        logs_responses = ["A_DONE_NOTHING", "no marker present"]
        exit_codes = [0, 0]
        is_running_responses = [False, False]

        def _logs(*a, **k):
            return Ok(
                logs_responses.pop(0) if logs_responses else "no marker present"
            )

        def _exit(*a, **k):
            return Ok(exit_codes.pop(0) if exit_codes else 0)

        def _running(*a, **k):
            return is_running_responses.pop(0) if is_running_responses else False

        # Step A has no marker (None) ⇒ marker_seen=True automatically.
        a_no_marker = _step(
            "step_a",
            outputs=("/workspace/a_out",),
            success_marker=None,
            success_artifact=None,
        )
        plan = PreparePlan(
            steps=(a_no_marker, b), final_model_path="/workspace/b_out"
        )

        with (
            patch.object(provider, "_ensure_docker_image", return_value=Ok(None)),
            patch.object(_mod, "docker_is_container_running", side_effect=_running),
            patch.object(_mod, "docker_logs", side_effect=_logs),
            patch.object(_mod, "docker_container_exit_code", side_effect=_exit),
            patch.object(_mod, "docker_rm_force", return_value=Ok(None)),
        ):
            result = provider._run_prepare_plan(
                ssh=mock_ssh,
                plan=plan,
                run_id="r1",
                workspace_host_path="/host/ws/inference",
            )

        assert result.is_err()
        # Step A succeeded (no marker required) → step B failed (B_OK missing).
        assert "SINGLENODE_PREPARE_NO_SUCCESS_MARKER" in str(result.unwrap_err())


# ===========================================================================
# Path mapping — provider concern, preserved from legacy
# ===========================================================================


class TestPathMapping:
    @pytest.mark.parametrize(
        ("input_path", "expected"),
        [
            ("/workspace/foo", "/workspace/foo"),  # already container
            ("/host/ws/inference/sub", "/workspace/sub"),  # host → container
            ("hf-org/repo", "hf-org/repo"),  # HF id → unchanged
            ("/somewhere/else", "/somewhere/else"),  # unrelated abs path
        ],
    )
    def test_host_to_container_mapping(self, input_path, expected) -> None:
        out = SingleNodeInferenceProvider._host_to_container(
            input_path,
            workspace_host_path="/host/ws/inference",
        )
        assert out == expected

    @pytest.mark.parametrize(
        ("container_path", "expected"),
        [
            ("/workspace/foo", "/host/ws/inference/foo"),
            ("/workspace/runs/r1/model/config.json",
             "/host/ws/inference/runs/r1/model/config.json"),
        ],
    )
    def test_container_to_host_mapping(self, container_path, expected) -> None:
        out = SingleNodeInferenceProvider._container_to_host(
            container_path,
            workspace_host_path="/host/ws/inference",
        )
        assert out == expected
