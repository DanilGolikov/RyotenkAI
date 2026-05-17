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
from ryotenkai_shared.errors import InferenceUnavailableError, ProviderUnavailableError

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
def fake_docker():
    """Canonical :class:`IDockerClient` fake — see ``tests/_fakes/docker.py``."""
    from tests._fakes.docker import FakeDockerClient

    return FakeDockerClient()


@pytest.fixture()
def provider(provider_cfg, engine_cfg, secrets, fake_docker):
    """Build a real ``SingleNodeInferenceProvider`` via ``ProviderContext`` —
    the legitimate constructor; matches how the registry instantiates it
    in production.

    The provider is injected with :class:`FakeDockerClient` so tests
    control container state, log content, exit codes, etc. via the
    fake's chaos surface rather than monkey-patching module-level
    docker functions.
    """
    from ryotenkai_providers.registry import ProviderContext

    pipeline_cfg = _mk_pipeline_config(provider_cfg, engine_cfg)
    ctx = ProviderContext(
        provider_id="single_node",
        pipeline_config=pipeline_cfg,
        provider_block=provider_cfg.model_dump(mode="python"),
        secrets=secrets,
    )
    p = SingleNodeInferenceProvider(ctx, docker=fake_docker)
    p._run_id = "run_test"
    return p


def _seed_prepare_step(
    fake_docker,
    *,
    run_id: str = "r1",
    step_name: str = "merge_lora",
    exit_code: int = 0,
    logs: str = "MERGE_SUCCESS",
    initially_running: bool = False,
) -> str:
    """Seed the fake with a container in the post-run state expected by
    ``_run_prepare_plan``'s polling loop.

    The plan-runner polls ``is_container_running`` until it returns
    False, then reads logs + exit code. A pre-seeded "already exited"
    container short-circuits the polling loop on the first iteration,
    keeping tests fast and deterministic.
    """
    container_name = f"helix-prepare-{run_id}-{step_name}"
    state = "running" if initially_running else "exited"
    fake_docker.register_container(
        container_name, state=state, exit_code=exit_code,
    )
    if logs:
        fake_docker.append_logs(container_name, logs)
    return container_name


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
        # Phase A2 Batch 12: _run_prepare_plan returns None on success, raises on failure.
        provider._run_prepare_plan(
            ssh=mock_ssh,
            plan=PreparePlan.empty(),
            run_id="r1",
            workspace_host_path="/host/ws/inference",
        )
        # No image pulls, no container runs.
        mock_ssh.exec_command.assert_not_called()

    def test_single_step_success(self, provider, fake_docker) -> None:
        plan = _plan(_step())
        mock_ssh = MagicMock()
        mock_ssh.exec_command.return_value = (True, "OK", "")
        _seed_prepare_step(
            fake_docker, logs="Some output\nMERGE_SUCCESS\n",
        )
        with patch.object(provider, "_ensure_docker_image", return_value=None):
            provider._run_prepare_plan(
                ssh=mock_ssh,
                plan=plan,
                run_id="r1",
                workspace_host_path="/host/ws/inference",
            )

    def test_two_step_plan_runs_both(self, provider, fake_docker) -> None:
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
        _seed_prepare_step(fake_docker, step_name="step_a")
        _seed_prepare_step(fake_docker, step_name="step_b")
        with patch.object(provider, "_ensure_docker_image", return_value=None):
            provider._run_prepare_plan(
                ssh=mock_ssh,
                plan=plan,
                run_id="r1",
                workspace_host_path="/host/ws/inference",
            )


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
        with pytest.raises(InferenceUnavailableError) as exc_info:
            provider._run_prepare_plan(
                ssh=MagicMock(),
                plan=plan,
                run_id="r1",
                workspace_host_path="/host/ws/inference",
            )
        assert exc_info.value.context.get("reason") == "SINGLENODE_PREPARE_SPEC_VERSION_UNSUPPORTED"

    def test_image_pull_fails(self, provider) -> None:
        plan = _plan(_step())
        with patch.object(
            provider,
            "_ensure_docker_image",
            side_effect=InferenceUnavailableError(
                detail="pull fail",
                context={"reason": "X"},
            ),
        ):
            with pytest.raises(InferenceUnavailableError) as exc_info:
                provider._run_prepare_plan(
                    ssh=MagicMock(),
                    plan=plan,
                    run_id="r1",
                    workspace_host_path="/host/ws/inference",
                )
        assert exc_info.value.context.get("reason") == "SINGLENODE_PREPARE_IMAGE_PULL_FAILED"

    def test_container_start_fails(self, provider) -> None:
        plan = _plan(_step())
        mock_ssh = MagicMock()
        # First exec_command (rm/mkdir cleanup) succeeds; the format_prepare_step
        # docker run call is the next exec_command — return failure.
        mock_ssh.exec_command.side_effect = [
            (True, "", ""),  # rm -rf && mkdir -p
            (False, "", "docker daemon not reachable"),  # docker run fail
        ]
        with patch.object(provider, "_ensure_docker_image", return_value=None):
            with pytest.raises(InferenceUnavailableError) as exc_info:
                provider._run_prepare_plan(
                    ssh=mock_ssh,
                    plan=plan,
                    run_id="r1",
                    workspace_host_path="/host/ws/inference",
                )
        assert exc_info.value.context.get("reason") == "SINGLENODE_PREPARE_CONTAINER_START_FAILED"

    def test_exit_code_nonzero(self, provider, fake_docker) -> None:
        plan = _plan(_step())
        mock_ssh = MagicMock()
        mock_ssh.exec_command.return_value = (True, "", "")
        _seed_prepare_step(fake_docker, exit_code=1)
        with patch.object(provider, "_ensure_docker_image", return_value=None):
            with pytest.raises(InferenceUnavailableError) as exc_info:
                provider._run_prepare_plan(
                    ssh=mock_ssh,
                    plan=plan,
                    run_id="r1",
                    workspace_host_path="/host/ws/inference",
                )
        assert exc_info.value.context.get("reason") == "SINGLENODE_PREPARE_CONTAINER_FAILED"

    def test_no_success_marker_in_logs(self, provider, fake_docker) -> None:
        plan = _plan(_step())
        mock_ssh = MagicMock()
        mock_ssh.exec_command.return_value = (True, "", "")
        _seed_prepare_step(fake_docker, logs="nothing notable here")
        with patch.object(provider, "_ensure_docker_image", return_value=None):
            with pytest.raises(InferenceUnavailableError) as exc_info:
                provider._run_prepare_plan(
                    ssh=mock_ssh,
                    plan=plan,
                    run_id="r1",
                    workspace_host_path="/host/ws/inference",
                )
        assert exc_info.value.context.get("reason") == "SINGLENODE_PREPARE_NO_SUCCESS_MARKER"

    def test_artifacts_not_found(self, provider, fake_docker) -> None:
        plan = _plan(_step())
        mock_ssh = MagicMock()
        # rm/mkdir, docker run, then artifact verify ⇒ MISSING.
        mock_ssh.exec_command.side_effect = [
            (True, "", ""),  # rm/mkdir
            (True, "", ""),  # docker run
            (True, "MISSING", ""),  # test -f artifact
            (True, "(directory listing)", ""),  # ls -lah
        ]
        _seed_prepare_step(fake_docker)
        with patch.object(provider, "_ensure_docker_image", return_value=None):
            with pytest.raises(InferenceUnavailableError) as exc_info:
                provider._run_prepare_plan(
                    ssh=mock_ssh,
                    plan=plan,
                    run_id="r1",
                    workspace_host_path="/host/ws/inference",
                )
        assert exc_info.value.context.get("reason") == "SINGLENODE_PREPARE_ARTIFACTS_NOT_FOUND"


# ===========================================================================
# Boundary — special-shape steps
# ===========================================================================


class TestBoundary:
    def test_no_marker_no_artifact_passes_with_zero_exit(
        self, provider, fake_docker,
    ) -> None:
        """A step may have no marker and no artifact ⇒ exit code is the
        sole success criterion."""
        plan = _plan(
            _step(success_marker=None, success_artifact=None)
        )
        mock_ssh = MagicMock()
        mock_ssh.exec_command.return_value = (True, "", "")
        _seed_prepare_step(fake_docker, logs="any output")
        with patch.object(provider, "_ensure_docker_image", return_value=None):
            provider._run_prepare_plan(
                ssh=mock_ssh,
                plan=plan,
                run_id="r1",
                workspace_host_path="/host/ws/inference",
            )

    def test_step_with_explicit_image_overrides_serve_image(
        self, provider, fake_docker,
    ) -> None:
        """``step.image`` set ⇒ provider uses it instead of engine serve image."""
        plan = _plan(_step(image="custom/converter:1.0", success_artifact=None))
        mock_ssh = MagicMock()
        mock_ssh.exec_command.return_value = (True, "", "")
        _seed_prepare_step(fake_docker)

        ensure_calls = []

        def _capture(*, ssh, image):
            ensure_calls.append(image)
            return None

        with patch.object(provider, "_ensure_docker_image", side_effect=_capture):
            provider._run_prepare_plan(
                ssh=mock_ssh,
                plan=plan,
                run_id="r1",
                workspace_host_path="/host/ws/inference",
            )

        assert ensure_calls == ["custom/converter:1.0"]

    def test_step_with_image_none_uses_engine_serve_image(
        self, provider, fake_docker,
    ) -> None:
        """``image=None`` ⇒ provider falls through to ``_resolve_engine_image``."""
        plan = _plan(_step(image=None, success_artifact=None))
        mock_ssh = MagicMock()
        mock_ssh.exec_command.return_value = (True, "", "")
        _seed_prepare_step(fake_docker)

        ensure_calls = []

        def _capture(*, ssh, image):
            ensure_calls.append(image)
            return None

        with patch.object(provider, "_ensure_docker_image", side_effect=_capture):
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
    def test_output_dirs_cleaned_before_run(self, provider, fake_docker) -> None:
        """Idempotency invariant: rm -rf + mkdir -p before docker run."""
        plan = _plan(
            _step(
                outputs=("/workspace/o1", "/workspace/o2"),
                success_artifact=None,  # simplifies mock
            )
        )
        mock_ssh = MagicMock()
        mock_ssh.exec_command.return_value = (True, "", "")
        _seed_prepare_step(fake_docker)
        with patch.object(provider, "_ensure_docker_image", return_value=None):
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

    def test_container_removed_after_run(self, provider, fake_docker) -> None:
        """``docker rm -f`` is always called post-step (success or failure)."""
        plan = _plan(_step(success_artifact=None))
        mock_ssh = MagicMock()
        mock_ssh.exec_command.return_value = (True, "", "")
        _seed_prepare_step(fake_docker)
        with patch.object(provider, "_ensure_docker_image", return_value=None):
            provider._run_prepare_plan(
                ssh=mock_ssh,
                plan=plan,
                run_id="r1",
                workspace_host_path="/host/ws/inference",
            )
        # The fake records every call; ``rm_force`` must appear at least once.
        assert len(fake_docker.calls_for("rm_force")) >= 1


# ===========================================================================
# Logic-specific — MLflow events + container naming
# ===========================================================================


class TestLogicSpecific:
    def test_no_mlflow_log_event_calls_in_phase_7(self, provider, fake_docker) -> None:
        """Phase 7: per-step ``log_event_*`` calls were retired. The
        provider no longer touches the (removed) MLflow event-log API;
        prepare lifecycle is observable via the typed event journal in
        future iterations. Here we assert the calls are absent.
        """
        mlflow = MagicMock()
        # No log_event_* methods exist on the post-Phase-7 manager — make
        # the mock raise if they're accessed, to catch regressions.
        for name in (
            "log_event",
            "log_event_start",
            "log_event_complete",
            "log_event_error",
            "log_event_warning",
            "log_event_info",
            "log_event_checkpoint",
        ):
            setattr(
                type(mlflow),
                name,
                property(lambda _self, _n=name: (_ for _ in ()).throw(
                    AttributeError(f"Phase 7: {_n!r} removed"),
                )),
            )
        provider._mlflow_manager = mlflow
        plan = _plan(_step(success_artifact=None))
        mock_ssh = MagicMock()
        mock_ssh.exec_command.return_value = (True, "", "")
        _seed_prepare_step(fake_docker)
        with patch.object(provider, "_ensure_docker_image", return_value=None):
            # Should not raise — adapter must not call log_event_* anymore.
            provider._run_prepare_plan(
                ssh=mock_ssh,
                plan=plan,
                run_id="r1",
                workspace_host_path="/host/ws/inference",
            )

    def test_container_name_uses_run_id_and_step_name(
        self, provider, fake_docker,
    ) -> None:
        plan = _plan(_step("custom_step", success_artifact=None))
        mock_ssh = MagicMock()
        captured_cmds: list[str] = []

        def _capture(cmd, *a, **k):
            captured_cmds.append(cmd)
            return (True, "", "")

        mock_ssh.exec_command.side_effect = _capture
        _seed_prepare_step(fake_docker, step_name="custom_step")
        with patch.object(provider, "_ensure_docker_image", return_value=None):
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
    def test_second_step_failure_aborts_chain(self, provider, fake_docker) -> None:
        """When step B fails, plan returns Err immediately. Step A's outputs
        are NOT cleaned up (no rollback)."""
        b = _step(
            "step_b",
            outputs=("/workspace/b_out",),
            success_marker="B_OK",
            success_artifact=None,
        )

        mock_ssh = MagicMock()
        mock_ssh.exec_command.return_value = (True, "", "")

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

        # Step A succeeds (no marker required); step B's logs do NOT
        # contain ``B_OK`` ⇒ NO_SUCCESS_MARKER.
        _seed_prepare_step(fake_docker, step_name="step_a", logs="A_DONE_NOTHING")
        _seed_prepare_step(fake_docker, step_name="step_b", logs="no marker present")

        with patch.object(provider, "_ensure_docker_image", return_value=None):
            with pytest.raises(InferenceUnavailableError) as exc_info:
                provider._run_prepare_plan(
                    ssh=mock_ssh,
                    plan=plan,
                    run_id="r1",
                    workspace_host_path="/host/ws/inference",
                )

        # Step A succeeded (no marker required) → step B failed (B_OK missing).
        assert exc_info.value.context.get("reason") == "SINGLENODE_PREPARE_NO_SUCCESS_MARKER"


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
