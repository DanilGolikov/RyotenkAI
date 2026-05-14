"""``VLLMEngineRuntime.prepare_model`` — engine-side merge plan builder.

Categories: positive, negative, boundary, invariant, dependency-error,
regression (legacy parity), logic-specific (CLI flag generation),
combinatorial.

Pure data-builder tests — no IO, no docker, no SSH. The provider's
``_run_prepare_plan`` is tested separately in providers/tests/.
"""

from __future__ import annotations

import itertools

import pytest

from ryotenkai_engines.interfaces import PreparePlan, PrepareStep
from ryotenkai_engines.vllm.config import VLLMEngineConfig
from ryotenkai_engines.vllm.runtime import (
    _HF_CACHE_DIR_IN_CONTAINER,
    _MERGE_SCRIPT_IN_CONTAINER,
    _MERGE_SUCCESS_MARKER,
    _MERGE_TIMEOUT_S,
    VLLMEngineRuntime,
)
from ryotenkai_shared.errors import EngineConfigInvalidError

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prepare(
    *,
    cfg: VLLMEngineConfig | None = None,
    base_model: str = "meta-llama/Llama-3-8B",
    adapter_path_in_container: str | None = "/workspace/adapters/lora_x",
    workspace_host_path: str = "/host/workspace",
    run_id: str = "run_abcdef",
    trust_remote_code: bool = False,
) -> PreparePlan:
    cfg = cfg or VLLMEngineConfig()
    return VLLMEngineRuntime().prepare_model(
        cfg=cfg,
        base_model=base_model,
        adapter_path_in_container=adapter_path_in_container,
        workspace_host_path=workspace_host_path,
        run_id=run_id,
        trust_remote_code=trust_remote_code,
    )


# ===========================================================================
# Positive — happy paths
# ===========================================================================


class TestPositive:
    def test_single_step_with_adapter(self) -> None:
        plan = _prepare()
        assert isinstance(plan, PreparePlan)
        assert len(plan.steps) == 1
        assert plan.spec_version == 1
        assert plan.final_model_path == "/workspace/runs/run_abcdef/model"

    def test_step_is_named_merge_lora(self) -> None:
        plan = _prepare()
        assert plan.steps[0].name == "merge_lora"

    def test_step_uses_serve_image_by_default(self) -> None:
        """``image=None`` ⇒ provider falls back to engine's serve image."""
        plan = _prepare()
        assert plan.steps[0].image is None

    def test_step_entrypoint_is_python3(self) -> None:
        """vLLM image's default ENTRYPOINT is the inference server, so the
        merge step overrides it."""
        plan = _prepare()
        assert plan.steps[0].entrypoint == ("python3",)

    def test_step_args_contain_required_flags_in_order(self) -> None:
        plan = _prepare(
            base_model="org/base",
            adapter_path_in_container="/workspace/ad",
            run_id="r1",
        )
        args = plan.steps[0].args
        # Script path is the first positional.
        assert args[0] == _MERGE_SCRIPT_IN_CONTAINER
        # Flags appear with paired values in the canonical order.
        assert "--base-model" in args
        assert args[args.index("--base-model") + 1] == "org/base"
        assert "--adapter" in args
        assert args[args.index("--adapter") + 1] == "/workspace/ad"
        assert "--output" in args
        assert (
            args[args.index("--output") + 1] == "/workspace/runs/r1/model"
        )
        assert "--cache-dir" in args
        assert (
            args[args.index("--cache-dir") + 1] == _HF_CACHE_DIR_IN_CONTAINER
        )

    def test_outputs_match_final_model_path(self) -> None:
        """Provider passes ``plan.final_model_path`` to ``build_launch_spec``,
        so it must match the step's first output."""
        plan = _prepare(run_id="r1")
        assert plan.steps[0].outputs == ("/workspace/runs/r1/model",)
        assert plan.final_model_path == plan.steps[0].outputs[0]

    def test_inputs_carry_adapter_path(self) -> None:
        """Provider may verify inputs exist before running."""
        plan = _prepare(adapter_path_in_container="/workspace/adapters/A")
        assert plan.steps[0].inputs == ("/workspace/adapters/A",)

    def test_volumes_mount_workspace(self) -> None:
        plan = _prepare(workspace_host_path="/srv/data/ws")
        assert plan.steps[0].volumes == (("/srv/data/ws", "/workspace"),)

    def test_env_carries_hf_cache_paths(self) -> None:
        plan = _prepare()
        env = plan.steps[0].env
        assert env["HF_HOME"] == _HF_CACHE_DIR_IN_CONTAINER
        assert env["HUGGINGFACE_HUB_CACHE"] == _HF_CACHE_DIR_IN_CONTAINER
        assert env["TRANSFORMERS_CACHE"] == _HF_CACHE_DIR_IN_CONTAINER

    def test_success_marker_and_artifact(self) -> None:
        plan = _prepare(run_id="r1")
        assert plan.steps[0].success_marker == _MERGE_SUCCESS_MARKER
        assert (
            plan.steps[0].success_artifact
            == "/workspace/runs/r1/model/config.json"
        )

    def test_timeout_seconds(self) -> None:
        plan = _prepare()
        assert plan.steps[0].timeout_seconds == _MERGE_TIMEOUT_S


# ===========================================================================
# Negative — error branches
# ===========================================================================


class TestNegative:
    def test_wrong_config_type_raises(self) -> None:
        """Engine guards against dispatch errors. The discriminated union
        SHOULD prevent this in practice (Pydantic narrows by ``kind``),
        but we guard defensively at the engine boundary."""
        from typing import Literal as _Literal

        from ryotenkai_engines.interfaces import BaseEngineConfig

        class NotVLLMConfig(BaseEngineConfig):
            kind: _Literal["other"] = "other"

        with pytest.raises(EngineConfigInvalidError) as exc_info:
            VLLMEngineRuntime().prepare_model(
                cfg=NotVLLMConfig(),
                base_model="x",
                adapter_path_in_container="/workspace/a",
                workspace_host_path="/host",
                run_id="r1",
                trust_remote_code=False,
            )
        err = exc_info.value
        assert err.context["reason"] == "vllm_config_type_mismatch"
        assert err.context["got"] == "NotVLLMConfig"
        assert err.context["expected"] == "VLLMEngineConfig"
        assert err.status == 422
        assert "VLLMEngineConfig" in (err.detail or "")


# ===========================================================================
# Boundary — empty plan branches
# ===========================================================================


class TestBoundary:
    def test_no_adapter_returns_empty_plan(self) -> None:
        plan = _prepare(adapter_path_in_container=None)
        assert plan.steps == ()
        assert plan.final_model_path is None
        assert plan.spec_version == 1

    def test_merge_before_deploy_false_returns_empty_plan(self) -> None:
        """When the engine is configured for live LoRA loading, no merge
        is needed. ``validate_config`` rejects this in MVP, but the
        ``prepare_model`` branch is correct for when the gate lifts."""
        cfg = VLLMEngineConfig(merge_before_deploy=False)
        plan = _prepare(cfg=cfg)
        assert plan.steps == ()
        assert plan.final_model_path is None

    def test_no_adapter_with_live_lora_returns_empty(self) -> None:
        """Both branches simultaneously."""
        cfg = VLLMEngineConfig(merge_before_deploy=False)
        plan = _prepare(cfg=cfg, adapter_path_in_container=None)
        assert plan.steps == ()
        assert plan.final_model_path is None


# ===========================================================================
# Invariants
# ===========================================================================


class TestInvariants:
    def test_returned_plan_validates(self) -> None:
        """Engine-built plans must satisfy PreparePlan's own validators
        (steps ⇒ final_model_path, unique names, …)."""
        plan = _prepare()
        # If construction succeeded, validators passed.
        assert isinstance(plan, PreparePlan)

    def test_plan_is_frozen(self) -> None:
        from pydantic import ValidationError

        plan = _prepare()
        with pytest.raises(ValidationError, match="frozen|Instance is frozen"):
            plan.spec_version = 99  # type: ignore[misc]

    def test_step_is_frozen(self) -> None:
        from pydantic import ValidationError

        plan = _prepare()
        with pytest.raises(ValidationError, match="frozen|Instance is frozen"):
            plan.steps[0].name = "other"  # type: ignore[misc]

    def test_args_are_flat_tuple_no_embedded_separators(self) -> None:
        """Regression: legacy code built a multi-line backslash-continuation
        shell string. Args must be pre-split tokens — provider does shell
        quoting."""
        plan = _prepare()
        for arg in plan.steps[0].args:
            assert "\n" not in arg, f"arg contains newline: {arg!r}"
            assert "\\\n" not in arg, f"arg contains line continuation: {arg!r}"

    def test_args_no_shell_metachars_in_paths(self) -> None:
        """Defensive — engine-supplied paths are not shell-quoted; the
        provider's ``shlex.quote`` handles it. But engine should never emit
        embedded quotes that confuse the provider."""
        plan = _prepare(
            base_model="org/base",
            adapter_path_in_container="/workspace/lora_x",
            run_id="r1",
        )
        for arg in plan.steps[0].args:
            # No literal embedded shell quotes — engine builds raw tokens.
            assert '"' not in arg
            assert "'" not in arg


# ===========================================================================
# Dependency-error — engine has NO IO
# ===========================================================================


class TestNoIO:
    """Sentinel-style: ensure ``prepare_model`` doesn't accidentally
    introduce IO. Full sentinel is ``test_no_io_in_engine_prepare.py``."""

    def test_does_not_import_paramiko_or_subprocess_at_module_level(
        self,
    ) -> None:
        import ryotenkai_engines.vllm.runtime as runtime_mod

        # The runtime module MUST NOT bring in IO libraries at import time.
        for forbidden in ("paramiko", "subprocess"):
            assert not hasattr(runtime_mod, forbidden), (
                f"runtime imported {forbidden} — engines must be IO-free"
            )


# ===========================================================================
# Regression — parity with legacy ``_build_merge_command`` semantics
# ===========================================================================


class TestRegression:
    def test_trust_remote_code_flag_appended_when_true(self) -> None:
        """Legacy: trailing ``--trust-remote-code`` flag iff requested."""
        plan = _prepare(trust_remote_code=True)
        assert "--trust-remote-code" in plan.steps[0].args
        # Always last — engine convention.
        assert plan.steps[0].args[-1] == "--trust-remote-code"

    def test_trust_remote_code_flag_absent_when_false(self) -> None:
        plan = _prepare(trust_remote_code=False)
        assert "--trust-remote-code" not in plan.steps[0].args

    def test_output_path_uses_run_id(self) -> None:
        """Legacy used ``/workspace/runs/{run_id}/model`` as the merge
        output. Preserved verbatim — provider relies on this."""
        for run_id in ("test_run_001", "abc-def_xyz", "r1"):
            plan = _prepare(run_id=run_id)
            assert plan.final_model_path == f"/workspace/runs/{run_id}/model"

    def test_cache_dir_unchanged(self) -> None:
        """Legacy: ``/workspace/hf_cache``."""
        plan = _prepare()
        assert _HF_CACHE_DIR_IN_CONTAINER == "/workspace/hf_cache"
        assert (
            plan.steps[0].args[
                plan.steps[0].args.index("--cache-dir") + 1
            ]
            == "/workspace/hf_cache"
        )

    def test_merge_script_path_unchanged(self) -> None:
        """Legacy: ``/opt/helix/merge_lora.py`` (PR-15 Dockerfile contract)."""
        assert _MERGE_SCRIPT_IN_CONTAINER == "/opt/helix/merge_lora.py"

    def test_success_marker_unchanged(self) -> None:
        """Legacy: provider polled stdout for ``MERGE_SUCCESS``."""
        assert _MERGE_SUCCESS_MARKER == "MERGE_SUCCESS"

    def test_timeout_unchanged(self) -> None:
        """Legacy: ``MERGE_TIMEOUT = 3600`` in provider; now in engine."""
        assert _MERGE_TIMEOUT_S == 3600


# ===========================================================================
# Logic-specific — flag generation rules
# ===========================================================================


class TestLogicSpecific:
    def test_args_have_no_embedded_value_glue(self) -> None:
        """Each value follows its flag as a SEPARATE token. ``--base-model
        org/base`` not ``--base-model=org/base``. Important for shlex.quote
        to quote each token independently."""
        plan = _prepare()
        for flag in ("--base-model", "--adapter", "--output", "--cache-dir"):
            assert flag in plan.steps[0].args, f"{flag} missing"
            # The token immediately after the flag is the value.
            idx = plan.steps[0].args.index(flag)
            assert idx + 1 < len(plan.steps[0].args)
            value = plan.steps[0].args[idx + 1]
            assert not value.startswith("--"), (
                f"{flag} value {value!r} looks like another flag"
            )

    def test_flag_order_is_base_adapter_output_cache(self) -> None:
        """Order matters for the golden snapshot — keep canonical."""
        args = list(_prepare().args if False else _prepare().steps[0].args)
        i_base = args.index("--base-model")
        i_adapter = args.index("--adapter")
        i_output = args.index("--output")
        i_cache = args.index("--cache-dir")
        assert i_base < i_adapter < i_output < i_cache

    def test_run_id_with_special_chars_passes_through(self) -> None:
        """Engine doesn't sanitize run_id — caller's responsibility. But
        the path is built by string concat; we just verify it doesn't
        crash and the value is plumbed."""
        plan = _prepare(run_id="weird/run id")
        assert plan.final_model_path == "/workspace/runs/weird/run id/model"


# ===========================================================================
# Combinatorial — 2³ matrix
# ===========================================================================


class TestCombinatorial:
    @pytest.mark.parametrize(
        ("merge_before_deploy", "adapter_present", "trust_remote_code"),
        list(itertools.product([True, False], [True, False], [True, False])),
    )
    def test_branch_matrix(
        self,
        *,
        merge_before_deploy: bool,
        adapter_present: bool,
        trust_remote_code: bool,
    ) -> None:
        cfg = VLLMEngineConfig(merge_before_deploy=merge_before_deploy)
        adapter = "/workspace/ad" if adapter_present else None
        plan = _prepare(
            cfg=cfg,
            adapter_path_in_container=adapter,
            trust_remote_code=trust_remote_code,
        )
        # Empty plan iff merge is off OR there's no adapter.
        if merge_before_deploy and adapter_present:
            assert len(plan.steps) == 1
            assert plan.final_model_path is not None
            # trust_remote_code flag plumbed correctly.
            if trust_remote_code:
                assert "--trust-remote-code" in plan.steps[0].args
            else:
                assert "--trust-remote-code" not in plan.steps[0].args
        else:
            assert plan.steps == ()
            assert plan.final_model_path is None
