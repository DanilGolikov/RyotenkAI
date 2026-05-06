"""``VLLMEngineRuntime`` — IInferenceEngine compliance + behaviour.

Categories: positive, negative, boundary, invariant, dependency-error,
regression (legacy parity), logic-specific (CLI flag generation),
combinatorial.
"""

from __future__ import annotations

import itertools

import pytest

from ryotenkai_engines.capabilities import EngineCapabilities
from ryotenkai_engines.interfaces import IInferenceEngine, LaunchSpec
from ryotenkai_engines.vllm.config import VLLMEngineConfig
from ryotenkai_engines.vllm.runtime import VLLMEngineRuntime

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build(cfg: VLLMEngineConfig | None = None, **overrides) -> LaunchSpec:  # type: ignore[no-untyped-def]
    cfg = cfg or VLLMEngineConfig()
    return VLLMEngineRuntime().build_launch_spec(
        cfg=cfg,
        image="img:1.0.0",
        container_name="ryo_vllm_test",
        port=8000,
        workspace_host_path="/host/ws",
        model_path_in_container="/workspace/model",
        **overrides,
    )


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------


class TestProtocolCompliance:
    def test_implements_iinferenceengine(self) -> None:
        """Runtime-checkable Protocol — instance MUST satisfy."""
        assert isinstance(VLLMEngineRuntime(), IInferenceEngine)

    def test_engine_id_classvar(self) -> None:
        assert VLLMEngineRuntime.engine_id == "vllm"

    def test_config_class_classvar(self) -> None:
        assert VLLMEngineRuntime.config_class is VLLMEngineConfig

    def test_zero_arg_constructible(self) -> None:
        """Registry calls ``runtime_cls()`` — must work without args."""
        VLLMEngineRuntime()


# ---------------------------------------------------------------------------
# get_capabilities
# ---------------------------------------------------------------------------


class TestCapabilities:
    def test_capabilities_match_manifest_shape(self) -> None:
        caps = VLLMEngineRuntime().get_capabilities()
        assert isinstance(caps, EngineCapabilities)
        assert caps.api_dialect == "openai_compatible"
        assert caps.supports_lora is True
        assert caps.supports_quantization is True
        assert caps.supports_streaming is True
        assert caps.supports_tensor_parallel is True
        assert "awq" in caps.supported_quantizations
        assert "bfloat16" in caps.supported_dtypes
        assert caps.default_port == 8000

    def test_capabilities_frozen(self) -> None:
        caps = VLLMEngineRuntime().get_capabilities()
        with pytest.raises(Exception):  # noqa: BLE001 — Pydantic ValidationError
            caps.api_dialect = "custom"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# build_launch_spec — happy path
# ---------------------------------------------------------------------------


class TestLaunchSpecPositive:
    def test_default_launch_spec(self) -> None:
        spec = _build()
        assert spec.image == "img:1.0.0"
        assert spec.container_name == "ryo_vllm_test"
        assert spec.port == 8000
        assert ("/host/ws", "/workspace") in spec.volumes
        # First-positional args: ``serve <model_path>``.
        assert spec.args[0] == "serve"
        assert spec.args[1] == "/workspace/model"

    def test_default_includes_required_flags(self) -> None:
        spec = _build()
        assert "--host" in spec.args
        assert "0.0.0.0" in spec.args
        assert "--port" in spec.args
        assert "8000" in spec.args
        assert "--tensor-parallel-size" in spec.args
        assert "--max-model-len" in spec.args
        assert "--gpu-memory-utilization" in spec.args

    def test_default_omits_optional_flags(self) -> None:
        spec = _build()
        assert "--quantization" not in spec.args
        assert "--enforce-eager" not in spec.args

    def test_env_includes_hf_cache(self) -> None:
        spec = _build()
        assert spec.env["HF_HOME"] == "/workspace/hf_cache"
        assert spec.env["HUGGINGFACE_HUB_CACHE"] == "/workspace/hf_cache"
        assert spec.env["TRANSFORMERS_CACHE"] == "/workspace/hf_cache"


# ---------------------------------------------------------------------------
# build_launch_spec — logic-specific (flag conditional on cfg)
# ---------------------------------------------------------------------------


class TestLaunchSpecLogic:
    def test_quantization_flag_added(self) -> None:
        spec = _build(VLLMEngineConfig(quantization="awq"))
        idx = spec.args.index("--quantization")
        assert spec.args[idx + 1] == "awq"

    def test_quantization_none_omits_flag(self) -> None:
        spec = _build(VLLMEngineConfig(quantization=None))
        assert "--quantization" not in spec.args

    def test_enforce_eager_flag_added(self) -> None:
        spec = _build(VLLMEngineConfig(enforce_eager=True))
        assert "--enforce-eager" in spec.args

    def test_enforce_eager_false_omits_flag(self) -> None:
        spec = _build(VLLMEngineConfig(enforce_eager=False))
        assert "--enforce-eager" not in spec.args

    def test_args_pre_split_no_shell_quoting_needed(self) -> None:
        """Each arg is its own tuple element — no shell-style quoting
        needed by the provider when forming docker run."""
        spec = _build(VLLMEngineConfig(quantization="awq"))
        # Arg list contains atomic strings, no embedded spaces in flag values
        for arg in spec.args:
            assert isinstance(arg, str)
            assert "\n" not in arg


# ---------------------------------------------------------------------------
# build_launch_spec — type errors
# ---------------------------------------------------------------------------


class TestLaunchSpecNegative:
    def test_wrong_config_type_raises(self) -> None:
        from ryotenkai_engines.interfaces import BaseEngineConfig

        class FakeCfg(BaseEngineConfig):
            kind: str = "fake"

        with pytest.raises(TypeError, match="VLLMEngineConfig"):
            VLLMEngineRuntime().build_launch_spec(
                cfg=FakeCfg(),
                image="x",
                container_name="x",
                port=8000,
                workspace_host_path="/x",
                model_path_in_container="/x",
            )


# ---------------------------------------------------------------------------
# Healthcheck + endpoint URL
# ---------------------------------------------------------------------------


class TestHealthcheckAndEndpoint:
    def test_healthcheck_curl_format(self) -> None:
        cmd = VLLMEngineRuntime().build_healthcheck_command(host="1.2.3.4", port=8000)
        assert "curl" in cmd
        assert "1.2.3.4" in cmd
        assert "8000" in cmd
        assert "/v1/models" in cmd

    def test_endpoint_url_format(self) -> None:
        url = VLLMEngineRuntime().build_default_endpoint_url(host="host", port=8000)
        assert url == "http://host:8000/v1"


# ---------------------------------------------------------------------------
# validate_config — engine-side invariants
# ---------------------------------------------------------------------------


class TestValidateConfig:
    def test_default_config_passes(self) -> None:
        result = VLLMEngineRuntime().validate_config(VLLMEngineConfig())
        assert result.is_ok()

    def test_merge_before_deploy_false_fails(self) -> None:
        cfg = VLLMEngineConfig(merge_before_deploy=False)
        result = VLLMEngineRuntime().validate_config(cfg)
        assert result.is_err()
        err = result.unwrap_err()
        assert err.code == "VLLM_LIVE_LORA_NOT_SUPPORTED"

    def test_supported_quantization_passes(self) -> None:
        result = VLLMEngineRuntime().validate_config(
            VLLMEngineConfig(quantization="awq"),
        )
        assert result.is_ok()

    def test_unsupported_quantization_fails(self) -> None:
        result = VLLMEngineRuntime().validate_config(
            VLLMEngineConfig(quantization="not-a-real-quant"),
        )
        assert result.is_err()
        err = result.unwrap_err()
        assert err.code == "VLLM_QUANTIZATION_UNSUPPORTED"

    def test_wrong_config_type_returns_err(self) -> None:
        from ryotenkai_engines.interfaces import BaseEngineConfig

        class FakeCfg(BaseEngineConfig):
            kind: str = "fake"

        result = VLLMEngineRuntime().validate_config(FakeCfg())
        assert result.is_err()
        assert result.unwrap_err().code == "VLLM_CONFIG_TYPE_MISMATCH"


# ---------------------------------------------------------------------------
# Combinatorial — (quantization, enforce_eager) × (TP=1/4)
# ---------------------------------------------------------------------------


class TestCombinatorial:
    @pytest.mark.parametrize(
        "quant,eager,tp",
        list(itertools.product(["awq", "gptq", None], [True, False], [1, 4])),
    )
    def test_arg_consistency(self, quant: str | None, eager: bool, tp: int) -> None:
        cfg = VLLMEngineConfig(
            quantization=quant,
            enforce_eager=eager,
            tensor_parallel_size=tp,
        )
        spec = _build(cfg)

        if quant:
            assert "--quantization" in spec.args
        else:
            assert "--quantization" not in spec.args
        if eager:
            assert "--enforce-eager" in spec.args
        else:
            assert "--enforce-eager" not in spec.args

        idx = spec.args.index("--tensor-parallel-size")
        assert spec.args[idx + 1] == str(tp)


# ---------------------------------------------------------------------------
# Regression — parity with legacy VLLMEngine.build_docker_run_command output
# (smoke check that the same flag set comes out of the new shape).
# ---------------------------------------------------------------------------


class TestLegacyParity:
    def test_legacy_flag_set_unchanged(self) -> None:
        """The flag set produced by the new LaunchSpec.args must contain
        everything the legacy ``VLLMEngine.build_docker_run_command``
        produced. Specifically: serve, model path, --host, --port,
        --tensor-parallel-size, --max-model-len, --gpu-memory-utilization."""
        spec = _build()
        legacy_required = {
            "serve",
            "--host",
            "--port",
            "--tensor-parallel-size",
            "--max-model-len",
            "--gpu-memory-utilization",
        }
        actual = set(spec.args)
        missing = legacy_required - actual
        assert not missing, f"Missing legacy flags: {missing}"
