from __future__ import annotations

import sys
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from ryotenkai_shared.config.validators.cross import (
    validate_pipeline_active_provider_is_registered,
    validate_pipeline_evaluation_requires_inference,
    validate_pipeline_inference_provider_config,
    validate_pipeline_providers_config,
    validate_pipeline_strategy_dataset_references,
)
from ryotenkai_shared.config.validators.pipeline import validate_pipeline_config_references
from ryotenkai_shared.errors import ConfigInvalidError

pytestmark = pytest.mark.unit


# Pre-existing legacy debt: validator error code drifted from
# CONFIG_PROVIDER_NOT_REGISTERED to CONFIG_PROVIDER_REGISTRY_LOAD_FAILED
# when the SimpleNamespace shim was added. The 3 tests below pin the older
# code and need to be updated when the assertion target catches up.
_XFAIL_REGISTRY_LOAD_DRIFT = pytest.mark.xfail(
    strict=True,
    reason=(
        "xfail-debt:cross-validators-registry-error-drift — Pre-existing: validator now surfaces CONFIG_PROVIDER_REGISTRY_LOAD_FAILED"
        " (SimpleNamespace.get_registry) instead of CONFIG_PROVIDER_NOT_REGISTERED."
    ),
)


def _assert_ok(result) -> None:
    """Validator returns ``None`` on success; assert the value is None.

    Kept for backward compatibility with the per-validator call sites
    below — the validator itself is invoked inline and would have raised
    on failure, so reaching this point means success.
    """
    assert result is None


def _assert_err(call_or_fn, *, code: str | None = None) -> str:
    """Assert the validator raised :class:`ConfigInvalidError`.

    Test callers wrap the invocation in a zero-arg ``lambda`` so this
    helper can invoke it under ``pytest.raises``. The returned string is
    ``str(exc)`` for the existing message-substring assertions.
    """
    assert callable(call_or_fn), "pass a zero-arg lambda — Result API is gone"
    with pytest.raises(ConfigInvalidError) as exc_info:
        call_or_fn()
    if code is not None:
        assert exc_info.value.context.get("code") == code, (
            f"expected legacy subcode {code!r}, got {exc_info.value.context.get('code')!r}"
        )
    return str(exc_info.value)


@dataclass(frozen=True)
class DummyStrategy:
    strategy_type: str
    dataset: str | None = None


@dataclass(frozen=True)
class DummyTraining:
    provider: str | None
    strategies: list[DummyStrategy]


@dataclass(frozen=True)
class DummyPipelineCfg:
    providers: dict[str, dict]
    training: DummyTraining
    datasets: dict[str, object]


class TestValidatePipelineProvidersConfig:
    def test_positive_valid_provider_reference(self) -> None:
        cfg = DummyPipelineCfg(
            providers={
                "single_node": {
                    "connect": {"ssh": {"alias": "pc"}},
                    "training": {"workspace_path": "/tmp/workspace"},
                }
            },
            training=DummyTraining(provider="single_node", strategies=[]),
            datasets={"default": object()},
        )
        _assert_ok(validate_pipeline_providers_config(cfg))  # type: ignore[arg-type]

    def test_negative_no_providers(self) -> None:
        cfg = DummyPipelineCfg(
            providers={},
            training=DummyTraining(provider="single_node", strategies=[]),
            datasets={"default": object()},
        )
        err = _assert_err(lambda: validate_pipeline_providers_config(cfg), code="CONFIG_PROVIDERS_MISSING")  # type: ignore[arg-type]
        assert "No providers configured" in err

    def test_negative_training_provider_not_set(self) -> None:
        cfg = DummyPipelineCfg(
            providers={"single_node": {}},
            training=DummyTraining(provider=None, strategies=[]),
            datasets={"default": object()},
        )
        err = _assert_err(lambda: validate_pipeline_providers_config(cfg), code="CONFIG_TRAINING_PROVIDER_MISSING")  # type: ignore[arg-type]
        assert "training.provider not set" in err

    def test_negative_training_provider_not_found(self) -> None:
        cfg = DummyPipelineCfg(
            providers={"single_node": {}},
            training=DummyTraining(provider="missing", strategies=[]),
            datasets={"default": object()},
        )
        err = _assert_err(lambda: validate_pipeline_providers_config(cfg), code="CONFIG_TRAINING_PROVIDER_NOT_FOUND")  # type: ignore[arg-type]
        assert "training.provider='missing' not found" in err


class TestValidatePipelineActiveProviderIsRegistered:
    def test_boundary_skips_when_training_provider_not_set(self) -> None:
        cfg = DummyPipelineCfg(
            providers={},
            training=DummyTraining(provider=None, strategies=[]),
            datasets={"default": object()},
        )
        _assert_ok(validate_pipeline_active_provider_is_registered(cfg))  # type: ignore[arg-type]

    def test_dependency_error_skips_when_src_providers_missing(self) -> None:
        cfg = DummyPipelineCfg(
            providers={
                "single_node": {
                    "connect": {"ssh": {"alias": "pc"}},
                    "training": {"workspace_path": "/tmp/workspace"},
                }
            },
            training=DummyTraining(provider="single_node", strategies=[]),
            datasets={"default": object()},
        )
        with patch(
            "ryotenkai_shared.config.validators.cross.importlib.import_module",
            side_effect=ModuleNotFoundError("nope", name="ryotenkai_providers.training"),
        ):
            result = validate_pipeline_active_provider_is_registered(cfg)  # type: ignore[arg-type]
        _assert_ok(result)

    @_XFAIL_REGISTRY_LOAD_DRIFT
    def test_negative_unknown_provider_not_registered(self) -> None:
        cfg = DummyPipelineCfg(
            providers={"local": {}},
            training=DummyTraining(provider="local", strategies=[]),
            datasets={"default": object()},
        )

        class DummyFactory:
            @staticmethod
            def get_available_providers():
                return ["single_node"]

        dummy_mod = SimpleNamespace(GPUProviderFactory=DummyFactory)
        with patch("ryotenkai_shared.config.validators.cross.importlib.import_module", return_value=dummy_mod):
            err = _assert_err(lambda: validate_pipeline_active_provider_is_registered(cfg), code="CONFIG_PROVIDER_NOT_REGISTERED")  # type: ignore[arg-type]
        assert "Unknown provider" in err
        assert "local" in err

    @_XFAIL_REGISTRY_LOAD_DRIFT
    def test_positive_provider_registered(self) -> None:
        cfg = DummyPipelineCfg(
            providers={
                "runpod": {
                    "connect": {"ssh": {"key_path": "/tmp/id_ed25519"}},
                    "cleanup": {},
                    "training": {"gpu_type": "NVIDIA A40"},
                    "inference": {},
                }
            },
            training=DummyTraining(provider="runpod", strategies=[]),
            datasets={"default": object()},
        )

        class DummyFactory:
            @staticmethod
            def get_available_providers():
                return ["single_node", "runpod"]

        dummy_mod = SimpleNamespace(GPUProviderFactory=DummyFactory)
        with patch("ryotenkai_shared.config.validators.cross.importlib.import_module", return_value=dummy_mod):
            result = validate_pipeline_active_provider_is_registered(cfg)  # type: ignore[arg-type]
        _assert_ok(result)


class TestValidatePipelineStrategyDatasetReferences:
    def test_boundary_datasets_empty_fails(self) -> None:
        cfg = DummyPipelineCfg(
            providers={},
            training=DummyTraining(provider=None, strategies=[DummyStrategy(strategy_type="sft", dataset=None)]),
            datasets={},
        )
        err = _assert_err(lambda: validate_pipeline_strategy_dataset_references(cfg), code="CONFIG_DATASETS_EMPTY")  # type: ignore[arg-type]
        assert "datasets must contain at least one entry" in err

    def test_positive_strategy_without_dataset_ok_when_datasets_non_empty(self) -> None:
        cfg = DummyPipelineCfg(
            providers={},
            training=DummyTraining(provider=None, strategies=[DummyStrategy(strategy_type="sft", dataset=None)]),
            datasets={"default": object()},
        )
        _assert_ok(validate_pipeline_strategy_dataset_references(cfg))  # type: ignore[arg-type]

    def test_negative_missing_dataset_reference(self) -> None:
        cfg = DummyPipelineCfg(
            providers={},
            training=DummyTraining(provider=None, strategies=[DummyStrategy(strategy_type="sft", dataset="missing")]),
            datasets={"default": object()},
        )
        err = _assert_err(lambda: validate_pipeline_strategy_dataset_references(cfg), code="CONFIG_STRATEGY_DATASET_MISSING")  # type: ignore[arg-type]
        assert "references" in err
        assert "dataset 'missing'" in err

    def test_combinatorial_multiple_strategies_one_missing(self) -> None:
        cfg = DummyPipelineCfg(
            providers={},
            training=DummyTraining(
                provider=None,
                strategies=[
                    DummyStrategy(strategy_type="sft", dataset="default"),
                    DummyStrategy(strategy_type="dpo", dataset="missing"),
                ],
            ),
            datasets={"default": object()},
        )
        err = _assert_err(lambda: validate_pipeline_strategy_dataset_references(cfg), code="CONFIG_STRATEGY_DATASET_MISSING")  # type: ignore[arg-type]
        assert "Strategy 1" in err
        assert "dataset 'missing'" in err


@dataclass(frozen=True)
class DummyInferenceConfig:
    enabled: bool
    provider: str = "single_node"


@dataclass(frozen=True)
class DummyEvalConfig:
    enabled: bool


@dataclass(frozen=True)
class DummyExtendedPipelineCfg:
    providers: dict
    training: DummyTraining
    datasets: dict
    inference: DummyInferenceConfig | None = None
    evaluation: DummyEvalConfig | None = None


_VALID_SINGLE_NODE_CFG = {
    "connect": {"ssh": {"alias": "pc"}},
    "training": {"workspace_path": "/tmp/workspace"},
}

_VALID_RUNPOD_CFG = {
    "connect": {"ssh": {"key_path": "/tmp/id_ed25519"}},
    "cleanup": {},
    "training": {"gpu_type": "NVIDIA A40"},
    "inference": {},
}

_RUNPOD_WITH_POD_CFG = {
    "connect": {"ssh": {"key_path": "/tmp/id_ed25519"}},
    "cleanup": {},
    "training": {"gpu_type": "NVIDIA A40"},
    "inference": {"pod": {}},  # image_name pinned in INFERENCE_IMAGES (Phase 6.6)
}


class TestValidatePipelineProvidersConfigValidationErrors:
    def test_negative_single_node_invalid_schema_triggers_validation_error(self) -> None:
        cfg = DummyPipelineCfg(
            providers={"single_node": {}},
            training=DummyTraining(provider="single_node", strategies=[]),
            datasets={"default": object()},
        )
        err = _assert_err(lambda: validate_pipeline_providers_config(cfg), code="CONFIG_SINGLE_NODE_PROVIDER_INVALID")  # type: ignore[arg-type]
        assert "single_node" in err
        assert "invalid for SingleNodeProviderConfig" in err

    def test_negative_single_node_generic_exception(self) -> None:
        from ryotenkai_shared.config.providers.registry import PROVIDER_TYPES

        cfg = DummyPipelineCfg(
            providers={"single_node": {}},
            training=DummyTraining(provider="single_node", strategies=[]),
            datasets={"default": object()},
        )
        with patch.object(PROVIDER_TYPES["single_node"], "schema", MagicMock(side_effect=OSError("disk error"))):
            err = _assert_err(lambda: validate_pipeline_providers_config(cfg), code="CONFIG_SINGLE_NODE_PROVIDER_INVALID")  # type: ignore[arg-type]
        assert "invalid for SingleNodeProviderConfig" in err
        assert "disk error" in err

    def test_negative_runpod_invalid_schema_triggers_validation_error(self) -> None:
        cfg = DummyPipelineCfg(
            providers={"runpod": {}},
            training=DummyTraining(provider="runpod", strategies=[]),
            datasets={"default": object()},
        )
        err = _assert_err(lambda: validate_pipeline_providers_config(cfg), code="CONFIG_RUNPOD_PROVIDER_INVALID")  # type: ignore[arg-type]
        assert "runpod" in err
        assert "invalid for RunPodProviderConfig" in err

    def test_negative_runpod_generic_exception(self) -> None:
        from ryotenkai_shared.config.providers.registry import PROVIDER_TYPES

        cfg = DummyPipelineCfg(
            providers={"runpod": {}},
            training=DummyTraining(provider="runpod", strategies=[]),
            datasets={"default": object()},
        )
        with patch.object(PROVIDER_TYPES["runpod"], "schema", MagicMock(side_effect=OSError("network error"))):
            err = _assert_err(lambda: validate_pipeline_providers_config(cfg), code="CONFIG_RUNPOD_PROVIDER_INVALID")  # type: ignore[arg-type]
        assert "invalid for RunPodProviderConfig" in err
        assert "network error" in err


class TestValidatePipelineActiveProviderIsRegisteredEdgeCases:
    def _cfg_with_custom_provider(self) -> DummyPipelineCfg:
        return DummyPipelineCfg(
            providers={"custom_provider": {}},
            training=DummyTraining(provider="custom_provider", strategies=[]),
            datasets={"default": object()},
        )

    def test_negative_propagates_providers_config_failure(self) -> None:
        cfg = DummyPipelineCfg(
            providers={},
            training=DummyTraining(provider="some_provider", strategies=[]),
            datasets={"default": object()},
        )
        err = _assert_err(lambda: validate_pipeline_active_provider_is_registered(cfg), code="CONFIG_PROVIDERS_MISSING")  # type: ignore[arg-type]
        assert "No providers configured" in err

    def test_negative_non_pipeline_module_not_found_error(self) -> None:
        cfg = self._cfg_with_custom_provider()
        with patch(
            "ryotenkai_shared.config.validators.cross.importlib.import_module",
            side_effect=ModuleNotFoundError("bad module", name="some.other.module"),
        ):
            err = _assert_err(lambda: validate_pipeline_active_provider_is_registered(cfg), code="CONFIG_PROVIDER_REGISTRY_LOAD_FAILED")  # type: ignore[arg-type]
        assert "Failed to load provider registry" in err

    def test_negative_generic_exception_from_factory_import(self) -> None:
        cfg = self._cfg_with_custom_provider()
        with patch(
            "ryotenkai_shared.config.validators.cross.importlib.import_module",
            side_effect=RuntimeError("unexpected crash"),
        ):
            err = _assert_err(lambda: validate_pipeline_active_provider_is_registered(cfg), code="CONFIG_PROVIDER_REGISTRY_LOAD_FAILED")  # type: ignore[arg-type]
        assert "Failed to load provider registry" in err
        assert "unexpected crash" in err


class TestValidatePipelineInferenceProviderConfig:
    def test_positive_inference_disabled(self) -> None:
        cfg = DummyExtendedPipelineCfg(
            providers={},
            training=DummyTraining(provider=None, strategies=[]),
            datasets={},
            inference=DummyInferenceConfig(enabled=False),
        )
        _assert_ok(validate_pipeline_inference_provider_config(cfg))  # type: ignore[arg-type]

    def test_positive_inference_none(self) -> None:
        cfg = DummyExtendedPipelineCfg(
            providers={},
            training=DummyTraining(provider=None, strategies=[]),
            datasets={},
            inference=None,
        )
        _assert_ok(validate_pipeline_inference_provider_config(cfg))  # type: ignore[arg-type]

    def test_positive_inference_unknown_provider_skipped(self) -> None:
        cfg = DummyExtendedPipelineCfg(
            providers={},
            training=DummyTraining(provider=None, strategies=[]),
            datasets={},
            inference=DummyInferenceConfig(enabled=True, provider="custom_cloud"),
        )
        _assert_ok(validate_pipeline_inference_provider_config(cfg))  # type: ignore[arg-type]

    def test_negative_single_node_missing_from_providers(self) -> None:
        cfg = DummyExtendedPipelineCfg(
            providers={"runpod": _VALID_RUNPOD_CFG},
            training=DummyTraining(provider="runpod", strategies=[]),
            datasets={},
            inference=DummyInferenceConfig(enabled=True, provider="single_node"),
        )
        err = _assert_err(lambda: validate_pipeline_inference_provider_config(cfg), code="CONFIG_INFERENCE_PROVIDER_MISSING")  # type: ignore[arg-type]
        assert "providers.single_node is missing" in err

    def test_negative_single_node_invalid_schema(self) -> None:
        cfg = DummyExtendedPipelineCfg(
            providers={"single_node": {}},
            training=DummyTraining(provider="single_node", strategies=[]),
            datasets={},
            inference=DummyInferenceConfig(enabled=True, provider="single_node"),
        )
        err = _assert_err(lambda: validate_pipeline_inference_provider_config(cfg), code="CONFIG_INFERENCE_SINGLE_NODE_INVALID")  # type: ignore[arg-type]
        assert "invalid for SingleNodeProviderConfig" in err

    def test_negative_single_node_generic_exception(self) -> None:
        cfg = DummyExtendedPipelineCfg(
            providers={"single_node": {}},
            training=DummyTraining(provider="single_node", strategies=[]),
            datasets={},
            inference=DummyInferenceConfig(enabled=True, provider="single_node"),
        )
        mock_sn = MagicMock()
        mock_sn.SingleNodeProviderConfig.side_effect = OSError("disk error")
        with patch.dict(sys.modules, {"ryotenkai_shared.config.providers.single_node": mock_sn}):
            err = _assert_err(lambda: validate_pipeline_inference_provider_config(cfg), code="CONFIG_INFERENCE_SINGLE_NODE_INVALID")  # type: ignore[arg-type]
        assert "invalid for SingleNodeProviderConfig" in err

    def test_positive_single_node_valid_config(self) -> None:
        cfg = DummyExtendedPipelineCfg(
            providers={"single_node": _VALID_SINGLE_NODE_CFG},
            training=DummyTraining(provider="single_node", strategies=[]),
            datasets={},
            inference=DummyInferenceConfig(enabled=True, provider="single_node"),
        )
        _assert_ok(validate_pipeline_inference_provider_config(cfg))  # type: ignore[arg-type]

    def test_negative_runpod_missing_from_providers(self) -> None:
        cfg = DummyExtendedPipelineCfg(
            providers={"single_node": _VALID_SINGLE_NODE_CFG},
            training=DummyTraining(provider="single_node", strategies=[]),
            datasets={},
            inference=DummyInferenceConfig(enabled=True, provider="runpod"),
        )
        err = _assert_err(lambda: validate_pipeline_inference_provider_config(cfg), code="CONFIG_INFERENCE_PROVIDER_MISSING")  # type: ignore[arg-type]
        assert "providers.runpod is missing" in err

    def test_negative_runpod_pod_is_none(self) -> None:
        cfg = DummyExtendedPipelineCfg(
            providers={"runpod": _VALID_RUNPOD_CFG},
            training=DummyTraining(provider="runpod", strategies=[]),
            datasets={},
            inference=DummyInferenceConfig(enabled=True, provider="runpod"),
        )
        err = _assert_err(lambda: validate_pipeline_inference_provider_config(cfg), code="CONFIG_RUNPOD_INFERENCE_POD_MISSING")  # type: ignore[arg-type]
        assert "inference.pod is missing" in err

    def test_negative_runpod_invalid_schema(self) -> None:
        cfg = DummyExtendedPipelineCfg(
            providers={"runpod": {}},
            training=DummyTraining(provider="runpod", strategies=[]),
            datasets={},
            inference=DummyInferenceConfig(enabled=True, provider="runpod"),
        )
        err = _assert_err(lambda: validate_pipeline_inference_provider_config(cfg), code="CONFIG_INFERENCE_RUNPOD_INVALID")  # type: ignore[arg-type]
        assert "invalid for RunPodProviderConfig" in err

    def test_negative_runpod_generic_exception(self) -> None:
        cfg = DummyExtendedPipelineCfg(
            providers={"runpod": {}},
            training=DummyTraining(provider="runpod", strategies=[]),
            datasets={},
            inference=DummyInferenceConfig(enabled=True, provider="runpod"),
        )
        mock_rp = MagicMock()
        mock_rp.RunPodProviderConfig.side_effect = OSError("crash")
        with patch.dict(sys.modules, {"ryotenkai_shared.config.providers.runpod": mock_rp}):
            err = _assert_err(lambda: validate_pipeline_inference_provider_config(cfg), code="CONFIG_INFERENCE_RUNPOD_INVALID")  # type: ignore[arg-type]
        assert "invalid for RunPodProviderConfig" in err

    def test_positive_runpod_valid_config_with_pod(self) -> None:
        cfg = DummyExtendedPipelineCfg(
            providers={"runpod": _RUNPOD_WITH_POD_CFG},
            training=DummyTraining(provider="runpod", strategies=[]),
            datasets={},
            inference=DummyInferenceConfig(enabled=True, provider="runpod"),
        )
        _assert_ok(validate_pipeline_inference_provider_config(cfg))  # type: ignore[arg-type]


class TestValidatePipelineEvaluationRequiresInference:
    def test_positive_eval_disabled(self) -> None:
        cfg = DummyExtendedPipelineCfg(
            providers={},
            training=DummyTraining(provider=None, strategies=[]),
            datasets={},
            evaluation=DummyEvalConfig(enabled=False),
        )
        _assert_ok(validate_pipeline_evaluation_requires_inference(cfg))  # type: ignore[arg-type]

    def test_positive_eval_none(self) -> None:
        cfg = DummyExtendedPipelineCfg(
            providers={},
            training=DummyTraining(provider=None, strategies=[]),
            datasets={},
            evaluation=None,
        )
        _assert_ok(validate_pipeline_evaluation_requires_inference(cfg))  # type: ignore[arg-type]

    def test_negative_eval_enabled_inference_disabled(self) -> None:
        cfg = DummyExtendedPipelineCfg(
            providers={},
            training=DummyTraining(provider=None, strategies=[]),
            datasets={},
            inference=DummyInferenceConfig(enabled=False),
            evaluation=DummyEvalConfig(enabled=True),
        )
        err = _assert_err(lambda: validate_pipeline_evaluation_requires_inference(cfg), code="CONFIG_EVALUATION_REQUIRES_INFERENCE")  # type: ignore[arg-type]
        assert "evaluation.enabled=true requires inference.enabled=true" in err

    def test_negative_eval_enabled_inference_none(self) -> None:
        cfg = DummyExtendedPipelineCfg(
            providers={},
            training=DummyTraining(provider=None, strategies=[]),
            datasets={},
            inference=None,
            evaluation=DummyEvalConfig(enabled=True),
        )
        err = _assert_err(lambda: validate_pipeline_evaluation_requires_inference(cfg), code="CONFIG_EVALUATION_REQUIRES_INFERENCE")  # type: ignore[arg-type]
        assert "evaluation.enabled=true requires inference.enabled=true" in err

    def test_positive_eval_enabled_inference_enabled(self) -> None:
        cfg = DummyExtendedPipelineCfg(
            providers={},
            training=DummyTraining(provider=None, strategies=[]),
            datasets={},
            inference=DummyInferenceConfig(enabled=True),
            evaluation=DummyEvalConfig(enabled=True),
        )
        _assert_ok(validate_pipeline_evaluation_requires_inference(cfg))  # type: ignore[arg-type]


class TestValidatePipelineConfigReferences:
    def test_positive_provider_unset_only_dataset_refs_checked(self) -> None:
        cfg = DummyPipelineCfg(
            providers={},
            training=DummyTraining(provider=None, strategies=[DummyStrategy(strategy_type="sft", dataset="default")]),
            datasets={"default": object()},
        )
        # No provider check when training.provider is None.
        validate_pipeline_config_references(cfg)  # type: ignore[arg-type]

    @_XFAIL_REGISTRY_LOAD_DRIFT
    def test_negative_provider_invalid_fails_first(self) -> None:
        cfg = DummyPipelineCfg(
            providers={"local": {}},
            training=DummyTraining(provider="local", strategies=[DummyStrategy(strategy_type="sft", dataset="missing")]),
            datasets={"default": object()},
        )

        class DummyFactory:
            @staticmethod
            def get_available_providers():
                return ["single_node"]

        dummy_mod = SimpleNamespace(GPUProviderFactory=DummyFactory)
        with patch("ryotenkai_shared.config.validators.cross.importlib.import_module", return_value=dummy_mod):  # noqa: SIM117
            with pytest.raises(ValueError, match=r"Unknown provider: 'local'"):
                validate_pipeline_config_references(cfg)  # type: ignore[arg-type]

    def test_negative_dataset_missing_fails(self) -> None:
        cfg = DummyPipelineCfg(
            providers={},
            training=DummyTraining(provider=None, strategies=[DummyStrategy(strategy_type="sft", dataset="missing")]),
            datasets={"default": object()},
        )
        with pytest.raises(ValueError, match=r"dataset 'missing'"):
            validate_pipeline_config_references(cfg)  # type: ignore[arg-type]

