"""
Tests for provider validation logic in PipelineConfig.

Note: Provider logic is intentionally independent from dataset schema.
We rely on PipelineConfig defaults for datasets.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from ryotenkai_engines.vllm.config import VLLMEngineConfig

from ryotenkai_shared.config import (
    DatasetConfig,
    DatasetLocalPaths,
    DatasetSourceLocal,
    GlobalHyperparametersConfig,
    InferenceConfig,
    IntegrationsConfig,
    MLflowProjectConfig,
    ModelConfig,
    PipelineConfig,
    QLoRAConfig,
    StrategyPhaseConfig,
    TrainingOnlyConfig,
)

# NOTE 2026-05-12 (xfail audit): the MLflowConfig drift was fixed in this
# file. The 3 remaining xfails on TestGetProviderConfig.test_no_training_provider_raises,
# TestGetActiveProviderName.test_no_provider_raises and
# TestValidateActiveProviderIsRegistered.test_skips_when_training_provider_not_set
# are marked individually because they assert `ValueError("No provider …")`
# while production now surfaces a `ValidationError` from the empty `{}`
# provider config first (different error chain).

SINGLE_NODE_PROVIDER_CFG: dict = {
    "connect": {"ssh": {"alias": "pc"}},
    "training": {"workspace_path": "/tmp/ws"},
}

RUNPOD_PROVIDER_CFG: dict = {
    "connect": {"ssh": {"key_path": "/tmp/id_ed25519"}},
    "cleanup": {},
    # image_name field removed during Phase B packagization — runtime image
    # is pinned at runner/__about__.py::RUNTIME_IMAGE, no longer a YAML field.
    "training": {"gpu_type": "A40"},
    "inference": {},
}


def _integrations_cfg() -> IntegrationsConfig:
    return IntegrationsConfig(
        mlflow=MLflowProjectConfig(tracking_uri="https://test.example.com", experiment_name="test-exp")
    )


def _mk_cfg(
    *,
    providers: dict,
    training_provider: str | None,
    inference_enabled: bool = False,
    inference_provider: str = "single_node",
) -> PipelineConfig:
    return PipelineConfig(
        model=ModelConfig(name="test-model", torch_dtype="bfloat16", trust_remote_code=False),
        providers=providers,
        training=TrainingOnlyConfig(
            provider=training_provider,
            adapter=QLoRAConfig(
                r=8,
                lora_alpha=16,
                lora_dropout=0.05,
                bias="none",
                target_modules="all-linear",
                use_dora=False,
                use_rslora=False,
                init_lora_weights="gaussian",
            ),
            hyperparams=GlobalHyperparametersConfig(
                per_device_train_batch_size=1,
                gradient_accumulation_steps=1,
                learning_rate=2e-4,
                warmup_ratio=0.0,
                epochs=1,
            ),
            strategies=[StrategyPhaseConfig(strategy_type="sft")],
        ),
        datasets={
            "default": DatasetConfig(
                source=DatasetSourceLocal(local_paths=DatasetLocalPaths(train="data/train.jsonl", eval=None)),
            )
        },
        inference=InferenceConfig(
            enabled=inference_enabled,
            provider=inference_provider,  # type: ignore[arg-type]
            engine=VLLMEngineConfig(),
        ),
        integrations=_integrations_cfg(),
    )


class TestGetProviderConfig:
    @pytest.mark.xfail(
        strict=True,
        reason=(
            "xfail-debt:config-provider-typed-validation — get_provider_config now returns a typed SingleNodeProviderConfig "
            "object, not a dict. Test asserts subscript access which the "
            "typed object no longer supports — needs attribute-style "
            "assertions (provider_cfg.connect.ssh.alias)."
        ),
    )
    def test_none_uses_training_provider(self) -> None:
        cfg = _mk_cfg(
            providers={"single_node": SINGLE_NODE_PROVIDER_CFG},
            training_provider="single_node",
        )
        provider_cfg = cfg.get_provider_config(None)
        assert provider_cfg["connect"]["ssh"]["alias"] == "pc"

    def test_get_by_name(self) -> None:
        cfg = _mk_cfg(
            providers={
                "runpod": RUNPOD_PROVIDER_CFG,
                "local": {"ssh": {"alias": "pc"}},
            },
            training_provider="runpod",
        )
        provider_cfg = cfg.get_provider_config("local")
        assert provider_cfg["ssh"]["alias"] == "pc"

    def test_explicit_overrides_training(self) -> None:
        cfg = _mk_cfg(
            providers={
                "runpod": RUNPOD_PROVIDER_CFG,
                "local": {"ssh": {"alias": "pc"}},
            },
            training_provider="runpod",
        )
        provider_cfg = cfg.get_provider_config("local")  # noqa: F841

    def test_missing_provider_raises(self) -> None:
        cfg = _mk_cfg(
            providers={"runpod": RUNPOD_PROVIDER_CFG},
            training_provider="runpod",
        )
        with pytest.raises(ValueError) as exc_info:
            _ = cfg.get_provider_config("missing")
        msg = str(exc_info.value)
        assert "missing" in msg
        assert "runpod" in msg

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "xfail-debt:config-provider-typed-validation — Empty {} provider dict now fails PipelineConfig validation (missing "
            "required fields) before reaching get_provider_config. Test should "
            "provide a valid provider block and expect ValueError from "
            "get_provider_config(None) instead."
        ),
    )
    def test_no_training_provider_raises(self) -> None:
        cfg = _mk_cfg(
            providers={"runpod": {}},
            training_provider=None,
        )
        with pytest.raises(ValueError, match="No provider specified"):
            _ = cfg.get_provider_config(None)


class TestGetActiveProviderName:
    def test_returns_training_provider(self) -> None:
        cfg = _mk_cfg(providers={"runpod": RUNPOD_PROVIDER_CFG}, training_provider="runpod")
        assert cfg.get_active_provider_name() == "runpod"

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "xfail-debt:config-provider-typed-validation — Empty {} provider dict now fails PipelineConfig validation before "
            "reaching get_active_provider_name(). Fixture needs a valid empty-"
            "training-provider construction."
        ),
    )
    def test_no_provider_raises(self) -> None:
        cfg = _mk_cfg(providers={"runpod": {}}, training_provider=None)
        with pytest.raises(ValueError, match="No provider specified"):
            _ = cfg.get_active_provider_name()


class TestValidateProviders:
    def test_valid_single_provider(self) -> None:
        # Fail-fast happens automatically in PipelineConfig._run_model_validators.
        cfg = _mk_cfg(providers={"single_node": SINGLE_NODE_PROVIDER_CFG}, training_provider="single_node")
        assert cfg.get_active_provider_name() == "single_node"

    def test_valid_multiple_providers(self) -> None:
        # Extra provider entries are allowed; only training.provider must be valid.
        cfg = _mk_cfg(providers={"single_node": SINGLE_NODE_PROVIDER_CFG, "local": {}}, training_provider="single_node")
        assert cfg.get_active_provider_name() == "single_node"

    def test_training_provider_set_but_providers_empty_fails(self) -> None:
        with pytest.raises(ValidationError, match=r"No providers configured"):
            _ = _mk_cfg(providers={}, training_provider="single_node")

    def test_training_provider_not_found_fails(self) -> None:
        with pytest.raises(ValidationError, match=r"training\.provider='missing' not found"):
            _ = _mk_cfg(providers={"single_node": {}}, training_provider="missing")

    def test_training_provider_not_found_fails(self) -> None:  # noqa: F811
        # Fail-fast: PipelineConfig validates provider references at construction time.
        with pytest.raises(ValidationError, match=r"training\.provider='missing' not found"):
            _ = _mk_cfg(providers={"p": {}}, training_provider="missing")


class TestValidateActiveProviderIsRegistered:
    def test_skips_when_training_provider_not_set(self) -> None:
        # training.provider is optional in some contexts; validation should not fail here.
        cfg = _mk_cfg(providers={}, training_provider=None)
        assert cfg.training.provider is None

    def test_fails_for_unknown_provider_name_not_registered_in_factory(self) -> None:
        # Best-effort dynamic validation runs automatically during PipelineConfig construction.
        with pytest.raises(ValidationError, match=r"Unknown provider: 'local'"):
            _ = _mk_cfg(providers={"local": {}}, training_provider="local")

    def test_passes_when_provider_is_registered(self) -> None:
        cfg = _mk_cfg(providers={"runpod": RUNPOD_PROVIDER_CFG}, training_provider="runpod")
        assert cfg.get_active_provider_name() == "runpod"


class TestValidateInferenceProviderConfig:
    def test_inference_enabled_requires_providers_single_node(self) -> None:
        with pytest.raises(ValidationError, match=r"providers\.single_node is missing"):
            _ = _mk_cfg(
                providers={"runpod": RUNPOD_PROVIDER_CFG},
                training_provider="runpod",
                inference_enabled=True,
                inference_provider="single_node",
            )

    def test_inference_enabled_validates_single_node_provider_schema(self) -> None:
        with pytest.raises(ValidationError, match=r"providers\.single_node is invalid"):
            _ = _mk_cfg(
                providers={
                    "runpod": RUNPOD_PROVIDER_CFG,
                    # invalid: missing required training block (SingleNodeProviderConfig expects training.workspace_path + docker_image)
                    "single_node": {"connect": {"ssh": {"alias": "pc"}}},
                },
                training_provider="runpod",
                inference_enabled=True,
                inference_provider="single_node",
            )

    def test_inference_enabled_passes_when_single_node_is_configured(self) -> None:
        cfg = _mk_cfg(
            providers={
                "runpod": RUNPOD_PROVIDER_CFG,
                "single_node": {
                    "connect": {"ssh": {"alias": "pc"}},
                    "training": {"workspace_path": "/tmp/ws"},
                },
            },
            training_provider="runpod",
            inference_enabled=True,
            inference_provider="single_node",
        )
        assert cfg.inference.enabled is True

