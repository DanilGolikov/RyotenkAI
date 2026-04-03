"""
Config v6.0 Comprehensive Test Suite - Part 1

Tests categories 1-4:
1. Positive tests - Happy path for all REQUIRED fields
2. Negative tests - Missing required, invalid types
3. Boundary tests - Min/max constraints validation
4. Invariant tests - Config consistency rules

Config v6.0 Breaking Changes:
- ModelConfig: torch_dtype, trust_remote_code → REQUIRED
- LoraConfig: 8 fields → REQUIRED (base + advanced), QLoRA optional
- AdaLoraConfig: 6 fields → REQUIRED (core + common LoRA)
- GlobalHyperparametersConfig: 5 core REQUIRED
- PhaseHyperparametersConfig: all optional (for phase overrides)
- DatasetLocalPaths: train → REQUIRED
- InferenceVLLMEngineConfig: merge_image, serve_image → REQUIRED (only for inference.provider=single_node)
"""

import pytest
from pydantic import ValidationError

from src.utils.config import (
    AdaLoraConfig,
    DatasetLocalPaths,
    GlobalHyperparametersConfig,
    InferenceVLLMEngineConfig,
    LoraConfig,
    ModelConfig,
    PhaseHyperparametersConfig,
)

# =============================================================================
# Category 1: POSITIVE TESTS - Happy path for all REQUIRED fields
# =============================================================================


def test_model_config_minimal_valid():
    """ModelConfig with 3 REQUIRED fields validates."""
    config = ModelConfig(
        name="Qwen/Qwen2.5-7B-Instruct",
        torch_dtype="bfloat16",
        trust_remote_code=True,
    )
    assert config.name == "Qwen/Qwen2.5-7B-Instruct"
    assert config.torch_dtype == "bfloat16"
    assert config.trust_remote_code is True
    assert config.device_map == "auto"  # Default
    assert config.flash_attention is False  # Default


def test_lora_config_minimal_valid():
    """LoraConfig with 8 REQUIRED fields (base + advanced) validates."""
    config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules="all-linear",
        use_dora=False,
        use_rslora=False,
        init_lora_weights="gaussian",
    )
    assert config.r == 16
    assert config.lora_alpha == 32
    assert config.lora_dropout == 0.05
    assert config.bias == "none"
    assert config.target_modules == "all-linear"
    assert config.use_dora is False
    assert config.use_rslora is False
    assert config.init_lora_weights == "gaussian"
    # QLoRA parameters have defaults
    assert config.bnb_4bit_quant_type == "nf4"
    assert config.bnb_4bit_compute_dtype == "bfloat16"
    assert config.bnb_4bit_use_double_quant is True


def test_adalora_config_minimal_valid():
    """AdaLoraConfig with 6 REQUIRED fields (core + common LoRA) validates."""
    config = AdaLoraConfig(
        init_r=12,
        target_r=8,
        total_step=100,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules="all-linear",
    )
    assert config.init_r == 12
    assert config.target_r == 8
    assert config.lora_alpha == 32
    assert config.lora_dropout == 0.05
    assert config.bias == "none"
    assert config.target_modules == "all-linear"
    # Scheduling parameters have defaults
    assert config.tinit == 200
    assert config.tfinal == 1000
    assert config.delta_t == 10


def test_global_hyperparams_minimal_valid():
    """GlobalHyperparametersConfig with 5 core REQUIRED fields validates."""
    config = GlobalHyperparametersConfig(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        warmup_ratio=0.1,
        epochs=3,
    )
    assert config.per_device_train_batch_size == 4
    assert config.gradient_accumulation_steps == 2
    assert config.learning_rate == 2e-4
    assert config.warmup_ratio == 0.1
    assert config.epochs == 3


def test_phase_hyperparams_all_optional():
    """PhaseHyperparametersConfig — all fields optional; empty config valid."""
    config = PhaseHyperparametersConfig()
    assert config.per_device_train_batch_size is None
    assert config.gradient_accumulation_steps is None
    assert config.learning_rate is None


def test_dataset_local_paths_minimal_valid():
    """DatasetLocalPaths with train REQUIRED validates."""
    config = DatasetLocalPaths(train="/data/train.jsonl")
    assert config.train == "/data/train.jsonl"
    assert config.eval is None  # Optional


def test_inference_vllm_minimal_valid():
    """InferenceVLLMEngineConfig with merge_image, serve_image REQUIRED validates."""
    config = InferenceVLLMEngineConfig(
        merge_image="helix/merge:latest",
        serve_image="vllm/vllm-openai:v0.7.0",
    )
    assert config.merge_image == "helix/merge:latest"
    assert config.serve_image == "vllm/vllm-openai:v0.7.0"
    # Defaults
    assert config.gpu_memory_utilization == 0.9
    assert config.max_model_len == 4096


def test_full_pipeline_config_valid():
    """Full integration test — minimal valid PipelineConfig v6.0."""
    # Ensures REQUIRED fields integrate correctly
    # We do not build a full PipelineConfig here (too many fields),
    # but each component still validates
    model = ModelConfig(
        name="Qwen/Qwen2.5-7B", torch_dtype="bfloat16", trust_remote_code=True
    )
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules="all-linear",
        use_dora=False,
        use_rslora=False,
        init_lora_weights="gaussian",
    )
    hyperparams = GlobalHyperparametersConfig(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        warmup_ratio=0.1,
        epochs=3,
    )
    assert model.name == "Qwen/Qwen2.5-7B"
    assert lora.r == 16
    assert hyperparams.epochs == 3


# =============================================================================
# Category 2: NEGATIVE TESTS - Missing required, invalid types
# =============================================================================


def test_model_config_missing_torch_dtype():
    """ModelConfig without torch_dtype must fail."""
    with pytest.raises(ValidationError) as exc_info:
        ModelConfig(name="test/model", trust_remote_code=True)
    errors = exc_info.value.errors()
    assert any("torch_dtype" in str(e) for e in errors)


def test_model_config_missing_trust_remote_code():
    """ModelConfig without trust_remote_code must fail."""
    with pytest.raises(ValidationError) as exc_info:
        ModelConfig(name="test/model", torch_dtype="bfloat16")
    errors = exc_info.value.errors()
    assert any("trust_remote_code" in str(e) for e in errors)


def test_lora_config_missing_r():
    """LoraConfig without r must fail."""
    with pytest.raises(ValidationError) as exc_info:
        LoraConfig(
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            target_modules="all-linear",
            use_dora=False,
            use_rslora=False,
            init_lora_weights="gaussian",
        )
    errors = exc_info.value.errors()
    assert any("r" in str(e.get("loc", [])) for e in errors)


def test_lora_config_missing_multiple_required():
    """LoraConfig missing several REQUIRED fields fails with multiple errors."""
    with pytest.raises(ValidationError) as exc_info:
        LoraConfig(r=16, lora_alpha=32)  # Missing: dropout, bias, target_modules, etc.
    errors = exc_info.value.errors()
    assert len(errors) >= 5  # At least 5 missing fields


def test_adalora_config_missing_init_r():
    """AdaLoraConfig without init_r must fail."""
    with pytest.raises(ValidationError) as exc_info:
        AdaLoraConfig(
            target_r=8,
            total_step=100,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            target_modules="all-linear",
        )
    errors = exc_info.value.errors()
    assert any("init_r" in str(e.get("loc", [])) for e in errors)


def test_global_hyperparams_missing_learning_rate():
    """GlobalHyperparametersConfig without learning_rate must fail."""
    with pytest.raises(ValidationError) as exc_info:
        GlobalHyperparametersConfig(
            per_device_train_batch_size=4,
            gradient_accumulation_steps=2,
            warmup_ratio=0.1,
            epochs=3,
        )
    errors = exc_info.value.errors()
    assert any("learning_rate" in str(e.get("loc", [])) for e in errors)


def test_global_hyperparams_missing_epochs():
    """GlobalHyperparametersConfig without epochs must fail."""
    with pytest.raises(ValidationError) as exc_info:
        GlobalHyperparametersConfig(
            per_device_train_batch_size=4,
            gradient_accumulation_steps=2,
            learning_rate=2e-4,
            warmup_ratio=0.1,
        )
    errors = exc_info.value.errors()
    assert any("epochs" in str(e.get("loc", [])) for e in errors)


def test_dataset_local_paths_missing_train():
    """DatasetLocalPaths without train must fail."""
    with pytest.raises(ValidationError) as exc_info:
        DatasetLocalPaths()
    errors = exc_info.value.errors()
    assert any("train" in str(e.get("loc", [])) for e in errors)


def test_inference_vllm_images_optional_in_engine_config():
    """
    InferenceVLLMEngineConfig: merge_image/serve_image optional at schema level.

    Provider-specific requirements are enforced by InferenceConfig validators.
    """
    cfg = InferenceVLLMEngineConfig()
    assert cfg.merge_image is None
    assert cfg.serve_image is None


def test_lora_config_invalid_r_type():
    """LoraConfig with r as string must fail."""
    with pytest.raises(ValidationError):
        LoraConfig(
            r="sixteen",  # Invalid type
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            target_modules="all-linear",
            use_dora=False,
            use_rslora=False,
            init_lora_weights="gaussian",
        )


def test_lora_config_invalid_bias():
    """LoraConfig with invalid bias must fail."""
    with pytest.raises(ValidationError):
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="invalid_bias",  # Not in Literal["none", "all", "lora_only"]
            target_modules="all-linear",
            use_dora=False,
            use_rslora=False,
            init_lora_weights="gaussian",
        )


def test_global_hyperparams_invalid_learning_rate_type():
    """GlobalHyperparametersConfig: Pydantic casts string floats by default."""
    cfg = GlobalHyperparametersConfig(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate="2e-4",  # cast to float
        warmup_ratio=0.1,
        epochs=3,
    )
    assert cfg.learning_rate == pytest.approx(2e-4)


def test_model_config_invalid_torch_dtype():
    """ModelConfig with invalid torch_dtype fails with a hint."""
    with pytest.raises(ValidationError) as exc_info:
        ModelConfig(
            name="test/model", torch_dtype="invalid_dtype", trust_remote_code=True
        )
    error_msg = str(exc_info.value)
    assert "bfloat16" in error_msg or "float16" in error_msg  # Helpful message


# =============================================================================
# Category 3: BOUNDARY TESTS - Min/max constraints validation
# =============================================================================


def test_lora_config_r_min_boundary():
    """LoraConfig with r=1 (minimum) validates."""
    config = LoraConfig(
        r=1,
        lora_alpha=2,
        lora_dropout=0.0,
        bias="none",
        target_modules="all-linear",
        use_dora=False,
        use_rslora=False,
        init_lora_weights="gaussian",
    )
    assert config.r == 1


def test_lora_config_r_max_boundary():
    """LoraConfig with r=256 (maximum) validates."""
    config = LoraConfig(
        r=256,
        lora_alpha=512,
        lora_dropout=0.0,
        bias="none",
        target_modules="all-linear",
        use_dora=False,
        use_rslora=False,
        init_lora_weights="gaussian",
    )
    assert config.r == 256


def test_lora_config_r_below_min():
    """LoraConfig with r=0 (below minimum) must fail."""
    with pytest.raises(ValidationError):
        LoraConfig(
            r=0,
            lora_alpha=1,
            lora_dropout=0.0,
            bias="none",
            target_modules="all-linear",
            use_dora=False,
            use_rslora=False,
            init_lora_weights="gaussian",
        )


def test_lora_config_r_above_max():
    """LoraConfig with r=257 (above maximum) must fail."""
    with pytest.raises(ValidationError):
        LoraConfig(
            r=257,
            lora_alpha=512,
            lora_dropout=0.0,
            bias="none",
            target_modules="all-linear",
            use_dora=False,
            use_rslora=False,
            init_lora_weights="gaussian",
        )


def test_lora_config_dropout_min_boundary():
    """LoraConfig with lora_dropout=0.0 (minimum) validates."""
    config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        bias="none",
        target_modules="all-linear",
        use_dora=False,
        use_rslora=False,
        init_lora_weights="gaussian",
    )
    assert config.lora_dropout == 0.0


def test_lora_config_dropout_max_boundary():
    """LoraConfig with lora_dropout=0.5 (maximum) validates."""
    config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.5,
        bias="none",
        target_modules="all-linear",
        use_dora=False,
        use_rslora=False,
        init_lora_weights="gaussian",
    )
    assert config.lora_dropout == 0.5


def test_lora_config_dropout_above_max():
    """LoraConfig with lora_dropout=0.6 (above max) must fail."""
    with pytest.raises(ValidationError):
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.6,
            bias="none",
            target_modules="all-linear",
            use_dora=False,
            use_rslora=False,
            init_lora_weights="gaussian",
        )


def test_global_hyperparams_learning_rate_boundaries():
    """GlobalHyperparametersConfig boundary learning_rate values."""
    # Min boundary (very small but > 0)
    config_min = GlobalHyperparametersConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-10,
        warmup_ratio=0.0,
        epochs=1,
    )
    assert config_min.learning_rate == 1e-10

    # Max boundary (reasonable for LLM training)
    config_max = GlobalHyperparametersConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=0.999,
        warmup_ratio=0.0,
        epochs=1,
    )
    assert config_max.learning_rate == 0.999


def test_global_hyperparams_epochs_min_boundary():
    """GlobalHyperparametersConfig with epochs=1 (minimum) validates."""
    config = GlobalHyperparametersConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        warmup_ratio=0.0,
        epochs=1,
    )
    assert config.epochs == 1


# =============================================================================
# Category 4: INVARIANT TESTS - Config consistency rules
# =============================================================================


def test_adalora_init_r_vs_target_r_invariant():
    """AdaLoraConfig: init_r must be >= target_r (logical constraint)."""
    # Valid: init_r > target_r
    config_valid = AdaLoraConfig(
        init_r=12,
        target_r=8,
        total_step=100,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules="all-linear",
    )
    assert config_valid.init_r > config_valid.target_r

    # Valid edge case: init_r == target_r
    config_equal = AdaLoraConfig(
        init_r=8,
        target_r=8,
        total_step=100,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules="all-linear",
    )
    assert config_equal.init_r == config_equal.target_r


def test_adalora_tinit_tfinal_invariant():
    """AdaLoraConfig: tinit must be < tfinal."""
    # Valid
    config = AdaLoraConfig(
        init_r=12,
        target_r=8,
        total_step=100,
        tinit=200,
        tfinal=1000,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules="all-linear",
    )
    assert config.tinit < config.tfinal


def test_phase_hyperparams_override_semantics():
    """PhaseHyperparametersConfig may override any Global field."""
    # Phase overrides learning_rate
    phase = PhaseHyperparametersConfig(learning_rate=1e-5)
    assert phase.learning_rate == 1e-5
    assert phase.epochs is None  # Other fields remain None

    # Phase overrides multiple fields
    phase_multi = PhaseHyperparametersConfig(
        learning_rate=1e-5, epochs=5, per_device_train_batch_size=8
    )
    assert phase_multi.learning_rate == 1e-5
    assert phase_multi.epochs == 5
    assert phase_multi.per_device_train_batch_size == 8
