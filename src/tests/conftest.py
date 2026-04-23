"""
Shared fixtures for RyotenkAI tests.

Provides:
- Mock configurations (pipeline, training, strategies)
- Test datasets (chat, instruction, text formats)
- Mock models, trainers, and RunPod API
- Common utilities for testing

Usage:
    def test_something(mock_config, test_chat_dataset, mock_model):
        # Use fixtures in your tests
        pass
"""

from __future__ import annotations

import json
import os
import shutil
from contextlib import nullcontext
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest


@pytest.fixture(scope="session", autouse=True)
def _cleanup_repo_root_test_side_effect_dirs():
    """
    Best-effort cleanup for noisy repo-root directories that some tests may create.

    We keep this conservative to avoid deleting user data:
    - Always remove the known test artifact: outputs/models/mock-model-checkpoint (if present).
    - Remove repo-root ./outputs or ./mlruns only if they did NOT exist at session start.
    """
    repo_root = Path(__file__).resolve().parents[2]
    repo_outputs = repo_root / "outputs"
    repo_mlruns = repo_root / "mlruns"

    outputs_existed = repo_outputs.exists()
    mlruns_existed = repo_mlruns.exists()

    yield

    # 1) Always remove the known mock artifact path (safe / test-only)
    mock_dir = repo_outputs / "models" / "mock-model-checkpoint"
    if mock_dir.exists():
        shutil.rmtree(mock_dir, ignore_errors=True)
        # prune empty parents
        for p in (repo_outputs / "models", repo_outputs):
            try:
                p.rmdir()
            except OSError:
                pass

    # 2) If these dirs were created during this test session, remove them
    if not outputs_existed and repo_outputs.exists():
        shutil.rmtree(repo_outputs, ignore_errors=True)

    if not mlruns_existed and repo_mlruns.exists():
        shutil.rmtree(repo_mlruns, ignore_errors=True)

    # 3) If repo-root mlruns/ is empty, remove it (safe even if it existed before)
    if repo_mlruns.exists():
        try:
            repo_mlruns.rmdir()
        except OSError:
            pass


@pytest.fixture(autouse=True)
def _isolate_mlflow_tracking_uri(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """
    Prevent MLflow from creating ./mlruns in the project root during tests.

    If a test (or imported code) uses MLflow without explicitly setting tracking_uri,
    MLflow defaults to a local file-store under the current working directory.
    We redirect that default to a per-test temporary directory.
    """
    if os.getenv("MLFLOW_TRACKING_URI") is None:
        monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file:{tmp_path / 'mlruns'}")
    yield

# =============================================================================
# GLOBAL TEST HYGIENE
# =============================================================================


@pytest.fixture(autouse=True)
def _isolate_hf_secret_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep ``HF_TOKEN`` and friends out of the shared process env.

    Several tests exercise ``StartupValidator.set_hf_token_env`` or construct
    pipelines that mutate ``os.environ["HF_TOKEN"]``. Without isolation those
    writes leak into the ``load_secrets`` happy-path tests, whose assertions
    expect a completely empty token. ``monkeypatch.delenv`` is automatically
    reverted on teardown, so tests that *want* ``HF_TOKEN`` set can still
    opt in explicitly.
    """
    for key in ("HF_TOKEN", "RUNPOD_API_KEY"):
        monkeypatch.delenv(key, raising=False)


# =============================================================================
# PATH FIXTURES
# =============================================================================


@pytest.fixture
def project_root() -> Path:
    """Get project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def test_fixtures_dir() -> Path:
    """Get test fixtures directory."""
    return Path(__file__).parent / "fixtures"


# =============================================================================
# MOCK CONFIGURATION FIXTURES
# =============================================================================


@pytest.fixture
def mock_config():
    """
    Create a minimal mock PipelineConfig for testing.

    This fixture provides a complete mock config that can be used
    for most tests without loading actual YAML files.
    """
    config = MagicMock()

    # Model config
    config.model.name = "test-model/test-llm"
    config.model.tokenizer_name = None
    config.model.torch_dtype = "bfloat16"
    config.model.device_map = "auto"
    config.model.trust_remote_code = False
    config.model.flash_attention = False

    # Training config
    config.training.type = "qlora"
    config.training.get_effective_load_in_4bit = MagicMock(return_value=True)

    # Mock hyperparams
    config.training.hyperparams = MagicMock()
    config.training.hyperparams.per_device_train_batch_size = 4
    config.training.hyperparams.gradient_accumulation_steps = 4
    config.training.hyperparams.warmup_ratio = 0.05
    config.training.hyperparams.lr_scheduler_type = "cosine"
    config.training.hyperparams.fp16 = False
    config.training.hyperparams.bf16 = True
    config.training.hyperparams.gradient_checkpointing = True
    config.training.hyperparams.logging_steps = 10
    config.training.hyperparams.save_steps = 100

    config.training.remote_config_path = "/workspace/config.yaml"

    # Default strategy
    from src.constants import STRATEGY_SFT

    mock_strategy = MagicMock()
    mock_strategy.strategy_type = STRATEGY_SFT
    mock_strategy.dataset = "default"
    mock_strategy.hyperparams = MagicMock()
    mock_strategy.hyperparams.epochs = 1
    mock_strategy.hyperparams.learning_rate = 2e-4
    mock_strategy.hyperparams.beta = None
    config.training.get_strategy_chain.return_value = [mock_strategy]
    config.training.strategies = [mock_strategy]  # Add explicit list for direct access
    config.training.is_multi_phase.return_value = False

    # QLoRA config
    config.qlora.r = 16
    config.qlora.lora_alpha = 32
    config.qlora.lora_dropout = 0.05
    config.qlora.bias = "none"
    config.qlora.target_modules = None
    config.qlora.task_type = "CAUSAL_LM"
    config.qlora.optimizer = "paged_adamw_8bit"

    # Adapter config helper
    config.get_adapter_config.return_value = config.qlora

    # New strict config shape: adapters live under training.*
    config.training.lora = config.qlora
    config.training.adalora = None

    # Dataset config
    mock_dataset = MagicMock()
    mock_dataset.get_source_type = MagicMock(return_value="local")
    mock_dataset.source_local = MagicMock()
    mock_dataset.source_local.local_paths = MagicMock()
    mock_dataset.source_local.local_paths.train = "data/datasets/train.jsonl"
    mock_dataset.source_local.local_paths.eval = "data/datasets/validation.jsonl"
    # training_paths removed in v6.0 - no longer in config
    mock_dataset.source_hf = None
    mock_dataset.max_samples = None
    mock_dataset.adapter_type = "instruction"
    mock_dataset.adapter_config = {"use_chat_template": True}
    mock_dataset.validations = MagicMock()
    mock_dataset.validations.plugins = []
    mock_dataset.validations.mode = "fast"
    mock_dataset.validations.critical_failures = 0
    config.get_primary_dataset.return_value = mock_dataset
    config.get_dataset.return_value = mock_dataset

    # Provider config (mock for tests)
    mock_provider_config = MagicMock()
    mock_provider_config.gpu_type = "NVIDIA RTX 4060"
    mock_provider_config.workspace_path = "/workspace"
    mock_provider_config.mock_mode = True  # Important: always mock for tests
    mock_provider_config.cleanup_workspace = False
    mock_provider_config.keep_on_error = True
    mock_provider_config.training_start_timeout = 120

    config.get_provider_config.return_value = mock_provider_config
    config.get_active_provider_name.return_value = "single_node"

    # HuggingFace config
    config.huggingface.repo_id = "test/test-model"
    config.huggingface.private = True

    # Evaluation config
    config.evaluation.metrics = ["perplexity"]
    config.evaluation.baseline_model = None
    config.evaluation.min_improvement_threshold = 0.05
    config.evaluation.test_batch_size = 4

    return config


@pytest.fixture
def mock_config_multi_phase(mock_config):
    """Create mock config with multi-phase training (CPT → SFT → CoT)."""
    from src.constants import STRATEGY_COT, STRATEGY_CPT, STRATEGY_SFT

    # Create strategy phases
    cpt_phase = MagicMock()
    cpt_phase.strategy_type = STRATEGY_CPT
    cpt_phase.dataset = "corpus"
    cpt_phase.hyperparams = MagicMock()
    cpt_phase.hyperparams.epochs = 1
    cpt_phase.hyperparams.learning_rate = 1e-5
    cpt_phase.hyperparams.beta = None

    sft_phase = MagicMock()
    sft_phase.strategy_type = STRATEGY_SFT
    sft_phase.dataset = "default"
    sft_phase.hyperparams = MagicMock()
    sft_phase.hyperparams.epochs = 2
    sft_phase.hyperparams.learning_rate = 2e-4
    sft_phase.hyperparams.beta = None

    cot_phase = MagicMock()
    cot_phase.strategy_type = STRATEGY_COT
    cot_phase.dataset = "default"
    cot_phase.hyperparams = MagicMock()
    cot_phase.hyperparams.epochs = 1
    cot_phase.hyperparams.learning_rate = 1e-4
    cot_phase.hyperparams.beta = None

    mock_config.training.get_strategy_chain.return_value = [cpt_phase, sft_phase, cot_phase]
    mock_config.training.strategies = [cpt_phase, sft_phase, cot_phase]  # Add explicit list
    mock_config.training.is_multi_phase.return_value = True

    return mock_config


@pytest.fixture
def mock_secrets():
    """Create mock Secrets."""
    secrets = MagicMock()
    secrets.runpod_api_key = "test_runpod_key"
    secrets.hf_token = "test_hf_token"
    return secrets


# =============================================================================
# TEST DATASET FIXTURES
# =============================================================================


@pytest.fixture
def test_chat_dataset(tmp_path) -> Path:
    """
    Create test chat dataset (10 samples).

    Format: {"messages": [{"role": "user/assistant", "content": "..."}]}
    """
    data = [
        {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"What is question number {i}?"},
                {
                    "role": "assistant",
                    "content": f"This is answer number {i}. It contains detailed information about the topic.",
                },
            ]
        }
        for i in range(10)
    ]

    path = tmp_path / "test_chat.jsonl"
    with path.open("w") as f:
        for sample in data:
            f.write(json.dumps(sample) + "\n")

    return path


@pytest.fixture
def test_instruction_dataset(tmp_path) -> Path:
    """
    Create test instruction dataset (10 samples).

    Format: {"instruction": "...", "response": "..."}
    """
    data = [
        {
            "instruction": f"Explain concept number {i} in detail.",
            "response": f"Concept {i} is an important topic. Here is a detailed explanation of how it works and why it matters.",
        }
        for i in range(10)
    ]

    path = tmp_path / "test_instruction.jsonl"
    with path.open("w") as f:
        for sample in data:
            f.write(json.dumps(sample) + "\n")

    return path


@pytest.fixture
def test_text_dataset(tmp_path) -> Path:
    """
    Create test text dataset (10 samples).

    Format: {"text": "..."}
    """
    data = [
        {
            "text": f"This is sample text number {i}. It contains multiple sentences about various topics. "
            f"The purpose is to test the text adapter functionality with sufficient length."
        }
        for i in range(10)
    ]

    path = tmp_path / "test_text.jsonl"
    with path.open("w") as f:
        for sample in data:
            f.write(json.dumps(sample) + "\n")

    return path


@pytest.fixture
def test_invalid_dataset(tmp_path) -> Path:
    """Create invalid dataset for negative testing."""
    data = [
        {"invalid_field": "no proper fields"},
        {"also_invalid": 123},
    ]

    path = tmp_path / "test_invalid.jsonl"
    with path.open("w") as f:
        for sample in data:
            f.write(json.dumps(sample) + "\n")

    return path


@pytest.fixture
def test_empty_dataset(tmp_path) -> Path:
    """Create empty dataset for negative testing."""
    path = tmp_path / "test_empty.jsonl"
    path.touch()
    return path


# =============================================================================
# MOCK MODEL FIXTURES
# =============================================================================


@pytest.fixture
def mock_model():
    """
    Create mock PreTrainedModel.

    Provides essential model interface without loading real weights.
    """
    model = MagicMock()
    model.config = MagicMock()
    model.config.model_type = "qwen2"
    model.config.hidden_size = 768
    model.config.num_hidden_layers = 12
    model.config.vocab_size = 32000

    # Parameters for trainable param counting
    mock_param = MagicMock()
    mock_param.numel.return_value = 1000
    mock_param.requires_grad = True
    model.parameters.return_value = [mock_param] * 10

    # PEFT compatibility
    model.print_trainable_parameters = MagicMock()

    # Save methods
    model.save_pretrained = MagicMock()

    # Device
    model.device = "cpu"

    return model


@pytest.fixture
def mock_tokenizer():
    """
    Create mock PreTrainedTokenizer.

    Provides essential tokenizer interface for testing.
    """
    tokenizer = MagicMock()
    tokenizer.pad_token = "<pad>"
    tokenizer.pad_token_id = 0
    tokenizer.eos_token = "</s>"
    tokenizer.eos_token_id = 2
    tokenizer.bos_token = "<s>"
    tokenizer.bos_token_id = 1
    tokenizer.model_max_length = 2048

    # Chat template
    def mock_apply_chat_template(messages, tokenize=False, add_generation_prompt=False):
        result = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            result += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        if add_generation_prompt:
            result += "<|im_start|>assistant\n"
        return result

    tokenizer.apply_chat_template = mock_apply_chat_template
    tokenizer.save_pretrained = MagicMock()

    return tokenizer


# =============================================================================
# MOCK TRAINER FIXTURES
# =============================================================================


@pytest.fixture
def mock_trainer():
    """
    Create mock SFTTrainer.

    Simulates training without actual computation.
    """
    trainer = MagicMock()

    # Training state
    trainer.state = MagicMock()
    trainer.state.global_step = 100
    trainer.state.epoch = 1.0
    trainer.state.log_history = [
        {"loss": 2.5, "step": 10},
        {"loss": 2.0, "step": 50},
        {"loss": 1.5, "step": 100},
    ]

    # Train method
    def mock_train():
        return MagicMock(
            training_loss=1.5,
            metrics={"train_loss": 1.5, "epoch": 1.0},
        )

    trainer.train = mock_train

    # Save method
    trainer.save_model = MagicMock()

    # Model access
    trainer.model = MagicMock()

    return trainer


# =============================================================================
# MOCK MEMORY MANAGER FIXTURES
# =============================================================================


@pytest.fixture
def mock_memory_manager():
    """
    Create mock MemoryManager for testing without GPU.

    Implements IMemoryManager interface.
    """

    class MockMemoryManager:
        """Mock memory manager for testing."""

        def __init__(self):
            self.operations: list[str] = []
            self._is_critical = False

        def get_memory_stats(self) -> dict[str, Any]:
            return {
                "allocated_mb": 1000,
                "cached_mb": 500,
                "free_mb": 6500,
                "total_mb": 8000,
            }

        def is_memory_critical(self) -> bool:
            return self._is_critical

        def clear_cache(self) -> int:
            self.operations.append("clear_cache")
            return 100

        def aggressive_cleanup(self) -> int:
            self.operations.append("aggressive_cleanup")
            return 200

        def get_training_recommendations(self) -> dict[str, Any]:
            return {
                "gpu_name": "MockGPU RTX 4060",
                "gpu_tier": "testing",
                "total_vram_gb": 8,
                "max_batch_size": 4,
                "max_model": "7B",
            }

        def safe_operation(self, operation_name: str):
            self.operations.append(f"safe_operation:{operation_name}")
            return nullcontext()

        def set_critical(self, critical: bool):
            """Helper for testing OOM scenarios."""
            self._is_critical = critical

    return MockMemoryManager()


# =============================================================================
# MOCK RUNPOD FIXTURES
# =============================================================================


@pytest.fixture
def mock_runpod_api():
    """
    Create mock RunPod API client.

    Simulates RunPod operations without network calls.
    """

    class MockRunPodAPI:
        """Mock RunPod API for testing."""

        def __init__(self):
            self.created_pods: list[dict] = []
            self.terminated_pods: list[str] = []
            self._next_pod_id = "pod_test_123"
            self._pod_status = "RUNNING"

        def create_pod(self, **kwargs) -> dict:
            pod = {
                "id": self._next_pod_id,
                "status": self._pod_status,
                "machine": {"gpu": "NVIDIA RTX 4060"},
                "costPerHr": 0.5,
                **kwargs,
            }
            self.created_pods.append(pod)
            return pod

        def query_pod(self, pod_id: str) -> dict | None:
            for pod in self.created_pods:
                if pod["id"] == pod_id:
                    return pod
            return None

        def terminate_pod(self, pod_id: str) -> bool:
            self.terminated_pods.append(pod_id)
            return True

        def set_pod_status(self, status: str):
            """Helper for testing different pod states."""
            self._pod_status = status
            for pod in self.created_pods:
                pod["status"] = status

    return MockRunPodAPI()


@pytest.fixture
def mock_runpod_ssh():
    """
    Create mock RunPod SSH client.

    Simulates SSH operations without actual connections.
    """

    class MockRunPodSSH:
        """Mock SSH client for testing."""

        def __init__(self):
            self.commands_executed: list[str] = []
            self.files_uploaded: list[tuple[str, str]] = []
            self.files_downloaded: list[tuple[str, str]] = []
            self._connected = False
            self._should_fail = False

        def test_connection(self) -> tuple[bool, str]:
            if self._should_fail:
                return False, "Connection refused"
            self._connected = True
            return True, ""

        def exec_command(self, cmd: str) -> tuple[bool, str]:
            if self._should_fail:
                return False, "Command failed"
            self.commands_executed.append(cmd)
            return True, ""

        def upload_file(self, local: str, remote: str) -> tuple[bool, str]:
            if self._should_fail:
                return False, "Upload failed"
            self.files_uploaded.append((local, remote))
            return True, ""

        def download_directory(self, remote: str, local: str) -> tuple[bool, str]:
            if self._should_fail:
                return False, "Download failed"
            self.files_downloaded.append((remote, local))
            return True, ""

        def set_should_fail(self, fail: bool):
            """Helper for testing failure scenarios."""
            self._should_fail = fail

    return MockRunPodSSH()


# =============================================================================
# MOCK FACTORY FIXTURES
# =============================================================================


@pytest.fixture
def mock_strategy_factory():
    """Create mock StrategyFactory for DI testing."""

    class MockStrategyFactory:
        def __init__(self):
            self.created: list[str] = []

        def create(self, strategy_type: str, config: Any) -> Any:
            self.created.append(strategy_type)
            strategy = MagicMock()
            strategy.get_metadata.return_value = MagicMock(
                name=f"{strategy_type}_strategy",
                strategy_type=strategy_type,
            )
            strategy.validate_dataset.return_value = MagicMock(is_success=lambda: True)
            return strategy

        def list_strategies(self) -> list[str]:
            from src.constants import ALL_STRATEGIES

            return list(ALL_STRATEGIES)

    return MockStrategyFactory()


@pytest.fixture
def mock_trainer_factory():
    """Create mock TrainerFactory for DI testing."""

    class MockTrainerFactory:
        def __init__(self):
            self.created: list[str] = []

        def create(
            self,
            strategy_type: str,
            model: Any,
            tokenizer: Any,
            train_dataset: Any,
            config: Any,
            **kwargs,
        ) -> Any:
            self.created.append(strategy_type)
            return MagicMock()

    return MockTrainerFactory()


# =============================================================================
# MOCK NOTIFIER FIXTURES
# =============================================================================


@pytest.fixture
def mock_notifier():
    """Create mock ICompletionNotifier for testing."""

    class MockNotifier:
        def __init__(self):
            self.complete_calls: list[dict] = []
            self.failed_calls: list[tuple[str, dict]] = []

        def notify_complete(self, data: dict[str, Any]) -> None:
            self.complete_calls.append(data)

        def notify_failed(self, error: str, data: dict[str, Any]) -> None:
            self.failed_calls.append((error, data))

    return MockNotifier()


# =============================================================================
# UTILITY FIXTURES
# =============================================================================


@pytest.fixture
def temp_output_dir(tmp_path) -> Path:
    """Create temporary output directory for tests."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def mock_dataset():
    """Create mock HuggingFace Dataset."""
    dataset = MagicMock()
    dataset.__len__ = MagicMock(return_value=10)
    dataset.column_names = ["messages"]
    dataset.map = MagicMock(return_value=dataset)
    dataset.filter = MagicMock(return_value=dataset)

    # Make iterable
    sample = {"messages": [{"role": "user", "content": "test"}]}
    dataset.__iter__ = MagicMock(return_value=iter([sample] * 10))

    return dataset


# =============================================================================
# EVALUATION TEST DOUBLES
# =============================================================================


@pytest.fixture
def mock_judge_provider():
    """MockJudgeProvider — test double for IJudgeProvider. Returns a fixed score."""

    from src.evaluation.plugins.llm_judge.interface import JudgeResponse

    class MockJudgeProvider:
        def __init__(self, fixed_score: int = 4):
            self.fixed_score = fixed_score
            self.calls: list[dict] = []

        def judge(self, question: str, expected: str, model_answer: str) -> JudgeResponse:
            self.calls.append({"question": question, "expected": expected, "model_answer": model_answer})
            return JudgeResponse(
                score=self.fixed_score,
                reasoning="mock reasoning",
                raw_response='{"score": %d, "reasoning": "mock reasoning"}' % self.fixed_score,
            )

    return MockJudgeProvider


@pytest.fixture
def mock_secrets_resolver():
    """MockSecretsResolver — test double for SecretsResolver."""

    class MockSecretsResolver:
        def __init__(self, data: dict[str, str]):
            self.data = data

        def resolve(self, keys: tuple[str, ...]) -> dict[str, str]:
            return {k: self.data[k] for k in keys}

    return MockSecretsResolver


# =============================================================================
# SKIP MARKERS
# =============================================================================


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "gpu: marks tests that require GPU (deselect with '-m \"not gpu\"')")
    config.addinivalue_line("markers", "integration: marks integration tests (deselect with '-m \"not integration\"')")
    config.addinivalue_line("markers", "e2e: marks end-to-end tests (deselect with '-m \"not e2e\"')")
