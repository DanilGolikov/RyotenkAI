"""
Unit tests for Training Strategies (SFT, CPT, CoT).

Tests prepare_dataset, validate_dataset, and get_metadata methods
for all training strategy types.

Updated for Phase 3: TRL-native integration (no manual preprocessing).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.training.strategies.base import StrategyMetadata
from src.training.strategies.cot import CoTStrategy
from src.training.strategies.cpt import CPTStrategy
from src.training.strategies.sft import SFTStrategy

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_pipeline_config():
    """Create mock config for strategies."""
    config = MagicMock()
    config.training.hyperparams.per_device_train_batch_size = 4
    return config


@pytest.fixture
def mock_tokenizer():
    """Create mock tokenizer for strategies."""
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 2
    tokenizer.model_max_length = 2048

    def apply_chat_template(messages, tokenize=False):
        return " ".join([m.get("content", "") for m in messages])

    tokenizer.apply_chat_template = apply_chat_template

    def tokenize(text, truncation=True, max_length=512, padding=False):
        return {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}

    tokenizer.side_effect = tokenize
    tokenizer.return_value = {"input_ids": [[1, 2, 3]], "attention_mask": [[1, 1, 1]]}

    return tokenizer


@pytest.fixture
def sft_strategy(mock_pipeline_config):
    """Create SFTStrategy instance."""
    return SFTStrategy(mock_pipeline_config)


@pytest.fixture
def cpt_strategy(mock_pipeline_config):
    """Create CPTStrategy instance."""
    return CPTStrategy(mock_pipeline_config)


@pytest.fixture
def cot_strategy(mock_pipeline_config):
    """Create CoTStrategy instance."""
    return CoTStrategy(mock_pipeline_config)


# =============================================================================
# TEST CLASS: SFTStrategy
# =============================================================================


class TestSFTStrategy:
    """Unit tests for SFTStrategy."""

    def test_validate_dataset_with_messages(self, sft_strategy):
        """
        Given: Dataset with messages field
        When: validate_dataset is called
        Then: Returns Ok(True)
        """
        mock_dataset = MagicMock()
        mock_dataset.column_names = ["messages"]

        result = sft_strategy.validate_dataset(mock_dataset)

        assert result.is_success()
        assert result.unwrap() is True

    def test_validate_dataset_with_instruction(self, sft_strategy):
        """
        Given: Dataset with instruction field
        When: validate_dataset is called
        Then: Returns Ok(True)
        """
        mock_dataset = MagicMock()
        mock_dataset.column_names = ["instruction", "response"]

        result = sft_strategy.validate_dataset(mock_dataset)

        assert result.is_success()

    def test_validate_dataset_with_text(self, sft_strategy):
        """
        Given: Dataset with text field
        When: validate_dataset is called
        Then: Returns Ok(True) - SFT now accepts text format too!
        """
        mock_dataset = MagicMock()
        mock_dataset.column_names = ["text"]

        result = sft_strategy.validate_dataset(mock_dataset)

        assert result.is_success()

    def test_validate_dataset_missing_fields(self, sft_strategy):
        """
        Given: Dataset without any valid field
        When: validate_dataset is called
        Then: Returns Err
        """
        mock_dataset = MagicMock()
        mock_dataset.column_names = ["random_field"]  # No valid field

        result = sft_strategy.validate_dataset(mock_dataset)

        assert result.is_failure()

    def test_prepare_dataset_messages_format(self, sft_strategy, mock_tokenizer):
        """
        Given: Dataset in chat format (messages)
        When: prepare_dataset is called
        Then: Returns dataset as-is (TRL handles formatting)
        """
        mock_dataset = MagicMock()
        mock_dataset.column_names = ["messages"]
        mock_dataset.__len__ = MagicMock(return_value=10)

        result = sft_strategy.prepare_dataset(mock_dataset, mock_tokenizer)

        assert result.is_success()
        # TRL handles messages natively, no .map() needed
        mock_dataset.map.assert_not_called()

    def test_prepare_dataset_text_format(self, sft_strategy, mock_tokenizer):
        """
        Given: Dataset with text column
        When: prepare_dataset is called
        Then: Returns dataset as-is (TRL handles text)
        """
        mock_dataset = MagicMock()
        mock_dataset.column_names = ["text"]
        mock_dataset.__len__ = MagicMock(return_value=10)

        result = sft_strategy.prepare_dataset(mock_dataset, mock_tokenizer)

        assert result.is_success()
        mock_dataset.map.assert_not_called()

    def test_prepare_dataset_instruction_format(self, sft_strategy, mock_tokenizer):
        """
        Given: Dataset in instruction format (legacy)
        When: prepare_dataset is called
        Then: Converts to text format
        """
        mock_dataset = MagicMock()
        mock_dataset.column_names = ["instruction", "response"]
        mock_dataset.map.return_value = MagicMock()
        mock_dataset.map.return_value.__len__ = MagicMock(return_value=10)

        result = sft_strategy.prepare_dataset(mock_dataset, mock_tokenizer)

        assert result.is_success()
        # Legacy format needs conversion
        mock_dataset.map.assert_called_once()

    def test_prepare_dataset_failure(self, sft_strategy, mock_tokenizer):
        """
        Given: Dataset without valid format
        When: prepare_dataset is called
        Then: Returns Err
        """
        mock_dataset = MagicMock()
        mock_dataset.column_names = ["random"]

        result = sft_strategy.prepare_dataset(mock_dataset, mock_tokenizer)

        assert result.is_failure()

    def test_get_training_objective(self, sft_strategy):
        """
        Given: SFTStrategy
        When: get_training_objective is called
        Then: Returns supervised_learning
        """
        objective = sft_strategy.get_training_objective()

        assert objective == "supervised_learning"

    def test_get_trainer_type(self, sft_strategy):
        """
        Given: SFTStrategy
        When: get_trainer_type is called
        Then: Returns 'sft'
        """
        trainer_type = sft_strategy.get_trainer_type()

        assert trainer_type == "sft"

    def test_get_metadata(self, sft_strategy):
        """
        Given: SFTStrategy
        When: get_metadata is called
        Then: Returns valid StrategyMetadata
        """
        metadata = sft_strategy.get_metadata()

        assert isinstance(metadata, StrategyMetadata)
        assert metadata.strategy_type == "sft"
        assert "sft" in metadata.name
        assert metadata.version is not None


# =============================================================================
# TEST CLASS: CPTStrategy
# =============================================================================


class TestCPTStrategy:
    """Unit tests for CPTStrategy."""

    def test_validate_dataset_with_text(self, cpt_strategy):
        """
        Given: Dataset with text field
        When: validate_dataset is called
        Then: Returns Ok(True)
        """
        mock_dataset = MagicMock()
        mock_dataset.column_names = ["text"]

        result = cpt_strategy.validate_dataset(mock_dataset)

        assert result.is_success()
        assert result.unwrap() is True

    def test_validate_dataset_missing_text(self, cpt_strategy):
        """
        Given: Dataset without text field
        When: validate_dataset is called
        Then: Returns Err
        """
        mock_dataset = MagicMock()
        mock_dataset.column_names = ["content"]  # Wrong field name

        result = cpt_strategy.validate_dataset(mock_dataset)

        assert result.is_failure()
        assert "text" in str(result.unwrap_err())

    def test_prepare_dataset_success(self, cpt_strategy, mock_tokenizer):
        """
        Given: Dataset with text field
        When: prepare_dataset is called
        Then: Returns dataset as-is (TRL handles tokenization)
        """
        mock_dataset = MagicMock()
        mock_dataset.column_names = ["text"]
        mock_dataset.__len__ = MagicMock(return_value=10)

        result = cpt_strategy.prepare_dataset(mock_dataset, mock_tokenizer)

        assert result.is_success()
        # TRL handles text natively, no .map() needed
        mock_dataset.map.assert_not_called()

    def test_prepare_dataset_missing_text(self, cpt_strategy, mock_tokenizer):
        """
        Given: Dataset without text field
        When: prepare_dataset is called
        Then: Returns Err
        """
        mock_dataset = MagicMock()
        mock_dataset.column_names = ["content"]

        result = cpt_strategy.prepare_dataset(mock_dataset, mock_tokenizer)

        assert result.is_failure()

    def test_get_training_objective(self, cpt_strategy):
        """
        Given: CPTStrategy
        When: get_training_objective is called
        Then: Returns language_modeling
        """
        objective = cpt_strategy.get_training_objective()

        assert objective == "language_modeling"

    def test_get_trainer_type(self, cpt_strategy):
        """
        Given: CPTStrategy
        When: get_trainer_type is called
        Then: Returns 'sft' (SFTTrainer works for CPT)
        """
        trainer_type = cpt_strategy.get_trainer_type()

        assert trainer_type == "sft"

    def test_get_metadata(self, cpt_strategy):
        """
        Given: CPTStrategy
        When: get_metadata is called
        Then: Returns valid StrategyMetadata
        """
        metadata = cpt_strategy.get_metadata()

        assert isinstance(metadata, StrategyMetadata)
        assert metadata.strategy_type == "cpt"
        assert "cpt" in metadata.name
        assert metadata.data_format == "text"

    def test_get_recommended_hyperparameters(self, cpt_strategy):
        """
        Given: CPTStrategy
        When: get_recommended_hyperparameters is called
        Then: Returns dict with CPT-specific settings
        """
        hyperparams = cpt_strategy.get_recommended_hyperparameters()

        assert isinstance(hyperparams, dict)
        assert "learning_rate" in hyperparams
        # CPT uses lower LR than SFT
        assert hyperparams["learning_rate"] <= 1e-4


# =============================================================================
# TEST CLASS: CoTStrategy
# =============================================================================


class TestCoTStrategy:
    """Unit tests for CoTStrategy."""

    def test_validate_dataset_with_messages(self, cot_strategy):
        """
        Given: Dataset with messages field (ChatML format)
        When: validate_dataset is called
        Then: Returns Ok(True)
        """
        mock_dataset = MagicMock()
        mock_dataset.column_names = ["messages"]

        result = cot_strategy.validate_dataset(mock_dataset)

        assert result.is_success()

    def test_validate_dataset_valid(self, cot_strategy):
        """
        Given: Dataset with instruction, reasoning, answer
        When: validate_dataset is called
        Then: Returns Ok(True)
        """
        mock_dataset = MagicMock()
        mock_dataset.column_names = ["instruction", "reasoning", "answer"]

        result = cot_strategy.validate_dataset(mock_dataset)

        assert result.is_success()

    def test_validate_dataset_with_chain_of_thought(self, cot_strategy):
        """
        Given: Dataset with chain_of_thought instead of reasoning
        When: validate_dataset is called
        Then: Returns Ok(True)
        """
        mock_dataset = MagicMock()
        mock_dataset.column_names = ["instruction", "chain_of_thought", "answer"]

        result = cot_strategy.validate_dataset(mock_dataset)

        assert result.is_success()

    def test_validate_dataset_missing_reasoning(self, cot_strategy):
        """
        Given: Dataset without reasoning field
        When: validate_dataset is called
        Then: Returns Err
        """
        mock_dataset = MagicMock()
        mock_dataset.column_names = ["instruction", "answer"]  # Missing reasoning

        result = cot_strategy.validate_dataset(mock_dataset)

        assert result.is_failure()
        assert "reasoning" in str(result.unwrap_err()) or "chain_of_thought" in str(result.unwrap_err())

    def test_validate_dataset_missing_answer(self, cot_strategy):
        """
        Given: Dataset without answer field
        When: validate_dataset is called
        Then: Returns Err
        """
        mock_dataset = MagicMock()
        mock_dataset.column_names = ["instruction", "reasoning"]  # Missing answer

        result = cot_strategy.validate_dataset(mock_dataset)

        assert result.is_failure()

    def test_prepare_dataset_messages_format(self, cot_strategy, mock_tokenizer):
        """
        Given: Dataset with messages format
        When: prepare_dataset is called
        Then: Returns dataset as-is (TRL handles ChatML)
        """
        mock_dataset = MagicMock()
        mock_dataset.column_names = ["messages"]
        mock_dataset.__len__ = MagicMock(return_value=10)

        result = cot_strategy.prepare_dataset(mock_dataset, mock_tokenizer)

        assert result.is_success()
        mock_dataset.map.assert_not_called()

    def test_prepare_dataset_success(self, cot_strategy, mock_tokenizer):
        """
        Given: Valid CoT dataset with explicit fields
        When: prepare_dataset is called
        Then: Returns formatted dataset with think/answer tags
        """
        mock_dataset = MagicMock()
        mock_dataset.column_names = ["instruction", "reasoning", "answer"]
        mock_dataset.map.return_value = MagicMock()
        mock_dataset.map.return_value.__len__ = MagicMock(return_value=10)

        result = cot_strategy.prepare_dataset(mock_dataset, mock_tokenizer)

        assert result.is_success()
        mock_dataset.map.assert_called_once()

    def test_get_training_objective(self, cot_strategy):
        """
        Given: CoTStrategy
        When: get_training_objective is called
        Then: Returns chain_of_thought_learning
        """
        objective = cot_strategy.get_training_objective()

        assert objective == "chain_of_thought_learning"

    def test_get_trainer_type(self, cot_strategy):
        """
        Given: CoTStrategy
        When: get_trainer_type is called
        Then: Returns 'sft'
        """
        trainer_type = cot_strategy.get_trainer_type()

        assert trainer_type == "sft"

    def test_get_metadata(self, cot_strategy):
        """
        Given: CoTStrategy
        When: get_metadata is called
        Then: Returns valid StrategyMetadata
        """
        metadata = cot_strategy.get_metadata()

        assert isinstance(metadata, StrategyMetadata)
        assert metadata.strategy_type == "cot"
        assert "cot" in metadata.name
        assert "reasoning" in metadata.objective or "chain_of_thought" in metadata.description.lower()

    def test_get_recommended_hyperparameters(self, cot_strategy):
        """
        Given: CoTStrategy
        When: get_recommended_hyperparameters is called
        Then: Returns dict with CoT-specific settings (longer sequences)
        """
        hyperparams = cot_strategy.get_recommended_hyperparameters()

        assert isinstance(hyperparams, dict)
        assert "max_seq_length" in hyperparams
        # CoT needs longer context for reasoning
        assert hyperparams["max_seq_length"] >= 4096


# =============================================================================
# TEST CLASS: StrategyFactory
# =============================================================================


class TestStrategyFactory:
    """Tests for StrategyFactory."""

    def test_create_sft_strategy(self, mock_pipeline_config):
        """
        Given: strategy_type='sft'
        When: StrategyFactory().create is called
        Then: Returns SFTStrategy
        """
        from src.training.strategies.factory import StrategyFactory

        factory = StrategyFactory()
        strategy = factory.create("sft", mock_pipeline_config)

        assert isinstance(strategy, SFTStrategy)

    def test_create_cpt_strategy(self, mock_pipeline_config):
        """
        Given: strategy_type='cpt'
        When: StrategyFactory().create is called
        Then: Returns CPTStrategy
        """
        from src.training.strategies.factory import StrategyFactory

        factory = StrategyFactory()
        strategy = factory.create("cpt", mock_pipeline_config)

        assert isinstance(strategy, CPTStrategy)

    def test_create_cot_strategy(self, mock_pipeline_config):
        """
        Given: strategy_type='cot'
        When: StrategyFactory().create is called
        Then: Returns CoTStrategy
        """
        from src.training.strategies.factory import StrategyFactory

        factory = StrategyFactory()
        strategy = factory.create("cot", mock_pipeline_config)

        assert isinstance(strategy, CoTStrategy)

    def test_create_unknown_strategy(self, mock_pipeline_config):
        """
        Given: Unknown strategy_type
        When: StrategyFactory().create is called
        Then: Raises ValueError
        """
        from src.training.strategies.factory import StrategyFactory

        factory = StrategyFactory()
        with pytest.raises(ValueError):
            factory.create("unknown", mock_pipeline_config)

    def test_list_strategies(self):
        """
        Given: StrategyFactory
        When: list_available is called
        Then: Returns dict with sft, cpt, cot
        """
        from src.training.strategies.factory import StrategyFactory

        factory = StrategyFactory()
        strategies = factory.list_available()

        assert "sft" in strategies
        assert "cpt" in strategies
        assert "cot" in strategies

    def test_get_default_hyperparameters(self):
        """
        Given: StrategyFactory
        When: get_default_hyperparameters is called
        Then: Returns dict with defaults for strategy
        """
        from src.training.strategies.factory import StrategyFactory

        factory = StrategyFactory()
        defaults = factory.get_default_hyperparameters("sft")

        assert isinstance(defaults, dict)


# =============================================================================
# TEST CLASS: Strategy Base Interface
# =============================================================================


class TestStrategyInterface:
    """Tests for TrainingStrategy interface compliance."""

    def test_sft_implements_interface(self, sft_strategy):
        """
        Given: SFTStrategy
        Then: Implements TrainingStrategy interface
        """
        assert hasattr(sft_strategy, "prepare_dataset")
        assert hasattr(sft_strategy, "validate_dataset")
        assert hasattr(sft_strategy, "get_training_objective")
        assert hasattr(sft_strategy, "get_metadata")
        assert hasattr(sft_strategy, "get_trainer_type")
        assert callable(sft_strategy.prepare_dataset)
        assert callable(sft_strategy.validate_dataset)
        assert callable(sft_strategy.get_trainer_type)

    def test_cpt_implements_interface(self, cpt_strategy):
        """
        Given: CPTStrategy
        Then: Implements TrainingStrategy interface
        """
        assert hasattr(cpt_strategy, "prepare_dataset")
        assert hasattr(cpt_strategy, "validate_dataset")
        assert hasattr(cpt_strategy, "get_training_objective")
        assert hasattr(cpt_strategy, "get_metadata")
        assert hasattr(cpt_strategy, "get_trainer_type")

    def test_cot_implements_interface(self, cot_strategy):
        """
        Given: CoTStrategy
        Then: Implements TrainingStrategy interface
        """
        assert hasattr(cot_strategy, "prepare_dataset")
        assert hasattr(cot_strategy, "validate_dataset")
        assert hasattr(cot_strategy, "get_training_objective")
        assert hasattr(cot_strategy, "get_metadata")
        assert hasattr(cot_strategy, "get_trainer_type")

    def test_all_strategies_return_result(self, sft_strategy, cpt_strategy, cot_strategy):
        """
        Given: All strategies
        When: validate_dataset is called
        Then: Returns Result type (Ok or Err)
        """
        mock_dataset = MagicMock()
        mock_dataset.column_names = []

        for strategy in [sft_strategy, cpt_strategy, cot_strategy]:
            result = strategy.validate_dataset(mock_dataset)
            assert hasattr(result, "is_success") or hasattr(result, "is_failure")


# =============================================================================
# ADDITIONAL FIXTURES
# =============================================================================


@pytest.fixture
def orpo_strategy(mock_pipeline_config):
    """Create ORPOStrategy instance."""
    from src.training.strategies.orpo import ORPOStrategy

    return ORPOStrategy(mock_pipeline_config)


@pytest.fixture
def dpo_strategy(mock_pipeline_config):
    """Create DPOStrategy instance."""
    from src.training.strategies.dpo import DPOStrategy

    return DPOStrategy(mock_pipeline_config)


@pytest.fixture
def sapo_strategy(mock_pipeline_config):
    """Create SAPOStrategy instance."""
    from src.training.strategies.sapo import SAPOStrategy

    return SAPOStrategy(mock_pipeline_config)


@pytest.fixture
def sapo_tokenizer():
    """Mock tokenizer that accepts add_generation_prompt kwarg (needed for SAPO)."""
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 2
    tokenizer.model_max_length = 2048
    tokenizer.chat_template = "{% for m in messages %}{{ m.content }}{% endfor %}"

    def apply_chat_template(messages, tokenize=False, **kwargs):
        return " ".join([m.get("content", "") for m in messages])

    tokenizer.apply_chat_template = apply_chat_template
    return tokenizer


# =============================================================================
# HELPERS
# =============================================================================


def _make_preference_dataset(n: int = 1):
    """Build a valid chosen/rejected Dataset."""
    from datasets import Dataset

    row = {
        "chosen": [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
        ],
        "rejected": [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "I don't know."},
        ],
    }
    return Dataset.from_list([row] * n)


# =============================================================================
# TEST CLASS: ORPOStrategy
# =============================================================================


class TestORPOStrategy:
    """Unit tests for ORPOStrategy."""

    # --- validate_dataset: positive ---

    def test_validate_dataset_valid_data(self, orpo_strategy):
        """
        Given: Dataset with valid chosen/rejected message pairs
        When: validate_dataset is called
        Then: Returns Ok(True)
        """
        ds = _make_preference_dataset()
        result = orpo_strategy.validate_dataset(ds)

        assert result.is_success()
        assert result.unwrap() is True

    def test_validate_dataset_empty_chosen_list_boundary(self, orpo_strategy):
        """
        Given: Dataset where chosen/rejected are empty lists
        When: validate_dataset is called
        Then: Returns Ok(True) — skips per-message structure checks
        """
        from datasets import Dataset

        ds = Dataset.from_list([{"chosen": [], "rejected": []}])
        result = orpo_strategy.validate_dataset(ds)

        assert result.is_success()

    # --- validate_dataset: error branches ---

    def test_validate_dataset_missing_chosen_column(self, orpo_strategy):
        """
        Given: Dataset with only 'rejected' column
        When: validate_dataset is called
        Then: Returns Err with code ORPO_MISSING_CHOSEN_COLUMN
        """
        from datasets import Dataset

        ds = Dataset.from_list([{"rejected": [{"role": "user", "content": "Q"}]}])
        result = orpo_strategy.validate_dataset(ds)

        assert result.is_failure()
        assert result.unwrap_err().code == "ORPO_MISSING_CHOSEN_COLUMN"

    def test_validate_dataset_missing_rejected_column(self, orpo_strategy):
        """
        Given: Dataset with only 'chosen' column
        When: validate_dataset is called
        Then: Returns Err with code ORPO_MISSING_REJECTED_COLUMN
        """
        from datasets import Dataset

        ds = Dataset.from_list([{"chosen": [{"role": "user", "content": "Q"}]}])
        result = orpo_strategy.validate_dataset(ds)

        assert result.is_failure()
        assert result.unwrap_err().code == "ORPO_MISSING_REJECTED_COLUMN"

    def test_validate_dataset_invalid_message_format_not_lists(self, orpo_strategy):
        """
        Given: chosen/rejected are strings (not lists)
        When: validate_dataset is called
        Then: Returns Err with code ORPO_INVALID_MESSAGE_FORMAT
        """
        from datasets import Dataset

        ds = Dataset.from_list([{"chosen": "plain string", "rejected": "another string"}])
        result = orpo_strategy.validate_dataset(ds)

        assert result.is_failure()
        assert result.unwrap_err().code == "ORPO_INVALID_MESSAGE_FORMAT"

    def test_validate_dataset_invalid_message_structure_not_dicts(self, orpo_strategy):
        """
        Given: chosen is a list of strings (not dicts)
        When: validate_dataset is called
        Then: Returns Err with code ORPO_INVALID_MESSAGE_STRUCTURE
        """
        from datasets import Dataset

        ds = Dataset.from_list([{"chosen": ["just a string"], "rejected": ["another"]}])
        result = orpo_strategy.validate_dataset(ds)

        assert result.is_failure()
        assert result.unwrap_err().code == "ORPO_INVALID_MESSAGE_STRUCTURE"

    def test_validate_dataset_missing_message_keys(self, orpo_strategy):
        """
        Given: Messages are dicts but missing 'role' / 'content' keys
        When: validate_dataset is called
        Then: Returns Err with code ORPO_MISSING_MESSAGE_KEYS
        """
        from datasets import Dataset

        ds = Dataset.from_list(
            [
                {
                    "chosen": [{"text": "no role/content here"}],
                    "rejected": [{"text": "missing keys too"}],
                }
            ]
        )
        result = orpo_strategy.validate_dataset(ds)

        assert result.is_failure()
        assert result.unwrap_err().code == "ORPO_MISSING_MESSAGE_KEYS"

    # --- prepare_dataset ---

    def test_prepare_dataset_passes_through_on_valid(self, orpo_strategy, mock_tokenizer):
        """
        Given: Valid preference dataset
        When: prepare_dataset is called
        Then: Returns Ok with the same dataset object (TRL handles formatting)
        """
        ds = _make_preference_dataset(3)
        result = orpo_strategy.prepare_dataset(ds, mock_tokenizer)

        assert result.is_success()
        assert result.unwrap() is ds

    def test_prepare_dataset_returns_err_when_validate_fails(self, orpo_strategy, mock_tokenizer):
        """
        Given: Dataset missing required columns
        When: prepare_dataset is called
        Then: Propagates Err from validate_dataset
        """
        from datasets import Dataset

        ds = Dataset.from_list([{"random": "field"}])
        result = orpo_strategy.prepare_dataset(ds, mock_tokenizer)

        assert result.is_failure()
        assert result.unwrap_err().code == "ORPO_MISSING_CHOSEN_COLUMN"

    # --- build_config_kwargs ---

    def test_build_config_kwargs_beta_none_defaults_to_0_1(self, orpo_strategy):
        """
        Given: hp.beta is None
        When: build_config_kwargs is called
        Then: beta defaults to 0.1
        """
        hp = MagicMock()
        hp.beta = None
        hp.max_length = 512

        kwargs = orpo_strategy.build_config_kwargs(hp)

        assert kwargs["beta"] == 0.1

    def test_build_config_kwargs_beta_explicit_value(self, orpo_strategy):
        """
        Given: hp.beta is 0.5
        When: build_config_kwargs is called
        Then: Uses actual value 0.5
        """
        hp = MagicMock()
        hp.beta = 0.5
        hp.max_length = 1024

        kwargs = orpo_strategy.build_config_kwargs(hp)

        assert kwargs["beta"] == 0.5
        assert kwargs["max_length"] == 1024

    # --- metadata ---

    def test_get_trainer_type(self, orpo_strategy):
        assert orpo_strategy.get_trainer_type() == "orpo"

    def test_get_training_objective(self, orpo_strategy):
        assert orpo_strategy.get_training_objective() == "combined_sft_preference"

    def test_get_metadata(self, orpo_strategy):
        """
        Given: ORPOStrategy
        When: get_metadata is called
        Then: Returns valid StrategyMetadata with orpo type
        """
        metadata = orpo_strategy.get_metadata()

        assert isinstance(metadata, StrategyMetadata)
        assert metadata.strategy_type == "orpo"
        assert "orpo" in metadata.name.lower()
        assert metadata.version is not None

    def test_get_recommended_hyperparameters(self, orpo_strategy):
        """
        Given: ORPOStrategy
        When: get_recommended_hyperparameters is called
        Then: Returns dict with beta and learning_rate
        """
        hp = orpo_strategy.get_recommended_hyperparameters()

        assert isinstance(hp, dict)
        assert "learning_rate" in hp
        assert "beta" in hp
        assert hp["beta"] == 0.1


# =============================================================================
# TEST CLASS: DPOStrategy
# =============================================================================


class TestDPOStrategy:
    """Unit tests for DPOStrategy."""

    # --- validate_dataset: positive ---

    def test_validate_dataset_valid_data(self, dpo_strategy):
        """
        Given: Dataset with valid chosen/rejected message pairs
        When: validate_dataset is called
        Then: Returns Ok(True)
        """
        ds = _make_preference_dataset()
        result = dpo_strategy.validate_dataset(ds)

        assert result.is_success()
        assert result.unwrap() is True

    def test_validate_dataset_empty_chosen_list_boundary(self, dpo_strategy):
        """
        Given: Dataset where chosen/rejected are empty lists (boundary case)
        When: validate_dataset is called
        Then: Returns Ok(True)
        """
        from datasets import Dataset

        ds = Dataset.from_list([{"chosen": [], "rejected": []}])
        result = dpo_strategy.validate_dataset(ds)

        assert result.is_success()

    # --- validate_dataset: error branches ---

    def test_validate_dataset_missing_chosen_column(self, dpo_strategy):
        """
        Given: Dataset missing 'chosen' column
        When: validate_dataset is called
        Then: Returns Err with code DPO_MISSING_CHOSEN_COLUMN
        """
        from datasets import Dataset

        ds = Dataset.from_list([{"rejected": [{"role": "user", "content": "Q"}]}])
        result = dpo_strategy.validate_dataset(ds)

        assert result.is_failure()
        assert result.unwrap_err().code == "DPO_MISSING_CHOSEN_COLUMN"

    def test_validate_dataset_missing_rejected_column(self, dpo_strategy):
        """
        Given: Dataset missing 'rejected' column
        When: validate_dataset is called
        Then: Returns Err with code DPO_MISSING_REJECTED_COLUMN
        """
        from datasets import Dataset

        ds = Dataset.from_list([{"chosen": [{"role": "user", "content": "Q"}]}])
        result = dpo_strategy.validate_dataset(ds)

        assert result.is_failure()
        assert result.unwrap_err().code == "DPO_MISSING_REJECTED_COLUMN"

    def test_validate_dataset_invalid_message_format_not_lists(self, dpo_strategy):
        """
        Given: chosen/rejected are strings instead of lists
        When: validate_dataset is called
        Then: Returns Err with code DPO_INVALID_MESSAGE_FORMAT
        """
        from datasets import Dataset

        ds = Dataset.from_list([{"chosen": "plain string", "rejected": "another string"}])
        result = dpo_strategy.validate_dataset(ds)

        assert result.is_failure()
        assert result.unwrap_err().code == "DPO_INVALID_MESSAGE_FORMAT"

    def test_validate_dataset_invalid_message_structure_not_dicts(self, dpo_strategy):
        """
        Given: chosen is list of strings (not dicts)
        When: validate_dataset is called
        Then: Returns Err with code DPO_INVALID_MESSAGE_STRUCTURE
        """
        from datasets import Dataset

        ds = Dataset.from_list([{"chosen": ["just a string"], "rejected": ["another"]}])
        result = dpo_strategy.validate_dataset(ds)

        assert result.is_failure()
        assert result.unwrap_err().code == "DPO_INVALID_MESSAGE_STRUCTURE"

    def test_validate_dataset_missing_message_keys(self, dpo_strategy):
        """
        Given: Messages are dicts missing 'role' and 'content' keys
        When: validate_dataset is called
        Then: Returns Err with code DPO_MISSING_MESSAGE_KEYS
        """
        from datasets import Dataset

        ds = Dataset.from_list(
            [
                {
                    "chosen": [{"text": "no role/content here"}],
                    "rejected": [{"text": "missing keys"}],
                }
            ]
        )
        result = dpo_strategy.validate_dataset(ds)

        assert result.is_failure()
        assert result.unwrap_err().code == "DPO_MISSING_MESSAGE_KEYS"

    # --- prepare_dataset ---

    def test_prepare_dataset_passes_through_on_valid(self, dpo_strategy, mock_tokenizer):
        """
        Given: Valid preference dataset
        When: prepare_dataset is called
        Then: Returns Ok with the same dataset object
        """
        ds = _make_preference_dataset(3)
        result = dpo_strategy.prepare_dataset(ds, mock_tokenizer)

        assert result.is_success()
        assert result.unwrap() is ds

    def test_prepare_dataset_returns_err_when_validate_fails(self, dpo_strategy, mock_tokenizer):
        """
        Given: Dataset missing required columns
        When: prepare_dataset is called
        Then: Propagates Err from validate_dataset
        """
        from datasets import Dataset

        ds = Dataset.from_list([{"random": "field"}])
        result = dpo_strategy.prepare_dataset(ds, mock_tokenizer)

        assert result.is_failure()
        assert result.unwrap_err().code == "DPO_MISSING_CHOSEN_COLUMN"

    # --- build_config_kwargs ---

    def test_build_config_kwargs_beta_none_defaults_to_0_1(self, dpo_strategy):
        """
        Given: hp.beta is None
        When: build_config_kwargs is called
        Then: beta defaults to 0.1
        """
        hp = MagicMock()
        hp.beta = None
        hp.max_length = 512

        kwargs = dpo_strategy.build_config_kwargs(hp)

        assert kwargs["beta"] == 0.1

    def test_build_config_kwargs_beta_explicit_value(self, dpo_strategy):
        """
        Given: hp.beta is 0.3
        When: build_config_kwargs is called
        Then: Uses actual value 0.3
        """
        hp = MagicMock()
        hp.beta = 0.3
        hp.max_length = 2048

        kwargs = dpo_strategy.build_config_kwargs(hp)

        assert kwargs["beta"] == 0.3

    # --- metadata ---

    def test_get_trainer_type(self, dpo_strategy):
        assert dpo_strategy.get_trainer_type() == "dpo"

    def test_get_training_objective(self, dpo_strategy):
        assert dpo_strategy.get_training_objective() == "preference_optimization"

    def test_get_metadata(self, dpo_strategy):
        """
        Given: DPOStrategy
        When: get_metadata is called
        Then: Returns valid StrategyMetadata with dpo type
        """
        metadata = dpo_strategy.get_metadata()

        assert isinstance(metadata, StrategyMetadata)
        assert metadata.strategy_type == "dpo"
        assert "dpo" in metadata.name.lower()
        assert metadata.version is not None

    def test_get_recommended_hyperparameters(self, dpo_strategy):
        """
        Given: DPOStrategy
        When: get_recommended_hyperparameters is called
        Then: Returns dict with very low learning_rate and beta
        """
        hp = dpo_strategy.get_recommended_hyperparameters()

        assert isinstance(hp, dict)
        assert "learning_rate" in hp
        assert "beta" in hp
        # DPO needs much lower LR than SFT
        assert hp["learning_rate"] <= 1e-5

    # --- combinatorial: both columns missing ---

    def test_validate_dataset_no_relevant_columns(self, dpo_strategy):
        """
        Given: Dataset with completely unrelated columns
        When: validate_dataset is called
        Then: Fails on missing chosen (first check)
        """
        from datasets import Dataset

        ds = Dataset.from_list([{"instruction": "Q", "response": "A"}])
        result = dpo_strategy.validate_dataset(ds)

        assert result.is_failure()
        assert result.unwrap_err().code == "DPO_MISSING_CHOSEN_COLUMN"


# =============================================================================
# TEST CLASS: SAPOStrategy
# =============================================================================


class TestSAPOStrategy:
    """Unit tests for SAPOStrategy."""

    # --- validate_dataset ---

    def test_validate_dataset_with_prompt_column(self, sapo_strategy):
        """
        Given: Dataset with 'prompt' column
        When: validate_dataset is called
        Then: Returns Ok(True)
        """
        from datasets import Dataset

        ds = Dataset.from_list([{"prompt": "Explain gravity", "reference_answer": "QUERY ExplainGravity () => RETURN 1"}])
        result = sapo_strategy.validate_dataset(ds)

        assert result.is_success()
        assert result.unwrap() is True

    def test_validate_dataset_with_messages_column(self, sapo_strategy):
        """
        Given: Dataset with 'messages' column (no prompt)
        When: validate_dataset is called
        Then: Returns Ok(True)
        """
        from datasets import Dataset

        ds = Dataset.from_list(
            [{"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "QUERY Hi () => RETURN 1"}]}]
        )
        result = sapo_strategy.validate_dataset(ds)

        assert result.is_success()

    def test_validate_dataset_missing_prompt_and_messages(self, sapo_strategy):
        """
        Given: Dataset with neither 'prompt' nor 'messages' column
        When: validate_dataset is called
        Then: Returns Err
        """
        from datasets import Dataset

        ds = Dataset.from_list([{"instruction": "Q", "answer": "A"}])
        result = sapo_strategy.validate_dataset(ds)

        assert result.is_failure()

    # --- prepare_dataset ---

    def test_prepare_dataset_with_prompt_column_returns_directly(self, sapo_strategy, sapo_tokenizer):
        """
        Given: Dataset already has 'prompt' column
        When: prepare_dataset is called
        Then: Returns Ok with same dataset (no .map call needed)
        """
        dataset = MagicMock()
        dataset.features = {"prompt": MagicMock(), "reference_answer": MagicMock(), "schema_context": MagicMock()}

        result = sapo_strategy.prepare_dataset(dataset, sapo_tokenizer)

        assert result.is_success()
        assert result.unwrap() is dataset
        dataset.map.assert_not_called()

    def test_prepare_dataset_with_messages_extracts_prompt(self, sapo_strategy, sapo_tokenizer):
        """
        Given: Dataset has 'messages' column but no 'prompt'
        When: prepare_dataset is called
        Then: Returns Ok with prepared dataset (calls .map to extract prompt)
        """
        prepared = MagicMock()
        dataset = MagicMock()
        dataset.features = {"messages": MagicMock()}
        dataset.map.return_value = prepared

        result = sapo_strategy.prepare_dataset(dataset, sapo_tokenizer)

        assert result.is_success()
        assert result.unwrap() is prepared
        dataset.map.assert_called_once()

    def test_prepare_dataset_no_usable_column_returns_err(self, sapo_strategy, sapo_tokenizer):
        """
        Given: Dataset has neither 'prompt' nor 'messages'
        When: prepare_dataset is called
        Then: Returns Err
        """
        dataset = MagicMock()
        dataset.features = {"instruction": MagicMock()}

        result = sapo_strategy.prepare_dataset(dataset, sapo_tokenizer)

        assert result.is_failure()

    def test_prepare_dataset_map_exception_returns_err(self, sapo_strategy, sapo_tokenizer):
        """
        Given: Dataset has 'messages' column but .map() raises
        When: prepare_dataset is called
        Then: Returns Err
        """
        dataset = MagicMock()
        dataset.features = {"messages": MagicMock()}
        dataset.map.side_effect = RuntimeError("map exploded")

        result = sapo_strategy.prepare_dataset(dataset, sapo_tokenizer)

        assert result.is_failure()

    # --- build_config_kwargs ---

    def test_build_config_kwargs_all_none_returns_loss_type_only(self, sapo_strategy):
        """
        Given: All optional hp fields are None
        When: build_config_kwargs is called
        Then: Returns empty dict (no defaults injected)
        """
        hp = MagicMock()
        hp.num_generations = None
        hp.max_prompt_length = None
        hp.max_completion_length = None
        hp.beta = None
        hp.sapo_temperature_pos = None
        hp.sapo_temperature_neg = None

        kwargs = sapo_strategy.build_config_kwargs(hp)

        assert kwargs == {"loss_type": "sapo"}

    def test_build_config_kwargs_all_set(self, sapo_strategy):
        """
        Given: All optional hp fields are set
        When: build_config_kwargs is called
        Then: All values appear in the returned dict
        """
        hp = MagicMock()
        hp.num_generations = 4
        hp.max_prompt_length = 256
        hp.max_completion_length = 512
        hp.beta = 0.05
        hp.sapo_temperature_pos = 0.9
        hp.sapo_temperature_neg = 1.1

        kwargs = sapo_strategy.build_config_kwargs(hp)

        assert kwargs["num_generations"] == 4
        assert kwargs["loss_type"] == "sapo"
        assert kwargs["max_prompt_length"] == 256
        assert kwargs["max_completion_length"] == 512
        assert kwargs["beta"] == 0.05
        assert kwargs["temperature"] == 0.9
        assert kwargs["sapo_temperature_neg"] == 1.1

    def test_build_config_kwargs_partial_set(self, sapo_strategy):
        """
        Given: Only some hp fields are set (num_generations, beta)
        When: build_config_kwargs is called
        Then: Only those keys appear in the result
        """
        hp = MagicMock()
        hp.num_generations = 8
        hp.max_prompt_length = None
        hp.max_completion_length = None
        hp.beta = 0.1
        hp.sapo_temperature_pos = None
        hp.sapo_temperature_neg = None

        kwargs = sapo_strategy.build_config_kwargs(hp)

        assert kwargs["num_generations"] == 8
        assert kwargs["loss_type"] == "sapo"
        assert kwargs["beta"] == 0.1
        assert "max_prompt_length" not in kwargs
        assert "max_completion_length" not in kwargs
        assert "temperature" not in kwargs

    # --- metadata ---

    def test_get_trainer_type(self, sapo_strategy):
        assert sapo_strategy.get_trainer_type() == "sapo"

    def test_get_training_objective(self, sapo_strategy):
        assert sapo_strategy.get_training_objective() == "soft_adaptive_policy_optimization"

    def test_get_metadata(self, sapo_strategy):
        """
        Given: SAPOStrategy
        When: get_metadata is called
        Then: Returns valid StrategyMetadata with sapo type
        """
        metadata = sapo_strategy.get_metadata()

        assert isinstance(metadata, StrategyMetadata)
        assert metadata.strategy_type == "sapo"
        assert metadata.version is not None

    def test_get_recommended_hyperparameters_not_defined(self, sapo_strategy):
        """
        Given: SAPOStrategy inherits base get_recommended_hyperparameters
        When: called
        Then: Returns a dict (base implementation)
        """
        hp = sapo_strategy.get_recommended_hyperparameters()
        assert isinstance(hp, dict)


# =============================================================================
# TEST CLASS: SFTStrategy – extended with real Dataset
# =============================================================================


class TestSFTStrategyExtended:
    """Additional SFTStrategy tests using real Dataset objects."""

    def test_prepare_dataset_instruction_format_real_dataset(self, sft_strategy, mock_tokenizer):
        """
        Given: Real Dataset with instruction/response columns
        When: prepare_dataset is called
        Then: format_instruction inside .map() runs; output has 'text' with headers
        """
        from datasets import Dataset

        ds = Dataset.from_list(
            [
                {"instruction": "Tell me a joke", "response": "Why did the chicken cross the road?"},
                {"instruction": "What is Python?", "response": "A programming language."},
            ]
        )

        result = sft_strategy.prepare_dataset(ds, mock_tokenizer)

        assert result.is_success()
        prepared = result.unwrap()
        assert "text" in prepared.column_names
        assert "### Instruction:" in prepared[0]["text"]
        assert "### Response:" in prepared[0]["text"]
        assert "Tell me a joke" in prepared[0]["text"]

    def test_prepare_dataset_instruction_format_output_format(self, sft_strategy, mock_tokenizer):
        """
        Given: Dataset with instruction and output (alternative key) columns
        When: prepare_dataset is called
        Then: format_instruction handles 'output' key as fallback for 'response'
        """
        from datasets import Dataset

        ds = Dataset.from_list(
            [
                {"instruction": "Explain gravity", "output": "Gravity pulls objects together."},
            ]
        )

        result = sft_strategy.prepare_dataset(ds, mock_tokenizer)

        assert result.is_success()
        prepared = result.unwrap()
        assert "text" in prepared.column_names

    def test_prepare_dataset_instruction_all_samples_converted(self, sft_strategy, mock_tokenizer):
        """
        Given: Dataset with 5 instruction rows
        When: prepare_dataset is called
        Then: All 5 rows are converted to 'text' format
        """
        from datasets import Dataset

        ds = Dataset.from_list(
            [{"instruction": f"Q{i}", "response": f"A{i}"} for i in range(5)]
        )

        result = sft_strategy.prepare_dataset(ds, mock_tokenizer)

        assert result.is_success()
        assert len(result.unwrap()) == 5

    def test_validate_dataset_boundary_multiple_valid_columns(self, sft_strategy):
        """
        Given: Dataset with all three valid column types present
        When: validate_dataset is called
        Then: Returns Ok (takes the first valid path)
        """
        mock_dataset = MagicMock()
        mock_dataset.column_names = ["messages", "text", "instruction"]

        result = sft_strategy.validate_dataset(mock_dataset)

        assert result.is_success()


# =============================================================================
# TEST CLASS: CoTStrategy – extended with real Dataset
# =============================================================================


class TestCoTStrategyExtended:
    """Additional CoTStrategy tests using real Dataset objects."""

    def test_prepare_dataset_reasoning_field_produces_think_tags(self, cot_strategy, mock_tokenizer):
        """
        Given: Real Dataset with instruction/reasoning/answer columns
        When: prepare_dataset is called
        Then: format_cot inside .map() wraps reasoning in <think> tags
        """
        from datasets import Dataset

        ds = Dataset.from_list(
            [
                {"instruction": "What is 2+2?", "reasoning": "2 plus 2 equals 4", "answer": "4"},
                {"instruction": "Capital of France?", "reasoning": "Paris is the capital", "answer": "Paris"},
            ]
        )

        result = cot_strategy.prepare_dataset(ds, mock_tokenizer)

        assert result.is_success()
        prepared = result.unwrap()
        assert "text" in prepared.column_names
        assert "<think>" in prepared[0]["text"]
        assert "</think>" in prepared[0]["text"]
        assert "<answer>" in prepared[0]["text"]
        assert "</answer>" in prepared[0]["text"]
        assert "2 plus 2 equals 4" in prepared[0]["text"]
        assert "4" in prepared[0]["text"]

    def test_prepare_dataset_chain_of_thought_field_used(self, cot_strategy, mock_tokenizer):
        """
        Given: Dataset with chain_of_thought instead of reasoning
        When: prepare_dataset is called
        Then: chain_of_thought content appears inside <think> tags
        """
        from datasets import Dataset

        ds = Dataset.from_list(
            [
                {"instruction": "What is 3*3?", "chain_of_thought": "3 times 3 is 9", "answer": "9"},
            ]
        )

        result = cot_strategy.prepare_dataset(ds, mock_tokenizer)

        assert result.is_success()
        prepared = result.unwrap()
        assert "3 times 3 is 9" in prepared[0]["text"]
        assert "<think>" in prepared[0]["text"]

    def test_prepare_dataset_all_samples_have_tags(self, cot_strategy, mock_tokenizer):
        """
        Given: Dataset with 4 CoT rows
        When: prepare_dataset is called
        Then: Every row in the output has think/answer tags
        """
        from datasets import Dataset

        ds = Dataset.from_list(
            [{"instruction": f"Q{i}", "reasoning": f"R{i}", "answer": f"A{i}"} for i in range(4)]
        )

        result = cot_strategy.prepare_dataset(ds, mock_tokenizer)

        assert result.is_success()
        prepared = result.unwrap()
        assert len(prepared) == 4
        for row in prepared:
            assert "<think>" in row["text"]
            assert "<answer>" in row["text"]

    def test_prepare_dataset_messages_format_no_map_called(self, cot_strategy, mock_tokenizer):
        """
        Given: Real Dataset with 'messages' column
        When: prepare_dataset is called
        Then: Returns dataset as-is without calling map (TRL handles ChatML)
        """
        from datasets import Dataset

        ds = Dataset.from_list(
            [
                {
                    "messages": [
                        {"role": "user", "content": "Hello"},
                        {"role": "assistant", "content": "<think>thinking</think><answer>Hi!</answer>"},
                    ]
                }
            ]
        )

        result = cot_strategy.prepare_dataset(ds, mock_tokenizer)

        assert result.is_success()
        # Should be the same dataset object, not processed through map
        assert result.unwrap() is ds

    def test_validate_dataset_both_required_missing(self, cot_strategy):
        """
        Given: Dataset has only 'reasoning' but not 'instruction' or 'answer'
        When: validate_dataset is called
        Then: Returns Err (missing required columns)
        """
        mock_dataset = MagicMock()
        mock_dataset.column_names = ["reasoning"]

        result = cot_strategy.validate_dataset(mock_dataset)

        assert result.is_failure()
