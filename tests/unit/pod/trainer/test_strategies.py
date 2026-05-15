"""
Unit tests for Training Strategies (SFT, CPT, CoT).

Tests prepare_dataset, validate_dataset, and get_metadata methods
for all training strategy types.

Updated for Phase 3: TRL-native integration (no manual preprocessing).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ryotenkai_pod.trainer.strategies.base import StrategyMetadata
from ryotenkai_pod.trainer.strategies.cot import CoTStrategy
from ryotenkai_pod.trainer.strategies.cpt import CPTStrategy
from ryotenkai_pod.trainer.strategies.sft import SFTStrategy
from ryotenkai_shared.errors import DatasetValidationFailedError

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
        Then: Returns None (no exception)
        """
        mock_dataset = MagicMock()
        mock_dataset.column_names = ["messages"]

        # Should not raise
        assert sft_strategy.validate_dataset(mock_dataset) is None

    def test_validate_dataset_with_text(self, sft_strategy):
        """
        Given: Dataset with text field
        When: validate_dataset is called
        Then: Returns None - SFT now accepts text format too!
        """
        mock_dataset = MagicMock()
        mock_dataset.column_names = ["text"]

        # Should not raise
        assert sft_strategy.validate_dataset(mock_dataset) is None

    def test_validate_dataset_missing_fields(self, sft_strategy):
        """
        Given: Dataset without any valid field
        When: validate_dataset is called
        Then: Raises DatasetValidationFailedError with SFT legacy code
        """
        mock_dataset = MagicMock()
        mock_dataset.column_names = ["random_field"]  # No valid field

        with pytest.raises(DatasetValidationFailedError) as excinfo:
            sft_strategy.validate_dataset(mock_dataset)

        exc = excinfo.value
        assert exc.context.get("legacy_code") == "SFT_MISSING_REQUIRED_COLUMN"
        assert exc.context.get("available_columns") == ["random_field"]
        assert "messages" in exc.detail or "text" in exc.detail

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
        Then: Returns None (no exception)
        """
        mock_dataset = MagicMock()
        mock_dataset.column_names = ["text"]

        # Should not raise
        assert cpt_strategy.validate_dataset(mock_dataset) is None

    def test_validate_dataset_missing_text(self, cpt_strategy):
        """
        Given: Dataset without text field
        When: validate_dataset is called
        Then: Raises DatasetValidationFailedError with CPT legacy code
        """
        mock_dataset = MagicMock()
        mock_dataset.column_names = ["content"]  # Wrong field name

        with pytest.raises(DatasetValidationFailedError) as excinfo:
            cpt_strategy.validate_dataset(mock_dataset)

        exc = excinfo.value
        assert exc.context.get("legacy_code") == "CPT_MISSING_TEXT_COLUMN"
        assert exc.context.get("missing_column") == "text"
        assert "text" in exc.detail

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



# =============================================================================
# TEST CLASS: CoTStrategy
# =============================================================================


class TestCoTStrategy:
    """Unit tests for CoTStrategy."""

    def test_validate_dataset_with_messages(self, cot_strategy):
        """
        Given: Dataset with messages field (ChatML format)
        When: validate_dataset is called
        Then: Returns None (no exception)
        """
        mock_dataset = MagicMock()
        mock_dataset.column_names = ["messages"]

        assert cot_strategy.validate_dataset(mock_dataset) is None

    def test_validate_dataset_with_text(self, cot_strategy):
        """
        Given: Dataset with text field
        When: validate_dataset is called
        Then: Returns None (no exception)
        """
        mock_dataset = MagicMock()
        mock_dataset.column_names = ["text"]

        assert cot_strategy.validate_dataset(mock_dataset) is None

    def test_validate_dataset_missing_required_columns(self, cot_strategy):
        """
        Given: Dataset without messages or text
        When: validate_dataset is called
        Then: Raises DatasetValidationFailedError with CoT legacy code
        """
        mock_dataset = MagicMock()
        mock_dataset.column_names = ["instruction", "answer"]  # Legacy format not accepted

        with pytest.raises(DatasetValidationFailedError) as excinfo:
            cot_strategy.validate_dataset(mock_dataset)

        exc = excinfo.value
        assert exc.context.get("legacy_code") == "COT_MISSING_REQUIRED_COLUMNS"
        assert exc.context.get("expected_one_of") == ["messages", "text"]


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
        from ryotenkai_pod.trainer.strategies.factory import StrategyFactory

        factory = StrategyFactory()
        strategy = factory.create("sft", mock_pipeline_config)

        assert isinstance(strategy, SFTStrategy)

    def test_create_cpt_strategy(self, mock_pipeline_config):
        """
        Given: strategy_type='cpt'
        When: StrategyFactory().create is called
        Then: Returns CPTStrategy
        """
        from ryotenkai_pod.trainer.strategies.factory import StrategyFactory

        factory = StrategyFactory()
        strategy = factory.create("cpt", mock_pipeline_config)

        assert isinstance(strategy, CPTStrategy)

    def test_create_cot_strategy(self, mock_pipeline_config):
        """
        Given: strategy_type='cot'
        When: StrategyFactory().create is called
        Then: Returns CoTStrategy
        """
        from ryotenkai_pod.trainer.strategies.factory import StrategyFactory

        factory = StrategyFactory()
        strategy = factory.create("cot", mock_pipeline_config)

        assert isinstance(strategy, CoTStrategy)

    def test_create_unknown_strategy(self, mock_pipeline_config):
        """
        Given: Unknown strategy_type
        When: StrategyFactory().create is called
        Then: Raises ValueError
        """
        from ryotenkai_pod.trainer.strategies.factory import StrategyFactory

        factory = StrategyFactory()
        with pytest.raises(ValueError):
            factory.create("unknown", mock_pipeline_config)

    def test_list_strategies(self):
        """
        Given: StrategyFactory
        When: list_available is called
        Then: Returns dict with sft, cpt, cot
        """
        from ryotenkai_pod.trainer.strategies.factory import StrategyFactory

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
        from ryotenkai_pod.trainer.strategies.factory import StrategyFactory

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
        assert hasattr(sft_strategy, "validate_dataset")
        assert hasattr(sft_strategy, "get_training_objective")
        assert hasattr(sft_strategy, "get_metadata")
        assert hasattr(sft_strategy, "get_trainer_type")
        assert callable(sft_strategy.validate_dataset)
        assert callable(sft_strategy.get_trainer_type)

    def test_cpt_implements_interface(self, cpt_strategy):
        """
        Given: CPTStrategy
        Then: Implements TrainingStrategy interface
        """
        assert hasattr(cpt_strategy, "validate_dataset")
        assert hasattr(cpt_strategy, "get_training_objective")
        assert hasattr(cpt_strategy, "get_metadata")
        assert hasattr(cpt_strategy, "get_trainer_type")

    def test_cot_implements_interface(self, cot_strategy):
        """
        Given: CoTStrategy
        Then: Implements TrainingStrategy interface
        """
        assert hasattr(cot_strategy, "validate_dataset")
        assert hasattr(cot_strategy, "get_training_objective")
        assert hasattr(cot_strategy, "get_metadata")
        assert hasattr(cot_strategy, "get_trainer_type")

    def test_all_strategies_raise_on_missing_columns(self, sft_strategy, cpt_strategy, cot_strategy):
        """
        Given: All strategies given an empty-columns dataset
        When: validate_dataset is called
        Then: Each raises DatasetValidationFailedError (uniform contract)
        """
        mock_dataset = MagicMock()
        mock_dataset.column_names = []

        for strategy in [sft_strategy, cpt_strategy, cot_strategy]:
            with pytest.raises(DatasetValidationFailedError):
                strategy.validate_dataset(mock_dataset)


# =============================================================================
# ADDITIONAL FIXTURES
# =============================================================================


@pytest.fixture
def orpo_strategy(mock_pipeline_config):
    """Create ORPOStrategy instance."""
    from ryotenkai_pod.trainer.strategies.orpo import ORPOStrategy

    return ORPOStrategy(mock_pipeline_config)


@pytest.fixture
def dpo_strategy(mock_pipeline_config):
    """Create DPOStrategy instance."""
    from ryotenkai_pod.trainer.strategies.dpo import DPOStrategy

    return DPOStrategy(mock_pipeline_config)


@pytest.fixture
def sapo_strategy(mock_pipeline_config):
    """Create SAPOStrategy instance."""
    from ryotenkai_pod.trainer.strategies.sapo import SAPOStrategy

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
        Then: Returns None (no exception)
        """
        ds = _make_preference_dataset()
        assert orpo_strategy.validate_dataset(ds) is None

    def test_validate_dataset_empty_chosen_list_boundary(self, orpo_strategy):
        """
        Given: Dataset where chosen/rejected are empty lists
        When: validate_dataset is called
        Then: Returns None — skips per-message structure checks
        """
        from datasets import Dataset

        ds = Dataset.from_list([{"chosen": [], "rejected": []}])
        assert orpo_strategy.validate_dataset(ds) is None

    # --- validate_dataset: error branches ---

    def test_validate_dataset_missing_chosen_column(self, orpo_strategy):
        """
        Given: Dataset with only 'rejected' column
        When: validate_dataset is called
        Then: Raises DatasetValidationFailedError with legacy_code ORPO_MISSING_CHOSEN_COLUMN
        """
        from datasets import Dataset

        ds = Dataset.from_list([{"rejected": [{"role": "user", "content": "Q"}]}])
        with pytest.raises(DatasetValidationFailedError) as excinfo:
            orpo_strategy.validate_dataset(ds)

        assert excinfo.value.context.get("legacy_code") == "ORPO_MISSING_CHOSEN_COLUMN"
        assert excinfo.value.context.get("missing_column") == "chosen"

    def test_validate_dataset_missing_rejected_column(self, orpo_strategy):
        """
        Given: Dataset with only 'chosen' column
        When: validate_dataset is called
        Then: Raises DatasetValidationFailedError with legacy_code ORPO_MISSING_REJECTED_COLUMN
        """
        from datasets import Dataset

        ds = Dataset.from_list([{"chosen": [{"role": "user", "content": "Q"}]}])
        with pytest.raises(DatasetValidationFailedError) as excinfo:
            orpo_strategy.validate_dataset(ds)

        assert excinfo.value.context.get("legacy_code") == "ORPO_MISSING_REJECTED_COLUMN"
        assert excinfo.value.context.get("missing_column") == "rejected"

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
        Then: Returns None (no exception)
        """
        ds = _make_preference_dataset()
        assert dpo_strategy.validate_dataset(ds) is None

    def test_validate_dataset_empty_chosen_list_boundary(self, dpo_strategy):
        """
        Given: Dataset where chosen/rejected are empty lists (boundary case)
        When: validate_dataset is called
        Then: Returns None (no exception)
        """
        from datasets import Dataset

        ds = Dataset.from_list([{"chosen": [], "rejected": []}])
        assert dpo_strategy.validate_dataset(ds) is None

    # --- validate_dataset: error branches ---

    def test_validate_dataset_missing_chosen_column(self, dpo_strategy):
        """
        Given: Dataset missing 'chosen' column
        When: validate_dataset is called
        Then: Raises DatasetValidationFailedError with legacy_code DPO_MISSING_CHOSEN_COLUMN
        """
        from datasets import Dataset

        ds = Dataset.from_list([{"rejected": [{"role": "user", "content": "Q"}]}])
        with pytest.raises(DatasetValidationFailedError) as excinfo:
            dpo_strategy.validate_dataset(ds)

        assert excinfo.value.context.get("legacy_code") == "DPO_MISSING_CHOSEN_COLUMN"
        assert excinfo.value.context.get("missing_column") == "chosen"

    def test_validate_dataset_missing_rejected_column(self, dpo_strategy):
        """
        Given: Dataset missing 'rejected' column
        When: validate_dataset is called
        Then: Raises DatasetValidationFailedError with legacy_code DPO_MISSING_REJECTED_COLUMN
        """
        from datasets import Dataset

        ds = Dataset.from_list([{"chosen": [{"role": "user", "content": "Q"}]}])
        with pytest.raises(DatasetValidationFailedError) as excinfo:
            dpo_strategy.validate_dataset(ds)

        assert excinfo.value.context.get("legacy_code") == "DPO_MISSING_REJECTED_COLUMN"
        assert excinfo.value.context.get("missing_column") == "rejected"

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

    # --- combinatorial: both columns missing ---

    def test_validate_dataset_no_relevant_columns(self, dpo_strategy):
        """
        Given: Dataset with completely unrelated columns
        When: validate_dataset is called
        Then: Fails on missing chosen (first check)
        """
        from datasets import Dataset

        ds = Dataset.from_list([{"instruction": "Q", "response": "A"}])
        with pytest.raises(DatasetValidationFailedError) as excinfo:
            dpo_strategy.validate_dataset(ds)

        assert excinfo.value.context.get("legacy_code") == "DPO_MISSING_CHOSEN_COLUMN"


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
        Then: Returns None (no exception)
        """
        from datasets import Dataset

        ds = Dataset.from_list([{"prompt": "Explain gravity", "reference_answer": "QUERY ExplainGravity () => RETURN 1"}])
        assert sapo_strategy.validate_dataset(ds) is None

    def test_validate_dataset_missing_prompt(self, sapo_strategy):
        """
        Given: Dataset without 'prompt' column
        When: validate_dataset is called
        Then: Raises DatasetValidationFailedError with legacy_code RL_MISSING_PROMPT_COLUMN
        """
        from datasets import Dataset

        ds = Dataset.from_list([{"instruction": "Q", "answer": "A"}])
        with pytest.raises(DatasetValidationFailedError) as excinfo:
            sapo_strategy.validate_dataset(ds)

        assert excinfo.value.context.get("legacy_code") == "RL_MISSING_PROMPT_COLUMN"
        assert excinfo.value.context.get("missing_column") == "prompt"

    # --- build_config_kwargs ---

    def test_build_config_kwargs_all_none_returns_loss_type_only(self, sapo_strategy):
        """
        Given: All optional hp fields are None
        When: build_config_kwargs is called
        Then: Returns empty dict (no defaults injected)
        """
        hp = MagicMock()
        hp.num_generations = None
        hp.generation_batch_size = None
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


# =============================================================================
# TEST CLASS: SFTStrategy – extended with real Dataset
# =============================================================================


class TestSFTStrategyExtended:
    """Additional SFTStrategy tests using real Dataset objects."""

    def test_validate_dataset_boundary_multiple_valid_columns(self, sft_strategy):
        """
        Given: Dataset with both messages and text columns
        When: validate_dataset is called
        Then: Returns None (matches on messages)
        """
        mock_dataset = MagicMock()
        mock_dataset.column_names = ["messages", "text"]

        assert sft_strategy.validate_dataset(mock_dataset) is None


# =============================================================================
# TEST CLASS: TrainingStrategy base contract (covers base.py mutation surface)
# =============================================================================


class TestTrainingStrategyBaseContract:
    """Pins behaviour declared by ``TrainingStrategy`` base class.

    These tests target the base.py mutation surface (default property
    flags, abstract-method enforcement, ``StrategyMetadata`` defaults).
    Each concrete strategy inherits the defaults, so flipping any
    default flag from ``False`` → ``True`` (or removing ``@abstractmethod``)
    must surface as a measurable difference.
    """

    # --- defaults: requires_reward_plugin / requires_reference_dataset ---

    @pytest.mark.parametrize(
        "strategy_fixture",
        ["sft_strategy", "cpt_strategy", "cot_strategy", "dpo_strategy", "orpo_strategy"],
    )
    def test_non_rl_strategies_do_not_require_reward_plugin(self, strategy_fixture, request):
        """SFT/CPT/CoT/DPO/ORPO inherit ``requires_reward_plugin = False``.

        A mutation flipping the default to ``True`` would silently break
        TrainerFactory (it would demand a reward plugin name for plain
        SFT). Pin the default explicitly.
        """
        strategy = request.getfixturevalue(strategy_fixture)
        assert strategy.requires_reward_plugin is False

    @pytest.mark.parametrize(
        "strategy_fixture",
        ["sft_strategy", "cpt_strategy", "cot_strategy", "dpo_strategy", "orpo_strategy"],
    )
    def test_non_rl_strategies_do_not_require_reference_dataset(self, strategy_fixture, request):
        """SFT/CPT/CoT/DPO/ORPO inherit ``requires_reference_dataset = False``.

        Flipping this default to ``True`` would cause TrainerFactory to
        pass ``train_dataset`` into every strategy's build_trainer_kwargs,
        breaking strategies that don't expect that key.
        """
        strategy = request.getfixturevalue(strategy_fixture)
        assert strategy.requires_reference_dataset is False

    def test_rl_subclasses_override_to_true(self, sapo_strategy):
        """SAPO (BaseRLStrategy) overrides both flags to True — sanity check."""
        assert sapo_strategy.requires_reward_plugin is True
        assert sapo_strategy.requires_reference_dataset is True

    # --- StrategyMetadata.__post_init__: dependencies defaults to {} ---

    def test_metadata_post_init_replaces_none_dependencies_with_dict(self):
        """``StrategyMetadata(dependencies=None)`` must become ``{}`` post-init.

        Mutation ``if self.dependencies is None`` → ``is not None`` would
        wipe legitimate dependency dicts and skip the None-replacement.
        Two assertions cover both branches.
        """
        md = StrategyMetadata(
            name="x",
            version="1.0",
            description="d",
            strategy_type="sft",
            data_format="text",
            objective="o",
            recommended_use="r",
            dependencies=None,
        )
        assert md.dependencies == {}

    def test_metadata_post_init_preserves_existing_dependencies(self):
        """``StrategyMetadata(dependencies={"trl": ">=0.8"})`` must NOT be cleared."""
        md = StrategyMetadata(
            name="x",
            version="1.0",
            description="d",
            strategy_type="sft",
            data_format="text",
            objective="o",
            recommended_use="r",
            dependencies={"trl": ">=0.8"},
        )
        assert md.dependencies == {"trl": ">=0.8"}

    # --- abstract methods enforced ---

    def test_training_strategy_is_abstract_cannot_instantiate_directly(self):
        """Removing ``@abstractmethod`` from validate_dataset/get_trainer_type/
        get_trainer_class/get_config_class would make the ABC concrete.
        Pin by attempting direct instantiation.
        """
        from ryotenkai_pod.trainer.strategies.base import TrainingStrategy

        with pytest.raises(TypeError):
            TrainingStrategy(config=MagicMock())  # type: ignore[abstract]

    # --- default get_training_objective falls back to get_trainer_type ---

    def test_default_training_objective_uses_trainer_type(self, mock_pipeline_config):
        """Base class default: ``get_training_objective`` returns trainer_type.

        Each concrete strategy overrides this with a domain-specific
        objective. Pin the SFT path explicitly so a mutation collapsing
        the override is visible.
        """
        # SFT overrides this; check the override is meaningful.
        sft = SFTStrategy(mock_pipeline_config)
        assert sft.get_training_objective() == "supervised_learning"
        assert sft.get_trainer_type() == "sft"
        # They must NOT be identical strings — the override matters.
        assert sft.get_training_objective() != sft.get_trainer_type()

    # --- repr stability (covers __repr__) ---

    def test_repr_contains_class_name_and_strategy_type(self, sft_strategy):
        r = repr(sft_strategy)
        assert "SFTStrategy" in r
        assert "type=sft" in r


# =============================================================================
# TEST CLASS: BaseRLStrategy.prepare_prompts_for_chat_template
# =============================================================================


class TestBaseRLPromptPreparation:
    """Pins behaviour of ``BaseRLStrategy.prepare_prompts_for_chat_template``.

    The mutation surface here is dense (4 nested guards + a .map call).
    Each branch must be measurable so flipping ``and``/``or``, ``not``,
    or boundary conditions surfaces as a test failure.
    """

    @pytest.fixture
    def rl_strategy(self, mock_pipeline_config):
        from ryotenkai_pod.trainer.strategies.sapo import SAPOStrategy

        return SAPOStrategy(mock_pipeline_config)

    # --- Guard 1: tokenizer has no chat_template → noop ---

    def test_noop_when_tokenizer_has_no_chat_template(self, rl_strategy):
        """No chat_template attr or None value → datasets returned unchanged."""
        from datasets import Dataset

        train = Dataset.from_list([{"prompt": "hello"}])
        eval_ds = Dataset.from_list([{"prompt": "world"}])
        tok = MagicMock(spec=[])  # no chat_template attr

        out_train, out_eval = rl_strategy.prepare_prompts_for_chat_template(train, eval_ds, tok)
        assert out_train is train
        assert out_eval is eval_ds

    def test_noop_when_chat_template_is_none(self, rl_strategy):
        """chat_template=None falsy → bail out, return originals."""
        from datasets import Dataset

        train = Dataset.from_list([{"prompt": "hi"}])
        tok = MagicMock()
        tok.chat_template = None

        out_train, out_eval = rl_strategy.prepare_prompts_for_chat_template(train, None, tok)
        assert out_train is train
        assert out_eval is None

    # --- Guard 2: prompt column missing → noop ---

    def test_noop_when_prompt_column_missing_from_train(self, rl_strategy):
        """train has no 'prompt' column → bail out."""
        from datasets import Dataset

        train = Dataset.from_list([{"text": "hi"}])
        tok = MagicMock()
        tok.chat_template = "{{messages}}"

        out_train, out_eval = rl_strategy.prepare_prompts_for_chat_template(train, None, tok)
        assert out_train is train
        assert out_eval is None

    # --- Guard 3: prompt is not a str (already conversational) → noop ---

    def test_noop_when_prompt_already_conversational(self, rl_strategy):
        """If first prompt is already a list-of-dicts, leave the dataset alone."""
        from datasets import Dataset

        train = Dataset.from_list(
            [{"prompt": [{"role": "user", "content": "hi"}]}]
        )
        tok = MagicMock()
        tok.chat_template = "{{messages}}"

        out_train, out_eval = rl_strategy.prepare_prompts_for_chat_template(train, None, tok)
        assert out_train is train

    # --- Happy path: train converted ---

    def test_converts_string_prompts_to_conversational_form(self, rl_strategy):
        """``"hi"`` → ``[{"role": "user", "content": "hi"}]``."""
        from datasets import Dataset

        train = Dataset.from_list([{"prompt": "what is 2+2?"}])
        tok = MagicMock()
        tok.chat_template = "{{messages}}"

        out_train, _ = rl_strategy.prepare_prompts_for_chat_template(train, None, tok)
        first = out_train[0]["prompt"]
        assert isinstance(first, list)
        assert first == [{"role": "user", "content": "what is 2+2?"}]

    # --- Happy path: eval converted only when it has prompt column ---

    def test_converts_eval_when_eval_has_prompt_column(self, rl_strategy):
        """eval dataset with prompt column gets the same treatment as train."""
        from datasets import Dataset

        train = Dataset.from_list([{"prompt": "Q1"}])
        eval_ds = Dataset.from_list([{"prompt": "Q2"}])
        tok = MagicMock()
        tok.chat_template = "{{messages}}"

        out_train, out_eval = rl_strategy.prepare_prompts_for_chat_template(train, eval_ds, tok)
        assert out_eval is not None
        assert out_eval[0]["prompt"] == [{"role": "user", "content": "Q2"}]
        assert out_train[0]["prompt"] == [{"role": "user", "content": "Q1"}]

    def test_skips_eval_conversion_when_eval_lacks_prompt_column(self, rl_strategy):
        """eval without prompt column: train is converted but eval is returned as-is."""
        from datasets import Dataset

        train = Dataset.from_list([{"prompt": "Q1"}])
        eval_ds = Dataset.from_list([{"text": "raw eval text"}])
        tok = MagicMock()
        tok.chat_template = "{{messages}}"

        out_train, out_eval = rl_strategy.prepare_prompts_for_chat_template(train, eval_ds, tok)
        # Train was converted...
        assert out_train[0]["prompt"] == [{"role": "user", "content": "Q1"}]
        # ...but eval is unchanged (still only has 'text').
        assert out_eval is not None
        assert out_eval[0]["text"] == "raw eval text"
        assert "prompt" not in (out_eval.column_names or [])

    def test_eval_none_is_passed_through(self, rl_strategy):
        """eval_dataset=None must propagate as None through the entire path."""
        from datasets import Dataset

        train = Dataset.from_list([{"prompt": "Q"}])
        tok = MagicMock()
        tok.chat_template = "{{messages}}"

        _, out_eval = rl_strategy.prepare_prompts_for_chat_template(train, None, tok)
        assert out_eval is None


# =============================================================================
# TEST CLASS: BaseRLStrategy default schema_extractor + base_rl_config_kwargs
# =============================================================================


class TestBaseRLConfigKwargs:
    """Pins ``_base_rl_config_kwargs`` field-by-field gating.

    Each ``if hp.<field> is not None`` branch must be visible: a mutation
    flipping ``is not None`` → ``is None`` would emit the wrong keys.
    """

    @pytest.fixture
    def rl_strategy(self, mock_pipeline_config):
        from ryotenkai_pod.trainer.strategies.sapo import SAPOStrategy

        return SAPOStrategy(mock_pipeline_config)

    def test_default_schema_extractor_is_noop(self, mock_pipeline_config):
        """Default schema_extractor returns empty string (no domain knowledge)."""
        from ryotenkai_pod.trainer.strategies.sapo import SAPOStrategy

        s = SAPOStrategy(mock_pipeline_config)
        assert s._schema_extractor("anything") == ""  # noqa: SLF001

    def test_injected_schema_extractor_is_used(self, mock_pipeline_config):
        """Injected extractor replaces the default no-op."""
        from ryotenkai_pod.trainer.strategies.sapo import SAPOStrategy

        s = SAPOStrategy(mock_pipeline_config, schema_extractor=lambda _: "SCHEMA")
        assert s._schema_extractor("anything") == "SCHEMA"  # noqa: SLF001

    def test_base_rl_config_kwargs_all_none_returns_empty(self, rl_strategy):
        """No hp fields set → empty kwargs (every guard short-circuits)."""
        hp = MagicMock()
        hp.num_generations = None
        hp.generation_batch_size = None
        hp.max_prompt_length = None
        hp.max_completion_length = None
        hp.beta = None

        out = rl_strategy._base_rl_config_kwargs(hp)  # noqa: SLF001
        assert out == {}

    def test_base_rl_config_kwargs_picks_only_set_fields(self, rl_strategy):
        """Only fields with non-None values are forwarded."""
        hp = MagicMock()
        hp.num_generations = 4
        hp.generation_batch_size = None
        hp.max_prompt_length = 256
        hp.max_completion_length = None
        hp.beta = 0.1

        out = rl_strategy._base_rl_config_kwargs(hp)  # noqa: SLF001
        assert out == {
            "num_generations": 4,
            "max_prompt_length": 256,
            "beta": 0.1,
        }

    def test_base_rl_config_kwargs_all_set(self, rl_strategy):
        """All 5 fields set → all 5 keys forwarded with the same values."""
        hp = MagicMock()
        hp.num_generations = 4
        hp.generation_batch_size = 8
        hp.max_prompt_length = 256
        hp.max_completion_length = 512
        hp.beta = 0.05

        out = rl_strategy._base_rl_config_kwargs(hp)  # noqa: SLF001
        assert out == {
            "num_generations": 4,
            "generation_batch_size": 8,
            "max_prompt_length": 256,
            "max_completion_length": 512,
            "beta": 0.05,
        }


# =============================================================================
# TEST CLASS: CoTStrategy – extended with real Dataset
# =============================================================================


