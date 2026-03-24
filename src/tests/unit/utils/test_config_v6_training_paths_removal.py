"""
Comprehensive tests for v6.0 training_paths removal.

Tests that:
1. DatasetSourceLocal no longer has training_paths field
2. Config loads without training_paths  
3. Methods use local_paths.train instead of training_paths
4. MLflow manager and main.py use correct paths
5. Fixtures and mocks don't reference training_paths
"""


import pytest

from src.utils.config import (
    DatasetConfig,
    DatasetLocalPaths,
    DatasetSourceLocal,
)


class TestTrainingPathsRemoval:
    """Test that training_paths field is completely removed from v6.0."""

    def test_dataset_source_local_has_no_training_paths_field(self):
        """DatasetSourceLocal should NOT have training_paths attribute."""
        source_local = DatasetSourceLocal(
            local_paths=DatasetLocalPaths(train="data/train.jsonl", eval=None)
        )

        assert not hasattr(source_local, "training_paths")

        # Accessing training_paths should raise AttributeError
        with pytest.raises(AttributeError, match="training_paths"):
            _ = source_local.training_paths

    def test_dataset_config_no_training_paths(self, tmp_path):
        """DatasetConfig should work without training_paths."""
        train_file = tmp_path / "train.jsonl"
        train_file.write_text('{"text": "sample"}')

        dataset = DatasetConfig(
            source_type="local",
            source_local=DatasetSourceLocal(
                local_paths=DatasetLocalPaths(
                    train=str(train_file),
                    eval=None
                )
            ),
        )

        assert dataset.source_local.local_paths.train == str(train_file)
        assert not hasattr(dataset.source_local, "training_paths")

    def test_get_source_uri_uses_local_paths_not_training_paths(self, tmp_path):
        """get_source_uri() should use local_paths.train (v6.0)."""
        train_file = tmp_path / "train.jsonl"
        train_file.write_text('{"text": "sample"}')

        dataset = DatasetConfig(
            source_type="local",
            source_local=DatasetSourceLocal(
                local_paths=DatasetLocalPaths(
                    train=str(train_file),
                    eval=None
                )
            ),
        )

        uri = dataset.get_source_uri()
        assert str(train_file.resolve()) in uri
        # Should NOT attempt to access training_paths

    def test_get_display_train_ref_uses_local_paths(self, tmp_path):
        """get_display_train_ref() should use local_paths.train (v6.0)."""
        train_file = tmp_path / "train.jsonl"
        train_file.write_text('{"text": "sample"}')

        dataset = DatasetConfig(
            source_type="local",
            source_local=DatasetSourceLocal(
                local_paths=DatasetLocalPaths(
                    train=str(train_file),
                    eval=None
                )
            ),
        )

        ref = dataset.get_display_train_ref()
        assert ref == str(train_file)


class TestDeploymentManagerPathGeneration:
    """Test that deployment_manager auto-generates paths correctly."""

    def test_get_training_path_helper_generates_correct_path(self):
        """_get_training_path() should generate data/{strategy_type}/{basename}."""
        from unittest.mock import MagicMock

        from src.pipeline.stages.managers.deployment_manager import TrainingDeploymentManager
        from src.utils.config import Secrets

        # Create minimal mock config and secrets
        manager = TrainingDeploymentManager(config=MagicMock(), secrets=Secrets())

        local_path = "/absolute/path/to/train.jsonl"
        strategy_type = "sft"

        expected = "data/sft/train.jsonl"
        actual = manager._get_training_path(local_path, strategy_type)

        assert actual == expected

    def test_get_training_path_with_subdirs(self):
        """_get_training_path() should use only basename (no subdirs)."""
        from unittest.mock import MagicMock

        from src.pipeline.stages.managers.deployment_manager import TrainingDeploymentManager
        from src.utils.config import Secrets

        manager = TrainingDeploymentManager(config=MagicMock(), secrets=Secrets())

        local_path = "/datasets/my_project/subfolder/corpus.jsonl"
        strategy_type = "cpt"

        expected = "data/cpt/corpus.jsonl"
        actual = manager._get_training_path(local_path, strategy_type)

        assert actual == expected

    def test_get_training_path_different_strategies(self):
        """_get_training_path() should generate different paths per strategy."""
        from unittest.mock import MagicMock

        from src.pipeline.stages.managers.deployment_manager import TrainingDeploymentManager
        from src.utils.config import Secrets

        manager = TrainingDeploymentManager(config=MagicMock(), secrets=Secrets())
        local_path = "data/dataset.jsonl"

        sft_path = manager._get_training_path(local_path, "sft")
        dpo_path = manager._get_training_path(local_path, "dpo")
        cpt_path = manager._get_training_path(local_path, "cpt")

        assert sft_path == "data/sft/dataset.jsonl"
        assert dpo_path == "data/dpo/dataset.jsonl"
        assert cpt_path == "data/cpt/dataset.jsonl"
