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

from ryotenkai_shared.config import (
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
            source=DatasetSourceLocal(
                local_paths=DatasetLocalPaths(
                    train=str(train_file),
                    eval=None
                )
            ),
        )

        assert dataset.source.local_paths.train == str(train_file)
        assert not hasattr(dataset.source, "training_paths")

    def test_get_source_uri_uses_local_paths_not_training_paths(self, tmp_path):
        """get_source_uri() should use local_paths.train (v6.0)."""
        train_file = tmp_path / "train.jsonl"
        train_file.write_text('{"text": "sample"}')

        dataset = DatasetConfig(
            source=DatasetSourceLocal(
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
            source=DatasetSourceLocal(
                local_paths=DatasetLocalPaths(
                    train=str(train_file),
                    eval=None
                )
            ),
        )

        ref = dataset.get_display_train_ref()
        assert ref == str(train_file)


# DEAD: `FileUploader._get_training_path` was removed during Phase 6.6.
# Training paths are now generated via `PodLayout` (see
# `ryotenkai_shared.infrastructure.pod_layout`). The tests below
# anchored on the removed private helper and exercise nothing real
# under the current API.
