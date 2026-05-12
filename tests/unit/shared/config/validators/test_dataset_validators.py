"""Dataset config validation — post discriminated-unions.

The legacy ``validate_dataset_source_blocks`` rule (source_type=X
requires source_X block) is gone — Pydantic's Tag-based discriminated
union enforces structural correctness at YAML load. This test file
validates the new shape and that legacy shapes are rejected.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ryotenkai_shared.config import DatasetConfig

pytestmark = pytest.mark.unit


class TestDiscriminatedSource:
    def test_positive_local_source(self) -> None:
        cfg = DatasetConfig.model_validate({
            "source": {
                "kind": "local",
                "local_paths": {"train": "data/train.jsonl", "eval": None},
            }
        })
        assert cfg.source.kind == "local"
        assert cfg.is_huggingface() is False

    def test_positive_huggingface_source(self) -> None:
        cfg = DatasetConfig.model_validate({
            "source": {
                "kind": "huggingface",
                "train_id": "tatsu-lab/alpaca",
                "eval_id": None,
            }
        })
        assert cfg.source.kind == "huggingface"
        assert cfg.is_huggingface() is True

    def test_unknown_kind_rejected(self) -> None:
        with pytest.raises(ValidationError):
            DatasetConfig.model_validate({"source": {"kind": "s3"}})

    def test_missing_kind_rejected(self) -> None:
        with pytest.raises(ValidationError):
            DatasetConfig.model_validate({
                "source": {"local_paths": {"train": "data.jsonl"}}
            })

    def test_local_kind_with_hf_field_rejected(self) -> None:
        """kind=local with huggingface-only field train_id ⇒ extra=forbid."""
        with pytest.raises(ValidationError, match=r"Extra inputs|extra"):
            DatasetConfig.model_validate({
                "source": {
                    "kind": "local",
                    "local_paths": {"train": "data.jsonl"},
                    "train_id": "should-not-be-here",
                }
            })

    def test_legacy_shape_rejected(self) -> None:
        """Old YAML shape (source_type + source_local flat fields) is
        rejected — extra='forbid' on DatasetConfig kicks in."""
        with pytest.raises(ValidationError, match=r"Extra inputs|extra|Field required"):
            DatasetConfig.model_validate({
                "source_type": "local",
                "source_local": {"local_paths": {"train": "data.jsonl"}},
            })

    def test_source_kind_returns_local(self) -> None:
        cfg = DatasetConfig.model_validate({
            "source": {"kind": "local", "local_paths": {"train": "x.jsonl"}},
        })
        assert cfg.source.kind == "local"

    def test_source_kind_returns_huggingface(self) -> None:
        cfg = DatasetConfig.model_validate({
            "source": {"kind": "huggingface", "train_id": "x"},
        })
        assert cfg.source.kind == "huggingface"
