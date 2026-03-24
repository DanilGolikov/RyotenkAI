from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.config.validators.datasets import validate_dataset_source_blocks
from src.utils.config import DatasetConfig

pytestmark = pytest.mark.unit


class TestValidateDatasetSourceBlocks:
    def test_positive_local_with_source_local(self) -> None:
        _ = DatasetConfig(
            source_type="local",
            source_local={"local_paths": {"train": "data/train.jsonl", "eval": None}},
        )

    def test_positive_huggingface_with_source_hf(self) -> None:
        _ = DatasetConfig(
            source_type="huggingface",
            source_hf={"train_id": "tatsu-lab/alpaca", "eval_id": None},
        )

    def test_boundary_autodetect_hf_when_source_type_none(self) -> None:
        _ = DatasetConfig(
            source_type=None,
            source_hf={"train_id": "openai/gsm8k", "eval_id": None},
        )

    def test_boundary_autodetect_local_when_source_type_none(self) -> None:
        _ = DatasetConfig(
            source_type=None,
            source_local={"local_paths": {"train": "data/train.jsonl", "eval": None}},
        )

    def test_invariant_allows_both_blocks_when_source_type_local(self) -> None:
        # Current policy: we only require the active block, we do not forbid the other one.
        _ = DatasetConfig(
            source_type="local",
            source_local={"local_paths": {"train": "data/train.jsonl", "eval": None}},
            source_hf={"train_id": "tatsu-lab/alpaca", "eval_id": None},
        )

    def test_invariant_allows_both_blocks_when_source_type_huggingface(self) -> None:
        _ = DatasetConfig(
            source_type="huggingface",
            source_hf={"train_id": "tatsu-lab/alpaca", "eval_id": None},
            source_local={"local_paths": {"train": "data/train.jsonl", "eval": None}},
        )

    def test_negative_local_missing_source_local(self) -> None:
        with pytest.raises(ValidationError, match=r"source_type='local' requires 'source_local:' block"):
            _ = DatasetConfig(source_type="local")

    def test_negative_huggingface_missing_source_hf(self) -> None:
        with pytest.raises(ValidationError, match=r"source_type='huggingface' requires 'source_hf:' block"):
            _ = DatasetConfig(source_type="huggingface")

    def test_boundary_unknown_source_type_raises_value_error(self) -> None:
        # get_source_type() is normally constrained by Pydantic (Literal["local","huggingface"]),
        # but we still keep this branch tested for robustness.
        class Dummy:
            def get_source_type(self):
                return "weird"

            source_hf = None
            source_local = None

        with pytest.raises(ValueError, match=r"non supported source_type='weird'"):
            validate_dataset_source_blocks(Dummy())  # type: ignore[arg-type]

