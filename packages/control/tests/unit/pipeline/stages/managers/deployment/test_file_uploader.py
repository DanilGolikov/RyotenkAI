"""Tests for the HTTP-based :class:`FileUploader` (Phase 3 PR-3.3).

After transport-unification-v2 the uploader's responsibility is
dataset collection + thin delegation to ``JobClient.upload_file``.
Wire-protocol tests for the endpoint itself live in
``packages/pod/tests/unit/runner/test_api_files.py``.

Test categories:
* positive       — successful upload of config + datasets
* negative       — missing config file → ConfigError
* negative       — referenced dataset missing on disk
* logic-specific — dataset collection walks strategy chain
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from ryotenkai_control.pipeline.stages.managers.deployment.file_uploader import (
    DEFAULT_WORKSPACE,
    FileUploader,
)
from ryotenkai_shared.contracts.runner_api.files import FileUploadTarget


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_strategy(strategy_type: str = "sft", dataset: str | None = None):  # type: ignore[no-untyped-def]
    s = MagicMock()
    s.strategy_type = strategy_type
    s.dataset = dataset
    return s


def _make_dataset_local(train: Path | None, eval_: Path | None = None):  # type: ignore[no-untyped-def]
    ds = MagicMock()
    ds.get_source_type.return_value = "local"
    paths = MagicMock()
    paths.train = str(train) if train else None
    paths.eval = str(eval_) if eval_ else None
    src = MagicMock()
    src.local_paths = paths
    ds.source_local = src
    return ds


def _make_uploader(
    config: MagicMock | None = None,
    workspace: str = DEFAULT_WORKSPACE,
) -> FileUploader:
    secrets = MagicMock()
    uploader = FileUploader(
        config=config or MagicMock(),
        secrets=secrets,
    )
    uploader.set_workspace(workspace)
    return uploader


@pytest.fixture
def fake_client() -> MagicMock:
    client = MagicMock()
    client.upload_file = AsyncMock(return_value=MagicMock(bytes_written=42))
    return client


# ---------------------------------------------------------------------------
# Positive
# ---------------------------------------------------------------------------


class TestPositive:
    def test_upload_config_only(self, fake_client, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
        config_file = tmp_path / "pipeline_config.yaml"
        config_file.write_text("model: test\n")

        cfg = MagicMock()
        cfg.training.get_strategy_chain.return_value = []
        cfg.get_primary_dataset.return_value = None  # no datasets

        uploader = _make_uploader(cfg)
        result = uploader.upload_via_http(
            fake_client, {"config_path": str(config_file)},
        )

        assert result.is_success(), result
        # Exactly one call — config only.
        assert fake_client.upload_file.await_count == 1
        target = fake_client.upload_file.await_args.args[0]
        assert target == FileUploadTarget.CONFIG.value

    def test_upload_config_plus_datasets(
        self, fake_client, tmp_path: Path,
    ) -> None:  # type: ignore[no-untyped-def]
        config_file = tmp_path / "pipeline_config.yaml"
        config_file.write_text("model: test\n")
        train_file = tmp_path / "train.jsonl"
        train_file.write_text('{"x": 1}\n')

        cfg = MagicMock()
        strategy = _make_strategy(strategy_type="sft", dataset="default")
        cfg.training.get_strategy_chain.return_value = [strategy]
        cfg.get_dataset_for_strategy.return_value = _make_dataset_local(train_file)
        cfg.resolve_path.return_value = train_file

        uploader = _make_uploader(cfg)
        result = uploader.upload_via_http(
            fake_client, {"config_path": str(config_file)},
        )

        assert result.is_success(), result
        # Two calls: config + dataset.
        assert fake_client.upload_file.await_count == 2
        targets = [c.args[0] for c in fake_client.upload_file.await_args_list]
        assert FileUploadTarget.CONFIG.value in targets
        assert FileUploadTarget.DATASET.value in targets


# ---------------------------------------------------------------------------
# Negative
# ---------------------------------------------------------------------------


class TestNegative:
    def test_missing_config_returns_err(self, fake_client) -> None:  # type: ignore[no-untyped-def]
        cfg = MagicMock()
        cfg.training.get_strategy_chain.return_value = []
        cfg.get_primary_dataset.return_value = None
        uploader = _make_uploader(cfg)

        result = uploader.upload_via_http(
            fake_client, {"config_path": "/nope/missing.yaml"},
        )

        assert result.is_failure()
        err = result.unwrap_err()
        assert err.code == "CONFIG_FILE_NOT_FOUND"
        # Client never invoked when config missing.
        assert fake_client.upload_file.await_count == 0

    def test_dataset_missing_on_disk_returns_err(
        self, fake_client, tmp_path: Path,
    ) -> None:  # type: ignore[no-untyped-def]
        config_file = tmp_path / "pipeline_config.yaml"
        config_file.write_text("x: 1\n")

        cfg = MagicMock()
        strategy = _make_strategy(dataset="default")
        cfg.training.get_strategy_chain.return_value = [strategy]
        cfg.get_dataset_for_strategy.return_value = _make_dataset_local(
            train=Path("/missing/file.jsonl"),
        )
        cfg.resolve_path.return_value = Path("/missing/file.jsonl")  # doesn't exist

        uploader = _make_uploader(cfg)
        result = uploader.upload_via_http(
            fake_client, {"config_path": str(config_file)},
        )

        assert result.is_failure()
        assert result.unwrap_err().code == "DATASET_FILE_NOT_FOUND"


# ---------------------------------------------------------------------------
# Logic-specific
# ---------------------------------------------------------------------------


class TestLogicSpecific:
    def test_dataset_collection_walks_strategy_chain(self, tmp_path: Path) -> None:
        train1 = tmp_path / "a.jsonl"
        train1.write_text("x")
        train2 = tmp_path / "b.jsonl"
        train2.write_text("y")

        cfg = MagicMock()
        s1 = _make_strategy(dataset="ds1")
        s2 = _make_strategy(dataset="ds2")
        cfg.training.get_strategy_chain.return_value = [s1, s2]

        def side_effect(strategy):  # type: ignore[no-untyped-def]
            if strategy.dataset == "ds1":
                return _make_dataset_local(train1)
            if strategy.dataset == "ds2":
                return _make_dataset_local(train2)
            return None

        cfg.get_dataset_for_strategy.side_effect = side_effect

        def resolve(p):  # type: ignore[no-untyped-def]
            return Path(p)

        cfg.resolve_path.side_effect = resolve

        uploader = _make_uploader(cfg)
        files, missing = uploader._collect_local_datasets()
        assert len(files) == 2
        assert {p.name for p in files} == {"a.jsonl", "b.jsonl"}
        assert missing == []
