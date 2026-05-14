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
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from ryotenkai_control.pipeline.stages.managers.deployment.file_uploader import (
    DEFAULT_WORKSPACE,
    FileUploader,
)
from ryotenkai_shared.contracts.runner_api.files import FileUploadTarget
from ryotenkai_shared.errors import ConfigInvalidError, SSHTransferFailedError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_strategy(strategy_type: str = "sft", dataset: str | None = None):  # type: ignore[no-untyped-def]
    s = SimpleNamespace(strategy_type=strategy_type, dataset=dataset)
    return s


def _make_dataset_local(train: Path | None, eval_: Path | None = None):  # type: ignore[no-untyped-def]
    from ryotenkai_shared.config import DatasetSourceLocal
    from ryotenkai_shared.config.datasets.sources import DatasetLocalPaths

    ds = MagicMock()
    src = DatasetSourceLocal(
        local_paths=DatasetLocalPaths(
            train=str(train) if train else "",
            eval=str(eval_) if eval_ else None,
        ),
    )
    ds.source = src
    ds.source_local = src  # legacy shim
    ds.get_source_type.return_value = "local"  # legacy shim
    return ds


def _make_uploader(
    config: MagicMock | None = None,
    workspace: str = DEFAULT_WORKSPACE,
) -> FileUploader:
    secrets = SimpleNamespace()
    uploader = FileUploader(
        config=config or MagicMock(),
        secrets=secrets,
    )
    uploader.set_workspace(workspace)
    return uploader


@pytest.fixture
def fake_client() -> MagicMock:
    client = SimpleNamespace(upload_file=AsyncMock(return_value=MagicMock(bytes_written=42)))
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
        # Phase A2 Batch 9 (raise-based): success returns None.
        result = uploader.upload_via_http(
            fake_client, {"config_path": str(config_file)},
        )
        assert result is None
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
        assert result is None
        # Two calls: config + dataset.
        assert fake_client.upload_file.await_count == 2
        targets = [c.args[0] for c in fake_client.upload_file.await_args_list]
        assert FileUploadTarget.CONFIG.value in targets
        assert FileUploadTarget.DATASET.value in targets


# ---------------------------------------------------------------------------
# Negative
# ---------------------------------------------------------------------------


class TestNegative:
    def test_missing_config_raises_config_invalid(self, fake_client) -> None:  # type: ignore[no-untyped-def]
        """Phase A2 Batch 9 (raise-based): missing config raises
        :class:`ConfigInvalidError` with ``reason="CONFIG_FILE_NOT_FOUND"``."""
        cfg = MagicMock()
        cfg.training.get_strategy_chain.return_value = []
        cfg.get_primary_dataset.return_value = None
        uploader = _make_uploader(cfg)

        with pytest.raises(ConfigInvalidError) as exc_info:
            uploader.upload_via_http(
                fake_client, {"config_path": "/nope/missing.yaml"},
            )
        assert exc_info.value.context.get("reason") == "CONFIG_FILE_NOT_FOUND"
        # Client never invoked when config missing.
        assert fake_client.upload_file.await_count == 0

    def test_dataset_missing_on_disk_raises_config_invalid(
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
        with pytest.raises(ConfigInvalidError) as exc_info:
            uploader.upload_via_http(
                fake_client, {"config_path": str(config_file)},
            )
        assert exc_info.value.context.get("reason") == "DATASET_FILE_NOT_FOUND"
        assert exc_info.value.context.get("missing") == ["/missing/file.jsonl"]

    def test_transport_failure_raises_ssh_transfer_failed(
        self, tmp_path: Path,
    ) -> None:
        """Boundary: when the underlying HTTP upload raises, the
        wrapper translates to :class:`SSHTransferFailedError` so
        callers can distinguish transport from validation failures."""
        config_file = tmp_path / "pipeline_config.yaml"
        config_file.write_text("model: test\n")

        cfg = MagicMock()
        cfg.training.get_strategy_chain.return_value = []
        cfg.get_primary_dataset.return_value = None

        # Client that raises on upload — simulates a closed tunnel.
        boom_client = MagicMock()
        boom_client.upload_file = AsyncMock(side_effect=RuntimeError("conn closed"))

        uploader = _make_uploader(cfg)
        with pytest.raises(SSHTransferFailedError) as exc_info:
            uploader.upload_via_http(
                boom_client, {"config_path": str(config_file)},
            )
        assert exc_info.value.context.get("reason") == "HTTP_FILE_UPLOAD_FAILED"
        # __cause__ preserves the original RuntimeError for trace fidelity.
        assert isinstance(exc_info.value.__cause__, RuntimeError)


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
