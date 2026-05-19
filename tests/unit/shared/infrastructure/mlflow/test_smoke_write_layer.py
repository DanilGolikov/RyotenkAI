"""Phase M1 smoke tests for the MLflow write layer.

Confirms that the five new modules under
``ryotenkai_shared.infrastructure.mlflow`` are importable, that their
public surfaces match the documented Protocols / signatures, and that
:class:`DeadLetterBuffer` round-trips writes through drain.

These tests deliberately avoid the network — :class:`MlflowTransport`
construction is exercised against a stub ``mlflow`` module installed
into ``sys.modules`` (the real ``mlflow`` may not be importable in CI
trim builds).
"""

from __future__ import annotations

import sys
import types
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def stub_mlflow() -> Iterator[MagicMock]:
    """Install a stub ``mlflow`` module so MlflowTransport.__init__ runs.

    ``MlflowTransport`` performs a one-shot ``mlflow.set_tracking_uri``
    on construction. We patch the module surface enough to let that
    call succeed without a real MLflow install. The fixture also
    patches ``mlflow.tracking.MlflowClient`` for callers that exercise
    the lazy client accessor.
    """
    saved_modules = {
        name: sys.modules.get(name)
        for name in ("mlflow", "mlflow.tracking", "mlflow.entities", "mlflow.data")
    }

    stub_root = types.ModuleType("mlflow")
    stub_root.set_tracking_uri = MagicMock(name="mlflow.set_tracking_uri")

    stub_tracking = types.ModuleType("mlflow.tracking")
    stub_tracking.MlflowClient = MagicMock(name="MlflowClient")

    stub_entities = types.ModuleType("mlflow.entities")
    stub_entities.Metric = MagicMock(name="Metric")

    stub_data = types.ModuleType("mlflow.data")

    sys.modules["mlflow"] = stub_root
    sys.modules["mlflow.tracking"] = stub_tracking
    sys.modules["mlflow.entities"] = stub_entities
    sys.modules["mlflow.data"] = stub_data
    stub_root.tracking = stub_tracking
    stub_root.entities = stub_entities
    stub_root.data = stub_data
    try:
        yield stub_root
    finally:
        for name, mod in saved_modules.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod


# ---------------------------------------------------------------------------
# Import-surface smoke
# ---------------------------------------------------------------------------


def test_write_layer_modules_importable() -> None:
    """All five Phase M1 modules import without side-effects."""
    from ryotenkai_shared.infrastructure.mlflow.dataset import HFDatasetLogger
    from ryotenkai_shared.infrastructure.mlflow.dead_letter import DeadLetterBuffer
    from ryotenkai_shared.infrastructure.mlflow.journal_uploader import JournalUploader
    from ryotenkai_shared.infrastructure.mlflow.metric_sink import MetricSink
    from ryotenkai_shared.infrastructure.mlflow.transport import MlflowTransport

    # Each symbol must be a class (not e.g. a function alias).
    for cls in (
        DeadLetterBuffer,
        JournalUploader,
        MetricSink,
        MlflowTransport,
        HFDatasetLogger,
    ):
        assert isinstance(cls, type), f"{cls!r} should be a class"


# ---------------------------------------------------------------------------
# DeadLetterBuffer round-trip
# ---------------------------------------------------------------------------


def test_dead_letter_buffer_round_trip(tmp_path: Path) -> None:
    """write() then drain() returns the same triples in append order."""
    from ryotenkai_shared.infrastructure.mlflow.dead_letter import DeadLetterBuffer

    buf = DeadLetterBuffer(tmp_path / "dlq.jsonl")
    buf.write("run-1", {"loss": 0.5, "acc": 0.9}, step=10)
    buf.write("run-2", {"loss": 0.4}, step=11)
    assert buf.size_bytes() > 0
    assert not buf.is_full()

    drained = list(buf.drain())
    assert len(drained) == 2

    run_a, metrics_a, step_a = drained[0]
    assert run_a == "run-1"
    assert metrics_a == {"loss": 0.5, "acc": 0.9}
    assert step_a == 10

    run_b, metrics_b, step_b = drained[1]
    assert run_b == "run-2"
    assert metrics_b == {"loss": 0.4}
    assert step_b == 11

    # File should be unlinked after drain.
    assert buf.size_bytes() == 0


def test_dead_letter_buffer_rejects_empty_run_id(tmp_path: Path) -> None:
    """write() must refuse an empty run_id to catch caller bugs early."""
    from ryotenkai_shared.infrastructure.mlflow.dead_letter import DeadLetterBuffer

    buf = DeadLetterBuffer(tmp_path / "dlq.jsonl")
    with pytest.raises(ValueError, match="non-empty run_id"):
        buf.write("", {"loss": 0.1}, step=0)


def test_dead_letter_buffer_max_bytes_validation(tmp_path: Path) -> None:
    """max_bytes must be positive."""
    from ryotenkai_shared.infrastructure.mlflow.dead_letter import DeadLetterBuffer

    with pytest.raises(ValueError, match="max_bytes"):
        DeadLetterBuffer(tmp_path / "dlq.jsonl", max_bytes=0)


# ---------------------------------------------------------------------------
# MlflowTransport construction (with stub mlflow)
# ---------------------------------------------------------------------------


def test_mlflow_transport_construction_stamps_uri(stub_mlflow: MagicMock) -> None:
    """__init__ calls mlflow.set_tracking_uri exactly once with the URI."""
    from ryotenkai_shared.infrastructure.mlflow.auth import _AuthNone
    from ryotenkai_shared.infrastructure.mlflow.transport import MlflowTransport
    from ryotenkai_shared.infrastructure.mlflow.uri import RuntimeUri

    uri = RuntimeUri(uri="http://localhost:5000", role="control_plane")
    transport = MlflowTransport(uri, _AuthNone())

    stub_mlflow.set_tracking_uri.assert_called_once_with("http://localhost:5000")
    assert transport.tracking_uri == "http://localhost:5000"


def test_mlflow_transport_rejects_invalid_timeouts(stub_mlflow: MagicMock) -> None:
    """Non-positive timeouts must be rejected in __init__."""
    from ryotenkai_shared.infrastructure.mlflow.auth import _AuthNone
    from ryotenkai_shared.infrastructure.mlflow.transport import MlflowTransport
    from ryotenkai_shared.infrastructure.mlflow.uri import RuntimeUri

    uri = RuntimeUri(uri="http://localhost:5000", role="control_plane")
    with pytest.raises(ValueError, match="positive"):
        MlflowTransport(uri, _AuthNone(), connect_timeout_s=0)


# ---------------------------------------------------------------------------
# MetricSink / JournalUploader / HFDatasetLogger construction
# ---------------------------------------------------------------------------


def test_metric_sink_construction_with_dead_letter(
    stub_mlflow: MagicMock,
    tmp_path: Path,
) -> None:
    """MetricSink can be constructed against a transport stub + DLQ."""
    from ryotenkai_shared.infrastructure.mlflow.auth import _AuthNone
    from ryotenkai_shared.infrastructure.mlflow.dead_letter import DeadLetterBuffer
    from ryotenkai_shared.infrastructure.mlflow.metric_sink import MetricSink
    from ryotenkai_shared.infrastructure.mlflow.transport import MlflowTransport
    from ryotenkai_shared.infrastructure.mlflow.uri import RuntimeUri

    uri = RuntimeUri(uri="http://localhost:5000", role="control_plane")
    transport = MlflowTransport(uri, _AuthNone())
    dlq = DeadLetterBuffer(tmp_path / "dlq.jsonl")
    sink = MetricSink(transport, dead_letter=dlq)

    # Empty dict must short-circuit (no MLflow call); we just need it
    # to not raise.
    sink.log("run-1", {}, step=0)


def test_journal_uploader_construction(stub_mlflow: MagicMock) -> None:
    """JournalUploader exposes upload(run_id, journal_path, sha256)."""
    from ryotenkai_shared.infrastructure.mlflow.auth import _AuthNone
    from ryotenkai_shared.infrastructure.mlflow.journal_uploader import JournalUploader
    from ryotenkai_shared.infrastructure.mlflow.transport import MlflowTransport
    from ryotenkai_shared.infrastructure.mlflow.uri import RuntimeUri

    uri = RuntimeUri(uri="http://localhost:5000", role="control_plane")
    transport = MlflowTransport(uri, _AuthNone())
    uploader = JournalUploader(transport, retry_delays_s=(0.0, 0.0))
    assert callable(uploader.upload)


def test_hf_dataset_logger_construction(stub_mlflow: MagicMock) -> None:
    """HFDatasetLogger exposes log(run_id, ds, *, context, ...)."""
    from ryotenkai_shared.infrastructure.mlflow.auth import _AuthNone
    from ryotenkai_shared.infrastructure.mlflow.dataset import HFDatasetLogger
    from ryotenkai_shared.infrastructure.mlflow.transport import MlflowTransport
    from ryotenkai_shared.infrastructure.mlflow.uri import RuntimeUri

    uri = RuntimeUri(uri="http://localhost:5000", role="control_plane")
    transport = MlflowTransport(uri, _AuthNone())
    ds_logger = HFDatasetLogger(transport)
    assert callable(ds_logger.log)


# ---------------------------------------------------------------------------
# JournalUploader upload-skip on missing file
# ---------------------------------------------------------------------------


def test_journal_uploader_skips_missing_file(
    stub_mlflow: MagicMock,
    tmp_path: Path,
) -> None:
    """upload() is a no-op when the journal path doesn't exist."""
    from ryotenkai_shared.infrastructure.mlflow.auth import _AuthNone
    from ryotenkai_shared.infrastructure.mlflow.journal_uploader import JournalUploader
    from ryotenkai_shared.infrastructure.mlflow.transport import MlflowTransport
    from ryotenkai_shared.infrastructure.mlflow.uri import RuntimeUri

    uri = RuntimeUri(uri="http://localhost:5000", role="control_plane")
    transport = MlflowTransport(uri, _AuthNone())
    uploader = JournalUploader(transport, retry_delays_s=())
    # Should NOT raise even though the file is missing — by contract
    # JournalUploader never raises (manifest carries the failure flag).
    uploader.upload("run-1", tmp_path / "does-not-exist.jsonl", "abc")


# ---------------------------------------------------------------------------
# _ensure_experiment: handle MLflow's soft-delete lifecycle
# ---------------------------------------------------------------------------


def test_ensure_experiment_restores_soft_deleted(stub_mlflow: MagicMock) -> None:
    """A soft-deleted experiment must be restored, not blindly reused.

    MLflow's ``get_experiment_by_name`` returns experiments in both
    ``active`` and ``deleted`` lifecycle stages. Creating a run under a
    ``deleted`` experiment fails with
    ``INVALID_PARAMETER_VALUE: The experiment X must be in the 'active'
    state``. The transport must transparently restore in-place so the
    operator does not have to manually un-delete after touching the UI.
    """
    from ryotenkai_shared.infrastructure.mlflow.auth import _AuthNone
    from ryotenkai_shared.infrastructure.mlflow.transport import MlflowTransport
    from ryotenkai_shared.infrastructure.mlflow.uri import RuntimeUri

    deleted_exp = MagicMock()
    deleted_exp.experiment_id = "17"
    deleted_exp.lifecycle_stage = "deleted"

    client = MagicMock()
    client.get_experiment_by_name.return_value = deleted_exp

    uri = RuntimeUri(uri="http://localhost:5000", role="control_plane")
    transport = MlflowTransport(uri, _AuthNone())
    transport._client = client  # bypass lazy initialiser

    exp_id = transport._ensure_experiment("helixql-nl2hql")

    assert exp_id == "17"
    client.restore_experiment.assert_called_once_with("17")
    client.create_experiment.assert_not_called()


def test_ensure_experiment_returns_active_unchanged(stub_mlflow: MagicMock) -> None:
    """Active experiments are returned as-is — no restore, no create."""
    from ryotenkai_shared.infrastructure.mlflow.auth import _AuthNone
    from ryotenkai_shared.infrastructure.mlflow.transport import MlflowTransport
    from ryotenkai_shared.infrastructure.mlflow.uri import RuntimeUri

    active_exp = MagicMock()
    active_exp.experiment_id = "42"
    active_exp.lifecycle_stage = "active"

    client = MagicMock()
    client.get_experiment_by_name.return_value = active_exp

    uri = RuntimeUri(uri="http://localhost:5000", role="control_plane")
    transport = MlflowTransport(uri, _AuthNone())
    transport._client = client

    exp_id = transport._ensure_experiment("helixql-nl2hql")

    assert exp_id == "42"
    client.restore_experiment.assert_not_called()
    client.create_experiment.assert_not_called()


def test_ensure_experiment_creates_on_miss(stub_mlflow: MagicMock) -> None:
    """When the name is unknown, the experiment is created fresh."""
    from ryotenkai_shared.infrastructure.mlflow.auth import _AuthNone
    from ryotenkai_shared.infrastructure.mlflow.transport import MlflowTransport
    from ryotenkai_shared.infrastructure.mlflow.uri import RuntimeUri

    client = MagicMock()
    client.get_experiment_by_name.return_value = None
    client.create_experiment.return_value = "99"

    uri = RuntimeUri(uri="http://localhost:5000", role="control_plane")
    transport = MlflowTransport(uri, _AuthNone())
    transport._client = client

    exp_id = transport._ensure_experiment("new-exp")

    assert exp_id == "99"
    client.create_experiment.assert_called_once_with("new-exp")
    client.restore_experiment.assert_not_called()
