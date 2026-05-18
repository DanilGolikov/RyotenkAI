"""Unit tests for :class:`ModelPublisher` (Phase M4)."""

from __future__ import annotations

import pytest

from ryotenkai_pod.trainer.mlflow.model_publisher import ModelPublisher
from tests._fakes.mlflow_model_registry import FakeModelRegistry


def test_publish_registers_with_runs_uri_and_sets_alias() -> None:
    registry = FakeModelRegistry()
    publisher = ModelPublisher(registry=registry)

    version = publisher.publish(
        run_id="run-abc",
        artifact_path="model",
        registered_name="ryotenkai/exp/family",
        alias_on_success="challenger",
    )

    assert version.name == "ryotenkai/exp/family"
    assert version.version == "1"
    assert version.run_id == "run-abc"
    # FakeModelRegistry records the URI on the register call.
    assert registry.register_calls[0].model_uri == "runs:/run-abc/model"
    assert registry.set_alias_calls[0].alias == "challenger"


def test_publish_default_alias_is_challenger() -> None:
    registry = FakeModelRegistry()
    publisher = ModelPublisher(registry=registry)
    version = publisher.publish(
        run_id="r1",
        artifact_path="model",
        registered_name="m",
    )
    assert registry.set_alias_calls[-1].alias == "challenger"
    assert registry.set_alias_calls[-1].version == version.version


def test_publish_alias_failure_propagates() -> None:
    class _ExplodingRegistry(FakeModelRegistry):
        def set_alias(self, name: str, alias: str, version: str) -> None:
            raise RuntimeError("registry offline")

    publisher = ModelPublisher(registry=_ExplodingRegistry())
    with pytest.raises(RuntimeError, match="registry offline"):
        publisher.publish(
            run_id="r1",
            artifact_path="model",
            registered_name="m",
        )


def test_publish_does_not_accept_save_pretrained_kwarg() -> None:
    """Phase M5: ``save_pretrained`` was removed from the publisher API.

    The artifact-logging step (``mlflow.transformers.log_model`` with
    ``save_pretrained=True``, R-21) is the caller's responsibility and
    runs BEFORE :meth:`ModelPublisher.publish`. The publisher only
    owns the registry half (register + alias).
    """
    registry = FakeModelRegistry()
    publisher = ModelPublisher(registry=registry)
    with pytest.raises(TypeError):
        publisher.publish(  # type: ignore[call-arg]
            run_id="r1",
            artifact_path="model",
            registered_name="m",
            save_pretrained=True,
        )
