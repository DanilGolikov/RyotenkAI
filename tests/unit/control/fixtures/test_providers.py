"""Phase 14.D+F — :class:`FakeGPUProvider` fixture self-test.

Pin that the centralized test double conforms to
:class:`IGPUProvider` so future Protocol additions don't silently
drift away from the fake.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest

# Stub `runpod` SDK at module load — slim CI venv doesn't have it.
if "runpod" not in sys.modules:
    _stub = types.ModuleType("runpod")
    _stub.api_key = ""
    _stub.create_pod = MagicMock()
    _stub.get_pod = MagicMock()
    _stub.stop_pod = MagicMock()
    _stub.resume_pod = MagicMock()
    _stub.terminate_pod = MagicMock()
    sys.modules["runpod"] = _stub


from ryotenkai_providers.training.interfaces import (  # noqa: E402
    IGPUProvider,
    ProviderCapabilities,
    VolumeKind,
)
from .providers import (  # noqa: E402
    FailingGPUProvider,
    FakeGPUProvider,
)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Pre-existing IGPUProvider Protocol drift — production Protocol "
        "gained ``pod_layout_for_run`` and ``provider_id`` methods (and "
        "``ProviderCapabilities`` gained 7 new fields) that the legacy "
        "FakeGPUProvider does not yet implement. Tracked in xfail_debt.md."
    ),
)
class TestProtocolConformance:
    def test_fake_conforms_to_igpuprovider(self) -> None:
        assert isinstance(FakeGPUProvider(), IGPUProvider)

    def test_failing_conforms_to_igpuprovider(self) -> None:
        assert isinstance(FailingGPUProvider(), IGPUProvider)


class TestDefaults:
    def test_default_provider_name(self) -> None:
        assert FakeGPUProvider().provider_name == "fake"

    def test_default_capabilities_is_local_true(self) -> None:
        # Default fake mimics a local provider — useful for the
        # majority of tests that don't care about provider type.
        caps = FakeGPUProvider().get_capabilities()
        assert caps.is_local is True
        assert caps.volume_kind == VolumeKind.LOCAL_DISK

    def test_default_required_secrets_empty(self) -> None:
        assert FakeGPUProvider().required_secrets() == ()


class TestOverrides:
    def test_can_inject_capabilities(self) -> None:
        custom_caps = ProviderCapabilities(
            provider_type="cloud",
            is_local=False,
            supports_log_download=True,
            volume_kind=VolumeKind.PERSISTENT,
        )
        provider = FakeGPUProvider(capabilities=custom_caps)
        assert provider.get_capabilities().provider_type == "cloud"
        assert provider.get_capabilities().is_local is False

    def test_can_inject_required_secrets(self) -> None:
        provider = FakeGPUProvider(
            required_secrets_value=("MY_SECRET",),
        )
        assert provider.required_secrets() == ("MY_SECRET",)


class TestFailingVariant:
    def test_connect_returns_err(self) -> None:
        provider = FailingGPUProvider()
        result = provider.connect(run=None)
        assert result.is_failure()

    def test_check_gpu_returns_err(self) -> None:
        provider = FailingGPUProvider()
        result = provider.check_gpu()
        assert result.is_failure()
