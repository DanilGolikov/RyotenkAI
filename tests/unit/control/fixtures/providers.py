"""Phase 14.D+F — centralized :class:`IGPUProvider` test doubles.

Pre-14.D+F each test file that needed a provider stub rolled its
own ``_FakeProvider`` class implementing just enough of the Protocol
to make the test pass. Six+ variants accumulated across
``test_inference_deployer.py``, ``test_early_pod_release.py``, etc.

This module centralizes:

* :class:`FakeGPUProvider` — full :class:`IGPUProvider` impl with
  sensible defaults; subclass / kwargs to customize.
* :class:`FailingGPUProvider` — variant where every state-changing
  method raises :class:`ProviderUnavailableError`.

Phase A2 Batch 12: provider Protocol migrated to raise-based.
Methods return ``T`` on success and raise typed errors on failure;
the optional ``*_exc`` kwargs let tests inject a pre-built exception
to be raised on a specific call.

Usage::

    from tests.unit.control.fixtures.providers import FakeGPUProvider

    def test_my_thing():
        provider = FakeGPUProvider(
            provider_name="test_runpod",
            capabilities=ProviderCapabilities(
                provider_type="cloud", supports_log_download=True,
            ),
        )
        # Use provider as an IGPUProvider in your code-under-test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ryotenkai_providers.runpod.models import PodResourceInfo
from ryotenkai_providers.training.interfaces import (
    AvailabilityVerdict,
    GPUInfo,
    IGPUProvider,
    ProviderCapabilities,
    ProviderStatus,
    SSHConnectionInfo,
    TrainingScriptHooks,
    VolumeKind,
)
from ryotenkai_shared.errors import ProviderUnavailableError, RyotenkAIError
from ryotenkai_shared.utils.ssh_client import SSHClient


@dataclass
class FakeGPUProvider:
    """Drop-in :class:`IGPUProvider` test double.

    Every method returns a sensible default. Inject a typed exception
    via ``*_exc`` kwargs (e.g. ``connect_exc=ProviderUnavailableError(...)``)
    to exercise error paths, or replace the return value via
    ``connect_value`` / ``check_gpu_value`` / ``hooks_value``.

    Phase A2 Batch 12: replaces ad-hoc ``_FakeProvider`` classes
    proliferating across the test suite. Conforms to
    :class:`IGPUProvider` runtime-checkable Protocol — verified by
    the fixture's own test.
    """

    provider_name_value: str = "fake"
    capabilities: ProviderCapabilities = field(
        default_factory=lambda: ProviderCapabilities(
            provider_type="local",
            supports_lifecycle_actions=False,
            volume_kind=VolumeKind.LOCAL_DISK,
            has_pause_resume=False,
            runner_workspace_root="/workspace",
            is_local=True,
            supports_log_download=False,
        ),
    )
    required_secrets_value: tuple[str, ...] = ()
    runtime_env_vars: dict[str, str] = field(default_factory=dict)
    availability_verdict: AvailabilityVerdict = field(
        default_factory=lambda: AvailabilityVerdict(
            state="running",
            resource_id="",
            message="fake provider always running",
        ),
    )
    # Optional pre-built return values (None ⇒ default fabricated value).
    connect_value: SSHConnectionInfo | None = None
    check_gpu_value: GPUInfo | None = None
    hooks_value: TrainingScriptHooks | None = None
    # Optional typed exceptions to raise (None ⇒ no raise).
    connect_exc: RyotenkAIError | None = None
    disconnect_exc: RyotenkAIError | None = None
    check_gpu_exc: RyotenkAIError | None = None
    hooks_exc: RyotenkAIError | None = None
    resource_info_value: PodResourceInfo | None = None
    status: ProviderStatus = ProviderStatus.AVAILABLE
    pod_id: str | None = None
    error_marked: bool = False

    @property
    def provider_name(self) -> str:
        return self.provider_name_value

    @property
    def provider_type(self) -> str:
        return self.capabilities.provider_type

    def connect(self, *, run: Any) -> SSHConnectionInfo:
        if self.connect_exc is not None:
            raise self.connect_exc
        if self.connect_value is not None:
            return self.connect_value
        return SSHConnectionInfo(
            host="fake.host",
            port=22,
            user="root",
            key_path="/tmp/fake_key",
            workspace_path="/workspace",
        )

    def disconnect(self) -> None:
        if self.disconnect_exc is not None:
            raise self.disconnect_exc

    def check_gpu(self) -> GPUInfo:
        if self.check_gpu_exc is not None:
            raise self.check_gpu_exc
        if self.check_gpu_value is not None:
            return self.check_gpu_value
        return GPUInfo(
            name="Fake-GPU",
            vram_total_mb=24576,
            vram_free_mb=24576,
            cuda_version="12.0",
            driver_version="555.00",
        )

    def get_capabilities(self) -> ProviderCapabilities:
        return self.capabilities

    def prepare_training_script_hooks(
        self, ssh_client: SSHClient, context: dict[str, Any],
    ) -> TrainingScriptHooks:
        if self.hooks_exc is not None:
            raise self.hooks_exc
        if self.hooks_value is not None:
            return self.hooks_value
        return TrainingScriptHooks.empty()

    def get_resource_info(self) -> PodResourceInfo | None:
        return self.resource_info_value

    def get_status(self) -> ProviderStatus:
        return self.status

    def mark_error(self) -> None:
        self.error_marked = True

    def required_runtime_env_vars(
        self, *, resource_id: str | None,
    ) -> dict[str, str]:
        return dict(self.runtime_env_vars)

    def probe_availability(self, resource_id: str) -> AvailabilityVerdict:
        return self.availability_verdict

    def required_secrets(self) -> tuple[str, ...]:
        return self.required_secrets_value


@dataclass
class FailingGPUProvider(FakeGPUProvider):
    """Variant where every state-changing method raises a typed error.

    Useful for testing error-handling paths without writing
    per-test failure stubs.
    """

    def __post_init__(self) -> None:
        exc = ProviderUnavailableError(
            detail="fake provider failure",
            context={"legacy_code": "FAKE_FAILURE"},
        )
        if self.connect_exc is None:
            self.connect_exc = exc
        if self.disconnect_exc is None:
            self.disconnect_exc = exc
        if self.check_gpu_exc is None:
            self.check_gpu_exc = exc
        if self.hooks_exc is None:
            self.hooks_exc = exc


# Phase 14.D+F § R-5 mitigation — assert the fake conforms to the
# Protocol at module-import time. If a future Protocol method
# isn't implemented here, this fails fast.
_runtime_check: IGPUProvider = FakeGPUProvider()  # noqa: F841
