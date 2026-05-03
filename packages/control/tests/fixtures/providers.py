"""Phase 14.D+F — centralized :class:`IGPUProvider` test doubles.

Pre-14.D+F each test file that needed a provider stub rolled its
own ``_FakeProvider`` class implementing just enough of the Protocol
to make the test pass. Six+ variants accumulated across
``test_inference_deployer.py``, ``test_early_pod_release.py``, etc.

This module centralizes:

* :class:`FakeGPUProvider` — full :class:`IGPUProvider` impl with
  sensible defaults; subclass / kwargs to customize.
* :class:`FailingGPUProvider` — variant where every state-changing
  method returns :class:`Err`.

Usage::

    from src.tests.fixtures.providers import FakeGPUProvider

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

from src.providers.runpod.models import PodResourceInfo
from src.providers.training.interfaces import (
    AvailabilityVerdict,
    GPUInfo,
    IGPUProvider,
    ProviderCapabilities,
    ProviderStatus,
    SSHConnectionInfo,
    TrainingScriptHooks,
    VolumeKind,
)
from src.utils.result import Err, Ok, ProviderError, Result
from src.utils.ssh_client import SSHClient


@dataclass
class FakeGPUProvider:
    """Drop-in :class:`IGPUProvider` test double.

    Every method returns :class:`Ok` with sensible defaults.
    Override specific methods via constructor kwargs (e.g.
    ``connect_result=Err(...)``) or subclass for richer stubs.

    Phase 14.D+F: replaces ad-hoc ``_FakeProvider`` classes
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
    connect_result: Result[SSHConnectionInfo, ProviderError] | None = None
    disconnect_result: Result[None, ProviderError] | None = None
    check_gpu_result: Result[GPUInfo, ProviderError] | None = None
    hooks_result: Result[TrainingScriptHooks, ProviderError] | None = None
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

    def connect(self, *, run: Any) -> Result[SSHConnectionInfo, ProviderError]:
        if self.connect_result is not None:
            return self.connect_result
        return Ok(
            SSHConnectionInfo(
                host="fake.host",
                port=22,
                user="root",
                key_path="/tmp/fake_key",
                workspace_path="/workspace",
            ),
        )

    def disconnect(self) -> Result[None, ProviderError]:
        if self.disconnect_result is not None:
            return self.disconnect_result
        return Ok(None)

    def check_gpu(self) -> Result[GPUInfo, ProviderError]:
        if self.check_gpu_result is not None:
            return self.check_gpu_result
        return Ok(
            GPUInfo(
                name="Fake-GPU",
                vram_total_mb=24576,
                vram_used_mb=0,
                vram_free_mb=24576,
                utilization_percent=0.0,
            ),
        )

    def get_capabilities(self) -> ProviderCapabilities:
        return self.capabilities

    def prepare_training_script_hooks(
        self, ssh_client: SSHClient, context: dict[str, Any],
    ) -> Result[TrainingScriptHooks, ProviderError]:
        if self.hooks_result is not None:
            return self.hooks_result
        return Ok(TrainingScriptHooks.empty())

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
    """Variant where every state-changing method returns :class:`Err`.

    Useful for testing error-handling paths without writing
    per-test failure stubs.
    """

    def __post_init__(self) -> None:
        err = Err(
            ProviderError(
                message="fake provider failure",
                code="FAKE_FAILURE",
                details={},
            ),
        )
        if self.connect_result is None:
            self.connect_result = err
        if self.disconnect_result is None:
            self.disconnect_result = err
        if self.check_gpu_result is None:
            self.check_gpu_result = err
        if self.hooks_result is None:
            self.hooks_result = err


# Phase 14.D+F § R-5 mitigation — assert the fake conforms to the
# Protocol at module-import time. If a future Protocol method
# isn't implemented here, this fails fast.
_runtime_check: IGPUProvider = FakeGPUProvider()  # noqa: F841
