"""Phase 14.A — factory-level capability ↔ Protocol conformance invariant.

THE invariant of Phase 14.A's two-Protocol split:

    For every concrete IGPUProvider impl,
    ``provider.get_capabilities().supports_lifecycle_actions``
    MUST equal ``isinstance(provider, ITerminalActionProvider)``.

If this drifts (provider author updates the flag but forgets to
inherit the Protocol — or vice versa), the entire Phase 14.B/C
runtime dispatch breaks: callers that gate on `isinstance` will
treat the provider as non-lifecycle, while callers that gate on the
flag will try to call methods that don't exist (or vice versa).

This test pins the invariant for both production providers. Adding
a third provider in the future MUST add a row here.
"""

from __future__ import annotations

import sys
import types
from typing import Any
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


from src.providers.runpod.training.provider import RunPodProvider  # noqa: E402
from src.providers.single_node.training.provider import SingleNodeProvider  # noqa: E402
from src.providers.training.interfaces import (  # noqa: E402
    IGPUProvider,
    ITerminalActionProvider,
    ProviderStatus,
)
from src.utils.config import Secrets  # noqa: E402


def _mk_runpod() -> RunPodProvider:
    """Build a RunPodProvider via __new__ to bypass the heavy
    Pydantic config validator (same pattern as
    ``test_provider_capabilities.py``)."""
    provider = object.__new__(RunPodProvider)
    provider._api_key = "rk_test"
    provider._status = ProviderStatus.AVAILABLE
    provider._pod_id = None
    provider._ssh_connection_info = None
    provider._gpu_info = None
    provider._pod_info = None
    provider._had_error = False
    cleanup_stub = MagicMock()
    cleanup_stub.keep_pod_on_error = False
    cleanup_stub.auto_stop_after_training = True
    training_stub = MagicMock()
    training_stub.gpu_type = "NVIDIA RTX A6000"
    config_stub = MagicMock()
    config_stub.cleanup = cleanup_stub
    config_stub.training = training_stub
    provider._config = config_stub
    provider._graphql_api_client = MagicMock()
    provider._api_client = MagicMock()
    return provider


def _mk_single_node() -> SingleNodeProvider:
    cfg: dict[str, Any] = {
        "connect": {"ssh": {"alias": "pc"}},
        "training": {"workspace_path": "/workspace"},
    }
    secrets = Secrets(HF_TOKEN="hf_test")
    return SingleNodeProvider(config=cfg, secrets=secrets)


# ---------------------------------------------------------------------------
# THE invariant
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "factory,expected_lifecycle",
    [
        pytest.param(_mk_runpod, True, id="runpod"),
        pytest.param(_mk_single_node, False, id="single_node"),
    ],
)
def test_capability_flag_matches_protocol_conformance(
    factory: Any, expected_lifecycle: bool,
) -> None:
    """Phase 14.A two-source-of-truth invariant.

    Failing this test = a provider author forgot to update one of:
      * ``ProviderCapabilities.supports_lifecycle_actions`` flag
      * inheritance from :class:`ITerminalActionProvider`

    Both must be in sync — flag reflects conformance, conformance
    reflects flag. The factory enforces this at boot via runtime
    assertion (Phase 14.A risk R-4 mitigation).
    """
    provider = factory()
    caps = provider.get_capabilities()

    # 1. Flag matches expectation.
    assert caps.supports_lifecycle_actions == expected_lifecycle

    # 2. Protocol conformance matches flag.
    is_lifecycle = isinstance(provider, ITerminalActionProvider)
    assert is_lifecycle == expected_lifecycle

    # 3. Cross-check the invariant directly.
    assert caps.supports_lifecycle_actions == is_lifecycle, (
        f"INVARIANT VIOLATED for {type(provider).__name__}: "
        f"capabilities.supports_lifecycle_actions={caps.supports_lifecycle_actions}, "
        f"isinstance(provider, ITerminalActionProvider)={is_lifecycle}. "
        "These two MUST agree — Phase 14.A two-source-of-truth contract."
    )


@pytest.mark.parametrize(
    "factory",
    [_mk_runpod, _mk_single_node],
    ids=["runpod", "single_node"],
)
def test_provider_satisfies_igpuprovider(factory: Any) -> None:
    """Every concrete provider must satisfy :class:`IGPUProvider`."""
    provider = factory()
    assert isinstance(provider, IGPUProvider)


@pytest.mark.parametrize(
    "factory,expected_kind",
    [
        pytest.param(_mk_runpod, "persistent", id="runpod"),
        pytest.param(_mk_single_node, "local_disk", id="single_node"),
    ],
)
def test_volume_kind_matches_provider_type(
    factory: Any, expected_kind: str,
) -> None:
    """Phase 14.A: ``volume_kind`` is provider-static and matches
    the documented contract:
      * RunPod cloud pods → persistent volumes by default.
      * Single_node host → local disk (no cloud volume semantics).
    """
    provider = factory()
    caps = provider.get_capabilities()
    assert caps.volume_kind.value == expected_kind


@pytest.mark.parametrize(
    "factory,expected_root",
    [
        pytest.param(_mk_runpod, "/workspace", id="runpod"),
        pytest.param(_mk_single_node, "/workspace", id="single_node"),
    ],
)
def test_runner_workspace_root_static(
    factory: Any, expected_root: str,
) -> None:
    """Both providers currently use ``/workspace``. Pin the value so
    a refactor can't silently change it (Phase 14.D consumes this
    field to replace hardcoded paths in ``_build_job_env``)."""
    provider = factory()
    caps = provider.get_capabilities()
    assert caps.runner_workspace_root == expected_root


# ---------------------------------------------------------------------------
# Phase 14.D+F invariants — capability flags + required_secrets parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "factory,expected_is_local",
    [
        pytest.param(_mk_runpod, False, id="runpod"),
        pytest.param(_mk_single_node, True, id="single_node"),
    ],
)
def test_is_local_capability_matches_provider_kind(
    factory: Any, expected_is_local: bool,
) -> None:
    """Phase 14.D+F — ``is_local`` flag replaces the
    ``provider_name == "single_node"`` string-check pattern in
    ``training_launcher`` and ``dependency_installer``."""
    caps = factory().get_capabilities()
    assert caps.is_local is expected_is_local


@pytest.mark.parametrize(
    "factory,expected_log_download",
    [
        pytest.param(_mk_runpod, True, id="runpod"),
        pytest.param(_mk_single_node, False, id="single_node"),
    ],
)
def test_supports_log_download_capability(
    factory: Any, expected_log_download: bool,
) -> None:
    """Phase 14.D+F — ``supports_log_download`` flag replaces the
    ``provider_name == PROVIDER_RUNPOD`` checks in
    :class:`GPUDeployer`. Cloud providers fetch logs over SCP/HTTP;
    local hosts already have logs on the filesystem."""
    caps = factory().get_capabilities()
    assert caps.supports_log_download is expected_log_download


@pytest.mark.parametrize(
    "factory,expected_secrets",
    [
        pytest.param(_mk_runpod, ("RUNPOD_API_KEY",), id="runpod"),
        pytest.param(_mk_single_node, (), id="single_node"),
    ],
)
def test_required_secrets_per_provider(
    factory: Any, expected_secrets: tuple[str, ...],
) -> None:
    """Phase 14.D+F — ``required_secrets()`` is the canonical list of
    operator-environment secrets the provider needs at startup.
    The startup validator iterates this tuple."""
    provider = factory()
    assert provider.required_secrets() == expected_secrets


@pytest.mark.parametrize(
    "factory",
    [_mk_runpod, _mk_single_node],
    ids=["runpod", "single_node"],
)
def test_required_secrets_subset_of_runtime_env_keys(factory: Any) -> None:
    """Phase 14.D+F § R-1 invariant — every name in
    :meth:`required_secrets` MUST appear as a key in
    :meth:`required_runtime_env_vars`. Catches the "ask for a
    secret, never use it" drift."""
    provider = factory()
    secrets = provider.required_secrets()
    runtime_env = provider.required_runtime_env_vars(resource_id="test-id")
    for secret_name in secrets:
        assert secret_name in runtime_env, (
            f"{secret_name} listed in required_secrets() but NOT "
            f"injected by required_runtime_env_vars(). Provider "
            f"is asking operator for a secret it never uses."
        )
