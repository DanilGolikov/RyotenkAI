"""Phase 14.A — :class:`SingleNodeProvider` capability methods contract.

Pin the new Phase 14.A methods on single_node:
* :meth:`required_runtime_env_vars` returns ONLY the bootstrap env var.
* :meth:`probe_availability` returns ``state="running"`` instantly.
* :meth:`get_capabilities` populates new fields correctly.
* SingleNodeProvider does NOT conform to :class:`ITerminalActionProvider`.

7-category coverage. Slim-venv compatible — no RunPod SDK needed.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.constants import PROVIDER_SINGLE_NODE, RUNTIME_PROVIDER_ENV_VAR
from src.providers.single_node.training.provider import SingleNodeProvider
from src.providers.training.interfaces import (
    AvailabilityVerdict,
    ITerminalActionProvider,
    ProviderCapabilities,
    VolumeKind,
)
from src.utils.config import Secrets


def _mk_provider(**overrides: Any) -> SingleNodeProvider:
    # Phase 6.6 cutover removed `docker_image` (image is pinned to
    # the runtime release image now). The config schema rejects
    # unknown fields, so the test fixture stays minimal.
    cfg: dict[str, Any] = {
        "connect": {"ssh": {"alias": "pc"}},
        "training": {"workspace_path": "/workspace"},
    }
    cfg.update(overrides)
    secrets = Secrets(HF_TOKEN="hf_test")
    return SingleNodeProvider(config=cfg, secrets=secrets)


# ---------------------------------------------------------------------------
# 1. Positive — defaults match contract
# ---------------------------------------------------------------------------


class TestPositive:
    def test_required_runtime_env_vars_minimal(self) -> None:
        provider = _mk_provider()
        env = provider.required_runtime_env_vars(resource_id=None)
        # Single_node returns ONLY the bootstrap env var.
        assert env == {RUNTIME_PROVIDER_ENV_VAR: PROVIDER_SINGLE_NODE}

    def test_required_runtime_env_vars_resource_id_ignored(self) -> None:
        # Single_node has no per-resource credentials — resource_id
        # is intentionally ignored.
        provider = _mk_provider()
        env_with = provider.required_runtime_env_vars(resource_id="some-id")
        env_without = provider.required_runtime_env_vars(resource_id=None)
        assert env_with == env_without

    def test_probe_availability_returns_running_immediately(self) -> None:
        # Pin: NO network round-trip. The host is always-on.
        provider = _mk_provider()
        v = provider.probe_availability("any-id")
        assert isinstance(v, AvailabilityVerdict)
        assert v.state == "running"
        assert v.resource_id == "any-id"

    def test_get_capabilities_populates_phase_14a_fields(self) -> None:
        provider = _mk_provider()
        caps = provider.get_capabilities()
        assert isinstance(caps, ProviderCapabilities)
        # Phase 14.A capability surface:
        assert caps.supports_lifecycle_actions is False
        assert caps.volume_kind is VolumeKind.LOCAL_DISK
        assert caps.has_pause_resume is False
        assert caps.runner_workspace_root == "/workspace"


# ---------------------------------------------------------------------------
# 2. Negative — invariant: NOT an ITerminalActionProvider
# ---------------------------------------------------------------------------


class TestNegativeNoLifecycleActions:
    def test_does_not_conform_to_iterminal_action_provider(self) -> None:
        # Phase 14.A: SingleNodeProvider intentionally does NOT
        # implement ITerminalActionProvider so the type checker
        # rejects `provider.pause()` / `.resume()` / `.terminate()`
        # at the callsite.
        provider = _mk_provider()
        assert not isinstance(provider, ITerminalActionProvider)

    def test_lifecycle_methods_absent(self) -> None:
        # Pin: NO `terminate` / `pause` / `resume` methods on the class.
        # Adding them later would silently violate the contract.
        provider = _mk_provider()
        for method in ("terminate", "pause", "resume"):
            assert not hasattr(provider, method), (
                f"SingleNodeProvider must NOT expose {method!r} — "
                "violates Phase 14.A capability gating."
            )


# ---------------------------------------------------------------------------
# 3. Boundary — empty resource_id, None resource_id
# ---------------------------------------------------------------------------


class TestBoundary:
    def test_probe_with_empty_resource_id(self) -> None:
        provider = _mk_provider()
        v = provider.probe_availability("")
        assert v.state == "running"
        assert v.resource_id == ""

    def test_required_env_vars_returns_dict_not_view(self) -> None:
        # Caller must be able to mutate the returned dict (e.g. merge
        # into a larger env). Pin: must return a fresh dict.
        provider = _mk_provider()
        env = provider.required_runtime_env_vars(resource_id=None)
        env["X"] = "y"  # must not raise


# ---------------------------------------------------------------------------
# 4. Invariants — capability flag ↔ Protocol conformance
# ---------------------------------------------------------------------------


class TestInvariants:
    def test_capability_flag_matches_protocol_conformance(self) -> None:
        # Phase 14.A two-source-of-truth invariant.
        provider = _mk_provider()
        caps = provider.get_capabilities()
        assert caps.supports_lifecycle_actions is False
        assert not isinstance(provider, ITerminalActionProvider)

    def test_probe_never_raises(self) -> None:
        # Phase 14.A contract: probe_availability is "fast,
        # never-raises". Even pathological inputs should return a
        # verdict, not raise.
        provider = _mk_provider()
        for rid in ("", "x", "🦙", "x" * 1000):
            v = provider.probe_availability(rid)
            assert v.state == "running"


# ---------------------------------------------------------------------------
# 5. Dependency errors — N/A (no I/O in single_node capability methods)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 6. Regressions — existing get_capabilities behavior preserved
# ---------------------------------------------------------------------------


class TestRegressions:
    def test_legacy_capabilities_fields_preserved(self) -> None:
        # Pre-Phase-14.A fields still present and unchanged.
        provider = _mk_provider()
        caps = provider.get_capabilities()
        assert caps.provider_type == "local"
        assert caps.supports_multi_gpu is False
        assert caps.supports_spot_instances is False


# ---------------------------------------------------------------------------
# 7. Logic-specific — bootstrap env var contract
# ---------------------------------------------------------------------------


class TestLogicSpecific:
    def test_bootstrap_env_var_value_is_provider_name(self) -> None:
        # Phase 14.A: RYOTENKAI_RUNTIME_PROVIDER value MUST equal
        # provider.provider_name so the runner-side registry
        # (Phase 14.B) can lookup the right impl.
        provider = _mk_provider()
        env = provider.required_runtime_env_vars(resource_id=None)
        assert env[RUNTIME_PROVIDER_ENV_VAR] == provider.provider_name
        assert env[RUNTIME_PROVIDER_ENV_VAR] == PROVIDER_SINGLE_NODE
