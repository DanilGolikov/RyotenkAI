"""Phase 14.A — :mod:`src.providers.training.interfaces` contract.

Pure-stdlib unit tests for the new types introduced by Phase 14.A:
* :class:`VolumeKind` — enum, string-value contract.
* :class:`AvailabilityVerdict` — frozen dataclass, state literals.
* :class:`ProviderCapabilities` — extended fields with safe defaults.
* :class:`ITerminalActionProvider` — runtime_checkable Protocol.

7-category coverage applied at type level — no provider impls
exercised here (they live in their own test files).
"""

from __future__ import annotations

import pytest

from ryotenkai_providers.training.interfaces import (
    AvailabilityVerdict,
    IGPUProvider,
    ITerminalActionProvider,
    ProviderCapabilities,
    VolumeKind,
)


# ---------------------------------------------------------------------------
# 1. Positive — happy-path construction + values
# ---------------------------------------------------------------------------


class TestVolumeKindPositive:
    def test_enum_values_match_env_string_contract(self) -> None:
        # Phase 14.A § R-2: enum values must equal the legacy
        # RUNPOD_VOLUME_KIND env string contract so the env-boundary
        # translation in Phase 14.D is `VolumeKind(env_str)` — not a
        # mapping table.
        assert VolumeKind.PERSISTENT.value == "persistent"
        assert VolumeKind.NETWORK.value == "network"
        assert VolumeKind.LOCAL_DISK.value == "local_disk"

    def test_enum_round_trip_through_string(self) -> None:
        # `VolumeKind("persistent") == VolumeKind.PERSISTENT` —
        # required for env-boundary translation.
        for k in VolumeKind:
            assert VolumeKind(k.value) is k


class TestAvailabilityVerdictPositive:
    def test_minimal_construction(self) -> None:
        v = AvailabilityVerdict(state="running", resource_id="pod-123")
        assert v.state == "running"
        assert v.resource_id == "pod-123"
        assert v.raw_status is None
        assert v.message == ""

    def test_full_construction(self) -> None:
        v = AvailabilityVerdict(
            state="sleeping_resumable",
            resource_id="pod-123",
            raw_status="EXITED",
            message="recoverable via podResume",
        )
        assert v.state == "sleeping_resumable"
        assert v.raw_status == "EXITED"
        assert v.message == "recoverable via podResume"


class TestProviderCapabilitiesPositive:
    def test_defaults_safe(self) -> None:
        # Phase 14.A: caller can construct with just the original
        # fields and inherit safe defaults for the new ones.
        caps = ProviderCapabilities(provider_type="cloud")
        assert caps.supports_lifecycle_actions is False
        assert caps.volume_kind is VolumeKind.PERSISTENT
        assert caps.has_pause_resume is False
        assert caps.runner_workspace_root == "/workspace"

    def test_full_construction_keyword_args(self) -> None:
        caps = ProviderCapabilities(
            provider_type="cloud",
            supports_lifecycle_actions=True,
            volume_kind=VolumeKind.NETWORK,
            has_pause_resume=False,
            runner_workspace_root="/data",
        )
        assert caps.supports_lifecycle_actions is True
        assert caps.volume_kind is VolumeKind.NETWORK
        assert caps.has_pause_resume is False
        assert caps.runner_workspace_root == "/data"


# ---------------------------------------------------------------------------
# 2. Negative — invalid state literals are rejected at runtime by Pyright/mypy,
#    not by the dataclass itself (Literal is a type hint, not a runtime check).
#    This test pins the behavior so reviewers know we did NOT add runtime
#    validation here — that's handled at the Pyright/mypy layer.
# ---------------------------------------------------------------------------


class TestAvailabilityVerdictNegative:
    def test_runtime_does_not_validate_state_literal(self) -> None:
        # Pin: dataclass doesn't enforce Literal at runtime. This is
        # by design — type checker catches it at lint time.
        v = AvailabilityVerdict(state="not_a_real_state", resource_id="x")  # type: ignore[arg-type]
        assert v.state == "not_a_real_state"

    def test_frozen_dataclass(self) -> None:
        v = AvailabilityVerdict(state="running", resource_id="pod-123")
        with pytest.raises(Exception):  # FrozenInstanceError
            v.resource_id = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 3. Boundary — empty resource_id, all five state values
# ---------------------------------------------------------------------------


class TestAvailabilityVerdictBoundary:
    def test_empty_resource_id_accepted(self) -> None:
        # Single_node uses empty string (no resource_id concept).
        v = AvailabilityVerdict(state="running", resource_id="")
        assert v.resource_id == ""

    def test_all_five_state_values(self) -> None:
        # Pin the canonical state vocabulary — operator dashboards
        # may key off these strings.
        for state in ("running", "sleeping_resumable", "gone",
                      "probe_failed", "unknown"):
            v = AvailabilityVerdict(state=state, resource_id="x")  # type: ignore[arg-type]
            assert v.state == state


# ---------------------------------------------------------------------------
# 4. Invariants — Protocol conformance
# ---------------------------------------------------------------------------


class TestProtocolInvariants:
    def test_iterminal_action_provider_runtime_checkable(self) -> None:
        # Build a tiny class that implements all three methods —
        # isinstance() should return True without explicit
        # inheritance.
        class _Stub:
            def terminate(self, *, resource_id: str, reason: str) -> None: ...
            def pause(self, *, resource_id: str) -> None: ...
            def resume(self, *, resource_id: str) -> None: ...
        assert isinstance(_Stub(), ITerminalActionProvider)

    def test_iterminal_action_provider_rejects_partial_impls(self) -> None:
        # Class with only 2 of 3 methods does NOT conform.
        class _Partial:
            def terminate(self, *, resource_id: str, reason: str) -> None: ...
            def pause(self, *, resource_id: str) -> None: ...
            # missing resume()
        # Note: runtime_checkable Protocol only checks method NAMES,
        # not signatures. A class missing `resume` fails isinstance.
        assert not isinstance(_Partial(), ITerminalActionProvider)

    def test_igpuprovider_runtime_checkable(self) -> None:
        # IGPUProvider must remain runtime_checkable after Phase 14.A
        # added new methods.
        assert hasattr(IGPUProvider, "_is_runtime_protocol")


# ---------------------------------------------------------------------------
# 5. Dependency errors — N/A (these are pure types with no I/O)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 6. Regressions — pre-Phase-14.A ProviderCapabilities construction
# ---------------------------------------------------------------------------


class TestRegressions:
    def test_pre_phase_14a_capabilities_construction_unchanged(self) -> None:
        # Pin: old code that constructed ProviderCapabilities with
        # only the legacy fields must continue to work without changes.
        caps = ProviderCapabilities(
            provider_type="cloud",
            supports_multi_gpu=True,
            supports_spot_instances=True,
            max_runtime_hours=None,
            gpu_name="H100",
            gpu_vram_gb=80.0,
        )
        # Legacy fields preserved.
        assert caps.provider_type == "cloud"
        assert caps.supports_multi_gpu is True
        assert caps.gpu_name == "H100"
        # New fields default safely.
        assert caps.supports_lifecycle_actions is False
        assert caps.volume_kind is VolumeKind.PERSISTENT


# ---------------------------------------------------------------------------
# 7. Logic-specific — VolumeKind <-> env string round-trip
# ---------------------------------------------------------------------------


class TestVolumeKindLogicSpecific:
    def test_lower_case_env_string_round_trip(self) -> None:
        # Phase 14.D will read RUNPOD_VOLUME_KIND env (string) and
        # convert to enum via VolumeKind(env_str). Pin the contract.
        assert VolumeKind("persistent") is VolumeKind.PERSISTENT
        assert VolumeKind("network") is VolumeKind.NETWORK
        assert VolumeKind("local_disk") is VolumeKind.LOCAL_DISK

    def test_unknown_env_string_raises(self) -> None:
        with pytest.raises(ValueError):
            VolumeKind("unknown_kind")
