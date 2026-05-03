"""Phase 14.B — :mod:`src.runner.runtime.provider_registry` contract.

Pure-stdlib unit tests for the env-driven resolver. Slim-venv
compatible — uses ``sys.modules`` stub for ``runpod`` SDK so the
RunPod client builder import doesn't fail.

7-cat coverage. Tests pin the edge-case table from Phase 14.B
§ 14.B.3.3 + the Phase 14.B § 8.2 § 12 cross-Protocol invariant.
"""

from __future__ import annotations

import sys
import types
from typing import Any
from unittest.mock import MagicMock

import pytest

# Stub `runpod` SDK before any RunPod imports — slim CI venv lacks it.
if "runpod" not in sys.modules:
    _stub = types.ModuleType("runpod")
    _stub.api_key = ""
    _stub.create_pod = MagicMock()
    _stub.get_pod = MagicMock()
    _stub.stop_pod = MagicMock()
    _stub.resume_pod = MagicMock()
    _stub.terminate_pod = MagicMock()
    sys.modules["runpod"] = _stub


from ryotenkai_shared.constants import (  # noqa: E402
    PROVIDER_RUNPOD,
    PROVIDER_SINGLE_NODE,
    RUNTIME_PROVIDER_ENV_VAR,
)
from ryotenkai_pod.runner.runtime.lifecycle_client import IPodLifecycleClient  # noqa: E402
from ryotenkai_pod.runner.runtime.provider_registry import (  # noqa: E402
    BootstrapConfigError,
    registered_providers,
    resolve_keep_on_error_from_env,
    resolve_lifecycle_client_from_env,
    resolve_resource_id_from_env,
    resolve_volume_kind_from_env,
)


# ---------------------------------------------------------------------------
# 1. Positive — happy paths
# ---------------------------------------------------------------------------


class TestPositive:
    def test_runpod_full_env_resolves_to_runpod_client(self) -> None:
        env = {
            RUNTIME_PROVIDER_ENV_VAR: "runpod",
            "RUNPOD_API_KEY": "rk-secret",
            "RUNPOD_POD_ID": "pod-abc",
        }
        client = resolve_lifecycle_client_from_env(env)
        assert isinstance(client, IPodLifecycleClient)
        assert client.provider_name == PROVIDER_RUNPOD

    def test_single_node_minimal_env_resolves_to_noop(self) -> None:
        env = {RUNTIME_PROVIDER_ENV_VAR: "single_node"}
        client = resolve_lifecycle_client_from_env(env)
        assert isinstance(client, IPodLifecycleClient)
        assert client.provider_name == PROVIDER_SINGLE_NODE

    def test_single_node_does_not_require_runpod_env(self) -> None:
        # § R-9: single-node bootstrap with NO RunPod env still works.
        env = {RUNTIME_PROVIDER_ENV_VAR: "single_node"}
        client = resolve_lifecycle_client_from_env(env)
        assert client.provider_name == PROVIDER_SINGLE_NODE


# ---------------------------------------------------------------------------
# 2. Negative — bootstrap config errors
# ---------------------------------------------------------------------------


class TestNegativeBootstrap:
    def test_unset_provider_env_raises(self) -> None:
        with pytest.raises(BootstrapConfigError) as exc_info:
            resolve_lifecycle_client_from_env({})
        assert RUNTIME_PROVIDER_ENV_VAR in str(exc_info.value)
        # Operator-friendly message lists known providers.
        assert "single_node" in str(exc_info.value)
        assert "runpod" in str(exc_info.value)

    def test_empty_provider_env_raises(self) -> None:
        # Empty string treated same as unset.
        with pytest.raises(BootstrapConfigError):
            resolve_lifecycle_client_from_env({RUNTIME_PROVIDER_ENV_VAR: ""})

    def test_runpod_without_api_key_raises(self) -> None:
        env = {
            RUNTIME_PROVIDER_ENV_VAR: "runpod",
            "RUNPOD_POD_ID": "pod-abc",
        }
        with pytest.raises(BootstrapConfigError) as exc_info:
            resolve_lifecycle_client_from_env(env)
        assert "RUNPOD_API_KEY" in str(exc_info.value)

    def test_runpod_without_pod_id_raises(self) -> None:
        env = {
            RUNTIME_PROVIDER_ENV_VAR: "runpod",
            "RUNPOD_API_KEY": "rk-secret",
        }
        with pytest.raises(BootstrapConfigError) as exc_info:
            resolve_lifecycle_client_from_env(env)
        assert "RUNPOD_POD_ID" in str(exc_info.value)

    def test_unregistered_provider_raises(self) -> None:
        env = {RUNTIME_PROVIDER_ENV_VAR: "lambda"}
        with pytest.raises(BootstrapConfigError) as exc_info:
            resolve_lifecycle_client_from_env(env)
        assert "lambda" in str(exc_info.value)
        # Known list surfaced for operator-friendliness.
        assert "single_node" in str(exc_info.value)
        assert "runpod" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 3. Boundary — whitespace, casing, edge values
# ---------------------------------------------------------------------------


class TestBoundary:
    def test_whitespace_around_provider_name_is_NOT_stripped(self) -> None:
        # Phase 14.B contract: whitespace breaks the bootstrap.
        # Operators must set the env cleanly. Test pins this to
        # surface accidental shell-quoting bugs.
        env = {RUNTIME_PROVIDER_ENV_VAR: " runpod "}
        with pytest.raises(BootstrapConfigError):
            resolve_lifecycle_client_from_env(env)

    def test_provider_name_is_case_sensitive(self) -> None:
        # ``RUNPOD`` (uppercase) is not registered — sentinel for
        # "casing is part of the public contract".
        env = {RUNTIME_PROVIDER_ENV_VAR: "RUNPOD"}
        with pytest.raises(BootstrapConfigError):
            resolve_lifecycle_client_from_env(env)


# ---------------------------------------------------------------------------
# 4. Invariants — registered_providers + cross-Protocol invariant
# ---------------------------------------------------------------------------


class TestInvariants:
    def test_registered_providers_is_sorted_tuple(self) -> None:
        names = registered_providers()
        assert isinstance(names, tuple)
        assert names == tuple(sorted(names))
        # Currently exactly the two we ship.
        assert PROVIDER_RUNPOD in names
        assert PROVIDER_SINGLE_NODE in names

    def test_every_registered_builder_returns_matching_provider_name(
        self,
    ) -> None:
        # Phase 14.B § 8.2 § 11 invariant: builder output's
        # provider_name MUST match the registry key.
        # Forces parity if a provider author adds a new entry but
        # forgets to wire ``provider_name``.
        for name in registered_providers():
            env: dict[str, str] = {RUNTIME_PROVIDER_ENV_VAR: name}
            if name == PROVIDER_RUNPOD:
                env["RUNPOD_API_KEY"] = "k"
                env["RUNPOD_POD_ID"] = "p"
            client = resolve_lifecycle_client_from_env(env)
            assert client.provider_name == name


# ---------------------------------------------------------------------------
# 5. Dependency errors — N/A (no transport at this layer)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 6. Regressions — env helper round-trips preserve pre-14.B behaviour
# ---------------------------------------------------------------------------


class TestRegressionsEnvHelpers:
    def test_resolve_volume_kind_persistent_default(self) -> None:
        assert resolve_volume_kind_from_env({}) == "persistent"

    def test_resolve_volume_kind_lowercase_round_trip(self) -> None:
        assert resolve_volume_kind_from_env({"RUNPOD_VOLUME_KIND": "persistent"}) == "persistent"
        assert resolve_volume_kind_from_env({"RUNPOD_VOLUME_KIND": "network"}) == "network"

    def test_resolve_volume_kind_uppercase_normalized(self) -> None:
        assert resolve_volume_kind_from_env({"RUNPOD_VOLUME_KIND": "NETWORK"}) == "network"

    def test_resolve_volume_kind_invalid_clamped_to_persistent(self) -> None:
        # Pre-14.B behaviour: unknown values fall back, not raise.
        assert resolve_volume_kind_from_env({"RUNPOD_VOLUME_KIND": "foo"}) == "persistent"
        assert resolve_volume_kind_from_env({"RUNPOD_VOLUME_KIND": ""}) == "persistent"

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("true", True),
            ("True", True),
            ("TRUE", True),
            ("false", False),
            ("yes", False),       # only literal "true"
            ("1", False),         # not a truthy synonym
            ("", False),
        ],
    )
    def test_resolve_keep_on_error_only_literal_true(
        self, raw: str, expected: bool,
    ) -> None:
        assert resolve_keep_on_error_from_env({"RUNPOD_KEEP_ON_ERROR": raw}) == expected

    def test_resolve_keep_on_error_unset_is_false(self) -> None:
        assert resolve_keep_on_error_from_env({}) is False


# ---------------------------------------------------------------------------
# 7. Logic-specific — resource_id resolution per provider
# ---------------------------------------------------------------------------


class TestLogicSpecificResourceId:
    def test_runpod_resource_id_from_pod_id(self) -> None:
        env = {
            RUNTIME_PROVIDER_ENV_VAR: "runpod",
            "RUNPOD_POD_ID": "pod-zzz",
        }
        assert resolve_resource_id_from_env(env) == "pod-zzz"

    def test_single_node_resource_id_is_empty(self) -> None:
        env = {RUNTIME_PROVIDER_ENV_VAR: "single_node"}
        assert resolve_resource_id_from_env(env) == ""

    def test_runpod_resource_id_missing_returns_empty(self) -> None:
        # Defensive — caller should validate via
        # resolve_lifecycle_client_from_env first, but if helper
        # is invoked directly with bad env it returns empty string
        # rather than raising.
        env = {RUNTIME_PROVIDER_ENV_VAR: "runpod"}
        assert resolve_resource_id_from_env(env) == ""


# ---------------------------------------------------------------------------
# Cross-Protocol invariant (Phase 14.B § 8.2 § 12)
# ---------------------------------------------------------------------------


class TestCrossProtocolInvariant:
    """Phase 14.B § 1.1 + § 8.2 § 12 — for every Mac-side provider
    that conforms to :class:`ITerminalActionProvider`, there's a
    matching registered :class:`IPodLifecycleClient` builder.
    """

    def _build_runpod_provider(self) -> Any:
        # Same pattern as
        # :func:`tests.unit.providers.runpod.training.test_provider_capabilities._mk_provider`
        # — bypasses the heavy Pydantic config validator.
        from ryotenkai_providers.runpod.training.provider import RunPodProvider
        from ryotenkai_providers.training.interfaces import ProviderStatus

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

    def test_runpod_mac_side_lifecycle_provider_has_runner_side_client(
        self,
    ) -> None:
        # Mac side: RunPodProvider conforms to ITerminalActionProvider.
        from ryotenkai_providers.training.interfaces import ITerminalActionProvider
        provider = self._build_runpod_provider()
        assert isinstance(provider, ITerminalActionProvider)

        # Runner side: registry has a matching builder for the same
        # provider_name.
        registered = registered_providers()
        assert provider.provider_name in registered

    def test_single_node_no_lifecycle_protocol_no_runner_side_lifecycle_actions(
        self,
    ) -> None:
        # Mac side: SingleNodeProvider does NOT conform to
        # ITerminalActionProvider — but it's STILL registered on the
        # runner side (as NoOp). That's by design: every provider
        # needs a lifecycle client, even if it's a no-op.
        from ryotenkai_shared.constants import PROVIDER_SINGLE_NODE
        registered = registered_providers()
        assert PROVIDER_SINGLE_NODE in registered

        # And the builder produces a NoOp that returns SKIPPED — the
        # NoOp tests pin this.
