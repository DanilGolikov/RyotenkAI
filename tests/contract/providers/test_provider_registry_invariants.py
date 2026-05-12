"""Cross-provider invariant tests for the manifest-driven registry.

Auto-discovers every ``provider.toml`` under
``packages/providers/src/ryotenkai_providers/`` and parametrizes the
invariant matrix over the discovered set — adding a new provider in
the future means dropping a manifest and the tests pick it up
zero-touch.

Replaces the legacy ``test_factory_capability_invariant.py`` matrix
that hand-listed each provider; the auto-discovery rewrite is
discussed in concurrent-gathering-hippo plan §F.3.
"""

from __future__ import annotations

import pytest

from ryotenkai_providers.registry import (
    ProviderRegistry,
    reset_registry,
)
from ryotenkai_providers.training.interfaces import (
    ITerminalActionProvider,
    ProviderBase,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def registry() -> ProviderRegistry:
    """One ProviderRegistry instance for the whole module.

    Production singleton is bypassed so a flaky test can't poison other
    tests via state leakage; ``reset_registry()`` ensures a clean walk.
    """
    reset_registry()
    return ProviderRegistry.from_filesystem()


def _all_provider_ids(registry: ProviderRegistry) -> tuple[str, ...]:
    return registry.list()


# ---------------------------------------------------------------------------
# POSITIVE: discovery
# ---------------------------------------------------------------------------


def test_registry_discovers_in_tree_providers(registry: ProviderRegistry) -> None:
    """At least the two shipped providers are discovered."""
    ids = registry.list()
    assert "runpod" in ids, f"runpod missing from registry: {ids}"
    assert "single_node" in ids, f"single_node missing from registry: {ids}"


def test_registry_load_emits_no_failures(registry: ProviderRegistry) -> None:
    """In-tree manifests must be valid; any LoadFailure is a regression."""
    failures = registry.failures()
    assert not failures, (
        f"in-tree manifests failed to load: "
        f"{[(f.provider_id, f.reason) for f in failures]}"
    )


# ---------------------------------------------------------------------------
# INVARIANT: capability flag ↔ Protocol membership parity
# ---------------------------------------------------------------------------


def _resolve_training_class(registry: ProviderRegistry, provider_id: str) -> type:
    """Force class resolution via ``_resolve_class`` (sets ClassVars too)."""
    return registry._resolve_class(provider_id, role_key="training")


@pytest.fixture
def all_provider_ids(registry: ProviderRegistry) -> tuple[str, ...]:
    return _all_provider_ids(registry)


def test_capability_protocol_parity_terminal_action(
    registry: ProviderRegistry, all_provider_ids: tuple[str, ...]
) -> None:
    """``supports_lifecycle_actions`` ↔ ``ITerminalActionProvider`` membership."""
    for pid in all_provider_ids:
        manifest = registry.get_manifest(pid)
        if "training" not in manifest.provider.roles:
            continue
        cls = _resolve_training_class(registry, pid)
        flag = manifest.capabilities.supports_lifecycle_actions
        is_terminal = issubclass(cls, ITerminalActionProvider)
        assert flag == is_terminal, (
            f"provider {pid!r} drift: capabilities.supports_lifecycle_actions="
            f"{flag} but isinstance(cls, ITerminalActionProvider)={is_terminal}"
        )


def test_capability_protocol_parity_recovery_probe(
    registry: ProviderRegistry, all_provider_ids: tuple[str, ...]
) -> None:
    """``supports_recovery_probe`` ↔ ``IRecoveryProbeProvider`` membership."""
    for pid in all_provider_ids:
        manifest = registry.get_manifest(pid)
        if "training" not in manifest.provider.roles:
            continue
        cls = _resolve_training_class(registry, pid)
        flag = manifest.capabilities.supports_recovery_probe
        has_method = hasattr(cls, "attempt_recovery") and callable(
            getattr(cls, "attempt_recovery", None)
        )
        assert flag == has_method, (
            f"provider {pid!r} drift: supports_recovery_probe={flag} but "
            f"attempt_recovery callable on class={has_method}"
        )


def test_capability_protocol_parity_capacity_classifier(
    registry: ProviderRegistry, all_provider_ids: tuple[str, ...]
) -> None:
    """``supports_capacity_error_detection`` ↔ ``ICapacityErrorClassifier``."""
    for pid in all_provider_ids:
        manifest = registry.get_manifest(pid)
        if "training" not in manifest.provider.roles:
            continue
        cls = _resolve_training_class(registry, pid)
        flag = manifest.capabilities.supports_capacity_error_detection
        has_method = hasattr(cls, "is_capacity_error") and callable(
            getattr(cls, "is_capacity_error", None)
        )
        assert flag == has_method, (
            f"provider {pid!r} drift: "
            f"supports_capacity_error_detection={flag} but "
            f"is_capacity_error callable on class={has_method}"
        )


# ---------------------------------------------------------------------------
# INVARIANT: identity (manifest.id == class.provider_id == folder name)
# ---------------------------------------------------------------------------


def test_provider_id_matches_class_attribute(
    registry: ProviderRegistry, all_provider_ids: tuple[str, ...]
) -> None:
    """After registry resolution, class ClassVar matches manifest id.

    Catches the ``hand-overrode provider_name`` regression class —
    i.e. someone hardcoding ``provider_name = "runpod_v2"`` on the class
    while the manifest still says ``runpod``.
    """
    for pid in all_provider_ids:
        manifest = registry.get_manifest(pid)
        if "training" in manifest.provider.roles:
            cls = _resolve_training_class(registry, pid)
            assert (
                cls._manifest_provider_id == pid
            ), f"{cls.__name__}._manifest_provider_id={cls._manifest_provider_id!r} != {pid!r}"


# ---------------------------------------------------------------------------
# INVARIANT: ProviderBase inheritance for every concrete training class
# ---------------------------------------------------------------------------


def test_training_classes_inherit_provider_base(
    registry: ProviderRegistry, all_provider_ids: tuple[str, ...]
) -> None:
    """Every training-role class must inherit ``ProviderBase``.

    The default ``provider_id`` / ``provider_name`` / ``provider_type``
    / ``get_capabilities`` impls live there; a class that doesn't
    inherit it would crash at runtime (or worse, silently use stub
    values).
    """
    for pid in all_provider_ids:
        manifest = registry.get_manifest(pid)
        if "training" not in manifest.provider.roles:
            continue
        cls = _resolve_training_class(registry, pid)
        assert issubclass(cls, ProviderBase), (
            f"training class {cls.__name__} for {pid!r} does NOT inherit "
            f"ProviderBase — identity accessors will fail"
        )


# ---------------------------------------------------------------------------
# INVARIANT: required_secrets manifest ↔ pre-factory resolution
# ---------------------------------------------------------------------------


def test_required_secrets_match_manifest(
    registry: ProviderRegistry, all_provider_ids: tuple[str, ...]
) -> None:
    """``registry.required_secrets`` must equal the manifest's required_env."""
    for pid in all_provider_ids:
        manifest = registry.get_manifest(pid)
        for role in manifest.provider.roles:
            via_registry = registry.required_secrets(pid, role=role)
            via_manifest = manifest.required_secret_names(role=role)
            assert via_registry == via_manifest, (
                f"provider {pid!r} role={role!r}: "
                f"registry={via_registry} != manifest={via_manifest}"
            )


# ---------------------------------------------------------------------------
# NEGATIVE: registry rejects unknown / role-mismatch
# ---------------------------------------------------------------------------


def test_create_training_unknown_id_returns_err(registry: ProviderRegistry) -> None:
    from ryotenkai_providers.registry import ProviderContext

    ctx = ProviderContext(
        provider_id="i_do_not_exist",
        pipeline_config=None,  # type: ignore[arg-type]
        provider_block={},
        secrets=None,  # type: ignore[arg-type]
    )
    res = registry.create_training("i_do_not_exist", ctx)
    assert res.is_failure()
    assert res.unwrap_err().code == "PROVIDER_NOT_REGISTERED"


def test_create_resume_provider_unavailable_returns_err(
    registry: ProviderRegistry,
) -> None:
    """SingleNode declares no resume_factory — registry returns clean Err."""
    if "single_node" not in registry.list():
        pytest.skip("single_node not registered")
    res = registry.create_resume_provider("single_node")
    assert res.is_failure()
    assert res.unwrap_err().code == "PROVIDER_RESUME_UNAVAILABLE"


# ---------------------------------------------------------------------------
# INVARIANT: pod_lifecycle_client present iff supports_lifecycle_actions
# ---------------------------------------------------------------------------


def test_pod_lifecycle_client_parity(
    registry: ProviderRegistry, all_provider_ids: tuple[str, ...]
) -> None:
    """Schema-level invariant doubled at runtime.

    Manifest's ``model_validator`` already enforces this (rejects
    manifests at LOAD); this test catches a regression where the schema
    invariant gets weakened or where a manifest somehow makes it past
    the validator (e.g. a schema bug).
    """
    for pid in all_provider_ids:
        manifest = registry.get_manifest(pid)
        flag = manifest.capabilities.supports_lifecycle_actions
        has_locator = manifest.entry_points.pod_lifecycle_client is not None
        assert flag == has_locator, (
            f"provider {pid!r}: supports_lifecycle_actions={flag} but "
            f"pod_lifecycle_client locator presence={has_locator}"
        )
