"""Smoke tests proving PR-1 scaffolding compiles and is importable.

Real test suites (manifest schema, registry, config union, vLLM runtime)
land in PR-2/PR-3.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def test_package_import() -> None:
    """The top-level package imports cleanly."""
    import ryotenkai_engines

    assert ryotenkai_engines.__version__ == "1.0.0"
    # Public API is intentionally empty in PR-1.
    assert ryotenkai_engines.__all__ == ()


def test_stub_modules_import() -> None:
    """Every stub module imports without error — no syntax-level breakage."""
    import ryotenkai_engines._config_union  # noqa: F401
    import ryotenkai_engines.capabilities  # noqa: F401
    import ryotenkai_engines.errors  # noqa: F401
    import ryotenkai_engines.images  # noqa: F401
    import ryotenkai_engines.interfaces  # noqa: F401
    import ryotenkai_engines.manifest  # noqa: F401
    import ryotenkai_engines.registry  # noqa: F401


def test_manifest_schema_version_constant() -> None:
    """``LATEST_ENGINE_SCHEMA_VERSION`` is the only stable export from
    ``manifest`` in PR-1 — schema model itself lands in PR-2."""
    from ryotenkai_engines.manifest import LATEST_ENGINE_SCHEMA_VERSION

    assert LATEST_ENGINE_SCHEMA_VERSION == 1


def test_image_resolution_constants_present() -> None:
    """Convention image-naming constants are stable across all PRs that
    follow — they form part of the public env contract for operators."""
    from ryotenkai_engines.images import (
        DEFAULT_IMAGE_REGISTRY,
        ENV_IMAGE_OVERRIDE_PATTERN,
        ENV_IMAGE_REGISTRY,
    )

    assert ENV_IMAGE_REGISTRY == "RYOTENKAI_INFERENCE_IMAGE_REGISTRY"
    assert DEFAULT_IMAGE_REGISTRY == "ryotenkai"
    # The pattern's only required substitution token.
    assert "{engine_upper}" in ENV_IMAGE_OVERRIDE_PATTERN
