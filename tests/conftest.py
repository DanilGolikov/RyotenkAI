"""Shared fixtures and plugin registration for the greenfield tests/ tree."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

# WHY: when running tests inside a git worktree, the editable installs of
# ``ryotenkai_*`` resolve to the parent worktree's ``packages/*/src``. To
# pick up additive Protocol modules created in this worktree (e.g.
# ``ryotenkai_shared.infrastructure.runpod_api``) without forcing
# ``uv sync`` to re-pin the editable paths, we prepend the worktree's
# package src directories to ``sys.path``. Production code never relies
# on this — it is purely a test-time convenience.
_WORKTREE_ROOT = Path(__file__).resolve().parent.parent
for _pkg_src in sorted((_WORKTREE_ROOT / "packages").glob("*/src")):
    _path_str = str(_pkg_src)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

from tests._harness import debug_bundle, telemetry  # noqa: E402  -- after sys.path patch
from tests._harness.clock import ManualClock  # noqa: E402  -- after sys.path patch

# WHY hypothesis profile registration in conftest: Phase 4 introduces
# hypothesis property tests under ``tests/property/``. We register two
# profiles here so any property-test module can rely on them existing
# at collection time without each module having to redo the boilerplate
# (the ``openapi_drift`` test still registers its own profiles locally
# because it predates the global setup).
from hypothesis import HealthCheck, settings as _hyp_settings  # noqa: E402

_hyp_settings.register_profile(
    "ci",
    max_examples=50,
    deadline=None,
    suppress_health_check=[
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ],
)
_hyp_settings.register_profile(
    "nightly",
    max_examples=5000,
    deadline=None,
    suppress_health_check=[
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ],
)
_hyp_settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "ci"))

if TYPE_CHECKING:
    from collections.abc import Iterator

# conftest.py живёт в самой директории tests/, так что относительные пути
# артефактов всегда якорятся к ней — независимо от того, какой ``rootpath``
# pytest вычислит для текущего запуска.
_TESTS_ROOT = Path(__file__).resolve().parent


def pytest_configure(config: pytest.Config) -> None:
    telemetry.register(config, output_dir=_TESTS_ROOT / ".telemetry")
    debug_bundle.register(config, output_dir=_TESTS_ROOT / ".debug_bundles")
    os.environ.setdefault("RYOTENKAI_TEST_SEED", "0")


def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[None]) -> None:
    telemetry.pytest_runtest_makereport(item, call)


@pytest.fixture
def manual_clock() -> ManualClock:
    return ManualClock()


@pytest.fixture
def tmp_telemetry_dir(tmp_path: Path) -> Iterator[Path]:
    target = tmp_path / "telemetry"
    target.mkdir()
    yield target
