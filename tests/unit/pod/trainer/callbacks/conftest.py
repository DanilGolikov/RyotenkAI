"""Conftest for callbacks tests — clean up sys.modules pollution.

Several callback tests stub ``sys.modules["ryotenkai_pod.trainer"]``
with a bare ``ModuleType`` shell so they can ``importlib`` the
callback source file without triggering the trainer package's
heavy ``__init__`` chain. The stubs leak across tests and break
``tests/unit/pod/trainer/utils/test_container_unit.py`` which
expects the real package object to be present.

Session-scoped fixture force-imports the real
``ryotenkai_pod.trainer`` package up-front so that the stubs
either find the real module already in ``sys.modules`` (and skip
the shell installation entirely, via the
``if "ryotenkai_pod.trainer" not in _sys.modules`` guard) or are
replaced with the real package before downstream tests run.
"""

from __future__ import annotations

import importlib
import sys


def _force_load_trainer_package() -> None:
    """Replace any ``ModuleType`` shell with the real package object."""
    real = importlib.import_module("ryotenkai_pod.trainer")
    # The container module triggers the rest of the package's __init__
    # chain, ensuring the real package is fully loaded.
    importlib.import_module("ryotenkai_pod.trainer.container")
    importlib.import_module("ryotenkai_pod.trainer.memory_manager")
    sys.modules["ryotenkai_pod.trainer"] = real
    # Some callback test stubs also override the parent package's
    # ``trainer`` attribute; make sure the real one is restored.
    pod_pkg = sys.modules.get("ryotenkai_pod")
    if pod_pkg is not None and getattr(pod_pkg, "trainer", None) is not real:
        pod_pkg.trainer = real  # type: ignore[attr-defined]


# Run at import time so the real trainer package is in ``sys.modules``
# before any test module in this directory does
# ``if "ryotenkai_pod.trainer" not in _sys.modules: install shell``.
_force_load_trainer_package()
