"""Drift guard for ``runtime_check._REQUIRED_SRC_MODULES``.

The post-sync importability gate (PR-A) imports each entry of
``docker/training/runtime_check.py:_REQUIRED_SRC_MODULES`` via
``importlib.import_module``. Critically: importing a module **executes
its top-level imports transitively**, so listing the trainer entrypoint
(``src.training.run_training``) automatically validates every
``src.*`` module the trainer touches at cold-start time. Explicit
entries below the entrypoint are belt-and-braces for failure modes
seen in production: the recurring ``src.providers`` missing-package
incident (``run_20260429_171726_49j32`` and the 15-crash incident on
2026-05-02), where importing the trainer worked in test but
``src.providers.<concrete_provider>`` was missing on the pod due to a
partial rsync.

These two invariants protect the gate from drift:

1. **Entrypoints present**: the trainer's CLI entry
   (``src.training.run_training``) and the runner's HTTP entry
   (``src.runner.main``) must always be in the list. Without them the
   gate would not even attempt to validate the actual processes that
   spawn on the pod.
2. **Sanity**: every entry is uniquely named and looks like a
   ``src.…`` dotted path. Catches typos that would silently turn into
   ``ImportError`` on every run.

We deliberately do NOT enforce "every top-level import in
``run_training.py`` must be a separate entry" — that would force-list
~30 transitively-validated modules with zero added safety, since the
entrypoint already imports them.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


_RUNTIME_CHECK_PATH = Path("docker/training/runtime_check.py")

# Entrypoints whose absence in the gate list would defeat the gate's
# whole purpose. Keep this set exactly aligned with what the pod runs.
_REQUIRED_ENTRYPOINTS: set[str] = {
    "ryotenkai_pod.trainer.run_training",  # trainer CLI invoked by training_launcher
    "ryotenkai_pod.runner.main",            # uvicorn FastAPI app the runner spins up
}


def _load_required_src_modules() -> list[str]:
    """Import ``_REQUIRED_SRC_MODULES`` from runtime_check.py via spec
    loader so we don't depend on PYTHONPATH including ``docker/``."""
    spec = importlib.util.spec_from_file_location(
        "runtime_check_under_test",
        _RUNTIME_CHECK_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return list(module._REQUIRED_SRC_MODULES)  # type: ignore[attr-defined]


def test_runtime_check_lists_both_entrypoints():
    """If this fails: the gate stopped checking either the trainer or
    the runner entrypoint. Importing the entrypoint transitively
    validates every top-level import of the actual process the pod will
    spawn — without it, the gate is blind to entire crash classes.

    Action: add the missing entry back to
    ``docker/training/runtime_check.py:_REQUIRED_SRC_MODULES``.
    """
    required = _load_required_src_modules()
    missing = sorted(_REQUIRED_ENTRYPOINTS - set(required))
    assert not missing, (
        f"Required entrypoints absent from _REQUIRED_SRC_MODULES: {missing}. "
        f"Without them the import gate would not validate the actual processes "
        f"that spawn on the pod. Re-add to docker/training/runtime_check.py."
    )


def test_runtime_check_required_modules_are_unique_and_dotted():
    """Sanity: no duplicates, all entries look like ``ryotenkai_<pkg>.…`` paths."""
    required = _load_required_src_modules()
    assert len(required) == len(set(required)), f"Duplicate entries: {required}"
    for name in required:
        assert name.startswith("ryotenkai_"), f"Non-ryotenkai_*.* entry: {name!r}"
        assert " " not in name, f"Whitespace in entry: {name!r}"


def test_runtime_check_required_modules_nonempty():
    """Empty list = gate validates nothing = pointless gate.

    This is the simplest invariant we can assert and the cheapest check
    to surface a misconfiguration like an accidental wholesale deletion.
    """
    required = _load_required_src_modules()
    assert required, "_REQUIRED_SRC_MODULES is empty — gate cannot protect anything"
