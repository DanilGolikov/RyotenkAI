"""Greenfield conftest for ``tests/unit/control/``.

Carries over the autouse env-isolation fixtures from the legacy
``packages/control/tests/conftest.py`` so that migrated control tests do
NOT leak ``HF_TOKEN`` / ``RUNPOD_API_KEY`` / ``MLFLOW_TRACKING_URI`` into
later test modules (e.g.
``tests/unit/shared/config/test_secrets_hf_hub_propagation.py`` which
expects an empty token).

This file is INTENTIONALLY minimal — only the autouse fixtures are
ported. Per-test fixtures (``mock_config`` etc.) are kept local to each
migrated test module (Batch 6b pattern); we do NOT re-export them here.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_mlflow_tracking_uri(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Prevent MLflow from creating ``./mlruns`` in the repo root.

    Without this, an imported module that calls ``mlflow.set_tracking_uri``
    with no argument falls back to ``file:./mlruns`` and persists the
    artefact between tests. We redirect to a per-test ``tmp_path``.
    """
    if os.getenv("MLFLOW_TRACKING_URI") is None:
        monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file:{tmp_path / 'mlruns'}")


@pytest.fixture(autouse=True)
def _isolate_hf_secret_env_vars(monkeypatch: pytest.MonkeyPatch):
    """Strip ``HF_TOKEN`` / ``RUNPOD_API_KEY`` for the duration of the test
    AND on teardown.

    Several control tests exercise ``StartupValidator.set_hf_token_env``
    (or other paths) that DIRECTLY assign to ``os.environ["HF_TOKEN"]``
    inside production code. ``monkeypatch.delenv`` only tracks values it
    knew about at setup time — it does NOT revert mutations the test
    itself caused. Without the explicit teardown below, those writes
    leak into later test modules — notably
    ``tests/unit/shared/config/test_secrets_hf_hub_propagation.py`` whose
    happy-path assertions expect an empty token.
    """
    for key in ("HF_TOKEN", "RUNPOD_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    yield
    # Explicitly drop any value the test (or its CUT) installed.
    for key in ("HF_TOKEN", "RUNPOD_API_KEY"):
        os.environ.pop(key, None)
