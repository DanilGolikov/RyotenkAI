"""Tests for the UX-layer pre-validation integration resolver.

The resolver walks a raw YAML dict (post ``yaml.safe_load`` but
pre ``PipelineConfig`` validation), inlines ``integration: <id>``
shortcuts from ``~/.ryotenkai/integrations/<id>/current.yaml``, and
returns a clean dict.

Categories: positive, negative, boundary, invariants, dependency-error,
regression, logic-specific.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from src.workspace.integrations.exceptions import (
    IntegrationNotFoundError,
    IntegrationUnresolvedError,
)
from src.workspace.integrations.registry import IntegrationRegistry
from src.workspace.integrations.resolver import resolve_yaml_integrations
from src.workspace.integrations.store import IntegrationStore

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _create_integration(
    workspace_root: Path,
    *,
    integration_id: str,
    integration_type: str,
    current_yaml: str,
) -> None:
    """Lay down a registered integration on disk."""
    registry = IntegrationRegistry(root=workspace_root)
    integration_path = registry.default_integration_path(integration_id)
    store = IntegrationStore(integration_path)
    store.create(id=integration_id, name=integration_id, type=integration_type)
    (integration_path / "current.yaml").write_text(current_yaml, encoding="utf-8")
    registry.register(
        integration_id=integration_id,
        name=integration_id,
        type=integration_type,
        path=integration_path,
    )


_MLFLOW_INTEGRATION_YAML = """
tracking_uri: "https://mlflow.example.com"
local_tracking_uri: "http://localhost:5002"
ca_bundle_path: null
system_metrics:
  callback_enabled: true
"""


# ---------------------------------------------------------------------------
# Positive
# ---------------------------------------------------------------------------


class TestPositive:
    def test_mlflow_integration_inlined(self, tmp_path: Path) -> None:
        _create_integration(
            tmp_path,
            integration_id="prod-mlflow",
            integration_type="mlflow",
            current_yaml=_MLFLOW_INTEGRATION_YAML,
        )
        registry = IntegrationRegistry(root=tmp_path)

        raw = {
            "integrations": {
                "mlflow": {
                    "integration": "prod-mlflow",
                    "experiment_name": "my-exp",
                },
            },
        }

        out = resolve_yaml_integrations(raw, registry=registry)
        mlflow = out["integrations"]["mlflow"]
        assert mlflow["tracking_uri"] == "https://mlflow.example.com"
        assert mlflow["local_tracking_uri"] == "http://localhost:5002"
        assert mlflow["experiment_name"] == "my-exp"
        # Integration id retained as secrets-tag.
        assert mlflow["integration"] == "prod-mlflow"

    def test_project_keys_win_over_integration(self, tmp_path: Path) -> None:
        """``tracking_uri:`` written explicitly in the project YAML
        overrides the integration default. Useful for dev branches that
        want a different tracker without forking the integration."""
        _create_integration(
            tmp_path,
            integration_id="prod-mlflow",
            integration_type="mlflow",
            current_yaml=_MLFLOW_INTEGRATION_YAML,
        )
        registry = IntegrationRegistry(root=tmp_path)

        raw = {
            "integrations": {
                "mlflow": {
                    "integration": "prod-mlflow",
                    "experiment_name": "my-exp",
                    "tracking_uri": "https://override.example.com",
                },
            },
        }

        out = resolve_yaml_integrations(raw, registry=registry)
        # Project wins.
        assert (
            out["integrations"]["mlflow"]["tracking_uri"]
            == "https://override.example.com"
        )
        # Integration default still fills in fields the project didn't override.
        assert (
            out["integrations"]["mlflow"]["local_tracking_uri"]
            == "http://localhost:5002"
        )

    def test_huggingface_existence_check_only(self, tmp_path: Path) -> None:
        """HF is a registry presence check + secrets tag — no content
        merge (the integration's body is empty/just a token blob)."""
        _create_integration(
            tmp_path,
            integration_id="my-hf",
            integration_type="huggingface",
            current_yaml="\n",  # body irrelevant for HF
        )
        registry = IntegrationRegistry(root=tmp_path)

        raw = {
            "integrations": {
                "huggingface": {
                    "integration": "my-hf",
                    "repo_id": "user/repo",
                    "private": True,
                },
            },
        }

        out = resolve_yaml_integrations(raw, registry=registry)
        # HF block left structurally unchanged — integration is the
        # secrets-tag, repo_id / private are project-side.
        assert out["integrations"]["huggingface"] == {
            "integration": "my-hf",
            "repo_id": "user/repo",
            "private": True,
        }


# ---------------------------------------------------------------------------
# Negative
# ---------------------------------------------------------------------------


class TestNegative:
    def test_missing_mlflow_integration_raises_not_found(
        self, tmp_path: Path
    ) -> None:
        registry = IntegrationRegistry(root=tmp_path)
        raw = {
            "integrations": {
                "mlflow": {
                    "integration": "ghost-mlflow",
                    "experiment_name": "x",
                },
            },
        }
        with pytest.raises(IntegrationNotFoundError) as exc:
            resolve_yaml_integrations(raw, registry=registry)
        assert exc.value.integration_id == "ghost-mlflow"
        assert exc.value.integration_type == "mlflow"

    def test_missing_huggingface_integration_raises_not_found(
        self, tmp_path: Path
    ) -> None:
        registry = IntegrationRegistry(root=tmp_path)
        raw = {
            "integrations": {
                "huggingface": {
                    "integration": "ghost-hf",
                    "repo_id": "user/repo",
                },
            },
        }
        with pytest.raises(IntegrationNotFoundError):
            resolve_yaml_integrations(raw, registry=registry)

    def test_type_mismatch_raises_unresolved(self, tmp_path: Path) -> None:
        """Project YAML asks for an mlflow integration but the registered
        id is huggingface-typed."""
        _create_integration(
            tmp_path,
            integration_id="my-hf",
            integration_type="huggingface",
            current_yaml="\n",
        )
        registry = IntegrationRegistry(root=tmp_path)
        raw = {
            "integrations": {
                "mlflow": {
                    "integration": "my-hf",
                    "experiment_name": "x",
                },
            },
        }
        with pytest.raises(IntegrationUnresolvedError) as exc:
            resolve_yaml_integrations(raw, registry=registry)
        assert "huggingface" in str(exc.value).lower()
        assert "mlflow" in str(exc.value).lower()

    def test_empty_integration_yaml_raises_unresolved(
        self, tmp_path: Path
    ) -> None:
        _create_integration(
            tmp_path,
            integration_id="empty-mlflow",
            integration_type="mlflow",
            current_yaml="",
        )
        registry = IntegrationRegistry(root=tmp_path)
        raw = {
            "integrations": {
                "mlflow": {
                    "integration": "empty-mlflow",
                    "experiment_name": "x",
                },
            },
        }
        with pytest.raises(IntegrationUnresolvedError) as exc:
            resolve_yaml_integrations(raw, registry=registry)
        assert "empty" in str(exc.value).lower()


# ---------------------------------------------------------------------------
# Boundary
# ---------------------------------------------------------------------------


class TestBoundary:
    def test_no_integrations_block_passthrough(
        self, tmp_path: Path
    ) -> None:
        """Project YAML without ``integrations`` section at all
        — resolver no-ops."""
        registry = IntegrationRegistry(root=tmp_path)
        raw = {"model": {"name": "stub"}}
        out = resolve_yaml_integrations(raw, registry=registry)
        assert out == raw

    def test_no_integration_keyword_passthrough(self, tmp_path: Path) -> None:
        """Project YAML with all-inline mlflow values (no
        ``integration:`` shorthand) — resolver leaves it alone."""
        registry = IntegrationRegistry(root=tmp_path)
        raw = {
            "integrations": {
                "mlflow": {
                    "tracking_uri": "https://inline.example.com",
                    "experiment_name": "x",
                },
            },
        }
        out = resolve_yaml_integrations(raw, registry=registry)
        assert out == raw

    def test_empty_integration_string_passthrough(self, tmp_path: Path) -> None:
        """``integration: ""`` → treated like absent (Pydantic will
        reject the resulting block downstream for missing tracking_uri).
        We don't lookup the registry."""
        registry = IntegrationRegistry(root=tmp_path)
        raw = {
            "integrations": {
                "mlflow": {
                    "integration": "",
                    "experiment_name": "x",
                },
            },
        }
        out = resolve_yaml_integrations(raw, registry=registry)
        # Empty integration → no registry lookup, no error. The block
        # is left structurally intact (downstream Pydantic will complain
        # about missing tracking_uri).
        assert "integration" in out["integrations"]["mlflow"]
        assert out["integrations"]["mlflow"]["integration"] == ""


# ---------------------------------------------------------------------------
# Invariants
# ---------------------------------------------------------------------------


class TestInvariants:
    def test_input_is_not_mutated(self, tmp_path: Path) -> None:
        """The resolver returns a NEW dict — caller can keep the
        original to show in UI."""
        _create_integration(
            tmp_path,
            integration_id="prod-mlflow",
            integration_type="mlflow",
            current_yaml=_MLFLOW_INTEGRATION_YAML,
        )
        registry = IntegrationRegistry(root=tmp_path)
        raw = {
            "integrations": {
                "mlflow": {
                    "integration": "prod-mlflow",
                    "experiment_name": "x",
                },
            },
        }
        snapshot = copy.deepcopy(raw)
        resolve_yaml_integrations(raw, registry=registry)
        assert raw == snapshot

    def test_idempotent_when_already_inlined(self, tmp_path: Path) -> None:
        """Resolving an already-inlined dict twice yields the same
        result."""
        _create_integration(
            tmp_path,
            integration_id="prod-mlflow",
            integration_type="mlflow",
            current_yaml=_MLFLOW_INTEGRATION_YAML,
        )
        registry = IntegrationRegistry(root=tmp_path)
        raw = {
            "integrations": {
                "mlflow": {
                    "integration": "prod-mlflow",
                    "experiment_name": "x",
                },
            },
        }
        first = resolve_yaml_integrations(raw, registry=registry)
        second = resolve_yaml_integrations(first, registry=registry)
        assert first == second


# ---------------------------------------------------------------------------
# Logic-specific
# ---------------------------------------------------------------------------


class TestLogicSpecific:
    def test_mlflow_and_huggingface_resolved_in_one_pass(
        self, tmp_path: Path
    ) -> None:
        _create_integration(
            tmp_path,
            integration_id="prod-mlflow",
            integration_type="mlflow",
            current_yaml=_MLFLOW_INTEGRATION_YAML,
        )
        _create_integration(
            tmp_path,
            integration_id="my-hf",
            integration_type="huggingface",
            current_yaml="\n",
        )
        registry = IntegrationRegistry(root=tmp_path)

        raw = {
            "integrations": {
                "mlflow": {
                    "integration": "prod-mlflow",
                    "experiment_name": "x",
                },
                "huggingface": {
                    "integration": "my-hf",
                    "repo_id": "user/repo",
                },
            },
        }
        out = resolve_yaml_integrations(raw, registry=registry)
        assert (
            out["integrations"]["mlflow"]["tracking_uri"]
            == "https://mlflow.example.com"
        )
        assert (
            out["integrations"]["huggingface"]["integration"]
            == "my-hf"
        )

    def test_resolver_does_not_validate_pipeline_schema(
        self, tmp_path: Path
    ) -> None:
        """Resolver is a pre-validation pass — even if downstream
        Pydantic validation would fail, the resolver succeeds."""
        _create_integration(
            tmp_path,
            integration_id="prod-mlflow",
            integration_type="mlflow",
            current_yaml=_MLFLOW_INTEGRATION_YAML,
        )
        registry = IntegrationRegistry(root=tmp_path)
        # ``experiment_name`` missing — Pydantic would reject
        # ``MLflowConfig`` later, but the resolver doesn't care.
        raw = {
            "integrations": {
                "mlflow": {"integration": "prod-mlflow"},
            },
        }
        out = resolve_yaml_integrations(raw, registry=registry)
        # Still returned a sensible inlined dict; Pydantic-level error
        # is the next layer's responsibility.
        assert (
            out["integrations"]["mlflow"]["tracking_uri"]
            == "https://mlflow.example.com"
        )
