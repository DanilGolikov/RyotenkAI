"""Custom exceptions for integration resolution.

Surface targeted, actionable errors when an integration referenced by a
project YAML cannot be resolved against the Settings → Integrations
registry. The CLI top-level handler converts these into clean,
human-readable messages instead of raw Python tracebacks.
"""

from __future__ import annotations


class IntegrationResolverError(Exception):
    """Base class for resolver failures.

    Caught at the CLI top-level handler to render a clean error message.
    """


class IntegrationNotFoundError(IntegrationResolverError):
    """The integration id referenced by a project ref is not in the registry.

    Example trigger: project YAML declares
    ``experiment_tracking.mlflow.integration: "helixql-mlflow"`` but the
    user never registered ``helixql-mlflow`` in Settings → Integrations.
    """

    def __init__(self, integration_id: str, integration_type: str) -> None:
        self.integration_id = integration_id
        self.integration_type = integration_type
        super().__init__(
            f"integration {integration_id!r} (type={integration_type}) is "
            f"referenced by the project YAML but not registered in "
            f"Settings → Integrations. Add it via the Web UI or CLI, "
            f"then retry."
        )


class IntegrationUnresolvedError(IntegrationResolverError):
    """The integration exists in the registry but cannot be resolved.

    Common causes:
    - ``current.yaml`` is empty (integration was created but never configured).
    - ``current.yaml`` fails schema validation (malformed URL, missing
      required field, …).
    - Type mismatch: project ref expects ``mlflow``-type integration but
      registry entry has type ``huggingface``.
    """

    def __init__(self, integration_id: str, reason: str) -> None:
        self.integration_id = integration_id
        self.reason = reason
        super().__init__(
            f"integration {integration_id!r} cannot be resolved: {reason}"
        )


__all__ = [
    "IntegrationNotFoundError",
    "IntegrationResolverError",
    "IntegrationUnresolvedError",
]
