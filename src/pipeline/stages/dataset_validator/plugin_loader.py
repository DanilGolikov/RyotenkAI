"""Load and instantiate validation plugins for the DatasetValidator stage.

Two entry points:

* :meth:`PluginLoader.load_for_default_dataset` — uses
  ``config.get_primary_dataset()`` as the source of plugin
  configuration (legacy convenience for callers that don't yet
  iterate the strategy chain).
* :meth:`PluginLoader.load_for_dataset` — per-dataset config lookup
  with default-plugin fallback. The DatasetValidator orchestration
  uses this for every dataset it validates.

Plugin instances come from :attr:`src.data.validation.registry.
validation_registry`. DTST_* secrets are injected via
:class:`SecretsResolver` for plugins that declare them in their
manifest. Returns the canonical 4-tuple
``(plugin_id, plugin_name, plugin_instance, apply_to_set)``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.data.validation.registry import validation_registry
from src.pipeline.stages.dataset_validator.constants import (
    SPLIT_EVAL,
    SPLIT_TRAIN,
    VALIDATIONS_ATTR,
)
from src.utils.logger import logger

if TYPE_CHECKING:
    from src.config.secrets.model import Secrets
    from src.utils.config import PipelineConfig


PluginTuple = tuple[str, str, Any, set[str]]


class PluginLoader:
    """Resolve plugin configs into instantiated ``ValidationPlugin`` tuples."""

    def __init__(self, config: PipelineConfig, secrets: Secrets | None) -> None:
        self._config = config
        self._secrets = secrets

    def load_for_default_dataset(self) -> list[PluginTuple]:
        """Load plugins for ``config.get_primary_dataset()``.

        Priority:
        1. Explicit ``validations.plugins`` on the primary dataset.
        2. Default plugin set (sensible thresholds, used when no
           ``validations`` block is configured).
        """
        return self.load_for_dataset(self._config.get_primary_dataset())

    def load_for_dataset(self, dataset_config: Any) -> list[PluginTuple]:
        """Load plugins for one specific dataset config or fall back to defaults."""
        validations = getattr(dataset_config, VALIDATIONS_ATTR, None)
        if validations is not None and getattr(validations, "plugins", None):
            return self._load_configured_plugins(validations.plugins)

        logger.info("[VALIDATOR] No validation config, using default plugins")
        return self._get_default_plugins()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load_configured_plugins(self, plugin_configs: list[Any]) -> list[PluginTuple]:
        """Instantiate each configured plugin via the validation registry."""
        logger.info(f"[VALIDATOR] Loading {len(plugin_configs)} validation plugins")

        from src.community.catalog import catalog

        catalog.ensure_loaded()

        secrets_resolver = self._build_secrets_resolver()

        plugins: list[PluginTuple] = []
        for pc in plugin_configs:
            try:
                plugin_id = pc.id
                plugin_name = pc.plugin
                plugin_params = pc.params or {}
                plugin_thresholds = pc.thresholds or {}
                apply_to = set(pc.apply_to or [SPLIT_TRAIN, SPLIT_EVAL])

                # Secret injection is centralised in registry.instantiate(). Plugins
                # that don't declare ``_required_secrets`` simply ignore the resolver.
                plugin = validation_registry.instantiate(
                    plugin_name,
                    resolver=secrets_resolver,
                    params=plugin_params,
                    thresholds=plugin_thresholds,
                )

                plugins.append((plugin_id, plugin_name, plugin, apply_to))
                logger.debug(f"[VALIDATOR] Loaded plugin instance: {plugin_id} ({plugin_name})")

            except KeyError as e:
                available = validation_registry.list_ids()
                logger.error(f"[VALIDATOR] Failed to load plugin: {e}. Available: {available}")
                raise

        return plugins

    def _build_secrets_resolver(self):
        """Build a DTST_* SecretsResolver if secrets are available."""
        if self._secrets is None:
            return None
        from src.data.validation.secrets import SecretsResolver

        return SecretsResolver(self._secrets)

    def _get_default_plugins(self) -> list[PluginTuple]:
        """Sensible default plugin set when no ``validations`` block is present."""
        from src.community.catalog import catalog

        catalog.ensure_loaded()

        from src.utils.config import DatasetValidationPluginConfig

        default_configs = [
            DatasetValidationPluginConfig(id="min_samples", plugin="min_samples", thresholds={"threshold": 100}),
            DatasetValidationPluginConfig(id="avg_length", plugin="avg_length", thresholds={"min": 50, "max": 8192}),
            DatasetValidationPluginConfig(id="empty_ratio", plugin="empty_ratio", thresholds={"max_ratio": 0.05}),
            DatasetValidationPluginConfig(
                id="diversity_score",
                plugin="diversity_score",
                thresholds={"min_score": 0.3},
            ),
        ]
        return self._load_configured_plugins(default_configs)


__all__ = ["PluginLoader", "PluginTuple"]
