"""Load and instantiate validation plugins for the DatasetValidator stage.

Single entry point :meth:`PluginLoader.load_for_dataset` resolves the
plugins declared on a dataset's ``validations.plugins`` block. If the
dataset has no ``validations`` block, or the block has an empty
``plugins`` list, **no plugins run**: validation for that dataset is
effectively a no-op (format check still runs, since it lives in
:class:`FormatChecker`).

There is intentionally no built-in default plugin set — running
hidden plugins behind the user's back makes pipeline behaviour
opaque and surprising. Validation must be explicit.

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

    def load_for_dataset(self, dataset_config: Any) -> list[PluginTuple]:
        """Load plugins for one dataset config.

        Returns an empty list if the dataset has no ``validations`` block
        or an empty ``plugins`` list — validation runs no plugins for
        that dataset (explicit behaviour, no hidden defaults).
        """
        validations = getattr(dataset_config, VALIDATIONS_ATTR, None)
        plugin_configs = getattr(validations, "plugins", None) if validations is not None else None
        if not plugin_configs:
            logger.info("[VALIDATOR] No validation plugins configured; skipping plugin checks")
            return []
        return self._load_configured_plugins(plugin_configs)

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


__all__ = ["PluginLoader", "PluginTuple"]
