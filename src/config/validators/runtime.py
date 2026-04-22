"""
Runtime validators.

Unlike the pure schema validators in cross.py / pipeline.py, the functions
in this module are allowed to read runtime objects (secrets, loaded plugins,
etc.) because the checks cannot be performed with the config schema alone.

Rules:
- Functions here MUST accept both a config object and a runtime object.
- Functions here MUST NOT perform I/O (disk, network).  Reading an already-
  loaded in-memory object (e.g. Secrets) is fine.
- Functions here are called from the orchestrator __init__, not from Pydantic
  model validators.
- Return convention: raise ValueError with an actionable message on failure,
  return None on success (same style as pipeline.py).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.config.pipeline.schema import PipelineConfig
    from src.config.secrets.model import Secrets


def validate_eval_plugin_secrets(cfg: PipelineConfig, secrets: Secrets) -> None:
    """
    Fail-fast: verify that every EVAL_* secret required by an enabled
    evaluation plugin is present in secrets.env before any stage runs.

    Motivation:
    - Plugin secrets are not part of the config schema (they live in
      secrets.env), so Pydantic validators cannot check them.
    - Failing here (orchestrator init) is far cheaper than discovering the
      missing secret after a multi-hour training run.

    Raises:
        ValueError: with the plugin id/name, missing keys, and a hint to add
            them to secrets.env.
    """
    eval_cfg = getattr(cfg, "evaluation", None)
    if not eval_cfg or not getattr(eval_cfg, "enabled", False):
        return

    from src.community.catalog import catalog
    from src.evaluation.plugins.registry import EvaluatorPluginRegistry
    from src.evaluation.plugins.secrets import SecretsResolver

    catalog.ensure_loaded()

    resolver = SecretsResolver(secrets)
    for plugin_cfg in eval_cfg.evaluators.plugins:
        if not plugin_cfg.enabled:
            continue

        plugin_name = plugin_cfg.plugin
        if not EvaluatorPluginRegistry.is_registered(plugin_name):
            continue

        plugin_cls = EvaluatorPluginRegistry.get(plugin_name)
        required_keys: tuple[str, ...] | None = getattr(plugin_cls, "_required_secrets", None)
        if not required_keys:
            continue

        try:
            resolver.resolve(required_keys)
        except RuntimeError as exc:
            missing = [k for k in required_keys if not (secrets.model_extra or {}).get(k.lower())]
            raise ValueError(
                f"Evaluation plugin instance '{plugin_cfg.id}' ({plugin_name}) is enabled "
                f"but required secret(s) are missing from secrets.env: {missing}. "
                f"Add them to secrets.env and restart. "
                f"Detail: {exc}"
            ) from exc


__all__ = [
    "validate_eval_plugin_secrets",
]
