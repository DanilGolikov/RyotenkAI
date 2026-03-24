from __future__ import annotations

from typing import Any

import yaml

from src.reports.core.constants import KEY_HYPERPARAMS
from src.reports.document.nodes import CodeBlock, DocBlock, Heading, HorizontalRule, Paragraph, emph, inlines, txt
from src.reports.plugins.interfaces import ReportBlock, ReportPluginContext
from src.reports.plugins.registry import ReportPluginRegistry

KEY_LORA = "lora"


@ReportPluginRegistry.register
class ConfigDumpBlockPlugin:
    plugin_id = "config_dump"
    title = "Configuration"
    order = 110

    def _convert_param_value(self, value: Any) -> Any:
        """Convert string MLflow param value to a Python type."""
        if not isinstance(value, str):
            return value

        v = value.strip()
        if v.lower() == "true":
            return True
        if v.lower() == "false":
            return False
        if v.lower() == "none":
            return None

        try:
            if "." in v:
                return float(v)
            return int(v)
        except ValueError:
            pass

        if "," in v:
            return [self._convert_param_value(x.strip()) for x in v.split(",")]

        return v

    def _reconstruct_config_from_params(self, params: dict[str, str]) -> dict[str, Any]:
        """
        Reconstruct nested config dict from flat MLflow params.

        Supported prefixes:
        - config.*                       (main config snapshot)
        - training.hyperparams.*         (global hyperparams snapshot)
        """
        config: dict[str, Any] = {}

        for key, value in params.items():
            if key.startswith("config."):
                path = key[7:].split(".")
            elif key.startswith(f"training.{KEY_HYPERPARAMS}."):
                suffix = key[len(f"training.{KEY_HYPERPARAMS}.") :]
                path = ["training", KEY_HYPERPARAMS, *suffix.split(".")]
            else:
                continue

            current: dict[str, Any] = config
            for part in path[:-1]:
                if part not in current or not isinstance(current[part], dict):
                    current[part] = {}
                current = current[part]

            current[path[-1]] = self._convert_param_value(value)

        # Normalize shape to match pipeline YAML semantics.
        training = config.get("training")
        if not isinstance(training, dict):
            training = {}
            config["training"] = training

        # Move lora.* under training.lora (display-only)
        if isinstance(config.get(KEY_LORA), dict) and KEY_LORA not in training:
            training[KEY_LORA] = config.pop(KEY_LORA)

        # Move top-level strategy.{i} -> training.strategies[{i}]
        strategies_raw = config.pop("strategy", None)
        strategies_list: list[dict[str, Any]] = []
        if isinstance(strategies_raw, dict):
            sortable: list[tuple[int | None, str, Any]] = []
            for k, v in strategies_raw.items():
                try:
                    idx = int(k)
                except (TypeError, ValueError):
                    idx = None
                sortable.append((idx, str(k), v))

            sortable.sort(key=lambda t: (t[0] is None, t[0] if t[0] is not None else t[1]))
            for _idx, _k, strat in sortable:
                if not isinstance(strat, dict):
                    continue
                if "type" in strat and "strategy_type" not in strat:
                    strat["strategy_type"] = strat.pop("type")
                strategies_list.append(strat)

        if strategies_list:
            training["strategies"] = strategies_list

        # Merge global training.hyperparams into each strategy hyperparams (strategy overrides win)
        training_hp = training.get(KEY_HYPERPARAMS)
        if isinstance(training_hp, dict) and isinstance(training.get("strategies"), list):
            base_hp = {k: v for k, v in training_hp.items() if not isinstance(v, dict)}
            for strat in training["strategies"]:
                if not isinstance(strat, dict):
                    continue
                strat_hp = strat.get(KEY_HYPERPARAMS)
                merged_hp: dict[str, Any] = dict(base_hp)
                if isinstance(strat_hp, dict):
                    merged_hp.update(strat_hp)
                strat[KEY_HYPERPARAMS] = merged_hp

        return config

    def render(self, ctx: ReportPluginContext) -> ReportBlock:
        config = ctx.report.config
        nodes: list[DocBlock] = [Heading(2, inlines(txt("🧾 Configuration")))]

        if config.source_config:
            nodes.append(Heading(3, inlines(txt("📄 Source Configuration (Original)"))))
            nodes.append(
                CodeBlock(
                    code=yaml.dump(config.source_config, default_flow_style=False, allow_unicode=True).rstrip(),
                    language="yaml",
                )
            )
            nodes.append(HorizontalRule())

        if config.params_config:
            reconstructed = self._reconstruct_config_from_params(config.params_config)
            if reconstructed:
                nodes.append(Heading(3, inlines(txt("📝 Full Configuration (from MLflow params)"))))
                nodes.append(
                    CodeBlock(
                        code=yaml.dump(reconstructed, default_flow_style=False, allow_unicode=True).rstrip(),
                        language="yaml",
                    )
                )
            else:
                nodes.append(Paragraph(inlines(emph("Could not reconstruct config from MLflow params."))))

        nodes.append(HorizontalRule())
        return ReportBlock(block_id=self.plugin_id, title=self.title, order=self.order, nodes=nodes)


__all__ = ["ConfigDumpBlockPlugin"]
