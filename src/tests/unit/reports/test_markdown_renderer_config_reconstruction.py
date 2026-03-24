from __future__ import annotations

import logging
from datetime import datetime

import yaml

from src.reports.domain.entities import ExperimentData, RunStatus
from src.reports.models.report import (
    ConfigInfo,
    ExperimentHealth,
    ExperimentReport,
    ModelInfo,
    ReportSummary,
    ResourcesInfo,
)
from src.reports.plugins.builtins.config_dump import ConfigDumpBlockPlugin
from src.reports.plugins.interfaces import ReportPluginContext
from src.reports.plugins.markdown_block_renderer import MarkdownBlockRenderer


class _DummyProvider:
    def load(self, run_id: str):  # pragma: no cover
        raise NotImplementedError


def _extract_first_yaml_code_block(markdown: str) -> str:
    fence = "```yaml"
    start = markdown.index(fence)
    start = markdown.index("\n", start) + 1
    end = markdown.index("```", start)
    return markdown[start:end]


def _render_full_config_dict(params: dict[str, str]) -> dict:
    report = ExperimentReport(
        generated_at=datetime.now(),
        summary=ReportSummary(
            run_id="test_run",
            run_name="test_experiment",
            experiment_name="unit_test",
            status=RunStatus.FINISHED,
            health=ExperimentHealth.GREEN,
            health_explanation="All good",
            duration_total_seconds=1.0,
        ),
        model=ModelInfo(name="TestModel"),
        phases=[],
        resources=ResourcesInfo(),
        timeline=[],
        issues=[],
        config=ConfigInfo(source_config=None, params_config=params),
        memory_management=None,
        validation=None,
    )

    data = ExperimentData(
        run_id=report.summary.run_id,
        run_name=report.summary.run_name,
        experiment_name=report.summary.experiment_name,
        status=report.summary.status,
        start_time=datetime.now(),
        end_time=None,
        duration_seconds=0.0,
        phases=[],
    )

    plugin = ConfigDumpBlockPlugin()
    ctx = ReportPluginContext(
        run_id=data.run_id,
        data_provider=_DummyProvider(),
        data=data,
        report=report,
        logger=logging.getLogger(__name__),
    )
    block = plugin.render(ctx)
    markdown = MarkdownBlockRenderer().render([block])

    reconstructed_yaml = _extract_first_yaml_code_block(markdown)
    return yaml.safe_load(reconstructed_yaml) or {}


class TestMarkdownRendererConfigReconstruction:
    def test_reconstruct_includes_training_hyperparams_and_merges_into_strategies(self) -> None:
        """
        Regression: Full Configuration in report was missing training.hyperparams.* because
        renderer only reconstructed `config.*` keys.

        We now also include training.hyperparams.* and merge them into each strategy hyperparams
        (strategy overrides win).
        """
        params = {
            # config.* snapshot
            "config.model.name": "Qwen/Qwen2.5-0.5B-Instruct",
            "config.training.type": "qlora",
            "config.training.output_dir": "output",
            "config.lora.r": "16",
            "config.lora.alpha": "32",
            # global hyperparams snapshot (no config.* prefix)
            "training.hyperparams.per_device_train_batch_size": "1",
            "training.hyperparams.gradient_accumulation_steps": "8",
            "training.hyperparams.logging_steps": "10",
            "training.hyperparams.save_steps": "200",
            "training.hyperparams.bf16": "True",
            "training.hyperparams.fp16": "False",
            "training.hyperparams.max_length": "2048",
            # strategy 0
            "config.strategy.0.type": "sft",
            "config.strategy.0.dataset": "default",
            "config.strategy.0.hyperparams.epochs": "1",
            "config.strategy.0.hyperparams.learning_rate": "0.0002",
            "config.strategy.0.hyperparams.max_length": "1024",  # override global
            "config.strategy.0.hyperparams.packing": "False",
        }

        cfg = _render_full_config_dict(params)

        assert cfg["model"]["name"] == "Qwen/Qwen2.5-0.5B-Instruct"

        training = cfg["training"]
        assert training["type"] == "qlora"
        assert training["output_dir"] == "output"

        # lora moved under training.lora for display
        assert training["lora"]["r"] == 16
        assert training["lora"]["alpha"] == 32

        # training.hyperparams included
        assert training["hyperparams"]["per_device_train_batch_size"] == 1
        assert training["hyperparams"]["gradient_accumulation_steps"] == 8
        assert training["hyperparams"]["bf16"] is True
        assert training["hyperparams"]["fp16"] is False

        # strategies are moved under training.strategies and merged
        assert "strategy" not in cfg  # normalized shape
        assert len(training["strategies"]) == 1
        s0 = training["strategies"][0]
        assert s0["strategy_type"] == "sft"
        assert s0["dataset"] == "default"

        # merged hyperparams include globals + overrides
        assert s0["hyperparams"]["epochs"] == 1
        assert s0["hyperparams"]["learning_rate"] == 0.0002
        assert s0["hyperparams"]["per_device_train_batch_size"] == 1
        assert s0["hyperparams"]["gradient_accumulation_steps"] == 8
        assert s0["hyperparams"]["logging_steps"] == 10
        assert s0["hyperparams"]["save_steps"] == 200
        assert s0["hyperparams"]["bf16"] is True
        assert s0["hyperparams"]["fp16"] is False
        assert s0["hyperparams"]["max_length"] == 1024  # override wins

    def test_reconstruct_handles_strategy_chains_sorted_by_index(self) -> None:
        params = {
            "config.training.type": "qlora",
            "training.hyperparams.per_device_train_batch_size": "1",
            # out of order on purpose
            "config.strategy.1.type": "dpo",
            "config.strategy.1.hyperparams.epochs": "2",
            "config.strategy.0.type": "sft",
            "config.strategy.0.hyperparams.epochs": "1",
        }

        cfg = _render_full_config_dict(params)

        strategies = cfg["training"]["strategies"]
        assert len(strategies) == 2
        assert strategies[0]["strategy_type"] == "sft"
        assert strategies[0]["hyperparams"]["epochs"] == 1
        assert strategies[0]["hyperparams"]["per_device_train_batch_size"] == 1
        assert strategies[1]["strategy_type"] == "dpo"
        assert strategies[1]["hyperparams"]["epochs"] == 2
        assert strategies[1]["hyperparams"]["per_device_train_batch_size"] == 1


