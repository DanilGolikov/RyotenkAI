"""``make_pipeline_config`` — factory for a real :class:`PipelineConfig`.

Replaces ``MagicMock(spec=PipelineConfig)`` in tests that drive
:class:`PipelineOrchestrator` end-to-end. The orchestrator's bootstrap
unpacks **typed attributes** off the config (``training.strategies``,
``integrations.mlflow``, ``training.get_strategy_chain()``, …). A
``MagicMock`` answers every attribute access with another ``MagicMock``,
which works until the production code reads a *typed shape* — at which
point tests have to over-mock to match.

Building a real :class:`PipelineConfig` once, with sensible defaults,
sidesteps the entire problem:

* The discriminated unions are real (adapter, dataset source).
* Strategy validation runs (``[SFT]`` by default — single phase, valid).
* Cross-reference validators (``validate_pipeline_strategy_dataset_references``)
  pass because there's a populated ``datasets`` registry.
* ``_source_path`` is set so ``PipelineBootstrap.build()`` doesn't
  reject the config.

WHY a factory instead of a fixture: factories compose. A test can
build a non-default config (``make_pipeline_config(provider='runpod')``)
with one call instead of mutating a fixture out of band.

Per the Phase 4-followup plan (docs/migration/phase_4bc_log.md) this is
the third leg of the orchestrator-test mock-elimination strategy:

  * `mlflow_manager` kwarg — Phase 4B (in `orchestrator.py`)
  * `stages_override` kwarg — Phase 4-followup (in `orchestrator.py`
    + `pipeline_bootstrap.py`)
  * `make_pipeline_config()` — this module
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ryotenkai_shared.config import (
    DatasetConfig,
    DatasetSourceLocal,
    GlobalHyperparametersConfig,
    IntegrationsConfig,
    MLflowConfig,
    ModelConfig,
    PipelineConfig,
    QLoRAConfig,
    TrainingOnlyConfig,
)
from ryotenkai_shared.config.datasets import DatasetLocalPaths


def _default_model() -> ModelConfig:
    return ModelConfig(
        name="test/model",
        torch_dtype="bfloat16",
        trust_remote_code=False,
    )


def _default_adapter() -> QLoRAConfig:
    return QLoRAConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules="all-linear",
        use_dora=False,
        use_rslora=False,
        init_lora_weights="gaussian",
    )


def _default_hyperparams() -> GlobalHyperparametersConfig:
    return GlobalHyperparametersConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        warmup_ratio=0.0,
        epochs=1,
    )


def _default_training() -> TrainingOnlyConfig:
    return TrainingOnlyConfig(
        adapter=_default_adapter(),
        hyperparams=_default_hyperparams(),
    )


def _default_dataset() -> DatasetConfig:
    return DatasetConfig(
        source=DatasetSourceLocal(
            local_paths=DatasetLocalPaths(train="data/train.jsonl", eval=None),
        ),
    )


def _default_integrations() -> IntegrationsConfig:
    """MLflow enabled by default so ``_setup_mlflow`` has a config to
    bootstrap; tests that want it disabled pass ``integrations=None``
    or override ``integrations=IntegrationsConfig()``."""
    return IntegrationsConfig(
        mlflow=MLflowConfig(
            tracking_uri="http://localhost:5002",
            experiment_name="test-exp",
        ),
    )


def make_pipeline_config(
    *,
    source_path: Path | None = None,
    model: ModelConfig | None = None,
    training: TrainingOnlyConfig | None = None,
    providers: dict[str, Any] | None = None,
    datasets: dict[str, DatasetConfig] | None = None,
    integrations: IntegrationsConfig | None = None,
    **extra: Any,
) -> PipelineConfig:
    """Build a real :class:`PipelineConfig` with sensible test defaults.

    All kwargs are optional — pass only what your test needs to override.
    The default produced config is the simplest valid one that passes every
    cross-block validator:

      * single ``[SFT]`` strategy with no dataset reference;
      * one ``"default"`` local-path dataset (``data/train.jsonl``);
      * QLoRA adapter, batch_size=1, LR=2e-4, 1 epoch;
      * MLflow enabled (tests that don't care can ignore it).

    Args:
        source_path: When supplied, stamped onto ``_source_path`` so
            :meth:`PipelineBootstrap.build` does not raise the
            "no _source_path" error. Defaults to ``Path("config.yaml")``
            (a sentinel — the path itself is rarely read by tests).
        model: Override the default :class:`ModelConfig`.
        training: Override the default :class:`TrainingOnlyConfig`.
        providers: Override the providers registry. Default ``{}``.
        datasets: Override the datasets registry. Default
            ``{"default": <local-path-dataset>}``.
        integrations: Override the integrations block. Default has
            MLflow enabled.
        **extra: Forwarded as additional kwargs to :class:`PipelineConfig`
            for fields like ``inference``, ``evaluation``, ``reports``
            (those have working defaults so they're rarely needed).

    Returns:
        A real, validated :class:`PipelineConfig` with ``_source_path`` set.
    """
    cfg = PipelineConfig(
        model=model or _default_model(),
        training=training or _default_training(),
        providers=providers if providers is not None else {},
        datasets=datasets if datasets is not None else {"default": _default_dataset()},
        integrations=integrations if integrations is not None else _default_integrations(),
        **extra,
    )
    # ``_source_path`` is a PrivateAttr — use object.__setattr__ to bypass
    # Pydantic's frozen-attribute guard (Pydantic v2 PrivateAttr).
    path = source_path if source_path is not None else Path("config.yaml")
    object.__setattr__(cfg, "_source_path", path)
    return cfg


__all__ = ["make_pipeline_config"]
