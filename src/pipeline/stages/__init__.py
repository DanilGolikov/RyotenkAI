"""Pipeline stages package — exposes ONLY light-weight contract names.

Stage classes (DatasetValidator, GPUDeployer, …) are deliberately NOT
re-exported from here. Importing them eagerly used to drag the full
training stack (data loaders → src.utils.container → torch / transformers
/ mlflow) into every caller that just wanted a status enum, slowing CLI
startup and making lazy-import patches fragile.

Layout:

    stages/
    ├── base.py                 # PipelineStage abstract base
    ├── constants.py            # StageNames, CANONICAL_STAGE_ORDER, PipelineContextKeys
    ├── gpu_deployer.py         # GPUDeployer + IEarlyReleasable
    ├── inference_deployer.py   # InferenceDeployer
    ├── model_evaluator.py      # ModelEvaluator
    ├── training_monitor.py     # TrainingMonitor
    ├── dataset_validator/      # DatasetValidator + ValidationArtifactManager
    ├── model_retriever/        # ModelRetriever + uploader / model card helpers
    └── managers/               # LogManager + TrainingDeploymentManager

Importing a stage class:

    # Lightweight — orchestrator path:
    from src.pipeline.stages.dataset_validator import DatasetValidator
    from src.pipeline.stages.gpu_deployer import GPUDeployer

    # Lightweight — anywhere that just needs the enum:
    from src.pipeline.stages import StageNames

The heavy-import guardrail in test_architectural_guardrails enforces
that ``from src.pipeline.stages import StageNames`` does NOT pull torch,
transformers, mlflow, datasets, or src.training.* into sys.modules.
"""

from src.pipeline.stages.constants import (
    CANONICAL_STAGE_ORDER,
    PipelineContextKeys,
    StageNames,
)

__all__ = ["CANONICAL_STAGE_ORDER", "PipelineContextKeys", "StageNames"]
