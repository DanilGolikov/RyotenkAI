"""
Pipeline stages for the RyotenkAI automated training system.

Each stage implements a specific part of the training workflow.
Stages are executed in order by PipelineOrchestrator.

Structure:
    stages/
    ├── base.py                 # PipelineStage base class
    ├── constants.py            # StageNames enum (no magic strings)
    ├── dataset_validator.py    # DatasetValidator stage
    ├── gpu_deployer.py         # GPUDeployer stage
    ├── inference_deployer.py   # InferenceDeployer stage (optional)
    ├── model_evaluator.py      # ModelEvaluator stage (optional)
    ├── model_retriever.py      # ModelRetriever stage
    ├── training_monitor.py     # TrainingMonitor stage
    └── managers/               # Helper classes (not stages)
        ├── deployment_manager.py
        └── log_manager.py

Usage:
    from src.pipeline.stages import StageNames
    from src.pipeline.stages import DatasetValidator, GPUDeployer, ...

    # Access context with typed keys (no magic strings!)
    deployer_ctx = context.get(StageNames.GPU_DEPLOYER, {})
"""

# Constants and types
from .constants import CANONICAL_STAGE_ORDER, PipelineContextKeys, StageNames

# Stages
from .dataset_validator import DatasetValidator
from .gpu_deployer import GPUDeployer
from .inference_deployer import InferenceDeployer

# Re-export managers for backward compatibility
from .managers import RunPodLogManager, TrainingDeploymentManager
from .model_evaluator import ModelEvaluator
from .model_retriever import ModelRetriever
from .training_monitor import TrainingMonitor

__all__ = [
    "CANONICAL_STAGE_ORDER",
    "DatasetValidator",
    "GPUDeployer",
    "InferenceDeployer",
    "ModelEvaluator",
    "ModelRetriever",
    "PipelineContextKeys",
    "RunPodLogManager",
    "StageNames",
    "TrainingDeploymentManager",
    "TrainingMonitor",
]
