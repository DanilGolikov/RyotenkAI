"""Cross-cutting constants for the pipeline package.

Stage- and manager-specific values live next to their consumers (e.g.
``stages/managers/deployment_constants.py``,
``stages/model_retriever/constants.py``,
``stages/dataset_validator/constants.py``) so that touching one stage's
limits doesn't generate a diff that reads as "constants got bumped"
across unrelated subsystems.

Anything here must satisfy: read by ≥ 2 different stages OR by the
orchestrator/MLflow integration directly.
"""

from src.constants import CONSOLE_LINE_WIDTH

# Console / report rendering
SEPARATOR_CHAR = "="
SEPARATOR_LINE_WIDTH = CONSOLE_LINE_WIDTH
SUMMARY_LINE_WIDTH = 70

# MLflow categories / source labels — orchestrator + every category-tagged
# event consumer reads these.
MLFLOW_CATEGORY_PIPELINE = "pipeline"
MLFLOW_CATEGORY_VALIDATION = "validation"
MLFLOW_CATEGORY_INFERENCE = "inference"
MLFLOW_CATEGORY_EVALUATION = "evaluation"
MLFLOW_SOURCE_ORCHESTRATOR = "PipelineOrchestrator"

# Pipeline-context dict keys (read by orchestrator, summary reporter,
# context propagator).
CTX_PROVIDER_NAME_UNKNOWN = "unknown"
CTX_PROVIDER_TYPE_UNKNOWN = "unknown"
CTX_RUNTIME_SECONDS = "runtime_seconds"
CTX_TRAINING_INFO = "training_info"
CTX_TRAINING_DURATION = "training_duration_seconds"
CTX_UPLOAD_DURATION = "upload_duration_seconds"

# Time
SECONDS_PER_HOUR = 3600

# Exit codes
EXIT_CODE_SIGINT = 130  # 128 + 2 (SIGINT)
