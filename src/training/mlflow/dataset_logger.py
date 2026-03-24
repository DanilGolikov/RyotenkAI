"""
MLflowDatasetLogger — dataset logging and tracking in MLflow.

Responsibilities (Single Responsibility):
  - Log datasets (pandas DataFrame, HuggingFace Dataset, dict)
  - Log dataset files (JSONL, CSV, Parquet)
  - Log dataset metadata without loading full data
  - Create MLflow Dataset objects for experiment→dataset→run linking
  - Link dataset inputs to runs

Depends on:
  - mlflow module (for data.from_pandas, log_input, etc.)
  - IMLflowPrimitives (for log_params, set_tags, log_dict in log_dataset_info)
  - Active run state is tracked via a has_active_run callable
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.training.constants import MLFLOW_CONTEXT_TRAINING
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

    from src.training.mlflow.primitives import IMLflowPrimitives

logger = get_logger(__name__)


class MLflowDatasetLogger:
    """
    Dataset logging for MLflow experiment tracking.

    Args:
        mlflow_module: The imported mlflow module
        primitives: Logging primitives (log_params, set_tags, log_dict)
        has_active_run: Callable returning True if an active run exists
    """

    def __init__(
        self,
        mlflow_module: Any,
        primitives: IMLflowPrimitives,
        has_active_run: Callable[[], bool],
    ) -> None:
        self._mlflow = mlflow_module
        self._primitives = primitives
        self._has_active_run = has_active_run

    def log_dataset(
        self,
        data: Any,
        name: str,
        source: str | None = None,
        context: str = MLFLOW_CONTEXT_TRAINING,
        targets: str | None = None,
        predictions: str | None = None,
    ) -> bool:
        """
        Log dataset with versioning to current run.

        Args:
            data: Dataset (pandas DataFrame, HuggingFace Dataset, or dict)
            name: Dataset name for identification
            source: Optional source path/URL
            context: Context tag ("training", "validation", "test")
            targets: Column name for targets (optional)
            predictions: Column name for predictions (optional)

        Returns:
            True if logged successfully
        """
        if self._mlflow is None or not self._has_active_run():
            return False

        try:
            mlflow_dataset = None

            if hasattr(data, "to_dict") and hasattr(data, "columns"):
                mlflow_dataset = self._mlflow.data.from_pandas(
                    data,
                    source=source,
                    name=name,
                    targets=targets,
                    predictions=predictions,
                )
            elif hasattr(data, "to_pandas"):
                df = data.to_pandas()
                mlflow_dataset = self._mlflow.data.from_pandas(
                    df,
                    source=source,
                    name=name,
                    targets=targets,
                    predictions=predictions,
                )
            elif isinstance(data, dict):
                import pandas as pd_module

                pd: Any = pd_module
                df = pd.DataFrame(data)
                mlflow_dataset = self._mlflow.data.from_pandas(
                    df,
                    source=source,
                    name=name,
                    targets=targets,
                    predictions=predictions,
                )

            if mlflow_dataset is None:
                logger.warning(f"[MLFLOW:DATASET] Unsupported data type: {type(data)}")
                return False

            self._mlflow.log_input(mlflow_dataset, context=context)
            logger.info(f"[MLFLOW:DATASET] Logged: {name} ({context})")
            return True

        except Exception as e:
            logger.warning(f"[MLFLOW:DATASET] Failed to log dataset: {e}")
            return False

    def log_dataset_from_file(
        self,
        file_path: str,
        name: str | None = None,
        context: str = MLFLOW_CONTEXT_TRAINING,
    ) -> bool:
        """
        Log dataset from file (JSONL, CSV, Parquet).

        Args:
            file_path: Path to dataset file
            name: Dataset name (default: filename without extension)
            context: Context tag

        Returns:
            True if logged successfully
        """
        if self._mlflow is None or not self._has_active_run():
            return False

        path = Path(file_path)
        if not path.exists():
            logger.warning(f"[MLFLOW:DATASET] File not found: {file_path}")
            return False

        dataset_name = name or path.stem

        try:
            import pandas as pd_module

            pd: Any = pd_module

            suffix = path.suffix.lower()
            if suffix in (".jsonl", ".json"):
                df = pd.read_json(path, lines=(suffix == ".jsonl"))
            elif suffix == ".csv":
                df = pd.read_csv(path)
            elif suffix == ".parquet":
                df = pd.read_parquet(path)
            else:
                logger.warning(f"[MLFLOW:DATASET] Unsupported file format: {suffix}")
                return False

            return self.log_dataset(
                data=df,
                name=dataset_name,
                source=str(path.absolute()),
                context=context,
            )

        except Exception as e:
            logger.warning(f"[MLFLOW:DATASET] Failed to load from {file_path}: {e}")
            return False

    def log_dataset_info(
        self,
        name: str,
        path: str | None = None,
        source: str | None = None,
        version: str | None = None,
        num_rows: int = 0,
        num_samples: int | None = None,
        num_features: int | None = None,
        context: str = MLFLOW_CONTEXT_TRAINING,
        extra_info: dict[str, Any] | None = None,
        extra_tags: dict[str, str] | None = None,
    ) -> None:
        """
        Log dataset metadata without loading full data.

        Args:
            name: Dataset name
            path: Path to dataset directory/file
            source: Source type (local, huggingface, etc.)
            version: Dataset version string
            num_rows: Number of samples/rows
            num_samples: Deprecated alias for num_rows
            num_features: Number of features/columns
            context: Context tag for parameter naming
            extra_info: Additional metadata dict (saved as JSON artifact)
            extra_tags: Additional tags to set on the run
        """
        samples = num_rows if num_rows > 0 else (num_samples or 0)

        params: dict[str, Any] = {
            f"dataset_{context}_name": name,
            f"dataset_{context}_samples": samples,
        }
        if path:
            params[f"dataset_{context}_path"] = path
        if source:
            params[f"dataset_{context}_source"] = source
        if version:
            params[f"dataset_{context}_version"] = version
        if num_features:
            params[f"dataset_{context}_features"] = num_features

        self._primitives.log_params(params)

        if extra_tags:
            self._primitives.set_tags(extra_tags)

        if extra_info:
            self._primitives.log_dict(extra_info, f"dataset_{context}_info.json")

        logger.debug(f"[MLFLOW:DATASET_INFO] {name}: {samples} samples, version={version}")

    def create_mlflow_dataset(
        self,
        data: Any,
        name: str,
        source: str,
        targets: str | None = None,
    ) -> Any:
        """
        Create MLflow Dataset object from data.

        Args:
            data: HuggingFace Dataset or pandas DataFrame
            name: Dataset name for MLflow
            source: Source URI
            targets: Target column name (optional)

        Returns:
            MLflow Dataset object or None
        """
        if self._mlflow is None:
            return None

        try:
            import pandas as pd_module

            pd: Any = pd_module

            if hasattr(data, "to_pandas"):
                df = data.to_pandas()
            elif isinstance(data, pd.DataFrame):
                df = data
            else:
                df = pd.DataFrame(data)

            mlflow_dataset = self._mlflow.data.from_pandas(
                df,
                source=source,
                name=name,
                targets=targets,
            )

            logger.debug(f"[MLFLOW:DATASET_CREATED] name={name}, source={source}, rows={len(df)}")
            return mlflow_dataset

        except Exception as e:
            logger.warning(f"[MLFLOW:DATASET_CREATE_FAILED] {e}")
            return None

    def log_dataset_input(
        self,
        dataset: Any,
        context: str = MLFLOW_CONTEXT_TRAINING,
    ) -> bool:
        """
        Link dataset to current MLflow run.

        Args:
            dataset: MLflow Dataset object (from create_mlflow_dataset)
            context: Context label

        Returns:
            True if logged successfully
        """
        if self._mlflow is None or not self._has_active_run() or dataset is None:
            return False

        try:
            self._mlflow.log_input(dataset, context=context)
            logger.debug(f"[MLFLOW:DATASET_INPUT] context={context}")
            return True
        except Exception as e:
            logger.warning(f"[MLFLOW:DATASET_INPUT_FAILED] {e}")
            return False


__all__ = ["MLflowDatasetLogger"]
