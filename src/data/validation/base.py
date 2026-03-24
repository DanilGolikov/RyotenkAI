"""
Base classes for validation plugin system.

Defines:
- ValidationResult: Standardized result format
- ValidationPlugin: Abstract base class for all plugins
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

from src.utils.logger import logger
from src.utils.plugin_base import BasePlugin

if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset


@dataclass
class ValidationErrorGroup:
    """
    Structured sample-level validation errors grouped by error type.

    Attributes:
        error_type: Stable machine-readable error identifier
        sample_indices: Sample indices included in the current payload
        total_count: Total number of samples with this error
    """

    error_type: str
    sample_indices: list[int]
    total_count: int


@dataclass
class ValidationResult:
    """
    Standardized result from a validation plugin.

    Attributes:
        plugin_name: Name of the plugin that produced this result
        passed: Whether validation passed
        params: Runtime parameters used during plugin execution
        thresholds: Pass/fail criteria used by the plugin
        metrics: Dictionary of numeric metrics (actual validation results)
        warnings: List of warning messages
        errors: List of error messages
        error_groups: Structured grouped sample-level errors
        execution_time_ms: Time taken to execute validation
    """

    plugin_name: str
    passed: bool
    params: dict[str, Any]
    thresholds: dict[str, Any]
    metrics: dict[str, float]
    warnings: list[str]
    errors: list[str]
    execution_time_ms: float
    error_groups: list[ValidationErrorGroup] = field(default_factory=list)


class ValidationPlugin(BasePlugin, ABC):
    """
    Abstract base class for dataset validation plugins.

    Plugins validate specific aspects of datasets (e.g., sample count,
    diversity, format correctness). They work with both regular Dataset
    and IterableDataset (streaming) objects.

    Class Attributes:
        name: Unique plugin identifier
        priority: Execution priority (lower = earlier, default: 50)
        expensive: Whether plugin is computationally expensive
        required_fields: List of required dataset fields
        supports_streaming: Whether plugin supports IterableDataset

    Methods to implement:
        get_description(): Human-readable description of what plugin checks
        validate(): Perform validation and return result
        get_recommendations(): Generate recommendations on failure

    Example:
        @ValidationPluginRegistry.register
        class MinSamplesValidator(ValidationPlugin):
            name = "min_samples"

            @classmethod
            def get_description(cls) -> str:
                return "Checks minimum number of samples"

            def validate(self, dataset):
                threshold = self.thresholds.get("threshold", 100)
                count = len(dataset)
                passed = count >= threshold
                return ValidationResult(...)

            def get_recommendations(self, result):
                return ["Add more samples", "Use data augmentation"]
    """

    # name / priority / version — inherited from BasePlugin
    # Plugin-specific metadata (override in subclasses)
    expensive: ClassVar[bool] = False
    required_fields: ClassVar[list[str]] = []
    supports_streaming: ClassVar[bool] = True

    def __init__(self, params: dict[str, Any] | None = None, thresholds: dict[str, Any] | None = None):
        """
        Initialize plugin with configuration.

        Args:
            params: Plugin runtime and execution settings
            thresholds: Plugin pass/fail criteria
        """
        self.params = dict(params or {})
        self.thresholds = dict(thresholds or {})
        self._validate_contract()

    @classmethod
    @abstractmethod
    def get_description(cls) -> str:
        """Return a human-readable description of what this plugin checks."""
        ...

    @abstractmethod
    def validate(
        self,
        dataset: Dataset | IterableDataset,
    ) -> ValidationResult:
        """
        Perform validation on dataset.

        Args:
            dataset: HuggingFace Dataset or IterableDataset (streaming)
        Returns:
            ValidationResult with metrics, errors, and warnings
        """
        ...

    @abstractmethod
    def get_recommendations(self, result: ValidationResult) -> list[str]:
        """
        Generate actionable recommendations based on validation result.

        Called only when validation fails (result.passed == False).

        Args:
            result: ValidationResult from validate()

        Returns:
            List of recommendation strings
        """
        ...

    def _validate_contract(self) -> None:
        """
        Validate plugin configuration contract.

        Override in subclass to validate `params` and `thresholds`.
        Raise ValueError if configuration is invalid.
        """
        pass

    # =========================================================================
    # HELPER METHODS (for subclasses)
    # =========================================================================

    def _get_sample(
        self,
        dataset: Dataset | IterableDataset,
        sample_size: int,
    ) -> Dataset | list[dict[str, Any]]:
        """
        Get sample from dataset (works with both types).

        Args:
            dataset: Regular or streaming dataset
            sample_size: Number of samples to retrieve

        Returns:
            Dataset (for regular) or list of sample dictionaries (for streaming)
        """
        from datasets import IterableDataset

        if isinstance(dataset, IterableDataset):
            # Streaming dataset - use .take()
            logger.debug(f"[{self.name}] Sampling {sample_size} from IterableDataset")
            return list(dataset.take(sample_size))
        else:
            # Regular dataset - use .select()
            actual_size = min(sample_size, len(dataset))
            logger.debug(f"[{self.name}] Sampling {actual_size} from Dataset")
            return dataset.select(range(actual_size))

    @staticmethod
    def _is_large_dataset(dataset: Dataset | IterableDataset) -> bool:
        """
        Check if dataset is large (streaming or >100k samples).

        Args:
            dataset: Dataset to check

        Returns:
            True if dataset is large
        """
        from datasets import IterableDataset

        if isinstance(dataset, IterableDataset):
            return True  # Treat streaming as large
        else:
            from src.data.constants import VALIDATION_LARGE_DATASET_THRESHOLD

            return len(dataset) > VALIDATION_LARGE_DATASET_THRESHOLD

    @staticmethod
    def _extract_text(sample: Any) -> str:
        """
        Universal text extraction from sample.

        Handles different dataset formats:
        - text: Direct text field
        - messages: Conversational format
        - input/output: Instruction format

        Args:
            sample: Dataset sample

        Returns:
            Extracted text
        """
        if isinstance(sample, Mapping):
            if "text" in sample:
                return str(sample["text"])
            if "messages" in sample:
                messages = sample.get("messages")
                if isinstance(messages, list):
                    parts: list[str] = []
                    for msg in messages:
                        if isinstance(msg, Mapping):
                            parts.append(str(msg.get("content", "")))
                        else:
                            parts.append(str(msg))
                    return " ".join(parts)
                return str(messages)
            if "input" in sample and "output" in sample:
                return f"{sample.get('input', '')} {sample.get('output', '')}".strip()
        return str(sample)

    @staticmethod
    def _new_issue_lists(_tag: str | None = None) -> tuple[list[str], list[str]]:
        """Return fresh (errors, warnings) lists for validate() implementations."""
        return [], []

    @staticmethod
    def _safe_ratio(numerator: int, denominator: int, _tag: str | None = None) -> float:
        """Compute numerator/denominator with zero-division safety."""
        return numerator / denominator if denominator else 0.0

    def _get_samples_for_validation(
        self,
        dataset: Dataset | IterableDataset,
        *,
        default_sample_size: int = 10000,
    ) -> Dataset | list[dict[str, Any]]:
        """
        Return samples for validation.

        - For large/streaming datasets: returns a sampled subset (via `_get_sample`).
        - For small regular datasets: returns the dataset itself (lazy iteration).
        """
        if self._is_large_dataset(dataset):
            sample_size = int(self.params.get("sample_size", default_sample_size))
            samples = self._get_sample(dataset, sample_size)
            logger.debug(f"[{self.name}] Checking {len(samples)} samples")
            return samples

        return dataset  # pyright: ignore[reportReturnType]

    def _effective_config(self, *, include_description: bool = False) -> dict[str, Any]:
        config: dict[str, Any] = {
            "params": dict(self.params),
            "thresholds": dict(self.thresholds),
        }
        if include_description:
            config["description"] = self.get_description()
        return config

    def _param(self, key: str, default: Any = None) -> Any:
        return self.params.get(key, default)

    def _threshold(self, key: str, default: Any = None) -> Any:
        return self.thresholds.get(key, default)

    def _max_error_examples(self, default: int = 5) -> int:
        """
        Return configured limit for sample-level error examples.

        Semantics:
        - `-1` → keep all indices
        - `0`  → suppress sample-level indices
        - `N`  → keep first N indices per group
        """
        return int(self.params.get("max_error_examples", default))

    def _should_collect_error_example(self, collected_count: int, *, default: int = 5) -> bool:
        """
        Return True if another sample index should be kept for the current group.
        """
        max_error_examples = self._max_error_examples(default=default)
        if max_error_examples == -1:
            return True
        return collected_count < max_error_examples

    def _build_error_groups(
        self,
        indices_by_error: Mapping[str, list[int]],
        *,
        counts_by_error: Mapping[str, int] | None = None,
    ) -> list[ValidationErrorGroup]:
        """
        Build structured error groups from collected sample indices.
        """
        groups: list[ValidationErrorGroup] = []
        for error_type in sorted(indices_by_error):
            sample_indices = list(indices_by_error[error_type])
            if not sample_indices:
                continue
            total_count = (
                int(counts_by_error.get(error_type, len(sample_indices))) if counts_by_error else len(sample_indices)
            )
            groups.append(
                ValidationErrorGroup(
                    error_type=error_type,
                    sample_indices=sample_indices,
                    total_count=total_count,
                )
            )
        return groups

    @staticmethod
    def render_error_groups(error_groups: list[ValidationErrorGroup]) -> list[str]:
        """
        Render structured error groups into log-friendly strings.
        """
        lines: list[str] = []
        for group in error_groups:
            visible = ", ".join(str(idx) for idx in group.sample_indices)
            lines.append(f"{group.error_type}: [{visible}]")
        return lines

    @staticmethod
    def _check_max_ratio(
        *,
        ratio: float,
        max_ratio: float,
        errors: list[str],
        error_message: str,
        warnings: list[str] | None = None,
        near_message: str | None = None,
        near_fraction: float = 0.5,
    ) -> bool:
        """
        Validate that `ratio` does not exceed `max_ratio`.

        Optionally appends a "near threshold" warning when `ratio` is close to the limit.
        """
        passed = ratio <= max_ratio
        if not passed:
            errors.append(error_message)
            return False

        if warnings is not None and near_message and ratio > near_fraction * max_ratio:
            warnings.append(near_message)

        return True


__all__ = ["ValidationErrorGroup", "ValidationPlugin", "ValidationResult"]
