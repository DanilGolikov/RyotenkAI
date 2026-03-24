"""
Deduplication validation plugin (SFT-specific).

Checks for duplicate examples in dataset.
"""

from __future__ import annotations

import hashlib
import time
from typing import TYPE_CHECKING

from src.data.validation.base import ValidationPlugin, ValidationResult
from src.data.validation.registry import ValidationPluginRegistry
from src.utils.logger import logger

if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset

_PLUGIN_NAME = "deduplication"
_NEAR_WARNING_THRESHOLD = 0.3


@ValidationPluginRegistry.register
class DeduplicationValidator(ValidationPlugin):
    """
    Validates dataset for duplicate examples.

    Computes hash of each example and detects duplicates.

    Params:
        max_duplicate_ratio (float): Maximum allowed duplicate ratio (default: 0.1 = 10%)
        sample_size (int): For large/streaming datasets (default: 10000)
        hash_fields (list): Fields to use for hashing (default: auto-detect)

    Recommendations on failure:
        - Remove duplicate examples
        - Check data collection pipeline
        - Use deduplication tools
    """

    name = "deduplication"
    priority = 40
    expensive = True  # Hashing is expensive
    supports_streaming = True

    @classmethod
    def get_description(cls) -> str:
        return "Checks for duplicate examples in the dataset (SFT)"

    def validate(
        self,
        dataset: Dataset | IterableDataset,
    ) -> ValidationResult:
        """Check for duplicate examples."""
        start_time = time.time()
        max_ratio = self._threshold("max_duplicate_ratio", 0.1)
        errors, warnings = self._new_issue_lists(_PLUGIN_NAME)

        # Get samples
        samples = self._get_samples_for_validation(dataset)

        # Compute hashes
        seen_hashes: dict[str, int] = {}
        duplicate_count = 0

        for i, sample in enumerate(samples):
            text = self._extract_text(sample)
            # Use MD5 for fast hashing
            text_hash = hashlib.md5(memoryview(text.encode())).hexdigest()

            if text_hash in seen_hashes:
                duplicate_count += 1
                if duplicate_count <= 5:  # Log first few duplicates
                    orig_idx = seen_hashes[text_hash]
                    logger.debug(f"  Duplicate found: sample {i} matches sample {orig_idx}")
            else:
                seen_hashes[text_hash] = i

        duplicate_ratio = self._safe_ratio(duplicate_count, len(samples), _PLUGIN_NAME)
        passed = self._check_max_ratio(
            ratio=duplicate_ratio,
            max_ratio=max_ratio,
            errors=errors,
            warnings=warnings,
            error_message=f"Too many duplicates: {duplicate_ratio:.2%} > {max_ratio:.2%}",
            near_message=f"Duplicate ratio is close to the threshold: {duplicate_ratio:.2%} (max: {max_ratio:.2%})",
        )

        execution_time = (time.time() - start_time) * 1000

        return ValidationResult(
            plugin_name=self.name,
            passed=passed,
            params=dict(self.params),
            thresholds={
                "max_duplicate_ratio": max_ratio,
            },
            metrics={
                "duplicate_ratio": round(duplicate_ratio, 4),
                "duplicate_count": float(duplicate_count),
                "unique_count": float(len(seen_hashes)),
                "total_checked": float(len(samples)),
            },
            warnings=warnings,
            errors=errors,
            execution_time_ms=execution_time,
        )

    def get_recommendations(self, result: ValidationResult) -> list[str]:
        """Generate recommendations on failure."""
        dup_ratio = result.metrics["duplicate_ratio"]
        dup_count = int(result.metrics["duplicate_count"])
        total = int(result.metrics["total_checked"])
        max_threshold = result.thresholds["max_duplicate_ratio"]

        recommendations = [
            f"Found {dup_count} duplicates out of {total} ({dup_ratio:.2%})",
            f"Threshold: {max_threshold:.2%}",
            "Recommendations:",
            "  1. Remove duplicate examples from the dataset",
            "  2. Check the data collection pipeline for repeats",
            "  3. Use deduplication tools (e.g. MinHash)",
            "  4. Check for circular data imports",
        ]

        if dup_ratio > _NEAR_WARNING_THRESHOLD:
            recommendations.append("  ⚠️ CRITICAL: >30% duplicates! This seriously hurts training quality.")

        return recommendations


__all__ = ["DeduplicationValidator"]
