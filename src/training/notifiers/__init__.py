"""
Training Completion Notifiers.

Provides pluggable notification strategies for training completion:
- MarkerFileNotifier: Creates marker files (for RunPod integration)
- LogNotifier: Just logs completion (for local development)

Example:
    >>> from src.training.notifiers import MarkerFileNotifier
    >>> notifier = MarkerFileNotifier(base_path="/workspace")
    >>> notifier.notify_complete({"output_path": "/workspace/model"})
"""

from src.training.notifiers.log import LogNotifier
from src.training.notifiers.marker_file import MarkerFileNotifier

__all__ = [
    "LogNotifier",
    "MarkerFileNotifier",
]
