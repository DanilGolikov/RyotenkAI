"""Dataset preview utilities — paginated reads for UI / API, never the
full pipeline load path."""

from src.data.preview.loader import DatasetPreviewLoader, PreviewPage

__all__ = ["DatasetPreviewLoader", "PreviewPage"]
