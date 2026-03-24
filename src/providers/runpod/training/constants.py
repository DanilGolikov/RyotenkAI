"""
RunPod provider constants.

We intentionally keep these values out of user configs to reduce misconfiguration
surface area for production pipelines.
"""

# Canonical RunPod GraphQL API base URL.
RUNPOD_API_BASE_URL = "https://api.runpod.io"

__all__ = [
    "RUNPOD_API_BASE_URL",
]
