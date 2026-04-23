"""Cerebras LLM-as-judge plugin — package entry point.

The community loader reads ``entry_point.class = "CerebrasJudgePlugin"``
from ``manifest.toml`` and pulls it from this package namespace.
"""

from .interface import IJudgeProvider, JudgeResponse
from .main import CerebrasJudgePlugin
from .provider import CerebrasProvider

__all__ = [
    "CerebrasJudgePlugin",
    "CerebrasProvider",
    "IJudgeProvider",
    "JudgeResponse",
]
