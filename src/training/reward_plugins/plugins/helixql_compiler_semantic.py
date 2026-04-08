from __future__ import annotations

import os
import platform
import shutil
import stat
import subprocess
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any
from urllib.request import urlopen, Request
import re

from src.training.reward_plugins.base import RewardPlugin
from src.training.reward_plugins.registry import RewardPluginRegistry
from src.utils.domains.helixql import extract_query_text, extract_schema_block, semantic_match_details
from src.utils.domains.helixql_cli import HelixCompiler
from src.utils.logger import logger

_DEFAULT_TIMEOUT_SECONDS = 10
_BACKEND_COMPILE = "compile"
_BACKEND_SEMANTIC_ONLY = "semantic_only"
_SUPPORTED_BACKENDS = frozenset({_BACKEND_COMPILE, _BACKEND_SEMANTIC_ONLY})
_QUERY_PREFIX_RE = re.compile(r"^\s*QUERY\b", flags=re.IGNORECASE)
_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

_HELIX_GITHUB_REPO = "HelixDB/helix-db"
_HELIX_INSTALL_DIR = Path("/usr/local/bin")
_HELIX_FALLBACK_DIR = Path.home() / ".local" / "bin"


def _resolve_helix_asset_name() -> str:
    """Map current platform to the GitHub release asset name."""
    system = platform.system().lower()
    machine = platform.machine().lower()

    if system == "linux":
        if machine in ("x86_64", "amd64"):
            return "helix-x86_64-unknown-linux-gnu"
        if machine in ("aarch64", "arm64"):
            return "helix-aarch64-unknown-linux-gnu"
    elif system == "darwin":
        if machine in ("x86_64", "amd64"):
            return "helix-x86_64-apple-darwin"
        if machine in ("aarch64", "arm64"):
            return "helix-aarch64-apple-darwin"

    raise RuntimeError(f"No pre-built helix binary for {system}/{machine}")


def _install_helix_cli(*, version: str = "latest") -> Path:
    """Download helix CLI from GitHub releases and install it.

    Returns the path to the installed binary.
    """
    asset_name = _resolve_helix_asset_name()

    if version == "latest":
        download_url = (
            f"https://github.com/{_HELIX_GITHUB_REPO}/releases/latest/download/{asset_name}"
        )
    else:
        download_url = (
            f"https://github.com/{_HELIX_GITHUB_REPO}/releases/download/{version}/{asset_name}"
        )

    install_dir = _HELIX_INSTALL_DIR if os.access(_HELIX_INSTALL_DIR, os.W_OK) else _HELIX_FALLBACK_DIR
    install_dir.mkdir(parents=True, exist_ok=True)
    target = install_dir / "helix"

    logger.info("[HELIX_SETUP] Downloading %s → %s", download_url, target)
    req = Request(download_url, headers={"User-Agent": "RyotenkAI-RewardPlugin/1.0"})
    with urlopen(req, timeout=120) as resp:  # noqa: S310
        data = resp.read()

    target.write_bytes(data)
    target.chmod(target.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    try:
        result = subprocess.run(
            [str(target), "--version"], capture_output=True, text=True, timeout=10
        )
        ver = result.stdout.strip() or result.stderr.strip()
        logger.info("[HELIX_SETUP] Installed: %s", ver)
    except Exception as exc:
        logger.warning("[HELIX_SETUP] Installed binary but version check failed: %s", exc)

    return target


@RewardPluginRegistry.register
class HelixQLCompilerSemanticRewardPlugin(RewardPlugin):
    """Domain plugin for HelixQL GRPO/SAPO reward."""

    name = "helixql_compiler_semantic"

    def __init__(self, params: dict[str, Any]):
        self._compiler: HelixCompiler | None = None
        super().__init__(params)

    def _validate_params(self) -> None:
        backend = str(self.params.get("validation_backend", _BACKEND_COMPILE)).strip().lower()
        if backend not in _SUPPORTED_BACKENDS:
            raise ValueError(
                f"Unsupported validation_backend={backend!r}. Expected one of {sorted(_SUPPORTED_BACKENDS)}"
            )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def setup(self) -> None:
        if self._backend() != _BACKEND_COMPILE:
            return

        if shutil.which("helix") is not None:
            logger.info("[HELIX_SETUP] helix CLI already on PATH — skipping install")
            return

        logger.info("[HELIX_SETUP] helix CLI not found — installing from GitHub releases ...")
        installed_path = _install_helix_cli()

        if shutil.which("helix") is None:
            bin_dir = str(installed_path.parent)
            os.environ["PATH"] = f"{bin_dir}:{os.environ.get('PATH', '')}"
            logger.info("[HELIX_SETUP] Added %s to PATH", bin_dir)

        if shutil.which("helix") is None:
            raise RuntimeError(
                f"helix CLI installed to {installed_path} but still not found on PATH. "
                "Check permissions and PATH configuration."
            )

    def _backend(self) -> str:
        return str(self.params.get("validation_backend", _BACKEND_COMPILE)).strip().lower()

    def _get_compiler(self) -> HelixCompiler:
        if self._compiler is None:
            timeout = int(self.params.get("timeout_seconds", _DEFAULT_TIMEOUT_SECONDS))
            self._compiler = HelixCompiler(timeout_seconds=timeout)
        return self._compiler

    def _semantic_only_score(self, *, output: str, reference: str, prompt: str) -> float:
        """
        Softer reward for smoke RL when compile-time validation is unavailable.

        We still prefer the domain-aware score from `semantic_match_details`, but when it
        collapses to 0.0 (common for early random generations), we fall back to a lexical
        similarity score so GRPO/SAPO can see non-zero variation and actually update.
        """
        details = semantic_match_details(candidate=output, expected=reference, user_text=prompt)
        score = float(details["score"])
        if score > 0.0:
            return score

        output_text = (output or "").strip()
        reference_text = (reference or "").strip()
        if not output_text or not reference_text:
            return 0.0

        output_lower = output_text.lower()
        reference_lower = reference_text.lower()
        seq = SequenceMatcher(a=output_lower, b=reference_lower).ratio()
        output_tokens = set(_TOKEN_RE.findall(output_lower))
        reference_tokens = set(_TOKEN_RE.findall(reference_lower))
        union = output_tokens | reference_tokens
        jaccard = (len(output_tokens & reference_tokens) / len(union)) if union else 0.0
        prefix_bonus = 0.1 if _QUERY_PREFIX_RE.search(output_text) else 0.0

        fallback = (0.65 * seq) + (0.25 * jaccard) + prefix_bonus
        return round(max(0.0, min(1.0, fallback)), 4)

    def build_trainer_kwargs(
        self,
        *,
        train_dataset: Any,
        phase_config: Any,
        pipeline_config: Any,
    ) -> dict[str, Any]:
        del phase_config, pipeline_config
        features = getattr(train_dataset, "features", {}) or {}
        available_fields = set(features.keys()) if hasattr(features, "keys") else set()
        required = {"prompt", "reference_answer"}
        missing = sorted(required - available_fields)
        if missing:
            raise ValueError(
                "Reward plugin 'helixql_compiler_semantic' requires dataset fields "
                f"{sorted(required)}. Missing: {missing}"
            )

        backend = self._backend()
        compiler = self._get_compiler() if backend == _BACKEND_COMPILE else None

        def compiler_reward(completions: Any, **kwargs: Any) -> list[float]:
            outputs = [extract_query_text(item) for item in completions]
            prompts = _coerce_column(kwargs, "prompt", len(outputs))
            schemas = _coerce_column(kwargs, "schema_context", len(outputs))

            scores: list[float] = []
            for idx, output in enumerate(outputs):
                schema_text = schemas[idx] or extract_schema_block(prompts[idx])
                if not schema_text.strip() or not output.strip():
                    scores.append(-1.0)
                    continue
                if compiler is None:
                    scores.append(0.0)
                    continue
                result = compiler.validate(schema=schema_text, query=output)
                scores.append(1.0 if result.ok else -1.0)
            return scores

        def semantic_reward(completions: Any, **kwargs: Any) -> list[float]:
            outputs = [extract_query_text(item) for item in completions]
            prompts = _coerce_column(kwargs, "prompt", len(outputs))
            schemas = _coerce_column(kwargs, "schema_context", len(outputs))
            references = _coerce_column(kwargs, "reference_answer", len(outputs))

            scores: list[float] = []
            for idx, output in enumerate(outputs):
                if not output.strip():
                    scores.append(0.0)
                    continue
                if backend == _BACKEND_COMPILE:
                    schema_text = schemas[idx] or extract_schema_block(prompts[idx])
                    if not schema_text.strip():
                        scores.append(0.0)
                        continue
                    if compiler is None:
                        scores.append(0.0)
                        continue
                    result = compiler.validate(schema=schema_text, query=output)
                    if not result.ok:
                        scores.append(0.0)
                        continue
                if backend == _BACKEND_SEMANTIC_ONLY:
                    scores.append(
                        self._semantic_only_score(output=output, reference=references[idx], prompt=prompts[idx])
                    )
                    continue
                details = semantic_match_details(candidate=output, expected=references[idx], user_text=prompts[idx])
                scores.append(float(details["score"]))
            return scores

        compiler_reward.__name__ = "compiler_reward"
        semantic_reward.__name__ = "semantic_reward"
        reward_funcs = [compiler_reward, semantic_reward] if backend == _BACKEND_COMPILE else [semantic_reward]
        return {"reward_funcs": reward_funcs}

    def build_config_kwargs(
        self,
        *,
        train_dataset: Any,
        phase_config: Any,
        pipeline_config: Any,
    ) -> dict[str, Any]:
        del train_dataset, phase_config, pipeline_config
        backend = self._backend()
        reward_weights = [1.0, 1.0] if backend == _BACKEND_COMPILE else [1.0]
        return {"reward_weights": reward_weights}


def _coerce_column(kwargs: dict[str, Any], key: str, size: int) -> list[str]:
    value = kwargs.get(key)
    if value is None:
        return [""] * size
    if isinstance(value, list):
        coerced = [extract_query_text(item) for item in value]
        if len(coerced) < size:
            coerced.extend([""] * (size - len(coerced)))
        return coerced
    return [extract_query_text(value) for _ in range(size)]


__all__ = ["HelixQLCompilerSemanticRewardPlugin"]
