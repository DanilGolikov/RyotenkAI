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


_CARGO_INSTALL_TIMEOUT = 600  # 10 min for rustup + cargo build


def _helix_binary_works(binary_path: str) -> bool:
    """Return True if the helix binary at *binary_path* runs successfully."""
    try:
        proc = subprocess.run(
            [binary_path, "--version"], capture_output=True, text=True, timeout=10,
        )
        if proc.returncode == 0:
            logger.info("[HELIX_SETUP] Binary OK: %s", proc.stdout.strip())
            return True
        combined = (proc.stdout + proc.stderr).strip()
        logger.warning("[HELIX_SETUP] Binary failed (rc=%d): %s", proc.returncode, combined[:300])
    except Exception as exc:
        logger.warning("[HELIX_SETUP] Binary test error: %s", exc)
    return False


def _install_helix_via_cargo() -> Path:
    """Install helix CLI by compiling from source with cargo.

    Installs Rust toolchain via rustup if cargo is not available, then builds
    helix-cli from the GitHub repository.  Returns the path to the binary.
    """
    if not shutil.which("cargo"):
        logger.info("[HELIX_SETUP] Installing Rust toolchain via rustup ...")
        rustup_proc = subprocess.run(
            ["sh", "-c", "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y"],
            capture_output=True, text=True, timeout=120,
        )
        if rustup_proc.returncode != 0:
            raise RuntimeError(
                f"rustup install failed (rc={rustup_proc.returncode}): "
                f"{(rustup_proc.stdout + rustup_proc.stderr).strip()[:500]}"
            )
        cargo_home = Path.home() / ".cargo" / "bin"
        os.environ["PATH"] = f"{cargo_home}:{os.environ.get('PATH', '')}"
        logger.info("[HELIX_SETUP] Rust installed, cargo at %s", shutil.which("cargo") or cargo_home / "cargo")

    cargo_bin = shutil.which("cargo")
    if not cargo_bin:
        raise RuntimeError("cargo not found on PATH after rustup install")

    # Install build dependencies required by helix-cli (OpenSSL, pkg-config)
    if shutil.which("apt-get"):
        logger.info("[HELIX_SETUP] Installing build dependencies (pkg-config, libssl-dev) ...")
        deps_proc = subprocess.run(
            ["sh", "-c", "apt-get update -qq && apt-get install -y -qq pkg-config libssl-dev"],
            capture_output=True, text=True, timeout=120,
        )
        if deps_proc.returncode != 0:
            logger.warning("[HELIX_SETUP] apt-get deps install failed (non-fatal): %s", deps_proc.stderr.strip()[-300:])

    logger.info("[HELIX_SETUP] Building helix-cli from source (this may take several minutes) ...")
    build_proc = subprocess.run(
        [cargo_bin, "install", "--git", f"https://github.com/{_HELIX_GITHUB_REPO}", "helix-cli"],
        capture_output=True, text=True, timeout=_CARGO_INSTALL_TIMEOUT,
    )
    if build_proc.returncode != 0:
        raise RuntimeError(
            f"cargo install helix-cli failed (rc={build_proc.returncode}): "
            f"{(build_proc.stdout + build_proc.stderr).strip()[-500:]}"
        )

    helix_bin = shutil.which("helix")
    if helix_bin is None:
        cargo_home_bin = Path.home() / ".cargo" / "bin"
        candidate = cargo_home_bin / "helix"
        if candidate.exists():
            os.environ["PATH"] = f"{cargo_home_bin}:{os.environ.get('PATH', '')}"
            helix_bin = str(candidate)

    if helix_bin is None:
        raise RuntimeError("helix binary not found after cargo install")

    logger.info("[HELIX_SETUP] Built from source: %s", helix_bin)
    return Path(helix_bin)


def _install_helix_cli(*, version: str = "latest") -> Path:
    """Download helix CLI from GitHub releases and install it.

    Tries the pre-built binary first.  If it fails at runtime (e.g. GLIBC
    incompatibility), falls back to building from source via cargo.

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

    if _helix_binary_works(str(target)):
        return target

    logger.warning("[HELIX_SETUP] Pre-built binary unusable (likely GLIBC mismatch), falling back to cargo build ...")
    target.unlink(missing_ok=True)
    logger.info("[HELIX_SETUP] Removed broken pre-built binary at %s", target)
    return _install_helix_via_cargo()


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

        existing = shutil.which("helix")
        if existing is not None and _helix_binary_works(existing):
            logger.info("[HELIX_SETUP] helix CLI already on PATH and working — skipping install")
            return

        logger.info("[HELIX_SETUP] helix CLI not found or not working — installing ...")
        installed_path = _install_helix_cli()

        if shutil.which("helix") is None:
            bin_dir = str(installed_path.parent)
            os.environ["PATH"] = f"{bin_dir}:{os.environ.get('PATH', '')}"
            logger.info("[HELIX_SETUP] Added %s to PATH", bin_dir)

        final = shutil.which("helix")
        if final is None or not _helix_binary_works(final):
            raise RuntimeError(
                f"helix CLI installed to {installed_path} but does not work. "
                "Check GLIBC version, permissions, and PATH configuration."
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

        _debug_logged = [False]

        def semantic_reward(completions: Any, **kwargs: Any) -> list[float]:
            outputs = [extract_query_text(item) for item in completions]
            prompts = _coerce_column(kwargs, "prompt", len(outputs))
            references = _coerce_column(kwargs, "reference_answer", len(outputs))

            if not _debug_logged[0]:
                _debug_logged[0] = True
                logger.info(
                    "[REWARD_DEBUG] kwargs_keys=%s completions_type=%s n_outputs=%d "
                    "ref_sample=%r out_sample=%r prompt_sample=%r",
                    sorted(kwargs.keys()),
                    type(completions).__name__,
                    len(outputs),
                    references[0][:100] if references else "EMPTY",
                    outputs[0][:100] if outputs else "EMPTY",
                    prompts[0][:100] if prompts else "EMPTY",
                )

            scores: list[float] = []
            for idx, output in enumerate(outputs):
                if not output.strip():
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
