"""
SystemPromptLoader — single resolution point for the system prompt.

Supported sources (mutually exclusive; controlled by InferenceLLMConfig):
    - Local file: InferenceLLMConfig.system_prompt_path
    - MLflow Registry: InferenceLLMConfig.system_prompt_mlflow_name

Result is SystemPromptResult: prompt text + source metadata for auditing.
Consumers (ModelEvaluator, providers) only use the string result.text.

Source metadata is written by providers to inference_manifest.json as
manifest["llm"]["system_prompt_source"] for observability and debugging.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from src.utils.logger import logger

if TYPE_CHECKING:
    from src.config.inference.common import InferenceLLMConfig
    from src.config.integrations.mlflow import MLflowConfig
    from src.infrastructure.mlflow.gateway import IMLflowGateway


@dataclass
class SystemPromptResult:
    """
    Resolved system prompt with its origin metadata.

    Attributes:
        text:   The raw prompt string ready to pass to the LLM.
        source: Audit metadata describing where the prompt came from.
                For file source:  {"type": "file",   "path": "<path>"}
                For MLflow source: {"type": "mlflow", "name": "<name>", "version": "<n>"}
    """

    text: str
    source: dict[str, str] = field(default_factory=dict)


class SystemPromptLoader:
    """
    Loads a system prompt from the configured source.

    All methods are static — no instance state needed.
    Source selection logic is centralised here; callers only call load().
    """

    @staticmethod
    def load(
        llm_cfg: InferenceLLMConfig,
        mlflow_cfg: MLflowConfig | None = None,
        gateway: IMLflowGateway | None = None,
    ) -> SystemPromptResult | None:
        """
        Resolve the system prompt from the configured source.

        Priority (both fields are mutually exclusive by InferenceLLMConfig validator,
        but MLflow is checked first for defensive coding):
            1. MLflow Prompt Registry  — if system_prompt_mlflow_name is set
            2. Local file              — if system_prompt_path is set
            3. None                    — if neither is configured

        Args:
            llm_cfg:    InferenceLLMConfig carrying source configuration.
            mlflow_cfg: MLflow integration config. Used as fallback to build a gateway
                        when no gateway is provided. Required when mlflow source is used.
            gateway:    Pre-built IMLflowGateway. Takes precedence over mlflow_cfg
                        when loading from MLflow. Pass this from orchestrator/providers
                        that already have a gateway instance.

        Returns:
            SystemPromptResult with text and source metadata, or None if not configured.
        """
        if llm_cfg.system_prompt_mlflow_name:
            resolved_gateway = SystemPromptLoader._resolve_gateway(
                llm_cfg.system_prompt_mlflow_name,
                mlflow_cfg=mlflow_cfg,
                gateway=gateway,
            )
            return SystemPromptLoader._from_mlflow(llm_cfg.system_prompt_mlflow_name, resolved_gateway)
        if llm_cfg.system_prompt_path:
            return SystemPromptLoader._from_file(llm_cfg.system_prompt_path)
        return None

    @staticmethod
    def _resolve_gateway(
        _mlflow_name: str,
        *,
        mlflow_cfg: MLflowConfig | None,
        gateway: IMLflowGateway | None,
    ) -> IMLflowGateway:
        """
        Resolve the gateway to use for loading from MLflow.

        If a pre-built gateway is provided it is used directly.
        Otherwise a new MLflowGateway is constructed from mlflow_cfg.

        Raises:
            ValueError: if neither gateway nor a valid mlflow_cfg is available.
        """
        if gateway is not None:
            return gateway

        if mlflow_cfg is None:
            raise ValueError(
                "inference.llm.system_prompt_mlflow_name is configured but "
                "experiment_tracking.mlflow is not. "
                "Add an mlflow block under experiment_tracking in pipeline_config.yaml."
            )

        # Build a gateway from config (used by unit tests or callers without a pre-built gateway)
        from src.infrastructure.mlflow.gateway import MLflowGateway
        from src.infrastructure.mlflow.uri_resolver import resolve_mlflow_uris

        resolved_uris = resolve_mlflow_uris(mlflow_cfg, runtime_role="control_plane")
        return MLflowGateway(
            resolved_uris.effective_local_tracking_uri,
            ca_bundle_path=mlflow_cfg.ca_bundle_path,
        )

    @staticmethod
    def _from_file(path_str: str) -> SystemPromptResult | None:
        """
        Read system prompt text from a local file.

        Returns None (with a warning) if the file is missing or empty.
        Never raises — caller continues without a system prompt on any error.
        """
        from pathlib import Path

        path = Path(path_str).expanduser()
        if not path.exists():
            logger.warning(
                f"[SYSTEM_PROMPT] system_prompt_path configured but file not found: {path}. "
                "Continuing without system prompt."
            )
            return None

        try:
            content = path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            logger.warning(f"[SYSTEM_PROMPT] Failed to read system prompt file {path}: {exc}. Continuing without.")
            return None

        if not content:
            logger.warning(f"[SYSTEM_PROMPT] system_prompt_path file is empty: {path}. Continuing without.")
            return None

        logger.info(f"[SYSTEM_PROMPT] Loaded from file: {path} ({len(content)} chars)")
        return SystemPromptResult(
            text=content,
            source={"type": "file", "path": str(path)},
        )

    @staticmethod
    def _from_mlflow(name: str, gateway: IMLflowGateway) -> SystemPromptResult | None:
        """
        Load system prompt from MLflow Prompt Registry via gateway.

        The gateway applies an explicit timeout to the HTTP call so this
        method never blocks indefinitely, even when the MLflow server is
        unreachable.

        Supported name_or_uri formats (native MLflow):
            'my-prompt'                      → latest version
            'prompts:/my-prompt/3'           → specific version (immutable)
            'prompts:/my-prompt@production'  → alias (mutable)

        Returns None (with a warning) on any MLflow connectivity or lookup error.
        """
        prompt: Any = gateway.load_prompt(name)
        if prompt is None:
            # gateway already logged the warning
            return None

        template = getattr(prompt, "template", None)
        if not template:
            logger.warning(
                f"[SYSTEM_PROMPT] MLflow prompt {name!r} (v{prompt.version}) has an empty template. "
                "Continuing without system prompt."
            )
            return None

        text = str(template).strip()
        if not text:
            logger.warning(
                f"[SYSTEM_PROMPT] MLflow prompt {name!r} (v{prompt.version}) template is blank. "
                "Continuing without system prompt."
            )
            return None

        source: dict[str, str] = {
            "type": "mlflow",
            "name": str(prompt.name),
            "version": str(prompt.version),
        }
        logger.info(
            f"[SYSTEM_PROMPT] Loaded from MLflow: name={prompt.name!r} "
            f"version={prompt.version} uri={gateway.uri!r} ({len(text)} chars)"
        )
        return SystemPromptResult(text=text, source=source)


__all__ = [
    "SystemPromptLoader",
    "SystemPromptResult",
]
