"""
MLflowModelRegistry — MLflow Model Registry operations.

Responsibilities (Single Responsibility):
  - Register models to MLflow Model Registry
  - Manage model aliases (set, delete, get, promote)
  - Load models by alias
  - Query model metadata

Does not depend on run state (_run, _parent_run_id, etc.).
`register_model` accepts run_id as an explicit parameter.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.training.constants import MLFLOW_KEY_TAGS
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.infrastructure.mlflow.gateway import IMLflowGateway

logger = get_logger(__name__)


class MLflowModelRegistry:
    """
    MLflow Model Registry — register, alias, and load models.

    Args:
        gateway: MLflow gateway for API access
        mlflow_module: The imported mlflow module
        log_model_enabled: Whether model registration is enabled in config
    """

    def __init__(
        self,
        gateway: IMLflowGateway,
        mlflow_module: Any,
        log_model_enabled: bool = True,
    ) -> None:
        self._gateway = gateway
        self._mlflow = mlflow_module
        self._log_model_enabled = log_model_enabled

    # =========================================================================
    # REGISTRATION
    # =========================================================================

    def register_model(
        self,
        model_name: str,
        run_id: str,
        model_uri: str | None = None,
        alias: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> str | None:
        """
        Register model to MLflow Model Registry.

        Args:
            model_name: Name for the registered model
            run_id: Run ID to build default model URI from
            model_uri: URI to model artifacts (default: runs:/{run_id}/model)
            alias: Optional alias (e.g., "champion", "staging")
            tags: Optional tags for the model version

        Returns:
            Model version number or None if failed
        """
        if self._mlflow is None or not run_id:
            return None

        if not self._log_model_enabled:
            logger.debug("[MLFLOW:REGISTRY] Model registry disabled (log_model=false)")
            return None

        try:
            if model_uri is None:
                model_uri = f"runs:/{run_id}/model"

            result = self._mlflow.register_model(model_uri, model_name)
            version = result.version
            logger.info(f"[MLFLOW:REGISTRY] Model registered: {model_name} v{version}")

            if alias:
                client = self._gateway.get_client()
                client.set_registered_model_alias(model_name, alias, version)
                logger.info(f"[MLFLOW:REGISTRY] Alias set: {model_name}@{alias} -> v{version}")

            if tags:
                client = self._gateway.get_client()
                for key, value in tags.items():
                    client.set_model_version_tag(model_name, version, key, value)

            return version

        except Exception as e:
            logger.warning(f"[MLFLOW:REGISTRY] Model registration failed: {e}")
            return None

    # =========================================================================
    # ALIAS MANAGEMENT
    # =========================================================================

    def set_model_alias(
        self,
        model_name: str,
        alias: str,
        version: int | str,
    ) -> bool:
        """
        Set alias for a registered model version.

        Args:
            model_name: Name of the registered model
            alias: Alias name (e.g., "champion", "staging")
            version: Model version number

        Returns:
            True if successful
        """
        if self._mlflow is None:
            return False

        try:
            client = self._gateway.get_client()
            client.set_registered_model_alias(model_name, alias, str(version))
            logger.info(f"[MLFLOW:REGISTRY] Alias set: {model_name}@{alias} -> v{version}")
            return True
        except Exception as e:
            logger.warning(f"[MLFLOW:REGISTRY] Failed to set alias: {e}")
            return False

    def get_model_by_alias(
        self,
        model_name: str,
        alias: str,
    ) -> dict[str, Any] | None:
        """
        Get model version info by alias.

        Args:
            model_name: Name of the registered model
            alias: Alias name (e.g., "champion")

        Returns:
            Dict with version info or None if not found
        """
        if self._mlflow is None:
            return None

        try:
            client = self._gateway.get_client()
            mv = client.get_model_version_by_alias(model_name, alias)
            return {
                "version": mv.version,
                "run_id": mv.run_id,
                "source": mv.source,
                "status": mv.status,
                "creation_timestamp": mv.creation_timestamp,
                "last_updated_timestamp": mv.last_updated_timestamp,
                "description": mv.description,
                MLFLOW_KEY_TAGS: mv.tags,
            }
        except Exception as e:
            logger.debug(f"[MLFLOW:REGISTRY] Alias not found: {model_name}@{alias}: {e}")
            return None

    def delete_model_alias(self, model_name: str, alias: str) -> bool:
        """
        Delete alias from a registered model.

        Args:
            model_name: Name of the registered model
            alias: Alias name to delete

        Returns:
            True if successful
        """
        if self._mlflow is None:
            return False

        try:
            client = self._gateway.get_client()
            client.delete_registered_model_alias(model_name, alias)
            logger.info(f"[MLFLOW:REGISTRY] Alias deleted: {model_name}@{alias}")
            return True
        except Exception as e:
            logger.warning(f"[MLFLOW:REGISTRY] Failed to delete alias: {e}")
            return False

    def promote_model(
        self,
        model_name: str,
        from_alias: str = "staging",
        to_alias: str = "champion",
    ) -> bool:
        """
        Promote model from one alias to another.

        Args:
            model_name: Name of the registered model
            from_alias: Source alias (default: "staging")
            to_alias: Target alias (default: "champion")

        Returns:
            True if successful
        """
        if self._mlflow is None:
            return False

        try:
            source_info = self.get_model_by_alias(model_name, from_alias)
            if not source_info:
                logger.warning(f"[MLFLOW:REGISTRY] Source alias not found: {model_name}@{from_alias}")
                return False

            version = source_info["version"]
            success = self.set_model_alias(model_name, to_alias, version)
            if success:
                logger.info(f"[MLFLOW:REGISTRY] Model promoted: {model_name} v{version} ({from_alias} -> {to_alias})")
            return success

        except Exception as e:
            logger.warning(f"[MLFLOW:REGISTRY] Failed to promote model: {e}")
            return False

    def get_model_aliases(self, model_name: str) -> dict[str, int]:
        """
        Get all aliases for a registered model.

        Args:
            model_name: Name of the registered model

        Returns:
            Dict mapping alias names to version numbers
        """
        if self._mlflow is None:
            return {}

        try:
            client = self._gateway.get_client()
            model = client.get_registered_model(model_name)

            aliases: dict[str, int] = {}
            if hasattr(model, "aliases") and model.aliases:
                for alias, version in model.aliases.items():
                    aliases[alias] = int(version)

            return aliases
        except Exception as e:
            logger.debug(f"[MLFLOW:REGISTRY] Failed to get aliases for {model_name}: {e}")
            return {}

    # =========================================================================
    # LOAD
    # =========================================================================

    def load_model_by_alias(self, model_name: str, alias: str = "champion") -> Any:
        """
        Load model from registry by alias.

        Args:
            model_name: Name of the registered model
            alias: Alias to load (default: "champion")

        Returns:
            Loaded model or None if failed
        """
        if self._mlflow is None:
            return None

        try:
            model_uri = f"models:/{model_name}@{alias}"
            model = self._mlflow.pyfunc.load_model(model_uri)
            logger.info(f"[MLFLOW:REGISTRY] Model loaded: {model_uri}")
            return model
        except Exception as e:
            logger.warning(f"[MLFLOW:REGISTRY] Failed to load model: {e}")
            return None


__all__ = ["MLflowModelRegistry"]
