"""Config hash computation and drift validation.

Factored out of PipelineOrchestrator. Exposes two pure operations:

* ``build_config_hashes`` — compute the three hashes the pipeline stores
  on each attempt (training_critical, late_stage, model_dataset).
* ``validate_drift`` — compare the freshly-computed hashes against a
  persisted state and, when the user tries to resume/restart into a
  scope that no longer matches, return a ``ConfigDriftError``.

The validator is stateless apart from the config it holds; the persisted
state is passed in on each call so tests can build fixtures trivially.

Legacy fallback (state without ``model_dataset_config_hash``) compares the
full ``training_critical`` hash — behaviour preserved verbatim from the
original orchestrator implementation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.pipeline.stages import StageNames
from src.pipeline.state import hash_payload
from src.utils.result import AppError, ConfigDriftError

if TYPE_CHECKING:
    from src.pipeline.state import PipelineState
    from src.utils.config import PipelineConfig


def compute_config_hashes(config: PipelineConfig) -> dict[str, str]:
    """Pure function: three scoped hashes for restart/resume drift detection.

    Shared by :class:`ConfigDriftValidator` (orchestrator path, takes
    ``self._config``) and
    :func:`src.pipeline.launch.restart_options.compute_config_hashes`
    (lightweight launch-options path). Single source of truth for the
    payload shape — drifting the two used to be a real risk.

    Returns a dict with three keys:

    * ``training_critical`` — model + training + datasets + provider; any
      change here invalidates resume across attempts.
    * ``late_stage`` — inference + evaluation; allowed to change for manual
      restart from those stages but not for plain resume.
    * ``model_dataset`` — model + training + datasets only (no provider);
      lets provider-only edits resume without re-tripping training_critical.
    """
    model_dataset_payload = {
        "model": config.model.model_dump(mode="json"),
        "training": config.training.model_dump(mode="json"),
        "datasets": {name: cfg.model_dump(mode="json") for name, cfg in config.datasets.items()},
    }
    training_payload = {
        **model_dataset_payload,
        "provider_name": config.get_active_provider_name(),
        "provider": config.get_provider_config(),
    }
    late_payload = {
        "inference": config.inference.model_dump(mode="json"),
        "evaluation": config.evaluation.model_dump(mode="json"),
    }
    return {
        "training_critical": hash_payload(training_payload),
        "late_stage": hash_payload(late_payload),
        "model_dataset": hash_payload(model_dataset_payload),
    }


class ConfigDriftValidator:
    """Compute config hashes and detect drift across attempts."""

    def __init__(self, config: PipelineConfig) -> None:
        self._config = config

    def build_config_hashes(self) -> dict[str, str]:
        """Compute the three scoped config hashes for the current config."""
        return compute_config_hashes(self._config)

    def validate_drift(
        self,
        *,
        state: PipelineState,
        start_stage_name: str,
        config_hashes: dict[str, str],
        resume: bool,
    ) -> AppError | None:
        """Return a ConfigDriftError when the stored config scope has drifted.

        Policy (unchanged from pre-refactor orchestrator):

        * ``model_dataset`` drift is fatal for any resume/restart.
        * ``late_stage`` drift is allowed only when the user is manually
          restarting from Inference Deployer or Model Evaluator.

        Legacy states without ``model_dataset_config_hash`` fall back to
        comparing ``training_critical`` hash.
        """
        # Prefer the fine-grained model_dataset hash when present; legacy states
        # without it fall back to the broader training_critical hash so they
        # still detect drift (provider-only changes are allowed on modern runs).
        if state.model_dataset_config_hash:
            model_dataset_changed = state.model_dataset_config_hash != config_hashes["model_dataset"]
            drift_scope = "model_dataset"
        else:
            model_dataset_changed = state.training_critical_config_hash != config_hashes["training_critical"]
            drift_scope = "training_critical"

        late_changed = state.late_stage_config_hash != config_hashes["late_stage"]

        if model_dataset_changed:
            return ConfigDriftError(
                message=(
                    f"{drift_scope} config changed for existing logical run; "
                    "resume/restart is blocked. Use the original config or start a new run."
                ),
                details={
                    "scope": drift_scope,
                    "start_stage_name": start_stage_name,
                    "resume": resume,
                },
            )
        if late_changed and (
            resume or start_stage_name not in {StageNames.INFERENCE_DEPLOYER, StageNames.MODEL_EVALUATOR}
        ):
            return ConfigDriftError(
                message=(
                    "late_stage config changed; only manual restart from "
                    "Inference Deployer or Model Evaluator is allowed."
                ),
                details={
                    "scope": "late_stage",
                    "start_stage_name": start_stage_name,
                    "resume": resume,
                },
            )
        return None


__all__ = ["ConfigDriftValidator"]
