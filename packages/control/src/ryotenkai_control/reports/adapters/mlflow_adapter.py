"""
MLflow Adapter Implementation.

Fetches data from MLflow and adapts it to Domain Entities.
Handles all the "dirty work" of traversing run hierarchy and normalizing keys.
"""

from __future__ import annotations

import json
import tempfile
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import mlflow
from mlflow.tracking import MlflowClient

from src.reports.domain.entities import (
    ExperimentData,
    MemoryEvent,
    MetricHistory,
    PhaseData,
    RunStatus,
)
from src.reports.domain.interfaces import IExperimentDataProvider
from src.utils.logger import get_logger

FILE_TRAINING_EVENTS = "training_events.json"
KEY_BATCH_SIZE = "batch_size"
KEY_END_TIME = "end_time"
KEY_LEARNING_RATE = "learning_rate"
KEY_MESSAGE = "message"
KEY_METRICS = "metrics"
KEY_START_TIME = "start_time"
KEY_STRATEGY_TYPE = "strategy_type"
KEY_UNKNOWN_LOWER = "unknown"

# Per-stage artifact filenames (one per pipeline stage)
STAGE_ARTIFACT_FILES: tuple[str, ...] = (
    "dataset_validator_results.json",
    "gpu_deployer_results.json",
    "training_monitor_results.json",
    "model_retriever_results.json",
    "inference_deployer_results.json",
    "evaluation_results.json",
)

if TYPE_CHECKING:
    from mlflow.entities import Run

    from src.infrastructure.mlflow.gateway import IMLflowGateway

logger = get_logger(__name__)

# =============================================================================
# CONSTANTS & MAPPINGS
# =============================================================================

# Metrics to fetch history for (immutable for WPS407)
METRICS_WITH_HISTORY: tuple[str, ...] = (
    "loss",
    "train_loss",
    "eval/loss",
    KEY_LEARNING_RATE,
    "entropy",
    "mean_token_accuracy",
    "grad_norm",
    "rewards/accuracies",
    "rewards/margins",
    "logps/chosen",
    "logps/rejected",
    "reward",
    "reward_std",
    "kl",
    "completion_length",
    "system/gpu_0_utilization_percentage",
    "system/gpu_0_memory_usage_megabytes",
    "system/gpu_0_memory_usage_percentage",
    "system/cpu_utilization_percentage",
    "system/system_memory_usage_megabytes",
    "system/system_memory_usage_percentage",
)


def _make_param_mapping() -> MappingProxyType[str, tuple[str, ...]]:
    """Build immutable param mapping (WPS407)."""
    return MappingProxyType(
        {
            KEY_LEARNING_RATE: (
                "training.hyperparams.actual.learning_rate",
                "training.hyperparams.learning_rate",
                "config.training.learning_rate",
                KEY_LEARNING_RATE,
                "lr",
            ),
            KEY_BATCH_SIZE: (
                "training.hyperparams.per_device_train_batch_size",
                "config.training.batch_size",
                KEY_BATCH_SIZE,
            ),
            "model_name": ("config.model.name", "model_name"),
            KEY_STRATEGY_TYPE: ("training.strategy_type", "chain_type", "strategy"),
            "grad_accum": (
                "training.hyperparams.gradient_accumulation_steps",
                "gradient_accumulation_steps",
                "config.training.hyperparams.gradient_accumulation_steps",
            ),
            "scheduler": ("training.hyperparams.lr_scheduler_type", "lr_scheduler_type", "learning_rate_scheduler"),
            "warmup_ratio": ("training.hyperparams.warmup_ratio", "warmup_ratio"),
        }
    )


PARAM_MAPPING = _make_param_mapping()


class MLflowAdapter(IExperimentDataProvider):
    """
    Adapter for MLflow data source.
    """

    def __init__(self, tracking_uri: str | None = None, *, gateway: IMLflowGateway | None = None):
        if gateway is not None:
            self._tracking_uri = gateway.uri
            self._client = gateway.get_client()
            mlflow.set_tracking_uri(gateway.uri)
        elif tracking_uri is not None:
            self._tracking_uri = tracking_uri
            self._client = MlflowClient(tracking_uri=tracking_uri)
            mlflow.set_tracking_uri(tracking_uri)
        else:
            raise ValueError("Either tracking_uri or gateway must be provided")

    def load(self, run_id: str) -> ExperimentData:
        """
        Load complete experiment data from MLflow run.
        """
        from src.pipeline.artifacts.base import StageArtifactEnvelope

        logger.info(f"[ADAPTER] Loading experiment data for run {run_id[:8]}...")

        try:
            root_run = self._client.get_run(run_id)
        except Exception as e:
            raise ValueError(f"Run not found: {run_id}") from e

        # 1. Fetch Artifacts (Events, Configs)
        artifacts_dir = self._download_essential_artifacts(run_id)

        # 2. Parse Global Training Events (Remote PC — still needed for phases/memory/gpu)
        training_events = self._load_json_events(artifacts_dir / FILE_TRAINING_EVENTS)

        # 3. Parse Per-Stage Envelope JSONs
        stage_envelopes: list[StageArtifactEnvelope] = []
        missing_artifacts: list[str] = []

        # Check for old-style pipeline_events.json (backward compat for legacy runs)
        legacy_pipeline_events = self._load_json_events(artifacts_dir / "pipeline_events.json")
        all_events_for_memory = legacy_pipeline_events + training_events

        for artifact_name in STAGE_ARTIFACT_FILES:
            local_path = artifacts_dir / artifact_name
            if local_path.exists():
                try:
                    raw = json.loads(local_path.read_text(encoding="utf-8"))
                    envelope = StageArtifactEnvelope.from_dict(raw)
                    stage_envelopes.append(envelope)
                    logger.debug(f"[ADAPTER] Loaded envelope: {artifact_name} (status={envelope.status})")
                except Exception as exc:
                    logger.warning(f"[ADAPTER] Failed to parse {artifact_name}: {exc}")
                    missing_artifacts.append(artifact_name)
            else:
                # Artifact absent means the stage never started — not an issue
                logger.debug(f"[ADAPTER] Stage artifact not found (stage did not run): {artifact_name}")

        # Populate typed result fields from envelopes
        validation_results: dict[str, Any] | None = None
        evaluation_results: dict[str, Any] | None = None
        deployment_results: dict[str, Any] | None = None
        training_stage_results: dict[str, Any] | None = None
        model_results: dict[str, Any] | None = None
        inference_results: dict[str, Any] | None = None

        for envelope in stage_envelopes:
            stage = envelope.stage
            if stage == "dataset_validator":
                validation_results = envelope.data
            elif stage == "model_evaluator":
                evaluation_results = envelope.data
            elif stage == "gpu_deployer":
                deployment_results = envelope.data
            elif stage == "training_monitor":
                training_stage_results = envelope.data
            elif stage == "model_retriever":
                model_results = envelope.data
            elif stage == "inference_deployer":
                inference_results = envelope.data

        # 4. Parse Memory Events from training_events (primary) + legacy pipeline_events
        memory_events = self._extract_memory_events(all_events_for_memory)

        # 5. Parse Source Config
        source_config = self._load_yaml_config(artifacts_dir)

        # Capture Root Params (including config.*)
        root_params = self._normalize_params(root_run.data.params)

        # 6. Find Training Context (Intermediate Run)
        training_run = self._find_training_run(run_id)
        training_params = training_run.data.params if training_run else {}

        # Merge training params into root_params (runtime context)
        root_params.update(self._normalize_params(training_params))

        # Fetch Global Resource History from Training Run
        resource_history = {}
        if training_run:
            for key in METRICS_WITH_HISTORY:
                if "system/" in key or "gpu" in key or "cpu" in key or "memory" in key:
                    hist = self._fetch_metric_history(training_run.info.run_id, key)
                    if hist:
                        resource_history[key] = hist

        # 7. Build Phases
        phases = self._build_phases(
            root_run,
            artifacts_dir,
            training_events=training_events,
            root_params=root_params,
        )

        # 8. Extract Hardware Info
        combined_params = root_params.copy()
        gpu_info = self._extract_gpu_info(all_events_for_memory, combined_params)

        # 9. Extract Model Info (Extra)
        model_extra_info = self._extract_model_info(all_events_for_memory)
        root_params.update(model_extra_info)

        # 10. Construct ExperimentData
        start_time = datetime.fromtimestamp(root_run.info.start_time / 1000)
        end_time = datetime.fromtimestamp(root_run.info.end_time / 1000) if root_run.info.end_time else None
        duration = (end_time - start_time).total_seconds() if end_time else 0.0

        experiment = self._client.get_experiment(root_run.info.experiment_id)

        if not training_events and not (artifacts_dir / FILE_TRAINING_EVENTS).exists():
            missing_artifacts.append(FILE_TRAINING_EVENTS)

        return ExperimentData(
            run_id=run_id,
            run_name=root_run.data.tags.get("mlflow.runName", run_id),
            experiment_name=experiment.name,
            status=RunStatus(root_run.info.status)
            if root_run.info.status in RunStatus.__members__
            else RunStatus.UNKNOWN,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
            phases=phases,
            source_config=source_config,
            root_params=root_params,
            resource_history=resource_history,
            memory_events=memory_events,
            stage_envelopes=stage_envelopes,
            validation_results=validation_results,
            evaluation_results=evaluation_results,
            deployment_results=deployment_results,
            training_stage_results=training_stage_results,
            model_results=model_results,
            inference_results=inference_results,
            gpu_info=gpu_info,
            missing_artifacts=missing_artifacts,
        )

    # =========================================================================
    # TREE TRAVERSAL & PHASES
    # =========================================================================

    def _build_phases(
        self,
        root_run: Run,
        _artifacts_dir: Path,
        *,
        training_events: list[dict[str, Any]] | None = None,
        root_params: dict[str, Any] | None = None,
    ) -> list[PhaseData]:
        """
        Traverse run hierarchy to build phases.
        Logic: Root -> Container (optional) -> Phases.
        """
        phases: list[PhaseData] = []

        # Get all children (sorted by time)
        children = self._get_sorted_children(root_run.info.run_id)

        # Strategy: Flatten hierarchy.
        # If a child is a container (has children), unpack its children.
        flattened_runs = []
        for child in children:
            grand_children = self._get_sorted_children(child.info.run_id)
            if grand_children:
                # It's a container, add grandchildren that look like phases
                # Filter for "phase_" or training runs
                phase_runs = [gc for gc in grand_children if "phase_" in gc.data.tags.get("mlflow.runName", "")]
                if phase_runs:
                    flattened_runs.extend(phase_runs)
                else:
                    # Maybe the grandchildren ARE the training runs directly?
                    # Let's add them all for now.
                    flattened_runs.extend(grand_children)
            else:
                flattened_runs.append(child)

        # Now build PhaseData from these runs
        for i, run in enumerate(flattened_runs):
            # Skip runs that are clearly not phases (e.g. system evaluations if any)
            # Use heuristic: Must have some metrics or training params
            if not run.data.metrics and not run.data.params:
                continue

            phase = self._run_to_phase_data(run, i)
            phases.append(phase)

        # Fallback: Some pipelines log phase metrics only as structured training events
        # (no nested MLflow child runs). In that case, reconstruct PhaseData from events.
        if not phases and training_events and root_params is not None:
            phases = self._build_phases_from_training_events(training_events, root_params)

        return phases

    @staticmethod
    def _build_phases_from_training_events(
        training_events: list[dict[str, Any]],
        root_params: dict[str, Any],
    ) -> list[PhaseData]:
        """
        Build phases from structured training events (PhaseExecutor/DataBuffer).

        Expected event shapes (from training_events.json):
        - PhaseExecutor start/complete with attributes.phase_idx and attributes.strategy_type
        - complete contains scalar metrics: train_loss, epoch, global_step, train_runtime, ...
        """
        # Local import to avoid circulars at import time
        from datetime import datetime

        phases_by_idx: dict[int, dict[str, Any]] = {}

        def parse_ts(ts_str: str | None) -> datetime | None:
            if not ts_str:
                return None
            with suppress(ValueError):
                return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            return None

        for e in training_events:
            attrs = e.get("attributes") or {}
            if not isinstance(attrs, dict):
                continue
            if "phase_idx" not in attrs:
                continue

            phase_idx_raw = attrs.get("phase_idx")
            if phase_idx_raw is None:
                continue
            try:
                idx = int(phase_idx_raw)
            except (TypeError, ValueError):
                continue

            src = str(e.get("source") or "")
            etype = str(e.get("event_type") or "").lower()

            rec = phases_by_idx.setdefault(
                idx,
                {
                    "idx": idx,
                    KEY_STRATEGY_TYPE: None,
                    KEY_START_TIME: None,
                    KEY_END_TIME: None,
                    KEY_METRICS: {},
                },
            )

            stype = attrs.get(KEY_STRATEGY_TYPE) or attrs.get("strategy")
            if stype:
                rec[KEY_STRATEGY_TYPE] = str(stype)

            ts = parse_ts(e.get("timestamp"))

            # Prefer PhaseExecutor events for timing/metrics
            if src.startswith("PhaseExecutor") and etype == "start":
                rec[KEY_START_TIME] = ts or rec.get(KEY_START_TIME)
            elif src.startswith("PhaseExecutor") and etype == "complete":
                rec[KEY_END_TIME] = ts or rec.get(KEY_END_TIME)

                # Extract scalar metrics from attributes
                for k in [
                    "train_loss",
                    "loss",
                    "epoch",
                    "global_step",
                    "train_runtime",
                    "train_samples_per_second",
                    "train_steps_per_second",
                    "learning_rate",
                ]:
                    if k in attrs and attrs[k] is not None:
                        try:
                            rec[KEY_METRICS][k] = float(attrs[k])
                        except (TypeError, ValueError):
                            continue

        # Convert collected records into PhaseData list
        phases: list[PhaseData] = []
        for idx in sorted(phases_by_idx.keys()):
            rec = phases_by_idx[idx]
            stype_raw = (
                rec.get(KEY_STRATEGY_TYPE) or root_params.get(f"config.strategy.{idx}.type") or KEY_UNKNOWN_LOWER
            )
            stype = str(stype_raw)

            # Build config (minimal, but enough for builder + e2e tests)
            phase_config: dict[str, Any] = {
                KEY_STRATEGY_TYPE: stype.upper(),
            }

            # Prefer per-phase learning_rate from events, then from config.strategy.* hyperparams, then global lr
            lr = (
                rec.get(KEY_METRICS, {}).get(KEY_LEARNING_RATE)
                or root_params.get(f"config.strategy.{idx}.hyperparams.learning_rate")
                or root_params.get(KEY_LEARNING_RATE)
            )
            if lr is not None:
                phase_config[KEY_LEARNING_RATE] = float(lr)

            # Propagate batch size if available
            bs = root_params.get(KEY_BATCH_SIZE) or root_params.get("training.hyperparams.per_device_train_batch_size")
            if bs is not None:
                with suppress(TypeError, ValueError):
                    phase_config[KEY_BATCH_SIZE] = int(bs)

            # Duration
            start_time = rec.get(KEY_START_TIME)
            end_time = rec.get(KEY_END_TIME)
            duration = rec.get(KEY_METRICS, {}).get("train_runtime")
            if duration is None and start_time and end_time:
                duration = max(0.0, (end_time - start_time).total_seconds())
            duration_seconds = float(duration or 0.0)

            metrics = dict(rec.get(KEY_METRICS) or {})

            phases.append(
                PhaseData(
                    idx=idx,
                    name=f"phase_{idx}_{stype}",
                    strategy=stype.upper(),
                    status=RunStatus.FINISHED,
                    duration_seconds=duration_seconds,
                    start_time=start_time,
                    end_time=end_time,
                    config=phase_config,
                    metrics=metrics,
                    history={},
                )
            )

        return phases

    def _find_training_run(self, root_run_id: str) -> Run | None:
        """
        Find the intermediate Training Run (child of Pipeline, parent of Phases).
        This run contains hardware info and runtime params.
        """
        children = self._get_sorted_children(root_run_id)
        if not children:
            return None

        # Strategy 1: Look for specific runtime keys
        for child in children:
            if "gpu_name" in child.data.params or "training_type" in child.data.params:
                return child

        # Strategy 2: Look for container run (has children)
        for child in children:
            grand_children = self._get_sorted_children(child.info.run_id)
            if grand_children:
                return child

        return None

    def _get_sorted_children(self, parent_id: str) -> list[Run]:
        """Fetch children sorted by start time."""
        try:
            run = self._client.get_run(parent_id)
            return self._client.search_runs(
                experiment_ids=[run.info.experiment_id],
                filter_string=f"tags.`mlflow.parentRunId` = '{parent_id}'",
                order_by=["start_time ASC"],
            )
        except (mlflow.exceptions.MlflowException, OSError, KeyError, TypeError, ValueError):  # type: ignore[attr-defined]
            return []

    def _run_to_phase_data(self, run: Run, idx: int) -> PhaseData:
        """Convert a single Run to PhaseData."""
        # Normalize Config
        config = self._normalize_params(run.data.params)

        # Extract name/strategy
        run_name = run.data.tags.get("mlflow.runName", f"phase_{idx}")
        strategy = (
            run.data.tags.get("training.strategy_type")
            or run.data.tags.get("chain_type")
            or config.get(KEY_STRATEGY_TYPE, "UNKNOWN")
        )

        # Normalize Metrics
        metrics = dict(run.data.metrics)

        # Fetch Histories for key metrics
        history = {}
        for key in METRICS_WITH_HISTORY:
            # Check if metric exists in this run
            if key in run.data.metrics:  # Only fetch if we have a final value
                hist = self._fetch_metric_history(run.info.run_id, key)
                if hist:
                    history[key] = hist

        # Duration
        start_ts = run.info.start_time
        end_ts = run.info.end_time or start_ts
        duration = (end_ts - start_ts) / 1000.0

        start_dt = datetime.fromtimestamp(start_ts / 1000)
        end_dt = datetime.fromtimestamp(end_ts / 1000) if run.info.end_time else None

        return PhaseData(
            idx=idx,
            name=run_name,
            strategy=strategy.upper(),
            status=RunStatus(run.info.status) if run.info.status in RunStatus.__members__ else RunStatus.UNKNOWN,
            duration_seconds=duration,
            start_time=start_dt,
            end_time=end_dt,
            config=config,
            metrics=metrics,
            history=history,
        )

    def _fetch_metric_history(self, run_id: str, key: str) -> MetricHistory | None:
        """Fetch history from MLflow."""
        try:
            history = self._client.get_metric_history(run_id, key)
            if not history:
                return None

            return MetricHistory(
                key=key,
                values=[h.value for h in history],
                steps=[h.step for h in history],
                timestamps=[h.timestamp for h in history],
            )
        except (mlflow.exceptions.MlflowException, OSError, KeyError, TypeError, ValueError):  # type: ignore[attr-defined]
            return None

    # =========================================================================
    # HELPERS
    # =========================================================================

    def _normalize_params(self, params: dict[str, str]) -> dict[str, Any]:
        """
        Normalize MLflow params (flat dict with dots) to Domain config.
        """
        config = {}

        # 1. Apply Mapping
        for domain_key, possible_keys in PARAM_MAPPING.items():
            for pk in possible_keys:
                if pk in params:
                    config[domain_key] = self._parse_value(params[pk])
                    break

        # 2. Keep all original params too (for detailed inspection)
        # But try to parse values
        for k, v in params.items():
            if k not in config:  # Don't overwrite normalized ones
                config[k] = self._parse_value(v)

        return config

    @staticmethod
    def _parse_value(val: str) -> Any:
        """Try to parse string value to int/float/bool."""
        if val.lower() == "true":
            return True
        if val.lower() == "false":
            return False
        try:
            return int(val)
        except ValueError:
            try:
                return float(val)
            except ValueError:
                return val

    def _download_essential_artifacts(self, run_id: str) -> Path:
        """Download events and config to temp dir."""
        temp_dir = Path(tempfile.mkdtemp(prefix=f"report_{run_id[:8]}_"))

        # 1. List available artifacts first to avoid hangs on missing files
        available_artifacts: set[str] = set()
        available_paths_flat: set[str] = set()  # flattened, including subfolder entries
        try:
            artifacts = self._client.list_artifacts(run_id)
            for a in artifacts:
                available_artifacts.add(a.path)
                available_paths_flat.add(a.path)
            # Also enumerate subfolders (e.g. "evaluation/")
            for a in artifacts:
                if a.is_dir:
                    sub = self._client.list_artifacts(run_id, a.path)
                    for s in sub:
                        available_paths_flat.add(s.path)
        except Exception as e:
            logger.warning(f"[ADAPTER] Failed to list artifacts for {run_id}: {e}")
            return temp_dir

        # 2. Download training_events.json + legacy pipeline_events.json (backward compat)
        essential_jsons = [FILE_TRAINING_EVENTS, "pipeline_events.json"]
        for artifact in essential_jsons:
            if artifact in available_artifacts or artifact in available_paths_flat:
                try:
                    self._client.download_artifacts(run_id, artifact, str(temp_dir))
                except Exception as e:
                    logger.warning(f"[ADAPTER] Failed to download {artifact}: {e}")

        # 3. Download per-stage JSON artifacts
        for artifact_name in STAGE_ARTIFACT_FILES:
            if artifact_name in available_paths_flat or artifact_name in available_artifacts:
                dest_subdir = str(temp_dir)
                # Preserve subfolder structure (e.g. "evaluation/")
                if "/" in artifact_name:
                    subfolder = artifact_name.rsplit("/", 1)[0]
                    dest_path = temp_dir / subfolder
                    dest_path.mkdir(parents=True, exist_ok=True)
                    dest_subdir = str(temp_dir)
                try:
                    self._client.download_artifacts(run_id, artifact_name, dest_subdir)
                except Exception as e:
                    logger.warning(f"[ADAPTER] Failed to download {artifact_name}: {e}")

        # 4. Download Configs (YAML)
        for artifact in available_artifacts:
            if artifact.endswith(".yaml") or artifact.endswith(".yml"):
                try:
                    self._client.download_artifacts(run_id, artifact, str(temp_dir))
                except Exception as e:
                    logger.warning(f"[ADAPTER] Failed to download config {artifact}: {e}")

        return temp_dir

    @staticmethod
    def _load_json_events(path: Path) -> list[dict[str, Any]]:
        """Load events from JSON file."""
        if not path.exists():
            return []
        try:
            content = path.read_text()
            data = json.loads(content)
            events = data.get("events", [])
            logger.debug(f"[ADAPTER] Loaded {len(events)} events from {path.name}")
            return events
        except Exception as e:
            logger.warning(f"[ADAPTER] Failed to load events from {path}: {e}")
            return []

    @staticmethod
    def _load_yaml_config(temp_dir: Path) -> dict[str, Any]:
        """Load first available YAML config."""
        import yaml as yaml_module

        yaml: Any = yaml_module

        for ext in ["*.yaml", "*.yml"]:
            for f in temp_dir.glob(ext):
                with suppress(Exception):
                    return yaml.safe_load(f.read_text()) or {}
        return {}

    @staticmethod
    def _extract_memory_events(events: list[dict[str, Any]]) -> list[MemoryEvent]:
        """Extract MemoryManager events."""
        memory_events = []
        for e in events:
            if e.get("source") != "MemoryManager":
                continue

            # Parse timestamp safely
            ts_str = e.get("timestamp")
            ts = datetime.now()
            if ts_str:
                with suppress(ValueError):
                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))

            # Determine type
            etype = e.get("event_type", "info")
            msg = e.get(KEY_MESSAGE, "").lower()

            if "cache cleared" in msg:
                etype = "cache_clear"
            elif "oom" in msg or "oom" in etype.lower():
                etype = "oom"
            elif etype in ("warning", "critical"):
                pass

            memory_events.append(
                MemoryEvent(
                    timestamp=ts,
                    event_type=etype,
                    message=e.get(KEY_MESSAGE, ""),
                    phase=e.get("phase"),
                    freed_mb=e.get("freed_mb"),
                    utilization_percent=e.get("utilization_percent"),
                    operation=e.get("operation"),
                )
            )

        return memory_events

    @staticmethod
    def _extract_gpu_info(events: list[dict[str, Any]], params: dict[str, str]) -> dict[str, Any]:
        """Extract GPU info from events or params."""
        gpu_info = {}

        # Try params first
        if "mm.gpu_name" in params:
            gpu_info["name"] = params["mm.gpu_name"]

        for e in events:
            # 1. Try Structured Data (New format)
            attrs = e.get("attributes", {})
            if "gpu_name" in attrs:
                gpu_info["name"] = attrs["gpu_name"]

                # Support both key variants (MemoryManager vs RunTraining)
                vram = attrs.get("total_vram_gb") or attrs.get("vram_gb")
                if vram:
                    gpu_info["total_vram"] = f"{vram:.1f}GB"
                    gpu_info["total_vram_gb_raw"] = vram

                tier = attrs.get("gpu_tier") or attrs.get("tier")
                if tier:
                    gpu_info["tier"] = tier

                # If we found full info, break. Else keep searching for better event.
                if vram and tier:
                    break

        return gpu_info

    @staticmethod
    def _extract_model_info(events: list[dict[str, Any]]) -> dict[str, Any]:
        """Extract model info from events."""
        model_info = {}
        for e in events:
            # 1. Try Structured Data
            attrs = e.get("attributes", {})
            if "total_parameters" in attrs:
                model_info["total_parameters"] = attrs["total_parameters"]
                model_info["trainable_parameters"] = attrs.get("trainable_parameters")
                model_info["trainable_percent"] = attrs.get("trainable_percent")
                model_info["model_loading_time_seconds"] = attrs.get("model_loading_time_seconds")
                # Don't return yet, look for other info

            # 2. Extract Adapter Size from Model Retriever
            if e.get("source") == "Model Retriever" and "Model size:" in e.get("message", ""):
                import re

                match = re.search(r"Model size: ([\d.]+) MB", e["message"])
                if match:
                    model_info["model_size_mb"] = float(match.group(1))

        return model_info
