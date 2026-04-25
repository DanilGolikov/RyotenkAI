"""HTTP API for the Datasets project tab.

Three endpoints, all scoped to a single dataset of a project:

  * ``GET  /preview``    — paginated rows for the UI list / table
  * ``GET  /path-check`` — quick "does the file exist / is the HF repo
    reachable" check, used by the source-config form
  * ``POST /validate``   — one-shot quality validation: format-check vs
    every strategy that references this dataset, then a sequential
    plugin pass. Does NOT spin up the full pipeline / GPU.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from src.api.dependencies import DatasetRequestContext, resolve_dataset_key
from src.config.datasets.constants import SOURCE_TYPE_HUGGINGFACE, SOURCE_TYPE_LOCAL
from src.data.preview.loader import (
    PREVIEW_LIMIT_MAX,
    DatasetPreviewLoader,
)
from src.data.validation.standalone import (
    FormatCheckResult,
    StandalonePluginRun,
    check_dataset_format,
    run_plugins,
)
from src.utils.logger import logger

router = APIRouter(prefix="/projects/{project_id}/datasets", tags=["datasets"])

Split = Literal["train", "eval"]


# =============================================================================
# Response models
# =============================================================================


class PreviewRow(BaseModel):
    """Marker subclass — JSON-shaped, free-form keys. We expose ``dict``
    rather than a typed model so the UI can render arbitrary jsonl
    schemas without a frontend regen on every dataset shape change."""

    model_config = {"extra": "allow"}


class PreviewResponse(BaseModel):
    rows: list[dict[str, Any]]
    total: int | None = Field(
        default=None,
        description="Total row count, or null when streaming (HF datasets).",
    )
    has_more: bool
    schema_hint: list[str] = Field(
        default_factory=list,
        description="Union of keys across the returned rows — stable column order for the structured view.",
    )


class PathCheckSplit(BaseModel):
    exists: bool
    line_count: int | None = None
    size_bytes: int | None = None
    error: str | None = None


class PathCheckResponse(BaseModel):
    source_type: Literal["local", "huggingface"]
    train: PathCheckSplit
    eval: PathCheckSplit | None = None


class FormatCheckPayload(BaseModel):
    strategy_type: str
    ok: bool
    message: str = ""


class PluginRunPayload(BaseModel):
    plugin_id: str
    plugin_name: str
    passed: bool
    crashed: bool = False
    duration_ms: float
    metrics: dict[str, float] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    error_groups: list["ErrorGroupPayload"] = Field(default_factory=list)


class ErrorGroupPayload(BaseModel):
    error_type: str
    sample_indices: list[int]
    total_count: int


class ValidateRequest(BaseModel):
    split: Split = "train"
    max_samples: int | None = Field(
        default=1000,
        description="Cap rows fed to the plugins. None = whole dataset (slow).",
    )


class ValidateResponse(BaseModel):
    duration_ms: int
    format_check: list[FormatCheckPayload]
    format_check_error: str | None = None
    plugin_results: list[PluginRunPayload]


PluginRunPayload.model_rebuild()


# =============================================================================
# Endpoints
# =============================================================================


@router.get("/{dataset_key}/preview", response_model=PreviewResponse, name="preview_dataset")
def preview_dataset(
    split: Split = Query("train"),
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=PREVIEW_LIMIT_MAX),
    ctx: DatasetRequestContext = Depends(resolve_dataset_key),
) -> PreviewResponse:
    loader = DatasetPreviewLoader()
    cfg = ctx.dataset_config
    started = time.perf_counter()
    try:
        page = _preview_page(loader, ctx.project_root, cfg, split=split, offset=offset, limit=limit)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"path_not_found: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=f"hf_load_failed: {exc}") from exc
    duration_ms = int((time.perf_counter() - started) * 1000)
    logger.info(
        "[API:DATASETS] preview project=%s key=%s split=%s offset=%d limit=%d → %d rows in %dms",
        ctx.project_id,
        ctx.dataset_key,
        split,
        offset,
        limit,
        len(page.rows),
        duration_ms,
    )
    return PreviewResponse(
        rows=page.rows,
        total=page.total,
        has_more=page.has_more,
        schema_hint=page.schema_hint,
    )


@router.get("/{dataset_key}/path-check", response_model=PathCheckResponse, name="check_dataset_paths")
def check_dataset_paths(
    ctx: DatasetRequestContext = Depends(resolve_dataset_key),
) -> PathCheckResponse:
    cfg = ctx.dataset_config
    source_type = cfg.get_source_type()

    if source_type == SOURCE_TYPE_LOCAL:
        assert cfg.source_local is not None
        train_split = _local_split_check(ctx.project_root, cfg.source_local.local_paths.train)
        eval_split: PathCheckSplit | None = None
        eval_path = cfg.source_local.local_paths.eval
        if eval_path:
            eval_split = _local_split_check(ctx.project_root, eval_path)
        return PathCheckResponse(source_type=SOURCE_TYPE_LOCAL, train=train_split, eval=eval_split)

    # huggingface
    assert cfg.source_hf is not None
    train_split = _hf_repo_check(cfg.source_hf.train_id)
    eval_split = _hf_repo_check(cfg.source_hf.eval_id) if cfg.source_hf.eval_id else None
    return PathCheckResponse(source_type=SOURCE_TYPE_HUGGINGFACE, train=train_split, eval=eval_split)


@router.post("/{dataset_key}/validate", response_model=ValidateResponse, name="validate_dataset")
def validate_dataset(
    body: ValidateRequest,
    ctx: DatasetRequestContext = Depends(resolve_dataset_key),
) -> ValidateResponse:
    started = time.perf_counter()
    cfg = ctx.dataset_config

    # 1. Load dataset (sample-only for speed). For local jsonl we go
    #    via the existing loader pipeline so plugins see the same
    #    Dataset object they would in a real run; for HF we use
    #    streaming to avoid downloading huge corpora.
    try:
        from datasets import Dataset
    except ImportError as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"datasets_missing: {exc}") from exc

    try:
        dataset = _load_for_validation(ctx.project_root, cfg, split=body.split, max_samples=body.max_samples)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"path_not_found: {exc}") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=f"load_failed: {exc}") from exc
    if not isinstance(dataset, Dataset):
        # IterableDataset path — wrap into a list-backed Dataset so
        # plugins that expect column metadata work uniformly.
        dataset = _iterable_to_dataset(dataset, max_samples=body.max_samples)

    # 2. Format check — reverse-lookup strategy_phases that reference
    #    this dataset_key. Strict 1:1 business rule: ≥2 phases pointing
    #    here with DIFFERENT strategy_type is a config error.
    strategy_phases = _strategy_phases_for_dataset(ctx)
    pipeline_cfg = _try_build_pipeline_config(ctx.parsed_pipeline_config)
    format_check_payloads: list[FormatCheckPayload] = []
    format_check_error: str | None = None
    if pipeline_cfg is not None and strategy_phases:
        bundle = check_dataset_format(dataset, ctx.dataset_key, strategy_phases, pipeline_cfg)
        if bundle.is_failure():
            format_check_error = bundle.unwrap_err().message
        else:
            format_check_payloads = [
                FormatCheckPayload(strategy_type=r.strategy_type, ok=r.ok, message=r.message)
                for r in bundle.unwrap()
            ]
    elif not strategy_phases:
        format_check_error = (
            "No strategy references this dataset — skipping format check. "
            "Attach a training strategy to validate compatibility."
        )

    # 3. Plugin pass (sequential, per-plugin try/except inside).
    plugin_runs = _run_configured_plugins(dataset, cfg)

    duration_ms = int((time.perf_counter() - started) * 1000)
    return ValidateResponse(
        duration_ms=duration_ms,
        format_check=format_check_payloads,
        format_check_error=format_check_error,
        plugin_results=[_plugin_payload(r) for r in plugin_runs],
    )


# =============================================================================
# Internal helpers
# =============================================================================


def _resolve_local_path(project_root: Path, raw_path: str) -> Path:
    """
    Resolve a JSONL path. Relative paths are anchored at the project
    root; absolute paths are accepted as-is (datasets often live on a
    shared NAS outside the project tree). Both paths are ``resolve()``-d
    to follow symlinks before any further checks.
    """
    p = Path(raw_path).expanduser()
    if not p.is_absolute():
        p = project_root / p
    return p.resolve()


def _local_split_check(project_root: Path, raw_path: str) -> PathCheckSplit:
    if not raw_path:
        return PathCheckSplit(exists=False, error="path_empty")
    try:
        path = _resolve_local_path(project_root, raw_path)
    except OSError as exc:
        return PathCheckSplit(exists=False, error=f"resolve_failed: {exc}")
    if not path.exists():
        return PathCheckSplit(exists=False, error="missing")
    if not path.is_file():
        return PathCheckSplit(exists=False, error="not_a_file")
    try:
        size = path.stat().st_size
    except OSError as exc:
        return PathCheckSplit(exists=True, error=f"stat_failed: {exc}")
    # Line count is on a hot path for the UI, so we lazy-count via the
    # loader's mtime cache.
    line_count: int | None
    try:
        from src.data.preview.loader import _count_lines  # type: ignore[attr-defined]

        line_count = _count_lines(path)
    except Exception as exc:
        logger.warning("[API:DATASETS] line count failed for %s: %s", path, exc)
        line_count = None
    return PathCheckSplit(exists=True, line_count=line_count, size_bytes=size)


def _hf_repo_check(repo_id: str) -> PathCheckSplit:
    """Lightweight reachability check — HEAD on the public HF API. We
    deliberately do NOT instantiate the dataset (cold start can take
    10-15 seconds); existence is enough."""
    if not repo_id:
        return PathCheckSplit(exists=False, error="repo_id_empty")
    try:
        import urllib.error
        import urllib.request

        url = f"https://huggingface.co/api/datasets/{repo_id}"
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return PathCheckSplit(exists=resp.status == 200)
    except urllib.error.HTTPError as exc:
        if exc.code == 401 or exc.code == 403:
            return PathCheckSplit(exists=True, error="auth_required")
        if exc.code == 404:
            return PathCheckSplit(exists=False, error="not_found")
        return PathCheckSplit(exists=False, error=f"http_{exc.code}")
    except Exception as exc:
        return PathCheckSplit(exists=False, error=f"unreachable: {exc}")


def _preview_page(
    loader: DatasetPreviewLoader,
    project_root: Path,
    cfg: Any,
    *,
    split: Split,
    offset: int,
    limit: int,
):
    source_type = cfg.get_source_type()
    if source_type == SOURCE_TYPE_LOCAL:
        assert cfg.source_local is not None
        raw_path = cfg.source_local.local_paths.train if split == "train" else cfg.source_local.local_paths.eval
        if not raw_path:
            raise FileNotFoundError(f"{split}_path_not_configured")
        path = _resolve_local_path(project_root, raw_path)
        return loader.preview_local_jsonl(path, offset=offset, limit=limit)

    # huggingface
    assert cfg.source_hf is not None
    repo_id = cfg.source_hf.train_id if split == "train" else cfg.source_hf.eval_id
    if not repo_id:
        raise FileNotFoundError(f"{split}_repo_not_configured")
    hf_token = _resolve_hf_token()
    return loader.preview_hf(repo_id, split=split, offset=offset, limit=limit, hf_token=hf_token)


def _load_for_validation(project_root: Path, cfg: Any, *, split: Split, max_samples: int | None):
    """Load a Dataset suitable for plugin validation. Local: read jsonl
    directly. HF: stream with `take(max_samples)`."""
    from datasets import Dataset, load_dataset

    source_type = cfg.get_source_type()
    if source_type == SOURCE_TYPE_LOCAL:
        assert cfg.source_local is not None
        raw_path = cfg.source_local.local_paths.train if split == "train" else cfg.source_local.local_paths.eval
        if not raw_path:
            raise FileNotFoundError(f"{split}_path_not_configured")
        path = _resolve_local_path(project_root, raw_path)
        if not path.exists():
            raise FileNotFoundError(str(path))
        ds = Dataset.from_json(str(path))
        if max_samples is not None and max_samples < len(ds):
            ds = ds.select(range(max_samples))
        return ds

    assert cfg.source_hf is not None
    repo_id = cfg.source_hf.train_id if split == "train" else cfg.source_hf.eval_id
    if not repo_id:
        raise FileNotFoundError(f"{split}_repo_not_configured")
    hf_token = _resolve_hf_token()
    kwargs: dict[str, Any] = {"streaming": True}
    if hf_token:
        kwargs["token"] = hf_token
    iter_ds = load_dataset(repo_id, split=split, **kwargs)
    if max_samples is not None:
        iter_ds = iter_ds.take(max_samples)
    return iter_ds


def _iterable_to_dataset(iter_ds: Any, max_samples: int | None):
    from datasets import Dataset

    rows: list[dict[str, Any]] = []
    cap = max_samples if max_samples is not None else 10_000
    for idx, row in enumerate(iter_ds):
        if idx >= cap:
            break
        if isinstance(row, dict):
            rows.append(row)
    return Dataset.from_list(rows)


def _strategy_phases_for_dataset(ctx: DatasetRequestContext) -> list[Any]:
    """Reverse-lookup: which `training.strategies[*]` reference this
    dataset_key? Strict 1:1 — but we still return the list so the
    format-check surface multiple strategy mismatches at once."""
    training = ctx.parsed_pipeline_config.get("training") or {}
    if not isinstance(training, dict):
        return []
    strategies = training.get("strategies") or []
    if not isinstance(strategies, list):
        return []
    # Pydantic StrategyPhaseConfig has `dataset: str | None`; default
    # None meaning "use the 'default' dataset". We treat key="default"
    # as the implicit fallback for None entries.
    matching: list[Any] = []
    for raw in strategies:
        if not isinstance(raw, dict):
            continue
        ref = raw.get("dataset")
        if ref is None and ctx.dataset_key == "default":
            matching.append(_phase_from_dict(raw))
        elif ref == ctx.dataset_key:
            matching.append(_phase_from_dict(raw))
    return matching


def _phase_from_dict(raw: dict[str, Any]):
    """Materialise a StrategyPhaseConfig from a raw dict, swallowing
    pydantic errors — the user might still be editing the form."""
    from src.config.training.strategies.phase import StrategyPhaseConfig

    try:
        return StrategyPhaseConfig.model_validate(raw)
    except Exception:
        # Fall back to a minimal dummy with strategy_type if available
        # so format-check can still flag "unknown strategy".
        class _Dummy:
            strategy_type = raw.get("strategy_type", "unknown")

        return _Dummy()


def _try_build_pipeline_config(parsed: dict[str, Any]):
    """Attempt to construct a PipelineConfig from the parsed dict. If
    the user is still editing and other parts of the config are broken,
    returns None — the format check is then skipped with a soft
    warning, but the plugin pass still runs (it doesn't need it)."""
    try:
        from src.utils.config import PipelineConfig

        return PipelineConfig.model_validate(parsed)
    except Exception as exc:
        logger.info(
            "[API:DATASETS] PipelineConfig parse failed (format check skipped): %s",
            exc,
        )
        return None


def _run_configured_plugins(dataset: Any, cfg: Any) -> list[StandalonePluginRun]:
    """Resolve plugin instances from `cfg.validations.plugins` via the
    registry, then invoke `standalone.run_plugins`.

    Plugins that fail to instantiate (e.g. the referenced plugin id
    isn't in the community catalog) are NOT silently dropped — they
    come back as crashed ``StandalonePluginRun`` entries so the UI
    shows the full picture and the user can debug why a referenced
    plugin isn't available.
    """
    plugin_configs = list(cfg.validations.plugins) if cfg.validations and cfg.validations.plugins else []
    if not plugin_configs:
        return []

    from src.community import catalog
    from src.data.validation.registry import ValidationPluginRegistry

    catalog.ensure_loaded()  # idempotent
    registry = ValidationPluginRegistry()

    # Manifest defaults are merged into the YAML params/thresholds as a
    # FALLBACK so existing instances saved before a plugin gained a
    # `suggested_params` block don't crash on init. The user-supplied
    # value always wins; we only fill in keys the user never provided.
    manifests_by_id: dict[str, Any] = {}
    try:
        from src.community.catalog import catalog as community_catalog

        for loaded in community_catalog.plugins("validation"):
            manifests_by_id[loaded.manifest.plugin.id] = loaded.manifest
    except Exception as exc:  # defensive — never block validation
        logger.debug("[API:DATASETS] catalog manifest scan failed: %s", exc)

    def _merged(raw_params: dict[str, Any], suggested: dict[str, Any]) -> dict[str, Any]:
        """User keys win; manifest fills missing keys only."""
        out = dict(suggested)
        out.update(raw_params)
        return out

    # Separate lists for plugins that loaded vs plugins that failed —
    # failed entries bypass `run_plugins` (nothing to call) but still
    # reach the response so the UI can render them with a crash pill.
    instances: list[tuple[str, str, Any]] = []
    failed: list[StandalonePluginRun] = []
    for raw in plugin_configs:
        manifest = manifests_by_id.get(raw.plugin)
        suggested_params: dict[str, Any] = {}
        suggested_thresholds: dict[str, Any] = {}
        if manifest is not None:
            suggested_params = dict(getattr(manifest, "suggested_params", {}) or {})
            suggested_thresholds = dict(getattr(manifest, "suggested_thresholds", {}) or {})
        try:
            plugin = registry.get_plugin(
                raw.plugin,
                params=_merged(dict(raw.params or {}), suggested_params),
                thresholds=_merged(dict(raw.thresholds or {}), suggested_thresholds),
            )
        except Exception as exc:
            logger.warning(
                "[API:DATASETS] plugin instantiation failed (id=%s plugin=%s): %s",
                raw.id,
                raw.plugin,
                exc,
            )
            # Be precise about the failure mode so the UI message
            # points the user at the right fix:
            #   - "not found" → plugin id is wrong (rename in YAML, pick
            #     another from the catalog)
            #   - anything else → usually a params/thresholds validation
            #     failure from the plugin's own constructor; the exc
            #     message comes straight through so the user sees the
            #     missing field / bad value.
            exc_text = str(exc).strip().strip('"')
            is_missing = "not found" in exc_text.lower() and raw.plugin in exc_text
            if is_missing:
                error_msg = (
                    f"Plugin '{raw.plugin}' is not registered in the community "
                    f"catalog. Pick another plugin or remove this entry."
                )
                recommendations: list[str] = [
                    "Check that the plugin directory under community/validation/",
                    "has a valid manifest.toml and that its `id` matches.",
                ]
            else:
                error_msg = (
                    f"Plugin '{raw.plugin}' failed to initialise: {exc_text}"
                )
                recommendations = [
                    "Open Configure on this instance and review params/thresholds.",
                ]
            failed.append(
                StandalonePluginRun(
                    plugin_id=raw.id,
                    plugin_name=raw.plugin,
                    passed=False,
                    crashed=True,
                    errors=[error_msg],
                    recommendations=recommendations,
                )
            )
            continue
        instances.append((raw.id, raw.plugin, plugin))

    runs = run_plugins(dataset, instances)
    # Preserve the YAML ordering so the UI reads in the same order the
    # user attached the plugins.
    order = {raw.id: idx for idx, raw in enumerate(plugin_configs)}
    combined = runs + failed
    combined.sort(key=lambda r: order.get(r.plugin_id, len(order)))
    return combined


def _plugin_payload(run: StandalonePluginRun) -> PluginRunPayload:
    return PluginRunPayload(
        plugin_id=run.plugin_id,
        plugin_name=run.plugin_name,
        passed=run.passed,
        crashed=run.crashed,
        duration_ms=run.duration_ms,
        metrics=run.metrics,
        warnings=run.warnings,
        errors=run.errors,
        recommendations=run.recommendations,
        error_groups=[
            ErrorGroupPayload(
                error_type=g.error_type,
                sample_indices=list(g.sample_indices),
                total_count=g.total_count,
            )
            for g in run.error_groups
        ],
    )


def _resolve_hf_token() -> str | None:
    """Pull HF token from existing Secrets layer (set via Settings →
    Integrations → HuggingFace). None → use anonymous access."""
    try:
        from src.config.secrets.model import Secrets

        return Secrets().hf_token
    except Exception as exc:
        logger.info("[API:DATASETS] HF token unavailable: %s", exc)
        return None


__all__ = ["router"]
