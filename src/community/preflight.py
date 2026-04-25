"""Pre-launch gate for community plugins — env availability + instance shape.

Walks an already-validated :class:`PipelineConfig`, collects every
plugin its stages will actually instantiate (validation / evaluation /
reward / reports), looks up each plugin's manifest in the catalog,
and reports two classes of problems:

1. **Missing envs**: any non-optional ``[[required_env]]`` entry whose
   value is not set in the merged secret + env environment
   (:class:`MissingEnv` rows).
2. **Instance shape violations**: any ``params`` / ``thresholds``
   block in the YAML config that doesn't satisfy the plugin's
   ``params_schema`` / ``thresholds_schema``
   (:class:`InstanceValidationError` rows from
   :mod:`src.community.instance_validator`). Catches
   type/enum/range/required-field mistakes that the Configure modal
   already prevents in the UI but hand-edited YAML can still produce.

Wired in two places:

- ``POST /plugins/preflight`` (API): the Launch modal calls it before
  enabling the launch button so the user sees a "set up before launch"
  chip with deep-link hints (managed_by → Settings tab) instead of a
  4-minute training run that crashes on a missing key.
- :class:`PipelineOrchestrator` (runtime): same check is rerun before
  forking the worker subprocess so a YAML-driven launch (no UI) still
  fails fast with a readable message instead of mid-pipeline.

The check is **read-only** — it never instantiates plugins, never
opens network connections, and never touches the file system beyond
the one catalog scan ``ensure_loaded`` does.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple

from src.community.instance_validator import (
    InstanceValidationError,
    validate_instance,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from src.community.manifest import PluginKind, PluginManifest
    from src.config.secrets.model import Secrets
    from src.utils.config import PipelineConfig


class _PluginRef(NamedTuple):
    """Internal helper: ties a plugin's identity to its YAML payload.

    Used by both the env gate (which ignores ``params``/``thresholds``)
    and the instance-shape gate (which ignores them only when the
    manifest has no schema). Kept narrow on purpose — this is a private
    enumeration shape, not part of the public API.
    """

    kind: PluginKind
    plugin_name: str
    instance_id: str
    params: dict[str, Any]
    thresholds: dict[str, Any]


@dataclass(frozen=True, slots=True)
class MissingEnv:
    """One required env that the preflight gate couldn't resolve.

    The UI banner uses ``managed_by`` to deep-link the user into the
    right Settings tab (HF Integrations / Provider workspaces) rather
    than nudging them to edit ``env.json`` for a credential the system
    already manages elsewhere.
    """

    plugin_kind: PluginKind
    plugin_name: str  # registered plugin id (e.g. ``"cerebras_judge"``)
    plugin_instance_id: str  # YAML-level id (``id`` field on the config row)
    name: str
    description: str
    secret: bool
    managed_by: str  # "" | "integrations" | "providers"


# ---------------------------------------------------------------------------
# Plugin enumeration — one helper per kind, all yield :class:`_PluginRef`.
# ---------------------------------------------------------------------------


def _enabled_validation_plugins(
    config: PipelineConfig,
) -> Iterable[_PluginRef]:
    for ds_id, ds in (config.datasets or {}).items():
        validations = getattr(ds, "validations", None)
        if validations is None:
            continue
        for plugin_cfg in getattr(validations, "plugins", None) or []:
            instance_id = f"{ds_id}.{plugin_cfg.id}"
            yield _PluginRef(
                kind="validation",
                plugin_name=plugin_cfg.plugin,
                instance_id=instance_id,
                params=dict(plugin_cfg.params or {}),
                thresholds=dict(plugin_cfg.thresholds or {}),
            )


def _enabled_evaluation_plugins(
    config: PipelineConfig,
) -> Iterable[_PluginRef]:
    eval_cfg = getattr(config, "evaluation", None)
    if eval_cfg is None or not getattr(eval_cfg, "enabled", False):
        return
    evaluators = getattr(eval_cfg, "evaluators", None)
    if evaluators is None:
        return
    for plugin_cfg in getattr(evaluators, "plugins", None) or []:
        if not getattr(plugin_cfg, "enabled", True):
            continue
        yield _PluginRef(
            kind="evaluation",
            plugin_name=plugin_cfg.plugin,
            instance_id=plugin_cfg.id,
            params=dict(plugin_cfg.params or {}),
            thresholds=dict(plugin_cfg.thresholds or {}),
        )


def _enabled_reward_plugins(
    config: PipelineConfig,
) -> Iterable[_PluginRef]:
    training = getattr(config, "training", None)
    if training is None:
        return
    for idx, phase in enumerate(getattr(training, "strategies", None) or []):
        phase_params = getattr(phase, "params", None) or {}
        plugin_name = str(phase_params.get("reward_plugin") or "").strip()
        if not plugin_name:
            continue
        # Reward plugins consume ``reward_params`` (a sub-dict) — the
        # outer ``phase.params`` carries TRL config that the trainer
        # interprets, not the plugin. Mirror what
        # ``build_reward_plugin_result`` does at runtime so we validate
        # exactly what the plugin will receive.
        reward_params = phase_params.get("reward_params") or {}
        if not isinstance(reward_params, dict):
            reward_params = {}
        instance_id = f"{phase.strategy_type}#{idx}"
        yield _PluginRef(
            kind="reward",
            plugin_name=plugin_name,
            instance_id=instance_id,
            params=dict(reward_params),
            thresholds={},
        )


def _enabled_report_plugins(
    config: PipelineConfig,
) -> Iterable[_PluginRef]:
    """Enumerate report plugin ids from ``reports.sections``.

    ``sections=None`` means "use the built-in default" — preflight
    queries the registry for that default rather than re-importing the
    constant module to keep this file decoupled. Reports take no
    constructor kwargs, so ``params`` / ``thresholds`` are always
    empty.
    """
    from src.reports.plugins.defaults import DEFAULT_REPORT_SECTIONS

    reports_cfg = getattr(config, "reports", None)
    sections: tuple[str, ...] = (
        tuple(reports_cfg.sections)
        if reports_cfg is not None and reports_cfg.sections is not None
        else DEFAULT_REPORT_SECTIONS
    )
    for plugin_id in sections:
        # Reports use the plugin id as the instance id — there's no
        # ``id``-vs-``plugin`` distinction (one section is one plugin).
        yield _PluginRef(
            kind="reports",
            plugin_name=plugin_id,
            instance_id=plugin_id,
            params={},
            thresholds={},
        )


# ---------------------------------------------------------------------------
# Manifest → required envs filter
# ---------------------------------------------------------------------------


def _required_envs(manifest: PluginManifest) -> list[tuple[str, str, bool, str]]:
    """Return ``(name, description, secret, managed_by)`` for non-optional envs.

    Optional envs are **not** part of the gate — the UI surfaces them
    so the user can override defaults but the pipeline runs without
    them. Same applies for envs marked ``managed_by="integrations" |
    "providers"`` whose values come from a Settings workspace token
    rather than ``env.json`` — those are checked per-resource by the
    Settings layer, not here.
    """
    return [
        (spec.name, spec.description, spec.secret, spec.managed_by)
        for spec in manifest.required_env
        if not spec.optional and not spec.managed_by
    ]


# ---------------------------------------------------------------------------
# Env / secrets resolution
# ---------------------------------------------------------------------------


def _build_value_lookup(
    secrets: Secrets | None,
    project_env: Mapping[str, str] | None,
) -> dict[str, str]:
    """Merge process env + secrets.model_extra + project_env into one dict.

    Lower-precedence sources first, higher-precedence sources later —
    this matches the launcher's actual merge order: process env from
    the operator's shell is the floor, ``secrets.env`` is layered on
    top, and the project's ``env.json`` is layered on top of that.

    Lookups are case-sensitive for env names (``EVAL_X`` and ``eval_x``
    are different) but the secrets ``model_extra`` storage normalises
    keys to lowercase, so we re-uppercase known plugin namespaces when
    folding it in.
    """
    merged: dict[str, str] = {}

    # 1) Process env (lowest precedence) — only non-empty entries.
    for k, v in os.environ.items():
        if v and v.strip():
            merged[k] = v

    # 2) Pydantic Secrets — typed fields + model_extra arbitrary keys.
    if secrets is not None:
        for typed_name, value in (
            ("HF_TOKEN", getattr(secrets, "hf_token", None)),
            ("RUNPOD_API_KEY", getattr(secrets, "runpod_api_key", None)),
        ):
            if value:
                merged[typed_name] = str(value)
        for key, value in (secrets.model_extra or {}).items():
            if value is None or not str(value).strip():
                continue
            # Plugin secrets are stored lowercase in model_extra; the
            # canonical env name is uppercase. Surface both spellings
            # so a manifest declaring ``EVAL_X`` matches the lowercase
            # storage as well.
            merged[key.upper()] = str(value)

    # 3) Project env (highest precedence) — wins over anything above.
    if project_env:
        for k, v in project_env.items():
            if v is not None and str(v).strip():
                merged[k] = str(v)

    return merged


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate_required_env(
    config: PipelineConfig,
    *,
    secrets: Secrets | None = None,
    project_env: Mapping[str, str] | None = None,
) -> list[MissingEnv]:
    """Return per-plugin missing-env rows for ``config``.

    Empty list ⇒ launch is safe. Non-empty ⇒ caller decides whether to
    block (pipeline orchestrator) or surface (API endpoint for UI).

    Plugins referenced in ``config`` but not registered in the catalog
    are *silently skipped* — the validation/runner already produces a
    clear "plugin not found" error at the right point in the lifecycle,
    and re-raising here would shadow it with a misleading "missing env"
    message.
    """
    from src.community.catalog import catalog
    from src.community.manifest import PluginKind as _PluginKindType  # noqa: F401

    catalog.ensure_loaded()
    lookup = _build_value_lookup(secrets, project_env)

    enumerators = (
        _enabled_validation_plugins,
        _enabled_evaluation_plugins,
        _enabled_reward_plugins,
        _enabled_report_plugins,
    )

    missing: list[MissingEnv] = []
    seen_keys: set[tuple[str, str, str]] = set()  # (kind, instance_id, name)

    for enumerate_fn in enumerators:
        for ref in enumerate_fn(config):
            try:
                loaded = catalog.get(ref.kind, ref.plugin_name)
            except KeyError:
                # Unknown plugin — let the runtime path surface that.
                continue
            for name, description, secret, managed_by in _required_envs(loaded.manifest):
                if name in lookup:
                    continue
                key = (ref.kind, ref.instance_id, name)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                missing.append(MissingEnv(
                    plugin_kind=ref.kind,
                    plugin_name=ref.plugin_name,
                    plugin_instance_id=ref.instance_id,
                    name=name,
                    description=description,
                    secret=secret,
                    managed_by=managed_by,
                ))

    return missing


def validate_instances(
    config: PipelineConfig,
) -> list[InstanceValidationError]:
    """Validate every plugin instance's ``params`` / ``thresholds``
    block against its manifest schema.

    Catches type/enum/range/required-field violations that the
    Configure modal already prevents in the UI but hand-edited YAML
    can still produce. Returns a flat list across all plugin kinds —
    each row carries enough context (``plugin_kind``, ``plugin_name``,
    ``plugin_instance_id``, ``location``) for the UI to deep-link
    into the right Configure modal field.

    Plugins not registered in the catalog are silently skipped — same
    rationale as :func:`validate_required_env` (the runtime path will
    surface a clearer "plugin not found" message).
    """
    from src.community.catalog import catalog

    catalog.ensure_loaded()

    enumerators = (
        _enabled_validation_plugins,
        _enabled_evaluation_plugins,
        _enabled_reward_plugins,
        _enabled_report_plugins,
    )

    errors: list[InstanceValidationError] = []
    for enumerate_fn in enumerators:
        for ref in enumerate_fn(config):
            try:
                loaded = catalog.get(ref.kind, ref.plugin_name)
            except KeyError:
                continue
            errors.extend(validate_instance(
                loaded.manifest,
                plugin_kind=ref.kind,
                plugin_name=ref.plugin_name,
                plugin_instance_id=ref.instance_id,
                params=ref.params,
                thresholds=ref.thresholds,
            ))
    return errors


@dataclass(frozen=True, slots=True)
class PreflightReport:
    """Combined env + instance-shape preflight result.

    Both lists are populated in a single catalog scan so callers don't
    pay for two ``ensure_loaded`` round-trips. ``ok`` is True only when
    *both* lists are empty.
    """

    missing_envs: list[MissingEnv]
    instance_errors: list[InstanceValidationError]

    @property
    def ok(self) -> bool:
        return not self.missing_envs and not self.instance_errors


def run_preflight(
    config: PipelineConfig,
    *,
    secrets: Secrets | None = None,
    project_env: Mapping[str, str] | None = None,
) -> PreflightReport:
    """Run both env and instance-shape gates in one pass.

    Convenience wrapper used by the API endpoint and the pipeline
    bootstrap so the two checks share the same catalog-loaded state
    and surface together. Individual functions remain available for
    callers that only need one half.
    """
    return PreflightReport(
        missing_envs=validate_required_env(
            config, secrets=secrets, project_env=project_env
        ),
        instance_errors=validate_instances(config),
    )


class LaunchAbortedError(RuntimeError):
    """Raised by the orchestrator when preflight finds problems.

    Carries the structured rows so the caller (CLI, API, pipeline log)
    can render the same message format every time. Constructible with
    a missing-envs list (back-compat with PR7), an instance-errors
    list, or both.
    """

    def __init__(
        self,
        missing: list[MissingEnv] | None = None,
        instance_errors: list[InstanceValidationError] | None = None,
    ) -> None:
        self.missing = missing or []
        self.instance_errors = instance_errors or []
        bits: list[str] = []
        if self.missing:
            env_names = ", ".join(f"{m.plugin_name}:{m.name}" for m in self.missing)
            bits.append(f"{len(self.missing)} required env(s) unset — {env_names}")
        if self.instance_errors:
            shape_names = ", ".join(
                f"{e.plugin_name}:{e.location}" for e in self.instance_errors
            )
            bits.append(
                f"{len(self.instance_errors)} instance shape error(s) — {shape_names}"
            )
        super().__init__("refusing to launch: " + "; ".join(bits))


__all__ = [
    "InstanceValidationError",
    "LaunchAbortedError",
    "MissingEnv",
    "PreflightReport",
    "run_preflight",
    "validate_instances",
    "validate_required_env",
]
