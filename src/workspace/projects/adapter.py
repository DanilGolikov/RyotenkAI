"""Project → orchestrator adapter (Variant 1 hexagonal boundary).

Reads everything the orchestrator needs out of a project's filesystem
layout and returns a frozen :class:`ProjectInputs` value object — the
core engine never imports anything from this module, the contract flows
the other way (callers translate "project context" into the pure
``(config, env, metadata)`` triple core understands).

Single public entry point:

    inputs = load_project_inputs("my-project", actor="cli")
    PipelineOrchestrator(config=inputs.config,
                         env=inputs.env,
                         metadata=inputs.metadata).run()

Filesystem layout consumed::

    ~/.ryotenkai/projects/<id>/
      project.json          # registry-backed; NOT read here
      configs/current.yaml  # → loaded via load_pipeline_config (resolves integrations)
      env.json              # → ProjectStore.read_env() → adapter env

The adapter is pure: no env mutation, no orchestrator construction, no
side effects beyond reading from disk. That's the point — callers can
decide whether to spawn a subprocess (Web-API), invoke the orchestrator
in-process (CLI), or pass the inputs along to a remote runner (future).
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.workspace.integrations.loader import load_pipeline_config
from src.workspace.projects.registry import (
    ProjectRegistry,
    ProjectRegistryError,
)
from src.workspace.projects.store import ProjectStore

if TYPE_CHECKING:
    from src.utils.config import PipelineConfig


class ProjectNotFoundError(LookupError):
    """Raised when a ``project_id`` doesn't resolve to a registered project.

    Surfaces as a clean ``die()`` in the CLI top-level handler — same
    pattern as ``IntegrationNotFoundError`` from Step 1.
    """

    def __init__(self, project_id: str, *, hint: str | None = None) -> None:
        msg = f"project not found: {project_id!r}"
        if hint:
            msg = f"{msg} ({hint})"
        super().__init__(msg)
        self.project_id = project_id


@dataclass(frozen=True, slots=True)
class ProjectInputs:
    """Frozen orchestrator inputs derived from a project's filesystem.

    Fields populated in :func:`load_project_inputs`:

    * ``config`` — fully-loaded :class:`PipelineConfig` with integration
      refs already resolved (``load_config`` runs the resolver as its
      final step).
    * ``env`` — project-specific env-var overrides (the JSON object the
      user typed in Settings → Env). Empty dict for projects without
      ``env.json``. Never includes process env — callers are expected
      to merge process env on top if they want that semantics.
    * ``metadata`` — invariants: ``project_id``, ``actor``,
      ``config_version_hash``. Optional caller-supplied keys flow
      through unchanged.
    * ``runs_base_dir`` — ``<project>/runs/``. Caller hands this to
      ``RuntimeSettings.runs_base_dir`` so launches from this project
      land **inside** the project workspace
      (``<project>/runs/<run_id>/``) instead of a global runs dir.
      Authoritative state for the run still lives in
      ``<run_dir>/pipeline_state.json``; project ownership is inferred
      from the directory location AND ``metadata.project_id``.
    """

    config: PipelineConfig
    runs_base_dir: Path
    env: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_project_inputs(
    project_id: str,
    *,
    config_override: Path | None = None,
    actor: str | None = None,
    extra_metadata: dict[str, Any] | None = None,
    registry: ProjectRegistry | None = None,
) -> ProjectInputs:
    """Read a project's filesystem and produce orchestrator inputs.

    Args:
        project_id: Project identifier as registered in
            ``~/.ryotenkai/projects.json``. Raises
            :class:`ProjectNotFoundError` when missing.
        config_override: Optional path to a YAML config that
            REPLACES the project's ``configs/current.yaml``. The
            project's ``env.json`` and ``metadata`` still flow through
            — useful for "run an experimental config inside this
            project's context". When ``None``, the project's own
            ``configs/current.yaml`` is used.
        actor: Caller identity for audit metadata. When ``None``, falls
            back to ``RYOTENKAI_ACTOR`` env var, then to the OS user.
        extra_metadata: Extra keys merged into ``metadata`` (without
            overriding the invariants ``project_id`` /
            ``config_version_hash`` / ``actor`` — those always win).
        registry: Optional :class:`ProjectRegistry` for testing
            isolation. When ``None`` a default-rooted registry is
            constructed; callers normally don't need to pass this.

    Returns:
        :class:`ProjectInputs` ready to feed into
        :class:`PipelineOrchestrator`.

    Raises:
        ProjectNotFoundError: ``project_id`` not in the registry, or
            its directory is missing on disk.
        ProjectStoreError: ``env.json`` malformed (bubbled unchanged).
        IntegrationNotFoundError / IntegrationUnresolvedError: from
            :func:`load_pipeline_config` when the project's YAML
            references an unregistered integration.
    """
    reg = registry if registry is not None else ProjectRegistry()
    try:
        entry = reg.resolve(project_id)
    except ProjectRegistryError as exc:
        raise ProjectNotFoundError(
            project_id,
            hint="run `ryotenkai project list` to see registered projects",
        ) from exc

    store = ProjectStore(Path(entry.path))
    if not store.exists():
        raise ProjectNotFoundError(
            project_id,
            hint=(
                f"registry points to {store.root} but the directory is missing — "
                "either restore it from backup or `project rm` the stale entry"
            ),
        )

    config_path = (
        config_override.expanduser().resolve()
        if config_override is not None
        else store.current_config_path
    )
    if not config_path.is_file():
        raise ProjectNotFoundError(
            project_id,
            hint=(
                f"no config at {config_path}; save one via the Web UI "
                "or pass --config explicitly"
            ),
        )

    # ``load_pipeline_config`` runs the UX-layer integration resolver
    # before Pydantic validation, so the returned ``PipelineConfig``
    # has every ``integration: <id>`` shorthand inlined.
    # ``IntegrationNotFoundError`` / ``IntegrationUnresolvedError``
    # surface to the caller — the CLI layer's top-level handler
    # renders them as clean ``die()`` errors.
    config = load_pipeline_config(config_path)

    # ProjectStore.read_env raises ProjectStoreError when env.json is
    # malformed; let that surface — caller renders it cleanly.
    env = store.read_env()

    metadata: dict[str, Any] = dict(extra_metadata) if extra_metadata else {}
    # Invariant keys win over caller-supplied ones — these reflect ground
    # truth about the run's origin and shouldn't be spoofable by a
    # caller passing a colliding ``project_id`` in extras.
    metadata["project_id"] = project_id
    metadata["actor"] = _resolve_actor(actor)
    metadata["config_version_hash"] = _hash_yaml(config_path)
    if config_override is not None:
        # Useful breadcrumb for debugging "why did my run pick that
        # YAML?" — the override file path is recorded so the lineage
        # is reconstructible after the fact.
        metadata["config_override_path"] = str(config_path)

    # Project's runs go inside its own workspace directory. Caller threads
    # this through ``RuntimeSettings.runs_base_dir`` so the orchestrator
    # creates ``<project>/runs/<run_id>/`` rather than a global location.
    runs_base_dir = store.runs_dir
    runs_base_dir.mkdir(parents=True, exist_ok=True)

    return ProjectInputs(
        config=config,
        runs_base_dir=runs_base_dir,
        env=env,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _resolve_actor(explicit: str | None) -> str:
    """Pick the actor identity for metadata.

    Precedence: explicit arg > ``RYOTENKAI_ACTOR`` env var > OS user >
    ``"unknown"``. Agents (Claude Code, CI bots, future LLM workflows)
    are expected to set ``RYOTENKAI_ACTOR=agent:<name>`` so audit logs
    can distinguish them from human runs.
    """
    if explicit and explicit.strip():
        return explicit.strip()
    env_actor = os.environ.get("RYOTENKAI_ACTOR")
    if env_actor and env_actor.strip():
        return env_actor.strip()
    user = os.environ.get("USER") or os.environ.get("USERNAME")
    if user and user.strip():
        return user.strip()
    return "unknown"


def _hash_yaml(path: Path) -> str:
    """SHA-256 hex of the YAML file's bytes — for run-level traceability.

    Distinct from :func:`src.pipeline.state.store.hash_payload`: this
    hash captures the *file as written*, including comments and
    whitespace, while ``hash_payload`` hashes the resolved config tree
    (used for drift detection). Both have a place; this one answers
    "what literal YAML did the user click run on?".

    Returns ``""`` if the file can't be read (defensive — adapter
    shouldn't crash a launch over an audit-trail nice-to-have).
    """
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


__all__ = [
    "ProjectInputs",
    "ProjectNotFoundError",
    "load_project_inputs",
]
