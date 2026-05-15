"""Project → spawn-launch adapter (workspace boundary).

Resolves a project's filesystem layout into the inputs the launcher
needs to fork the pipeline worker:

    resolved = resolve_project_launch_inputs("my-project")
    extra_env = build_subprocess_extra_env(resolved)
    spawn_launch(launch_request, extra_env=extra_env, attach_stdio=True)

The adapter is pure: no env mutation, no orchestrator construction,
no side effects beyond reading from disk. Launcher (CLI / Web API)
combines the resolver output with a ``LaunchRequest`` and calls
``spawn_launch``; the spawned worker (``src/pipeline/worker.py``)
loads the YAML and runs the orchestrator in-subprocess.

Filesystem layout consumed::

    ~/.ryotenkai/projects/<id>/
      project.json          # registry-backed; on-disk id is authoritative
      configs/current.yaml  # → loaded inside the worker subprocess
      env.json              # → ProjectStore.read_env() → extra_env merge
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path

from ryotenkai_control.workspace.projects.registry import (
    ProjectRegistry,
    ProjectRegistryError,
)
from ryotenkai_control.workspace.projects.store import ProjectStore
from ryotenkai_shared.errors import ProjectNotFoundError as _SharedProjectNotFoundError


class ProjectNotFoundError(_SharedProjectNotFoundError):
    """Raised when a ``project_id`` doesn't resolve to a registered project.

    Phase C: inherits from the shared typed ``ProjectNotFoundError``
    (404, ``PROJECT_NOT_FOUND``) so the RFC 9457 problem+json
    contract converts it without an ad-hoc adapter. ``.project_id``
    is preserved for back-compat with call sites that introspect it.
    Surfaces as a clean ``die()`` in the CLI top-level handler.
    """

    def __init__(self, project_id: str, *, hint: str | None = None) -> None:
        msg = f"project not found: {project_id!r}"
        if hint:
            msg = f"{msg} ({hint})"
        super().__init__(msg, context={"project_id": project_id})
        self.project_id = project_id


@dataclass(frozen=True, slots=True)
class ResolvedProject:
    """Slim project descriptor for the launch path.

    Holds *paths and metadata only* — the YAML is loaded by the
    spawned worker (``src/pipeline/worker.py``), not here. Callers
    use this to build the subprocess command + env for
    ``spawn_launch(extra_env=...)``.

    Fields:

    * ``config_path`` — resolved YAML path (project's
      ``configs/current.yaml`` or the explicit override).
    * ``env`` — project-scoped env-var overrides from ``env.json``.
      Empty dict for projects without an env.json. Passed as
      ``extra_env`` to ``spawn_launch`` (merged on top of process env
      before fork).
    * ``metadata`` — invariants ``project_id`` / ``actor`` /
      ``config_version_hash`` plus optional
      ``config_override_path``. Pushed to the subprocess as
      ``RYOTENKAI_*`` env vars (the worker's bootstrap reads them and
      stamps them onto :class:`PipelineState`).
    * ``runs_base_dir`` — ``<project>/runs/``. Pushed as
      ``RYOTENKAI_RUNS_BASE_DIR`` so the subprocess'
      ``RuntimeSettings`` lands launches inside the project workspace.
    """

    config_path: Path
    runs_base_dir: Path
    env: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def resolve_project_launch_inputs(
    project_id: str,
    *,
    config_override: Path | None = None,
    actor: str | None = None,
    registry: ProjectRegistry | None = None,
) -> ResolvedProject:
    """Read project filesystem, return paths + env + metadata for spawn.

    Used by both CLI (``ryotenkai run start --project X``) and Web API
    when an explicit project_id is known.

    Args:
        project_id: Identifier as registered in
            ``~/.ryotenkai/projects.json``.
        config_override: Optional path to a YAML that REPLACES the
            project's ``configs/current.yaml``. Project's env.json and
            metadata still flow through.
        actor: Caller identity. Falls back to ``RYOTENKAI_ACTOR`` env
            var, then to OS user, then to ``"unknown"``.
        registry: Optional :class:`ProjectRegistry` for testing
            isolation.

    Returns:
        :class:`ResolvedProject`.

    Raises:
        ProjectNotFoundError: ``project_id`` not in the registry, or
            its directory is missing on disk, or the YAML is missing.
        ProjectStoreError: ``env.json`` malformed (bubbled unchanged).
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

    env = store.read_env()

    metadata: dict[str, str] = {
        "project_id": project_id,
        "actor": _resolve_actor(actor),
        "config_version_hash": _hash_yaml(config_path),
    }
    if config_override is not None:
        metadata["config_override_path"] = str(config_path)

    runs_base_dir = store.runs_dir
    runs_base_dir.mkdir(parents=True, exist_ok=True)

    return ResolvedProject(
        config_path=config_path,
        runs_base_dir=runs_base_dir,
        env=env,
        metadata=metadata,
    )


def resolve_project_launch_inputs_from_run_dir(
    run_dir: Path,
    *,
    registry: ProjectRegistry | None = None,
) -> ResolvedProject | None:
    """Walk up from ``run_dir`` to find a sibling ``project.json``.

    Used by Web API's launch endpoint and CLI ``run resume/restart``
    (when project context isn't passed explicitly but the run lives
    inside a project workspace).

    Returns ``None`` if the run is not inside any project workspace —
    the caller treats that as "ad-hoc, no metadata to inject".

    Note: project_id is read from the on-disk ``project.json`` (the
    current name on disk), NOT from the registry — so a project that
    was renamed between attempts re-acquires its current id rather
    than the stale one. Filesystem is authoritative for the run.
    """
    run_dir = run_dir.expanduser().resolve()
    for candidate in (run_dir, *run_dir.parents):
        if not (candidate / "project.json").is_file():
            continue
        store = ProjectStore(candidate)
        try:
            metadata_obj = store.load()
        except Exception:
            # Malformed project.json — treat as no project context.
            return None
        try:
            return resolve_project_launch_inputs(
                metadata_obj.id,
                registry=registry,
            )
        except ProjectNotFoundError:
            # Registry doesn't know this project (e.g. user moved
            # the dir without re-registering). Filesystem wins —
            # build a ResolvedProject from the on-disk store directly,
            # bypassing the registry lookup.
            return _resolved_from_store(store, metadata_obj.id)
    return None


def build_subprocess_extra_env(
    resolved: ResolvedProject | None,
    *,
    default_actor: str | None = None,
) -> dict[str, str]:
    """Convert a :class:`ResolvedProject` into the ``extra_env`` map for spawn.

    Combines project ``env.json`` overrides with the ``RYOTENKAI_*``
    metadata env vars that the spawned worker's bootstrap reads to
    populate :class:`PipelineState.metadata` and MLflow ``meta.*`` tags.

    Returns ``{}`` for ad-hoc runs (``resolved is None``) — the worker
    treats absence of ``RYOTENKAI_PROJECT_ID`` as "anonymous run".

    ``default_actor`` is used when ``resolved.metadata`` carries no
    actor (e.g. Web API path with no auth — pass ``"agent:web-ui"``).
    """
    if resolved is None:
        return {}
    extra: dict[str, str] = dict(resolved.env)
    extra["RYOTENKAI_PROJECT_ID"] = resolved.metadata["project_id"]
    actor = resolved.metadata.get("actor") or (default_actor or "")
    if actor:
        extra["RYOTENKAI_ACTOR"] = actor
    if resolved.metadata.get("config_version_hash"):
        extra["RYOTENKAI_CONFIG_VERSION_HASH"] = resolved.metadata["config_version_hash"]
    extra["RYOTENKAI_RUNS_BASE_DIR"] = str(resolved.runs_base_dir)
    if "config_override_path" in resolved.metadata:
        extra["RYOTENKAI_CONFIG_OVERRIDE_PATH"] = resolved.metadata["config_override_path"]
    return extra


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _resolved_from_store(store: ProjectStore, project_id: str) -> ResolvedProject:
    """Build a ResolvedProject directly from a ProjectStore.

    Used as a fallback when the registry has no entry for a project
    found via filesystem walk-up — keeps "moved without re-register"
    runs working.
    """
    config_path = store.current_config_path
    env = store.read_env()
    metadata: dict[str, str] = {
        "project_id": project_id,
        "actor": _resolve_actor(None),
        "config_version_hash": _hash_yaml(config_path),
    }
    runs_base_dir = store.runs_dir
    runs_base_dir.mkdir(parents=True, exist_ok=True)
    return ResolvedProject(
        config_path=config_path,
        runs_base_dir=runs_base_dir,
        env=env,
        metadata=metadata,
    )


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
    "ProjectNotFoundError",
    "ResolvedProject",
    "build_subprocess_extra_env",
    "resolve_project_launch_inputs",
    "resolve_project_launch_inputs_from_run_dir",
]
