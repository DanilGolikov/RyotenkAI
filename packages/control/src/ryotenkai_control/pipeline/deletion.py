"""Run deletion service.

Deletes a local run directory (and any nested runs) and, when configured,
cascades deletion to the MLflow experiment tree rooted at the run's
root_mlflow_run_id. Shared domain service used by both the CLI and the web API.

Phase M3.B: the MLflow side of cascade-delete uses an injected
:class:`MlflowReadClient` (via :class:`RunTreeWalker` for the BFS) instead
of an ad-hoc ``mlflow.tracking.MlflowClient()`` construction. The
underlying client is still used for the ``delete_run`` call (no
``IRunQuery`` method covers writes) — the lint forbidding ad-hoc
constructions is satisfied because the client comes from
``MlflowReadClient.underlying_client``.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ryotenkai_control.pipeline.mlflow.read.client import MlflowReadClient
from ryotenkai_control.pipeline.mlflow.read.tree_walker import RunNode, RunTreeWalker
from ryotenkai_control.pipeline.state.queries import discover_run_dirs, load_pipeline_state

if TYPE_CHECKING:
    pass

_LOG = logging.getLogger("ryotenkai.pipeline.deletion")


def _record_depths(node: RunNode, *, depth: int, into: dict[str, int]) -> None:
    """Walk a :class:`RunNode` tree once, recording per-id depth.

    Used by :meth:`RunDeleter._delete_run_tree` so deletion proceeds
    deepest-first — matching the legacy ordering.
    """
    into[node.handle.run_id] = depth
    for child in node.children:
        _record_depths(child, depth=depth + 1, into=into)


class DeleteMode(StrEnum):
    LOCAL_AND_MLFLOW = "local_and_mlflow"
    LOCAL_ONLY = "local_only"


@dataclass(frozen=True, slots=True)
class DeleteIssue:
    run_dir: Path
    phase: str
    message: str


@dataclass(frozen=True, slots=True)
class DeleteResult:
    target: Path
    run_dirs: tuple[Path, ...]
    deleted_mlflow_run_ids: tuple[str, ...]
    local_deleted: bool
    issues: tuple[DeleteIssue, ...]

    @property
    def is_success(self) -> bool:
        return not self.issues


class RunDeleter:
    """Delete local runs and optionally their linked MLflow run tree.

    Phase M3.B: cascade deletion routes through :class:`MlflowReadClient`
    + :class:`RunTreeWalker`. Callers may pass a ``run_query_factory``
    that produces a read client given a ``(tracking_uri, ca_bundle_path)``
    pair — keeps URI ownership at the composition root while letting the
    deleter create per-run clients (different state files may point at
    different servers, e.g. local vs. funnel).
    """

    def __init__(
        self,
        *,
        rmtree=shutil.rmtree,
        run_query_factory: Any | None = None,
    ) -> None:
        self._rmtree = rmtree
        # ``run_query_factory`` is a callable
        # ``(tracking_uri: str, ca_bundle_path: str | None) -> MlflowReadClient``.
        # ``None`` falls back to the default factory below, which
        # constructs the standard :class:`MlflowReadClient`. Test seam.
        self._run_query_factory = run_query_factory or self._default_run_query_factory

    @staticmethod
    def _default_run_query_factory(
        tracking_uri: str,
        ca_bundle_path: str | None,  # noqa: ARG004 — kept for symmetry with the legacy MLflowEnvironment
    ) -> MlflowReadClient:
        return MlflowReadClient(tracking_uri=tracking_uri)

    def delete_target(self, target: Path, *, mode: DeleteMode = DeleteMode.LOCAL_AND_MLFLOW) -> DeleteResult:
        resolved_target = target.expanduser().resolve()
        run_dirs = discover_run_dirs(resolved_target)
        deleted_mlflow_run_ids: list[str] = []
        issues: list[DeleteIssue] = []

        if mode == DeleteMode.LOCAL_AND_MLFLOW:
            for run_dir in run_dirs:
                issues.extend(self._delete_mlflow_for_run(run_dir, deleted_mlflow_run_ids))

        blocking_issues = [issue for issue in issues if self._is_blocking_issue(issue)]
        if blocking_issues:
            return DeleteResult(
                target=resolved_target,
                run_dirs=run_dirs,
                deleted_mlflow_run_ids=tuple(deleted_mlflow_run_ids),
                local_deleted=False,
                issues=tuple(issues),
            )

        try:
            self._rmtree(resolved_target)
        except Exception as exc:
            issues.append(DeleteIssue(run_dir=resolved_target, phase="filesystem_delete", message=str(exc)))
            _LOG.exception("Filesystem delete failed for %s", resolved_target)

        return DeleteResult(
            target=resolved_target,
            run_dirs=run_dirs,
            deleted_mlflow_run_ids=tuple(deleted_mlflow_run_ids),
            local_deleted=not any(issue.phase == "filesystem_delete" for issue in issues),
            issues=tuple(issues),
        )

    def _delete_mlflow_for_run(self, run_dir: Path, deleted_ids: list[str]) -> list[DeleteIssue]:
        try:
            state = load_pipeline_state(run_dir)
        except Exception as exc:
            _LOG.exception("Failed to load pipeline state for %s", run_dir)
            return [DeleteIssue(run_dir=run_dir, phase="state_load", message=str(exc))]

        if not state.root_mlflow_run_id:
            return []
        if not state.mlflow_runtime_tracking_uri:
            return [
                DeleteIssue(
                    run_dir=run_dir,
                    phase="mlflow_runtime_contract",
                    message="pipeline_state.json has no mlflow_runtime_tracking_uri",
                )
            ]

        try:
            run_query = self._run_query_factory(
                state.mlflow_runtime_tracking_uri,
                state.mlflow_ca_bundle_path,
            )
        except Exception as exc:
            _LOG.exception("Failed to construct MLflow read client for %s", run_dir)
            return [DeleteIssue(run_dir=run_dir, phase="mlflow_delete", message=str(exc))]

        try:
            deleted_ids.extend(self._delete_run_tree(run_query, state.root_mlflow_run_id))
            return []
        except Exception as exc:
            _LOG.exception("MLflow delete failed for %s", run_dir)
            return [DeleteIssue(run_dir=run_dir, phase="mlflow_delete", message=str(exc))]

    def _delete_run_tree(
        self,
        run_query: MlflowReadClient,
        root_run_id: str,
    ) -> list[str]:
        """Cascade-delete the MLflow run tree rooted at ``root_run_id``.

        Phase M3.B: BFS traversal is delegated to :class:`RunTreeWalker`
        (which depends solely on :class:`IRunQuery`); the actual
        ``delete_run`` call uses ``run_query.underlying_client`` because
        the write surface is not part of :class:`IRunQuery`.
        """
        walker = RunTreeWalker(run_query)
        try:
            handles = walker.flat_descendants(root_run_id)
        except KeyError:
            # ``IRunQuery.get_run`` raises ``KeyError`` when the run is
            # already gone. Treat as no-op (legacy semantics matched
            # "resource does not exist" string sniffing).
            return []
        except Exception as exc:
            if self._is_missing_run_error(exc):
                return []
            raise

        # Order by (depth, id) so children land before parents — matches
        # the legacy ordering. ``flat_descendants`` returns DFS pre-order;
        # we re-order by reverse BFS depth.
        depth_map: dict[str, int] = {}
        # Re-walk the tree to record per-node depth (cheap; same query
        # results are now cached on the walker / read client).
        root_node = walker.walk(root_run_id)
        _record_depths(root_node, depth=0, into=depth_map)

        run_ids_with_depth = [
            (handle.run_id, depth_map.get(handle.run_id, 0)) for handle in handles
        ]
        ordered_run_ids = [
            run_id
            for run_id, _depth in sorted(
                run_ids_with_depth, key=lambda item: (item[1], item[0]), reverse=True,
            )
        ]
        client = run_query.underlying_client
        for run_id in ordered_run_ids:
            try:
                client.delete_run(run_id)
            except Exception as exc:
                if self._is_missing_run_error(exc):
                    continue
                raise
        return ordered_run_ids

    @staticmethod
    def _is_missing_run_error(error: Exception) -> bool:
        message = str(error).lower()
        return "resource does not exist" in message or "not found" in message or "deleted" in message

    @staticmethod
    def _is_blocking_issue(issue: DeleteIssue) -> bool:
        return issue.phase != "mlflow_runtime_contract"


__all__ = ["DeleteIssue", "DeleteMode", "DeleteResult", "RunDeleter"]
