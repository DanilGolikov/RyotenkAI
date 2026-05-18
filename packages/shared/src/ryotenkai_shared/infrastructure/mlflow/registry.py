"""MLflow Model Registry adapter implementing :class:`IModelRegistry`.

Concrete implementation of the alias-based promotion Protocol declared
in :mod:`ryotenkai_shared.infrastructure.mlflow.protocols`. Wraps the
official ``mlflow.MlflowClient`` registry surface (``register_model`` /
``set_registered_model_alias`` / ``get_model_version_by_alias``) under
a narrow typed interface so :class:`ModelPublisher` and the
``ryotenkai model promote`` CLI can depend on the Protocol without
pinning to a concrete client.

Aliases over stages
-------------------
MLflow's ``Staging`` / ``Production`` / ``Archived`` stages are
deprecated as of 2.9.0 and removed in 3.0. The supported mechanism is
registered-model aliases (movable pointers per name). Two aliases land
in production: ``challenger`` (set automatically on a successful
publish) and ``champion`` (promoted manually via the CLI gate).

See:

* ADR -- ``docs/plans/vectorized-fluttering-mist.md`` Phase M5
* Protocol -- :class:`IModelRegistry`
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ryotenkai_shared.utils.logger import get_logger

if TYPE_CHECKING:
    from ryotenkai_shared.infrastructure.mlflow.protocols import ModelVersion

logger = get_logger(__name__)


__all__ = ["MlflowModelRegistry", "MlflowModelVersion"]


@dataclass(frozen=True)
class MlflowModelVersion:
    """Concrete value object satisfying :class:`ModelVersion` Protocol.

    Holds the three read-only fields the Protocol exposes (the wider
    ``mlflow.entities.ModelVersion`` carries timestamps, source URI,
    tags, etc.). We narrow to the fields the publisher / CLI consume
    so test fakes stay trivial.
    """

    name: str
    version: str
    run_id: str | None


class MlflowModelRegistry:
    """Alias-based model registry against the real MLflow server.

    :param tracking_uri: Tracking URI of the MLflow server. Passed
        directly to :class:`mlflow.MlflowClient`. NEVER re-stamps
        ``mlflow.set_tracking_uri`` -- that singleton mutation belongs
        exclusively to :class:`MlflowTransport.__init__` (R-07).

    .. note::

        ``mlflow`` is lazy-imported so unit tests / CLI help screens
        without the package installed do not pay the ~400ms import
        cost. The class is therefore safe to construct in a thin
        composition root even when the real client cannot be reached.
    """

    def __init__(self, tracking_uri: str) -> None:
        if not tracking_uri or not tracking_uri.strip():
            msg = "MlflowModelRegistry.tracking_uri must be non-empty"
            raise ValueError(msg)
        self._tracking_uri = tracking_uri
        # Lazily constructed on first server-touching call.
        self._client: Any | None = None

    @property
    def tracking_uri(self) -> str:
        """The URI this registry was configured against (frozen)."""
        return self._tracking_uri

    # -- IModelRegistry surface -------------------------------------

    def register(self, model_uri: str, name: str) -> ModelVersion:
        """Register ``model_uri`` as a new version of ``name``.

        Delegates to :func:`mlflow.register_model`, which handles
        registered-model-creation on miss and returns the new
        :class:`mlflow.entities.ModelVersion`. We narrow the return
        type to :class:`MlflowModelVersion` so callers depend on the
        Protocol, not the wider mlflow proto.

        :param model_uri: Source URI (typically
            ``runs:/<run_id>/<artifact_path>``).
        :param name: Registered model name.
        :returns: :class:`MlflowModelVersion` describing the new version.
        :raises RuntimeError: When ``mlflow`` cannot be imported or
            the registration call fails.
        """
        mlflow = self._load_mlflow_module()
        logger.info(
            "[REGISTRY] register_model name=%s uri=%s tracking_uri=%s",
            name, model_uri, self._tracking_uri,
        )
        # ``register_model`` honours the global ``mlflow.set_tracking_uri``
        # stamp; ``MlflowTransport.__init__`` is responsible for that
        # stamping (the lint rule NO_SET_TRACKING_URI_GLOBAL guards it).
        # Calling ``register_model`` here is a read-against-stamped-URI
        # path -- no mutation.
        mv = mlflow.register_model(model_uri, name)
        return MlflowModelVersion(
            name=getattr(mv, "name", name),
            version=str(mv.version),
            run_id=getattr(mv, "run_id", None),
        )

    def set_alias(self, name: str, alias: str, version: str) -> None:
        """Point ``alias`` at ``version`` of registered model ``name``.

        :param name: Registered model name.
        :param alias: Alias label (``"challenger"`` / ``"champion"`` /
            user-defined).
        :param version: Existing version string.
        :raises RuntimeError: When the alias cannot be set (unknown
            version, server error, etc.).
        """
        client = self._get_client()
        logger.info(
            "[REGISTRY] set_alias name=%s alias=%s version=%s",
            name, alias, version,
        )
        client.set_registered_model_alias(name, alias, str(version))

    def resolve_alias(self, name: str, alias: str) -> ModelVersion:
        """Return the version currently pointed to by ``alias`` of ``name``.

        :param name: Registered model name.
        :param alias: Alias label.
        :returns: :class:`MlflowModelVersion` for the resolved version.
        :raises RuntimeError: When the alias is not registered.
        """
        client = self._get_client()
        mv = client.get_model_version_by_alias(name, alias)
        return MlflowModelVersion(
            name=getattr(mv, "name", name),
            version=str(mv.version),
            run_id=getattr(mv, "run_id", None),
        )

    # -- internals --------------------------------------------------

    def _get_client(self) -> Any:
        """Lazily construct the underlying :class:`MlflowClient`.

        This is the ONLY ``MlflowClient`` construction outside
        :class:`MlflowTransport` -- and it is allowlisted in
        ``scripts/lint/mlflow_rules.py`` (NO_AD_HOC_MLFLOW_CLIENT
        rule) accordingly.
        """
        if self._client is None:
            from mlflow.tracking import MlflowClient  # noqa: PLC0415

            self._client = MlflowClient(tracking_uri=self._tracking_uri)
        return self._client

    @staticmethod
    def _load_mlflow_module() -> Any:
        """Lazy import of ``mlflow`` (heavy side-effects in CI tests)."""
        import mlflow  # noqa: PLC0415

        return mlflow
