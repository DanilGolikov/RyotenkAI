"""MLflow infrastructure layer — narrow Protocols + transport.

Phase F retired the wide ``IMLflowManager`` / ``MLflowGateway`` /
``MLflowEnvironment`` / ``resolve_mlflow_uris`` surface. The
write-path is now expressed via narrow Protocols (see ``protocols``)
and concrete implementations (``transport``, ``registry``,
``journal_uploader``, ``prompt_registry``).

Module names re-exported by this package are deliberately kept thin
to avoid pulling the logger transitively (the logger imports
``ryotenkai_shared.config`` which imports
``mlflow_project`` which in turn imports
``infrastructure.mlflow.config`` -- a re-export loop here would cause
a circular import). Callers should import concrete classes from their
own module paths.
"""

__all__: list[str] = []
