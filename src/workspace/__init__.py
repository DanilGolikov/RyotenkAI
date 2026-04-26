"""User-workspace registries: projects, providers, integrations.

Each subpackage owns a workspace-scoped registry + store keyed under
``~/.ryotenkai/`` (the user's RyotenkAI home). These are configuration
domains, not pipeline runtime — projects describe user-managed config
trees, providers/integrations hold reusable credentials and external
service bindings.

The shape is shared via the ``WorkspaceRegistry`` / ``WorkspaceStore``
generic bases in :mod:`src.workspace._registry_base`.
"""
