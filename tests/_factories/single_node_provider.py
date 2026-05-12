"""``make_single_node_provider`` — factory for a real :class:`SingleNodeProvider`.

The :class:`SingleNodeProvider` constructor requires a fully-formed
:class:`ProviderContext` (post-Phase-B refactor) — building it inline
in every test produced 80+ lines of duplicated boilerplate. This
factory wraps :func:`tests._fakes.provider_context.make_provider_context`
with sensible single-node defaults so tests can write:

>>> provider = make_single_node_provider()                # alias mode
>>> provider = make_single_node_provider(host="1.2.3.4")  # explicit mode

…and immediately exercise behaviour without dragging fixture
construction into the assertions.

Follows the same pattern as :mod:`tests._factories.pipeline_config`:

* Real objects (no ``MagicMock(spec=...)``) so production typed dispatch
  paths fire.
* Tiny, composable surface — one positional class, many keyword
  overrides.
* Defaults match the most common test shape so the override list stays
  short.
"""

from __future__ import annotations

from typing import Any

from ryotenkai_providers.single_node.training.provider import SingleNodeProvider
from ryotenkai_shared.config import Secrets

from tests._fakes.provider_context import attach_manifest_capabilities, make_provider_context

# Single attachment at import time — every test importing this factory
# automatically gets a SingleNodeProvider that knows its manifest
# identity. Avoids "provider_name returns empty string" surprises when
# tests run in isolation (no ProviderRegistry stamping).
attach_manifest_capabilities(
    SingleNodeProvider,
    provider_id="single_node",
    provider_name="single_node",
    provider_type="local",
)


def make_single_node_provider(
    *,
    host: str | None = None,
    user: str | None = None,
    port: int = 22,
    alias: str = "pc",
    workspace_path: str = "/workspace",
    mock_mode: bool = False,
    hf_token: str = "hf_test",
    extra_provider_block: dict[str, Any] | None = None,
) -> SingleNodeProvider:
    """Build a :class:`SingleNodeProvider` ready for unit-test exercise.

    Default shape is **alias mode** (``connect.ssh.alias = "pc"``)
    because that's the lowest-friction config for tests and matches the
    way local development is typically wired.

    Switch to **explicit mode** by passing ``host=...`` (and optionally
    ``user=`` / ``port=``); the alias is dropped and the ssh block
    becomes ``{host, user, port}``.

    Args:
        host: Switch to explicit SSH mode and use this host. When set,
            the alias is omitted from the config.
        user: Explicit-mode username. Ignored if ``host`` is None.
        port: Explicit-mode port. Ignored if ``host`` is None.
        alias: Alias-mode SSH alias (default ``"pc"``). Ignored if
            ``host`` is set.
        workspace_path: Remote workspace root.
        mock_mode: Forward to ``training.mock_mode`` (skips real GPU
            checks). Default ``False``; common in tests is ``True``.
        hf_token: HF token threaded into the :class:`Secrets` block.
        extra_provider_block: Optional shallow-merged additions to the
            provider block — useful for setting ``cleanup`` /
            ``inference`` sub-blocks without rebuilding the whole dict.

    Returns:
        A fully-constructed :class:`SingleNodeProvider`. The provider's
        SSH client / GPU info are unpopulated (i.e. ``connect()`` has
        not yet been called).
    """
    if host is None:
        connect_block: dict[str, Any] = {"ssh": {"alias": alias}}
    else:
        connect_block = {"ssh": {"host": host, "user": user or "user", "port": port}}

    provider_block: dict[str, Any] = {
        "connect": connect_block,
        "training": {"workspace_path": workspace_path, "mock_mode": mock_mode},
    }
    if extra_provider_block:
        provider_block.update(extra_provider_block)

    ctx = make_provider_context(
        provider_id="single_node",
        config=provider_block,
        secrets=Secrets(HF_TOKEN=hf_token),
    )
    return SingleNodeProvider(ctx)


__all__ = ["make_single_node_provider"]
