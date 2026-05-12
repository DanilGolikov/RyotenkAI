"""Sentinel: protocol discovery in ``test_no_protocol_mocking.py`` stays healthy.

The no-Protocol-mocking sentinel discovers `Protocol`-decorated classes
dynamically by walking `packages/*/src/`. If that discovery regresses
(e.g. someone changes the AST walk or the seed list), the sentinel
silently allows mocks. This meta-test guards the discovery itself.

Invariants:
1. **Count floor**: at least 30 protocols are discovered. Allows
   growth (currently 38+) but catches regressions where discovery
   silently drops to 0.
2. **Anchor protocols present**: well-known Protocol names that we
   know exist in production MUST appear in the discovered set. If they
   disappear, either the protocol was renamed (update this list) or
   discovery is broken.
"""

from __future__ import annotations

import importlib

# Known Protocol names that production code defines. If a refactor
# renames one of these, update the list here AND verify the rename was
# intentional. The intent is to catch *silent regressions* in discovery.
ANCHOR_PROTOCOLS = (
    "IMLflowManager",
    "IPodLifecycleClient",
    "IDockerClient",
    "ISSHClient",
    "IHFHubClient",
    "IJobClient",
    "IRunPodAPI",
    "ITrainerSpawner",
)


def test_protocol_discovery_count_floor() -> None:
    """At least 30 protocols are discovered. Catches discovery regressions."""
    mod = importlib.import_module("tests._lint.test_no_protocol_mocking")
    discovered = mod._discover_protocols()
    assert len(discovered) >= 30, (
        f"Protocol discovery returned only {len(discovered)} protocols; "
        f"expected ≥ 30. Either many Protocols were deleted (verify by "
        f"checking git log) or discovery is broken. Found: "
        f"{sorted(discovered)[:20]}"
    )


def test_anchor_protocols_are_discovered() -> None:
    """Known production Protocols must appear in the discovered set."""
    mod = importlib.import_module("tests._lint.test_no_protocol_mocking")
    discovered = mod._discover_protocols()
    missing = [name for name in ANCHOR_PROTOCOLS if name not in discovered]
    assert not missing, (
        f"Anchor protocols missing from discovery: {missing}.\n"
        f"Either they were renamed (update ANCHOR_PROTOCOLS in this file) "
        f"or discovery is broken. Discovered: {sorted(discovered)}"
    )
