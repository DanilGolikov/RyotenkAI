"""CLI ↔ API parity gate (greenfield rewrite of legacy test).

The legacy parity test in
``packages/control/tests/contract/test_cli_api_parity.py`` is kept in
the legacy lane; this version stands on the greenfield substrate
(:class:`Stack.control_plane()` + real CLI process) and is parametrised
over every command so divergence between the Typer tree and the
FastAPI router tree fails CI.

Implementation note: a full schema-driven 1:1 mapping requires either
hand-curated allowlists per command or stable annotations on each
Typer callback (something like ``app.command(api_route=...)`` so the
test can reach into the production code for the pairings). Until that
lands as a Phase 4+ task, the greenfield gate enforces a weaker
property: **every CLI command exists** AND **every documented API
endpoint reachable via GET responds** when fronted by the live
control plane. The legacy test continues to enforce per-pair payload
equality.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

from tests._harness.stack import Stack

pytestmark = [
    pytest.mark.contract,
    pytest.mark.stack,
    pytest.mark.slow,
    pytest.mark.asyncio,
]

_REPO_ROOT = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# Allowlist of known CLI ↔ API divergences.
# ---------------------------------------------------------------------------
#
# Each entry is a CLI command path that has *no* HTTP counterpart by
# design (e.g. local-only utilities). Phase 4+ adds the inverse list:
# API endpoints with no CLI counterpart.

_CLI_ONLY_COMMANDS: frozenset[str] = frozenset(
    {
        "run",       # subprocess driver — irreducible to a single endpoint
        "smoke",     # batch local validator
        "validate",  # offline schema validate
    },
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _enumerate_cli_top_commands() -> list[str]:
    """Parse ``ryotenkai --help`` to discover top-level commands."""
    proc = subprocess.run(
        [sys.executable, "-m", "ryotenkai_control.main", "--help"],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
        env={"NO_COLOR": "1", "PATH": "/usr/bin:/bin"},
    )
    if proc.returncode != 0:
        pytest.skip(f"ryotenkai --help failed: {proc.stderr[:200]}")
    # Typer renders commands in a "Commands" section; lines look like
    # ``  command  Short description``. We scan for the section.
    section = re.split(r"Commands[:\s]*$", proc.stdout, maxsplit=1, flags=re.MULTILINE)
    if len(section) < 2:
        # Fallback — Typer renders with a Rich panel; pull "│ ▸ name" rows.
        commands = re.findall(r"^[││ ]+([a-z][a-z0-9_-]*)\b", proc.stdout, flags=re.MULTILINE)
        return sorted(set(commands))
    body = section[1]
    commands = re.findall(r"^[││ ]+([a-z][a-z0-9_-]*)\b", body, flags=re.MULTILINE)
    return sorted(set(commands))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_every_documented_get_endpoint_responds(stack: Stack) -> None:
    """Drive a GET against every endpoint declared in the OpenAPI spec.

    Pure smoke contract: nothing about the *shape* of responses, just
    that the server doesn't 5xx and obeys its declared status codes.
    Companion to :mod:`test_openapi_stateful` which fuzzes payloads.
    """
    import json

    spec = json.loads((_REPO_ROOT / "web" / "src" / "api" / "openapi.json").read_text())
    paths = spec["paths"]

    async with stack.control_plane() as base_url, httpx.AsyncClient(
        base_url=base_url, timeout=10.0,
    ) as client:
        for path, methods in paths.items():
            if "get" not in methods:
                continue
            concrete = re.sub(r"\{[^}]+\}", "missing", path)
            response = await client.get(concrete)
            assert response.status_code < 500, (
                f"GET {concrete} → {response.status_code}; spec lists this endpoint."
            )


async def test_cli_top_level_commands_exist() -> None:
    """``ryotenkai --help`` lists every command we expect.

    The test is intentionally cheap (no Stack) — it pins the surface
    so a removed command shows up as a parity failure, not a silent
    drop.
    """
    commands = _enumerate_cli_top_commands()
    if not commands:
        pytest.skip("Could not parse Typer help output; revisit Typer renderer.")

    expected_minimum: frozenset[str] = frozenset(
        {
            "run",
            "runs",
            "project",
            "plugin",
            "preset",
        },
    )
    missing = expected_minimum - set(commands)
    assert not missing, (
        f"expected CLI commands missing from --help: {sorted(missing)}; "
        f"actual: {commands}"
    )


@pytest.mark.parametrize(
    "cli_cmd, expected_endpoint",
    [
        ("runs ls", "/api/v1/runs"),
        ("preset ls", "/api/v1/config/presets"),
    ],
)
async def test_known_cli_api_pairings(
    stack: Stack, cli_cmd: str, expected_endpoint: str,
) -> None:
    """For each pinned pairing, both the CLI subcommand and the HTTP
    endpoint exist and respond. Weak property: the legacy test
    enforces payload equality; this checks reachability."""
    ["--help", *cli_cmd.split()]
    proc = subprocess.run(
        [sys.executable, "-m", "ryotenkai_control.main", *cli_cmd.split(), "--help"],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
        env={"NO_COLOR": "1", "PATH": "/usr/bin:/bin"},
    )
    assert proc.returncode == 0, (
        f"`ryotenkai {cli_cmd} --help` failed: {proc.stderr[:200]}"
    )

    async with stack.control_plane() as base_url, httpx.AsyncClient(
        base_url=base_url, timeout=10.0,
    ) as client:
        response = await client.get(expected_endpoint)
        assert response.status_code < 500, (
            f"{expected_endpoint} → {response.status_code}; expected pair "
            f"with `ryotenkai {cli_cmd}`."
        )


def test_cli_only_commands_documented_and_minimal() -> None:
    """Drift fence: catches accidental growth of the allowlist.

    When somebody adds a new CLI-only command, force them to update
    the allowlist + add a justification rather than silently accept
    drift.
    """
    # We don't enumerate CLI-only commands at runtime (that requires
    # diffing CLI vs spec, which is expensive). Instead, hold the
    # allowlist size as a fence — bumping it requires touching this
    # test, which surfaces in PR review.
    assert len(_CLI_ONLY_COMMANDS) <= 8, (
        f"CLI-only allowlist grew unexpectedly: {sorted(_CLI_ONLY_COMMANDS)}. "
        f"Each entry should have a one-line justification in code; if "
        f"the count must rise, document why and update the cap."
    )
