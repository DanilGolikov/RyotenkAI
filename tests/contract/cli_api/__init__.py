"""Greenfield CLI ↔ HTTP API parity contracts.

The legacy parity test at
``packages/control/tests/contract/test_cli_api_parity.py`` stays put.
This greenfield version uses :class:`Stack.control_plane()` to drive
both sides over real HTTP + a real CLI subprocess, surfacing drift
between the Typer command tree and the FastAPI route tree.
"""
