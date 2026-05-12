"""Run every ``gen_*.py`` generator. Idempotent.

CLI::

    .venv/bin/python -m tests._contracts._generators.regen_all
"""

from __future__ import annotations

import sys

from tests._contracts._generators import (
    gen_marker_schemas,
    gen_plugin_manifest,
    gen_runner_events,
    gen_runner_internal_events,
)


def main() -> int:
    rc = 0
    for module in (
        gen_plugin_manifest,
        gen_runner_internal_events,
        gen_marker_schemas,
        gen_runner_events,
    ):
        rc |= module.main()
    return rc


if __name__ == "__main__":
    sys.exit(main())
