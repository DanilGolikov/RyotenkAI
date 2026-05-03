"""HTTP + WebSocket surface of the runner.

Sub-modules:
- :mod:`jobs` — REST: ``POST /jobs``, ``GET /jobs/{id}``, ``POST /jobs/{id}/stop``.
- :mod:`events` — WebSocket ``/jobs/{id}/events?since=<offset>``.
- :mod:`internal` — loopback ``POST /internal/events`` (trainer → server).
"""

from __future__ import annotations
