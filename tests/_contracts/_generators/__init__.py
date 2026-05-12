"""Schema generators.

Each ``gen_<name>.py`` script in this package writes a single JSON
Schema next to its sibling ``<name>_schema.json`` in
``tests/_contracts/``. The scripts are deliberately self-contained
(no shared CLI) so they can run independently when one production
model changes.

Run all of them via :mod:`tests._contracts._generators.regen_all`.
"""

from __future__ import annotations
