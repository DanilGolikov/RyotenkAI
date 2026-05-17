"""Upcaster type definitions for schema evolution.

An upcaster is a pure function that migrates a raw envelope dict from
schema version ``from_version`` to ``to_version`` — one hop at a time.
Chains are composed by the registry runner (see :mod:`.__init__`).

Conventions enforced by code review (not by code):

* Pure: no IO, no logger, no datetime.now(), no random.
* Idempotent: applying the same hop twice with the same args yields the
  same dict.
* One hop at a time: an upcaster handles ``(N, N+1)``; multi-hop
  migrations compose hops in the registry.
* Never mutate the input dict in place — return a new (possibly partially
  copied) dict. The runner trusts the contract.

A semantic break (field removal, meaning change) is NOT a schema
migration — it requires a NEW dotted ``type`` instead. See the upcasters
README for the policy.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

# (raw_envelope_dict, from_version, to_version) -> new_raw_envelope_dict.
# Both versions are passed so a single upcaster can defensively assert it
# matches its declared (N, N+1) hop in tests.
Upcaster = Callable[[dict[str, Any], int, int], dict[str, Any]]


__all__ = ["Upcaster"]
