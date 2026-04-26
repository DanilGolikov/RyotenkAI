"""Test bootstrap for community_libs.helixql.

The test module below does ``from community_libs.helixql import …`` at
import time (module level). That runs during pytest's *collection*
phase, before any fixture has fired — so we cannot rely on a session
fixture to preload the namespace. Instead we run the preload
unconditionally at conftest import, which happens before pytest walks
into ``test_*.py`` files in this directory.

Idempotent: if the catalog has already been loaded earlier in the
test session, ``preload_community_libs`` is a no-op for the same root.
"""

from __future__ import annotations

from src.community.constants import COMMUNITY_ROOT
from src.community.libs import libs_root_for, preload_community_libs

preload_community_libs(libs_root_for(COMMUNITY_ROOT))
