"""
docker/training/Dockerfile.runtime — thin-image structural invariants.

The thin-image migration (v2.0.0+) inverted the previous "image bakes
src/runner" model: the image now ships ONLY environment (Python +
CUDA + libs + sshd + entrypoint) and the Mac control plane rsyncs
``src/...`` into ``/workspace/runs/<run_id>`` per run.

This file holds **regression guards** against silently re-baking
``src/`` back into the image — which would re-couple every
runner-code change to a 10-minute Docker rebuild and burn the CI
cycle savings the migration bought.

If any of these tests fails, the Dockerfile has drifted away from
``docs/architecture/thin-image.md``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_DOCKERFILE = Path(__file__).resolve().parents[4] / "docker" / "training" / "Dockerfile.runtime"


@pytest.fixture(scope="module")
def dockerfile_lines() -> list[str]:
    """Read Dockerfile.runtime once, strip whole-line comments.

    Comments are stripped because we DO want to leave a textual note
    explaining WHY we no longer ``COPY src`` — and that note must not
    trip the regression grep below.
    """
    raw = _DOCKERFILE.read_text(encoding="utf-8")
    out: list[str] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        out.append(line)
    return out


# ---------------------------------------------------------------------------
# Negative regressions — bakery is forbidden.
# ---------------------------------------------------------------------------


def test_no_copy_src_into_image(dockerfile_lines: list[str]) -> None:
    """No ``COPY src`` directive may exist.

    A ``COPY src ...`` brings the entire backend into the image and
    re-couples runner-code edits to a Docker rebuild — exactly the
    cycle the thin image removes. The matcher is anchored at line
    start to avoid colliding with copies of *other* files whose path
    happens to contain ``src`` (e.g. a hypothetical
    ``COPY some/script.py``).
    """
    for line in dockerfile_lines:
        # Match ``COPY src`` and ``COPY src/``; allow ``COPY scripts/...``,
        # ``COPY docker/...``, etc.
        if re.match(r"^\s*COPY\s+src(\s|/)", line):
            pytest.fail(
                f"forbidden ``COPY src`` directive present: {line.rstrip()!r} — "
                "the thin image must not bake src/ in",
            )


def test_no_pythonpath_opt_ryotenkai_env(dockerfile_lines: list[str]) -> None:
    """``ENV PYTHONPATH=...opt/ryotenkai...`` was the baked-in baseline.

    Removing it is the second half of the thin-image migration: with
    a baked baseline, even a missing rsync would silently launch
    uvicorn from /opt/ryotenkai. We want ModuleNotFoundError instead
    so the failure mode is loud.
    """
    for line in dockerfile_lines:
        if re.match(r"^\s*ENV\s+PYTHONPATH", line) and "/opt/ryotenkai" in line:
            pytest.fail(
                f"forbidden PYTHONPATH=/opt/ryotenkai env: {line.rstrip()!r} — "
                "thin-image migration removed the baked baseline",
            )


# ---------------------------------------------------------------------------
# Positive — the environment-spec parts that MUST stay.
# ---------------------------------------------------------------------------


def test_runtime_check_py_still_copied(dockerfile_lines: list[str]) -> None:
    """``runtime_check.py`` is the dependency-verifier the control
    plane runs after pull. It is part of the **environment spec** and
    must travel with the image.
    """
    found = any(
        "docker/training/runtime_check.py" in line and line.lstrip().startswith("COPY")
        for line in dockerfile_lines
    )
    assert found, "runtime_check.py must remain in the image (env-spec, not application code)"


def test_entrypoint_sh_still_copied(dockerfile_lines: list[str]) -> None:
    """The inert-pod ``entrypoint.sh`` (sshd + PUBLIC_KEY + sleep)
    is the bootstrap contract for every provider. Must stay baked.
    """
    found = any(
        "docker/training/entrypoint.sh" in line and line.lstrip().startswith("COPY")
        for line in dockerfile_lines
    )
    assert found, "entrypoint.sh must remain in the image"


def test_requirements_runtime_still_copied(dockerfile_lines: list[str]) -> None:
    """The pip-install layer is the body of the environment. Without
    ``requirements.runtime.txt`` baked in, the image ships no
    Python deps and fails at runtime_check time.
    """
    found = any(
        "requirements.runtime.txt" in line and line.lstrip().startswith("COPY")
        for line in dockerfile_lines
    )
    assert found, "requirements.runtime.txt must remain in the image"
