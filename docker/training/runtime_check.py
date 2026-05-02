#!/usr/bin/env python3
"""Runtime contract checker — baked into the training image at
``/opt/helix/runtime_check.py``.

Two modes (mutually exclusive flags):

* **No flag** (default): verify pip packages — torch, transformers, etc.
  Used by :class:`DependencyInstaller` after the runtime image is pulled
  to confirm the image profile is intact.
* **``--check-source``**: verify ``src.*`` Python modules importable from
  the *current* workspace's PYTHONPATH. Used by :class:`CodeSyncer` after
  rsync to enforce a fail-fast post-deployment contract: rsync rc=0 does
  not guarantee that all required src.* modules ended up on the pod
  (see ``code_syncer.py:86-93`` failure mode caught in
  ``run_20260429_171726_49j32`` and the 15-crash incident on 2026-05-02).

Output contracts (parsed by control plane):

Default mode — parsed by
:mod:`src.pipeline.stages.managers.deployment.dependency_installer`:

* First line: ``OK`` / ``FAIL``
* Subsequent lines: ``<name>=<version>`` for each required package.
* Missing required packages → ``<name>=missing (<ExceptionType>)``,
  exit code 1.
* Missing optional packages → ``<name>=unknown``, exit code unchanged.

``--check-source`` mode — parsed by
:mod:`src.pipeline.stages.managers.deployment.code_syncer`:

* First line: ``OK`` / ``FAILED``.
* Subsequent lines: ``<module>=importable`` on success, or
  ``<module>=NOT_IMPORTABLE (<ExceptionType>: <message>)`` on failure.
* Exit code 0 on success, 2 on failure (distinct from rc=1 of the
  default mode so the parser can disambiguate which contract was
  violated).

Why a separate script and not an ``import`` smoke test inline in the SSH
command: keeping the contract in one file makes "what does the image
promise?" inspectable from a published image without reading control
plane code, and the file gets versioned alongside
``requirements.runtime.txt``.
"""

from __future__ import annotations

import argparse
import importlib
import sys

# Required packages — ImportError on any of these = wrong/broken image.
# Order is the order of the printed manifest.
_REQUIRED: list[tuple[str, str]] = [
    # (import name, attr to read for version — empty = use module.__version__)
    ("torch", ""),
    ("transformers", ""),
    ("trl", ""),
    ("peft", ""),
    ("accelerate", ""),
    ("datasets", ""),
    ("mlflow", ""),
    ("pydantic", ""),
    ("yaml", ""),  # PyYAML — module name is ``yaml``
    ("omegaconf", ""),
    ("psutil", ""),
]

# Optional packages — best-effort. Missing = print ``unknown`` and keep
# going. Used for things like pynvml that may or may not ship in a given
# CUDA base image but the trainer can soft-fall-back without.
_OPTIONAL: list[tuple[str, str]] = [
    ("pynvml", ""),
]

# Required src.* modules — every entry must be importable on the pod for
# the trainer to spawn cleanly. Drift between this list and the trainer's
# actual top-level imports is guarded by
# ``src/tests/unit/training/test_required_modules_drift.py``.
#
# Why each module is here:
#   * ``src.workspace.integrations.loader`` — load_pipeline_config(),
#     called at run_training.py module-load.
#   * ``src.config`` — pydantic schemas the loader validates against.
#   * ``src.providers`` — provider lifecycle clients the runner imports
#     at startup. Recurring failure mode: missing → uvicorn dies before
#     binding 8080 (run_20260429_171726_49j32, plus 15-crash incident).
#   * ``src.training.run_training`` — trainer entrypoint itself.
#   * ``src.runner.main`` — runner entrypoint (shipped via thin-image
#     since Phase 6.6, no longer baked in the docker image).
#   * ``src.utils.config`` — config façade re-exported from src.config.
_REQUIRED_SRC_MODULES: list[str] = [
    "src.workspace.integrations.loader",
    "src.config",
    "src.providers",
    "src.training.run_training",
    "src.runner.main",
    "src.utils.config",
]


def _version(mod_name: str, attr: str) -> str:
    """Return the version string for ``mod_name``, or raise on failure.

    For most packages we just read ``module.__version__``. ``attr`` is
    a future-proof hook for libraries whose canonical version lives at
    a sub-attribute (e.g. ``ver.STRING``). Empty ``attr`` = use
    ``__version__``.
    """
    mod = importlib.import_module(mod_name)
    if attr:
        return str(getattr(mod, attr))
    return str(getattr(mod, "__version__", "unknown"))


def check_pip_packages() -> int:
    """Default mode — verify pip packages from the runtime image.

    Returns 0 on success, 1 if any required package is missing.
    """
    lines: list[str] = []
    failed = False

    for name, attr in _REQUIRED:
        try:
            v = _version(name, attr)
        except Exception as exc:
            lines.append(f"{name}=missing ({type(exc).__name__})")
            failed = True
        else:
            lines.append(f"{name}={v}")

    for name, attr in _OPTIONAL:
        try:
            v = _version(name, attr)
        except Exception:
            lines.append(f"{name}=unknown")
        else:
            lines.append(f"{name}={v}")

    # ``OK`` first so the parser sees a clean success token at the top
    # of stdout. Required packages are listed below it.
    if not failed:
        print("OK")
    else:
        print("FAIL")

    for line in lines:
        print(line)

    return 1 if failed else 0


def check_source_importable() -> int:
    """``--check-source`` mode — verify each required ``src.*`` module
    is import-able from the current PYTHONPATH.

    Returns 0 on success, 2 on failure. The non-1 exit code lets the
    parser distinguish "image broken" (rc=1) from "synced source broken"
    (rc=2).
    """
    failed: list[tuple[str, str]] = []
    for mod_name in _REQUIRED_SRC_MODULES:
        try:
            importlib.import_module(mod_name)
        except Exception as exc:
            failed.append((mod_name, f"{type(exc).__name__}: {exc}"))

    if not failed:
        print("OK")
        for m in _REQUIRED_SRC_MODULES:
            print(f"{m}=importable")
        return 0

    print("FAILED")
    # Print importable modules first, then the failing ones — operator
    # gets a full picture of what made it through.
    failed_names = {m for m, _ in failed}
    for m in _REQUIRED_SRC_MODULES:
        if m not in failed_names:
            print(f"{m}=importable")
    for mod_name, err in failed:
        print(f"{mod_name}=NOT_IMPORTABLE ({err})")
    return 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="runtime_check",
        description="Verify runtime contract on a training pod.",
    )
    parser.add_argument(
        "--check-source",
        action="store_true",
        help="Verify required src.* modules are importable (post-sync gate).",
    )
    args = parser.parse_args(argv)

    if args.check_source:
        return check_source_importable()
    return check_pip_packages()


if __name__ == "__main__":
    sys.exit(main())
