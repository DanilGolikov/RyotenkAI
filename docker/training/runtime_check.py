#!/usr/bin/env python3
"""Runtime contract checker — baked into the training image at
``/opt/helix/runtime_check.py``.

The control plane runs this inside the pod's container after pulling
the image to verify the runtime profile is intact (right CUDA build of
torch, all training deps importable, etc.). It is the single source of
truth for "what does this image actually have?" — read from outside via
``docker run --rm <image> python3 /opt/helix/runtime_check.py``.

Output contract (parsed by
:mod:`src.pipeline.stages.managers.deployment.dependency_installer`):

* First line: ``OK`` (literal). The verifier looks for this token in
  stdout to declare success.
* Subsequent lines: ``<name>=<version>`` for each required package, in
  a stable order. Missing packages emit ``<name>=missing`` and turn the
  exit code non-zero. Packages without ``__version__`` (e.g. pynvml)
  emit ``<name>=unknown`` but don't fail.

Why a separate script and not an ``import`` smoke test inline in the
SSH command: keeping the contract in one file makes "what does the
image promise?" inspectable from a published image without reading
control-plane code, and the file gets versioned alongside
``requirements.runtime.txt``.
"""

from __future__ import annotations

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


def main() -> int:
    """Print the manifest. Return non-zero if any required package is
    missing or fails to import — that's a broken-image signal the
    control plane will surface as a deployment failure."""
    lines: list[str] = []
    failed = False

    for name, attr in _REQUIRED:
        try:
            v = _version(name, attr)
        except Exception as exc:  # noqa: BLE001 — any import error means broken image
            lines.append(f"{name}=missing ({type(exc).__name__})")
            failed = True
        else:
            lines.append(f"{name}={v}")

    for name, attr in _OPTIONAL:
        try:
            v = _version(name, attr)
        except Exception:  # noqa: BLE001 — optional, soft-fail
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


if __name__ == "__main__":
    sys.exit(main())
