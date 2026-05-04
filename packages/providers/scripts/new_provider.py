#!/usr/bin/env python3
"""Generate the skeleton of a new provider.

Drops a provider folder under
``packages/providers/src/ryotenkai_providers/<id>/`` containing:

  * ``provider.toml`` — minimum-valid manifest with TODO-marked fields.
  * ``__init__.py`` — empty marker (registration is manifest-driven).
  * ``training/{__init__.py, provider.py}`` — when ``training`` in roles.
  * ``inference/{__init__.py, provider.py}`` — when ``inference`` in roles.
  * ``runtime/{__init__.py, lifecycle_client.py}`` — only when
    ``--with-lifecycle`` is passed (sets ``supports_lifecycle_actions=
    true`` in the manifest and emits a stub IPodLifecycleClient impl).

Plus a parallel test skeleton under
``packages/providers/tests/unit/providers/<id>/`` with smoke tests
that import the class and assert its ``ProviderBase`` inheritance.

Per the team's policy (concurrent-gathering-hippo plan §F.3): this is
a **dev-time tool**, not a user-facing CLI. Providers are added by
the core team, not by community plugin authors — so the script lives
in ``packages/providers/scripts/`` and is invoked directly with
``python``, not exposed through ``ryotenkai`` or as a console script.

Usage::

    python packages/providers/scripts/new_provider.py my_cloud_provider \\
        --roles training,inference \\
        [--with-lifecycle] [--with-recovery-probe] [--with-capacity-classifier] \\
        [--type cloud|local]

After scaffolding, the developer should:

1. Fill the TODO-marked fields in ``provider.toml`` (description, author,
   gpu config, etc.).
2. Implement the abstract methods in ``training/provider.py`` /
   ``inference/provider.py``.
3. Add a Pydantic config schema in
   ``packages/shared/src/ryotenkai_shared/config/providers/<id>/`` and
   point ``provider.toml`` at it via ``[entry_points.config_schema]``.
4. Run ``packages/providers/scripts/compile_pod_manifests.py`` to
   generate the pod sub-manifest.
5. Run ``packages/providers/scripts/check_manifests.py`` to validate
   no drift.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Final

#: Provider id format — must round-trip through folder name + Python
#: module name. Same regex the manifest's :class:`ProviderSpec`
#: validator uses; duplicated here so the script doesn't have to import
#: the Pydantic-bound model just for the regex.
_PROVIDER_ID_RE: Final[re.Pattern[str]] = re.compile(r"^[a-z][a-z0-9_]*$")


def _workspace_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file() and (parent / "packages").is_dir():
            return parent
    return Path.cwd()


_REPO_ROOT = _workspace_root()
_PROVIDERS_SRC = _REPO_ROOT / "packages" / "providers" / "src" / "ryotenkai_providers"
_PROVIDERS_TESTS = _REPO_ROOT / "packages" / "providers" / "tests" / "unit" / "providers"


def class_name_from_id(provider_id: str) -> str:
    """``my_cloud`` → ``MyCloudProvider``."""
    parts = [p for p in provider_id.split("_") if p]
    return "".join(p.capitalize() for p in parts) + "Provider"


def render_manifest(
    *,
    provider_id: str,
    class_name: str,
    roles: list[str],
    provider_type: str,
    with_lifecycle: bool,
    with_recovery_probe: bool,
    with_capacity_classifier: bool,
) -> str:
    """Render a minimum-valid ``provider.toml``.

    Capability flags pair 1:1 with the optional Protocol inheritance
    chosen for the training class — so the schema invariants will hold
    out of the box once the developer fills in the TODO-marked fields.
    """
    lines = [
        f"# {provider_id} provider manifest — TODO-marked fields need to be filled.",
        "",
        "schema_version = 1",
        "",
        "[provider]",
        f'id          = "{provider_id}"',
        f'name        = "TODO: human-readable name"',
        'version     = "0.1.0"',
        f'roles       = {roles!r}'.replace("'", '"'),
        'description = "TODO: one-line summary of what this provider does."',
        'author      = "TODO: Name <email>"',
        'stability   = "experimental"',
        "",
        "[capabilities]",
        f'provider_type                     = "{provider_type}"',
        f"is_local                          = {'true' if provider_type == 'local' else 'false'}",
        "supports_multi_gpu                = false   # TODO",
        "supports_spot_instances           = false   # TODO",
        f"supports_lifecycle_actions        = {'true' if with_lifecycle else 'false'}",
        f"has_pause_resume                  = {'true' if with_lifecycle else 'false'}   # TODO subset",
        f"supports_recovery_probe           = {'true' if with_recovery_probe else 'false'}",
        f"supports_capacity_error_detection = {'true' if with_capacity_classifier else 'false'}",
        "supports_log_download             = false   # TODO",
        f'volume_kind                       = "{"local_disk" if provider_type == "local" else "persistent"}"',
        'runner_workspace_root             = "/workspace"',
        "# max_runtime_hours = null   # null = unlimited",
        "",
    ]
    if "training" in roles:
        lines += [
            "[entry_points.training]",
            f'module = "ryotenkai_providers.{provider_id}.training.provider"',
            f'class  = "{class_name}"',
            "",
        ]
    if "inference" in roles:
        inf_class = class_name.replace("Provider", "InferenceProvider")
        lines += [
            "[entry_points.inference]",
            f'module = "ryotenkai_providers.{provider_id}.inference.provider"',
            f'class  = "{inf_class}"',
            "",
        ]
    if with_lifecycle:
        lc_class = class_name.replace("Provider", "PodLifecycleClient")
        lines += [
            "[entry_points.pod_lifecycle_client]",
            f'module = "ryotenkai_providers.{provider_id}.runtime.lifecycle_client"',
            f'class  = "{lc_class}"',
            "",
        ]
    lines += [
        "# TODO: point at your Pydantic config schema. Live under",
        "# packages/shared/src/ryotenkai_shared/config/providers/{provider_id}/",
        "# (see the runpod / single_node packages for the layout).",
        "[entry_points.config_schema]",
        f'module = "ryotenkai_shared.config.providers.{provider_id}"',
        f'class  = "TODO_{class_name}Config"',
        "",
        "# [[required_env]]",
        "# Declare any operator-supplied secrets here. The startup",
        "# validator pulls them via ``registry.required_secrets``;",
        "# missing required envs fail-fast at startup before any pipeline",
        "# stage runs. Use ``required_for_roles=[]`` (the default) when the",
        "# secret is shared across all declared roles.",
        '# name               = "MYPROV_API_KEY"',
        '# description        = "TODO: what this token is used for."',
        "# optional           = false",
        "# secret             = true",
        "# required_for_roles = []",
        "",
    ]
    return "\n".join(lines)


def render_training_provider_py(
    *,
    provider_id: str,
    class_name: str,
    with_recovery_probe: bool,
    with_capacity_classifier: bool,
    with_lifecycle: bool,
) -> str:
    bases = ["ProviderBase", "IGPUProvider"]
    extra_imports = ["IGPUProvider", "ProviderBase", "ProviderCapabilities"]
    if with_lifecycle:
        bases.append("ITerminalActionProvider")
        extra_imports.append("ITerminalActionProvider")
    if with_recovery_probe:
        bases.append("IRecoveryProbeProvider")
        extra_imports.append("IRecoveryProbeProvider")
    if with_capacity_classifier:
        bases.append("ICapacityErrorClassifier")
        extra_imports.append("ICapacityErrorClassifier")
    extra_imports.append("ProviderStatus")
    bases_str = ", ".join(bases)
    extra_imports_block = ",\n    ".join(sorted(set(extra_imports)))
    body = [
        '"""TODO: one-line module docstring."""',
        "",
        "from __future__ import annotations",
        "",
        "from typing import TYPE_CHECKING",
        "",
        "from ryotenkai_providers.training.interfaces import (",
        f"    {extra_imports_block},",
        ")",
        "",
        "if TYPE_CHECKING:",
        "    from ryotenkai_providers.registry import ProviderContext",
        "",
        "",
        f"class {class_name}({bases_str}):",
        '    """TODO: class docstring."""',
        "",
        '    def __init__(self, ctx: "ProviderContext") -> None:',
        '        """Initialize the provider from a :class:`ProviderContext`."""',
        "        self._ctx = ctx",
        '        raise NotImplementedError("TODO: parse provider_block, init API client.")',
        "",
        "    # IGPUProvider methods — TODO: implement each.",
        "",
        "    def connect(self, *, run):  # type: ignore[no-untyped-def]",
        '        raise NotImplementedError("TODO: connect")',
        "",
        "    def disconnect(self):",
        '        raise NotImplementedError("TODO: disconnect")',
        "",
        "    def get_status(self) -> ProviderStatus:",
        '        raise NotImplementedError("TODO: get_status")',
        "",
        "    def check_gpu(self):",
        '        raise NotImplementedError("TODO: check_gpu")',
        "",
        "    def get_resource_info(self):",
        "        return None",
        "",
        "    def required_runtime_env_vars(self, *, resource_id=None):",
        '        raise NotImplementedError("TODO: env vars for trainer")',
        "",
        "    def probe_availability(self, resource_id: str):",
        '        raise NotImplementedError("TODO: availability probe")',
        "",
        "    def pod_layout_for_run(self, run_id: str):",
        '        raise NotImplementedError("TODO: PodLayout")',
        "",
        "    def required_secrets(self) -> tuple[str, ...]:",
        '        # Manifest-driven; the registry reads required_env directly.',
        "        return ()",
        "",
        "    def prepare_training_script_hooks(self, ssh_client, context):",
        '        raise NotImplementedError("TODO: training hooks")',
        "",
    ]
    if with_lifecycle:
        body += [
            "    # ITerminalActionProvider — TODO: implement.",
            "",
            "    def terminate(self, *, resource_id: str, reason: str):",
            '        raise NotImplementedError("TODO: terminate")',
            "",
            "    def pause(self, *, resource_id: str):",
            '        raise NotImplementedError("TODO: pause")',
            "",
            "    def resume(self, *, resource_id: str):",
            '        raise NotImplementedError("TODO: resume")',
            "",
        ]
    if with_recovery_probe:
        body += [
            "    # IRecoveryProbeProvider — TODO: implement.",
            "",
            "    def attempt_recovery(self, *, resource_id: str):",
            '        raise NotImplementedError("TODO: probe + recover after WS loss")',
            "",
        ]
    if with_capacity_classifier:
        body += [
            "    # ICapacityErrorClassifier — TODO: implement.",
            "",
            "    def is_capacity_error(self, message: str) -> bool:",
            '        raise NotImplementedError("TODO: classify capacity errors")',
            "",
        ]
    return "\n".join(body)


def render_inference_provider_py(*, provider_id: str, class_name: str) -> str:
    inf_class = class_name.replace("Provider", "InferenceProvider")
    return "\n".join(
        [
            '"""TODO: one-line module docstring."""',
            "",
            "from __future__ import annotations",
            "",
            "from typing import TYPE_CHECKING",
            "",
            "from ryotenkai_providers.inference.interfaces import IInferenceProvider",
            "from ryotenkai_providers.training.interfaces import ProviderBase",
            "",
            "if TYPE_CHECKING:",
            "    from ryotenkai_providers.registry import ProviderContext",
            "",
            "",
            f"class {inf_class}(ProviderBase, IInferenceProvider):",
            '    """TODO: class docstring."""',
            "",
            '    def __init__(self, ctx: "ProviderContext") -> None:',
            "        self._ctx = ctx",
            '        raise NotImplementedError("TODO: parse pipeline_config.inference, init engine client.")',
            "",
            "    # IInferenceProvider — TODO: implement each method.",
            "",
        ]
    )


def render_lifecycle_client_py(*, provider_id: str, class_name: str) -> str:
    lc_class = class_name.replace("Provider", "PodLifecycleClient")
    return "\n".join(
        [
            '"""TODO: one-line module docstring."""',
            "",
            "from __future__ import annotations",
            "",
            "from ryotenkai_shared.infrastructure.lifecycle import (",
            "    IPodLifecycleClient,",
            "    LifecycleActionResult,",
            ")",
            "",
            "",
            f"class {lc_class}:",
            '    """TODO: docstring. Implements IPodLifecycleClient (async)."""',
            "",
            "    def __init__(self, *, api_key: str | None = None) -> None:",
            "        self._api_key = api_key",
            "",
            "    @property",
            "    def provider_name(self) -> str:",
            f'        return "{provider_id}"',
            "",
            "    async def terminate(self, *, resource_id: str) -> LifecycleActionResult:",
            '        raise NotImplementedError("TODO: terminate")',
            "",
            "    async def pause(self, *, resource_id: str) -> LifecycleActionResult:",
            '        raise NotImplementedError("TODO: pause")',
            "",
            "    async def resume(self, *, resource_id: str) -> LifecycleActionResult:",
            '        raise NotImplementedError("TODO: resume")',
            "",
        ]
    )


def render_test_smoke(*, provider_id: str, class_name: str) -> str:
    return "\n".join(
        [
            f'"""Smoke tests for the {provider_id} provider scaffold."""',
            "",
            "from __future__ import annotations",
            "",
            "import pytest",
            "",
            f"from ryotenkai_providers.{provider_id}.training.provider import {class_name}",
            "from ryotenkai_providers.training.interfaces import ProviderBase",
            "",
            "",
            f"def test_{provider_id}_inherits_provider_base() -> None:",
            f"    assert issubclass({class_name}, ProviderBase)",
            "",
            "",
            f"def test_{provider_id}_construction_raises_until_implemented() -> None:",
            "    # Scaffolded class raises NotImplementedError on init — replace this",
            "    # test as soon as the impl lands.",
            "    with pytest.raises(NotImplementedError):",
            f"        {class_name}(ctx=None)  # type: ignore[arg-type]",
            "",
        ]
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _write(path: Path, content: str, *, force: bool) -> None:
    if path.exists() and not force:
        raise SystemExit(f"refuse to overwrite existing {path} (use --force).")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    print(f"  wrote: {path.relative_to(_REPO_ROOT)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("provider_id", help="snake_case provider id (folder name).")
    parser.add_argument(
        "--roles",
        default="training,inference",
        help="Comma-separated roles. Default: training,inference",
    )
    parser.add_argument(
        "--type",
        choices=["cloud", "local"],
        default="cloud",
        help="provider_type for the manifest. Default: cloud",
    )
    parser.add_argument(
        "--with-lifecycle",
        action="store_true",
        help="Implement ITerminalActionProvider + emit a pod lifecycle client.",
    )
    parser.add_argument(
        "--with-recovery-probe",
        action="store_true",
        help="Implement IRecoveryProbeProvider (training-monitor recovery loop).",
    )
    parser.add_argument(
        "--with-capacity-classifier",
        action="store_true",
        help="Implement ICapacityErrorClassifier (resume-service retry classifier).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files. Use with care.",
    )
    args = parser.parse_args(argv)

    pid: str = args.provider_id
    if not _PROVIDER_ID_RE.match(pid):
        parser.error(
            f"provider_id must match {_PROVIDER_ID_RE.pattern!r} "
            f"(snake_case, lowercase, starts with a letter); got {pid!r}"
        )
    roles = [r.strip() for r in args.roles.split(",") if r.strip()]
    bad = [r for r in roles if r not in ("training", "inference")]
    if bad:
        parser.error(f"unknown role(s) {bad!r}; valid: training, inference")
    if not roles:
        parser.error("--roles must be non-empty.")
    if args.type == "local":
        # Schema invariant: is_local=true → no lifecycle / capacity-classifier.
        if args.with_lifecycle or args.with_capacity_classifier:
            parser.error(
                "--type=local is incompatible with --with-lifecycle / "
                "--with-capacity-classifier (manifest schema rejects this)."
            )

    class_name = class_name_from_id(pid)
    src_dir = _PROVIDERS_SRC / pid
    tests_dir = _PROVIDERS_TESTS / pid
    if src_dir.exists() and not args.force:
        parser.error(f"{src_dir} already exists. Use --force to overwrite.")

    print(f"Scaffolding provider {pid!r} ({class_name}) at {src_dir.relative_to(_REPO_ROOT)}/")

    # Manifest + package roots.
    manifest = render_manifest(
        provider_id=pid,
        class_name=class_name,
        roles=roles,
        provider_type=args.type,
        with_lifecycle=args.with_lifecycle,
        with_recovery_probe=args.with_recovery_probe,
        with_capacity_classifier=args.with_capacity_classifier,
    )
    _write(src_dir / "provider.toml", manifest, force=args.force)
    _write(
        src_dir / "__init__.py",
        f'"""{pid} provider — registration is manifest-driven."""\n',
        force=args.force,
    )

    if "training" in roles:
        _write(
            src_dir / "training" / "__init__.py",
            f'"""{pid} training provider package."""\n',
            force=args.force,
        )
        _write(
            src_dir / "training" / "provider.py",
            render_training_provider_py(
                provider_id=pid,
                class_name=class_name,
                with_recovery_probe=args.with_recovery_probe,
                with_capacity_classifier=args.with_capacity_classifier,
                with_lifecycle=args.with_lifecycle,
            ),
            force=args.force,
        )
    if "inference" in roles:
        _write(
            src_dir / "inference" / "__init__.py",
            f'"""{pid} inference provider package."""\n',
            force=args.force,
        )
        _write(
            src_dir / "inference" / "provider.py",
            render_inference_provider_py(provider_id=pid, class_name=class_name),
            force=args.force,
        )
    if args.with_lifecycle:
        _write(
            src_dir / "runtime" / "__init__.py",
            f'"""{pid} pod-side runtime package."""\n',
            force=args.force,
        )
        _write(
            src_dir / "runtime" / "lifecycle_client.py",
            render_lifecycle_client_py(provider_id=pid, class_name=class_name),
            force=args.force,
        )

    # Tests.
    _write(
        tests_dir / "__init__.py",
        '"""Tests for the scaffolded provider."""\n',
        force=args.force,
    )
    if "training" in roles:
        _write(
            tests_dir / "test_smoke.py",
            render_test_smoke(provider_id=pid, class_name=class_name),
            force=args.force,
        )

    print()
    print("Next steps:")
    print(f"  1. Fill TODOs in {src_dir.relative_to(_REPO_ROOT)}/provider.toml.")
    print(f"  2. Implement methods in {src_dir.relative_to(_REPO_ROOT)}/training/provider.py.")
    print(
        f"  3. Add a Pydantic config schema in "
        f"packages/shared/src/ryotenkai_shared/config/providers/{pid}/."
    )
    print("  4. Run python packages/providers/scripts/compile_pod_manifests.py")
    print("  5. Run python packages/providers/scripts/check_manifests.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
