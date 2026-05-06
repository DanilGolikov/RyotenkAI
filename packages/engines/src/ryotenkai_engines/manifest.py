"""Pydantic schema for ``engine.toml`` files.

Mirrors the design of ``packages/providers/src/ryotenkai_providers/manifest.py``:

* Each block is its own ``BaseModel`` with ``extra="forbid"``.
* Cross-field invariants in ``@model_validator(mode="after")``.
* Schema-version gate: manifests declaring a newer ``schema_version`` than
  the loader knows about are rejected with an upgrade hint.
* Drift detector (PR-10 ``check_engine_manifests.py``) cross-checks that
  every shipped engine's runtime ``get_capabilities()`` exactly matches
  this schema's ``[capabilities]`` block.

Top-level structure of an ``engine.toml``::

    schema_version = 1

    [engine]                # identity
    id          = "vllm"
    name        = "vLLM"
    version     = "1.0.0"
    upstream_version = "0.7.0"   # optional, informational
    description = "..."
    stability   = "stable"
    homepage    = "https://..."

    [capabilities]          # see EngineCapabilities
    api_dialect              = "openai_compatible"
    supports_lora            = true
    supports_quantization    = true
    supports_streaming       = true
    supports_tensor_parallel = true
    supported_quantizations  = ["awq", "gptq"]
    supported_dtypes         = ["bfloat16", "float16"]
    default_port             = 8000

    [image]                 # OPTIONAL — convention default if absent
    default = "ryotenkai/inference-vllm:1.0.0"

    [entry_points.runtime]  # required
    module = "ryotenkai_engines.vllm.runtime"
    class  = "VLLMEngineRuntime"

    [entry_points.config_schema]   # required
    module = "ryotenkai_engines.vllm.config"
    class  = "VLLMEngineConfig"
"""

from __future__ import annotations

import re
from typing import Literal

from packaging.version import InvalidVersion, Version
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ryotenkai_engines.capabilities import EngineCapabilities

#: Engine id format. ``snake_case``, starts with a letter; must equal the
#: package directory name (the loader walks
#: ``packages/engines/src/ryotenkai_engines/<id>/engine.toml``). Stable
#: identifiers are used in user YAML configs (the discriminator value),
#: image names, and operator dashboards — renaming is a contract change.
_ENGINE_ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")


#: Latest manifest schema version this loader understands. Bumped on
#: breaking changes only (additive optional fields don't bump). A manifest
#: declaring a newer number is rejected with an upgrade hint.
#:
#: History:
#:   v1 (2026-05) — initial shape.
LATEST_ENGINE_SCHEMA_VERSION = 1


#: Coarse trustworthiness of the engine implementation. Operator-visible.
#: ``"experimental"`` engines are deployable but warn at registry build.
Stability = Literal["stable", "beta", "experimental"]


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class EngineSpec(BaseModel):
    """Body of ``[engine]`` — identity + integration version."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Snake_case canonical id; equals folder name.")
    name: str = Field(description="Display name for UI / dashboards.")
    version: str = Field(
        description=(
            "PEP 440 / SemVer string for OUR integration contract — NOT "
            "the upstream engine version. Bumping this triggers an image "
            "rebuild via the convention default tag."
        ),
    )
    upstream_version: str = Field(
        default="",
        description=(
            "Optional informational field — the upstream engine release "
            "this integration packages (e.g. 'vLLM 0.7.0'). Surfaces in "
            "logs, MLflow tags, operator dashboards. NOT used in image "
            "naming. Empty string when irrelevant."
        ),
    )
    description: str = Field(default="", description="One-line summary.")
    stability: Stability = Field(
        default="experimental",
        description="Coarse maturity marker; UI surfaces; non-stable engines emit a warning at registry boot.",
    )
    homepage: str = Field(
        default="",
        description="Optional URL to upstream project / docs.",
    )

    @model_validator(mode="after")
    def _check_id_format(self) -> EngineSpec:
        if not _ENGINE_ID_RE.match(self.id):
            raise ValueError(
                f"engine.id must match {_ENGINE_ID_RE.pattern!r} "
                f"(snake_case identifier matching the folder name under "
                f"packages/engines/src/ryotenkai_engines/); got {self.id!r}"
            )
        return self

    @model_validator(mode="after")
    def _check_version_format(self) -> EngineSpec:
        try:
            Version(self.version)
        except InvalidVersion as exc:
            raise ValueError(
                f"engine {self.id!r} version {self.version!r} is not a "
                f"valid PEP 440 / SemVer version: {exc}"
            ) from exc
        # upstream_version is opaque — engines may use upstream's own
        # versioning (e.g. "0.7.0+cuda12.4") — only validate when present
        # AND the author wants strict checking. Keep informational.
        return self


class ImageSpec(BaseModel):
    """Body of ``[image]`` — optional explicit image override.

    When absent (the default), :func:`ryotenkai_engines.images.resolve_image`
    falls through to the convention:
    ``f"{prefix}/inference-{id}:{version}"``.

    When present, the explicit ``default`` wins over the convention but
    is still overridable by env / provider-side overrides.
    """

    model_config = ConfigDict(extra="forbid")

    default: str = Field(
        description=(
            "Fully-qualified image (registry/name:tag). MUST NOT use "
            "floating tags (`:latest`, `:dev`, `:main`) — drift detector "
            "rejects them. Use semver-pinned tags for reproducibility."
        ),
    )

    @model_validator(mode="after")
    def _check_pinned_tag(self) -> ImageSpec:
        if not self.default:
            raise ValueError("image.default must not be empty")
        # Floating-tag guard. We use a denylist (small, explicit) rather
        # than try to parse the entire docker tag grammar.
        floating_tags = {":latest", ":dev", ":main", ":master", ":nightly", ":edge"}
        for tag in floating_tags:
            if self.default.endswith(tag):
                raise ValueError(
                    f"image.default {self.default!r} uses a floating tag "
                    f"{tag!r}. Pin to a specific version for reproducibility."
                )
        return self


class ClassEntryPoint(BaseModel):
    """Locator for a runtime-resolved class.

    Loader splits ``module`` + ``class_name`` and uses
    ``importlib.import_module(module).<class_name>`` lazily — only when
    the corresponding role is requested. Keeps heavy engine SDKs out of
    the import graph for non-using deployments.
    """

    model_config = ConfigDict(extra="forbid")

    module: str = Field(description="Fully-qualified Python module path.")
    class_name: str = Field(
        alias="class",
        description="Class name within the module.",
    )


class EntryPointsSpec(BaseModel):
    """Body of the ``[entry_points.*]`` blocks.

    Both ``runtime`` and ``config_schema`` are required — engines with
    no runtime class are not real engines; engines with no config class
    can't participate in the discriminated union.
    """

    model_config = ConfigDict(extra="forbid")

    runtime: ClassEntryPoint = Field(
        description=(
            "Locator for the IInferenceEngine class. The registry "
            "instantiates it on demand; its ``engine_id`` ClassVar MUST "
            "equal ``engine.id``."
        ),
    )
    config_schema: ClassEntryPoint = Field(
        description=(
            "Locator for the BaseEngineConfig subclass. Used by the "
            "discriminated-union builder; its ``kind`` Literal MUST "
            "equal ``engine.id``."
        ),
    )


# ---------------------------------------------------------------------------
# Top-level manifest
# ---------------------------------------------------------------------------


class EngineManifest(BaseModel):
    """Full manifest loaded from
    ``packages/engines/src/ryotenkai_engines/<id>/engine.toml``.

    Single source of truth consumed by :class:`EngineRegistry`. The
    schema-level validators below cover every drift class; the loader
    catches the rest (importability, Protocol parity, ``kind`` Literal
    match) by walking the resolved classes.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(default=LATEST_ENGINE_SCHEMA_VERSION)
    engine: EngineSpec
    capabilities: EngineCapabilities
    image: ImageSpec | None = Field(
        default=None,
        description=(
            "Optional explicit image override. When absent, "
            ":func:`resolve_image` falls back to the convention "
            "``{prefix}/inference-{id}:{version}``."
        ),
    )
    entry_points: EntryPointsSpec

    # ---- whole-document invariants ----

    @model_validator(mode="after")
    def _check_schema_version(self) -> EngineManifest:
        if self.schema_version < 1:
            raise ValueError(
                f"engine schema_version must be >= 1; got {self.schema_version}"
            )
        if self.schema_version > LATEST_ENGINE_SCHEMA_VERSION:
            raise ValueError(
                f"engine schema_version {self.schema_version} is newer than this "
                f"loader supports (max: {LATEST_ENGINE_SCHEMA_VERSION}). Upgrade "
                f"ryotenkai_engines to read this manifest."
            )
        return self

    @model_validator(mode="after")
    def _check_capability_matches_dtypes(self) -> EngineManifest:
        # Bridge-level invariant: a quantization mode listed must not also
        # appear in supported_dtypes (those are mutually exclusive concepts
        # — quantization is "how weights are stored", dtype is "compute
        # precision"). Catch obvious copy-paste mistakes.
        overlap = set(self.capabilities.supported_quantizations) & set(self.capabilities.supported_dtypes)
        if overlap:
            raise ValueError(
                f"capabilities.supported_quantizations and "
                f"capabilities.supported_dtypes must not overlap; common: "
                f"{sorted(overlap)!r}. Quantization (storage format) and "
                f"dtype (compute precision) are distinct axes."
            )
        return self


__all__ = (
    "LATEST_ENGINE_SCHEMA_VERSION",
    "Stability",
    "EngineSpec",
    "ImageSpec",
    "ClassEntryPoint",
    "EntryPointsSpec",
    "EngineManifest",
)
