"""Pydantic models for community manifests (plugins, libs, presets).

Schema v3 (2026-04): plugin params/thresholds are described via
:class:`ParamFieldSchema` — a typed, field-level description that the
UI can render as a proper form (labels, descriptions, defaults,
required-markers) using the same ``FieldRenderer`` as the main
config builder. At API boundary :meth:`PluginManifest.ui_manifest`
emits the schemas in JSON Schema shape so the frontend does not
need to know about our TOML-specific keys.

Plugin schema v5 (2026-04): adds ``author`` to ``[plugin]`` and
replaces the v4 ``[plugin].libs = [...]`` shorthand with top-level
``[[lib_requirements]]`` blocks that carry an optional PEP 440
specifier (e.g. ``version = ">=1.0.0,<2.0.0"``) — symmetric with
the existing ``[[required_env]]`` shape.

Lib manifests live alongside plugin manifests under
``community/libs/<name>/manifest.toml``. A lib's manifest is
deliberately minimal: ``id`` + ``version`` (PEP 440) carry the
load-time + dependency-check semantics; ``description`` / ``author``
are surface-only metadata for the UI catalogue.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version
from pydantic import BaseModel, ConfigDict, Field, model_validator

#: Field names in ``[params_schema.X]`` / ``[thresholds_schema.X]`` must
#: be valid Python identifiers in ``snake_case`` so they round-trip
#: cleanly into TypedDict / dataclass codegen (PR10's longer-term goal)
#: AND so the scaffold CLI can emit a working ``self.params["x"]``
#: access stub. Names like ``min-samples`` or ``MaxLength`` would force
#: every consumer to special-case identifiers — we'd rather catch it
#: at manifest-load time.
_PARAM_FIELD_NAME_RE = re.compile(r"^[a-z_][a-z0-9_]*$")

#: Names in ``[plugin].libs`` are folder names under ``community/libs/``;
#: that folder is also imported as ``community_libs.<name>``, so the name
#: MUST be a valid Python identifier in ``snake_case``. Same rule as
#: :data:`_PARAM_FIELD_NAME_RE` — kept as a separate constant for the
#: sake of readable error messages.
_LIB_NAME_RE = re.compile(r"^[a-z_][a-z0-9_]*$")

PluginKind = Literal["validation", "evaluation", "reward", "reports"]

Stability = Literal["stable", "beta", "experimental"]

#: Latest manifest schema version this loader understands. Bumped whenever
#: a breaking shape change lands (e.g. removing a field, renaming a key).
#: Manifests omitting ``schema_version`` are treated as the latest, so legacy
#: TOMLs keep loading; future TOMLs declaring a newer number are rejected
#: with a clear error so the user knows to upgrade RyotenkAI.
#:
#: History:
#:   v3 (2026-04) — ``params_schema`` / ``thresholds_schema`` via
#:                  :class:`ParamFieldSchema`; introduced the ``required_env``
#:                  block alongside the legacy ``[secrets].required``.
#:   v4 (2026-04) — dropped ``[secrets].required`` — every required env
#:                  (secret or not, optional or not) is now declared via
#:                  ``[[required_env]]``. The loader derives the runtime
#:                  ``_required_secrets`` ClassVar from entries with
#:                  ``secret=true, optional=false``.
#:   v5 (2026-04) — current. Adds ``[plugin].author`` (free-form
#:                  ``"Name <email>"`` recommended). Replaces the v4
#:                  ``[plugin].libs = [...]`` shorthand with top-level
#:                  ``[[lib_requirements]]`` blocks: each entry has a
#:                  ``name`` (must match a ``community/libs/<name>/``
#:                  package) and an optional PEP 440 ``version``
#:                  specifier (e.g. ``">=1.0.0,<2.0.0"``). Empty
#:                  version means "any". Loader rejects the v4 shape
#:                  with a precise migration hint.
LATEST_SCHEMA_VERSION = 5

#: Lib manifests evolve independently of plugin manifests. Bump only
#: when the on-disk shape under ``community/libs/<name>/manifest.toml``
#: changes. v1 is the inaugural shape: ``[lib]`` with ``id``, ``version``,
#: ``description``, ``author``.
LATEST_LIB_SCHEMA_VERSION = 1

#: Leaf types accepted in ``[params_schema.X] type``. ``enum`` is rendered
#: as a JSON-Schema string with an ``enum`` list; everything else maps
#: directly to the same-named JSON Schema type.
ParamFieldType = Literal["integer", "number", "string", "boolean", "enum"]


class EntryPoint(BaseModel):
    """Entry point for plugin code imports."""

    model_config = ConfigDict(extra="forbid")

    module: str = Field(description="Python module name inside the plugin folder (without .py).")
    class_name: str = Field(
        alias="class",
        description="Fully-qualified class name within the module.",
    )


class PresetEntryPoint(BaseModel):
    """Entry point for a preset (points to YAML body)."""

    model_config = ConfigDict(extra="forbid")

    file: str = Field(description="Relative path to the preset YAML inside the preset folder.")


class RequiredEnvSpec(BaseModel):
    """Environment variable a plugin needs at runtime.

    Plugins declare this list explicitly so the UI can render a
    "Required environment variables" block in the Configure modal,
    point the user at the right Settings tab when the value is a
    managed credential (HF / RunPod / MLflow), and refuse to launch
    a pipeline if a non-optional env is unset.

    Fields:
      name        — env-var name. UPPER_SNAKE_CASE. Stored as-is in
                    the project's `env.json`.
      description — what the value is used for. Surfaces in the UI
                    next to the input.
      optional    — when True, plugin runs even if the var is unset.
      secret      — when True, UI renders a password-style input
                    (`type=password` + show/hide toggle). Default is
                    True because most envs are credentials; explicitly
                    set to False for non-secret config (URLs etc).
      managed_by  — informational hint surfaced in the UI when the
                    var is owned by another Settings surface
                    (`integrations` / `providers`). Lets the modal
                    redirect the user instead of accepting a plain-
                    text override.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    optional: bool = False
    secret: bool = True
    managed_by: Literal["integrations", "providers", ""] = ""


class CompatSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_core_version: str = ""


class LibRequirement(BaseModel):
    """A plugin's declared dependency on a community/libs/<name>/ package.

    Symmetric with :class:`RequiredEnvSpec` (top-level
    ``[[lib_requirements]]`` blocks in the plugin's ``manifest.toml``).
    Each entry pins a *name* (which must match a real lib folder under
    ``community/libs/``) and an optional PEP 440 specifier:

    .. code-block:: toml

        [[lib_requirements]]
        name = "helixql"
        version = ">=1.0.0,<2.0.0"

    An empty ``version`` means "any version" — the loader still checks
    that the lib *exists*, just doesn't constrain its version.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    version: str = ""

    @model_validator(mode="after")
    def _check_name_format(self) -> LibRequirement:
        if not _LIB_NAME_RE.match(self.name):
            raise ValueError(
                f"lib_requirements.name must match {_LIB_NAME_RE.pattern!r} "
                f"(snake_case identifier matching a community/libs/<name>/ "
                f"folder); got {self.name!r}"
            )
        return self

    @model_validator(mode="after")
    def _check_version_specifier(self) -> LibRequirement:
        if not self.version:
            return self
        try:
            SpecifierSet(self.version)
        except InvalidSpecifier as exc:
            raise ValueError(
                f"lib_requirements.version for {self.name!r} is not a valid "
                f"PEP 440 specifier set: {exc}. Examples: '>=1.0.0', "
                f"'>=1.0.0,<2.0.0', '~=1.2', '==1.0.0'."
            ) from exc
        return self


class LibSpec(BaseModel):
    """Body of [lib] section in lib manifest.toml."""

    model_config = ConfigDict(extra="forbid")

    id: str
    version: str = "1.0.0"
    description: str = ""
    #: Free-form attribution. See :class:`PluginSpec.author` — same
    #: shape and recommendation.
    author: str = ""

    @model_validator(mode="after")
    def _check_id_format(self) -> LibSpec:
        if not _LIB_NAME_RE.match(self.id):
            raise ValueError(
                f"lib.id must match {_LIB_NAME_RE.pattern!r} (snake_case "
                f"identifier — same shape as the folder name under "
                f"community/libs/); got {self.id!r}"
            )
        return self

    @model_validator(mode="after")
    def _check_version_format(self) -> LibSpec:
        try:
            Version(self.version)
        except InvalidVersion as exc:
            raise ValueError(
                f"lib {self.id!r} version {self.version!r} is not a valid "
                f"PEP 440 version: {exc}"
            ) from exc
        return self


class LibManifest(BaseModel):
    """Full lib manifest loaded from ``community/libs/<name>/manifest.toml``.

    Lib manifests evolve on their own ``schema_version`` axis (see
    :data:`LATEST_LIB_SCHEMA_VERSION`); changes here don't affect the
    plugin manifest schema number.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(default=LATEST_LIB_SCHEMA_VERSION)
    lib: LibSpec

    @model_validator(mode="after")
    def _check_schema_version(self) -> LibManifest:
        if self.schema_version < 1:
            raise ValueError(
                f"lib schema_version must be >= 1; got {self.schema_version}"
            )
        if self.schema_version > LATEST_LIB_SCHEMA_VERSION:
            raise ValueError(
                f"lib manifest declares schema_version={self.schema_version}, "
                f"but this RyotenkAI build supports up to "
                f"v{LATEST_LIB_SCHEMA_VERSION}. Upgrade the host to load "
                f"this lib."
            )
        return self

    def ui_manifest(self) -> dict[str, Any]:
        """Flatten into the shape consumed by the web UI / ``/libs`` API."""
        return {
            "schema_version": self.schema_version,
            "id": self.lib.id,
            "version": self.lib.version,
            "description": self.lib.description,
            "author": self.lib.author,
        }


class ParamFieldSchema(BaseModel):
    """Description of a single params/thresholds field.

    One source-of-truth for everything the UI needs to render a field:
    its type, bounds, default, label, help text, and whether the user
    MUST supply a value. Transformed into JSON Schema via
    :func:`field_to_json_schema` when the manifest goes over the wire.

    Invariants (enforced in ``_check_constraints``):

    - ``type == "enum"`` ⇒ ``options`` is a non-empty list.
    - ``options`` is only valid when ``type == "enum"``.
    - ``min`` / ``max`` are only valid for numeric types
      (``integer`` / ``number``). ``min`` ≤ ``max``.
    - ``required=True`` + ``default is not None`` is contradictory —
      reject it so authors pick one intention (see R6 in the plan).
    - ``default`` must fall inside ``options`` / ``[min, max]`` when
      relevant.
    """

    model_config = ConfigDict(extra="forbid")

    type: ParamFieldType
    title: str = ""
    description: str = ""
    default: Any = None
    min: float | int | None = None
    max: float | int | None = None
    options: list[Any] = Field(default_factory=list)
    required: bool = False
    secret: bool = False

    @model_validator(mode="after")
    def _check_constraints(self) -> ParamFieldSchema:
        # enum / options wiring
        if self.type == "enum":
            if not self.options:
                raise ValueError("type='enum' requires a non-empty 'options' list")
        elif self.options:
            raise ValueError("'options' is only valid for type='enum'")

        # min / max only for numeric scalars
        numeric_types = {"integer", "number"}
        if self.min is not None or self.max is not None:
            if self.type not in numeric_types:
                raise ValueError(
                    f"'min'/'max' are only valid for numeric types "
                    f"(integer/number); got type={self.type!r}"
                )
            if self.min is not None and self.max is not None and self.min > self.max:
                raise ValueError(f"'min' ({self.min}) is greater than 'max' ({self.max})")

        # required + default — contradictory
        if self.required and self.default is not None:
            raise ValueError(
                "'required=true' forbids 'default' — pick one: either the "
                "user must supply a value, or you offer a sensible default"
            )

        # default range checks
        if self.default is not None:
            if self.type == "enum" and self.default not in self.options:
                raise ValueError(
                    f"default={self.default!r} is not in options={self.options}"
                )
            if self.type in numeric_types:
                if self.min is not None and self.default < self.min:
                    raise ValueError(f"default={self.default} is below min={self.min}")
                if self.max is not None and self.default > self.max:
                    raise ValueError(f"default={self.default} is above max={self.max}")

        # secret only makes sense for strings (passwords, tokens, URLs)
        if self.secret and self.type != "string":
            raise ValueError("'secret=true' is only valid for type='string'")

        return self


def field_to_json_schema(field: ParamFieldSchema) -> dict[str, Any]:
    """Render one :class:`ParamFieldSchema` as a JSON Schema node."""
    node: dict[str, Any] = {}
    # enum → string with an enum list (standard JSON Schema idiom).
    if field.type == "enum":
        node["type"] = "string"
        node["enum"] = list(field.options)
    else:
        node["type"] = field.type
    if field.title:
        node["title"] = field.title
    if field.description:
        node["description"] = field.description
    if field.default is not None:
        node["default"] = field.default
    if field.min is not None:
        node["minimum"] = field.min
    if field.max is not None:
        node["maximum"] = field.max
    # Custom extension — UI uses this to render <input type="password">
    # and to hide the value from logs. Namespaced to avoid collision
    # with any future standard JSON Schema keyword.
    if field.secret:
        node["x-secret"] = True
    return node


def params_to_json_schema(
    fields: dict[str, ParamFieldSchema],
) -> dict[str, Any]:
    """Render a group of fields as a JSON Schema object."""
    return {
        "type": "object",
        "properties": {name: field_to_json_schema(f) for name, f in fields.items()},
        "required": sorted(name for name, f in fields.items() if f.required),
        "additionalProperties": False,
    }


class PluginSpec(BaseModel):
    """Body of [plugin] section in plugin manifest.toml."""

    model_config = ConfigDict(extra="forbid")

    id: str
    kind: PluginKind
    name: str = ""
    version: str = "1.0.0"
    category: str = ""
    stability: Stability = "stable"
    description: str = ""
    #: Free-form attribution string for the plugin author. Recommended
    #: format ``"Name <email>"`` (Cargo / npm convention) so downstream
    #: tooling can split it cheaply, but enforcement stops at "non-empty
    #: when set" — multi-author plugins or ones owned by a team can use
    #: whatever form fits. Empty string means "no author declared".
    author: str = ""
    entry_point: EntryPoint
    #: For kind="reward" this MUST be a non-empty list of strategy types
    #: the plugin is compatible with (e.g. ``["grpo", "sapo"]``). For
    #: every other kind the field MUST be empty — a validation plugin
    #: has no business declaring strategy compatibility.
    supported_strategies: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _reject_v4_libs_shape(cls, data: Any) -> Any:
        """Catch v4-style ``[plugin].libs = [...]`` with a migration hint.

        v4 shipped a string-list field on the plugin block; v5 moved
        that to a top-level ``[[lib_requirements]]`` array of tables
        with optional version constraints. A bare Pydantic
        ``extra="forbid"`` failure here would just say "libs not
        permitted" — actionable, but not super helpful for an author
        upgrading from a working v4 manifest.
        """
        if isinstance(data, dict) and "libs" in data:
            raise ValueError(
                "[plugin].libs was removed in schema v5. Move each entry "
                "into a top-level [[lib_requirements]] block with an "
                "optional `version` specifier:\n\n"
                "  [[lib_requirements]]\n"
                "  name = \"helixql\"\n"
                "  # version = \">=1.0.0\"  # optional PEP 440 specifier"
            )
        return data

    @model_validator(mode="after")
    def _fill_name(self) -> PluginSpec:
        if not self.name:
            object.__setattr__(self, "name", self.id)
        return self

    @model_validator(mode="after")
    def _check_supported_strategies(self) -> PluginSpec:
        if self.kind == "reward":
            if not self.supported_strategies:
                raise ValueError(
                    "reward plugins must declare 'supported_strategies' "
                    "(e.g. ['grpo', 'sapo']) — this field is how the UI "
                    "knows which strategies can attach this plugin"
                )
        elif self.supported_strategies:
            raise ValueError(
                f"'supported_strategies' is only meaningful for reward plugins; "
                f"plugin kind is {self.kind!r}"
            )
        return self


class PluginManifest(BaseModel):
    """Full plugin manifest loaded from manifest.toml."""

    model_config = ConfigDict(extra="forbid")

    #: Manifest schema version. Optional; missing values are treated as
    #: :data:`LATEST_SCHEMA_VERSION`. Higher-than-supported numbers are
    #: rejected with a clear error pointing the user at RyotenkAI upgrade.
    schema_version: int = Field(default=LATEST_SCHEMA_VERSION)
    plugin: PluginSpec
    params_schema: dict[str, ParamFieldSchema] = Field(default_factory=dict)
    thresholds_schema: dict[str, ParamFieldSchema] = Field(default_factory=dict)
    suggested_params: dict[str, Any] = Field(default_factory=dict)
    suggested_thresholds: dict[str, Any] = Field(default_factory=dict)
    compat: CompatSpec = Field(default_factory=CompatSpec)
    required_env: list[RequiredEnvSpec] = Field(
        default_factory=list,
        description=(
            "Environment variables this plugin needs at runtime. The web UI "
            "renders inputs for these in the Configure modal; values are "
            "persisted into the project's env.json (the same place general "
            "project env vars live)."
        ),
    )
    lib_requirements: list[LibRequirement] = Field(
        default_factory=list,
        description=(
            "Community libs this plugin imports from, with optional PEP 440 "
            "version constraints. Each entry's ``name`` must resolve to a "
            "package under community/libs/; the loader version-checks each "
            "satisfied lib against its manifest before the plugin loads."
        ),
    )

    @model_validator(mode="after")
    def _check_lib_requirements_unique(self) -> PluginManifest:
        names = [req.name for req in self.lib_requirements]
        if len(set(names)) != len(names):
            seen: set[str] = set()
            dups = sorted(
                {n for n in names if (n in seen) or seen.add(n)}  # type: ignore[func-returns-value]
            )
            raise ValueError(
                f"duplicate lib_requirements names: {dups}. Merge into a "
                f"single entry — version constraints can be combined with "
                f"a comma, e.g. version=\">=1.0.0,<2.0.0\"."
            )
        return self

    @model_validator(mode="after")
    def _check_schema_version(self) -> PluginManifest:
        if self.schema_version < 1:
            raise ValueError(
                f"schema_version must be >= 1; got {self.schema_version}"
            )
        if self.schema_version > LATEST_SCHEMA_VERSION:
            raise ValueError(
                f"manifest declares schema_version={self.schema_version}, "
                f"but this RyotenkAI build supports up to v{LATEST_SCHEMA_VERSION}. "
                f"Upgrade the host to load this plugin."
            )
        return self

    @model_validator(mode="after")
    def _check_suggested_against_schema(self) -> PluginManifest:
        self._assert_keys_subset(
            self.suggested_params, self.params_schema, label="suggested_params"
        )
        self._assert_keys_subset(
            self.suggested_thresholds,
            self.thresholds_schema,
            label="suggested_thresholds",
        )
        return self

    @model_validator(mode="after")
    def _check_schema_field_names(self) -> PluginManifest:
        """Reject param/threshold field names that aren't valid Python
        identifiers.

        ``params_schema.min-samples`` would render fine as JSON Schema
        but breaks any code that wants to access ``self.params["…"]``
        through an attribute, populate a TypedDict from the schema, or
        emit a working scaffold. We force ``snake_case`` at the
        manifest layer so every downstream tool can rely on it.
        """
        for label, schema in (
            ("params_schema", self.params_schema),
            ("thresholds_schema", self.thresholds_schema),
        ):
            invalid = sorted(k for k in schema if not _PARAM_FIELD_NAME_RE.match(k))
            if invalid:
                raise ValueError(
                    f"{label} field names must match {_PARAM_FIELD_NAME_RE.pattern!r} "
                    f"(snake_case Python identifiers); offenders: {invalid}"
                )
        return self

    @staticmethod
    def _assert_keys_subset(
        suggestions: dict[str, Any],
        schema: dict[str, ParamFieldSchema],
        *,
        label: str,
    ) -> None:
        if not schema:
            return
        extra = set(suggestions) - set(schema)
        if extra:
            raise ValueError(
                f"{label} keys {sorted(extra)} are not declared in the corresponding schema"
            )

    def required_secret_names(self) -> tuple[str, ...]:
        """Return the names of envs the runtime treats as required secrets.

        These are the ``[[required_env]]`` entries with both ``secret=true``
        and ``optional=false``. The community loader uses this to populate
        the runtime ``_required_secrets`` ClassVar that the per-kind
        :class:`PluginRegistry` consults during ``instantiate()`` to inject
        ``instance._secrets``.

        Optional or non-secret envs are *intentionally* excluded — those
        are fetched lazily by the plugin via ``_env(name)`` (B2) when the
        helper lands. The Configure modal still surfaces them all.
        """
        return tuple(
            spec.name
            for spec in self.required_env
            if spec.secret and not spec.optional
        )

    def ui_manifest(self) -> dict[str, Any]:
        """Flatten into the shape consumed by the web UI / ``/plugins`` API.

        ``params_schema`` / ``thresholds_schema`` come out as JSON Schema
        ``object`` nodes ready to drop into the same ``FieldRenderer``
        that powers the main config builder.
        """
        return {
            "schema_version": self.schema_version,
            "id": self.plugin.id,
            "name": self.plugin.name,
            "version": self.plugin.version,
            "description": self.plugin.description,
            "category": self.plugin.category,
            "stability": self.plugin.stability,
            "kind": self.plugin.kind,
            "supported_strategies": list(self.plugin.supported_strategies),
            "author": self.plugin.author,
            "params_schema": params_to_json_schema(self.params_schema),
            "thresholds_schema": params_to_json_schema(self.thresholds_schema),
            "suggested_params": dict(self.suggested_params),
            "suggested_thresholds": dict(self.suggested_thresholds),
            "required_env": [spec.model_dump() for spec in self.required_env],
            "lib_requirements": [
                req.model_dump() for req in self.lib_requirements
            ],
        }


class PresetScope(BaseModel):
    """Which top-level config keys the preset owns vs leaves alone.

    Apply semantics (see :mod:`src.community.preset_apply`):

    - ``replaces``  — these top-level keys are fully overwritten from the
      preset YAML. Anything the user had under these keys is discarded.
    - ``preserves`` — these keys are kept untouched; values present in the
      preset YAML for these keys are ignored.
    - Keys in the preset YAML that are in neither list fall through to
      ``replaces`` semantics (safe default: preset is authoritative over
      whatever it bothered to write down).

    When the whole ``[preset.scope]`` block is omitted, the preset runs
    in v1 compatibility mode — full replacement of the user config.
    """

    model_config = ConfigDict(extra="forbid")

    replaces: list[str] = Field(default_factory=list)
    preserves: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _no_overlap(self) -> PresetScope:
        overlap = set(self.replaces) & set(self.preserves)
        if overlap:
            raise ValueError(
                f"scope.replaces and scope.preserves overlap on {sorted(overlap)}"
            )
        return self


class PresetRequirements(BaseModel):
    """What the user's environment must provide for the preset to make sense."""

    model_config = ConfigDict(extra="forbid")

    # HF Hub models referenced by the preset; gated ones imply HF_TOKEN.
    hub_models: list[str] = Field(default_factory=list)
    # Preferred provider kinds (informational). Empty → preset is neutral.
    provider_kind: list[str] = Field(default_factory=list)
    # Community plugin ids the preset expects to be loaded (``kind:id``).
    required_plugins: list[str] = Field(default_factory=list)
    # Informational hardware hint (displayed in UI; not enforced).
    min_vram_gb: int | None = None


class PresetSpec(BaseModel):
    """Body of [preset] section in preset manifest.toml."""

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str = ""
    description: str = ""
    size_tier: str = ""
    version: str = "1.0.0"
    entry_point: PresetEntryPoint
    # v2 contract — all three blocks are optional; absent = v1 full replace.
    scope: PresetScope | None = None
    requirements: PresetRequirements | None = None
    # Dotted JSONPath → hint. Surfaced in the preview modal so the user
    # knows which fields *they* must fill after applying.
    placeholders: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _fill_name(self) -> PresetSpec:
        if not self.name:
            object.__setattr__(self, "name", self.id)
        return self


class PresetManifest(BaseModel):
    """Full preset manifest loaded from manifest.toml."""

    model_config = ConfigDict(extra="forbid")

    preset: PresetSpec


__all__ = [
    "CompatSpec",
    "EntryPoint",
    "LATEST_LIB_SCHEMA_VERSION",
    "LATEST_SCHEMA_VERSION",
    "LibManifest",
    "LibRequirement",
    "LibSpec",
    "ParamFieldSchema",
    "ParamFieldType",
    "PluginKind",
    "PluginManifest",
    "PluginSpec",
    "PresetEntryPoint",
    "PresetManifest",
    "PresetRequirements",
    "PresetScope",
    "PresetSpec",
    "RequiredEnvSpec",
    "Stability",
    "field_to_json_schema",
    "params_to_json_schema",
]
