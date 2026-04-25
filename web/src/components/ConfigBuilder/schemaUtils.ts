import type { JsonSchemaNode, PipelineJsonSchema } from '../../api/hooks/useConfigSchema'

/**
 * Resolve a single ``$ref`` step. Supports only JSON-pointer refs into the
 * schema root (``#/$defs/Foo`` and the Pydantic alias ``#/definitions/Foo``).
 * Returns the target node or the original when not resolvable.
 */
export function resolveRef(root: PipelineJsonSchema, node: JsonSchemaNode): JsonSchemaNode {
  if (!node || typeof node !== 'object') return node
  const ref = node.$ref
  if (typeof ref !== 'string' || !ref.startsWith('#/')) return node
  const segments = ref.slice(2).split('/')
  let cursor: unknown = root
  for (const seg of segments) {
    if (cursor && typeof cursor === 'object' && seg in (cursor as Record<string, unknown>)) {
      cursor = (cursor as Record<string, unknown>)[seg]
    } else {
      return node
    }
  }
  if (cursor && typeof cursor === 'object') {
    return { ...(cursor as JsonSchemaNode), ...node, $ref: undefined }
  }
  return node
}

export type FieldKind =
  | 'string'
  | 'number'
  | 'integer'
  | 'boolean'
  | 'enum'
  | 'object'
  | 'array'
  | 'union'
  | 'unknown'

export function detectKind(node: JsonSchemaNode): FieldKind {
  if (!node) return 'unknown'
  if (Array.isArray(node.enum) && node.enum.length > 0) return 'enum'
  const type = Array.isArray(node.type) ? node.type[0] : node.type
  if (type === 'object') return 'object'
  if (type === 'array') return 'array'
  if (type === 'integer') return 'integer'
  if (type === 'number') return 'number'
  if (type === 'boolean') return 'boolean'
  if (type === 'string') return 'string'
  if (Array.isArray(node.anyOf) && node.anyOf.length > 0) {
    const consts = node.anyOf.filter((n) => 'const' in n)
    if (consts.length === node.anyOf.length) return 'enum'
    return 'union'
  }
  return 'unknown'
}

/**
 * Humanise a JSON-schema title that comes straight from a Pydantic class
 * name. Strips the ``Config`` suffix so dropdowns/headers read ``Lora``
 * instead of ``LoraConfig``. Multi-word class names keep their spacing
 * via CamelCase splitting.
 */
export function prettifyTitle(raw: string): string {
  const stripped = raw.replace(/Config$/, '').trim()
  if (!stripped) return raw
  return stripped.replace(/([a-z])([A-Z])/g, '$1 $2')
}

export function titleOrKey(node: JsonSchemaNode, key: string): string {
  if (typeof node?.title === 'string' && node.title.length > 0) {
    return prettifyTitle(node.title)
  }
  return key
}

export function getDefault(node: JsonSchemaNode): unknown {
  return Object.prototype.hasOwnProperty.call(node ?? {}, 'default') ? node.default : undefined
}

/**
 * Preferred ordering for PipelineConfig top-level groups in the UI. Keys
 * not listed here fall through alphabetically at the end so unknown
 * future groups still render. Applied by ConfigBuilder + GroupSubtabs +
 * TocRail via {@link orderTopLevelKeys}.
 */
const TOP_LEVEL_ORDER = [
  'model',
  'training',
  'inference',
  'evaluation',
  'experiment_tracking',
]

/**
 * Human-friendly labels for top-level groups. Overrides the JSON-schema
 * ``title`` (which is the Pydantic class name, e.g. ``TrainingOnlyConfig``).
 */
const TOP_LEVEL_LABELS: Record<string, string> = {
  model: 'Model',
  training: 'Training',
  inference: 'Inference',
  evaluation: 'Evaluation',
  experiment_tracking: 'Experiment tracking',
  // Provider-scoped sections — reuse the same label path so the
  // ConfigBuilder section header reads "Connect" / "Cleanup" instead
  // of the bare YAML key.
  connect: 'Connect',
  cleanup: 'Cleanup',
}

/**
 * Top-level keys whose editors live outside the config form — selected
 * from Settings via pickers instead of being edited inline. Kept in the
 * saved value so round-trips are lossless; just hidden from the UI.
 */
const HIDDEN_TOP_LEVEL_KEYS = new Set<string>(['providers', 'datasets'])

export function topLevelLabel(key: string, fallback?: string): string {
  return TOP_LEVEL_LABELS[key] ?? fallback ?? key
}

export function orderTopLevelKeys(keys: string[]): string[] {
  const index = new Map(TOP_LEVEL_ORDER.map((k, i) => [k, i]))
  return [...keys].sort((a, b) => {
    const ai = index.get(a) ?? TOP_LEVEL_ORDER.length
    const bi = index.get(b) ?? TOP_LEVEL_ORDER.length
    if (ai !== bi) return ai - bi
    return a.localeCompare(b)
  })
}

/** Ordered keys with hidden groups filtered out. */
export function visibleTopLevelKeys(keys: string[]): string[] {
  return orderTopLevelKeys(keys).filter((k) => !HIDDEN_TOP_LEVEL_KEYS.has(k))
}

/**
 * Per-object required/visibility overrides, keyed by dotted path from
 * schema root. Paths match the ``pathPrefix`` threaded through
 * FieldRenderer/ObjectFields (array indices are real numbers, but
 * overrides here target object containers so no wildcard is needed).
 *
 * Semantics:
 * - ``requires``: promoted to the top of the form with a red asterisk.
 * - ``optional``: forced out of the original Pydantic ``required`` set
 *   (no asterisk).
 * - ``alwaysVisible``: pinned under the required fields but without an
 *   asterisk — use for fields the user should see with their defaults
 *   without clicking "Show optional".
 */
export interface RequiredOverride {
  requires?: string[]
  optional?: string[]
  alwaysVisible?: string[]
  /** Reorder fields within the rendered form. Unlisted keys keep their
   * original JSON-schema order after the listed ones. */
  fieldOrder?: string[]
  /** Skip the "Show N optional fields" toggle and render every field
   * inline. Useful for leaf configs where every knob is relevant once
   * the user has opened the section. */
  expandOptional?: boolean
  /** Collapsible wrapping this object starts closed (default: open). */
  defaultCollapsed?: boolean
  /** Hide these fields from the UI entirely (they stay in the value on
   *  round-trip so saved YAML keeps them). Use when a field is managed
   *  elsewhere or is out of scope for a given mode. */
  hidden?: string[]
  /** When the object has exactly one child and that child is itself an
   *  object, render the grandchildren inline instead of wrapping them
   *  in a ``CollapsibleCard``. Useful for provider ``connect`` blocks
   *  where the only field is ``ssh`` — the extra accordion reads as
   *  noise. The single-child key is preserved in the saved YAML. */
  inlineSingleChild?: boolean
}

/**
 * Value-aware override: invoked with the current object value so rules
 * can depend on sibling flags (e.g. ``inference.enabled=true`` promotes
 * provider/engine to required).
 */
type RequiredOverrideFn = (
  currentValue: Record<string, unknown>,
) => RequiredOverride | undefined

const REQUIRED_OVERRIDES: Record<string, RequiredOverride | RequiredOverrideFn> = {
  training: {
    requires: ['provider', 'type', 'strategies'],
    optional: ['hyperparams'],
    alwaysVisible: ['hyperparams'],
    // Hyperparams sits above strategies (per user request) while Type +
    // active branch still lead the form. The active branch name is
    // unknown at override time, so the discriminator bump still applies
    // on top of this order.
    fieldOrder: ['type', 'provider', 'hyperparams', 'strategies'],
  },
  // Strategy phase item: dataset is promoted to required (it's the
  // primary knob alongside strategy_type). Stays above the "Show N
  // optional" toggle and gets the red asterisk.
  //
  // `params` is hidden from the form: it's a free-form `dict[str, Any]`
  // owned by plugins (e.g. `reward_plugin: id` written by the Plugins
  // tab) — there's no schema for the user to fill in by hand here, and
  // surfacing an empty `Params {}` row was confusing. Power users can
  // still edit it via the YAML view; the value round-trips losslessly.
  'training.strategies.*': {
    requires: ['dataset'],
    fieldOrder: ['strategy_type', 'dataset'],
    hidden: ['params'],
  },
  // Phase-level leaf configs: once the user opens them, every knob is
  // relevant. Skip the "Show N optional" dance. Also collapse by
  // default — they're secondary to the strategy_type/dataset the user
  // tweaks first.
  'training.strategies.*.hyperparams': {
    expandOptional: true,
    defaultCollapsed: true,
  },
  'training.strategies.*.adapter_cache': {
    expandOptional: true,
    defaultCollapsed: true,
  },
  inference: (v) => {
    // PR3: ``enabled`` is hidden — the FE derives it from ``provider``
    // and writes it on every selection change (see
    // ``InferenceProviderField``), so it stays consistent in YAML.
    const active = typeof v.provider === 'string' && v.provider.length > 0
    if (active) {
      return {
        requires: ['provider', 'engine'],
        optional: ['common', 'engines'],
        alwaysVisible: ['common', 'engines'],
        fieldOrder: ['provider', 'engine', 'common', 'engines'],
        expandOptional: true,
        hidden: ['enabled'],
      }
    }
    // Inactive: only the provider picker is visible — picking one reveals
    // the rest of the form.
    return {
      requires: ['provider'],
      optional: ['engine', 'engines', 'common'],
      fieldOrder: ['provider', 'engine', 'common', 'engines'],
      hidden: ['enabled'],
    }
  },
  // Inference leaf configs: knobs only, no sub-sections worth hiding.
  // Collapsed by default; once opened, no "Show N optional" dance.
  'inference.common': { expandOptional: true, defaultCollapsed: true },
  'inference.common.health_check': { expandOptional: true },
  'inference.common.lora': { expandOptional: true },
  'inference.common.chat_ui': { expandOptional: true },
  'inference.engines': { expandOptional: true, defaultCollapsed: true },
  'inference.engines.vllm': { expandOptional: true },
  // Evaluation mirrors Inference: enabled flag gates the rest. Evaluators
  // is managed server-side (plugins registry) — hidden from the form. The
  // dataset block is a single-field wrapper (`path`) so we auto-expand it.
  evaluation: (v) => {
    const enabled = v.enabled === true
    if (enabled) {
      return {
        requires: ['enabled', 'dataset'],
        alwaysVisible: ['save_answers_md'],
        fieldOrder: ['enabled', 'save_answers_md', 'dataset'],
        hidden: ['evaluators'],
      }
    }
    return {
      requires: ['enabled'],
      optional: ['dataset'],
      fieldOrder: ['enabled', 'dataset'],
      hidden: ['evaluators', 'save_answers_md'],
    }
  },
  'evaluation.dataset': { expandOptional: true, defaultCollapsed: false },
  // Provider sections — once the user is on the dedicated TocRail
  // tab for a section, show everything immediately. Hiding fields
  // behind a second "Show N optional fields" toggle is pointless when
  // the user already navigated *into* the section.
  cleanup: { expandOptional: true },
  // Provider ``connect`` typically contains a single ``ssh`` sub-object;
  // wrapping it in a CollapsibleCard adds a header that reads as noise.
  // ``inlineSingleChild`` promotes the grandchildren to top-level so the
  // user sees the SSH fields directly. ``expandOptional`` prevents the
  // optional-fields toggle.
  connect: { inlineSingleChild: true, expandOptional: true },
  // Experiment tracking: both backends are optional in the schema, but
  // the whole tab is empty otherwise. Pin them as always-visible so the
  // user sees the available backends immediately on tab entry.
  experiment_tracking: {
    alwaysVisible: ['mlflow', 'huggingface'],
    fieldOrder: ['mlflow', 'huggingface'],
  },
  // MLflow block — PR3: only ``integration`` is visible until a value
  // is picked; ``experiment_name`` (+ optional ``run_description_file``)
  // are revealed once an integration is chosen.
  'experiment_tracking.mlflow': (v) => {
    const active = typeof v.integration === 'string' && v.integration.length > 0
    if (active) {
      return {
        requires: ['integration', 'experiment_name'],
        alwaysVisible: ['run_description_file'],
        fieldOrder: ['integration', 'experiment_name', 'run_description_file'],
        expandOptional: true,
      }
    }
    return {
      requires: ['integration'],
      optional: ['experiment_name', 'run_description_file'],
      fieldOrder: ['integration', 'experiment_name', 'run_description_file'],
      hidden: ['experiment_name', 'run_description_file'],
    }
  },
  // HuggingFace block — PR3: no more ``enabled`` flag. Only the
  // ``integration`` picker is visible until a value is selected; then
  // ``repo_id`` + ``private`` appear.
  'experiment_tracking.huggingface': (v) => {
    const active = typeof v.integration === 'string' && v.integration.length > 0
    if (active) {
      return {
        requires: ['integration', 'repo_id'],
        alwaysVisible: ['private'],
        fieldOrder: ['integration', 'repo_id', 'private'],
        expandOptional: true,
      }
    }
    return {
      requires: ['integration'],
      optional: ['repo_id', 'private'],
      fieldOrder: ['integration', 'repo_id', 'private'],
      hidden: ['repo_id', 'private'],
    }
  },
}

export function getRequiredOverride(
  path: string,
  currentValue: Record<string, unknown>,
): RequiredOverride | undefined {
  const key = normalizeArrayPath(path)
  const entry = REQUIRED_OVERRIDES[key]
  if (typeof entry === 'function') return entry(currentValue)
  return entry
}

/**
 * Per-field renderer overrides, keyed by wildcard dotted path. Numeric
 * array indices in the real path are normalised to ``*`` before lookup,
 * so ``training.strategies.0.dataset`` matches
 * ``training.strategies.*.dataset``.
 */
export interface FieldOverride {
  /** Force enum rendering with this value list, regardless of schema kind. */
  enumValues?: string[]
  /** Render a disabled "coming soon" placeholder instead of an input. */
  comingSoon?: string
}

/**
 * Tell the UI which enum values of a soft-discriminator map to sibling
 * branches, so helpers aren't misread as branches. Pydantic often emits
 * the discriminator as ``type: str`` (not ``Literal[...]``) with the
 * values listed in the description; without this override, every
 * object-like sibling (including ``hyperparams``) is treated as a branch
 * and gets hidden whenever the active type switches.
 */
export interface DiscriminatorOverride {
  enumKey: string
  values: string[]
}

const DISCRIMINATOR_OVERRIDES: Record<string, DiscriminatorOverride> = {
  training: { enumKey: 'type', values: ['lora', 'qlora', 'adalora'] },
}

export function getDiscriminatorOverride(path: string): DiscriminatorOverride | undefined {
  return DISCRIMINATOR_OVERRIDES[path]
}

const FIELD_OVERRIDES: Record<string, FieldOverride> = {
  'model.torch_dtype': {
    enumValues: ['auto', 'bfloat16', 'float16', 'float32'],
  },
  'training.strategies.*.strategy_type': {
    enumValues: ['cpt', 'sft', 'cot', 'dpo', 'orpo', 'grpo', 'sapo'],
  },
  'training.strategies.*.dataset': {
    comingSoon: 'Dataset selection — coming soon',
  },
  // target_modules is typed as `str | list[str]` in the schema which the
  // union path otherwise renders as a JSON preview. Pin it to a known
  // string enum so it's an editable dropdown.
  'training.lora.target_modules': {
    enumValues: ['all-linear', 'q_proj,k_proj,v_proj,o_proj'],
  },
  'training.qlora.target_modules': {
    enumValues: ['all-linear', 'q_proj,k_proj,v_proj,o_proj'],
  },
  'training.adalora.target_modules': {
    enumValues: ['all-linear', 'q_proj,k_proj,v_proj,o_proj'],
  },
  'inference.engine': {
    enumValues: ['vllm'],
  },
  // inference.provider is rendered by InferenceProviderField
  // (CUSTOM_FIELD_RENDERERS) — sourced from Settings providers with a
  // populated inference block. No hard-coded enum here.
}

function normalizeArrayPath(path: string): string {
  return path
    .split('.')
    .map((seg) => (/^\d+$/.test(seg) ? '*' : seg))
    .join('.')
}

export function getFieldOverride(path: string): FieldOverride | undefined {
  return FIELD_OVERRIDES[normalizeArrayPath(path)]
}
