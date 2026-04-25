import { useState } from 'react'
import type { JsonSchemaNode, PipelineJsonSchema } from '../../api/hooks/useConfigSchema'
import { ArrayField } from './ArrayField'
import { DatasetSelectField } from './DatasetSelectField'
import { FieldAnchor } from './FieldAnchor'
import { HelpTooltip } from './HelpTooltip'
import { AlertIcon } from '../icons'
import { HFModelField } from './HFModelField'
import { HuggingFaceIntegrationField } from './HuggingFaceIntegrationField'
import { InferenceProviderField } from './InferenceProviderField'
import { MLflowIntegrationField } from './MLflowIntegrationField'
import { SelectField } from './SelectField'
import { TrainingProviderField } from './TrainingProviderField'
import { UnionField } from './UnionField'
import { useClientFieldValidation, useFieldStatus, useValidationCtx } from './ValidationContext'
import type { FieldStatus } from './ValidationContext'
import { Toggle } from '../ui'

/**
 * Per-path custom components for fields that can't be described by the
 * generic schema-driven renderer (e.g. ones that need live Settings
 * data). Keyed by dotted path; numeric array indices are normalised to
 * ``*`` via the same path-normaliser used for FIELD_OVERRIDES.
 */
type CustomFieldProps = {
  value: unknown
  onChange: (next: unknown) => void
  /** Forwarded focus/blur so the validation context can track which
   *  field is currently being edited. Custom renderers can wire these
   *  into their main focusable element (input / trigger). */
  onFocus?: () => void
  onBlur?: () => void
  /** The whole form value — only needed by renderers that cross-edit
   *  sibling fields (e.g. ``inference.provider`` keeping
   *  ``inference.enabled`` in sync). Optional; other renderers ignore. */
  rootValue?: Record<string, unknown>
  onRootChange?: (next: Record<string, unknown>) => void
}
const CUSTOM_FIELD_RENDERERS: Record<
  string,
  React.ComponentType<CustomFieldProps>
> = {
  'training.provider': TrainingProviderField,
  'inference.provider': InferenceProviderField,
  'model.name': HFModelField,
  'experiment_tracking.mlflow.integration': MLflowIntegrationField,
  'experiment_tracking.huggingface.integration': HuggingFaceIntegrationField,
  // Every strategy phase gets a dataset picker instead of a free-text
  // input. Numeric array indices are normalised to `*` by the path
  // normaliser below, so the key matches every element in the chain.
  'training.strategies.*.dataset': DatasetSelectField,
}
import {
  detectKind,
  getDefault,
  getDiscriminatorOverride,
  getFieldOverride,
  getRequiredOverride,
  resolveRef,
  titleOrKey,
} from './schemaUtils'

type Setter = (value: unknown) => void

type FieldProps = {
  root: PipelineJsonSchema
  node: JsonSchemaNode
  value: unknown
  onChange: Setter
  labelKey: string
  required?: boolean
  depth?: number
  path?: string
  hashPrefix?: string
  /** The whole form value — threaded down so cross-editing custom
   *  renderers (e.g. ``inference.provider`` syncing
   *  ``inference.enabled``) can update sibling fields atomically. */
  rootValue?: Record<string, unknown>
  onRootChange?: (next: Record<string, unknown>) => void
}

/**
 * Linear/Stripe-style config row: label-left as plain text (no pill,
 * no background, no border) + input-right on `surface-inset` (вдавлено).
 *
 * Why no label-pill: pre-redesign the label sat in a 220px box with its
 * own border + bg-surface-1, which (a) added a 4th-level border on every
 * row, (b) made the row feel like a separate chunk, (c) created two
 * competing design languages on screens that mix scalar rows + cards.
 * Plain-text labels read like Linear/Vercel/Grafana settings — chrome
 * disappears, the eye scans labels and inputs on their own merits.
 *
 * Visual rules:
 *   - idle / ok / editing → no extra paint. Input contrast (вдавленный
 *     surface-inset) + focus-ring carries the affordance.
 *   - error → red ring on the input only, plus inline message below.
 *     The label asterisk turns red so the row reads as "errored" at a
 *     glance even when the user is scrolled past the input.
 */
// Error styling targets actual controls (input / select / textarea /
// toggle / listbox-button), NOT the wrapping div. Custom renderers
// often wrap the control in a flex container that fills the row —
// applying the ring on the wrapper makes a small 32 px field look
// like a giant errored block. Same logic as `.field-attention-pulse`
// in globals.css.
const STATUS_RING: Record<FieldStatus['state'], string> = {
  idle: '',
  editing: '',
  ok: '',
  error: 'cfg-error-target',
}

function LabelledRow({
  label,
  description,
  required,
  path,
  value,
  suppressBar,
  children,
}: {
  label: string
  description?: string
  required?: boolean
  path?: string
  value?: unknown
  /** Skip the input-side error ring. Used for checkboxes where a ring
   *  around a 16×16 box reads as a glitch. */
  suppressBar?: boolean
  children: React.ReactNode
}) {
  const status = useFieldStatus(path ?? '', Boolean(required), value)
  const ctx = useValidationCtx()
  const ring = path && !suppressBar ? STATUS_RING[status.state] : STATUS_RING.idle
  // Pulse halo on the INPUT side only — pre-redesign it cupped both
  // label pill and input wrapper, which felt loud. The single yellow
  // glow around the input is enough signal "look here", and it falls
  // off the moment the user focuses the field.
  const isFocused = path != null && ctx?.focusedPath === path
  const pulseCls =
    path && status.state === 'error' && !isFocused
      ? 'field-attention-pulse'
      : ''
  const inputId = path ? `cfg-${path.replace(/[^a-zA-Z0-9_-]/g, '_')}` : undefined
  const helpLabel = path ? `Help for ${label}` : 'Field help'
  // Required asterisk: muted brand-warm by default, flips to err when
  // the row is in error state — single visual signal that travels with
  // the label as the user scrolls.
  const asteriskCls =
    status.state === 'error' ? 'text-err' : 'text-brand-warm'
  return (
    <div className="py-1.5">
      <div className="grid grid-cols-1 sm:grid-cols-[200px_minmax(0,1fr)] gap-1 sm:gap-4 items-start sm:items-center">
        <div className="flex items-center gap-1.5 min-w-0 h-8 px-0.5">
          <label
            htmlFor={inputId}
            className="flex-1 min-w-0 text-xs text-ink-2 tracking-tight truncate"
          >
            {label}
            {required && (
              <>
                <span aria-hidden className={`ml-0.5 ${asteriskCls}`}>*</span>
                <span className="sr-only"> (required)</span>
              </>
            )}
          </label>
          {status.state === 'error' && (
            // Hover-triggered tooltip for the inline error message —
            // native `title` was too slow (~1.5 s delay) and easy to
            // miss. The popover lives inside the label cell and shows
            // up immediately on hover/focus, mirroring HelpTooltip's
            // visual style for consistency.
            <span className="relative inline-flex group/err shrink-0">
              <span
                aria-label={`Error: ${status.message ?? ''}`}
                role="img"
                tabIndex={0}
                className="inline-flex w-4 h-4 items-center justify-center text-err cursor-help focus:outline-none focus:ring-1 focus:ring-err/40 rounded"
              >
                <AlertIcon className="w-3.5 h-3.5" />
              </span>
              {status.message && (
                <span
                  role="tooltip"
                  className="pointer-events-none absolute left-0 top-full mt-1.5 z-30 w-72 max-w-[calc(100vw-2rem)] rounded-md border border-err/40 bg-surface-4 px-3 py-2 text-[0.7rem] leading-snug text-ink-1 shadow-card whitespace-normal break-words opacity-0 group-hover/err:opacity-100 group-focus-within/err:opacity-100 transition-opacity"
                >
                  <span className="text-err font-medium">Error:</span> {status.message}
                </span>
              )}
            </span>
          )}
          <HelpTooltip text={description} label={helpLabel} />
        </div>
        <div
          id={inputId}
          className={`w-full min-w-0 ${ring} ${pulseCls} transition-colors`}
          aria-invalid={status.state === 'error' || undefined}
          aria-describedby={
            status.state === 'error' && inputId ? `${inputId}-err` : undefined
          }
        >
          {children}
        </div>
      </div>
      {status.state === 'error' && status.message && (
        <div
          id={inputId ? `${inputId}-err` : undefined}
          className="mt-1 ml-0 sm:ml-[216px] text-[0.65rem] text-err font-mono break-words"
        >
          {status.message}
        </div>
      )}
    </div>
  )
}

/**
 * Focus/blur wiring for a single input. Reports focus entry/exit to the
 * validation context so the field can flip yellow while edited + red
 * once the user moves on from a required-but-empty field. Blur also
 * triggers a server re-validate (debounced inside the provider).
 */
function useFieldHandlers(path: string) {
  const ctx = useValidationCtx()
  if (!ctx) return {}
  return {
    onFocus: () => ctx.setFocusedPath(path),
    onBlur: () => {
      ctx.setFocusedPath(null)
      ctx.markDirty(path)
      ctx.requestValidate()
    },
  } as const
}

// Dense input baseline: 32px height, 13px text, monospace for values.
// Linear/repowise-flat: input sits on `surface-inset` (DARKER than the
// surrounding card surface), which makes it read as вдавленный — the
// contrast itself signals "edit me" without a heavy border. Border drops
// to hairline line-1, focus lifts to brand-violet without any coloured
// background. Eye scans values, not chrome.
const INPUT_BASE =
  'h-8 rounded-md bg-surface-inset border border-line-1 px-2.5 text-[13px] text-ink-1 font-mono focus:outline-none focus:border-brand hover:border-line-2 transition-colors placeholder:text-ink-4 placeholder:italic'

const WIDE_NAME_RE = /(path|url|uri|dir|file|repo|image|endpoint|bucket|model|name|prefix|suffix|volume|description|prompt|tracking_uri)/i

/**
 * Choose an input width for a string field. Short semantic names (name,
 * slug, id) get a medium 320px input; paths/URLs/secrets and anything
 * that tends to hold long values go full-width up to ~640px. Keeps
 * short fields from ballooning on wide screens.
 */
function stringWidthClass(key: string, node: JsonSchemaNode): string {
  const format = typeof node.format === 'string' ? node.format : ''
  if (format === 'uri' || format === 'path' || format === 'email') {
    return 'w-full max-w-[640px]'
  }
  if (WIDE_NAME_RE.test(key)) return 'w-full max-w-[640px]'
  return 'w-80'
}

export function FieldRenderer(props: FieldProps) {
  const { root, node: rawNode, value, onChange, labelKey, required, depth = 0 } = props
  const path = props.path ?? labelKey
  const hashPrefix = props.hashPrefix ?? ''
  const rawResolved = resolveRef(root, rawNode)
  // Unwrap nullable scalars: Pydantic ``Optional[str]`` comes through as
  // ``anyOf: [{type:'string'}, {type:'null'}]``, which detectKind sees as
  // a union and the union branch then falls back to the JSON preview.
  // Drop the null branch and keep the scalar so it renders as its own
  // kind. ``undefined`` means "not set" and already round-trips fine.
  const node = unwrapNullableScalar(root, rawResolved)
  const kind = detectKind(node)
  const label = titleOrKey(node, labelKey)
  const description = typeof node.description === 'string' ? node.description : undefined
  const fallback = value === undefined ? getDefault(node) : value

  const wrapAnchor = (el: React.ReactNode) => (
    <FieldAnchor path={path} hashPrefix={hashPrefix}>
      {el}
    </FieldAnchor>
  )
  const focusHandlers = useFieldHandlers(path)
  // Synchronous client-side range/enum/pattern check — merges into the
  // same field status pipe as the server errors, so UI styling doesn't
  // need to know which source produced the message.
  useClientFieldValidation(path, node, fallback)

  const normalizedPath = path
    .split('.')
    .map((seg) => (/^\d+$/.test(seg) ? '*' : seg))
    .join('.')
  const Custom = CUSTOM_FIELD_RENDERERS[normalizedPath]
  if (Custom) {
    return wrapAnchor(
      <LabelledRow label={label} description={description} required={required} path={path} value={fallback}>
        <Custom
          value={fallback}
          onChange={onChange}
          rootValue={props.rootValue}
          onRootChange={props.onRootChange}
          {...focusHandlers}
        />
      </LabelledRow>,
    )
  }

  const fieldOverride = getFieldOverride(path)
  if (fieldOverride?.comingSoon) {
    // Don't surface as required — widget isn't shipped yet, so client
    // can't satisfy it. Server still validates the underlying YAML, so
    // correctness isn't lost. Visual treatment: single inline row with
    // muted text + `pill-skip` chip — same vertical weight as a real
    // field, so it doesn't visually shout "broken section". Help-tooltip
    // carries the "edit in YAML" guidance instead of a 2-line caption.
    const tip =
      `${description ?? fieldOverride.comingSoon}\n\nPicker in progress — switch to YAML view to set this value today.`
    return wrapAnchor(
      <LabelledRow label={label} description={tip} required={false} path={path} value={fallback}>
        <div className="h-8 inline-flex items-center gap-2 text-xs text-ink-3 italic">
          <span>{fieldOverride.comingSoon}</span>
          <span className="pill pill-skip not-italic">soon</span>
        </div>
      </LabelledRow>,
    )
  }
  if (fieldOverride?.enumValues) {
    const current = typeof value === 'string' ? value : ''
    const needsEmpty = !fieldOverride.enumValues.includes(current)
    const schemaDefault = getDefault(node)
    const placeholder =
      typeof schemaDefault === 'string' ? schemaDefault : '—'
    return wrapAnchor(
      <LabelledRow label={label} description={description} required={required} path={path} value={fallback}>
        <SelectField
          value={current}
          options={fieldOverride.enumValues.map((v) => ({ value: v }))}
          onChange={(next) => onChange(next === '' ? undefined : next)}
          allowEmpty={needsEmpty}
          placeholder={placeholder}
          {...focusHandlers}
        />
      </LabelledRow>,
    )
  }

  if (kind === 'boolean') {
    const checked = Boolean(fallback)
    return wrapAnchor(
      <LabelledRow label={label} description={description} required={required} path={path} value={fallback} suppressBar>
        <Toggle
          checked={checked}
          onChange={(next) => onChange(next)}
          aria-label={typeof label === 'string' ? label : path}
          {...focusHandlers}
        />
      </LabelledRow>
    )
  }

  if (kind === 'enum') {
    const options = collectEnumOptions(node)
    // Show user-set value, or empty + schema default as greyed placeholder
    // when the user hasn't picked anything yet. Undefined still serializes
    // as "absent" — Pydantic fills in the default on load.
    const current = typeof value === 'string' ? value : ''
    const schemaDefault = getDefault(node)
    const placeholder =
      schemaDefault !== undefined && schemaDefault !== null
        ? String(schemaDefault)
        : '—'
    return wrapAnchor(
      <LabelledRow label={label} description={description} required={required} path={path} value={fallback}>
        <SelectField
          value={current}
          options={options.map((opt) => ({ value: String(opt) }))}
          onChange={(next) => onChange(next === '' ? undefined : next)}
          allowEmpty
          placeholder={placeholder}
          {...focusHandlers}
        />
      </LabelledRow>,
    )
  }

  if (kind === 'number' || kind === 'integer') {
    const schemaDefault = getDefault(node)
    const placeholder =
      schemaDefault !== undefined && schemaDefault !== null
        ? String(schemaDefault)
        : undefined
    return wrapAnchor(
      <LabelledRow label={label} description={description} required={required} path={path} value={fallback}>
        <input
          type="number"
          step={kind === 'integer' ? 1 : 'any'}
          value={value === undefined || value === null ? '' : String(value)}
          placeholder={placeholder}
          onChange={(e) => {
            if (e.target.value === '') onChange(undefined)
            else
              onChange(
                kind === 'integer' ? Number.parseInt(e.target.value, 10) : Number(e.target.value),
              )
          }}
          {...focusHandlers}
          className={`${INPUT_BASE} w-32`}
        />
      </LabelledRow>,
    )
  }

  if (kind === 'string') {
    const schemaDefault = getDefault(node)
    const placeholder =
      typeof schemaDefault === 'string' ? schemaDefault : undefined
    return wrapAnchor(
      <LabelledRow label={label} description={description} required={required} path={path} value={fallback}>
        <input
          type="text"
          value={typeof value === 'string' ? value : ''}
          placeholder={placeholder}
          onChange={(e) => onChange(e.target.value)}
          {...focusHandlers}
          className={`${INPUT_BASE} ${stringWidthClass(labelKey, node)}`}
        />
      </LabelledRow>,
    )
  }

  if (kind === 'union') {
    const rawBranches = (node.anyOf ?? node.oneOf ?? []) as JsonSchemaNode[]
    const objectBranches = rawBranches
      .map((b) => resolveRef(root, b))
      .filter((b) => (Array.isArray(b.type) ? b.type[0] : b.type) === 'object' || b.properties)
    if (objectBranches.length >= 1) {
      return wrapAnchor(
        <UnionField
          root={root}
          branches={rawBranches}
          value={fallback}
          onChange={onChange}
          label={label}
          required={required}
          renderBranch={(branch) => (
            <ObjectFields
              root={root}
              node={branch}
              value={fallback}
              onChange={onChange}
              depth={depth + 1}
              pathPrefix={path}
              hashPrefix={hashPrefix}
              rootValue={props.rootValue}
              onRootChange={props.onRootChange}
            />
          )}
        />,
      )
    }
    return wrapAnchor(
      <LabelledRow label={label} description={description} required={required} path={path} value={fallback}>
        <AdvancedJsonPreview value={fallback} />
      </LabelledRow>,
    )
  }

  if (kind === 'object') {
    const fields = Object.keys(node.properties ?? {})
    if (fields.length === 0) {
      return (
        <LabelledRow label={label} description={description} required={required} path={path} value={fallback}>
          <AdvancedJsonPreview value={fallback} />
        </LabelledRow>
      )
    }
    if (depth === 0) {
      // Flat render: the active subtab already owns the frame, no need for
      // a nested collapsible card inside it.
      //
      // Header — Linear/Stripe Settings pattern: H2 + one-line caption
      // (the schema description) + hairline divider. The eyebrow
      // ("SECTION") was decoration noise — the left-rail's own
      // "SECTIONS" header already carries that signal, repeating it on
      // the right pane was a tautology. With the eyebrow gone, the H2
      // can sit larger and become the single anchor.
      return (
        <div className="space-y-5">
          <header className="pb-3 border-b border-line-1">
            <div className="flex items-baseline gap-2">
              <h2 className="text-[1.25rem] font-semibold text-ink-1 tracking-tight leading-tight">
                {label}
              </h2>
              <HelpTooltip text={description} label={`Help for ${label}`} />
            </div>
            {description && (
              // Section descriptions in pydantic-derived schemas can be
              // very long (whole architecture notes). Clamp to 2 lines
              // so the header stays compact; full text is still
              // available via the help-tooltip "?" beside the title.
              <p className="mt-1 text-xs text-ink-3 max-w-2xl line-clamp-2">
                {description}
              </p>
            )}
          </header>
          <ObjectFields
            root={root}
            node={node}
            value={fallback}
            onChange={onChange}
            depth={depth + 1}
            pathPrefix={path}
            hashPrefix={hashPrefix}
            rootValue={props.rootValue}
            onRootChange={props.onRootChange}
          />
        </div>
      )
    }
    // Per-block recommendation chips were removed — starter values
    // now ship as top-level config presets (community/presets/*/preset.yaml,
    // loaded via the PresetPickerModal in ConfigTab). Keeps the form
    // chromeless inside groups and moves "I want a sensible starting
    // point" up one level where it belongs.
    return wrapAnchor(
      <CollapsibleCard
        label={label}
        description={description}
        required={required}
        headerExtra={null}
        bodyExtra={null}
      >
        <ObjectFields
          root={root}
          node={node}
          value={fallback}
          onChange={onChange}
          depth={depth + 1}
          pathPrefix={path}
          hashPrefix={hashPrefix}
          rootValue={props.rootValue}
          onRootChange={props.onRootChange}
        />
      </CollapsibleCard>,
    )
  }

  if (kind === 'array') {
    return (
      <ArrayField
        root={root}
        node={node}
        value={fallback}
        onChange={onChange}
        label={label}
        description={description}
        required={required}
        path={path}
        hashPrefix={hashPrefix}
      />
    )
  }

  if (kind === 'unknown') {
    return wrapAnchor(
      <LabelledRow label={label} description={description} required={required} path={path} value={fallback}>
        <AdvancedJsonPreview value={fallback} />
      </LabelledRow>,
    )
  }

  return null
}

/**
 * Renders an object's properties, sorting required fields before optional
 * ones. At depth >= 1 all optional fields are folded behind a
 * "Show <N> advanced" toggle (off by default).
 */
export function ObjectFields({
  root,
  node,
  value,
  onChange,
  depth,
  pathPrefix = '',
  hashPrefix = '',
  rootValue,
  onRootChange,
  forceExpandOptional = false,
}: {
  root: PipelineJsonSchema
  node: JsonSchemaNode
  value: unknown
  onChange: Setter
  depth: number
  pathPrefix?: string
  hashPrefix?: string
  rootValue?: Record<string, unknown>
  onRootChange?: (next: Record<string, unknown>) => void
  /** When true, render every field inline without the "Show N
   *  optional" collapse. Used by schema-driven forms rendered outside
   *  the main config builder (e.g. the plugin Configure modal) where
   *  the schema is typically small and the collapse would just hide
   *  the content the user just opened the modal to edit. */
  forceExpandOptional?: boolean
}) {
  const [showAdvanced, setShowAdvanced] = useState(false)
  const props = (node.properties ?? {}) as Record<string, JsonSchemaNode>
  const currentValue = isPlainRecord(value) ? (value as Record<string, unknown>) : {}
  const override = getRequiredOverride(pathPrefix, currentValue)

  // ``inlineSingleChild``: when this object has exactly one property and
  // that property is itself an object, promote the grandchildren so the
  // user doesn't see an extra ``CollapsibleCard`` wrapper (e.g. the
  // ``connect`` block of a provider where the only field is ``ssh``).
  const propKeys = Object.keys(props)
  if (override?.inlineSingleChild && propKeys.length === 1) {
    const onlyKey = propKeys[0]
    const onlyNode = resolveRef(root, props[onlyKey])
    const childIsObject =
      detectKind(onlyNode) === 'object' ||
      (Array.isArray(onlyNode.anyOf) && onlyNode.anyOf.some((b) => detectKind(resolveRef(root, b)) === 'object'))
    if (childIsObject) {
      const childValue = isPlainRecord(currentValue[onlyKey])
        ? (currentValue[onlyKey] as Record<string, unknown>)
        : {}
      return (
        <ObjectFields
          root={root}
          node={onlyNode}
          value={childValue}
          onChange={(next) => onChange({ ...currentValue, [onlyKey]: next })}
          depth={depth}
          pathPrefix={pathPrefix ? `${pathPrefix}.${onlyKey}` : onlyKey}
          hashPrefix={hashPrefix}
          rootValue={rootValue}
          onRootChange={onRootChange}
        />
      )
    }
  }

  const requiredSet = new Set<string>(Array.isArray(node.required) ? node.required : [])
  override?.requires?.forEach((k) => requiredSet.add(k))
  override?.optional?.forEach((k) => requiredSet.delete(k))
  const alwaysVisibleSet = new Set<string>(override?.alwaysVisible ?? [])
  const fieldOrder = override?.fieldOrder

  // Discriminator detection: look for an enum-typed property whose values
  // exactly match sibling property names. E.g. training.type ∈
  // {qlora,lora,adalora} alongside sibling object fields of the same
  // names. If detected, hide the non-matching siblings in the UI (keep
  // them in the value so switching back is lossless).
  const discriminator = detectDiscriminator(root, props, pathPrefix)
  const activeBranch: string | undefined = (() => {
    if (!discriminator) return undefined
    const fromValue = currentValue[discriminator.enumKey]
    if (typeof fromValue === 'string' && discriminator.siblings.has(fromValue)) {
      return fromValue
    }
    // Fall back to the schema default so the UI doesn't hide *every*
    // sibling while the value is still empty.
    const resolved = resolveRef(root, props[discriminator.enumKey])
    const fromDefault = typeof resolved.default === 'string' ? resolved.default : undefined
    if (fromDefault && discriminator.siblings.has(fromDefault)) return fromDefault
    return discriminator.enumValues[0]
  })()
  const hiddenSiblings = new Set<string>()
  if (discriminator) {
    for (const name of discriminator.siblings) {
      if (name !== activeBranch) hiddenSiblings.add(name)
    }
  }

  // When a discriminator is present, the active branch is conceptually
  // required: without it the form is incomplete. Pin it into requiredSet
  // so it shows up with an asterisk next to the enum key.
  if (discriminator && activeBranch && discriminator.siblings.has(activeBranch)) {
    requiredSet.add(activeBranch)
  }

  const orderedKeys = fieldOrder
    ? orderByHint(Object.keys(props), fieldOrder)
    : Object.keys(props)
  // Pinned bucket = required ∪ alwaysVisible. fieldOrder controls the
  // exact interleaving so an alwaysVisible field (e.g. hyperparams) can
  // sit above a required one (e.g. strategies) when the override says
  // so. Asterisks still come purely from requiredSet membership.
  const pinnedFields: string[] = []
  const optionalFields: string[] = []
  const hiddenFromOverride = new Set<string>(override?.hidden ?? [])
  for (const key of orderedKeys) {
    if (hiddenSiblings.has(key)) continue
    if (hiddenFromOverride.has(key)) continue
    if (requiredSet.has(key) || alwaysVisibleSet.has(key)) pinnedFields.push(key)
    else optionalFields.push(key)
  }
  // Place the active discriminator branch right after ``provider`` when
  // it's present (training's preferred layout), else right after the
  // enum key. fieldOrder can't express this because it doesn't know
  // which branch is active at config time.
  if (discriminator && activeBranch && discriminator.siblings.has(activeBranch)) {
    const branchIdx = pinnedFields.indexOf(activeBranch)
    const anchorKey = pinnedFields.includes('provider')
      ? 'provider'
      : discriminator.enumKey
    const anchorIdx = pinnedFields.indexOf(anchorKey)
    if (branchIdx >= 0 && anchorIdx >= 0 && branchIdx !== anchorIdx + 1) {
      pinnedFields.splice(branchIdx, 1)
      const reInsertAt = pinnedFields.indexOf(anchorKey) + 1
      pinnedFields.splice(reInsertAt, 0, activeBranch)
    }
  }

  function setKey(key: string, next: unknown) {
    const copy = { ...currentValue }
    if (next === undefined) delete copy[key]
    else copy[key] = next
    onChange(copy)
  }

  const renderField = (key: string) => {
    // When this is the soft-discriminator key, override the default
    // string renderer with a select built from the sibling names so the
    // user can't type typos.
    if (discriminator && key === discriminator.enumKey) {
      const resolved = resolveRef(root, props[key])
      if (detectKind(resolved) !== 'enum') {
        const label = titleOrKey(resolved, key)
        const desc =
          typeof resolved.description === 'string' ? resolved.description : undefined
        const fallback =
          typeof currentValue[key] === 'string'
            ? (currentValue[key] as string)
            : typeof resolved.default === 'string'
            ? (resolved.default as string)
            : ''
        const needsEmpty = !discriminator.enumValues.includes(fallback)
        const fieldPath = pathPrefix ? `${pathPrefix}.${key}` : key
        return (
          <LabelledRow
            key={key}
            label={label}
            description={desc}
            required={requiredSet.has(key)}
            path={fieldPath}
            value={fallback}
          >
            <SelectField
              value={fallback}
              options={discriminator.enumValues.map((v) => ({ value: v }))}
              onChange={(next) => setKey(key, next || undefined)}
              allowEmpty={needsEmpty}
            />
          </LabelledRow>
        )
      }
    }
    return (
      <FieldRenderer
        key={key}
        root={root}
        node={props[key]}
        value={currentValue[key]}
        onChange={(next) => setKey(key, next)}
        labelKey={key}
        required={requiredSet.has(key)}
        depth={depth}
        path={pathPrefix ? `${pathPrefix}.${key}` : key}
        hashPrefix={hashPrefix}
        rootValue={rootValue}
        onRootChange={onRootChange}
      />
    )
  }

  // Vertical rhythm rule:
  //   - flat schema (only scalar rows, no nested cards) → tight 4px
  //   - schema mixing cards + rows → 8px so cards read as distinct
  //     sections
  // Keyed off *visible* child kinds so a flat tab (e.g. Model: name,
  // torch_dtype, trust_remote_code) stays dense, while Training keeps
  // breathing room between its lora/hyperparams/strategies cards.
  // Detect card-shaped children via the same nullable-unwrap path the
  // renderer uses — otherwise an ``Optional[str]`` (anyOf [string,null])
  // looks like a "union" card and inflates the gap for flat tabs like
  // Model where every field is a scalar.
  const hasCardChild = [...pinnedFields, ...optionalFields].some((key) => {
    const n = unwrapNullableScalar(root, resolveRef(root, props[key]))
    const k = detectKind(n)
    return k === 'object' || k === 'array' || k === 'union'
  })
  const gap = hasCardChild ? 'space-y-1.5' : 'space-y-0.5'

  if (override?.expandOptional || forceExpandOptional) {
    return (
      <div className={gap}>
        {pinnedFields.map(renderField)}
        {optionalFields.map(renderField)}
      </div>
    )
  }

  return (
    <div className={gap}>
      {pinnedFields.map(renderField)}
      {optionalFields.length > 0 && (
        <div className={gap}>
          <button
            type="button"
            onClick={() => setShowAdvanced((v) => !v)}
            className="text-2xs text-ink-3 hover:text-ink-1 transition flex items-center gap-1.5 py-2"
          >
            <span className={`transition-transform ${showAdvanced ? 'rotate-90' : ''}`}>▸</span>
            {showAdvanced ? 'Hide' : 'Show'} {optionalFields.length} optional field
            {optionalFields.length === 1 ? '' : 's'}
          </button>
          {showAdvanced && (
            <div className={gap}>
              {optionalFields.map(renderField)}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function orderByHint(keys: string[], hint: string[]): string[] {
  const index = new Map(hint.map((k, i) => [k, i]))
  return [...keys].sort((a, b) => {
    const ai = index.get(a) ?? hint.length + keys.indexOf(a)
    const bi = index.get(b) ?? hint.length + keys.indexOf(b)
    return ai - bi
  })
}

function unwrapNullableScalar(root: PipelineJsonSchema, node: JsonSchemaNode): JsonSchemaNode {
  if (!Array.isArray(node.anyOf) || node.anyOf.length < 2) return node
  const resolved = node.anyOf.map((b) => resolveRef(root, b))
  const nonNull = resolved.filter((b) => (Array.isArray(b.type) ? b.type[0] : b.type) !== 'null')
  const hasNull = resolved.some((b) => (Array.isArray(b.type) ? b.type[0] : b.type) === 'null')
  if (!hasNull || nonNull.length !== 1) return node
  // Preserve the outer ``title``/``description``/``default`` so labels
  // and help text don't disappear when unwrapping.
  return { ...nonNull[0], title: node.title ?? nonNull[0].title, description: node.description ?? nonNull[0].description, default: node.default ?? nonNull[0].default, anyOf: undefined }
}

/**
 * Section card with a click-to-collapse header. Defaults to expanded
 * so first-time users see all fields; once collapsed, only the header
 * row (label + ? + chevron) stays visible and the values are still
 * preserved in the form state.
 */
function CollapsibleCard({
  label,
  description,
  required,
  defaultOpen = false,
  children,
  headerExtra,
  bodyExtra,
}: {
  label: string
  description?: string
  required?: boolean
  defaultOpen?: boolean
  children: React.ReactNode
  /** Rendered inside the header row next to the label — e.g. status
   *  chips or counts. Not toggle-sensitive. */
  headerExtra?: React.ReactNode
  /** Rendered at the top of the expanded body — e.g. LoRA one-click
   *  recommendations above the actual field list. */
  bodyExtra?: React.ReactNode
}) {
  const [open, setOpen] = useState(defaultOpen)
  // Flat group: hairline border + surface-2 (one tier up from canvas).
  // No left brand-bar, no gradient wash — those previously made nested
  // groups feel like a separate design language inside the form. Open
  // state shifts only the chevron rotation + a hairline divider between
  // header and body. Brand-policy intact: violet stays for CTA / focus
  // / active-tab, structural chrome is neutral.
  return (
    <div className="rounded-md border border-line-1 bg-surface-2 transition-colors">
      <div
        role="button"
        tabIndex={-1}
        onClick={(e) => {
          if ((e.target as HTMLElement).closest('[data-no-toggle]')) return
          setOpen((v) => !v)
        }}
        className={[
          'flex items-center gap-2 px-3.5 py-2.5 cursor-pointer transition-colors hover:bg-surface-3/30',
          open ? 'border-b border-line-1' : '',
        ].join(' ')}
      >
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation()
            setOpen((v) => !v)
          }}
          aria-expanded={open}
          className="flex items-center gap-2 text-left"
        >
          <span
            aria-hidden
            className={`text-[10px] text-ink-3 transition-transform ${
              open ? 'rotate-90' : ''
            }`}
          >
            ▸
          </span>
          <span className="text-xs text-ink-1 font-medium">
            {label}
            {required && (
              <span className="ml-1 text-brand-warm" aria-hidden>*</span>
            )}
          </span>
        </button>
        <span data-no-toggle>
          <HelpTooltip text={description} />
        </span>
        {headerExtra && <span data-no-toggle className="ml-auto">{headerExtra}</span>}
      </div>
      {open && (
        <div className="px-3.5 pb-3 pt-2 space-y-2">
          {bodyExtra}
          {children}
        </div>
      )}
    </div>
  )
}


function AdvancedJsonPreview({ value }: { value: unknown }) {
  return (
    <div className="rounded-md border border-line-1 bg-surface-0 p-2">
      <div className="text-[0.6rem] text-ink-4 uppercase tracking-wide">advanced — edit via YAML</div>
      <pre className="text-[0.65rem] text-ink-3 font-mono whitespace-pre-wrap break-words max-h-32 overflow-y-auto">
        {value === undefined ? '—' : JSON.stringify(value, null, 2)}
      </pre>
    </div>
  )
}

function collectEnumOptions(node: JsonSchemaNode): unknown[] {
  if (Array.isArray(node.enum)) return node.enum
  if (Array.isArray(node.anyOf)) {
    return node.anyOf
      .map((n) => ('const' in n ? (n as { const: unknown }).const : undefined))
      .filter((v) => v !== undefined)
  }
  return []
}

function isPlainRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

/**
 * Detect a discriminator pattern in an object schema. Two shapes are
 * recognised:
 *
 * 1. Strict: one property is an ``enum`` whose string values exactly
 *    match the names of ≥2 sibling properties. E.g.
 *    ``{ type: Literal["qlora","lora","adalora"], qlora, lora, adalora }``.
 *
 * 2. Soft: one property named ``type``/``kind``/``mode`` is a plain
 *    string AND ≥2 sibling properties are object-like (optionally
 *    nullable). The discriminator values default to the sibling names.
 *    Covers Pydantic models that haven't been tightened to ``Literal``
 *    (e.g. current ``TrainingOnlyConfig.type``).
 *
 * Returns the enum key + the set of sibling names + the values shown in
 * the dropdown, or ``null`` when no such pattern is present.
 */
function detectDiscriminator(
  root: PipelineJsonSchema,
  props: Record<string, JsonSchemaNode>,
  pathPrefix: string,
): { enumKey: string; siblings: Set<string>; enumValues: string[] } | null {
  const keys = Object.keys(props)

  // Path override: when the soft-discriminator heuristic would otherwise
  // over-match (e.g. training.type is plain ``str`` and every object
  // sibling looks like a branch), we pin the enumKey/values explicitly.
  const forced = getDiscriminatorOverride(pathPrefix)
  if (forced && keys.includes(forced.enumKey)) {
    const matching = forced.values.filter((v) => keys.includes(v) && v !== forced.enumKey)
    if (matching.length >= 2) {
      return {
        enumKey: forced.enumKey,
        siblings: new Set(matching),
        enumValues: forced.values,
      }
    }
  }
  const objectLikeSiblings = (self: string): string[] =>
    keys.filter((k) => {
      if (k === self) return false
      const resolved = resolveRef(root, props[k])
      const kind = detectKind(resolved)
      if (kind === 'object') return true
      // Nullable object: anyOf of object + null.
      if (Array.isArray(resolved.anyOf)) {
        const branches = resolved.anyOf.map((b) => resolveRef(root, b))
        if (
          branches.some((b) => detectKind(b) === 'object') &&
          branches.some((b) => (Array.isArray(b.type) ? b.type[0] : b.type) === 'null')
        ) {
          return true
        }
      }
      return false
    })

  // Strict: enum values ⊆ sibling names.
  for (const key of keys) {
    const resolved = resolveRef(root, props[key])
    if (detectKind(resolved) !== 'enum') continue

    const values: string[] = []
    if (Array.isArray(resolved.enum)) {
      for (const v of resolved.enum) if (typeof v === 'string') values.push(v)
    } else if (Array.isArray(resolved.anyOf)) {
      for (const branch of resolved.anyOf) {
        if (branch && typeof branch === 'object' && 'const' in branch) {
          const c = (branch as { const?: unknown }).const
          if (typeof c === 'string') values.push(c)
        }
      }
    }
    if (values.length < 2) continue

    const matching = values.filter((v) => keys.includes(v) && v !== key)
    if (matching.length >= 2 && matching.length === values.length) {
      return { enumKey: key, siblings: new Set(matching), enumValues: values }
    }
  }

  // Soft: a "type"/"kind"/"mode" string field + object-like siblings.
  const candidateKeys = ['type', 'kind', 'mode']
  for (const key of candidateKeys) {
    if (!keys.includes(key)) continue
    const resolved = resolveRef(root, props[key])
    if (detectKind(resolved) !== 'string') continue
    const siblings = objectLikeSiblings(key)
    if (siblings.length < 2) continue
    return {
      enumKey: key,
      siblings: new Set(siblings),
      enumValues: siblings,
    }
  }

  return null
}
