import { useState } from 'react'
import type { JsonSchemaNode, PipelineJsonSchema } from '../../api/hooks/useConfigSchema'
import { ArrayField } from './ArrayField'
import { FieldAnchor } from './FieldAnchor'
import { HelpTooltip } from './HelpTooltip'
import { AlertIcon } from '../icons'
import { HFModelField } from './HFModelField'
import { InferenceProviderField } from './InferenceProviderField'
import { SelectField } from './SelectField'
import { TrainingProviderField } from './TrainingProviderField'
import { UnionField } from './UnionField'
import { useClientFieldValidation, useFieldStatus, useValidationCtx } from './ValidationContext'
import type { FieldStatus } from './ValidationContext'

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
}
const CUSTOM_FIELD_RENDERERS: Record<
  string,
  React.ComponentType<CustomFieldProps>
> = {
  'training.provider': TrainingProviderField,
  'inference.provider': InferenceProviderField,
  'model.name': HFModelField,
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
}

/**
 * Dense label-left row with a hairline divider + subtle row hover.
 * Labels are intentionally muted (zinc-400) so input values pop — the
 * brightness delta is what lets the eye scan boundaries in a long form.
 * Hairlines come from `space-y-2` on the surrounding
 * ObjectFields container.
 */
// Visual rules (after iteration with design):
//   - idle / ok / editing → no bar. Focus ring on the input already
//     signals "you're here" via the existing `focus:border-brand` inside
//     INPUT_BASE; we mirror that onto the label pill so both halves of
//     the row light up together.
//   - error → red bar under the input + matching red border on the label
//     pill + inline message below.
const STATUS_BORDER: Record<FieldStatus['state'], string> = {
  idle: '',
  editing: '',
  ok: '',
  // Apply ring to the DIRECT CHILD (input / SelectField trigger / etc.),
  // not the wrapper — the wrapper fills the whole grid cell, so a ring on
  // it spans the full pane while the actual input may be narrow. Arbitrary
  // variant [&>*] lets us reach only the one rendered input element.
  // Thicker 2px ring + subtle red bg tint + red border: errors should
  // pop visually against the flat grey surface — previously a 1px ring
  // was too quiet to notice while scrolling a long form.
  error:
    '[&>*]:rounded [&>*]:ring-2 [&>*]:ring-err/40 [&>*]:border-err [&>*]:bg-err/[0.04]',
}

const LABEL_PILL_BORDER: Record<FieldStatus['state'], string> = {
  idle: 'border-line-1',
  editing: 'border-brand',
  ok: 'border-line-1',
  // Matching the stronger ring on the input side: thin red ring + red
  // border on the label pill so both halves of the row read as "errored".
  error: 'border-err ring-1 ring-err/30',
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
  /** Skip the input-side status bar. Used for checkboxes where a bar
   *  under a 16×16 box reads as a glitch. */
  suppressBar?: boolean
  children: React.ReactNode
}) {
  const status = useFieldStatus(path ?? '', Boolean(required), value)
  const ctx = useValidationCtx()
  const bar = path && !suppressBar ? STATUS_BORDER[status.state] : STATUS_BORDER.idle
  const labelBorder = path ? LABEL_PILL_BORDER[status.state] : LABEL_PILL_BORDER.idle
  // Pulse class: soft yellow attention halo, applied to both halves
  // while the field has an unresolved error AND isn't currently
  // focused. The moment the user clicks into the field, focusedPath
  // matches and the class falls off — animation stops cleanly. Kept
  // OFF the input-side wrapper when `suppressBar` is set (e.g.
  // checkboxes) to avoid the halo cupping a 16px box.
  const isFocused = path != null && ctx?.focusedPath === path
  const pulseCls =
    path && status.state === 'error' && !isFocused
      ? 'field-attention-pulse'
      : ''
  // Derive a stable DOM id from the dotted config path so the <label
  // htmlFor> binds to the input rendered by children, and screen
  // readers can announce "Field X, required, value Y" consistently.
  // Path is guaranteed unique within one form instance. CSS.escape
  // handles numeric array indices and any chars that aren't valid in
  // id selectors.
  const inputId = path ? `cfg-${path.replace(/[^a-zA-Z0-9_-]/g, '_')}` : undefined
  const helpLabel = path ? `Help for ${label}` : 'Field help'
  // `group/row` + `group-focus-within/row:*` lets the label column light
  // up brand-burgundy whenever any input inside the row is focused — a
  // single visual signal "you're editing this field" that survives
  // focus shifts between sub-inputs (e.g. combobox popovers). Brand tint
  // stays ≤10% alpha per the "brand doesn't flood forms" policy.
  return (
    <div className="group/row py-1.5">
      <div className="grid grid-cols-1 sm:grid-cols-[220px_minmax(0,1fr)] gap-2 sm:gap-4 items-start sm:items-center">
        <div
          className={`flex items-center gap-2 min-w-0 rounded bg-surface-1 border ${labelBorder} ${pulseCls} px-2.5 h-8 transition-colors group-focus-within/row:bg-brand-weak/10 group-focus-within/row:border-brand/30`}
        >
          {/* Required marker lives in a fixed-width slot on the left so
              every row aligns to the same column, regardless of label
              length. Pattern borrowed from Ant Design; absent rows get
              an invisible placeholder to preserve the grid. */}
          <span
            aria-hidden={!required}
            className={`inline-flex w-2 shrink-0 text-brand-warm text-xs leading-none ${required ? '' : 'invisible'}`}
          >
            *
          </span>
          <label
            htmlFor={inputId}
            className="flex-1 min-w-0 text-xs text-ink-2 font-medium tracking-tight truncate"
          >
            {label}
            {required && <span className="sr-only"> (required)</span>}
          </label>
          {/* Error glyph — sits to the LEFT of the help "?" so the
              user registers "something's wrong here" before they
              look at the input column. Only rendered on error; no
              tooltip interaction — the full message already lives
              inline below the field row. */}
          {status.state === 'error' && (
            <span
              aria-hidden
              title={status.message}
              className="inline-flex w-4 h-4 items-center justify-center text-err shrink-0"
            >
              <AlertIcon className="w-3.5 h-3.5" />
            </span>
          )}
          <HelpTooltip text={description} label={helpLabel} />
        </div>
        <div
          id={inputId}
          className={`w-full min-w-0 ${bar} ${pulseCls} transition-colors`}
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
          className="mt-1 ml-0 sm:ml-[236px] text-[0.65rem] text-err font-mono break-words"
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
// Grafana-flat: neutral surface-1 bg + hairline line-1 border. Focus
// lifts to brand border (violet) without any coloured background, so
// the form reads as chrome rather than decoration.
const INPUT_BASE =
  'h-8 rounded bg-surface-1 border border-line-1 px-2.5 text-[13px] text-ink-1 font-mono focus:outline-none focus:border-brand hover:border-line-2 transition-colors placeholder:text-ink-4 placeholder:italic'

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
        <Custom value={fallback} onChange={onChange} {...focusHandlers} />
      </LabelledRow>,
    )
  }

  const fieldOverride = getFieldOverride(path)
  if (fieldOverride?.comingSoon) {
    // Don't surface this field as "required" — the form widget isn't
    // built yet, so the client can't satisfy it anyway. Server still
    // validates the underlying YAML, so correctness isn't lost; we
    // just stop blocking the UI on a widget we haven't shipped. Show
    // a muted banner underneath so the user knows they must edit the
    // YAML view to set this value in the meantime.
    return wrapAnchor(
      <div className="space-y-1">
        <LabelledRow label={label} description={description} required={false} path={path} value={fallback}>
          <div className="h-8 inline-flex items-center gap-2 px-2.5 rounded-md border border-dashed border-line-1 bg-surface-0/40 text-xs text-ink-3 italic">
            <span>{fieldOverride.comingSoon}</span>
            <span className="not-italic text-[0.55rem] uppercase tracking-wide px-1.5 py-0.5 rounded bg-brand-alt/15 text-brand-alt">
              soon
            </span>
          </div>
        </LabelledRow>
        <div className="ml-0 sm:ml-[236px] text-[0.65rem] text-ink-4">
          Picker in progress — switch to the{' '}
          <span className="text-ink-3 font-medium">YAML</span> view to set this
          value today.
        </div>
      </div>,
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
        <label className="inline-flex items-center gap-2 text-xs cursor-pointer select-none">
          <input
            type="checkbox"
            checked={checked}
            onChange={(e) => onChange(e.target.checked)}
            {...focusHandlers}
            className="h-4 w-4 rounded-[2px] border border-line-2 bg-surface-1 accent-brand hover:border-ink-3 focus:outline-none focus-visible:ring-1 focus-visible:ring-brand transition-colors"
          />
          <span className="font-mono text-ink-2">{String(checked)}</span>
        </label>
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
      // Header design — follows the Linear/Stripe/GitHub pattern for
      // config forms:
      //   • Typography carries the hierarchy (18–20 px, font-weight 600).
      //     Decoration is minimal — no accent bars or gradient flourishes.
      //   • A tiny uppercase "eyebrow" above the title gives context
      //     ("Section") without bolder chrome.
      //   • Hairline divider below the whole block keeps rhythm between
      //     sections without a heavy box.
      return (
        <div className="space-y-5">
          <header className="pb-3 border-b border-line-1/50">
            <div className="text-[0.6rem] font-medium uppercase tracking-[0.16em] text-ink-4 mb-1">
              Section
            </div>
            <div className="flex items-baseline gap-2">
              <h3 className="text-[1.15rem] font-semibold text-ink-1 tracking-tight leading-tight">
                {label}
              </h3>
              <HelpTooltip text={description} label={`Help for ${label}`} />
            </div>
          </header>
          <ObjectFields
            root={root}
            node={node}
            value={fallback}
            onChange={onChange}
            depth={depth + 1}
            pathPrefix={path}
            hashPrefix={hashPrefix}
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
}: {
  root: PipelineJsonSchema
  node: JsonSchemaNode
  value: unknown
  onChange: Setter
  depth: number
  pathPrefix?: string
  hashPrefix?: string
}) {
  const [showAdvanced, setShowAdvanced] = useState(false)
  const props = (node.properties ?? {}) as Record<string, JsonSchemaNode>
  const currentValue = isPlainRecord(value) ? (value as Record<string, unknown>) : {}
  const override = getRequiredOverride(pathPrefix, currentValue)
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

  if (override?.expandOptional) {
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
  return (
    // Nested collapsibles wear a violet (`brand-alt`) left-line as the
    // secondary brand tone — top sections use burgundy, nested groups
    // use violet, so two levels read as two moods (see Brand-usage
    // policy in FRONTEND_GUIDELINES.md). When closed the card stays flat
    // on `surface-2`; when open, the header gets a soft violet-to-right
    // wash and the body drops to `surface-1` so the open group reads as
    // a well, not a pasted-on block.
    <div
      className={[
        'relative rounded border border-line-1 transition-colors',
        open
          ? 'bg-surface-1 border-l-2 border-l-brand-alt/50'
          : 'bg-surface-2 border-l-2 border-l-brand-alt/25 hover:border-l-brand-alt/40',
      ].join(' ')}
    >
      <div
        role="button"
        tabIndex={-1}
        onClick={(e) => {
          if ((e.target as HTMLElement).closest('[data-no-toggle]')) return
          setOpen((v) => !v)
        }}
        className={[
          'flex items-center gap-2 px-4 py-2.5 cursor-pointer transition-colors',
          open
            ? 'bg-gradient-to-r from-brand-alt/[0.12] via-transparent to-transparent'
            : 'hover:bg-surface-3/40',
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
            className={`text-[10px] transition-transform ${
              open ? 'rotate-90 text-brand-alt' : 'text-ink-3'
            }`}
          >
            ▸
          </span>
          <span className="text-xs text-ink-1 font-medium">
            {label}
            {required && <span className="ml-1 text-brand-warm">*</span>}
          </span>
        </button>
        <span data-no-toggle>
          <HelpTooltip text={description} />
        </span>
        {headerExtra && <span data-no-toggle className="ml-auto">{headerExtra}</span>}
      </div>
      {open && (
        <div className="px-4 pb-3 pt-1 space-y-3">
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
