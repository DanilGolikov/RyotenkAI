import { useState } from 'react'
import type { JsonSchemaNode, PipelineJsonSchema } from '../../api/hooks/useConfigSchema'
import { FieldAnchor } from './FieldAnchor'
import { HelpTooltip } from './HelpTooltip'
import { UnionField } from './UnionField'
import { detectKind, getDefault, resolveRef, titleOrKey } from './schemaUtils'

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

function LabelledRow({
  label,
  description,
  required,
  children,
}: {
  label: string
  description?: string
  required?: boolean
  children: React.ReactNode
}) {
  return (
    <div>
      <div className="flex items-center gap-2 mb-1.5">
        <label className="text-xs text-ink-1 font-medium">
          {label}
          {required && <span className="ml-1 text-brand">*</span>}
        </label>
        <HelpTooltip text={description} />
      </div>
      {children}
    </div>
  )
}

export function FieldRenderer(props: FieldProps) {
  const { root, node: rawNode, value, onChange, labelKey, required, depth = 0 } = props
  const path = props.path ?? labelKey
  const hashPrefix = props.hashPrefix ?? ''
  const node = resolveRef(root, rawNode)
  const kind = detectKind(node)
  const label = titleOrKey(node, labelKey)
  const description = typeof node.description === 'string' ? node.description : undefined
  const fallback = value === undefined ? getDefault(node) : value

  const wrapAnchor = (el: React.ReactNode) => (
    <FieldAnchor path={path} hashPrefix={hashPrefix}>
      {el}
    </FieldAnchor>
  )

  if (kind === 'boolean') {
    return wrapAnchor(
      <LabelledRow label={label} description={description} required={required}>
        <label className="inline-flex items-center gap-2 text-xs">
          <input
            type="checkbox"
            checked={Boolean(fallback)}
            onChange={(e) => onChange(e.target.checked)}
            className="accent-brand"
          />
          <span className="text-ink-2">{String(Boolean(fallback))}</span>
        </label>
      </LabelledRow>
    )
  }

  if (kind === 'enum') {
    const options = collectEnumOptions(node)
    return wrapAnchor(
      <LabelledRow label={label} description={description} required={required}>
        <select
          value={fallback === undefined || fallback === null ? '' : String(fallback)}
          onChange={(e) => onChange(e.target.value === '' ? undefined : e.target.value)}
          className="w-full rounded-md bg-surface-2 border border-line-1 px-3 py-2 text-sm font-mono focus:outline-none focus:border-brand"
        >
          <option value="">—</option>
          {options.map((opt) => (
            <option key={String(opt)} value={String(opt)}>
              {String(opt)}
            </option>
          ))}
        </select>
      </LabelledRow>,
    )
  }

  if (kind === 'number' || kind === 'integer') {
    return wrapAnchor(
      <LabelledRow label={label} description={description} required={required}>
        <input
          type="number"
          step={kind === 'integer' ? 1 : 'any'}
          value={fallback === undefined || fallback === null ? '' : String(fallback)}
          onChange={(e) => {
            if (e.target.value === '') onChange(undefined)
            else
              onChange(
                kind === 'integer' ? Number.parseInt(e.target.value, 10) : Number(e.target.value),
              )
          }}
          className="w-full rounded-md bg-surface-2 border border-line-1 px-3 py-2 text-sm font-mono focus:outline-none focus:border-brand"
        />
      </LabelledRow>,
    )
  }

  if (kind === 'string') {
    return wrapAnchor(
      <LabelledRow label={label} description={description} required={required}>
        <input
          type="text"
          value={typeof fallback === 'string' ? fallback : ''}
          onChange={(e) => onChange(e.target.value)}
          className="w-full rounded-md bg-surface-2 border border-line-1 px-3 py-2 text-sm font-mono focus:outline-none focus:border-brand"
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
      <LabelledRow label={label} description={description} required={required}>
        <AdvancedJsonPreview value={fallback} />
      </LabelledRow>,
    )
  }

  if (kind === 'object') {
    const fields = Object.keys(node.properties ?? {})
    if (fields.length === 0) {
      return (
        <LabelledRow label={label} description={description} required={required}>
          <AdvancedJsonPreview value={fallback} />
        </LabelledRow>
      )
    }
    if (depth === 0) {
      // Flat render: the active subtab already owns the frame, no need for
      // a nested collapsible card inside it.
      return (
        <div className="space-y-5">
          <header className="flex items-center gap-2">
            <h3 className="text-base font-semibold text-ink-1">{label}</h3>
            {required && <span className="text-[0.65rem] text-brand uppercase tracking-wide">required</span>}
            <HelpTooltip text={description} />
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
    return wrapAnchor(
      <div className="space-y-3 pl-3 border-l border-line-1">
        <div className="flex items-center gap-2">
          <div className="text-xs text-ink-2 font-medium">{label}</div>
          <HelpTooltip text={description} />
        </div>
        <ObjectFields
          root={root}
          node={node}
          value={fallback}
          onChange={onChange}
          depth={depth + 1}
          pathPrefix={path}
          hashPrefix={hashPrefix}
        />
      </div>,
    )
  }

  if (kind === 'array' || kind === 'unknown') {
    return wrapAnchor(
      <LabelledRow label={label} description={description} required={required}>
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
function ObjectFields({
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
  const requiredSet = new Set<string>(Array.isArray(node.required) ? node.required : [])
  const currentValue = isPlainRecord(value) ? (value as Record<string, unknown>) : {}

  const requiredFields: string[] = []
  const optionalFields: string[] = []
  for (const key of Object.keys(props)) {
    if (requiredSet.has(key)) requiredFields.push(key)
    else optionalFields.push(key)
  }

  function setKey(key: string, next: unknown) {
    const copy = { ...currentValue }
    if (next === undefined) delete copy[key]
    else copy[key] = next
    onChange(copy)
  }

  const renderField = (key: string) => (
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

  return (
    <div className="space-y-4">
      {requiredFields.map(renderField)}
      {optionalFields.length > 0 && (
        <div className="space-y-4">
          <button
            type="button"
            onClick={() => setShowAdvanced((v) => !v)}
            className="text-2xs text-ink-3 hover:text-ink-1 transition flex items-center gap-1.5"
          >
            <span className={`transition-transform ${showAdvanced ? 'rotate-90' : ''}`}>▸</span>
            {showAdvanced ? 'Hide' : 'Show'} {optionalFields.length} optional field
            {optionalFields.length === 1 ? '' : 's'}
          </button>
          {showAdvanced && <div className="space-y-4">{optionalFields.map(renderField)}</div>}
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
