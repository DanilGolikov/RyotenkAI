import type { JsonSchemaNode, PipelineJsonSchema } from '../../api/hooks/useConfigSchema'
import { FormGroup } from './FormGroup'
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
      <div className="flex items-baseline justify-between gap-2">
        <label className="text-2xs text-ink-2">
          {label}
          {required && <span className="ml-1 text-brand">*</span>}
        </label>
      </div>
      <div className="mt-1">{children}</div>
      {description && (
        <div className="text-[0.65rem] text-ink-4 mt-0.5 leading-snug">{description}</div>
      )}
    </div>
  )
}

export function FieldRenderer(props: FieldProps) {
  const { root, node: rawNode, value, onChange, labelKey, required, depth = 0 } = props
  const node = resolveRef(root, rawNode)
  const kind = detectKind(node)
  const label = titleOrKey(node, labelKey)
  const description = typeof node.description === 'string' ? node.description : undefined
  const fallback = value === undefined ? getDefault(node) : value

  if (kind === 'boolean') {
    return (
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
    return (
      <LabelledRow label={label} description={description} required={required}>
        <select
          value={fallback === undefined || fallback === null ? '' : String(fallback)}
          onChange={(e) => onChange(e.target.value === '' ? undefined : e.target.value)}
          className="w-full rounded-md bg-surface-2 border border-line-1 px-2 py-1.5 text-xs font-mono focus:outline-none focus:border-brand"
        >
          <option value="">—</option>
          {options.map((opt) => (
            <option key={String(opt)} value={String(opt)}>
              {String(opt)}
            </option>
          ))}
        </select>
      </LabelledRow>
    )
  }

  if (kind === 'number' || kind === 'integer') {
    return (
      <LabelledRow label={label} description={description} required={required}>
        <input
          type="number"
          step={kind === 'integer' ? 1 : 'any'}
          value={fallback === undefined || fallback === null ? '' : String(fallback)}
          onChange={(e) => {
            if (e.target.value === '') onChange(undefined)
            else onChange(kind === 'integer' ? Number.parseInt(e.target.value, 10) : Number(e.target.value))
          }}
          className="w-full rounded-md bg-surface-2 border border-line-1 px-2 py-1.5 text-xs font-mono focus:outline-none focus:border-brand"
        />
      </LabelledRow>
    )
  }

  if (kind === 'string') {
    return (
      <LabelledRow label={label} description={description} required={required}>
        <input
          type="text"
          value={typeof fallback === 'string' ? fallback : ''}
          onChange={(e) => onChange(e.target.value)}
          className="w-full rounded-md bg-surface-2 border border-line-1 px-2 py-1.5 text-xs font-mono focus:outline-none focus:border-brand"
        />
      </LabelledRow>
    )
  }

  if (kind === 'object') {
    const props = (node.properties ?? {}) as Record<string, JsonSchemaNode>
    const requiredSet = new Set<string>(Array.isArray(node.required) ? node.required : [])
    const currentValue = (isPlainRecord(fallback) ? fallback : {}) as Record<string, unknown>
    const fields = Object.keys(props)
    if (fields.length === 0) {
      return (
        <LabelledRow label={label} description={description} required={required}>
          <AdvancedJsonPreview value={fallback} />
        </LabelledRow>
      )
    }
    if (depth === 0) {
      return (
        <FormGroup title={label} description={description} required={required} defaultOpen={required || requiredSet.size > 0}>
          {fields.map((key) => (
            <FieldRenderer
              key={key}
              root={root}
              node={props[key]}
              value={currentValue[key]}
              onChange={(next) => {
                const copy = { ...currentValue }
                if (next === undefined) delete copy[key]
                else copy[key] = next
                onChange(copy)
              }}
              labelKey={key}
              required={requiredSet.has(key)}
              depth={depth + 1}
            />
          ))}
        </FormGroup>
      )
    }
    return (
      <div className="space-y-2 pl-3 border-l border-line-1">
        <div className="text-2xs text-ink-2 font-medium">{label}</div>
        {description && <div className="text-[0.65rem] text-ink-4">{description}</div>}
        {fields.map((key) => (
          <FieldRenderer
            key={key}
            root={root}
            node={props[key]}
            value={currentValue[key]}
            onChange={(next) => {
              const copy = { ...currentValue }
              if (next === undefined) delete copy[key]
              else copy[key] = next
              onChange(copy)
            }}
            labelKey={key}
            required={requiredSet.has(key)}
            depth={depth + 1}
          />
        ))}
      </div>
    )
  }

  if (kind === 'array' || kind === 'union' || kind === 'unknown') {
    return (
      <LabelledRow label={label} description={description} required={required}>
        <AdvancedJsonPreview value={fallback} />
      </LabelledRow>
    )
  }

  return null
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
