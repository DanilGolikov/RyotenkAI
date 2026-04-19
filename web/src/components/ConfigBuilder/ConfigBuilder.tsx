import type { JsonSchemaNode, PipelineJsonSchema } from '../../api/hooks/useConfigSchema'
import { FieldRenderer } from './FieldRenderer'

export function ConfigBuilder({
  schema,
  value,
  onChange,
}: {
  schema: PipelineJsonSchema
  value: Record<string, unknown>
  onChange: (next: Record<string, unknown>) => void
}) {
  const topProps = (schema.properties ?? {}) as Record<string, JsonSchemaNode>
  const topRequired = new Set<string>(Array.isArray(schema.required) ? schema.required : [])

  const keys = Object.keys(topProps)
  if (keys.length === 0) {
    return <div className="text-xs text-ink-3">Empty schema.</div>
  }

  return (
    <div className="space-y-3">
      {keys.map((key) => (
        <FieldRenderer
          key={key}
          root={schema}
          node={topProps[key]}
          value={value?.[key]}
          labelKey={key}
          required={topRequired.has(key)}
          depth={0}
          onChange={(next) => {
            const copy = { ...value }
            if (next === undefined) delete copy[key]
            else copy[key] = next
            onChange(copy)
          }}
        />
      ))}
    </div>
  )
}
