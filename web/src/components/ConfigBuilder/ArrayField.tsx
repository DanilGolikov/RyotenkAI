import { useMemo } from 'react'
import type { JsonSchemaNode, PipelineJsonSchema } from '../../api/hooks/useConfigSchema'
import { FieldAnchor } from './FieldAnchor'
import { FieldRenderer } from './FieldRenderer'
import { HelpTooltip } from './HelpTooltip'
import { detectKind, getDefault, resolveRef } from './schemaUtils'

interface Props {
  root: PipelineJsonSchema
  node: JsonSchemaNode
  value: unknown
  onChange: (value: unknown) => void
  label: string
  description?: string
  required?: boolean
  path: string
  hashPrefix?: string
}

function defaultItem(node: JsonSchemaNode): unknown {
  if (Object.prototype.hasOwnProperty.call(node, 'default')) {
    return getDefault(node)
  }
  const kind = detectKind(node)
  switch (kind) {
    case 'string':
      return ''
    case 'integer':
    case 'number':
      return 0
    case 'boolean':
      return false
    case 'array':
      return []
    case 'object':
      return {}
    case 'enum': {
      const opts = Array.isArray(node.enum)
        ? node.enum
        : Array.isArray(node.anyOf)
        ? node.anyOf
            .map((n) => ('const' in n ? (n as { const: unknown }).const : undefined))
            .filter((v) => v !== undefined)
        : []
      return opts[0] ?? ''
    }
    default:
      return null
  }
}

/**
 * Inline array editor. Replaces the "advanced — edit via YAML" fallback
 * for ``kind === 'array'``. Each row renders its element through
 * FieldRenderer using the resolved ``items`` schema; items can be added
 * or removed, but reordering is out of MVP scope.
 */
export function ArrayField({
  root,
  node,
  value,
  onChange,
  label,
  description,
  required,
  path,
  hashPrefix,
}: Props) {
  const itemsSchema = useMemo<JsonSchemaNode>(() => {
    const raw = (node.items as JsonSchemaNode | undefined) ?? {}
    return resolveRef(root, raw)
  }, [root, node.items])

  const list: unknown[] = Array.isArray(value) ? (value as unknown[]) : []

  function updateAt(idx: number, next: unknown) {
    const copy = list.slice()
    if (next === undefined) copy.splice(idx, 1)
    else copy[idx] = next
    onChange(copy)
  }

  function removeAt(idx: number) {
    const copy = list.slice()
    copy.splice(idx, 1)
    onChange(copy)
  }

  function appendItem() {
    const next = list.concat([defaultItem(itemsSchema)])
    onChange(next)
  }

  return (
    <FieldAnchor path={path} hashPrefix={hashPrefix ?? ''}>
      <div className="space-y-3">
        <div className="flex items-center gap-2">
          <div className="text-xs text-ink-1 font-medium">
            {label}
            {required && <span className="ml-1 text-brand">*</span>}
          </div>
          <HelpTooltip text={description} />
          <span className="ml-auto text-2xs text-ink-3">
            {list.length} item{list.length === 1 ? '' : 's'}
          </span>
          <button
            type="button"
            onClick={appendItem}
            className="rounded-md border border-line-1 px-2 py-1 text-2xs text-ink-2 hover:text-ink-1 hover:border-brand-alt transition"
          >
            + Add
          </button>
        </div>
        {list.length === 0 ? (
          <div className="rounded-md border border-dashed border-line-1 bg-surface-0/40 px-3 py-4 text-2xs text-ink-3 text-center">
            No items. Click <span className="text-ink-1">+ Add</span> to create one.
          </div>
        ) : (
          <div className="space-y-3">
            {list.map((item, idx) => (
              <div
                key={idx}
                className="relative rounded-md border border-line-1 bg-surface-1/60 p-3 pr-10"
              >
                <div className="text-[0.6rem] uppercase tracking-wide text-ink-4 mb-2">
                  item {idx + 1}
                </div>
                <FieldRenderer
                  root={root}
                  node={itemsSchema}
                  value={item}
                  onChange={(next) => updateAt(idx, next)}
                  labelKey={`${path}[${idx}]`}
                  depth={1}
                  path={`${path}.${idx}`}
                  hashPrefix={hashPrefix}
                />
                <button
                  type="button"
                  onClick={() => removeAt(idx)}
                  title="Remove item"
                  className="absolute top-2 right-2 w-6 h-6 rounded text-ink-3 hover:text-err hover:bg-err/10 transition text-xs"
                >
                  ✕
                </button>
              </div>
            ))}
          </div>
        )}
      </div>
    </FieldAnchor>
  )
}
