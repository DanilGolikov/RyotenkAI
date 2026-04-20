import { useMemo, useState } from 'react'
import type { JsonSchemaNode, PipelineJsonSchema } from '../../api/hooks/useConfigSchema'
import { FieldAnchor } from './FieldAnchor'
import { FieldRenderer, ObjectFields } from './FieldRenderer'
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

/**
 * Pick a human-readable title for an array item. Prefers well-known
 * discriminator-ish keys (``strategy_type``, ``type``, ``name``, ``id``)
 * so e.g. a Strategies row shows "sft" instead of "item 1".
 */
const ITEM_TITLE_KEYS = ['strategy_type', 'type', 'name', 'id']
function deriveItemTitle(item: unknown, idx: number): string {
  if (item && typeof item === 'object' && !Array.isArray(item)) {
    const rec = item as Record<string, unknown>
    for (const key of ITEM_TITLE_KEYS) {
      const v = rec[key]
      if (typeof v === 'string' && v.trim()) {
        // Strategy phase items read as "<NAME> Strategy Phase" so the
        // expanded card label matches the user's mental model instead of
        // just showing the raw enum value.
        if (key === 'strategy_type') return `${v.toUpperCase()} Strategy Phase`
        return v
      }
    }
  }
  return `Item ${idx + 1}`
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
  const initialList: unknown[] = Array.isArray(value) ? (value as unknown[]) : []
  // Auto-expand arrays that already carry content; empty arrays start
  // collapsed so a fresh form reads as "section present, no entries yet".
  const [open, setOpen] = useState(initialList.length > 0)
  const itemsSchema = useMemo<JsonSchemaNode>(() => {
    const raw = (node.items as JsonSchemaNode | undefined) ?? {}
    return resolveRef(root, raw)
  }, [root, node.items])

  const list: unknown[] = initialList

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
      <div className="relative rounded border border-line-1 bg-surface-2">
        {open && (
          <div
            aria-hidden
            className="absolute left-0 inset-y-1 w-0.5 bg-gradient-brand rounded-full pointer-events-none"
          />
        )}
        <div
          role="button"
          tabIndex={-1}
          onClick={(e) => {
            if ((e.target as HTMLElement).closest('[data-no-toggle]')) return
            setOpen((v) => !v)
          }}
          className="flex items-center gap-2 px-4 py-2.5 cursor-pointer hover:bg-surface-3/40 transition-colors"
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
              className={`text-ink-3 text-[10px] transition-transform ${open ? 'rotate-90' : ''}`}
            >
              ▸
            </span>
            <span className="text-xs text-ink-1 font-medium">
              {label}
              {required && <span className="ml-1 text-brand">*</span>}
            </span>
          </button>
          <span data-no-toggle>
            <HelpTooltip text={description} />
          </span>
          <span className="ml-auto text-2xs text-ink-3" data-no-toggle>
            {list.length} item{list.length === 1 ? '' : 's'}
          </span>
          <button
            type="button"
            data-no-toggle
            onClick={(e) => {
              e.stopPropagation()
              appendItem()
              setOpen(true)
            }}
            className="rounded border border-line-1 px-2 py-1 text-2xs text-ink-2 hover:text-ink-1 hover:border-line-2 transition"
          >
            + Add
          </button>
        </div>
        {open && (
          <div className="px-4 pb-3 pt-1 space-y-3">
            {list.length === 0 ? (
              <div className="rounded border border-dashed border-line-1 bg-surface-0/40 px-3 py-4 text-2xs text-ink-3 text-center">
                No items. Click <span className="text-ink-1">+ Add</span> to create one.
              </div>
            ) : (
              list.map((item, idx) => (
                <ArrayItem
                  key={idx}
                  title={deriveItemTitle(item, idx)}
                  onRemove={() => removeAt(idx)}
                >
                  {detectKind(itemsSchema) === 'object' ? (
                    // Skip the extra CollapsibleCard FieldRenderer would
                    // wrap the item's object schema in — ArrayItem already
                    // provides the title + toggle, so flatten the fields
                    // directly under it.
                    <ObjectFields
                      root={root}
                      node={itemsSchema}
                      value={item}
                      onChange={(next) => updateAt(idx, next)}
                      depth={1}
                      pathPrefix={`${path}.${idx}`}
                      hashPrefix={hashPrefix}
                    />
                  ) : (
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
                  )}
                </ArrayItem>
              ))
            )}
          </div>
        )}
      </div>
    </FieldAnchor>
  )
}

/**
 * Single collapsible array item. Header = title (e.g. "sft") + delete;
 * body = the FieldRenderer output. Starts open; per-item state lives
 * here so collapsing one item doesn't affect its siblings.
 */
function ArrayItem({
  title,
  onRemove,
  children,
  defaultOpen = true,
}: {
  title: string
  onRemove: () => void
  children: React.ReactNode
  /** Items open by default so the strategy phase's fields are visible
   *  without an extra click — callers pass false only for really long
   *  lists where auto-expansion would flood the screen. */
  defaultOpen?: boolean
}) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div className="relative rounded border border-line-1 bg-surface-2">
      {open && (
        <div
          aria-hidden
          className="absolute left-0 inset-y-1 w-0.5 bg-gradient-brand rounded-full pointer-events-none"
        />
      )}
      <div
        role="button"
        tabIndex={-1}
        onClick={(e) => {
          if ((e.target as HTMLElement).closest('[data-no-toggle]')) return
          setOpen((v) => !v)
        }}
        className="flex items-center gap-2 px-3 py-2 cursor-pointer hover:bg-surface-2/60 transition-colors"
      >
        <span
          aria-hidden
          className={`text-ink-3 text-[10px] transition-transform ${open ? 'rotate-90' : ''}`}
        >
          ▸
        </span>
        <span className="text-xs font-medium text-ink-1">{title}</span>
        <button
          type="button"
          data-no-toggle
          onClick={(e) => {
            e.stopPropagation()
            onRemove()
          }}
          title="Remove item"
          className="ml-auto w-6 h-6 rounded text-err hover:bg-err/10 transition text-xs"
        >
          ✕
        </button>
      </div>
      {open && <div className="px-3 pb-3 pt-1">{children}</div>}
    </div>
  )
}
