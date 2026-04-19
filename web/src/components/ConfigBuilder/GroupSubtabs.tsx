import type { JsonSchemaNode, PipelineJsonSchema } from '../../api/hooks/useConfigSchema'
import { resolveRef, titleOrKey } from './schemaUtils'
import type { GroupValidity } from './TocRail'

const DOT_CLS: Record<GroupValidity, string> = {
  ok: 'bg-ok',
  warn: 'bg-warn',
  err: 'bg-err',
  idle: 'bg-ink-4/40',
}

/**
 * Horizontal list of group subtabs — sticky at the top of the ConfigBuilder.
 * Semantically identical to TocRail but optimised for top-of-content navigation
 * and short schemas where the vertical rail wastes space.
 */
export function GroupSubtabs({
  schema,
  active,
  onSelect,
  validity,
}: {
  schema: PipelineJsonSchema
  active: string | null
  onSelect: (key: string) => void
  validity?: Partial<Record<string, GroupValidity>>
}) {
  const topProps = (schema.properties ?? {}) as Record<string, JsonSchemaNode>
  const required = new Set<string>(Array.isArray(schema.required) ? schema.required : [])
  const keys = Object.keys(topProps)

  return (
    <div className="sticky top-0 z-10 -mx-1 px-1 pb-2 pt-1 bg-surface-1/90 backdrop-blur border-b border-line-1 overflow-x-auto">
      <div className="flex gap-1 min-w-max">
        {keys.map((key) => {
          const node = resolveRef(schema, topProps[key])
          const label = titleOrKey(node, key)
          const isActive = key === active
          const dot = validity?.[key] ?? 'idle'
          return (
            <button
              key={key}
              type="button"
              onClick={() => onSelect(key)}
              className={[
                'flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs transition whitespace-nowrap border',
                isActive
                  ? 'bg-surface-2 text-ink-1 border-brand/60 shadow-inset-accent'
                  : 'text-ink-2 hover:text-ink-1 border-line-1 hover:border-line-2 hover:bg-surface-2/60',
              ].join(' ')}
            >
              <span className={`w-1.5 h-1.5 rounded-full ${DOT_CLS[dot]}`} />
              <span>{label}</span>
              {required.has(key) && !isActive && (
                <span className="text-[0.55rem] text-brand">req</span>
              )}
            </button>
          )
        })}
      </div>
    </div>
  )
}
