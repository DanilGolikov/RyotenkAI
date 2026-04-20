import type { JsonSchemaNode, PipelineJsonSchema } from '../../api/hooks/useConfigSchema'
import { resolveRef, titleOrKey, topLevelLabel, visibleTopLevelKeys } from './schemaUtils'
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
  const keys = visibleTopLevelKeys(Object.keys(topProps))

  return (
    <div className="sticky top-0 z-10 -mx-1 px-1 pt-1 bg-surface-1/90 backdrop-blur border-b border-line-1 overflow-x-auto">
      <div className="flex gap-1 min-w-max">
        {keys.map((key) => {
          const node = resolveRef(schema, topProps[key])
          const label = topLevelLabel(key, titleOrKey(node, key))
          const isActive = key === active
          // Default to red: if the server hasn't confirmed this group is
          // clean yet, assume there's still something to fix. Server
          // validation flips it to ok/warn/err based on real checks.
          const dot = validity?.[key] ?? 'err'
          return (
            <button
              key={key}
              type="button"
              onClick={() => onSelect(key)}
              className={[
                'flex items-center gap-1.5 px-3 py-2 text-xs transition whitespace-nowrap',
                // Flat underline subtab — mirrors the main Info/Config
                // tabs but uses brand-alt (violet) to distinguish the
                // hierarchy: main tabs = burgundy, subtabs = violet.
                isActive
                  ? 'text-ink-1 border-b-2 border-brand-alt -mb-px'
                  : 'text-ink-3 hover:text-ink-1 border-b-2 border-transparent',
              ].join(' ')}
            >
              <span className={`w-1.5 h-1.5 rounded-full ${DOT_CLS[dot]}`} />
              <span>{label}</span>
            </button>
          )
        })}
      </div>
    </div>
  )
}
