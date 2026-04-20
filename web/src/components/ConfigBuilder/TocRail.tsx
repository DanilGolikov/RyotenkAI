import type { JsonSchemaNode, PipelineJsonSchema } from '../../api/hooks/useConfigSchema'
import { resolveRef, titleOrKey, topLevelLabel, visibleTopLevelKeys } from './schemaUtils'

export type GroupValidity = 'ok' | 'warn' | 'err' | 'idle'

const DOT_CLS: Record<GroupValidity, string> = {
  ok: 'bg-ok',
  warn: 'bg-warn',
  err: 'bg-err',
  idle: 'bg-ink-4/40',
}

export function TocRail({
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
  const keys = visibleTopLevelKeys(Object.keys(topProps))

  return (
    <nav className="sticky top-4 space-y-0.5">
      <div className="text-2xs uppercase tracking-wide text-ink-3 px-2 mb-1">Sections</div>
      {keys.map((key) => {
        const node = resolveRef(schema, topProps[key])
        const label = topLevelLabel(key, titleOrKey(node, key))
        const isActive = key === active
        const dot = validity?.[key] ?? 'idle'
        return (
          <button
            key={key}
            type="button"
            onClick={() => onSelect(key)}
            className={[
              'w-full flex items-center gap-2 px-2 py-1.5 rounded-md text-xs text-left transition',
              isActive
                ? 'bg-surface-3 text-ink-1 border-l-2 border-brand -ml-0.5 pl-[0.625rem]'
                : 'text-ink-2 hover:text-ink-1 hover:bg-surface-2',
            ].join(' ')}
          >
            <span className={`w-1.5 h-1.5 rounded-full ${DOT_CLS[dot]}`} />
            <span className="truncate flex-1">{label}</span>
            {required.has(key) && key !== 'model' && (
              <span className="text-[0.55rem] text-brand">req</span>
            )}
          </button>
        )
      })}
    </nav>
  )
}
