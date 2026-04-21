import type { JsonSchemaNode, PipelineJsonSchema } from '../../api/hooks/useConfigSchema'
import { resolveRef, titleOrKey, topLevelLabel, visibleTopLevelKeys } from './schemaUtils'

export type GroupValidity = 'ok' | 'warn' | 'err' | 'idle'

const DOT_CLS: Record<GroupValidity, string> = {
  ok: 'bg-ok',
  warn: 'bg-warn',
  err: 'bg-err',
  idle: 'bg-ink-4/40',
}

const DOT_LABEL: Record<GroupValidity, string> = {
  ok: 'validated',
  warn: 'warning',
  err: 'has issues',
  idle: 'not validated yet',
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
  const keys = visibleTopLevelKeys(Object.keys(topProps))

  return (
    <nav className="sticky top-4 flex flex-col gap-0.5">
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
            aria-current={isActive ? 'page' : undefined}
            className={[
              // Minimal active state: brighter text + subtle bg lift.
              // Aside now has transparent bg (inherits card surface-2),
              // so active uses surface-3 — one step LIGHTER than the
              // surrounding card — for a clean "raised tile" read.
              'w-full flex items-center gap-2 px-2 py-1.5 rounded-md text-xs text-left transition-colors',
              isActive
                ? 'text-ink-1 font-medium bg-surface-3'
                : 'text-ink-2 hover:text-ink-1 hover:bg-surface-3/50',
            ].join(' ')}
          >
            <span
              aria-hidden="true"
              title={DOT_LABEL[dot]}
              className={`w-1.5 h-1.5 rounded-full shrink-0 ${DOT_CLS[dot]}`}
            />
            <span className="truncate flex-1">{label}</span>
          </button>
        )
      })}
    </nav>
  )
}
