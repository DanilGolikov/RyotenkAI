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
  const required = new Set<string>(Array.isArray(schema.required) ? schema.required : [])
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
              'w-full flex items-center gap-2 px-2 py-1.5 rounded-md text-xs text-left transition',
              isActive
                // Active section wears brand — weak burgundy bg + strong text +
                // thicker brand left border. Reads as "you are here" instead
                // of merely "lighter grey".
                ? 'bg-brand-weak/50 text-brand-strong border-l-2 border-brand -ml-0.5 pl-[0.625rem] shadow-[inset_0_0_0_1px_rgba(237,72,127,0.12)]'
                : 'text-ink-2 hover:text-ink-1 hover:bg-surface-2',
            ].join(' ')}
          >
            <span
              aria-hidden="true"
              title={DOT_LABEL[dot]}
              className={`w-1.5 h-1.5 rounded-full shrink-0 ${DOT_CLS[dot]}`}
            />
            <span className="truncate flex-1">{label}</span>
            {required.has(key) && key !== 'model' && (
              <span className="text-[0.55rem] text-brand">req</span>
            )}
          </button>
        )
      })}
      <div className="mt-3 px-2 flex items-center gap-2 text-[0.6rem] text-ink-4 select-none">
        <span className="w-1 h-1 rounded-full bg-ok" aria-hidden="true" />
        <span>ok</span>
        <span className="w-1 h-1 rounded-full bg-warn ml-1" aria-hidden="true" />
        <span>warn</span>
        <span className="w-1 h-1 rounded-full bg-err ml-1" aria-hidden="true" />
        <span>err</span>
      </div>
    </nav>
  )
}
