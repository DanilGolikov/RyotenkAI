import { useState } from 'react'
import type { ConfigValidationResult } from '../../api/types'
import { groupForCheck } from './validationMap'

interface Props {
  result: ConfigValidationResult | null
  isValidating: boolean
  hashPrefix?: string
  onJump?: (group: string) => void
}

export function ValidationBanner({ result, isValidating, hashPrefix, onJump }: Props) {
  const [expanded, setExpanded] = useState(false)

  if (!result && !isValidating) return null

  const failures = (result?.checks ?? []).filter((c) => c.status !== 'ok')
  const hasWarnings = failures.some((c) => c.status === 'warn')
  const ok = result?.ok ?? true

  const cls = ok
    ? hasWarnings
      ? 'border-warn/40 bg-warn/10 text-warn'
      : 'border-ok/40 bg-ok/10 text-ok'
    : 'border-err/40 bg-err/10 text-err'

  const title = isValidating
    ? 'Validating…'
    : ok
    ? hasWarnings
      ? `Valid (${failures.length} warning${failures.length === 1 ? '' : 's'})`
      : 'Configuration is valid'
    : `${failures.filter((c) => c.status === 'fail').length} issue${
        failures.filter((c) => c.status === 'fail').length === 1 ? '' : 's'
      } to fix`

  function jumpTo(group: string | null) {
    if (!group) return
    if (onJump) {
      onJump(group)
      return
    }
    const prefix = hashPrefix ? `${hashPrefix}:` : ''
    const nextHash = `#${prefix}${group}`
    if (window.location.hash !== nextHash) {
      history.replaceState(null, '', nextHash)
      window.dispatchEvent(new HashChangeEvent('hashchange'))
    }
  }

  return (
    <div className={`rounded-md border ${cls} text-xs`}>
      <button
        type="button"
        onClick={() => setExpanded((v) => !v)}
        className="w-full flex items-center gap-2 px-3 py-2 text-left hover:bg-black/10 transition"
      >
        <span className={`w-1.5 h-1.5 rounded-full ${ok ? (hasWarnings ? 'bg-warn' : 'bg-ok') : 'bg-err'}`} />
        <span className="font-medium">{title}</span>
        {failures.length > 0 && (
          <span className="ml-auto text-[0.65rem] opacity-70">
            {expanded ? 'hide' : 'details'}
          </span>
        )}
      </button>
      {expanded && failures.length > 0 && (
        <div className="border-t border-current/20 px-3 py-2 space-y-1.5">
          {failures.map((check, idx) => {
            const group = groupForCheck(check)
            return (
              <div key={idx} className="flex items-start gap-2 text-2xs">
                <span
                  className={[
                    'w-1 h-1 mt-1.5 rounded-full shrink-0',
                    check.status === 'warn' ? 'bg-warn' : 'bg-err',
                  ].join(' ')}
                />
                <div className="min-w-0 flex-1">
                  <div className="text-ink-1">{check.label}</div>
                  {check.detail && (
                    <div className="text-ink-3 font-mono truncate">{check.detail}</div>
                  )}
                </div>
                {group && (
                  <button
                    type="button"
                    onClick={() => jumpTo(group)}
                    className="text-[0.65rem] uppercase tracking-wide text-ink-1 hover:text-ink-2 whitespace-nowrap"
                  >
                    → {group}
                  </button>
                )}
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
