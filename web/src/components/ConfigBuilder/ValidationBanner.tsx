import { useEffect, useRef, useState } from 'react'
import type { ConfigValidationResult } from '../../api/types'
import { AlertIcon, CheckIcon, ChevronRightIcon, InfoIcon } from '../icons'
import { groupForCheck, SETTINGS_JUMP_TARGET } from './validationMap'

interface Props {
  result: ConfigValidationResult | null
  isValidating: boolean
  hashPrefix?: string
  onJump?: (group: string) => void
  /** Called when the user clicks a specific field error row with the
   *  full dotted path (e.g. ``training.strategies.0.strategy_type``).
   *  ConfigTab wires this to switch from YAML to Form view and set the
   *  hash so the FieldAnchor scroll kicks in. */
  onJumpToField?: (path: string) => void
}

export function ValidationBanner({
  result,
  isValidating,
  hashPrefix,
  onJump,
  onJumpToField,
}: Props) {
  const [expanded, setExpanded] = useState(false)

  const failures = (result?.checks ?? []).filter((c) => c.status !== 'ok')
  const hasWarnings = failures.some((c) => c.status === 'warn')
  const ok = result?.ok ?? true

  // Flatten backend-emitted field_errors into discrete rows. One
  // row per (path, message) pair, rendered as a simple bullet list.
  // Pydantic's "Value error, " prefix is noise for humans — strip it.
  const fieldErrorRows = Object.entries(result?.field_errors ?? {}).flatMap(
    ([path, msgs]) =>
      (msgs ?? []).map((raw) => ({
        path,
        msg: raw.replace(/^Value error,\s*/i, '').trim(),
      })),
  )
  const totalFieldErrors = fieldErrorRows.length

  // Brief green glow when validation first goes fully green.
  // Edge-trigger on the false → (ok && !hasWarnings) transition so
  // we don't re-glow on every re-render after success. Fades in ~800ms.
  const [successFlash, setSuccessFlash] = useState(false)
  const wasOkRef = useRef(false)
  useEffect(() => {
    const fullyOk = Boolean(result && result.ok && !hasWarnings && !isValidating)
    if (fullyOk && !wasOkRef.current) {
      setSuccessFlash(true)
      const t = window.setTimeout(() => setSuccessFlash(false), 900)
      wasOkRef.current = true
      return () => window.clearTimeout(t)
    }
    if (!fullyOk) wasOkRef.current = false
  }, [result, hasWarnings, isValidating])

  // Auto-expand when errors first appear, edge-triggered on the
  // no-errors → has-errors transition. Without this, a fail result
  // collapses down to just "3 field error(s)" and the user has to
  // know to click — they don't, and the paths stay hidden. Once
  // expanded, respect manual collapse until the error set clears
  // and comes back.
  const hadErrorsRef = useRef(false)
  const errorCount = failures.length + Object.keys(result?.field_errors ?? {}).length
  useEffect(() => {
    // Auto-EXPAND only. We used to also scrollIntoView + focus the
    // banner on the 0→N edge, but that fights the user: while they're
    // typing to fix one field, the error count briefly hits 0, the
    // next keystroke re-triggers validation with new errors, and
    // focus snaps back to the banner — losing their caret. The
    // GOV.UK focus-banner pattern is for SUBMIT flows, not live
    // validation, so we drop it here.
    if (errorCount > 0 && !hadErrorsRef.current) {
      setExpanded(true)
      hadErrorsRef.current = true
    } else if (errorCount === 0) {
      hadErrorsRef.current = false
    }
  }, [errorCount])

  if (!result && !isValidating) return null

  // State-driven palette. `left` is a vertical accent bar on the left
  // edge (the card silhouette) — pulls the banner out of "tinted fill
  // block" into "status card with accent", much less visually noisy.
  const tone = ok ? (hasWarnings ? 'warn' : 'ok') : 'err'
  const palette: Record<
    'err' | 'warn' | 'ok' | 'loading',
    { chrome: string; left: string; badge: string; icon: string }
  > = {
    err: {
      chrome: 'border-err/30 bg-err/[0.06] text-ink-1',
      left: 'bg-err',
      badge: 'bg-err/20 text-err border-err/40',
      icon: 'text-err',
    },
    warn: {
      chrome: 'border-warn/30 bg-warn/[0.05] text-ink-1',
      left: 'bg-warn',
      badge: 'bg-warn/20 text-warn border-warn/40',
      icon: 'text-warn',
    },
    ok: {
      chrome: 'border-ok/25 bg-ok/[0.04] text-ink-1',
      left: 'bg-ok',
      badge: 'bg-ok/20 text-ok border-ok/40',
      icon: 'text-ok',
    },
    loading: {
      chrome: 'border-line-2 bg-surface-2 text-ink-2',
      left: 'bg-info/60',
      badge: 'bg-info/15 text-info border-info/30',
      icon: 'text-info',
    },
  }
  const p = palette[isValidating ? 'loading' : tone]

  const failCount = failures.filter((c) => c.status === 'fail').length
  const warnCount = failures.filter((c) => c.status === 'warn').length

  const title = isValidating
    ? 'Validating…'
    : ok
    ? hasWarnings
      ? 'Configuration valid — with warnings'
      : 'Configuration is valid'
    : failCount === 1
    ? '1 issue to fix'
    : `${failCount} issues to fix`

  const subtitle = isValidating
    ? null
    : ok
    ? hasWarnings
      ? `${warnCount} warning${warnCount === 1 ? '' : 's'} — review when you can`
      : 'Ready to save or launch a run.'
    : warnCount > 0
    ? `${warnCount} warning${warnCount === 1 ? '' : 's'} + ${failCount} error${
        failCount === 1 ? '' : 's'
      }`
    : null

  const Icon = isValidating
    ? InfoIcon
    : tone === 'ok'
    ? CheckIcon
    : AlertIcon

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
    <div
      className={`relative overflow-hidden rounded-lg border ${p.chrome} transition-shadow duration-[700ms] ${
        successFlash ? 'shadow-[0_0_40px_rgba(74,222,128,0.28)]' : 'shadow-card'
      }`}
    >
      {/* Left accent bar — carries the status colour at full intensity
          so the banner reads from the corner of the eye without the fill
          having to compete with it. */}
      <span aria-hidden="true" className={`absolute inset-y-0 left-0 w-[3px] ${p.left}`} />

      <button
        type="button"
        onClick={() => setExpanded((v) => !v)}
        disabled={failures.length === 0 && totalFieldErrors === 0}
        className="w-full flex items-center gap-3 pl-4 pr-3 py-2.5 text-left hover:bg-white/[0.02] transition disabled:cursor-default"
      >
        <Icon className={`w-4 h-4 shrink-0 ${p.icon}`} />
        <div className="min-w-0 flex-1 flex items-baseline gap-3 flex-wrap">
          <span className="text-sm font-semibold text-ink-1">{title}</span>
          {subtitle && (
            <span className="text-[0.7rem] text-ink-3">{subtitle}</span>
          )}
        </div>
        {(failures.length > 0 || totalFieldErrors > 0) && (
          <>
            {failCount > 0 && (
              <span className={`inline-flex items-center h-5 px-1.5 rounded border text-[0.65rem] font-mono font-medium tabular-nums ${palette.err.badge}`}>
                {failCount}
              </span>
            )}
            {warnCount > 0 && (
              <span className={`inline-flex items-center h-5 px-1.5 rounded border text-[0.65rem] font-mono font-medium tabular-nums ${palette.warn.badge}`}>
                {warnCount}
              </span>
            )}
            <span className="text-ink-3 shrink-0">
              <ChevronRightIcon
                className={`w-3.5 h-3.5 transition-transform ${expanded ? 'rotate-90' : ''}`}
              />
            </span>
          </>
        )}
      </button>
      {expanded && (failures.length > 0 || totalFieldErrors > 0) && (
        <div className="border-t border-line-1/60 bg-surface-0/40 pl-4 pr-3 py-2 space-y-1.5">
          {(() => {
            // Promote secret-related failures to a prominent "Open
            // project Settings" CTA at the top of the expanded area.
            const settingsFailures = failures.filter(
              (c) => groupForCheck(c) === SETTINGS_JUMP_TARGET,
            )
            if (settingsFailures.length === 0) return null
            const label =
              settingsFailures.length === 1
                ? '1 secret missing'
                : `${settingsFailures.length} secrets missing`
            return (
              <button
                type="button"
                onClick={() => jumpTo(SETTINGS_JUMP_TARGET)}
                className="group w-full flex items-center justify-between gap-2 rounded-md border border-line-2 bg-surface-2 hover:bg-surface-3 hover:border-brand/40 px-3 py-2 mb-2 transition"
              >
                <span className="flex items-center gap-2.5">
                  <AlertIcon className="w-4 h-4 text-err shrink-0" />
                  <span className="flex flex-col items-start gap-0.5">
                    <span className="text-xs font-medium text-ink-1">
                      Open project Settings
                    </span>
                    <span className="text-[0.65rem] text-ink-3">
                      {label} — set env vars to unblock runs
                    </span>
                  </span>
                </span>
                <ChevronRightIcon className="w-3.5 h-3.5 text-ink-3 group-hover:text-ink-1 transition-colors" />
              </button>
            )
          })()}
          {/* Field-level errors from Pydantic validation. Shown first
              because they're the most actionable — each row carries a
              dotted path (so the user knows WHERE) and the full
              message with newlines preserved (so multi-line validator
              messages render correctly). */}
          {fieldErrorRows.length > 0 && (
            <div
              role="alert"
              aria-live="assertive"
              className="mb-2 space-y-1"
            >
              {fieldErrorRows.map((row, i) => (
                <div
                  key={`fe-${i}`}
                  className="flex items-start gap-2.5 py-1 text-[0.7rem]"
                >
                  <span
                    aria-hidden
                    className="w-1.5 h-1.5 mt-1.5 rounded-full shrink-0 bg-err"
                  />
                  <div className="min-w-0 flex-1">
                    <div className="text-ink-1 whitespace-pre-wrap break-words">
                      {row.msg}
                    </div>
                    <div className="mt-0.5 font-mono text-[0.65rem] text-ink-4 break-all">
                      {row.path}
                    </div>
                  </div>
                  {onJumpToField && (
                    <button
                      type="button"
                      onClick={() => onJumpToField(row.path)}
                      title={`Open ${row.path} in the form`}
                      className="inline-flex items-center gap-1 text-[0.6rem] uppercase tracking-wide text-ink-3 hover:text-brand transition whitespace-nowrap"
                    >
                      open
                      <ChevronRightIcon className="w-3 h-3" />
                    </button>
                  )}
                </div>
              ))}
            </div>
          )}
          {/* Suppress the "YAML schema — N field error(s)" summary
              row when we've already listed each field above. The
              count is redundant once the list is visible. */}
          {failures
            .filter(
              (c) => !(totalFieldErrors > 0 && c.label === 'YAML schema'),
            )
            .map((check, idx) => {
            const group = groupForCheck(check)
            const isSettings = group === SETTINGS_JUMP_TARGET
            const dotCls =
              check.status === 'warn'
                ? 'bg-warn'
                : check.status === 'fail'
                ? 'bg-err'
                : 'bg-ink-3'
            return (
              <div
                key={idx}
                className="flex items-start gap-2.5 py-1 text-[0.7rem]"
              >
                <span
                  aria-hidden
                  className={`w-1.5 h-1.5 mt-1.5 rounded-full shrink-0 ${dotCls}`}
                />
                <div className="min-w-0 flex-1">
                  <div className="text-ink-1">{check.label}</div>
                  {check.detail && (
                    <div className="text-ink-3 font-mono text-[0.65rem] truncate">
                      {check.detail}
                    </div>
                  )}
                </div>
                {group && !isSettings && (
                  <button
                    type="button"
                    onClick={() => jumpTo(group)}
                    className="inline-flex items-center gap-1 text-[0.6rem] uppercase tracking-wide text-ink-3 hover:text-brand transition whitespace-nowrap"
                  >
                    {group}
                    <ChevronRightIcon className="w-3 h-3" />
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
