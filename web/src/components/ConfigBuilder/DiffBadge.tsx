import { useEffect, useMemo, useRef, useState } from 'react'
import { deepDiff, type DiffEntry } from '../../lib/jsonDiff'

const KIND_CLS: Record<DiffEntry['kind'], string> = {
  changed: 'text-warn',
  added: 'text-ok',
  removed: 'text-err',
}

function formatVal(v: unknown): string {
  if (v === undefined) return '—'
  if (typeof v === 'string') return v
  try {
    const s = JSON.stringify(v)
    return s.length > 80 ? s.slice(0, 79) + '…' : s
  } catch {
    return String(v)
  }
}

interface Props {
  presetName: string
  baseline: Record<string, unknown>
  current: Record<string, unknown>
  onClear: () => void
}

export function DiffBadge({ presetName, baseline, current, onClear }: Props) {
  const [open, setOpen] = useState(false)
  const diff = useMemo(() => deepDiff(baseline, current), [baseline, current])

  // Brief brand-burgundy glow when the badge appears (a preset was
  // applied) or when the preset changes (swapped for another). Fades
  // after ~2.5s. Provides non-modal "something happened" signal so
  // the user doesn't miss the quiet state change in the corner.
  const [flash, setFlash] = useState(true)
  const prevNameRef = useRef(presetName)
  useEffect(() => {
    if (prevNameRef.current !== presetName) setFlash(true)
    prevNameRef.current = presetName
    const t = window.setTimeout(() => setFlash(false), 2500)
    return () => window.clearTimeout(t)
  }, [presetName])

  return (
    <div className="inline-flex items-center gap-2">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className={`rounded-md border border-brand-alt/50 bg-brand-alt/10 text-brand-alt px-2 py-1 text-[0.65rem] transition-shadow duration-[1500ms] ${
          flash
            ? 'shadow-[0_0_20px_rgba(237,72,127,0.45)]'
            : 'shadow-none'
        }`}
      >
        from <span className="font-mono">{presetName}</span>
        {diff.length > 0 && <span className="ml-1">· {diff.length} diff</span>}
      </button>
      <button
        type="button"
        onClick={onClear}
        className="text-ink-3 hover:text-ink-1 text-[0.65rem]"
        title="Clear preset baseline"
      >
        clear
      </button>
      {open && (
        <div
          onClick={() => setOpen(false)}
          className="fixed inset-0 z-40 bg-black/60 backdrop-blur-sm flex items-start justify-end p-4"
        >
          <div
            onClick={(e) => e.stopPropagation()}
            className="w-full max-w-xl max-h-[80vh] overflow-auto rounded-xl border border-line-2 bg-surface-1 shadow-card"
          >
            <div className="px-4 py-3 border-b border-line-1 flex items-center justify-between">
              <div>
                <div className="text-sm font-semibold">
                  Diff from preset <span className="font-mono text-brand-alt">{presetName}</span>
                </div>
                <div className="text-2xs text-ink-3">
                  {diff.length === 0
                    ? 'No differences — matches the baseline.'
                    : `${diff.length} field${diff.length === 1 ? '' : 's'} modified`}
                </div>
              </div>
              <button
                type="button"
                onClick={() => setOpen(false)}
                className="text-ink-3 hover:text-ink-1 text-xs"
              >
                close
              </button>
            </div>
            <div className="p-3 text-xs space-y-2">
              {diff.map((d) => (
                <div key={d.path} className="rounded-md border border-line-1 bg-surface-0 p-2">
                  <div className="flex items-baseline gap-2">
                    <span className={`text-[0.65rem] uppercase tracking-wide ${KIND_CLS[d.kind]}`}>
                      {d.kind}
                    </span>
                    <span className="font-mono text-ink-1">{d.path}</span>
                  </div>
                  <div className="mt-1 font-mono text-2xs space-y-0.5">
                    {d.kind !== 'added' && (
                      <div className="text-ink-3">
                        − <span className="text-err/80">{formatVal(d.baseline)}</span>
                      </div>
                    )}
                    {d.kind !== 'removed' && (
                      <div className="text-ink-3">
                        + <span className="text-ok/80">{formatVal(d.current)}</span>
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
