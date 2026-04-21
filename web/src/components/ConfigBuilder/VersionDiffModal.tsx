import { useRef } from 'react'
import { deepDiff } from '../../lib/jsonDiff'
import { safeYamlParse } from '../../lib/yaml'
import { useClickOutside } from '../../hooks/useClickOutside'

interface Props {
  /** Baseline YAML — the version the user is comparing AGAINST. */
  baselineYaml: string
  baselineLabel: string
  /** Current form value — already parsed, so we don't need YAML here. */
  current: Record<string, unknown>
  onClose: () => void
}

const KIND_CLS: Record<'changed' | 'added' | 'removed', string> = {
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

/**
 * Read-only diff view: "how does the current form differ from a prior
 * config snapshot?". Used by the "Compare with last run" button to
 * surface changes the user has made since the last saved version —
 * no apply action, just information. Reuses `deepDiff` semantics for
 * consistency with PresetPreviewModal.
 */
export function VersionDiffModal({ baselineYaml, baselineLabel, current, onClose }: Props) {
  const panelRef = useRef<HTMLDivElement | null>(null)
  useClickOutside(panelRef, true, onClose)

  const baseline = (safeYamlParse(baselineYaml) ?? {}) as Record<string, unknown>
  const diff = deepDiff(baseline, current)

  return (
    <div
      className="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm flex items-center justify-center p-6"
      role="dialog"
      aria-modal="true"
      aria-labelledby="version-diff-title"
    >
      <div
        ref={panelRef}
        className="w-full max-w-2xl max-h-[80vh] overflow-hidden rounded-xl border border-line-2 bg-surface-1 shadow-card flex flex-col"
      >
        <div className="px-4 py-3 border-b border-line-1 flex items-center justify-between">
          <div>
            <div id="version-diff-title" className="text-sm font-semibold text-ink-1">
              Changes since <span className="font-mono text-brand-alt">{baselineLabel}</span>
            </div>
            <div className="text-2xs text-ink-3">
              {diff.length === 0
                ? 'Current config matches the baseline.'
                : `${diff.length} field${diff.length === 1 ? '' : 's'} differ.`}
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="text-ink-3 hover:text-ink-1 text-xs"
          >
            close
          </button>
        </div>

        <div className="p-3 text-xs space-y-2 overflow-y-auto">
          {diff.length === 0 ? (
            <div className="text-ink-3 px-2 py-6 text-center">No differences.</div>
          ) : (
            diff.map((d) => (
              <div key={d.path} className="rounded-md border border-line-1 bg-surface-0 p-2">
                <div className="flex items-baseline gap-2">
                  <span
                    className={`text-[0.65rem] uppercase tracking-wide ${KIND_CLS[d.kind]}`}
                  >
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
            ))
          )}
        </div>
      </div>
    </div>
  )
}
