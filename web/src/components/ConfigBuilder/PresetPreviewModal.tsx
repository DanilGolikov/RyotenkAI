import { useEffect, useRef } from 'react'
import type { ConfigPreset } from '../../api/types'
import { deepDiff } from '../../lib/jsonDiff'
import { safeYamlParse } from '../../lib/yaml'
import { useClickOutside } from '../../hooks/useClickOutside'

interface Props {
  preset: ConfigPreset
  /** Current form value to diff against the preset's YAML. */
  current: Record<string, unknown>
  /** Whether the user has unsaved changes — used to reword the
   *  confirmation button so they realise "Apply" overwrites their
   *  work. */
  dirty: boolean
  onCancel: () => void
  onApply: () => void
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
 * "You're about to load preset X — here's exactly what will change"
 * dialog. Previously a preset click wrote the YAML in place and the
 * user had to reconstruct what was lost; this surfaces the diff
 * upfront so Apply is an informed choice. Reuses `deepDiff` (powers
 * DiffBadge) — diff semantics stay consistent across the app.
 */
export function PresetPreviewModal({ preset, current, dirty, onApply, onCancel }: Props) {
  const panelRef = useRef<HTMLDivElement | null>(null)
  useClickOutside(panelRef, true, onCancel)

  const parsed = safeYamlParse(preset.yaml) ?? {}
  const diff = deepDiff(current, parsed)

  // Auto-focus Apply button so keyboard users can confirm with Enter.
  const applyBtnRef = useRef<HTMLButtonElement | null>(null)
  useEffect(() => {
    applyBtnRef.current?.focus()
  }, [])

  return (
    <div
      className="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm flex items-center justify-center p-6"
      role="dialog"
      aria-modal="true"
      aria-labelledby="preset-preview-title"
    >
      <div
        ref={panelRef}
        className="w-full max-w-2xl max-h-[80vh] overflow-hidden rounded-xl border border-line-2 bg-surface-1 shadow-card flex flex-col"
      >
        <div className="px-4 py-3 border-b border-line-1 flex items-center justify-between">
          <div>
            <div id="preset-preview-title" className="text-sm font-semibold text-ink-1">
              Load preset <span className="font-mono text-brand-alt">{preset.name}</span>?
            </div>
            <div className="text-2xs text-ink-3">
              {diff.length === 0
                ? 'No changes — preset matches the current config.'
                : `${diff.length} field${diff.length === 1 ? '' : 's'} will change.`}
              {dirty && (
                <span className="ml-2 text-brand-strong">
                  You have unsaved changes that will be overwritten.
                </span>
              )}
            </div>
          </div>
          <button
            type="button"
            onClick={onCancel}
            className="text-ink-3 hover:text-ink-1 text-xs"
          >
            close
          </button>
        </div>

        {preset.description && (
          <div className="px-4 py-2 border-b border-line-1 text-xs text-ink-2 leading-snug">
            {preset.description}
          </div>
        )}

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

        <div className="px-4 py-3 border-t border-line-1 flex items-center justify-end gap-2">
          <button
            type="button"
            onClick={onCancel}
            className="btn-ghost h-8 text-xs"
          >
            Cancel
          </button>
          <button
            ref={applyBtnRef}
            type="button"
            onClick={onApply}
            className="btn-primary h-8 text-xs px-3"
          >
            {dirty ? 'Overwrite & apply' : 'Apply preset'}
          </button>
        </div>
      </div>
    </div>
  )
}
