import { useEffect, useRef } from 'react'
import { useClickOutside } from '../../hooks/useClickOutside'
import { usePresetPreview } from '../../api/hooks/usePresetPreview'
import type {
  ConfigPreset,
  PresetDiffEntry,
  PresetRequirementCheck,
} from '../../api/types'

interface Props {
  preset: ConfigPreset
  /** Current form value diffed against the preset's YAML on the backend. */
  current: Record<string, unknown>
  /** Unsaved edits marker — reflowed into the Apply label so the user
   *  knows they're overwriting something. */
  dirty: boolean
  onCancel: () => void
  onApply: () => void
}

const STATUS_STYLE: Record<PresetRequirementCheck['status'], string> = {
  ok: 'bg-ok/15 text-ok border-ok/30',
  warning: 'bg-warn/15 text-warn border-warn/30',
  missing: 'bg-err/15 text-err border-err/30',
}

const STATUS_GLYPH: Record<PresetRequirementCheck['status'], string> = {
  ok: '✓',
  warning: '!',
  missing: '✗',
}

/** Reason → short prefix chip next to each changed key. */
const REASON_STYLE: Record<PresetDiffEntry['reason'], { label: string; cls: string }> = {
  preset_replaced: { label: 'replaced', cls: 'bg-brand-alt/15 text-brand-alt border-brand-alt/30' },
  preset_added:    { label: 'added',    cls: 'bg-ok/15 text-ok border-ok/30' },
  preset_preserved:{ label: 'preserved',cls: 'bg-surface-2 text-ink-3 border-line-2' },
  no_scope:        { label: 'overwrite',cls: 'bg-warn/15 text-warn border-warn/30' },
}

function formatVal(v: unknown): string {
  if (v === undefined || v === null) return '—'
  if (typeof v === 'string') return v
  try {
    const s = JSON.stringify(v)
    return s.length > 80 ? s.slice(0, 79) + '…' : s
  } catch {
    return String(v)
  }
}

/**
 * Preset-apply preview — replaces the old client-side deepDiff view with
 * the backend's structured apply analysis (POST /config/presets/{id}/preview).
 * Four stacked sections so the user can reason about each aspect
 * independently:
 *
 * 1. **Requirements** — green/amber/red readiness checks (hub models,
 *    provider kind, plugins, VRAM hint).
 * 2. **What changes** — keys this preset rewrites or adds.
 * 3. **What's preserved** — keys the preset declared as ``preserves``,
 *    kept verbatim from your config.
 * 4. **Placeholders to fill** — paths you'll still need to edit after
 *    Apply (e.g. dataset JSONL path).
 *
 * A banner at the top surfaces any warnings (e.g. v1 "full overwrite"
 * legacy behaviour when the preset has no scope).
 */
export function PresetPreviewModal({ preset, current, dirty, onApply, onCancel }: Props) {
  const panelRef = useRef<HTMLDivElement | null>(null)
  useClickOutside(panelRef, true, onCancel)
  const applyBtnRef = useRef<HTMLButtonElement | null>(null)

  const { data, isLoading, error } = usePresetPreview(preset.name, current)

  useEffect(() => {
    applyBtnRef.current?.focus()
  }, [])

  const requirements = data?.requirements ?? []
  const diff = data?.diff ?? []
  const placeholders = data?.placeholders ?? []
  const warnings = data?.warnings ?? []

  // Partition diff by reason for the three content sections.
  const changedEntries = diff.filter(
    (d) => d.reason === 'preset_replaced' || d.reason === 'preset_added' || d.reason === 'no_scope',
  )
  const preservedEntries = diff.filter((d) => d.reason === 'preset_preserved')

  return (
    <div
      className="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm flex items-center justify-center p-6"
      role="dialog"
      aria-modal="true"
      aria-labelledby="preset-preview-title"
    >
      <div
        ref={panelRef}
        className="w-full max-w-2xl max-h-[85vh] overflow-hidden rounded-xl border border-line-2 bg-surface-1 shadow-card flex flex-col"
      >
        {/* Header */}
        <div className="px-4 py-3 border-b border-line-1 flex items-center justify-between">
          <div>
            <div id="preset-preview-title" className="text-sm font-semibold text-ink-1">
              Load preset{' '}
              <span className="font-semibold text-brand-alt">
                {preset.display_name || preset.name}
              </span>
              ?
            </div>
            <div className="text-2xs text-ink-3">
              {isLoading
                ? 'Computing diff…'
                : `${changedEntries.length} change${changedEntries.length === 1 ? '' : 's'}`
                  + ` · ${preservedEntries.length} preserved`
                  + ` · ${placeholders.length} to fill`}
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

        {/* Body */}
        <div className="p-3 text-xs space-y-4 overflow-y-auto">
          {error && (
            <div className="rounded-md border border-err/40 bg-err/10 text-err text-2xs px-3 py-2">
              Failed to compute preview: {String((error as Error).message)}
            </div>
          )}

          {/* 0. Warnings banner (e.g. legacy v1 full-overwrite) */}
          {warnings.length > 0 && (
            <div className="rounded-md border border-warn/40 bg-warn/10 text-warn text-2xs px-3 py-2 space-y-1">
              {warnings.map((w, i) => (
                <div key={i}>{w}</div>
              ))}
            </div>
          )}

          {/* 1. Requirements */}
          {requirements.length > 0 && (
            <Section title="Requirements" count={requirements.length}>
              <div className="space-y-1.5">
                {requirements.map((r, i) => (
                  <div
                    key={i}
                    className={`rounded-md border px-2.5 py-1.5 flex items-baseline gap-2 ${STATUS_STYLE[r.status]}`}
                  >
                    <span className="font-mono text-2xs" aria-hidden="true">
                      {STATUS_GLYPH[r.status]}
                    </span>
                    <span className="text-2xs font-medium">{r.label}</span>
                    {r.detail && (
                      <span className="text-[0.65rem] opacity-80 truncate">{r.detail}</span>
                    )}
                  </div>
                ))}
              </div>
            </Section>
          )}

          {/* 2. What changes */}
          <Section title="What changes" count={changedEntries.length}>
            {changedEntries.length === 0 ? (
              <Empty>No changes — preset matches your current config for these keys.</Empty>
            ) : (
              <div className="space-y-1.5">
                {changedEntries.map((d) => (
                  <DiffRow key={d.key} entry={d} />
                ))}
              </div>
            )}
          </Section>

          {/* 3. What's preserved */}
          <Section title="Preserved from your config" count={preservedEntries.length}>
            {preservedEntries.length === 0 ? (
              <Empty>
                Nothing preserved — this preset didn't declare any{' '}
                <code className="font-mono text-ink-2">preserves</code> keys.
              </Empty>
            ) : (
              <div className="flex flex-wrap gap-1.5">
                {preservedEntries.map((d) => (
                  <span
                    key={d.key}
                    className="inline-flex items-center rounded border border-line-2 bg-surface-2 px-1.5 py-0.5 text-[0.65rem] font-mono text-ink-2"
                    title="Kept from your config — preset declared this key as preserved."
                  >
                    {d.key}
                  </span>
                ))}
              </div>
            )}
          </Section>

          {/* 4. Placeholders */}
          {placeholders.length > 0 && (
            <Section title="Fields you'll still need to fill" count={placeholders.length}>
              <div className="space-y-1.5">
                {placeholders.map((p) => (
                  <div
                    key={p.path}
                    className="rounded-md border border-warn/30 bg-warn/5 px-2.5 py-1.5 space-y-0.5"
                  >
                    <div className="font-mono text-2xs text-warn">{p.path}</div>
                    {p.hint && (
                      <div className="text-[0.65rem] text-ink-3">{p.hint}</div>
                    )}
                  </div>
                ))}
              </div>
            </Section>
          )}
        </div>

        {/* Footer */}
        <div className="px-4 py-3 border-t border-line-1 flex items-center justify-end gap-2">
          <button
            type="button"
            onClick={onCancel}
            className="btn-ghost h-8 text-xs flex items-center gap-1"
          >
            <span aria-hidden="true">←</span> Back
          </button>
          <button
            ref={applyBtnRef}
            type="button"
            onClick={onApply}
            disabled={isLoading}
            className="btn-primary h-8 text-xs px-3 disabled:opacity-60"
          >
            {dirty ? 'Overwrite & apply' : 'Apply preset'}
          </button>
        </div>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Local UI helpers — kept in the same file because they're only used here
// and are essentially styling lambdas.
// ---------------------------------------------------------------------------

function Section({
  title,
  count,
  children,
}: {
  title: string
  count: number
  children: React.ReactNode
}) {
  return (
    <div className="space-y-1.5">
      <div className="flex items-baseline gap-2">
        <span className="text-2xs font-semibold text-ink-1">{title}</span>
        <span className="text-[0.6rem] text-ink-4">{count}</span>
      </div>
      {children}
    </div>
  )
}

function Empty({ children }: { children: React.ReactNode }) {
  return (
    <div className="rounded-md border border-dashed border-line-2 px-3 py-2 text-2xs text-ink-3">
      {children}
    </div>
  )
}

function DiffRow({ entry }: { entry: PresetDiffEntry }) {
  const reason = REASON_STYLE[entry.reason]
  return (
    <div className="rounded-md border border-line-1 bg-surface-0 p-2">
      <div className="flex items-baseline gap-2">
        <span
          className={`inline-flex items-center rounded border px-1.5 py-0.5 text-[0.6rem] uppercase tracking-wide ${reason.cls}`}
        >
          {reason.label}
        </span>
        <span className="font-mono text-xs text-ink-1">{entry.key}</span>
      </div>
      <div className="mt-1 font-mono text-2xs space-y-0.5">
        {entry.before !== undefined && entry.before !== null && (
          <div className="text-ink-3">
            <span className="text-err/80">−</span> {formatVal(entry.before)}
          </div>
        )}
        {entry.after !== undefined && entry.after !== null && (
          <div className="text-ink-3">
            <span className="text-ok/80">+</span> {formatVal(entry.after)}
          </div>
        )}
      </div>
    </div>
  )
}
