import { useEffect, useRef } from 'react'
import { useClickOutside } from '../../hooks/useClickOutside'
import { usePresetPreview } from '../../api/hooks/usePresetPreview'
import { HelpTooltip } from './HelpTooltip'
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
  ok: 'bg-ok/10 text-ok border-ok/30',
  warning: 'bg-warn/10 text-warn border-warn/30',
  missing: 'bg-err/10 text-err border-err/30',
}

const STATUS_GLYPH: Record<PresetRequirementCheck['status'], string> = {
  ok: '✓',
  warning: '!',
  missing: '✗',
}

const STATUS_TITLE: Record<PresetRequirementCheck['status'], string> = {
  ok: 'OK — this requirement is satisfied.',
  warning:
    'Warning — preset will apply, but something is off (e.g. missing auth token or VRAM headroom). Check the detail.',
  missing:
    'Missing — preset relies on this and nothing matching is configured. Fix before Apply or expect failures at runtime.',
}

/** Reason → chip label + colour class + tooltip. Tooltips explain the
 *  backend's reason enum in plain English. */
const REASON_STYLE: Record<
  PresetDiffEntry['reason'],
  { label: string; cls: string; title: string }
> = {
  preset_replaced: {
    label: 'replaced',
    cls: 'bg-brand-alt/15 text-brand-alt border-brand-alt/30',
    title:
      'This key existed in your current config and the preset overwrites it.',
  },
  preset_added: {
    label: 'added',
    cls: 'bg-ok/15 text-ok border-ok/30',
    title:
      'New key — your config did not have this; the preset is introducing it.',
  },
  preset_preserved: {
    label: 'preserved',
    cls: 'bg-surface-2 text-ink-3 border-line-2',
    title:
      'Preset explicitly declared this key as preserved — your current value stays untouched.',
  },
  no_scope: {
    label: 'overwrite',
    cls: 'bg-warn/15 text-warn border-warn/30',
    title:
      'Legacy v1 preset — no scope declared, so the entire top-level block is overwritten. Review carefully.',
  },
}

const SECTION_HELP = {
  requirements:
    'Readiness checks the backend ran against your current environment. Green = satisfied; amber = works with caveats; red = preset depends on this and nothing matching is set up.',
  changes:
    'Keys this preset will write into your config. Existing values are overwritten; new keys are appended. Red strikethrough line is what you have now; green line is what the preset puts in its place.',
  preserved:
    'Keys the preset author declared as "preserves" — these stay exactly as you have them even though the preset nominally touches their block. Useful for keeping things like dataset paths or project-specific overrides.',
  placeholders:
    'Paths the preset intentionally leaves empty or marked as "TODO" — you need to fill them in manually after Apply. Typical examples: dataset JSONL path, HF model name, project-specific output directory.',
}

/** Pretty-print arbitrary JSON-ish values for the diff rows.
 *  - Strings: rendered as-is so multi-line strings stay readable.
 *  - Scalars: ``String(v)``.
 *  - Objects/arrays: ``JSON.stringify(v, null, 2)`` so nested structure
 *    is readable across multiple lines rather than one truncated blob.
 *  Capped via CSS ``max-height`` + scroll on the row, so a huge training
 *  block doesn't blow out the modal. */
function formatDiffValue(v: unknown): string {
  if (v === undefined || v === null) return '—'
  if (typeof v === 'string') return v
  if (typeof v === 'number' || typeof v === 'boolean') return String(v)
  try {
    return JSON.stringify(v, null, 2)
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
 * Every section title has a ``?`` tooltip explaining it in plain
 * English; every chip has a ``title=`` hover hint. A banner at the
 * top surfaces any warnings (e.g. v1 "full overwrite" legacy behaviour
 * when the preset has no scope).
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
            <div className="text-2xs text-ink-3 flex items-center gap-2 flex-wrap">
              {isLoading ? (
                <span>Computing diff…</span>
              ) : (
                <>
                  <span
                    title="Number of keys this preset will overwrite or add to your config."
                    className="cursor-help"
                  >
                    <span className="text-ink-1 font-medium">{changedEntries.length}</span>{' '}
                    change{changedEntries.length === 1 ? '' : 's'}
                  </span>
                  <span className="text-ink-4">·</span>
                  <span
                    title="Keys the preset explicitly keeps from your current config, even though its block nominally covers them."
                    className="cursor-help"
                  >
                    <span className="text-ink-1 font-medium">{preservedEntries.length}</span>{' '}
                    preserved
                  </span>
                  <span className="text-ink-4">·</span>
                  <span
                    title="Paths you'll still need to fill manually after Apply (e.g. dataset file)."
                    className="cursor-help"
                  >
                    <span className="text-ink-1 font-medium">{placeholders.length}</span> to fill
                  </span>
                </>
              )}
              {dirty && (
                <span
                  className="ml-2 text-brand-strong cursor-help"
                  title="You edited the form without saving — clicking Apply will overwrite those edits."
                >
                  You have unsaved changes that will be overwritten.
                </span>
              )}
            </div>
          </div>
          <button
            type="button"
            onClick={onCancel}
            className="text-ink-3 hover:text-ink-1 text-xs"
            title="Close the preview and return to the preset picker."
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
            <div
              className="rounded-md border border-warn/40 bg-warn/10 text-warn text-2xs px-3 py-2 space-y-1"
              title="Caveats the backend wants you to see before applying — usually legacy-compat quirks."
            >
              {warnings.map((w, i) => (
                <div key={i}>{w}</div>
              ))}
            </div>
          )}

          {/* 1. Requirements */}
          {requirements.length > 0 && (
            <Section title="Requirements" count={requirements.length} help={SECTION_HELP.requirements}>
              <div className="space-y-1.5">
                {requirements.map((r, i) => (
                  <div
                    key={i}
                    className={`rounded-md border px-2.5 py-1.5 flex items-baseline gap-2 ${STATUS_STYLE[r.status]}`}
                    title={STATUS_TITLE[r.status]}
                  >
                    <span className="font-mono text-2xs w-4 text-center" aria-hidden="true">
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
          <Section title="What changes" count={changedEntries.length} help={SECTION_HELP.changes}>
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
          <Section
            title="Preserved from your config"
            count={preservedEntries.length}
            help={SECTION_HELP.preserved}
          >
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
                    title={`'${d.key}' is kept from your config — preset declared this key as preserved.`}
                  >
                    {d.key}
                  </span>
                ))}
              </div>
            )}
          </Section>

          {/* 4. Placeholders */}
          {placeholders.length > 0 && (
            <Section
              title="Fields you'll still need to fill"
              count={placeholders.length}
              help={SECTION_HELP.placeholders}
            >
              <div className="space-y-1.5">
                {placeholders.map((p) => (
                  <div
                    key={p.path}
                    className="rounded-md border border-warn/30 bg-warn/5 px-2.5 py-1.5 space-y-0.5"
                    title={p.hint || 'Placeholder left empty by the preset — fill in before launching.'}
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
            title="Go back to the preset picker — nothing is applied."
          >
            <span aria-hidden="true">←</span> Back
          </button>
          <button
            ref={applyBtnRef}
            type="button"
            onClick={onApply}
            disabled={isLoading}
            className="btn-primary h-8 text-xs px-3 disabled:opacity-60"
            title={
              dirty
                ? 'Apply this preset — your unsaved edits in overlapping keys will be overwritten.'
                : 'Apply this preset to your config.'
            }
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
  help,
  children,
}: {
  title: string
  count: number
  help?: string
  children: React.ReactNode
}) {
  return (
    <div className="space-y-1.5">
      <div className="flex items-baseline gap-2">
        <span className="text-2xs font-semibold text-ink-1">{title}</span>
        <span className="text-[0.6rem] text-ink-4">{count}</span>
        {help && <HelpTooltip text={help} label={`${title} help`} />}
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

/** Single row in "What changes" — restores the old github-style coloured
 *  diff (red-tinted ``−`` line + green-tinted ``+`` line) with the new
 *  reason chip (``REPLACED`` / ``ADDED`` / ``OVERWRITE``) layered on top.
 *  Complex values are pretty-printed as multi-line ``<pre>`` blocks so
 *  the user can actually read nested objects instead of a truncated
 *  one-liner. */
function DiffRow({ entry }: { entry: PresetDiffEntry }) {
  const reason = REASON_STYLE[entry.reason]
  const hasBefore = entry.before !== undefined && entry.before !== null
  const hasAfter = entry.after !== undefined && entry.after !== null
  return (
    <div className="rounded-md border border-line-1 bg-surface-0 overflow-hidden">
      <div className="flex items-baseline gap-2 px-2 py-1.5 border-b border-line-1 bg-surface-1">
        <span
          className={`inline-flex items-center rounded border px-1.5 py-0.5 text-[0.6rem] uppercase tracking-wide cursor-help ${reason.cls}`}
          title={reason.title}
        >
          {reason.label}
        </span>
        <span className="font-mono text-xs text-ink-1" title={`Config path: ${entry.key}`}>
          {entry.key}
        </span>
      </div>
      <div className="font-mono text-2xs">
        {hasBefore && (
          <div
            className="flex gap-1.5 px-2 py-1 bg-err/10 border-l-2 border-err/60 text-err"
            title="Your current value — will be removed."
          >
            <span className="select-none opacity-60">−</span>
            <pre className="flex-1 whitespace-pre-wrap break-all leading-snug max-h-32 overflow-y-auto m-0">
              {formatDiffValue(entry.before)}
            </pre>
          </div>
        )}
        {hasAfter && (
          <div
            className="flex gap-1.5 px-2 py-1 bg-ok/10 border-l-2 border-ok/60 text-ok"
            title="Value the preset will write in place."
          >
            <span className="select-none opacity-60">+</span>
            <pre className="flex-1 whitespace-pre-wrap break-all leading-snug max-h-32 overflow-y-auto m-0">
              {formatDiffValue(entry.after)}
            </pre>
          </div>
        )}
        {!hasBefore && !hasAfter && (
          <div className="px-2 py-1 text-ink-4">(empty value)</div>
        )}
      </div>
    </div>
  )
}
