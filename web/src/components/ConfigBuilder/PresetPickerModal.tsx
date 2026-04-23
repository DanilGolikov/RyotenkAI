import { useEffect, useMemo, useRef, useState } from 'react'
import { useConfigPresets } from '../../api/hooks/useConfigPresets'
import type { ConfigPreset } from '../../api/types'
import { PresetPreviewModal } from './PresetPreviewModal'

interface Props {
  dirty: boolean
  onLoad: (preset: ConfigPreset) => void
  /** Current form value — passed through to the preview modal so it can
   *  compute the exact list of fields that will change if the preset is
   *  applied. */
  current: Record<string, unknown>
  /** Force-close signal. Parent bumps this on view toggles (Form↔YAML
   *  switch) so the open modal dismisses — mouse-outside events don't
   *  fire for same-page toggles. */
  closeToken?: number
}

type PresetRow = ConfigPreset & {
  /** Pre-computed lowercase blob for fast substring match. */
  _haystack: string
}

/** Tier → short human chip shown inside the card header.
 *  Classes carry both the text and the colour bucket so large presets
 *  visually stand out from small ones without reading the label. */
const TIER_STYLE: Record<string, { label: string; cls: string }> = {
  small: { label: 'small', cls: 'bg-ok/15 text-ok border-ok/30' },
  medium: { label: 'medium', cls: 'bg-warn/15 text-warn border-warn/30' },
  large: { label: 'large', cls: 'bg-err/15 text-err border-err/30' },
}

// Chip palette — each requirement category gets its own semantic colour
// bucket so you can read a card at a glance without parsing any icons.
// Tokens come from the existing Tailwind theme (brand, ok, warn, err,
// ink, surface) — no new colours added.
const CHIP_BASE = 'inline-flex items-center rounded px-1.5 py-0.5 text-[0.6rem]'
// VRAM — most important requirement, brand accent.
const CHIP_VRAM = `${CHIP_BASE} bg-brand-alt/10 text-brand-alt border border-brand-alt/20`
// Hub model — mono font on a dim tile, neutral semantics (identifier).
const CHIP_MODEL = `${CHIP_BASE} bg-surface-2 font-mono text-ink-2`
// Provider kind — info-blue accent (separates at-a-glance from VRAM and models).
const CHIP_PROVIDER = `${CHIP_BASE} bg-brand/10 text-brand border border-brand/20`
// Required plugin — warn accent, signals "needs extra setup".
const CHIP_PLUGIN = `${CHIP_BASE} bg-warn/10 text-warn border border-warn/20 font-mono`
// Scope counters — neutral but slightly lighter than plain chips.
const CHIP_SCOPE = `${CHIP_BASE} bg-surface-2 text-ink-3`
// Placeholders counter — brighter warn to stand out: "you'll still edit N fields".
const CHIP_PLACEHOLDERS = `${CHIP_BASE} bg-warn/15 text-warn border border-warn/30`

function buildHaystack(p: ConfigPreset): string {
  const parts: string[] = [
    p.name,
    p.display_name ?? '',
    p.description ?? '',
    p.size_tier ?? '',
  ]
  const req = p.requirements
  if (req) {
    parts.push(...(req.hub_models ?? []))
    parts.push(...(req.provider_kind ?? []))
    parts.push(...(req.required_plugins ?? []))
  }
  return parts.filter(Boolean).join(' ').toLowerCase()
}

function matchScore(row: PresetRow, q: string): number {
  if (!q) return 1
  const needle = q.toLowerCase().trim()
  if (!needle) return 1
  if (!row._haystack.includes(needle)) return 0
  // Prefer matches on id / display_name over description.
  const name = (row.display_name || row.name).toLowerCase()
  const nameHit = name.includes(needle) ? 3 : 0
  const idHit = row.name.toLowerCase().includes(needle) ? 2 : 0
  const tierHit = (row.size_tier ?? '').toLowerCase().includes(needle) ? 1 : 0
  return nameHit + idHit + tierHit + 1
}

export function PresetPickerModal({ dirty, onLoad, current, closeToken }: Props) {
  const { data, isLoading } = useConfigPresets()
  const [open, setOpen] = useState(false)
  const [query, setQuery] = useState('')
  const [cursor, setCursor] = useState(0)
  const [pendingPreset, setPendingPreset] = useState<ConfigPreset | null>(null)
  const inputRef = useRef<HTMLInputElement | null>(null)
  const listRef = useRef<HTMLDivElement | null>(null)

  const rows: PresetRow[] = useMemo(() => {
    const presets = data?.presets ?? []
    return presets.map((p) => ({ ...p, _haystack: buildHaystack(p) }))
  }, [data])

  const results = useMemo(() => {
    return rows
      .map((r) => ({ row: r, score: matchScore(r, query) }))
      .filter((x) => x.score > 0)
      .sort((a, b) => b.score - a.score)
      .map((x) => x.row)
  }, [rows, query])

  // Auto-focus search input and reset state on open.
  useEffect(() => {
    if (open) {
      setQuery('')
      setCursor(0)
      setTimeout(() => inputRef.current?.focus(), 0)
    }
  }, [open])

  // Keep cursor inside current result range as the user types.
  useEffect(() => setCursor((c) => Math.min(c, Math.max(0, results.length - 1))), [results.length])

  // Parent-driven close signal.
  useEffect(() => {
    if (closeToken === undefined) return
    setOpen(false)
  }, [closeToken])

  // Esc closes the picker, but only if the preview isn't open on top —
  // otherwise pressing Esc to close the preview would also dismiss the
  // picker underneath and break the "Back" contract.
  useEffect(() => {
    if (!open) return
    function onKey(e: KeyboardEvent) {
      if (e.key === 'Escape' && !pendingPreset) setOpen(false)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [open, pendingPreset])

  // Scroll active card into view on cursor move.
  useEffect(() => {
    const el = listRef.current?.querySelector<HTMLElement>(`[data-idx="${cursor}"]`)
    el?.scrollIntoView({ block: 'nearest' })
  }, [cursor])

  const presets = data?.presets ?? []
  if (isLoading || presets.length === 0) return null

  // Keep the picker open behind the preview so ``Back`` in the preview
  // reveals the list again (same-page "sheet-over-sheet" pattern). The
  // preview sits on top via z-index stacking — clicks on its backdrop
  // dismiss only the preview, not the picker.
  function handlePick(preset: ConfigPreset) {
    setPendingPreset(preset)
  }

  function confirmApply() {
    if (pendingPreset) onLoad(pendingPreset)
    setPendingPreset(null)
    setOpen(false)
  }

  return (
    <>
      <button
        type="button"
        onClick={() => setOpen(true)}
        aria-haspopup="dialog"
        className="rounded-md border border-line-1 px-3 py-1.5 text-2xs text-ink-2 hover:text-ink-1 hover:border-line-2"
      >
        Load preset
      </button>

      {open && (
        <div
          role="dialog"
          aria-modal="true"
          aria-label="Pick a preset"
          className="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm flex items-start justify-center pt-24"
          onClick={() => {
            // When the preview is open on top, backdrop clicks belong to
            // it (handled by its own useClickOutside) — don't also close
            // the picker underneath, or the user has nothing to return to.
            if (pendingPreset) return
            setOpen(false)
          }}
        >
          <div
            onClick={(e) => e.stopPropagation()}
            className="w-full max-w-2xl rounded-xl border border-line-2 bg-surface-1 shadow-card overflow-hidden"
          >
            {/* Search bar */}
            <div className="px-4 py-3 border-b border-line-1 flex items-center gap-3">
              <span className="text-ink-3 text-xs font-mono">🔍</span>
              <input
                ref={inputRef}
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'ArrowDown') {
                    e.preventDefault()
                    setCursor((c) => Math.min(c + 1, results.length - 1))
                  } else if (e.key === 'ArrowUp') {
                    e.preventDefault()
                    setCursor((c) => Math.max(c - 1, 0))
                  } else if (e.key === 'Enter' && results[cursor]) {
                    e.preventDefault()
                    handlePick(results[cursor])
                  }
                }}
                placeholder="Search presets by name, size, model, strategy…"
                className="flex-1 bg-transparent text-sm focus:outline-none placeholder:text-ink-4"
              />
              <span className="text-[0.6rem] text-ink-4">
                {results.length} / {rows.length}
              </span>
              <button
                type="button"
                onClick={() => setOpen(false)}
                className="text-ink-3 hover:text-ink-1 text-2xs"
              >
                esc
              </button>
            </div>

            {/* Results */}
            <div ref={listRef} className="max-h-[480px] overflow-y-auto py-1">
              {results.length === 0 ? (
                <div className="px-4 py-10 text-center text-2xs text-ink-3">
                  No presets match <span className="font-mono text-ink-2">{query}</span>.
                </div>
              ) : (
                results.map((p, idx) => {
                  const tier = p.size_tier ? TIER_STYLE[p.size_tier] : null
                  const req = p.requirements
                  const placeholderCount = Object.keys(p.placeholders ?? {}).length
                  const replacesN = p.scope?.replaces?.length ?? 0
                  const preservesN = p.scope?.preserves?.length ?? 0
                  return (
                    <button
                      key={p.name}
                      type="button"
                      data-idx={idx}
                      onMouseEnter={() => setCursor(idx)}
                      onClick={() => handlePick(p)}
                      className={[
                        'w-full text-left px-4 py-3 flex flex-col gap-1.5 transition',
                        idx === cursor ? 'bg-surface-2 text-ink-1' : 'text-ink-2 hover:bg-surface-2/60',
                      ].join(' ')}
                    >
                      {/* Header: name · id · size-tier (colour-coded) */}
                      <div className="flex items-center gap-2">
                        <span className="text-xs font-medium text-ink-1">
                          {p.display_name || p.name}
                        </span>
                        <span className="text-[0.6rem] font-mono text-ink-4">{p.name}</span>
                        {tier && (
                          <span
                            className={`ml-auto rounded border px-1.5 py-0.5 text-[0.6rem] font-medium uppercase tracking-wide ${tier.cls}`}
                          >
                            {tier.label}
                          </span>
                        )}
                      </div>

                      {p.description && (
                        <div className="text-[0.65rem] text-ink-3 line-clamp-2">
                          {p.description}
                        </div>
                      )}

                      {/* Requirements row — VRAM + HF models + provider + plugins */}
                      {req && (
                        <div className="flex flex-wrap gap-1.5">
                          {req.min_vram_gb != null && (
                            <span className={CHIP_VRAM} title="Minimum GPU VRAM">
                              ≥{req.min_vram_gb} GB VRAM
                            </span>
                          )}
                          {(req.hub_models ?? []).map((m) => (
                            <span
                              key={`hub-${m}`}
                              className={CHIP_MODEL}
                              title="Hugging Face Hub model"
                            >
                              {m}
                            </span>
                          ))}
                          {(req.provider_kind ?? []).map((k) => (
                            <span
                              key={`prov-${k}`}
                              className={CHIP_PROVIDER}
                              title="Compatible provider kind"
                            >
                              {k}
                            </span>
                          ))}
                          {(req.required_plugins ?? []).map((pl) => (
                            <span
                              key={`pl-${pl}`}
                              className={CHIP_PLUGIN}
                              title="Required community plugin"
                            >
                              {pl}
                            </span>
                          ))}
                        </div>
                      )}

                      {/* Scope + placeholders summary */}
                      {(replacesN > 0 || preservesN > 0 || placeholderCount > 0) && (
                        <div className="flex flex-wrap gap-1.5">
                          {replacesN > 0 && (
                            <span
                              className={CHIP_SCOPE}
                              title={`Keys replaced by this preset: ${p.scope?.replaces?.join(', ')}`}
                            >
                              replaces {replacesN}
                            </span>
                          )}
                          {preservesN > 0 && (
                            <span
                              className={CHIP_SCOPE}
                              title={`Keys preserved from your config: ${p.scope?.preserves?.join(', ')}`}
                            >
                              preserves {preservesN}
                            </span>
                          )}
                          {placeholderCount > 0 && (
                            <span
                              className={CHIP_PLACEHOLDERS}
                              title="Fields you'll need to fill in after applying"
                            >
                              {placeholderCount} to fill
                            </span>
                          )}
                        </div>
                      )}
                    </button>
                  )
                })
              )}
            </div>

            {/* Footer */}
            <div className="px-4 py-2 border-t border-line-1 text-[0.6rem] text-ink-4 flex gap-4">
              <span>↑↓ navigate</span>
              <span>⏎ preview</span>
              <span>esc close</span>
            </div>
          </div>
        </div>
      )}

      {pendingPreset && (
        <PresetPreviewModal
          preset={pendingPreset}
          current={current}
          dirty={dirty}
          onCancel={() => setPendingPreset(null)}
          onApply={confirmApply}
        />
      )}
    </>
  )
}
