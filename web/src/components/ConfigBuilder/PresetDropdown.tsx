import { useCallback, useEffect, useRef, useState } from 'react'
import { useConfigPresets } from '../../api/hooks/useConfigPresets'
import type { ConfigPreset } from '../../api/types'
import { useClickOutside } from '../../hooks/useClickOutside'
import { PresetPreviewModal } from './PresetPreviewModal'

interface Props {
  dirty: boolean
  onLoad: (preset: ConfigPreset) => void
  /** Current form value — needed by the preview modal to compute the
   *  exact list of fields that will change if the preset is applied. */
  current: Record<string, unknown>
  /** Force-close signal. Parent bumps this (e.g. on Form↔YAML view
   *  switch) to make the dropdown dismiss itself — the `mousedown`
   *  outside-listener doesn't fire on view toggles because those happen
   *  in response to a button click inside the same page, not a mouse
   *  event outside the dropdown. */
  closeToken?: number
}

export function PresetDropdown({ dirty, onLoad, current, closeToken }: Props) {
  const { data, isLoading } = useConfigPresets()
  const [open, setOpen] = useState(false)
  const [pendingPreset, setPendingPreset] = useState<ConfigPreset | null>(null)
  const wrapperRef = useRef<HTMLDivElement | null>(null)

  const close = useCallback(() => setOpen(false), [])
  useClickOutside(wrapperRef, open, close)

  // Parent-driven close signal (view switch, preset applied, etc.).
  useEffect(() => {
    if (closeToken === undefined) return
    setOpen(false)
  }, [closeToken])

  const presets = data?.presets ?? []
  if (isLoading || presets.length === 0) return null

  // Clicking a preset opens the diff preview rather than applying
  // immediately. Replaces the old `window.confirm` dirty-guard with a
  // full list of what will change — so the "overwrite" decision is
  // informed, not blind.
  function handleClick(preset: ConfigPreset) {
    setPendingPreset(preset)
    setOpen(false)
  }

  function confirmApply() {
    if (pendingPreset) onLoad(pendingPreset)
    setPendingPreset(null)
  }

  return (
    <>
      <div ref={wrapperRef} className="relative">
        <button
          type="button"
          onClick={() => setOpen((v) => !v)}
          aria-expanded={open}
          className="rounded-md border border-line-1 px-3 py-1.5 text-2xs text-ink-2 hover:text-ink-1 hover:border-line-2"
        >
          Load preset ▾
        </button>
        {open && (
          <div
            role="menu"
            className="absolute right-0 z-40 mt-1 w-80 rounded-md border border-line-2 bg-surface-1 shadow-card overflow-hidden"
          >
            {presets.map((p) => (
              <button
                key={p.name}
                type="button"
                role="menuitem"
                onClick={() => handleClick(p)}
                className="w-full text-left px-3 py-2 hover:bg-surface-2 transition block"
              >
                <div className="text-xs text-ink-1 font-medium">
                  {p.display_name || p.name}
                </div>
                {p.description && (
                  <div className="text-[0.65rem] text-ink-3 line-clamp-2 mt-0.5">
                    {p.description}
                  </div>
                )}
              </button>
            ))}
          </div>
        )}
      </div>
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
