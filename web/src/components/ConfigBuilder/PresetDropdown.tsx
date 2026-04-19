import { useState } from 'react'
import { useConfigPresets } from '../../api/hooks/useConfigPresets'
import type { ConfigPreset } from '../../api/types'

interface Props {
  dirty: boolean
  onLoad: (preset: ConfigPreset) => void
}

export function PresetDropdown({ dirty, onLoad }: Props) {
  const { data, isLoading } = useConfigPresets()
  const [open, setOpen] = useState(false)

  const presets = data?.presets ?? []
  if (isLoading || presets.length === 0) return null

  function handleClick(preset: ConfigPreset) {
    if (dirty) {
      const ok = window.confirm(
        `You have unsaved changes. Replace the current config with preset "${preset.name}"?`,
      )
      if (!ok) return
    }
    onLoad(preset)
    setOpen(false)
  }

  return (
    <div className="relative">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="rounded-md border border-line-1 px-3 py-1.5 text-2xs text-ink-2 hover:text-ink-1 hover:border-line-2"
      >
        Load preset ▾
      </button>
      {open && (
        <div
          onMouseLeave={() => setOpen(false)}
          className="absolute right-0 z-10 mt-1 w-80 rounded-md border border-line-2 bg-surface-1 shadow-card overflow-hidden"
        >
          {presets.map((p) => (
            <button
              key={p.name}
              type="button"
              onClick={() => handleClick(p)}
              className="w-full text-left px-3 py-2 hover:bg-surface-2 transition block"
            >
              <div className="text-xs text-ink-1 font-mono">{p.name}</div>
              {p.description && (
                <div className="text-[0.65rem] text-ink-3 line-clamp-2 mt-0.5">{p.description}</div>
              )}
            </button>
          ))}
        </div>
      )}
    </div>
  )
}
