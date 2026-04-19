import { useState } from 'react'
import type { PluginKind } from '../../api/types'
import { PluginBrowser } from '../PluginBrowser'

const KINDS: { id: PluginKind; label: string }[] = [
  { id: 'reward', label: 'Reward' },
  { id: 'validation', label: 'Validation' },
  { id: 'evaluation', label: 'Evaluation' },
]

export function PluginsTab() {
  const [kind, setKind] = useState<PluginKind>('reward')
  return (
    <div className="space-y-4">
      <div className="flex gap-1">
        {KINDS.map((k) => (
          <button
            key={k.id}
            type="button"
            onClick={() => setKind(k.id)}
            className={[
              'text-2xs rounded-md px-3 py-1.5 border transition',
              kind === k.id
                ? 'border-brand text-ink-1 bg-surface-2'
                : 'border-line-1 text-ink-3 hover:border-line-2 hover:text-ink-1',
            ].join(' ')}
          >
            {k.label}
          </button>
        ))}
      </div>
      <PluginBrowser kind={kind} />
    </div>
  )
}
