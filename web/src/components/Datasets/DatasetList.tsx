/**
 * Sidebar list of datasets in the current project. Each item shows
 * key, source-type chip and "auto" tag for entries created together
 * with a strategy. Active selection is driven by the URL —
 * NavLink-isActive does the work.
 */

import { NavLink } from 'react-router-dom'
import type { DatasetEntry } from '../../api/hooks/useDatasets'

interface Props {
  projectId: string
  datasets: DatasetEntry[]
}

export function DatasetList({ datasets }: Props) {
  return (
    <nav className="flex flex-col gap-0.5 sticky top-0 pt-1">
      <div className="text-2xs uppercase tracking-wide text-ink-3 px-2 mb-1 flex items-center justify-between">
        <span>Datasets</span>
        <span className="text-ink-4 normal-case tracking-normal">{datasets.length}</span>
      </div>
      {datasets.length === 0 && (
        <div className="text-2xs text-ink-4 px-2 py-2 italic">
          empty — add one via Config → Training
        </div>
      )}
      {datasets.map((d) => (
        <NavLink
          key={d.key}
          to={encodeURIComponent(d.key)}
          className={({ isActive }) =>
            [
              'group flex items-center gap-2 px-2 py-1.5 rounded-md text-xs text-left transition-colors min-w-0',
              isActive
                ? 'text-ink-1 font-medium bg-surface-3'
                : 'text-ink-2 hover:text-ink-1 hover:bg-surface-3/50',
            ].join(' ')
          }
        >
          <span className="font-mono truncate flex-1 min-w-0">{d.key}</span>
          <span
            className={[
              'pill',
              d.sourceType === 'huggingface' ? 'pill-info' : 'pill-idle',
            ].join(' ')}
            title={d.sourceType}
          >
            {d.sourceType === 'huggingface' ? 'HF' : 'local'}
          </span>
          {d.autoCreated && (
            <span className="pill pill-skip" title="Auto-created together with a strategy">
              auto
            </span>
          )}
        </NavLink>
      ))}
    </nav>
  )
}
