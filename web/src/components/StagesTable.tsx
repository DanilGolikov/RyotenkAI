import type { StageRun } from '../api/types'
import { StatusBadge } from './StatusBadge'
import { formatDuration } from '../lib/format'

export function StagesTable({ stages }: { stages: StageRun[] }) {
  if (!stages.length) return <div className="text-sm text-gray-500 py-4">no stages</div>
  return (
    <table className="w-full text-sm">
      <thead className="text-left text-xs uppercase tracking-wider text-gray-500 border-b border-surface-muted">
        <tr>
          <th className="px-4 py-2">Status</th>
          <th className="px-4 py-2">Stage</th>
          <th className="px-4 py-2">Mode</th>
          <th className="px-4 py-2">Duration</th>
          <th className="px-4 py-2">Error</th>
        </tr>
      </thead>
      <tbody>
        {stages.map((stage) => (
          <tr key={stage.stage_name} className="border-b border-surface-muted/50">
            <td className="px-4 py-2"><StatusBadge status={stage.status} /></td>
            <td className="px-4 py-2">{stage.stage_name}</td>
            <td className="px-4 py-2 text-gray-400">{stage.mode_label || '—'}</td>
            <td className="px-4 py-2 text-gray-400">{formatDuration(stage.duration_seconds)}</td>
            <td className="px-4 py-2 text-rose-400 truncate max-w-xl">{stage.error || ''}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}
