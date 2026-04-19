import { Link } from 'react-router-dom'
import type { RunSummary } from '../api/types'
import { StatusBadge } from './StatusBadge'
import { formatDuration } from '../lib/format'

export function RunsTable({ rows }: { rows: RunSummary[] }) {
  if (rows.length === 0) {
    return <div className="text-sm text-gray-500 px-4 py-6">no runs yet</div>
  }
  return (
    <table className="w-full text-sm">
      <thead className="text-left text-xs uppercase tracking-wider text-gray-500 border-b border-surface-muted">
        <tr>
          <th className="px-4 py-2">Status</th>
          <th className="px-4 py-2">Run</th>
          <th className="px-4 py-2">Config</th>
          <th className="px-4 py-2">Attempts</th>
          <th className="px-4 py-2">Duration</th>
          <th className="px-4 py-2">Created</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((run) => (
          <tr key={run.run_id} className="border-b border-surface-muted/50 hover:bg-surface-raised">
            <td className="px-4 py-2"><StatusBadge status={run.status} /></td>
            <td className="px-4 py-2 text-accent">
              <Link to={`/runs/${encodeURIComponent(run.run_id)}`}>{run.run_id}</Link>
            </td>
            <td className="px-4 py-2 text-gray-400">{run.config_name}</td>
            <td className="px-4 py-2 text-gray-400">{run.attempts}</td>
            <td className="px-4 py-2 text-gray-400">{formatDuration(run.duration_seconds)}</td>
            <td className="px-4 py-2 text-gray-400">{run.created_at}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}
