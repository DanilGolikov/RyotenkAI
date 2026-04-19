import { useMemo, useState } from 'react'
import { useRuns } from '../api/hooks/useRuns'
import { RunsTable } from '../components/RunsTable'

export function RunsList() {
  const { data, isLoading, error } = useRuns()
  const [filter, setFilter] = useState('')

  const filteredGroups = useMemo(() => {
    if (!data) return {}
    if (!filter.trim()) return data.groups
    const query = filter.toLowerCase()
    return Object.fromEntries(
      Object.entries(data.groups).map(([group, rows]) => [
        group,
        rows.filter((r) => r.run_id.toLowerCase().includes(query) || (r.config_name ?? '').toLowerCase().includes(query)),
      ]),
    )
  }, [data, filter])

  return (
    <div className="p-6 space-y-4">
      <div className="flex items-center justify-between">
        <h1 className="text-xl">Runs</h1>
        <input
          value={filter}
          onChange={(event) => setFilter(event.target.value)}
          placeholder="filter by run id or config..."
          className="bg-surface-raised border border-surface-muted rounded px-3 py-1.5 text-sm w-72"
        />
      </div>

      {error && <div className="text-rose-400 text-sm">{(error as Error).message}</div>}
      {isLoading && <div className="text-gray-400 text-sm">loading…</div>}

      {Object.entries(filteredGroups).map(([group, rows]) => (
        <section key={group} className="bg-surface-raised rounded border border-surface-muted">
          <h2 className="px-4 py-2 border-b border-surface-muted text-sm text-gray-400">{group}</h2>
          <RunsTable rows={rows} />
        </section>
      ))}

      {!isLoading && Object.keys(filteredGroups).length === 0 && (
        <div className="text-gray-500 text-sm">no runs found — launch one via `ryotenkai train` or create via API</div>
      )}
    </div>
  )
}
