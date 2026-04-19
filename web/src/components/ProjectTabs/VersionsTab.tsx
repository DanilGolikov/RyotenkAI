import { useState } from 'react'
import {
  useProjectConfigVersions,
  useReadConfigVersion,
  useRestoreConfigVersion,
} from '../../api/hooks/useProjects'
import { Spinner } from '../ui'

export function VersionsTab({ projectId }: { projectId: string }) {
  const { data, isLoading, error } = useProjectConfigVersions(projectId)
  const [selected, setSelected] = useState<string | null>(null)
  const restoreMut = useRestoreConfigVersion(projectId)
  const detailQuery = useReadConfigVersion(projectId, selected)

  if (isLoading) {
    return (
      <div className="flex items-center gap-2 text-sm text-ink-3">
        <Spinner /> loading versions
      </div>
    )
  }
  if (error) return <div className="text-sm text-err">{(error as Error).message}</div>

  const versions = data?.versions ?? []

  if (versions.length === 0) {
    return (
      <div className="text-xs text-ink-3">
        No snapshots yet. Each config save creates one automatically.
      </div>
    )
  }

  return (
    <div className="grid grid-cols-[260px_1fr] gap-4">
      <div className="space-y-1 max-h-[480px] overflow-y-auto pr-2">
        {versions.map((v) => {
          const active = selected === v.filename
          return (
            <button
              key={v.filename}
              onClick={() => setSelected(v.filename)}
              className={[
                'w-full text-left rounded-md px-3 py-2 text-xs border transition',
                active
                  ? 'border-brand bg-surface-2 text-ink-1'
                  : 'border-line-1 hover:border-line-2 text-ink-2',
              ].join(' ')}
            >
              <div className="font-mono text-ink-1 text-2xs truncate">{v.filename}</div>
              <div className="text-ink-3 text-[0.65rem] mt-0.5">
                {v.created_at} · {v.size_bytes} bytes
              </div>
            </button>
          )
        })}
      </div>

      <div className="min-w-0">
        {!selected ? (
          <div className="text-xs text-ink-3">Select a snapshot to preview.</div>
        ) : detailQuery.isLoading ? (
          <div className="flex items-center gap-2 text-sm text-ink-3">
            <Spinner /> loading…
          </div>
        ) : detailQuery.error ? (
          <div className="text-sm text-err">{(detailQuery.error as Error).message}</div>
        ) : (
          <div className="space-y-2">
            <pre className="bg-surface-0 border border-line-1 rounded-md p-3 text-xs font-mono text-ink-1 overflow-auto max-h-[460px]">
              {detailQuery.data?.yaml}
            </pre>
            <div className="flex items-center justify-end gap-2">
              {restoreMut.error && (
                <span className="text-err text-2xs">
                  {(restoreMut.error as Error).message}
                </span>
              )}
              <button
                type="button"
                disabled={restoreMut.isPending}
                onClick={() => restoreMut.mutate(selected)}
                className="btn-primary px-3 py-1.5 text-xs"
              >
                {restoreMut.isPending ? 'Restoring…' : 'Restore as current'}
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
