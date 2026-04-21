import { useState } from 'react'
import type { UseMutationResult, UseQueryResult } from '@tanstack/react-query'
import type {
  ConfigVersionDetail,
  ConfigVersionsResponse,
  SaveConfigResponse,
} from '../api/types'
import { YamlView } from './YamlView'
import { Spinner } from './ui'

interface FavoriteHandler {
  onToggle: (filename: string, nextFavorite: boolean) => void
  pending: boolean
}

interface Props {
  versionsQuery: UseQueryResult<ConfigVersionsResponse>
  useReadVersion: (filename: string | null) => UseQueryResult<ConfigVersionDetail>
  restoreMutation: UseMutationResult<SaveConfigResponse, unknown, string>
  favorite?: FavoriteHandler
  emptyHint?: string
}

export function ConfigVersionsPanel({
  versionsQuery,
  useReadVersion,
  restoreMutation,
  favorite,
  emptyHint = 'No snapshots yet. Each save creates one automatically.',
}: Props) {
  const [selected, setSelected] = useState<string | null>(null)
  const detailQuery = useReadVersion(selected)

  if (versionsQuery.isLoading) {
    return (
      <div className="flex items-center gap-2 text-sm text-ink-3">
        <Spinner /> loading versions
      </div>
    )
  }
  if (versionsQuery.error) {
    return <div className="text-sm text-err">{(versionsQuery.error as Error).message}</div>
  }

  const versions = versionsQuery.data?.versions ?? []
  if (versions.length === 0) return <div className="text-xs text-ink-3">{emptyHint}</div>

  return (
    <div className="grid grid-cols-[280px_1fr] gap-4">
      <div className="space-y-1 max-h-[520px] overflow-y-auto pr-2">
        {versions.map((v) => {
          const active = selected === v.filename
          const fav = !!v.is_favorite
          return (
            <div
              key={v.filename}
              className={[
                // Hover lights the border in the active-state colour
                // WITHOUT filling the body (no `bg-surface-2` on hover).
                // This gives the "target outlined" affordance the user
                // expects without flashing a dark block behind the text,
                // which previously looked like a selection glitch.
                'rounded-md px-3 py-2 text-xs border transition-colors flex items-start gap-2',
                active
                  ? 'border-brand bg-surface-2 text-ink-1'
                  : fav
                  ? 'border-warn/60 hover:border-warn text-ink-1'
                  : 'border-line-1 hover:border-brand/60 text-ink-2 hover:text-ink-1',
              ].join(' ')}
            >
              {favorite && (
                <button
                  type="button"
                  disabled={favorite.pending}
                  onClick={(e) => {
                    e.stopPropagation()
                    favorite.onToggle(v.filename, !fav)
                  }}
                  title={fav ? 'Unpin from favorites' : 'Pin as favorite'}
                  className={[
                    'shrink-0 text-base leading-none transition',
                    fav ? 'text-warn' : 'text-ink-4 hover:text-warn',
                  ].join(' ')}
                >
                  {fav ? '★' : '☆'}
                </button>
              )}
              <button
                type="button"
                onClick={() => setSelected(v.filename)}
                className="flex-1 min-w-0 text-left"
              >
                <div className="font-mono text-ink-1 text-2xs truncate">{v.filename}</div>
                <div className="text-ink-3 text-[0.65rem] mt-0.5">
                  {v.created_at} · {v.size_bytes} bytes
                </div>
              </button>
            </div>
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
            <YamlView text={detailQuery.data?.yaml ?? ''} maxHeight="max-h-[460px]" />
            <div className="flex items-center justify-end gap-2">
              {restoreMutation.error ? (
                <span className="text-err text-2xs">
                  {(restoreMutation.error as Error).message}
                </span>
              ) : null}
              <button
                type="button"
                disabled={restoreMutation.isPending}
                onClick={() => restoreMutation.mutate(selected)}
                className="btn-primary px-3 py-1.5 text-xs"
              >
                {restoreMutation.isPending ? 'Restoring…' : 'Restore as current'}
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
