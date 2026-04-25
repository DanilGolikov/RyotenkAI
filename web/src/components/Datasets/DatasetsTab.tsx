/**
 * Datasets project tab — master/detail layout.
 *
 * Sidebar lists datasets parsed out of the project's current YAML
 * config. The right pane shows the active dataset's source config
 * (paths / HF repo), a paginated preview, and an on-demand validation
 * panel. Routes are nested under ``/projects/:id/datasets`` —
 * ``:datasetKey`` selects which dataset is open.
 */

import { useMemo } from 'react'
import { Navigate, NavLink, Route, Routes, useNavigate, useParams } from 'react-router-dom'
import { useProjectConfig, useSaveProjectConfig } from '../../api/hooks/useProjects'
import { useDatasetsList } from '../../api/hooks/useDatasets'
import type { DatasetEntry } from '../../api/hooks/useDatasets'
import { dumpYaml, safeYamlParse } from '../../lib/yaml'
import { Spinner } from '../ui'
import { DatasetDetail } from './DatasetDetail'
import { DatasetList } from './DatasetList'

interface Props {
  projectId: string
}

export function DatasetsTab({ projectId }: Props) {
  const configQuery = useProjectConfig(projectId)
  const saveMut = useSaveProjectConfig(projectId)

  const parsed = useMemo<Record<string, unknown>>(() => {
    if (!configQuery.data?.yaml) return {}
    return safeYamlParse(configQuery.data.yaml) ?? {}
  }, [configQuery.data?.yaml])

  const datasets = useDatasetsList(parsed)

  const persist = async (next: Record<string, unknown>) => {
    await saveMut.mutateAsync(dumpYaml(next))
  }

  if (configQuery.isLoading) {
    return (
      <div className="flex items-center gap-2 text-xs text-ink-3">
        <Spinner /> loading datasets
      </div>
    )
  }
  if (configQuery.error) {
    return <div className="text-xs text-err">{(configQuery.error as Error).message}</div>
  }

  return (
    <div className="grid grid-cols-[260px_minmax(0,1fr)] gap-4 h-full min-h-0">
      <aside className="min-w-0 border-r border-line-1 pr-3 overflow-y-auto">
        <DatasetList projectId={projectId} datasets={datasets} />
      </aside>
      <div className="min-w-0 overflow-y-auto">
        <Routes>
          <Route
            index
            element={
              datasets.length === 0 ? (
                <EmptyState />
              ) : (
                <Navigate to={encodeURIComponent(datasets[0].key)} replace />
              )
            }
          />
          <Route
            path=":datasetKey"
            element={
              <DatasetDetailRoute
                projectId={projectId}
                parsed={parsed}
                datasets={datasets}
                persist={persist}
                saving={saveMut.isPending}
                saveError={(saveMut.error as Error | null) ?? null}
              />
            }
          />
        </Routes>
      </div>
    </div>
  )
}

function DatasetDetailRoute({
  projectId,
  parsed,
  datasets,
  persist,
  saving,
  saveError,
}: {
  projectId: string
  parsed: Record<string, unknown>
  datasets: DatasetEntry[]
  persist: (next: Record<string, unknown>) => Promise<void>
  saving: boolean
  saveError: Error | null
}) {
  const { datasetKey } = useParams<{ datasetKey: string }>()
  const navigate = useNavigate()
  const decodedKey = datasetKey ? decodeURIComponent(datasetKey) : ''
  const entry = datasets.find((d) => d.key === decodedKey)

  if (!entry) {
    return (
      <div className="space-y-3">
        <div className="text-sm text-err">Dataset not found: <span className="font-mono">{decodedKey}</span></div>
        <NavLink to=".." className="text-2xs text-ink-3 hover:text-ink-1 underline">
          Back to list
        </NavLink>
      </div>
    )
  }

  return (
    <DatasetDetail
      projectId={projectId}
      parsed={parsed}
      entry={entry}
      persist={persist}
      saving={saving}
      saveError={saveError}
      onDeleted={() => navigate('..', { replace: true })}
    />
  )
}

function EmptyState() {
  return (
    <div className="rounded-md border border-dashed border-line-1 bg-surface-inset px-4 py-8 text-center text-xs text-ink-3 max-w-lg mx-auto mt-12">
      <div className="text-sm text-ink-1 font-medium mb-1">No datasets yet</div>
      <p className="leading-snug">
        Datasets show up here automatically when you add a training strategy in
        the <span className="text-ink-2">Config</span> tab. You can also add one
        manually using the <span className="text-ink-2">+ Add</span> button on
        the left.
      </p>
    </div>
  )
}
