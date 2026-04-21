import { useEffect, useState } from 'react'
import { NavLink, Navigate, Route, Routes, useParams } from 'react-router-dom'
import {
  useProject,
  useUpdateProjectDescription,
} from '../api/hooks/useProjects'
import type { ProjectDetail } from '../api/types'
import { ConfigTab } from '../components/ProjectTabs/ConfigTab'
import { PluginsTab } from '../components/ProjectTabs/PluginsTab'
import { RunsTab } from '../components/ProjectTabs/RunsTab'
import { SettingsTab } from '../components/ProjectTabs/SettingsTab'
import { VersionsTab } from '../components/ProjectTabs/VersionsTab'
import { Card, Spinner } from '../components/ui'

const TABS: { to: string; label: string }[] = [
  { to: 'info', label: 'Info' },
  { to: 'config', label: 'Config' },
  { to: 'versions', label: 'Versions' },
  { to: 'runs', label: 'Runs' },
  { to: 'plugins', label: 'Plugins' },
  { to: 'settings', label: 'Settings' },
]

function InfoTab({ project }: { project: ProjectDetail }) {
  return (
    <dl className="space-y-3 text-xs">
      <div className="flex gap-4">
        <dt className="w-28 text-ink-3 shrink-0">Name</dt>
        <dd className="text-ink-1 break-all">{project.name}</dd>
      </div>
      <div className="flex gap-4">
        <dt className="w-28 text-ink-3 shrink-0">ID</dt>
        <dd className="text-ink-2 font-mono break-all">{project.id}</dd>
      </div>
      <div className="flex gap-4">
        <dt className="w-28 text-ink-3 shrink-0 pt-1">Description</dt>
        <dd className="flex-1 min-w-0">
          <DescriptionEditor project={project} />
        </dd>
      </div>
      <div className="flex gap-4">
        <dt className="w-28 text-ink-3 shrink-0">Workspace</dt>
        <dd className="text-ink-4 font-mono break-all">{project.path}</dd>
      </div>
    </dl>
  )
}

function DescriptionEditor({ project }: { project: ProjectDetail }) {
  const mut = useUpdateProjectDescription(project.id)
  const [draft, setDraft] = useState(project.description)
  const [focused, setFocused] = useState(false)

  // Sync draft with server on remount or background refetch — but leave
  // the user's in-flight typing alone while the field is focused.
  useEffect(() => {
    if (!focused) setDraft(project.description)
  }, [project.description, focused])

  const dirty = draft !== project.description

  async function save() {
    try {
      await mut.mutateAsync(draft)
    } catch (exc) {
      window.alert((exc as Error).message || 'Failed to save description.')
    }
  }

  return (
    <div className="space-y-2">
      <textarea
        value={draft}
        onChange={(e) => setDraft(e.target.value)}
        onFocus={() => setFocused(true)}
        onBlur={() => setFocused(false)}
        rows={3}
        placeholder="Short summary of the experiment — goal, dataset, model."
        className="w-full rounded bg-surface-1 border border-line-1 hover:border-line-2 focus:border-brand focus:outline-none px-2.5 py-2 text-xs text-ink-1 placeholder:text-ink-4 resize-y transition-colors"
      />
      <div className="flex items-center gap-2">
        <button
          type="button"
          disabled={!dirty || mut.isPending}
          onClick={save}
          className="btn-primary h-7 px-3 text-2xs disabled:opacity-40 disabled:cursor-not-allowed"
        >
          {mut.isPending ? 'Saving…' : 'Save'}
        </button>
        {mut.error ? (
          <span className="text-err text-2xs">
            {(mut.error as Error).message}
          </span>
        ) : mut.isSuccess && !dirty ? (
          <span className="text-ok/80 text-2xs">Saved</span>
        ) : null}
      </div>
    </div>
  )
}

export function ProjectDetailPage() {
  const { id } = useParams<{ id: string }>()
  const { data: project, isLoading, error } = useProject(id)

  if (!id) return <Navigate to="/projects" replace />

  if (isLoading) {
    return (
      <div className="p-6 text-sm text-ink-3 flex items-center gap-2">
        <Spinner /> loading project
      </div>
    )
  }
  if (error) {
    return <div className="p-6 text-sm text-err">{(error as Error).message}</div>
  }
  if (!project) {
    return <div className="p-6 text-sm text-ink-3">Project not found.</div>
  }

  return (
    // Edge-to-edge project shell: no outer padding — the card fills
    // main completely. Removes the "frame" effect where page bg
    // peeked through around the card. h-full from AppShell cascades
    // down so the card occupies full viewport height minus topbar.
    <div className="h-full flex flex-col min-h-0">
      <Card padding="p-0" hero className="flex-1 min-h-0 flex flex-col !rounded-none !border-0">
        {/* Project-name strip + tabs — natural height, sit at top of
            card. They don't move because the scrolling region lives
            below them. */}
        <div className="px-5 py-3 border-b border-line-1/60 flex items-baseline gap-3 shrink-0">
          <div className="text-sm font-semibold text-ink-1 truncate">{project.name}</div>
          <div className="text-[0.65rem] font-mono text-ink-4 truncate">{project.id}</div>
        </div>

        <div className="px-3 pt-2 border-b border-line-1 flex gap-1 shrink-0">
          {TABS.map((t) => (
            <NavLink
              key={t.to}
              to={t.to}
              replace
              className={({ isActive }) =>
                [
                  'px-3 py-2 text-xs rounded-t-md transition',
                  isActive
                    ? 'text-ink-1 border-b-2 border-brand -mb-px'
                    : 'text-ink-3 hover:text-ink-1',
                ].join(' ')
              }
            >
              {t.label}
            </NavLink>
          ))}
        </div>

        {/* Scrolling region. Form-style tabs (Info, Config, Plugins,
            Settings) overflow naturally and this div provides the
            scrollbar. Tabs that want a bounded layout (Versions with
            its two-column grid) can set `h-full` on their root and
            manage their own internal scrolls. */}
        <div className="p-5 flex-1 min-h-0 overflow-y-auto">
          <Routes>
            <Route index element={<Navigate to="info" replace />} />
            <Route path="info" element={<InfoTab project={project} />} />
            <Route path="config" element={<ConfigTab projectId={project.id} />} />
            <Route path="versions" element={<VersionsTab projectId={project.id} />} />
            <Route path="runs" element={<RunsTab projectId={project.id} />} />
            <Route path="plugins" element={<PluginsTab projectId={project.id} />} />
            <Route path="settings" element={<SettingsTab projectId={project.id} />} />
            <Route path="*" element={<Navigate to="info" replace />} />
          </Routes>
        </div>
      </Card>
    </div>
  )
}
