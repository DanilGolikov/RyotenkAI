import { NavLink, Navigate, Route, Routes, useParams } from 'react-router-dom'
import { useProject } from '../api/hooks/useProjects'
import { ConfigTab } from '../components/ProjectTabs/ConfigTab'
import { PluginsTab } from '../components/ProjectTabs/PluginsTab'
import { RunsTab } from '../components/ProjectTabs/RunsTab'
import { VersionsTab } from '../components/ProjectTabs/VersionsTab'
import { Card, Spinner } from '../components/ui'

const TABS: { to: string; label: string }[] = [
  { to: 'config', label: 'Config' },
  { to: 'versions', label: 'Versions' },
  { to: 'runs', label: 'Runs' },
  { to: 'plugins', label: 'Plugins' },
]

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
    <div className="max-w-6xl mx-auto p-6 space-y-4">
      <Card padding="p-0">
        <div className="px-5 py-4 border-b border-line-1 bg-gradient-brand-soft">
          <div className="text-lg font-semibold text-ink-1">{project.name}</div>
          <div className="text-2xs text-ink-3 font-mono mt-0.5">{project.id}</div>
          {project.description && (
            <div className="text-xs text-ink-2 mt-2">{project.description}</div>
          )}
          <div className="text-[0.65rem] text-ink-4 font-mono mt-2 truncate">
            {project.path}
          </div>
        </div>

        <div className="px-3 pt-2 border-b border-line-1 flex gap-1">
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

        <div className="p-5">
          <Routes>
            <Route index element={<Navigate to="config" replace />} />
            <Route path="config" element={<ConfigTab projectId={project.id} />} />
            <Route path="versions" element={<VersionsTab projectId={project.id} />} />
            <Route path="runs" element={<RunsTab projectId={project.id} />} />
            <Route path="plugins" element={<PluginsTab projectId={project.id} />} />
            <Route path="*" element={<Navigate to="config" replace />} />
          </Routes>
        </div>
      </Card>
    </div>
  )
}
