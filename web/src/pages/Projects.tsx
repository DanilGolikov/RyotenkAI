import { useState } from 'react'
import { useProjects } from '../api/hooks/useProjects'
import { NewProjectModal } from '../components/NewProjectModal'
import { ProjectCard } from '../components/ProjectCard'
import { Card, EmptyState, SectionHeader, Spinner } from '../components/ui'

export function ProjectsPage() {
  const { data, isLoading, error } = useProjects()
  const [modalOpen, setModalOpen] = useState(false)

  const newProjectBtn = (
    <button
      type="button"
      onClick={() => setModalOpen(true)}
      className="btn-primary px-3 py-1.5 text-xs"
    >
      + New project
    </button>
  )

  return (
    <div className="px-6 py-6 space-y-4">
      <Card padding="p-0">
        <div className="px-4 pt-4">
          <SectionHeader
            title="Projects"
            subtitle="Experiment workspaces: config, versions, plugins, runs."
            action={newProjectBtn}
          />
        </div>
        <div className="p-4 pt-0">
          {error ? (
            <div className="px-3 py-4 text-sm text-err">{(error as Error).message}</div>
          ) : isLoading ? (
            <div className="px-3 py-4 text-sm text-ink-3 flex items-center gap-2">
              <Spinner /> loading
            </div>
          ) : !data || data.length === 0 ? (
            <EmptyState
              title="No projects yet"
              hint="Create a workspace to start building a config and tracking its runs."
              action={newProjectBtn}
            />
          ) : (
            <div className="grid gap-3 grid-cols-1 sm:grid-cols-2 lg:grid-cols-3">
              {data.map((p) => (
                <ProjectCard key={p.id} project={p} />
              ))}
            </div>
          )}
        </div>
      </Card>
      <NewProjectModal open={modalOpen} onClose={() => setModalOpen(false)} />
    </div>
  )
}
