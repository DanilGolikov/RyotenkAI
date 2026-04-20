import { useState } from 'react'
import { Link } from 'react-router-dom'
import { useDeleteProject } from '../api/hooks/useProjects'
import { timeAgo } from '../lib/format'
import type { ProjectSummary } from '../api/types'
import { DeleteProjectModal } from './DeleteProjectModal'

export function ProjectCard({ project }: { project: ProjectSummary }) {
  const del = useDeleteProject()
  const [modalOpen, setModalOpen] = useState(false)

  function openModal(e: React.MouseEvent) {
    e.preventDefault()
    e.stopPropagation()
    setModalOpen(true)
  }

  return (
    <Link
      to={`/projects/${encodeURIComponent(project.id)}`}
      className="group relative block rounded-lg border border-line-1 bg-surface-1 overflow-hidden hover:border-brand/60 hover:bg-surface-2 transition shadow-card"
    >
      <div className="h-0.5 bg-gradient-brand opacity-60 group-hover:opacity-100 transition" />
      <div className="p-4 space-y-3">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <div className="text-sm font-medium text-ink-1 truncate">{project.name}</div>
            <div className="text-2xs text-ink-3 mt-0.5 font-mono truncate">{project.id}</div>
          </div>
          <div className="flex items-center gap-2 shrink-0">
            <div className="text-2xs text-ink-3 whitespace-nowrap">
              {project.created_at ? timeAgo(project.created_at) : ''}
            </div>
            <button
              type="button"
              title="Delete project"
              aria-label={`Delete project ${project.name}`}
              disabled={del.isPending}
              onClick={openModal}
              className="w-6 h-6 inline-flex items-center justify-center rounded text-ink-4 hover:text-err hover:bg-err/10 transition disabled:opacity-50"
            >
              {del.isPending ? (
                <span className="text-[0.55rem] font-mono">…</span>
              ) : (
                <svg
                  className="w-3.5 h-3.5"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                >
                  <path d="M3 6h18" />
                  <path d="M8 6V4a1 1 0 0 1 1-1h6a1 1 0 0 1 1 1v2" />
                  <path d="M5 6l1 14a2 2 0 0 0 2 2h8a2 2 0 0 0 2-2l1-14" />
                  <path d="M10 11v6" />
                  <path d="M14 11v6" />
                </svg>
              )}
            </button>
          </div>
        </div>
        {project.description && (
          <div className="text-xs text-ink-2 line-clamp-2">{project.description}</div>
        )}
        <div className="pt-2 border-t border-line-1/60 text-[0.65rem] font-mono text-ink-4 truncate">
          {project.path}
        </div>
      </div>
      {modalOpen && (
        <DeleteProjectModal
          project={project}
          onClose={() => setModalOpen(false)}
          onConfirm={async (deleteFiles) => {
            try {
              await del.mutateAsync({ projectId: project.id, deleteFiles })
              setModalOpen(false)
            } catch (exc) {
              window.alert((exc as Error).message || 'Failed to delete project.')
            }
          }}
          pending={del.isPending}
        />
      )}
    </Link>
  )
}
