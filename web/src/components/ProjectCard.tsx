import { Link } from 'react-router-dom'
import { timeAgo } from '../lib/format'
import type { ProjectSummary } from '../api/types'

export function ProjectCard({ project }: { project: ProjectSummary }) {
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
          <div className="text-2xs text-ink-3 whitespace-nowrap">
            {project.created_at ? timeAgo(project.created_at) : ''}
          </div>
        </div>
        {project.description && (
          <div className="text-xs text-ink-2 line-clamp-2">{project.description}</div>
        )}
        <div className="pt-2 border-t border-line-1/60 text-[0.65rem] font-mono text-ink-4 truncate">
          {project.path}
        </div>
      </div>
    </Link>
  )
}
