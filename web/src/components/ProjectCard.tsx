import { Link } from 'react-router-dom'
import { timeAgo } from '../lib/format'
import type { ProjectSummary } from '../api/types'

export function ProjectCard({ project }: { project: ProjectSummary }) {
  return (
    <Link
      to={`/projects/${encodeURIComponent(project.id)}`}
      className="block card p-4 hover:border-line-2 transition"
    >
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
        <div className="mt-2 text-xs text-ink-2 line-clamp-2">{project.description}</div>
      )}
      <div className="mt-3 text-[0.65rem] font-mono text-ink-4 truncate">{project.path}</div>
    </Link>
  )
}
