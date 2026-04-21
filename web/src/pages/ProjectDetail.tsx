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
import { Spinner } from '../components/ui'

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
    // `max-w-4xl` keeps the metadata block from pooling to the left
    // on a 1200+ px canvas — the dl previously left ~70 % of the row
    // empty. Description gets its own full-width block so the
    // textarea can stretch comfortably; metadata sits above in a
    // 2-column grid that wraps naturally on narrower viewports.
    <div className="max-w-4xl space-y-6">
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-8 gap-y-4 text-xs">
        <InfoField label="Name" value={project.name} />
        <InfoField label="ID" value={project.id} mono />
        <InfoField label="Created" value={project.created_at} mono />
        <InfoField label="Updated" value={project.updated_at} mono />
      </div>

      <div className="space-y-2">
        <div className="text-[0.7rem] uppercase tracking-wider text-ink-4 font-medium">
          Description
        </div>
        <DescriptionEditor project={project} />
      </div>

      <div className="space-y-2">
        <div className="text-[0.7rem] uppercase tracking-wider text-ink-4 font-medium">
          Workspace
        </div>
        <WorkspacePath path={project.path} />
      </div>
    </div>
  )
}

function InfoField({ label, value, mono = false }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="space-y-1 min-w-0">
      <div className="text-[0.65rem] uppercase tracking-wider text-ink-4 font-medium">
        {label}
      </div>
      <div className={`text-ink-1 break-all ${mono ? 'font-mono text-[0.75rem]' : ''}`}>
        {value}
      </div>
    </div>
  )
}

function WorkspacePath({ path }: { path: string }) {
  const [copied, setCopied] = useState(false)
  async function copy() {
    try {
      await navigator.clipboard.writeText(path)
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    } catch {
      /* clipboard unavailable — leave silently */
    }
  }
  return (
    <div className="flex items-center gap-2 rounded-md border border-line-1 bg-surface-1 px-3 py-2">
      <code className="flex-1 min-w-0 text-[0.75rem] font-mono text-ink-2 break-all">
        {path}
      </code>
      <button
        type="button"
        onClick={copy}
        title={copied ? 'Copied' : 'Copy path'}
        className="text-2xs text-ink-3 hover:text-ink-1 transition shrink-0 px-2 py-1 rounded border border-line-1 hover:border-line-2"
      >
        {copied ? 'copied' : 'copy'}
      </button>
    </div>
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
    // main completely. Uses `.card-hero-ambient` so the root-level
    // ambient gradients (top-left burgundy, bottom-right violet)
    // bleed through, matching the visual of Overview/Projects/Settings
    // where those ambient washes are visible behind content.
    <div className="h-full flex flex-col min-h-0">
      <div className="card-hero-ambient flex-1 min-h-0 flex flex-col !rounded-none !border-0">
        {/* Project header: real h1 with burgundy-gradient text, ID
            slug below as a quiet caption, optional description on
            its OWN line below the ID. Previously id + description
            shared a single flex row which truncated both on narrow
            viewports and made description feel tacked-on. Split
            layout gives the description its full breathing room. */}
        <div className="px-6 pt-5 pb-4 border-b border-line-1/60 shrink-0">
          <h1 className="text-2xl font-semibold text-ink-1 leading-tight truncate">
            {project.name}
          </h1>
          <div className="mt-1 text-[0.65rem] font-mono text-ink-4 truncate">
            {project.id}
          </div>
          {project.description && (
            <div className="mt-1.5 text-xs text-ink-3 line-clamp-2 max-w-3xl">
              {project.description}
            </div>
          )}
        </div>

        {/* Tabs strip — aligned with header padding (px-6), pt-3 for
            breathing room from the title block above, slightly
            larger text-sm for easier hit-target on a 1200+ canvas,
            and a brighter line-2 bottom border so the boundary with
            the content area reads clearly. */}
        <div className="px-6 pt-3 border-b border-line-2 flex gap-1 shrink-0">
          {TABS.map((t) => (
            <NavLink
              key={t.to}
              to={t.to}
              replace
              className={({ isActive }) =>
                [
                  // Border widths reserved in base (1px all sides +
                  // 2px bottom) so toggling the colors doesn't shift
                  // layout. Colors are assigned separately per-state —
                  // putting `border-transparent` in the base collides
                  // with `border-line-2` in active (Tailwind generates
                  // both as `border-color`, and ordering in the
                  // stylesheet decides which wins per side), which
                  // is why the top/side frame was invisible before.
                  'relative px-3 py-2.5 text-sm rounded-t-md',
                  'border border-b-2',
                  'transition-colors duration-150',
                  isActive
                    // Three concurrent signals on active — matches the
                    // NN/g recommendation of "active needs ~3× weight
                    // over inactive":
                    //   1) brighter text (ink-1) + medium weight
                    //   2) burgundy bottom "полоска" (2px)
                    //   3) soft side+top frame (line-2) + bg lift to
                    //      surface-3, so the tab silhouette reads
                    //      as a card popping out of the strip.
                    ? 'text-ink-1 font-medium bg-surface-3 border-[#3c4046] border-b-brand-warm'
                    : 'text-ink-3 hover:text-ink-1 hover:bg-surface-2/50 border-transparent',
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
      </div>
    </div>
  )
}
