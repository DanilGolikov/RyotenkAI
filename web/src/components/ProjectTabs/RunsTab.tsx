export function RunsTab({ projectId }: { projectId: string }) {
  return (
    <div className="text-xs text-ink-3 space-y-2">
      <div>
        Project-scoped runs will land here as soon as you launch a pipeline with
        this workspace selected.
      </div>
      <div className="text-[0.65rem] font-mono text-ink-4">project = {projectId}</div>
    </div>
  )
}
