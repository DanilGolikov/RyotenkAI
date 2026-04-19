import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api } from '../api/client'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import type { RunSummary, ConfigValidationResult } from '../api/types'
import { Card, SectionHeader } from '../components/ui'
import { LaunchModal } from '../components/LaunchModal'
import { qk } from '../api/queryKeys'

export function LaunchPage() {
  const navigate = useNavigate()
  const qc = useQueryClient()
  const [subgroup, setSubgroup] = useState('')
  const [runId, setRunId] = useState('')
  const [configPath, setConfigPath] = useState('')
  const [validation, setValidation] = useState<ConfigValidationResult | null>(null)
  const [createdRun, setCreatedRun] = useState<RunSummary | null>(null)
  const [launchOpen, setLaunchOpen] = useState(false)

  const createMut = useMutation({
    mutationFn: (body: { run_id?: string; subgroup?: string }) =>
      api.post<RunSummary>('/runs', body),
    onSuccess: (run) => {
      setCreatedRun(run)
      qc.invalidateQueries({ queryKey: qk.runs() })
      setLaunchOpen(true)
    },
  })

  const validateMut = useMutation({
    mutationFn: (path: string) =>
      api.post<ConfigValidationResult>('/config/validate', { config_path: path }),
    onSuccess: (res) => setValidation(res),
  })

  return (
    <div className="p-5 space-y-5 max-w-[900px]">
      <section className="space-y-1">
        <h1 className="text-2xl font-semibold gradient-text">Launch</h1>
        <p className="text-xs text-ink-3">
          Create a new run directory and start the pipeline. Config validation is optional but recommended.
        </p>
      </section>

      <Card>
        <SectionHeader title="1 · Validate config (optional)" />
        <div className="flex gap-2">
          <input
            value={configPath}
            onChange={(event) => setConfigPath(event.target.value)}
            placeholder="config/pipeline.yaml"
            className="flex-1 bg-surface-2 border border-line-2 rounded-md px-3 py-2 text-sm font-mono focus:border-brand focus:outline-none"
          />
          <button
            type="button"
            disabled={!configPath || validateMut.isPending}
            onClick={() => validateMut.mutate(configPath)}
            className="btn-ghost"
          >
            {validateMut.isPending ? 'Validating…' : 'Validate'}
          </button>
        </div>
        {validateMut.error && (
          <div className="mt-3 text-xs text-err bg-err/10 border border-err/30 px-3 py-2 rounded">
            {(validateMut.error as Error).message}
          </div>
        )}
        {validation && (
          <div className="mt-4 space-y-1">
            <div className={`text-xs font-medium ${validation.ok ? 'text-ok' : 'text-err'}`}>
              {validation.ok ? 'Config OK' : 'Config has errors'}
            </div>
            <ul className="text-2xs space-y-0.5 font-mono">
              {validation.checks.map((check, idx) => (
                <li
                  key={`${idx}:${check.label}`}
                  className={
                    check.status === 'ok'   ? 'text-ok' :
                    check.status === 'warn' ? 'text-warn' :
                    'text-err'
                  }
                >
                  [{check.status.toUpperCase()}] {check.label}
                  {check.detail && <span className="text-ink-3"> — {check.detail}</span>}
                </li>
              ))}
            </ul>
          </div>
        )}
      </Card>

      <Card>
        <SectionHeader title="2 · Create run" subtitle="leave run id empty for auto-generated timestamped id" />
        <div className="grid grid-cols-2 gap-3">
          <label className="block">
            <span className="text-2xs uppercase tracking-wider text-ink-3">Run id</span>
            <input
              value={runId}
              onChange={(event) => setRunId(event.target.value)}
              placeholder="auto"
              className="w-full mt-1 bg-surface-2 border border-line-2 rounded-md px-3 py-2 text-sm font-mono focus:border-brand focus:outline-none"
            />
          </label>
          <label className="block">
            <span className="text-2xs uppercase tracking-wider text-ink-3">Subgroup</span>
            <input
              value={subgroup}
              onChange={(event) => setSubgroup(event.target.value)}
              placeholder="experiments/"
              className="w-full mt-1 bg-surface-2 border border-line-2 rounded-md px-3 py-2 text-sm font-mono focus:border-brand focus:outline-none"
            />
          </label>
        </div>
        {createMut.error && (
          <div className="mt-3 text-xs text-err bg-err/10 border border-err/30 px-3 py-2 rounded">
            {(createMut.error as Error).message}
          </div>
        )}
        <div className="mt-4 flex justify-end">
          <button
            type="button"
            className="btn-primary"
            disabled={createMut.isPending}
            onClick={() =>
              createMut.mutate({
                run_id: runId.trim() || undefined,
                subgroup: subgroup.trim() || undefined,
              })
            }
          >
            {createMut.isPending ? 'Creating…' : 'Create + Launch'}
          </button>
        </div>
      </Card>

      {createdRun && (
        <LaunchModal
          runId={createdRun.run_id}
          open={launchOpen}
          onClose={() => {
            setLaunchOpen(false)
            navigate(`/runs/${encodeURIComponent(createdRun.run_id)}`)
          }}
          defaultMode="new_run"
          defaultConfigPath={configPath || null}
        />
      )}
    </div>
  )
}
