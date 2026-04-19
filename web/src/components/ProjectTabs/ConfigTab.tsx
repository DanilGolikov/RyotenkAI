import { useEffect, useMemo, useState } from 'react'
import {
  useProjectConfig,
  useSaveProjectConfig,
  useValidateProjectConfig,
} from '../../api/hooks/useProjects'
import type { ConfigValidationResult } from '../../api/types'
import { Spinner } from '../ui'

export function ConfigTab({ projectId }: { projectId: string }) {
  const { data, isLoading, error } = useProjectConfig(projectId)
  const saveMut = useSaveProjectConfig(projectId)
  const validateMut = useValidateProjectConfig(projectId)

  const [yamlText, setYamlText] = useState<string>('')
  const [dirty, setDirty] = useState(false)

  useEffect(() => {
    if (data && !dirty) setYamlText(data.yaml)
  }, [data, dirty])

  const validationResult: ConfigValidationResult | undefined = validateMut.data

  const statusLine = useMemo(() => {
    if (saveMut.isPending) return 'Saving…'
    if (validateMut.isPending) return 'Validating…'
    if (dirty) return 'Unsaved changes'
    if (saveMut.isSuccess) return 'Saved'
    return ''
  }, [saveMut.isPending, saveMut.isSuccess, validateMut.isPending, dirty])

  if (isLoading) {
    return (
      <div className="flex items-center gap-2 text-sm text-ink-3">
        <Spinner /> loading config
      </div>
    )
  }
  if (error) {
    return <div className="text-sm text-err">{(error as Error).message}</div>
  }

  return (
    <div className="space-y-4">
      <div className="rounded-md border border-line-1 bg-surface-0 overflow-hidden">
        <textarea
          value={yamlText}
          onChange={(e) => {
            setYamlText(e.target.value)
            setDirty(true)
          }}
          spellCheck={false}
          rows={24}
          className="w-full bg-surface-0 text-ink-1 font-mono text-xs px-4 py-3 focus:outline-none resize-y"
          placeholder="# paste or build your pipeline config here"
        />
      </div>

      <div className="flex items-center gap-2 text-xs">
        <button
          type="button"
          onClick={() => validateMut.mutate(yamlText)}
          className="rounded-md border border-line-1 px-3 py-1.5 text-ink-2 hover:text-ink-1 hover:border-line-2"
          disabled={validateMut.isPending}
        >
          Validate
        </button>
        <button
          type="button"
          onClick={async () => {
            await saveMut.mutateAsync(yamlText)
            setDirty(false)
          }}
          className="btn-primary px-3 py-1.5"
          disabled={saveMut.isPending || !dirty}
        >
          {saveMut.isPending ? 'Saving…' : 'Save'}
        </button>
        <span className="text-ink-3 ml-auto">{statusLine}</span>
      </div>

      {validationResult && (
        <div className="rounded-md border border-line-1 bg-surface-1 p-3 space-y-1.5">
          <div
            className={`text-xs font-medium ${
              validationResult.ok ? 'text-ok' : 'text-err'
            }`}
          >
            {validationResult.ok ? 'Configuration looks valid' : 'Configuration has issues'}
          </div>
          {validationResult.checks.map((c, idx) => (
            <div key={idx} className="flex items-start gap-2 text-2xs">
              <span
                className={[
                  'w-1.5 h-1.5 mt-1.5 rounded-full shrink-0',
                  c.status === 'ok'
                    ? 'bg-ok'
                    : c.status === 'warn'
                    ? 'bg-warn'
                    : 'bg-err',
                ].join(' ')}
              />
              <div className="min-w-0">
                <div className="text-ink-1">{c.label}</div>
                {c.detail && <div className="text-ink-3 font-mono truncate">{c.detail}</div>}
              </div>
            </div>
          ))}
        </div>
      )}

      {saveMut.error && (
        <div className="rounded-md border border-err/40 bg-err/10 text-err text-xs px-3 py-2">
          {(saveMut.error as Error).message}
        </div>
      )}
    </div>
  )
}
