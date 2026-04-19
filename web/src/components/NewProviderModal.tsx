import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useCreateProvider, useProviderTypes } from '../api/hooks/useProviders'
import { ApiError } from '../api/client'

function slugify(value: string): string {
  return value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-|-$/g, '')
    .slice(0, 48)
}

export function NewProviderModal({ open, onClose }: { open: boolean; onClose: () => void }) {
  const [type, setType] = useState<string>('')
  const [name, setName] = useState('')
  const [idOverride, setIdOverride] = useState('')
  const [pathOverride, setPathOverride] = useState('')
  const [description, setDescription] = useState('')
  const [touchedId, setTouchedId] = useState(false)
  const typesQuery = useProviderTypes()
  const createMut = useCreateProvider()
  const navigate = useNavigate()

  useEffect(() => {
    if (!open) {
      setType('')
      setName('')
      setIdOverride('')
      setPathOverride('')
      setDescription('')
      setTouchedId(false)
      createMut.reset()
    } else if (!type && typesQuery.data && typesQuery.data.types.length > 0) {
      setType(typesQuery.data.types[0].id)
    }
  }, [open, typesQuery.data])

  if (!open) return null

  const derivedId = touchedId ? idOverride : slugify(name)

  async function onSubmit(event: React.FormEvent) {
    event.preventDefault()
    try {
      const summary = await createMut.mutateAsync({
        type,
        name: name.trim(),
        id: derivedId || undefined,
        path: pathOverride.trim() || undefined,
        description: description.trim(),
      })
      onClose()
      navigate(`/settings/providers/${summary.id}`)
    } catch {
      /* error shown below */
    }
  }

  const errorMsg = (() => {
    if (!createMut.error) return null
    if (createMut.error instanceof ApiError) return createMut.error.message
    return String(createMut.error)
  })()

  return (
    <div
      className="fixed inset-0 z-40 bg-black/70 backdrop-blur-sm flex items-center justify-center p-4"
      onClick={onClose}
    >
      <form
        onSubmit={onSubmit}
        onClick={(event) => event.stopPropagation()}
        className="w-full max-w-lg rounded-xl border border-line-2 bg-surface-1 shadow-card overflow-hidden"
      >
        <div className="px-5 py-4 bg-gradient-brand-soft border-b border-line-1">
          <div className="text-sm font-semibold">New provider</div>
          <div className="text-2xs text-ink-3">
            Reusable provider configuration (SingleNode / RunPod). Versioned on every save.
          </div>
        </div>

        <div className="p-5 space-y-4">
          <label className="block">
            <div className="text-2xs text-ink-3 mb-1">Type</div>
            <select
              value={type}
              onChange={(e) => setType(e.target.value)}
              className="w-full rounded-md bg-surface-2 border border-line-1 px-3 py-2 text-sm focus:outline-none focus:border-brand"
            >
              {typesQuery.data?.types.map((t) => (
                <option key={t.id} value={t.id}>
                  {t.label}
                </option>
              ))}
            </select>
          </label>

          <label className="block">
            <div className="text-2xs text-ink-3 mb-1">Name</div>
            <input
              autoFocus
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="w-full rounded-md bg-surface-2 border border-line-1 px-3 py-2 text-sm focus:outline-none focus:border-brand"
              placeholder={type === 'runpod' ? 'RunPod prod' : 'Local GPU'}
            />
          </label>

          <label className="block">
            <div className="text-2xs text-ink-3 mb-1">
              Provider id <span className="text-ink-4">(slug)</span>
            </div>
            <input
              value={derivedId}
              onChange={(e) => {
                setTouchedId(true)
                setIdOverride(e.target.value)
              }}
              className="w-full rounded-md bg-surface-2 border border-line-1 px-3 py-2 text-sm font-mono focus:outline-none focus:border-brand"
              placeholder="runpod-prod"
            />
          </label>

          <label className="block">
            <div className="text-2xs text-ink-3 mb-1">
              Path <span className="text-ink-4">(optional, absolute)</span>
            </div>
            <input
              value={pathOverride}
              onChange={(e) => setPathOverride(e.target.value)}
              className="w-full rounded-md bg-surface-2 border border-line-1 px-3 py-2 text-sm font-mono focus:outline-none focus:border-brand"
              placeholder="~/.ryotenkai/providers/<id>/"
            />
            <div className="text-[0.65rem] text-ink-4 mt-1">
              Default: <span className="font-mono">~/.ryotenkai/providers/{derivedId || '<id>'}/</span>
            </div>
          </label>

          <label className="block">
            <div className="text-2xs text-ink-3 mb-1">Description</div>
            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              rows={2}
              className="w-full rounded-md bg-surface-2 border border-line-1 px-3 py-2 text-sm focus:outline-none focus:border-brand"
              placeholder="Where is this provider used?"
            />
          </label>

          {errorMsg && (
            <div className="rounded-md border border-err/40 bg-err/10 text-err text-xs px-3 py-2">
              {errorMsg}
            </div>
          )}
        </div>

        <div className="px-5 py-3 border-t border-line-1 bg-surface-0 flex justify-end gap-2">
          <button
            type="button"
            onClick={onClose}
            className="rounded-md px-3 py-1.5 text-xs text-ink-2 hover:text-ink-1 border border-line-1 hover:border-line-2"
          >
            Cancel
          </button>
          <button
            type="submit"
            disabled={!name.trim() || !type || createMut.isPending}
            className="btn-primary px-4 py-1.5 text-xs disabled:opacity-50"
          >
            {createMut.isPending ? 'Creating…' : 'Create provider'}
          </button>
        </div>
      </form>
    </div>
  )
}
