import { useEffect, useMemo, useRef, useState } from 'react'
import type { JsonSchemaNode, PipelineJsonSchema } from '../../api/hooks/useConfigSchema'
import { detectKind, resolveRef, titleOrKey } from './schemaUtils'

interface IndexEntry {
  path: string
  dottedLabel: string
  title: string
  description: string
  group: string
}

const MAX_DEPTH = 6
const MAX_FILES = 600

function walkSchema(root: PipelineJsonSchema): IndexEntry[] {
  const out: IndexEntry[] = []
  const topProps = (root.properties ?? {}) as Record<string, JsonSchemaNode>

  function visit(
    rawNode: JsonSchemaNode,
    pathTokens: string[],
    titleTokens: string[],
    group: string,
    depth: number,
  ) {
    if (out.length >= MAX_FILES || depth > MAX_DEPTH) return
    const node = resolveRef(root, rawNode)
    const kind = detectKind(node)
    const title = titleTokens.join(' › ')
    const description = typeof node.description === 'string' ? node.description : ''
    const path = pathTokens.join('.')

    if (kind === 'object') {
      // Push the group itself + recurse into children.
      if (pathTokens.length > 0) {
        out.push({ path, dottedLabel: path, title, description, group })
      }
      const children = (node.properties ?? {}) as Record<string, JsonSchemaNode>
      for (const key of Object.keys(children)) {
        visit(children[key], [...pathTokens, key], [...titleTokens, titleOrKey(children[key], key)], group, depth + 1)
      }
      return
    }

    if (kind === 'union') {
      const branches = (node.anyOf ?? node.oneOf ?? []) as JsonSchemaNode[]
      for (const branch of branches) {
        visit(branch, pathTokens, titleTokens, group, depth + 1)
      }
      return
    }

    if (pathTokens.length > 0) {
      out.push({ path, dottedLabel: path, title, description, group })
    }
  }

  for (const key of Object.keys(topProps)) {
    visit(topProps[key], [key], [titleOrKey(topProps[key], key)], key, 0)
  }
  return out
}

function fuzzyMatch(entry: IndexEntry, q: string): number {
  if (!q) return 1
  const needle = q.toLowerCase()
  const hay = (entry.path + ' ' + entry.title + ' ' + entry.description).toLowerCase()
  if (!hay.includes(needle)) return 0
  // Prefer matches in path/title over description
  const pathScore = entry.path.toLowerCase().includes(needle) ? 3 : 0
  const titleScore = entry.title.toLowerCase().includes(needle) ? 2 : 0
  return pathScore + titleScore + 1
}

export function FieldSearchOmniBox({
  schema,
  hashPrefix = '',
}: {
  schema: PipelineJsonSchema
  hashPrefix?: string
}) {
  const [open, setOpen] = useState(false)
  const [query, setQuery] = useState('')
  const [cursor, setCursor] = useState(0)
  const inputRef = useRef<HTMLInputElement | null>(null)

  const index = useMemo(() => walkSchema(schema), [schema])
  const results = useMemo(() => {
    return index
      .map((e) => ({ entry: e, score: fuzzyMatch(e, query) }))
      .filter((r) => r.score > 0)
      .sort((a, b) => b.score - a.score)
      .slice(0, 50)
      .map((r) => r.entry)
  }, [index, query])

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      const active = document.activeElement as HTMLElement | null
      const inEditable =
        active &&
        (active.tagName === 'INPUT' ||
          active.tagName === 'TEXTAREA' ||
          active.tagName === 'SELECT' ||
          active.isContentEditable)

      if (e.key === '/' && !e.metaKey && !e.ctrlKey && !e.altKey && !inEditable) {
        e.preventDefault()
        setOpen(true)
      } else if (e.key === 'Escape' && open) {
        setOpen(false)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [open])

  useEffect(() => {
    if (open) {
      setQuery('')
      setCursor(0)
      setTimeout(() => inputRef.current?.focus(), 0)
    }
  }, [open])

  useEffect(() => setCursor(0), [query])

  function jump(entry: IndexEntry) {
    const prefix = hashPrefix ? `${hashPrefix}:` : ''
    const nextHash = `#${prefix}${entry.group}`
    if (window.location.hash !== nextHash) {
      history.replaceState(null, '', nextHash)
      window.dispatchEvent(new HashChangeEvent('hashchange'))
    }
    setOpen(false)
    // Attempt to scroll to a specific anchor. Will be proper in C.8.
    setTimeout(() => {
      const anchor = document.querySelector(`[data-field-path="${entry.path}"]`)
      if (anchor) anchor.scrollIntoView({ behavior: 'smooth', block: 'center' })
    }, 50)
  }

  if (!open) return null

  return (
    <div
      className="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm flex items-start justify-center pt-24"
      onClick={() => setOpen(false)}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        className="w-full max-w-2xl rounded-xl border border-line-2 bg-surface-1 shadow-card overflow-hidden"
      >
        <div className="px-4 py-3 border-b border-line-1 flex items-center gap-3">
          <span className="text-ink-3 text-xs font-mono">/</span>
          <input
            ref={inputRef}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'ArrowDown') {
                e.preventDefault()
                setCursor((c) => Math.min(c + 1, results.length - 1))
              } else if (e.key === 'ArrowUp') {
                e.preventDefault()
                setCursor((c) => Math.max(c - 1, 0))
              } else if (e.key === 'Enter' && results[cursor]) {
                e.preventDefault()
                jump(results[cursor])
              }
            }}
            placeholder="Jump to field…"
            className="flex-1 bg-transparent text-sm focus:outline-none placeholder:text-ink-4"
          />
          <button
            type="button"
            onClick={() => setOpen(false)}
            className="text-ink-3 hover:text-ink-1 text-2xs"
          >
            esc
          </button>
        </div>
        <div className="max-h-[400px] overflow-y-auto py-1">
          {results.length === 0 ? (
            <div className="px-4 py-6 text-center text-2xs text-ink-3">No matches.</div>
          ) : (
            results.map((r, idx) => (
              <button
                key={r.path}
                type="button"
                onMouseEnter={() => setCursor(idx)}
                onClick={() => jump(r)}
                className={[
                  'w-full text-left px-4 py-2 flex items-baseline gap-3 transition',
                  idx === cursor ? 'bg-surface-2 text-ink-1' : 'text-ink-2 hover:bg-surface-2/60',
                ].join(' ')}
              >
                <span className="text-2xs font-mono text-brand-alt">{r.group}</span>
                <span className="text-xs font-mono text-ink-1 truncate">{r.path}</span>
                <span className="text-[0.6rem] text-ink-3 truncate ml-auto">{r.title}</span>
              </button>
            ))
          )}
        </div>
        <div className="px-4 py-2 border-t border-line-1 text-[0.6rem] text-ink-4 flex gap-4">
          <span>↑↓ navigate</span>
          <span>⏎ jump</span>
          <span>esc close</span>
        </div>
      </div>
    </div>
  )
}
