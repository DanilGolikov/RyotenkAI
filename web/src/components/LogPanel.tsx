import { useEffect, useRef, useState } from 'react'
import { useLogStream } from '../api/hooks/useLogStream'

const FILES = ['pipeline.log', 'training.log', 'inference.log', 'eval.log'] as const

export function LogPanel({
  runId,
  attemptNo,
  enabled,
}: {
  runId: string
  attemptNo: number
  enabled: boolean
}) {
  const [file, setFile] = useState<(typeof FILES)[number]>('pipeline.log')
  const [autoScroll, setAutoScroll] = useState(true)
  const scrollRef = useRef<HTMLDivElement | null>(null)
  const { lines, connected, error } = useLogStream(runId, attemptNo, file, enabled)

  useEffect(() => {
    if (!autoScroll || !scrollRef.current) return
    scrollRef.current.scrollTop = scrollRef.current.scrollHeight
  }, [lines, autoScroll])

  return (
    <div className="flex flex-col h-[32rem] border border-surface-muted rounded bg-surface-raised">
      <div className="flex items-center justify-between px-3 py-2 border-b border-surface-muted text-xs">
        <div className="flex gap-2">
          {FILES.map((name) => (
            <button
              key={name}
              type="button"
              onClick={() => setFile(name)}
              className={`px-2 py-1 rounded ${file === name ? 'bg-accent-muted text-white' : 'text-gray-400 hover:text-gray-100'}`}
            >
              {name}
            </button>
          ))}
        </div>
        <div className="flex gap-3 items-center text-gray-500">
          {error && <span className="text-rose-400">error</span>}
          <span className={connected ? 'text-emerald-400' : 'text-gray-500'}>{connected ? 'live' : 'offline'}</span>
          <label className="flex items-center gap-1 cursor-pointer">
            <input
              type="checkbox"
              checked={autoScroll}
              onChange={(event) => setAutoScroll(event.target.checked)}
            />
            follow
          </label>
        </div>
      </div>
      <div
        ref={scrollRef}
        onScroll={(event) => {
          const el = event.currentTarget
          const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 40
          if (!atBottom && autoScroll) setAutoScroll(false)
        }}
        className="flex-1 overflow-auto text-xs font-mono whitespace-pre px-3 py-2"
      >
        {lines.length === 0
          ? <div className="text-gray-600">waiting for output…</div>
          : lines.map((line, idx) => <div key={`${idx}:${line.slice(0, 16)}`}>{line}</div>)}
      </div>
    </div>
  )
}
