import { useEffect, useRef, useState } from 'react'
import { useLogStream } from '../api/hooks/useLogStream'
import { Toggle } from './ui'

const FILES = ['pipeline.log', 'training.log', 'inference.log', 'eval.log'] as const
type LogFile = (typeof FILES)[number]

export function LogDock({
  runId,
  attemptNo,
  enabled,
  height = 'h-[22rem]',
}: {
  runId: string
  attemptNo: number
  enabled: boolean
  height?: string
}) {
  const [file, setFile] = useState<LogFile>('pipeline.log')
  const [autoScroll, setAutoScroll] = useState(true)
  const [collapsed, setCollapsed] = useState(false)
  const scrollRef = useRef<HTMLDivElement | null>(null)
  const { lines, connected, error } = useLogStream(runId, attemptNo, file, enabled && !collapsed)

  useEffect(() => {
    if (!autoScroll || !scrollRef.current) return
    scrollRef.current.scrollTop = scrollRef.current.scrollHeight
  }, [lines, autoScroll])

  return (
    <div className={`card-raised flex flex-col overflow-hidden ${collapsed ? 'h-auto' : height}`}>
      <div className="flex items-center justify-between px-3 py-2 border-b border-line-1 gap-3">
        <div className="flex items-center gap-2 min-w-0">
          <span className={`live-dot ${connected ? '' : 'opacity-30'}`} />
          <span className="text-xs text-ink-2">Live logs</span>
          <span className="text-2xs text-ink-3 font-mono truncate">/ {file}</span>
        </div>
        <div className="flex items-center gap-1.5 flex-wrap">
          {FILES.map((name) => (
            <button
              key={name}
              type="button"
              onClick={() => setFile(name)}
              className={[
                'px-2 py-0.5 rounded text-2xs font-mono transition border',
                file === name
                  ? 'bg-surface-3 text-ink-1 border-line-2'
                  : 'bg-transparent border-line-1 text-ink-3 hover:text-ink-1 hover:border-line-2',
              ].join(' ')}
            >
              {name}
            </button>
          ))}
          {error && <span className="text-2xs text-err">error</span>}
          <div className="flex items-center gap-1.5 text-2xs text-ink-3 ml-1">
            <Toggle
              checked={autoScroll}
              onChange={setAutoScroll}
              aria-label="Follow tail"
            />
            <span
              className="cursor-pointer select-none"
              onClick={() => setAutoScroll((v) => !v)}
            >
              follow
            </span>
          </div>
          <button
            type="button"
            onClick={() => setCollapsed((v) => !v)}
            className="text-2xs text-ink-3 hover:text-ink-1 px-1"
            aria-label={collapsed ? 'expand' : 'collapse'}
          >
            {collapsed ? '▴' : '▾'}
          </button>
        </div>
      </div>
      {!collapsed && (
        <div
          ref={scrollRef}
          onScroll={(event) => {
            const el = event.currentTarget
            const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 40
            if (!atBottom && autoScroll) setAutoScroll(false)
          }}
          className="flex-1 overflow-auto font-mono text-[11.5px] leading-relaxed px-3 py-2 whitespace-pre text-ink-2 bg-surface-1"
        >
          {lines.length === 0
            ? <div className="text-ink-4">waiting for output…</div>
            : lines.map((line, idx) => <div key={`${idx}:${line.slice(0, 16)}`}>{line}</div>)}
        </div>
      )}
    </div>
  )
}
