import { useMemo } from 'react'

/**
 * Tiny YAML highlighter rendered in a <pre>. Tokenises by line so we can
 * colour comments, keys, and scalar values without pulling in a syntax
 * library. Good enough for read-only previews of pipeline configs.
 */
interface Props {
  text: string
  className?: string
  maxHeight?: string
}

type Segment = { cls?: string; text: string }

const KEY_RE = /^(\s*)([-?]?\s*)([^\s:#][^:#]*?)(:)(\s|$)/
const NUMBER_RE = /^-?\d+(\.\d+)?([eE][-+]?\d+)?$/
const BOOL_RE = /^(true|false|null|yes|no|on|off|~)$/i

function tokenizeValue(value: string): Segment[] {
  const trimmed = value.trim()
  if (!trimmed) return [{ text: value }]
  if (trimmed.startsWith('"') || trimmed.startsWith("'")) {
    return [{ cls: 'text-ok/90', text: value }]
  }
  if (NUMBER_RE.test(trimmed)) return [{ cls: 'text-warn', text: value }]
  if (BOOL_RE.test(trimmed)) return [{ cls: 'text-brand-alt', text: value }]
  return [{ cls: 'text-ink-2', text: value }]
}

function tokenizeLine(raw: string): Segment[] {
  const commentIdx = (() => {
    // Find # not inside quotes. Simple heuristic.
    let inSingle = false
    let inDouble = false
    for (let i = 0; i < raw.length; i++) {
      const c = raw[i]
      if (c === "'" && !inDouble) inSingle = !inSingle
      else if (c === '"' && !inSingle) inDouble = !inDouble
      else if (c === '#' && !inSingle && !inDouble) return i
    }
    return -1
  })()

  const head = commentIdx >= 0 ? raw.slice(0, commentIdx) : raw
  const tail: Segment[] =
    commentIdx >= 0 ? [{ cls: 'text-ink-4 italic', text: raw.slice(commentIdx) }] : []

  const match = head.match(KEY_RE)
  if (match) {
    const [, indent, bullet, key, colon, trailing] = match
    const rest = head.slice(indent.length + bullet.length + key.length + colon.length + trailing.length)
    return [
      { text: indent },
      bullet ? { cls: 'text-brand-alt', text: bullet } : { text: '' },
      { cls: 'text-brand', text: key },
      { cls: 'text-ink-3', text: colon + (trailing ? trailing : '') },
      ...tokenizeValue(rest),
      ...tail,
    ]
  }

  // Plain line: could be a standalone scalar (list item, block scalar, etc.)
  const listMatch = head.match(/^(\s*)(-\s+)(.*)$/)
  if (listMatch) {
    const [, indent, bullet, rest] = listMatch
    return [
      { text: indent },
      { cls: 'text-brand-alt', text: bullet },
      ...tokenizeValue(rest),
      ...tail,
    ]
  }

  return [...tokenizeValue(head), ...tail]
}

export function YamlView({ text, className = '', maxHeight = 'max-h-[520px]' }: Props) {
  const lines = useMemo(() => text.split('\n'), [text])
  return (
    <pre
      className={[
        'bg-surface-0 border border-line-1 rounded-md p-3 text-xs font-mono overflow-auto leading-relaxed',
        maxHeight,
        className,
      ].join(' ')}
    >
      {lines.map((line, idx) => {
        const segs = tokenizeLine(line)
        return (
          <div key={idx} className="whitespace-pre">
            {segs.map((seg, i) => (
              <span key={i} className={seg.cls}>
                {seg.text}
              </span>
            ))}
            {segs.length === 0 && <span>&nbsp;</span>}
          </div>
        )
      })}
    </pre>
  )
}
