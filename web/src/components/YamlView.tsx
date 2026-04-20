import { useMemo } from 'react'
import { tokenizeYamlLine } from '../lib/yamlTokens'

interface Props {
  text: string
  className?: string
  maxHeight?: string
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
        const segs = tokenizeYamlLine(line)
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
