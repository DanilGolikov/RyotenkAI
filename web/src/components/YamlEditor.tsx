import { Fragment, useRef } from 'react'
import { tokenizeYamlLine } from '../lib/yamlTokens'

/**
 * YAML editor with live syntax highlighting.
 *
 * Overlay trick: a transparent ``<textarea>`` sits on top of a ``<pre>``
 * that renders the tokenised text. Both layers use the EXACT same
 * typography, padding, and wrapping, and the textarea's scroll is
 * mirrored to the pre so the highlighted text tracks the caret
 * perfectly.
 *
 * Two things are critical for alignment:
 *   - no block elements inside the pre; the highlighter emits inline
 *     ``<span>`` segments joined by literal ``\n`` so the pre renders
 *     newlines the same way the textarea does.
 *   - ``wrap="off"`` on the textarea + ``whitespace-pre`` on the pre —
 *     textarea's soft-wrap doesn't match CSS pre-wrap pixel-for-pixel,
 *     so we disable wrapping and let both scroll horizontally instead.
 */
interface Props {
  value: string
  onChange: (next: string) => void
  rows?: number
  maxHeight?: string
  placeholder?: string
  spellCheck?: boolean
  onBlur?: () => void
}

// Shared typography/padding — both layers MUST match for alignment.
// Explicit line-height in em keeps textarea and pre at identical metrics
// (leading-relaxed is 1.625 in Tailwind; setting the exact value avoids
// browsers applying subtly different defaults to textareas vs pres).
const EDITOR_TYPO =
  'px-3 py-3 text-xs font-mono leading-[1.5rem] whitespace-pre'

export function YamlEditor({
  value,
  onChange,
  rows = 18,
  maxHeight = 'max-h-[420px]',
  placeholder = '# paste or build your pipeline config here',
  spellCheck = false,
  onBlur,
}: Props) {
  const preRef = useRef<HTMLPreElement | null>(null)

  function syncScroll(e: React.UIEvent<HTMLTextAreaElement>) {
    const t = e.currentTarget
    if (preRef.current) {
      preRef.current.scrollTop = t.scrollTop
      preRef.current.scrollLeft = t.scrollLeft
    }
  }

  const showPlaceholder = !value
  const displayText = value || placeholder

  return (
    <div
      className={[
        'relative rounded-md border border-line-1 bg-surface-0',
        maxHeight,
      ].join(' ')}
    >
      {/* Highlighted background. Same geometry as the textarea, scroll
          mirrored in JS. pointer-events-none so the textarea owns input. */}
      <pre
        ref={preRef}
        aria-hidden
        className={[
          'absolute inset-0 m-0 overflow-hidden',
          EDITOR_TYPO,
          'pointer-events-none select-none',
          showPlaceholder ? 'text-ink-4' : '',
        ].join(' ')}
      >
        <HighlightedYaml text={displayText} />
        {/* Trailing newline so the caret at EOF has a row to land on
            without the pre shrinking by one line vs. the textarea. */}
        {'\n'}
      </pre>

      <textarea
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onBlur={onBlur}
        onScroll={syncScroll}
        spellCheck={spellCheck}
        rows={rows}
        wrap="off"
        placeholder={placeholder}
        className={[
          'relative block w-full',
          maxHeight,
          EDITOR_TYPO,
          // Text is invisible, caret visible via caret-color. Selection
          // bg is painted by the textarea so the user sees real native
          // selection rectangles over the highlighted text below.
          'bg-transparent text-transparent caret-ink-1',
          'selection:bg-brand/30 selection:text-transparent',
          'focus:outline-none resize-none overflow-auto',
          'placeholder:text-ink-4',
        ].join(' ')}
      />
    </div>
  )
}

function HighlightedYaml({ text }: { text: string }) {
  const lines = text.split('\n')
  return (
    <>
      {lines.map((line, i) => {
        const segs = tokenizeYamlLine(line)
        return (
          <Fragment key={i}>
            {segs.map((s, j) => (
              <span key={j} className={s.cls}>
                {s.text}
              </span>
            ))}
            {i < lines.length - 1 && '\n'}
          </Fragment>
        )
      })}
    </>
  )
}
