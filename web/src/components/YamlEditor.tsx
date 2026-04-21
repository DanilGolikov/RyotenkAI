import { useCallback, useEffect, useRef, useState } from 'react'
import { CollapseIcon, ExpandIcon } from './icons'
import { EditorState } from '@codemirror/state'
import {
  Decoration,
  EditorView,
  MatchDecorator,
  ViewPlugin,
  keymap,
  lineNumbers,
  highlightActiveLine,
} from '@codemirror/view'
import type { DecorationSet, ViewUpdate } from '@codemirror/view'
import { defaultKeymap, history, historyKeymap } from '@codemirror/commands'
import {
  bracketMatching,
  foldAll,
  foldGutter,
  foldKeymap,
  indentOnInput,
  syntaxHighlighting,
  unfoldAll,
  HighlightStyle,
} from '@codemirror/language'
import { yaml } from '@codemirror/lang-yaml'
import { tags as t } from '@lezer/highlight'

/**
 * YAML editor (and read-only viewer) backed by CodeMirror 6.
 *
 * Why CodeMirror and not Monaco: Monaco is ~500 KB gzipped for this
 * use case, mostly TypeScript intellisense we don't need. CodeMirror 6
 * is modular — the yaml language package + basic keymap gives us
 * syntax highlighting, folding, bracket matching, undo/redo, and line
 * numbers at about 50 KB. No web worker overhead, no IntelliSense UI,
 * no VSCode feeling bolted onto the page.
 *
 * This component is also used as a read-only viewer (pass
 * `readOnly`) so snapshots in the Versions tab get the same gutter +
 * fold affordances as the live editor — previously they used a plain
 * <pre>, which was cheaper to render but less navigable for long
 * configs.
 */

// Highlight style — values in white, only numbers and booleans carry
// accent hues so the colour tells the reader "this is a literal type"
// at a glance without turning every line into a rainbow. Plain-scalar
// strings (`qlora`, `all-linear`, bare words) render as neutral ink
// like any other value.
//
// Tag coverage:
//   propertyName → brand (pink), weight 500, anchors "this is a key"
//   number       → brand-alt (violet), numeric literal
//   bool         → info (sky blue), truthy literal (distinct from
//                  numbers on purpose — reading YAML on a dark panel,
//                  having two literals in the same hue was
//                  confusing).
//   keyword      → err (red), only `null`, `~`, errors
//   comment      → ink-4 italic, out of the way
//   everything else (string, name, content, literal, punctuation,
//   operator) falls back to default `color: #fafafa` so values read
//   as plain white text.
const yamlHighlight = HighlightStyle.define([
  { tag: t.propertyName, color: '#ed487f', fontWeight: '500' },
  { tag: t.number, color: '#b8a1fb' },
  { tag: t.bool, color: '#60a5fa' },
  { tag: t.keyword, color: '#f87171' },
  { tag: t.comment, color: '#71717a', fontStyle: 'italic' },
])

// ---------------------------------------------------------------------------
// Scalar-value decorator.
//
// `@codemirror/lang-yaml` only tags KEYS (propertyName). Bare scalars on the
// right-hand side — `16`, `false`, `qlora`, `0.05` — come through the
// parser as untagged text, so `HighlightStyle` has nothing to colour. We
// attach a viewport-scoped MatchDecorator that regex-matches the whole
// value after `: ` and classifies it into four buckets mapped to CSS
// classes. Palette follows the conventional Dracula / One-Dark mapping
// for readability (strings = yellow, numbers = orange, booleans =
// cyan, null = red, keys = brand pink).
//
// Single regex with one capture group; the JS callback inspects the
// captured text to decide the type. Alternation-in-regex with a
// fall-through would work too, but the JS classifier is easier to
// extend (e.g. add IP-addr detection later).
// ---------------------------------------------------------------------------
const VALUE_RE =
  /(?<=:\s)(null|~|true|false|True|False|TRUE|FALSE|yes|no|on|off|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|[^\s#][^\n#]*?)(?=\s*(?:#|$))/g

const BOOL_RE = /^(true|false|True|False|TRUE|FALSE|yes|no|on|off)$/
const NUM_RE = /^-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?$/

const valueMatcher = new MatchDecorator({
  regexp: VALUE_RE,
  decoration: (match) => {
    const v = match[1]
    if (v === 'null' || v === '~') return Decoration.mark({ class: 'cm-v-null' })
    if (BOOL_RE.test(v)) return Decoration.mark({ class: 'cm-v-bool' })
    if (NUM_RE.test(v)) return Decoration.mark({ class: 'cm-v-num' })
    return Decoration.mark({ class: 'cm-v-str' })
  },
})

const valueDecorations = ViewPlugin.fromClass(
  class {
    decorations: DecorationSet
    constructor(view: EditorView) {
      this.decorations = valueMatcher.createDeco(view)
    }
    update(update: ViewUpdate) {
      if (update.docChanged || update.viewportChanged) {
        this.decorations = valueMatcher.updateDeco(update, this.decorations)
      }
    }
  },
  { decorations: (p) => p.decorations },
)

const editorTheme = EditorView.theme(
  {
    '&': {
      backgroundColor: 'transparent',
      color: '#fafafa',
      // 12 px felt slightly cramped when shipped; 12.5 with a
      // proportional line-height lands in a comfortable range for
      // long config files and leaves room for more rows in viewport.
      fontSize: '12.5px',
      fontFamily: '"JetBrains Mono", ui-monospace, Menlo, monospace',
    },
    '.cm-content': {
      caretColor: '#fafafa',
      padding: '12px 0',
      lineHeight: '1.45',
    },
    '.cm-gutters': {
      backgroundColor: 'transparent',
      borderRight: '1px solid #2c3036',
      color: '#71717a',
      fontSize: '11px',
    },
    '.cm-activeLineGutter': {
      backgroundColor: 'transparent',
      color: '#d4d4d8',
    },
    '.cm-activeLine': {
      backgroundColor: 'rgba(255,255,255,0.02)',
    },
    // Fold gutter: default chevrons are ~8px — too small to click. We
    // widen the hit area to 18px, brighten the colour, and kick the
    // triangle to full brand on hover so the target is unambiguous.
    '.cm-foldGutter': {
      minWidth: '18px',
      padding: '0 2px',
    },
    '.cm-foldGutter .cm-gutterElement': {
      cursor: 'pointer',
      color: '#a1a1aa',
      fontSize: '13px',
      lineHeight: '1.5rem',
      textAlign: 'center',
      width: '18px',
      transition: 'color 0.12s ease, background-color 0.12s ease',
      borderRadius: '3px',
    },
    '.cm-foldGutter .cm-gutterElement:hover': {
      color: '#ed487f',
      backgroundColor: 'rgba(237, 72, 127, 0.14)',
    },
    // Value-literal decorations — palette follows the conventional
    // Dracula / One-Dark mapping (researched, see YamlEditor file
    // header). Strings yellow so they register as "literal text",
    // numbers orange so they pop against the muted greys, booleans
    // cyan so true/false are instantly scannable, nulls red as an
    // "absent/missing" warning.
    '.cm-v-str': {
      color: '#f1fa8c', // Dracula yellow
    },
    '.cm-v-num': {
      color: '#ffb86c', // Dracula orange
    },
    '.cm-v-bool': {
      color: '#8be9fd', // Dracula cyan (distinct from numbers)
    },
    '.cm-v-null': {
      color: '#ff5555', // Dracula red
      fontStyle: 'italic',
    },
    '.cm-cursor': {
      borderLeftColor: '#fafafa',
    },
    '.cm-selectionBackground, .cm-content ::selection': {
      backgroundColor: 'rgba(237, 72, 127, 0.28) !important',
    },
    '.cm-focused .cm-selectionBackground, .cm-focused .cm-content ::selection': {
      backgroundColor: 'rgba(237, 72, 127, 0.38) !important',
    },
  },
  { dark: true },
)

interface Props {
  value: string
  onChange?: (next: string) => void
  onBlur?: () => void
  maxHeight?: string
  /** Extra classes applied to the outermost wrapper. Useful when the
   *  editor sits inside a flex column and needs `flex-1 min-h-0` to
   *  absorb remaining height. */
  className?: string
  /** If true, disables editing but keeps gutters, folding, selection,
   *  and all the navigation affordances of the live editor. */
  readOnly?: boolean
  /** If true, renders the top toolbar with Collapse all / Expand all.
   *  Default true; version-preview callers can turn it off. */
  showToolbar?: boolean
  /** Optional extra content shown to the right of the toolbar (e.g.
   *  "preview: v21" label for the Versions tab). */
  toolbarExtra?: React.ReactNode
}

export function YamlEditor({
  value,
  onChange,
  onBlur,
  maxHeight = 'max-h-[420px]',
  className = '',
  readOnly = false,
  showToolbar = true,
  toolbarExtra,
}: Props) {
  const hostRef = useRef<HTMLDivElement | null>(null)
  const viewRef = useRef<EditorView | null>(null)
  // Keep refs to the latest handlers so the extension compartment can
  // close over the current callback without re-creating the view.
  const onChangeRef = useRef(onChange)
  onChangeRef.current = onChange
  const onBlurRef = useRef(onBlur)
  onBlurRef.current = onBlur

  useEffect(() => {
    if (!hostRef.current) return
    const extensions = [
      lineNumbers(),
      foldGutter({
        markerDOM: (open) => {
          const span = document.createElement('span')
          span.textContent = open ? '▾' : '▸'
          span.setAttribute('aria-hidden', 'true')
          return span
        },
      }),
      history(),
      indentOnInput(),
      bracketMatching(),
      highlightActiveLine(),
      yaml(),
      syntaxHighlighting(yamlHighlight),
      valueDecorations,
      editorTheme,
      keymap.of([...defaultKeymap, ...historyKeymap, ...foldKeymap]),
      EditorView.updateListener.of((u) => {
        if (u.docChanged && onChangeRef.current) {
          onChangeRef.current(u.state.doc.toString())
        }
      }),
      EditorView.domEventHandlers({
        blur: () => onBlurRef.current?.(),
      }),
      EditorView.editable.of(!readOnly),
      EditorState.readOnly.of(readOnly),
    ]
    const state = EditorState.create({ doc: value, extensions })
    const view = new EditorView({ state, parent: hostRef.current })
    viewRef.current = view
    return () => {
      view.destroy()
      viewRef.current = null
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [readOnly])

  // External value changes (preset load, YAML swap from form, version
  // selection) sync into the editor without destroying the view — this
  // keeps the undo stack, cursor position, and focus intact. Guard
  // against the no-op case so we don't loop on our own onChange.
  useEffect(() => {
    const view = viewRef.current
    if (!view) return
    const current = view.state.doc.toString()
    if (current === value) return
    view.dispatch({
      changes: { from: 0, to: current.length, insert: value },
    })
  }, [value])

  // Imperative fold controls for the toolbar. CodeMirror ships them as
  // StateCommands so they hook straight into the current view.
  const foldEverything = useCallback(() => {
    const view = viewRef.current
    if (!view) return
    foldAll(view)
  }, [])
  const unfoldEverything = useCallback(() => {
    const view = viewRef.current
    if (!view) return
    unfoldAll(view)
  }, [])

  // "Expand" toggle. This is NOT browser-level fullscreen — user
  // wants a floating panel with a bit of padding from the viewport
  // edges and the app chrome (sidebar, topbar) still visible but
  // blurred behind. Pure CSS overlay: `fixed inset-8` for the editor,
  // `fixed inset-0 backdrop-blur` for the dimmer. Esc closes it.
  // CodeMirror stays mounted throughout — we only flip the wrapper's
  // classes, so cursor, selection, fold, undo all persist.
  const wrapperRef = useRef<HTMLDivElement | null>(null)
  const [fullscreen, setFullscreen] = useState(false)
  const [closing, setClosing] = useState(false)
  const enterFullscreen = useCallback(() => setFullscreen(true), [])
  const exitFullscreen = useCallback(() => {
    setClosing(true)
    window.setTimeout(() => {
      setFullscreen(false)
      setClosing(false)
    }, 160)
  }, [])

  // Esc closes, and we lock body scroll so the dimmed background
  // doesn't jitter under the overlay.
  useEffect(() => {
    if (!fullscreen) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') exitFullscreen()
    }
    document.addEventListener('keydown', onKey)
    const prevOverflow = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    return () => {
      document.removeEventListener('keydown', onKey)
      document.body.style.overflow = prevOverflow
    }
  }, [fullscreen, exitFullscreen])

  // Ask CodeMirror to remeasure after the wrapper changes size.
  useEffect(() => {
    const view = viewRef.current
    if (!view) return
    const handle = requestAnimationFrame(() => view.requestMeasure())
    return () => cancelAnimationFrame(handle)
  }, [fullscreen])

  // Wrapper classes: normal flow by default, CSS overlay when
  // expanded. The overlay sits at `inset-8` (~32 px from every
  // viewport edge) with rounded-xl + shadow, matching the modal
  // feel user asked for. z-9999 sits atop every other UI element in
  // this page — sidebar + topbar stay visible but blurred behind
  // the backdrop. `!fixed` overrides the `relative` from `baseCls`
  // (both Tailwind utilities in the same layer would otherwise race
  // on source order).
  const baseCls =
    'relative rounded-md border border-line-1 bg-surface-0 overflow-hidden flex flex-col'
  const wrapperCls = fullscreen
    ? `${baseCls} !fixed inset-6 lg:inset-10 z-[9999] !rounded-xl shadow-card ${
        closing ? 'fs-exit' : 'fs-enter'
      }`
    : `${baseCls} ${className}`

  return (
    <>
      {fullscreen && (
        <div
          aria-hidden="true"
          onClick={exitFullscreen}
          className={[
            'fixed inset-0 z-[9998] bg-black/55 backdrop-blur-md',
            closing ? 'fs-backdrop-out' : 'fs-backdrop-in',
          ].join(' ')}
        />
      )}
      <div ref={wrapperRef} className={wrapperCls}>
        {showToolbar && (
          <div className="flex items-center gap-2 px-2 py-1.5 border-b border-line-1/60 bg-surface-1/50 text-2xs">
            <button
              type="button"
              onClick={foldEverything}
              title="Collapse all blocks (fold every top-level key)"
              className="rounded px-2 py-0.5 text-ink-3 hover:text-ink-1 hover:bg-surface-3/50 transition inline-flex items-center gap-1"
            >
              <span aria-hidden>▸</span>
              Collapse all
            </button>
            <button
              type="button"
              onClick={unfoldEverything}
              title="Expand all folded blocks"
              className="rounded px-2 py-0.5 text-ink-3 hover:text-ink-1 hover:bg-surface-3/50 transition inline-flex items-center gap-1"
            >
              <span aria-hidden>▾</span>
              Expand all
            </button>
            {readOnly && (
              <span className="ml-1 inline-flex items-center gap-1 rounded-full border border-line-2 px-1.5 py-0.5 text-[0.6rem] uppercase tracking-wider text-ink-4">
                read-only
              </span>
            )}
            <div className="ml-auto flex items-center gap-2">
              {toolbarExtra}
              <button
                type="button"
                onClick={fullscreen ? exitFullscreen : enterFullscreen}
                title={
                  fullscreen
                    ? 'Exit fullscreen (Esc)'
                    : 'Open editor fullscreen'
                }
                aria-label={fullscreen ? 'Exit fullscreen' : 'Open fullscreen'}
                aria-pressed={fullscreen}
                className="rounded px-1.5 py-0.5 text-ink-3 hover:text-ink-1 hover:bg-surface-3/50 transition inline-flex items-center"
              >
                {fullscreen ? (
                  <CollapseIcon className="w-3.5 h-3.5" />
                ) : (
                  <ExpandIcon className="w-3.5 h-3.5" />
                )}
              </button>
            </div>
          </div>
        )}
        <div className={`overflow-y-auto min-h-0 flex-1 ${fullscreen ? 'max-h-full' : maxHeight}`}>
          <div ref={hostRef} />
        </div>
      </div>
    </>
  )
}
