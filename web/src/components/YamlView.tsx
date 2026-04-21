import { YamlEditor } from './YamlEditor'

interface Props {
  text: string
  className?: string
  maxHeight?: string
  /** Render the Collapse-all/Expand-all toolbar. Default true — version
   *  previews are long and fold gutters are a key navigation aid. */
  showToolbar?: boolean
  /** Extra content in the toolbar (e.g. version label). */
  toolbarExtra?: React.ReactNode
}

/**
 * Read-only YAML display. Delegates to `YamlEditor` in read-only mode
 * so snapshots, version previews, and config dumps get the same
 * gutters, line numbers, fold gutter, and syntax highlighting as the
 * editable form. Previously this was a plain `<pre>` with a hand-
 * rolled tokenizer, which meant version previews missed all the
 * navigation affordances of the live editor.
 */
export function YamlView({
  text,
  className = '',
  maxHeight = 'max-h-[520px]',
  showToolbar = true,
  toolbarExtra,
}: Props) {
  return (
    <YamlEditor
      value={text}
      readOnly
      maxHeight={maxHeight}
      className={className}
      showToolbar={showToolbar}
      toolbarExtra={toolbarExtra}
    />
  )
}
