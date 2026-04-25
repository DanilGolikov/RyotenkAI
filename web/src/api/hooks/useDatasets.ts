/**
 * Hooks for the Datasets project tab.
 *
 *   - useDatasetsList(parsed)         — pure derivation from the project
 *                                       config (no network call). Returns
 *                                       ordered list of dataset entries.
 *   - useDatasetPathCheck(...)        — does the file/HF repo exist?
 *   - useDatasetPreview(...)          — paginated rows via useInfiniteQuery
 *   - useDatasetValidation(...)       — one-shot mutation, refreshes
 *                                       path-check on success
 */

import {
  useInfiniteQuery,
  useMutation,
  useQuery,
  useQueryClient,
} from '@tanstack/react-query'
import { useMemo } from 'react'
import { api } from '../client'
import { qk } from '../queryKeys'
import type {
  DatasetPathCheckResponse,
  DatasetPreviewResponse,
  DatasetSplit,
  DatasetValidateRequest,
  DatasetValidateResponse,
} from '../types'

const DEFAULT_PAGE_SIZE = 50

export interface DatasetEntry {
  /** Map key under `parsed.datasets` — the canonical reference name. */
  key: string
  sourceType: 'local' | 'huggingface'
  /** True when added by the auto-coupling logic (created together with a
   *  strategy). The UI shows this differently in confirm-delete prompts. */
  autoCreated: boolean
  trainPath: string
  evalPath: string | null
  hasEvalSplit: boolean
}

/**
 * Read-only derivation from the parsed project config. We keep this as
 * a hook so it composes with React's render cycle (memoised), but no
 * network call is made — the YAML is already fetched by the
 * ConfigTab/useProjectConfig path and we just project the relevant
 * subset.
 */
export function useDatasetsList(parsed: Record<string, unknown> | null | undefined): DatasetEntry[] {
  return useMemo(() => {
    if (!parsed || typeof parsed !== 'object') return []
    const block = (parsed as Record<string, unknown>).datasets
    if (!block || typeof block !== 'object' || Array.isArray(block)) return []
    const entries: DatasetEntry[] = []
    for (const [key, raw] of Object.entries(block as Record<string, unknown>)) {
      if (!raw || typeof raw !== 'object') continue
      const r = raw as Record<string, unknown>
      const sourceType = inferSourceType(r)
      entries.push({
        key,
        sourceType,
        autoCreated: r.auto_created === true,
        trainPath: extractTrainPath(r, sourceType),
        evalPath: extractEvalPath(r, sourceType),
        hasEvalSplit: Boolean(extractEvalPath(r, sourceType)),
      })
    }
    return entries
  }, [parsed])
}

function inferSourceType(r: Record<string, unknown>): 'local' | 'huggingface' {
  const explicit = r.source_type
  if (explicit === 'local' || explicit === 'huggingface') return explicit
  return r.source_hf ? 'huggingface' : 'local'
}

function extractTrainPath(r: Record<string, unknown>, t: 'local' | 'huggingface'): string {
  if (t === 'huggingface') {
    const hf = r.source_hf as Record<string, unknown> | undefined
    return typeof hf?.train_id === 'string' ? hf.train_id : ''
  }
  const local = r.source_local as Record<string, unknown> | undefined
  const paths = local?.local_paths as Record<string, unknown> | undefined
  return typeof paths?.train === 'string' ? paths.train : ''
}

function extractEvalPath(r: Record<string, unknown>, t: 'local' | 'huggingface'): string | null {
  if (t === 'huggingface') {
    const hf = r.source_hf as Record<string, unknown> | undefined
    const v = hf?.eval_id
    return typeof v === 'string' && v ? v : null
  }
  const local = r.source_local as Record<string, unknown> | undefined
  const paths = local?.local_paths as Record<string, unknown> | undefined
  const v = paths?.eval
  return typeof v === 'string' && v ? v : null
}

// ---------------------------------------------------------------------------
// path-check (lightweight existence + line-count probe)
// ---------------------------------------------------------------------------

export function useDatasetPathCheck(
  projectId: string,
  datasetKey: string | null,
  enabled = true,
) {
  return useQuery({
    queryKey: datasetKey ? qk.datasetPathCheck(projectId, datasetKey) : ['dataset-path-check', 'disabled'],
    queryFn: () =>
      api.get<DatasetPathCheckResponse>(
        `/projects/${encodeURIComponent(projectId)}/datasets/${encodeURIComponent(datasetKey!)}/path-check`,
      ),
    enabled: enabled && !!datasetKey,
    // Path/HF reachability isn't free (HEAD request). 60s is enough for
    // the common form-edit cycle while not making the user wait when
    // they tab back to the page.
    staleTime: 60 * 1000,
    refetchOnWindowFocus: true,
    retry: false,
  })
}

// ---------------------------------------------------------------------------
// preview (paginated)
// ---------------------------------------------------------------------------

export function useDatasetPreview(
  projectId: string,
  datasetKey: string | null,
  split: DatasetSplit,
  pageSize: number = DEFAULT_PAGE_SIZE,
) {
  return useInfiniteQuery({
    queryKey: datasetKey ? qk.datasetPreview(projectId, datasetKey, split) : ['dataset-preview', 'disabled'],
    enabled: !!datasetKey,
    initialPageParam: 0,
    queryFn: ({ pageParam }) =>
      api.get<DatasetPreviewResponse>(
        `/projects/${encodeURIComponent(projectId)}/datasets/${encodeURIComponent(datasetKey!)}/preview`,
        { split, offset: pageParam as number, limit: pageSize },
      ),
    getNextPageParam: (lastPage, allPages) =>
      lastPage.has_more ? allPages.length * pageSize : undefined,
    // Datasets don't change without an explicit save action, so the
    // first page can stay cached for a few minutes per session.
    staleTime: 5 * 60 * 1000,
    retry: 1,
  })
}

// ---------------------------------------------------------------------------
// one-off validation
// ---------------------------------------------------------------------------

export function useDatasetValidation(projectId: string, datasetKey: string) {
  const qc = useQueryClient()
  return useMutation({
    mutationKey: qk.datasetValidation(projectId, datasetKey),
    mutationFn: (body: DatasetValidateRequest) =>
      api.post<DatasetValidateResponse>(
        `/projects/${encodeURIComponent(projectId)}/datasets/${encodeURIComponent(datasetKey)}/validate`,
        body,
      ),
    onSuccess: () => {
      // path-check shows status — refresh so the list pill updates.
      qc.invalidateQueries({ queryKey: qk.datasetPathCheck(projectId, datasetKey) })
    },
  })
}
