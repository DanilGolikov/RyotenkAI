import { useMutation, useQueryClient } from '@tanstack/react-query'
import { api } from '../client'
import { qk } from '../queryKeys'
import type { DeleteResult } from '../types'

export function useDeleteRun() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (runId: string) =>
      api.del<DeleteResult>(`/runs/${encodeURIComponent(runId)}`, { mode: 'local_and_mlflow' }),
    onSuccess: () => qc.invalidateQueries({ queryKey: qk.runs() }),
  })
}
