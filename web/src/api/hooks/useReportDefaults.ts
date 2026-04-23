import { useQuery } from '@tanstack/react-query'
import { api } from '../client'

interface ReportDefaultsResponse {
  sections: string[]
}

/** Fetches the built-in report section order the backend uses when
 *  ``reports.sections`` is unset in the pipeline config. The frontend
 *  uses this to pre-fill the Reports section in the Plugins tab so the
 *  user sees what will actually render by default — without silently
 *  writing the list to YAML until they reorder or remove a section. */
export function useReportDefaults() {
  return useQuery({
    queryKey: ['plugins', 'reports', 'defaults'],
    queryFn: () => api.get<ReportDefaultsResponse>('/plugins/reports/defaults'),
    staleTime: 60 * 60 * 1000, // defaults change with code releases, not at runtime
  })
}
