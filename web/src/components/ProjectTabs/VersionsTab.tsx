import {
  useProjectConfigVersions,
  useReadConfigVersion,
  useRestoreConfigVersion,
} from '../../api/hooks/useProjects'
import { ConfigVersionsPanel } from '../ConfigVersionsPanel'

export function VersionsTab({ projectId }: { projectId: string }) {
  const versionsQuery = useProjectConfigVersions(projectId)
  const restoreMutation = useRestoreConfigVersion(projectId)
  return (
    <ConfigVersionsPanel
      versionsQuery={versionsQuery}
      useReadVersion={(filename) => useReadConfigVersion(projectId, filename)}
      restoreMutation={restoreMutation}
    />
  )
}
