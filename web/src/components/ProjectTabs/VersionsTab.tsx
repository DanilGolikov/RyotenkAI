import {
  useProjectConfigVersions,
  useReadConfigVersion,
  useRestoreConfigVersion,
  useToggleFavoriteVersion,
} from '../../api/hooks/useProjects'
import { ConfigVersionsPanel } from '../ConfigVersionsPanel'

export function VersionsTab({ projectId }: { projectId: string }) {
  const versionsQuery = useProjectConfigVersions(projectId)
  const restoreMutation = useRestoreConfigVersion(projectId)
  const favMut = useToggleFavoriteVersion(projectId)
  return (
    <ConfigVersionsPanel
      versionsQuery={versionsQuery}
      useReadVersion={(filename) => useReadConfigVersion(projectId, filename)}
      restoreMutation={restoreMutation}
      favorite={{
        pending: favMut.isPending,
        onToggle: (filename, favorite) => favMut.mutate({ filename, favorite }),
      }}
    />
  )
}
