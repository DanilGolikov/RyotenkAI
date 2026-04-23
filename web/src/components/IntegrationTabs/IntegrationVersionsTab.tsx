import {
  useIntegrationConfigVersions,
  useReadIntegrationVersion,
  useRestoreIntegrationVersion,
} from '../../api/hooks/useIntegrations'
import { ConfigVersionsPanel } from '../ConfigVersionsPanel'

export function IntegrationVersionsTab({ integrationId }: { integrationId: string }) {
  const versionsQuery = useIntegrationConfigVersions(integrationId)
  const restoreMutation = useRestoreIntegrationVersion(integrationId)
  return (
    <ConfigVersionsPanel
      versionsQuery={versionsQuery}
      useReadVersion={(filename) => useReadIntegrationVersion(integrationId, filename)}
      restoreMutation={restoreMutation}
    />
  )
}
