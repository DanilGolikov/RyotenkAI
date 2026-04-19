import {
  useProviderConfigVersions,
  useReadProviderVersion,
  useRestoreProviderVersion,
} from '../../api/hooks/useProviders'
import { ConfigVersionsPanel } from '../ConfigVersionsPanel'

export function ProviderVersionsTab({ providerId }: { providerId: string }) {
  const versionsQuery = useProviderConfigVersions(providerId)
  const restoreMutation = useRestoreProviderVersion(providerId)
  return (
    <ConfigVersionsPanel
      versionsQuery={versionsQuery}
      useReadVersion={(filename) => useReadProviderVersion(providerId, filename)}
      restoreMutation={restoreMutation}
    />
  )
}
