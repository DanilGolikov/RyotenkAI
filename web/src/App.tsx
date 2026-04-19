import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { AppShell } from './components/AppShell'
import { Overview } from './pages/Overview'
import { RunsWorkspace } from './pages/RunsWorkspace'
import { LaunchPage } from './pages/LaunchPage'
import { ProjectsPage } from './pages/Projects'
import { ProjectDetailPage } from './pages/ProjectDetail'
import { SettingsPage } from './pages/Settings'
import { ProvidersPage } from './pages/Providers'
import { ProviderDetailPage } from './pages/ProviderDetail'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      refetchOnWindowFocus: false,
    },
  },
})

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          <Route element={<AppShell />}>
            <Route index element={<Overview />} />
            <Route path="runs" element={<RunsWorkspace />}>
              <Route path=":runId" element={<RunsWorkspace />} />
              <Route path=":runId/attempts/:attemptNo" element={<RunsWorkspace />} />
            </Route>
            <Route path="runs/:runId" element={<RunsWorkspace />} />
            <Route path="runs/:runId/attempts/:attemptNo" element={<RunsWorkspace />} />
            <Route path="launch" element={<LaunchPage />} />
            <Route path="projects" element={<ProjectsPage />} />
            <Route path="projects/:id/*" element={<ProjectDetailPage />} />
            <Route path="settings" element={<SettingsPage />}>
              <Route index element={<Navigate to="providers" replace />} />
              <Route path="providers" element={<ProvidersPage />} />
              <Route path="providers/:id/*" element={<ProviderDetailPage />} />
            </Route>
            <Route path="*" element={<Navigate to="/" replace />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  )
}
