/**
 * DeleteProjectModal — phase-6 representative "modal/dialog" story.
 *
 * Three variants: idle, pending (save in flight), long-id project that
 * stresses the confirm-text gate. Modal renders into the same root as
 * the story canvas (no portal needed — it's a plain absolutely-
 * positioned overlay).
 */

import type { Meta, StoryObj } from '@storybook/react'
import { DeleteProjectModal } from './DeleteProjectModal'
import type { ProjectSummary } from '../api/types'

const FIXED_TS = '2026-05-01T09:30:00Z'

const baseProject: ProjectSummary = {
  id: 'sapo-grpo-research',
  name: 'SAPO vs GRPO research',
  path: '/projects/sapo-grpo-research',
  created_at: FIXED_TS,
} as ProjectSummary

const meta: Meta<typeof DeleteProjectModal> = {
  title: 'Modals/DeleteProjectModal',
  component: DeleteProjectModal,
  parameters: {
    layout: 'fullscreen',
  },
}
export default meta
type Story = StoryObj<typeof DeleteProjectModal>

export const Default: Story = {
  args: {
    project: baseProject,
    onClose: () => {},
    onConfirm: () => {},
    pending: false,
  },
}

export const Pending: Story = {
  args: {
    project: baseProject,
    onClose: () => {},
    onConfirm: () => {},
    pending: true,
  },
}

export const LongProjectId: Story = {
  args: {
    project: {
      ...baseProject,
      id: 'extremely-long-project-id-that-tests-input-truncation-2026-05-10',
      name: 'Edge case: long id stress test',
    },
    onClose: () => {},
    onConfirm: () => {},
    pending: false,
  },
}
