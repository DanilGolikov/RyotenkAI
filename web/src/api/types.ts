// Thin re-exports over the generated OpenAPI schema (./schema.d.ts).
// Pydantic on the backend is the single source of truth — this file
// just gives components friendly names and a couple of UI-only enums
// that aren't emitted as dedicated OpenAPI schemas.
//
// Do not hand-author payload shapes here. If a type is missing, run
// `make gen-api` and add it as a re-export from `components['schemas']`.

import type { components } from './schema'

type S = components['schemas']

// ───────── UI-only unions (enumerated inline on the backend — no $def) ─────────

/** Pipeline / attempt / stage lifecycle status, including the UI-only
 *  `unknown` fallback that the frontend can produce while a response is
 *  in-flight. */
export type Status =
  | 'pending'
  | 'running'
  | 'completed'
  | 'failed'
  | 'interrupted'
  | 'stale'
  | 'skipped'
  | 'unknown'

export type LaunchMode = 'new_run' | 'fresh' | 'resume' | 'restart'

export type PluginKind = 'reward' | 'validation' | 'evaluation'

// ───────── Runs / attempts / stages ─────────
//
// Narrow `status: string` from OpenAPI to the Status union — the backend
// only ever emits the enum values but Pydantic `str` widens it. Same for
// fields typed with ``default_factory=list/dict`` in Pydantic that FastAPI
// reports as optional: at runtime they're always present, so the UI treats
// them as required.

type Narrow<T, K extends keyof T, V> = Omit<T, K> & { [P in K]-?: V }

export type StageRun = Narrow<S['StageRun'], 'status', Status>
export type RunSummary = Narrow<S['RunSummary'], 'status', Status>
export type RunsListResponse =
  Narrow<S['RunsListResponse'], 'groups', Record<string, RunSummary[]>>
export type RunDetail =
  Narrow<
    Narrow<S['RunDetail'], 'status', Status>,
    'attempts',
    AttemptSummary[]
  > & { pipeline_status: Status }
export type AttemptDetail =
  Narrow<
    Narrow<S['AttemptDetail'], 'status', Status>,
    'stage_runs',
    Record<string, StageRun>
  >
/** AttemptDetail minus the typed stage_runs map. */
export type AttemptSummary = Omit<AttemptDetail, 'stage_runs'>
export type StagesResponse = Narrow<S['StagesResponse'], 'stages', StageRun[]>
export type LineageRef = S['LineageRefSchema']

// ───────── Logs ─────────

export type LogChunk = S['LogChunk']
export type LogFileInfo = S['LogFileInfo']

// ───────── Launch / interrupt ─────────

export type LaunchRequestBody = S['LaunchRequestSchema']
export type LaunchResponse = S['LaunchResponse']
export type InterruptResponse = S['InterruptResponse']
export type RestartPoint = S['RestartPoint']
export type RestartPointsResponse =
  Narrow<S['RestartPointsResponse'], 'points', RestartPoint[]>

// ───────── Delete ─────────

export type DeleteIssue = S['DeleteIssueSchema']
export type DeleteResult = S['DeleteResultSchema']

// ───────── Reports ─────────

export type ReportResponse = S['ReportResponse']

// ───────── Config validation (shared envelope) ─────────

export type ConfigCheck = S['ConfigCheck']
export type ConfigValidationResult =
  Narrow<S['ConfigValidationResult'], 'checks', ConfigCheck[]> & {
    /** Backend-emitted field errors keyed by dotted Pydantic loc
     * (e.g. ``"training.strategies.0.strategy_type"``). Added 2026-04.
     * Reflect manually here until ``openapi-typescript`` is re-run. */
    field_errors?: Record<string, string[]>
  }

// ───────── Health ─────────

export type HealthStatus = S['HealthStatus']

// ───────── Projects ─────────

export type ProjectSummary = S['ProjectSummary']
export type ProjectDetail = S['ProjectDetail']
export type CreateProjectRequest = S['CreateProjectRequest']
export type SaveConfigRequest = S['SaveConfigRequest']
export type SaveConfigResponse = S['SaveConfigResponse']
export type ConfigResponse = S['ConfigResponse']
export type ConfigVersion = S['ConfigVersion']
export type ConfigVersionsResponse = S['ConfigVersionsResponse']
export type ConfigVersionDetail = S['ConfigVersionDetail']
export type ToggleFavoriteRequest = S['ToggleFavoriteRequest']
export type ToggleFavoriteResponse = S['ToggleFavoriteResponse']

// ───────── Plugins catalogue ─────────

export type PluginManifest =
  Narrow<
    Narrow<
      Narrow<
        Narrow<S['PluginManifest'], 'params_schema', Record<string, unknown>>,
        'thresholds_schema',
        Record<string, unknown>
      >,
      'suggested_params',
      Record<string, unknown>
    >,
    'suggested_thresholds',
    Record<string, unknown>
  >
export type PluginListResponse =
  Narrow<S['PluginListResponse'], 'plugins', PluginManifest[]>

// ───────── Providers (reusable workspaces) ─────────

export type ProviderTypeInfo = S['ProviderTypeInfo']
export type ProviderTypesResponse = S['ProviderTypesResponse']
// Augment with has_inference until the OpenAPI spec is regenerated. The
// field is populated by the backend in provider_service.list_summaries;
// frontend filters use it directly (e.g. InferenceProviderField).
export type ProviderSummary = S['ProviderSummary'] & { has_inference?: boolean }
export type ProviderDetail = S['ProviderDetail']
export type CreateProviderRequest = S['CreateProviderRequest']
export type ProviderConfigResponse = S['ProviderConfigResponse']
export type ProviderSaveConfigRequest = S['ProviderSaveConfigRequest']
export type ProviderSaveConfigResponse = S['ProviderSaveConfigResponse']
export type ProviderConfigVersion = S['ProviderConfigVersion']
export type ProviderConfigVersionsResponse = S['ProviderConfigVersionsResponse']
export type ProviderConfigVersionDetail = S['ProviderConfigVersionDetail']

// ───────── Config presets ─────────

export type ConfigPreset = S['ConfigPreset']
export type ConfigPresetsResponse = S['ConfigPresetsResponse']
