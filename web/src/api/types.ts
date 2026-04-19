// Hand-written types mirroring src/api/schemas/*.py. Run `npm run gen:api` once
// you add openapi-typescript to replace this file with generated output.

export type Status =
  | 'pending'
  | 'running'
  | 'completed'
  | 'failed'
  | 'interrupted'
  | 'stale'
  | 'skipped'
  | 'unknown'

export interface RunSummary {
  run_id: string
  run_dir: string
  created_at: string
  created_ts: number
  status: Status
  status_icon?: string | null
  status_color?: string | null
  attempts: number
  config_name: string
  mlflow_run_id?: string | null
  started_at?: string | null
  completed_at?: string | null
  duration_seconds?: number | null
  error?: string | null
  group: string
}

export interface RunsListResponse {
  runs_dir: string
  groups: Record<string, RunSummary[]>
}

export interface LineageRef {
  attempt_id: string
  stage_name: string
  outputs: Record<string, unknown>
}

export interface RunDetail {
  schema_version: number
  logical_run_id: string
  run_directory: string
  config_path: string
  config_abspath?: string | null
  active_attempt_id?: string | null
  pipeline_status: Status
  training_critical_config_hash: string
  late_stage_config_hash: string
  model_dataset_config_hash: string
  root_mlflow_run_id?: string | null
  mlflow_runtime_tracking_uri?: string | null
  mlflow_ca_bundle_path?: string | null
  attempts: AttemptSummary[]
  current_output_lineage: Record<string, LineageRef>
  status: Status
  status_icon?: string | null
  status_color?: string | null
  running_attempt_no?: number | null
  next_attempt_no: number
  is_locked: boolean
  lock_pid?: number | null
}

export interface AttemptSummary {
  attempt_id: string
  attempt_no: number
  runtime_name: string
  requested_action: string
  effective_action: string
  restart_from_stage?: string | null
  status: Status
  started_at: string
  completed_at?: string | null
  error?: string | null
  training_critical_config_hash: string
  late_stage_config_hash: string
  model_dataset_config_hash: string
  root_mlflow_run_id?: string | null
  pipeline_attempt_mlflow_run_id?: string | null
  training_run_id?: string | null
  enabled_stage_names: string[]
  stage_runs: Record<string, unknown>
}

export interface StageRun {
  stage_name: string
  status: Status
  status_icon?: string | null
  status_color?: string | null
  execution_mode?: string | null
  mode_label?: string | null
  outputs: Record<string, unknown>
  error?: string | null
  failure_kind?: string | null
  reuse_from?: Record<string, unknown> | null
  skip_reason?: string | null
  started_at?: string | null
  completed_at?: string | null
  duration_seconds?: number | null
}

export interface AttemptDetail extends Omit<AttemptSummary, 'stage_runs'> {
  stage_runs: Record<string, StageRun>
  duration_seconds?: number | null
  status_icon?: string | null
  status_color?: string | null
}

export interface StagesResponse {
  stages: StageRun[]
}

export interface LogChunk {
  file: string
  offset: number
  next_offset: number
  eof: boolean
  content: string
}

export interface LogFileInfo {
  name: string
  path: string
  size_bytes: number
  exists: boolean
}

export type LaunchMode = 'new_run' | 'fresh' | 'resume' | 'restart'

export interface LaunchRequestBody {
  mode: LaunchMode
  config_path?: string | null
  restart_from_stage?: string | null
  log_level?: 'INFO' | 'DEBUG'
}

export interface LaunchResponse {
  pid: number
  launched_at: string
  command: string[]
  launcher_log: string
  run_dir: string
}

export interface InterruptResponse {
  interrupted: boolean
  pid?: number | null
  reason?: string | null
}

export interface RestartPoint {
  stage: string
  available: boolean
  mode: string
  reason: string
}

export interface RestartPointsResponse {
  config_path: string
  points: RestartPoint[]
}

export interface DeleteIssue {
  run_dir: string
  phase: string
  message: string
}

export interface DeleteResult {
  target: string
  run_dirs: string[]
  deleted_mlflow_run_ids: string[]
  local_deleted: boolean
  issues: DeleteIssue[]
  is_success: boolean
}

export interface ReportResponse {
  path: string
  markdown: string
  generated_at: string
  regenerated: boolean
}

export interface ConfigCheck {
  label: string
  status: 'ok' | 'warn' | 'fail'
  detail: string
}

export interface ConfigValidationResult {
  ok: boolean
  config_path: string
  checks: ConfigCheck[]
}

export interface HealthStatus {
  status: 'ok' | 'degraded'
  runs_dir: string
  runs_dir_readable: boolean
  version: string
}

// ───────── Projects ─────────

export interface ProjectSummary {
  id: string
  name: string
  path: string
  created_at: string
  description: string
}

export interface ProjectDetail extends ProjectSummary {
  description: string
  updated_at: string
  current_config_yaml: string
}

export interface CreateProjectRequest {
  name: string
  id?: string
  path?: string
  description?: string
}

export interface SaveConfigRequest {
  yaml: string
}

export interface SaveConfigResponse {
  ok: boolean
  snapshot_filename: string | null
}

export interface ConfigResponse {
  yaml: string
  parsed_json: Record<string, unknown> | null
}

export interface ConfigVersion {
  filename: string
  created_at: string
  size_bytes: number
  is_favorite?: boolean
}

export interface ConfigVersionsResponse {
  versions: ConfigVersion[]
}

export interface ConfigVersionDetail {
  filename: string
  yaml: string
}

// ───────── Plugins ─────────

export type PluginKind = 'reward' | 'validation' | 'evaluation'

export interface PluginManifest {
  id: string
  name: string
  version: string
  priority: number
  description: string
  category: string
  stability: string
  params_schema: Record<string, unknown>
  thresholds_schema: Record<string, unknown>
  suggested_params: Record<string, unknown>
  suggested_thresholds: Record<string, unknown>
}

export interface PluginListResponse {
  kind: PluginKind
  plugins: PluginManifest[]
}

// ───────── Providers (reusable workspaces) ─────────

export interface ProviderTypeInfo {
  id: string
  label: string
  json_schema: Record<string, unknown>
}

export interface ProviderTypesResponse {
  types: ProviderTypeInfo[]
}

export interface ProviderSummary {
  id: string
  name: string
  type: string
  path: string
  created_at: string
  description: string
}

export interface ProviderDetail extends ProviderSummary {
  updated_at: string
  current_config_yaml: string
}

export interface CreateProviderRequest {
  name: string
  type: string
  id?: string
  path?: string
  description?: string
}

// ───────── Config presets ─────────

export interface ConfigPreset {
  name: string
  description: string
  yaml: string
}

export interface ConfigPresetsResponse {
  presets: ConfigPreset[]
}
