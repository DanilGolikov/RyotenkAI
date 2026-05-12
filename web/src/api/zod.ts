/* eslint-disable */
// AUTO-GENERATED — DO NOT EDIT BY HAND.
// Regenerate with: `make regen-zod` (or `cd web && npm run gen:zod`).
// Source: web/src/api/openapi.json (produced by scripts/sync_openapi.py).
import { makeApi, Zodios, type ZodiosOptions } from "@zodios/core";
import { z } from "zod";

const HealthStatus = z
  .object({
    status: z.enum(["ok", "degraded"]),
    runs_dir: z.string(),
    runs_dir_readable: z.boolean(),
    version: z.string().optional().default("v0.1.0"),
  })
  .passthrough();
const ConfigValidateRequest = z
  .object({ config_path: z.string() })
  .passthrough();
const ConfigCheck = z
  .object({
    label: z.string(),
    status: z.enum(["ok", "warn", "fail"]),
    detail: z.string().optional().default(""),
  })
  .passthrough();
const ConfigValidationResult = z
  .object({
    ok: z.boolean(),
    config_path: z.string(),
    checks: z.array(ConfigCheck).optional(),
    field_errors: z.record(z.array(z.string())).optional(),
  })
  .passthrough();
const ValidationError = z
  .object({
    loc: z.array(z.union([z.string(), z.number()])),
    msg: z.string(),
    type: z.string(),
  })
  .passthrough();
const HTTPValidationError = z
  .object({ detail: z.array(ValidationError) })
  .partial()
  .passthrough();
const ConfigTemplate = z
  .object({ name: z.string(), path: z.string() })
  .passthrough();
const DefaultConfigResponse = z
  .object({
    runs_dir: z.string(),
    config_templates: z.array(ConfigTemplate).optional(),
  })
  .passthrough();
const PresetScopeOut = z
  .object({ replaces: z.array(z.string()), preserves: z.array(z.string()) })
  .partial()
  .passthrough();
const PresetRequirementsOut = z
  .object({
    hub_models: z.array(z.string()),
    provider_kind: z.array(z.string()),
    required_plugins: z.array(z.string()),
    min_vram_gb: z.union([z.number(), z.null()]),
  })
  .partial()
  .passthrough();
const ConfigPreset = z
  .object({
    name: z.string(),
    display_name: z.string().optional().default(""),
    description: z.string().optional().default(""),
    yaml: z.string(),
    size_tier: z.string().optional().default(""),
    scope: z.union([PresetScopeOut, z.null()]).optional(),
    requirements: z.union([PresetRequirementsOut, z.null()]).optional(),
    placeholders: z.record(z.string()).optional(),
  })
  .passthrough();
const ConfigPresetsResponse = z
  .object({ presets: z.array(ConfigPreset) })
  .partial()
  .passthrough();
const PresetPreviewRequest = z
  .object({ current_config: z.object({}).partial().passthrough() })
  .partial()
  .passthrough();
const PresetDiffEntry = z
  .object({
    key: z.string(),
    kind: z.enum(["added", "removed", "changed", "unchanged"]),
    reason: z.enum([
      "preset_replaced",
      "preset_added",
      "preset_preserved",
      "no_scope",
    ]),
    before: z.unknown().optional(),
    after: z.unknown().optional(),
  })
  .passthrough();
const PresetRequirementCheck = z
  .object({
    label: z.string(),
    status: z.enum(["ok", "missing", "warning"]),
    detail: z.string().optional().default(""),
  })
  .passthrough();
const PresetPlaceholderHint = z
  .object({ path: z.string(), hint: z.string() })
  .passthrough();
const PresetPreviewResponse = z
  .object({
    resulting_config: z.object({}).partial().passthrough(),
    diff: z.array(PresetDiffEntry),
    requirements: z.array(PresetRequirementCheck),
    placeholders: z.array(PresetPlaceholderHint),
    warnings: z.array(z.string()).optional(),
  })
  .passthrough();
const StageRun = z
  .object({
    stage_name: z.string(),
    status: z.string(),
    status_icon: z.union([z.string(), z.null()]).optional(),
    status_color: z.union([z.string(), z.null()]).optional(),
    execution_mode: z.union([z.string(), z.null()]).optional(),
    mode_label: z.union([z.string(), z.null()]).optional(),
    outputs: z.object({}).partial().passthrough().optional(),
    error: z.union([z.string(), z.null()]).optional(),
    failure_kind: z.union([z.string(), z.null()]).optional(),
    reuse_from: z
      .union([z.object({}).partial().passthrough(), z.null()])
      .optional(),
    skip_reason: z.union([z.string(), z.null()]).optional(),
    started_at: z.union([z.string(), z.null()]).optional(),
    completed_at: z.union([z.string(), z.null()]).optional(),
    duration_seconds: z.union([z.number(), z.null()]).optional(),
  })
  .passthrough();
const AttemptDetail = z
  .object({
    attempt_id: z.string(),
    attempt_no: z.number().int(),
    runtime_name: z.string(),
    requested_action: z.string(),
    effective_action: z.string(),
    restart_from_stage: z.union([z.string(), z.null()]).optional(),
    status: z.string(),
    status_icon: z.union([z.string(), z.null()]).optional(),
    status_color: z.union([z.string(), z.null()]).optional(),
    started_at: z.string(),
    completed_at: z.union([z.string(), z.null()]).optional(),
    error: z.union([z.string(), z.null()]).optional(),
    training_critical_config_hash: z.string().optional().default(""),
    late_stage_config_hash: z.string().optional().default(""),
    model_dataset_config_hash: z.string().optional().default(""),
    root_mlflow_run_id: z.union([z.string(), z.null()]).optional(),
    pipeline_attempt_mlflow_run_id: z.union([z.string(), z.null()]).optional(),
    training_run_id: z.union([z.string(), z.null()]).optional(),
    enabled_stage_names: z.array(z.string()).optional(),
    stage_runs: z.record(StageRun).optional(),
    duration_seconds: z.union([z.number(), z.null()]).optional(),
  })
  .passthrough();
const StagesResponse = z
  .object({ stages: z.array(StageRun) })
  .partial()
  .passthrough();
const PreviewResponse = z
  .object({
    rows: z.array(z.object({}).partial().passthrough()),
    total: z.union([z.number(), z.null()]).optional(),
    has_more: z.boolean(),
    schema_hint: z.array(z.string()).optional(),
  })
  .passthrough();
const PathCheckSplit = z
  .object({
    exists: z.boolean(),
    line_count: z.union([z.number(), z.null()]).optional(),
    size_bytes: z.union([z.number(), z.null()]).optional(),
    error: z.union([z.string(), z.null()]).optional(),
  })
  .passthrough();
const PathCheckResponse = z
  .object({
    source_type: z.enum(["local", "huggingface"]),
    train: PathCheckSplit,
    eval: z.union([PathCheckSplit, z.null()]).optional(),
  })
  .passthrough();
const ValidateRequest = z
  .object({
    split: z.enum(["train", "eval"]).default("train"),
    max_samples: z.union([z.number(), z.null()]).default(1000),
  })
  .partial()
  .passthrough();
const FormatCheckPayload = z
  .object({
    strategy_type: z.string(),
    ok: z.boolean(),
    message: z.string().optional().default(""),
  })
  .passthrough();
const ErrorGroupPayload = z
  .object({
    error_type: z.string(),
    sample_indices: z.array(z.number().int()),
    total_count: z.number().int(),
  })
  .passthrough();
const PluginRunPayload = z
  .object({
    plugin_id: z.string(),
    plugin_name: z.string(),
    passed: z.boolean(),
    crashed: z.boolean().optional().default(false),
    duration_ms: z.number(),
    metrics: z.record(z.number()).optional(),
    warnings: z.array(z.string()).optional(),
    errors: z.array(z.string()).optional(),
    recommendations: z.array(z.string()).optional(),
    error_groups: z.array(ErrorGroupPayload).optional(),
  })
  .passthrough();
const ValidateResponse = z
  .object({
    duration_ms: z.number().int(),
    format_check: z.array(FormatCheckPayload),
    format_check_error: z.union([z.string(), z.null()]).optional(),
    plugin_results: z.array(PluginRunPayload),
  })
  .passthrough();
const IntegrationTypeInfo = z
  .object({
    id: z.string(),
    label: z.string(),
    requires_token: z.boolean(),
    json_schema: z.object({}).partial().passthrough(),
  })
  .passthrough();
const IntegrationTypesResponse = z
  .object({ types: z.array(IntegrationTypeInfo) })
  .passthrough();
const IntegrationSummary = z
  .object({
    id: z.string(),
    name: z.string(),
    type: z.string(),
    path: z.string(),
    created_at: z.string(),
    description: z.string().optional().default(""),
    has_token: z.boolean().optional().default(false),
  })
  .passthrough();
const CreateIntegrationRequest = z
  .object({
    name: z.string().min(1).max(80),
    type: z.string().min(1),
    id: z.union([z.string(), z.null()]).optional(),
    path: z.union([z.string(), z.null()]).optional(),
    description: z.string().optional().default(""),
  })
  .passthrough();
const IntegrationDetail = z
  .object({
    id: z.string(),
    name: z.string(),
    type: z.string(),
    path: z.string(),
    created_at: z.string(),
    description: z.string().optional().default(""),
    has_token: z.boolean().optional().default(false),
    updated_at: z.string(),
    current_config_yaml: z.string().optional().default(""),
  })
  .passthrough();
const IntegrationConfigResponse = z
  .object({
    yaml: z.string(),
    parsed_json: z
      .union([z.object({}).partial().passthrough(), z.null()])
      .optional(),
  })
  .passthrough();
const IntegrationSaveConfigRequest = z
  .object({ yaml: z.string() })
  .passthrough();
const IntegrationSaveConfigResponse = z
  .object({
    ok: z.boolean(),
    snapshot_filename: z.union([z.string(), z.null()]).optional(),
  })
  .passthrough();
const IntegrationConfigVersion = z
  .object({
    filename: z.string(),
    created_at: z.string(),
    size_bytes: z.number().int(),
  })
  .passthrough();
const IntegrationConfigVersionsResponse = z
  .object({ versions: z.array(IntegrationConfigVersion) })
  .passthrough();
const IntegrationConfigVersionDetail = z
  .object({ filename: z.string(), yaml: z.string() })
  .passthrough();
const IntegrationTokenRequest = z
  .object({ token: z.string().min(1) })
  .passthrough();
const ConnectionTestResult = z
  .object({
    ok: z.boolean(),
    latency_ms: z.union([z.number(), z.null()]).optional(),
    detail: z.string().optional().default(""),
  })
  .passthrough();
const LogChunk = z
  .object({
    file: z.string(),
    offset: z.number().int().gte(0),
    next_offset: z.number().int().gte(0),
    eof: z.boolean(),
    content: z.string(),
  })
  .passthrough();
const LogFileInfo = z
  .object({
    name: z.string(),
    path: z.string(),
    size_bytes: z.number().int(),
    exists: z.boolean(),
  })
  .passthrough();
const config_path = z.union([z.string(), z.null()]).optional();
const RestartPoint = z
  .object({
    stage: z.string(),
    available: z.boolean(),
    mode: z.string(),
    reason: z.string(),
  })
  .passthrough();
const RestartPointsResponse = z
  .object({ config_path: z.string(), points: z.array(RestartPoint).optional() })
  .passthrough();
const LaunchRequestSchema = z
  .object({
    mode: z.enum(["new_run", "fresh", "resume", "restart"]),
    config_path: z.union([z.string(), z.null()]).optional(),
    restart_from_stage: z.union([z.string(), z.null()]).optional(),
    log_level: z.enum(["INFO", "DEBUG"]).optional().default("INFO"),
  })
  .passthrough();
const LaunchResponse = z
  .object({
    pid: z.number().int(),
    launched_at: z.string(),
    command: z.array(z.string()),
    launcher_log: z.string(),
    run_dir: z.string(),
  })
  .passthrough();
const InterruptResponse = z
  .object({
    interrupted: z.boolean(),
    pid: z.union([z.number(), z.null()]).optional(),
    reason: z.union([z.string(), z.null()]).optional(),
  })
  .passthrough();
const ReportDefaultsResponse = z
  .object({ sections: z.array(z.string()) })
  .passthrough();
const PreflightRequest = z
  .object({
    config: z.object({}).partial().passthrough(),
    project_env: z.record(z.string()).optional(),
  })
  .passthrough();
const MissingEnvSchema = z
  .object({
    plugin_kind: z.enum(["reward", "validation", "evaluation", "reports"]),
    plugin_name: z.string(),
    plugin_instance_id: z.string(),
    name: z.string(),
    description: z.string().optional().default(""),
    secret: z.boolean().optional().default(true),
    managed_by: z.string().optional().default(""),
  })
  .passthrough();
const InstanceErrorSchema = z
  .object({
    plugin_kind: z.enum(["reward", "validation", "evaluation", "reports"]),
    plugin_name: z.string(),
    plugin_instance_id: z.string(),
    location: z.string(),
    message: z.string(),
  })
  .passthrough();
const PreflightResponse = z
  .object({
    ok: z.boolean(),
    missing: z.array(MissingEnvSchema).optional(),
    instance_errors: z.array(InstanceErrorSchema).optional(),
  })
  .passthrough();
const RequiredEnvSpec = z.object({
  name: z.string(),
  description: z.string().optional().default(""),
  optional: z.boolean().optional().default(false),
  secret: z.boolean().optional().default(true),
  managed_by: z.enum(["integrations", "providers", ""]).optional().default(""),
});
const LibRequirement = z.object({
  name: z.string(),
  version: z.string().optional().default(""),
});
const PluginManifest = z
  .object({
    schema_version: z.number().int().optional().default(5),
    id: z.string(),
    name: z.string(),
    version: z.string().optional().default("1.0.0"),
    description: z.string().optional().default(""),
    category: z.string().optional().default(""),
    stability: z.string().optional().default("stable"),
    kind: z.enum(["reward", "validation", "evaluation", "reports"]),
    supported_strategies: z.array(z.string()).optional(),
    author: z.string().optional().default(""),
    params_schema: z.object({}).partial().passthrough().optional(),
    thresholds_schema: z.object({}).partial().passthrough().optional(),
    suggested_params: z.object({}).partial().passthrough().optional(),
    suggested_thresholds: z.object({}).partial().passthrough().optional(),
    required_env: z.array(RequiredEnvSpec).optional(),
    lib_requirements: z.array(LibRequirement).optional(),
  })
  .passthrough();
const PluginLoadError = z
  .object({
    entry_name: z.string(),
    plugin_id: z.union([z.string(), z.null()]).optional(),
    error_type: z.string(),
    message: z.string(),
    traceback: z.string().optional().default(""),
  })
  .passthrough();
const PluginListResponse = z
  .object({
    kind: z.enum(["reward", "validation", "evaluation", "reports"]),
    plugins: z.array(PluginManifest),
    errors: z.array(PluginLoadError).optional(),
  })
  .passthrough();
const ProjectSummary = z
  .object({
    id: z.string(),
    name: z.string(),
    path: z.string(),
    created_at: z.string(),
    description: z.string().optional().default(""),
  })
  .passthrough();
const CreateProjectRequest = z
  .object({
    name: z.string().min(1).max(80),
    id: z.union([z.string(), z.null()]).optional(),
    path: z.union([z.string(), z.null()]).optional(),
    description: z.string().optional().default(""),
  })
  .passthrough();
const ProjectDetail = z
  .object({
    id: z.string(),
    name: z.string(),
    path: z.string(),
    description: z.string(),
    created_at: z.string(),
    updated_at: z.string(),
    current_config_yaml: z.string().optional().default(""),
  })
  .passthrough();
const UpdateProjectDescriptionRequest = z
  .object({ description: z.string().max(2000).default("") })
  .partial()
  .passthrough();
const StalePluginEntry = z
  .object({
    plugin_kind: z.string(),
    plugin_name: z.string(),
    instance_id: z.string(),
    location: z.string(),
  })
  .passthrough();
const ConfigResponse = z
  .object({
    yaml: z.string(),
    parsed_json: z
      .union([z.object({}).partial().passthrough(), z.null()])
      .optional(),
    stale_plugins: z.array(StalePluginEntry).optional().default([]),
  })
  .passthrough();
const SaveConfigRequest = z.object({ yaml: z.string() }).passthrough();
const SaveConfigResponse = z
  .object({
    ok: z.boolean(),
    snapshot_filename: z.union([z.string(), z.null()]).optional(),
  })
  .passthrough();
const ProjectEnvResponse = z
  .object({ env: z.record(z.string()).default({}) })
  .partial()
  .passthrough();
const ProjectEnvRequest = z.object({ env: z.record(z.string()) }).passthrough();
const ConfigVersion = z
  .object({
    filename: z.string(),
    created_at: z.string(),
    size_bytes: z.number().int(),
    is_favorite: z.boolean().optional().default(false),
  })
  .passthrough();
const ConfigVersionsResponse = z
  .object({ versions: z.array(ConfigVersion) })
  .passthrough();
const ConfigVersionDetail = z
  .object({ filename: z.string(), yaml: z.string() })
  .passthrough();
const ToggleFavoriteRequest = z.object({ favorite: z.boolean() }).passthrough();
const ToggleFavoriteResponse = z
  .object({ favorite_versions: z.array(z.string()) })
  .passthrough();
const limit = z.union([z.number(), z.null()]).optional();
const ProjectRunEntry = z
  .object({
    run_id: z.string(),
    started_at: z.string(),
    status: z.string(),
    finished_at: z.union([z.string(), z.null()]).optional(),
    mlflow_run_id: z.union([z.string(), z.null()]).optional(),
    config_version_hash: z.union([z.string(), z.null()]).optional(),
    actor: z.union([z.string(), z.null()]).optional(),
    run_directory: z.union([z.string(), z.null()]).optional(),
  })
  .passthrough();
const ProjectRunsResponse = z
  .object({ runs: z.array(ProjectRunEntry).default([]) })
  .partial()
  .passthrough();
const ProviderTypeInfo = z
  .object({
    id: z.string(),
    label: z.string(),
    json_schema: z.object({}).partial().passthrough(),
  })
  .passthrough();
const ProviderTypesResponse = z
  .object({ types: z.array(ProviderTypeInfo) })
  .passthrough();
const ProviderSummary = z
  .object({
    id: z.string(),
    name: z.string(),
    type: z.string(),
    path: z.string(),
    created_at: z.string(),
    description: z.string().optional().default(""),
    has_inference: z.boolean().optional().default(false),
    has_training: z.boolean().optional().default(false),
    has_token: z.boolean().optional().default(false),
  })
  .passthrough();
const CreateProviderRequest = z
  .object({
    name: z.string().min(1).max(80),
    type: z.string().min(1),
    id: z.union([z.string(), z.null()]).optional(),
    path: z.union([z.string(), z.null()]).optional(),
    description: z.string().optional().default(""),
  })
  .passthrough();
const ProviderDetail = z
  .object({
    id: z.string(),
    name: z.string(),
    type: z.string(),
    path: z.string(),
    created_at: z.string(),
    description: z.string().optional().default(""),
    has_inference: z.boolean().optional().default(false),
    has_training: z.boolean().optional().default(false),
    has_token: z.boolean().optional().default(false),
    updated_at: z.string(),
    current_config_yaml: z.string().optional().default(""),
  })
  .passthrough();
const ProviderConfigResponse = z
  .object({
    yaml: z.string(),
    parsed_json: z
      .union([z.object({}).partial().passthrough(), z.null()])
      .optional(),
  })
  .passthrough();
const ProviderSaveConfigRequest = z.object({ yaml: z.string() }).passthrough();
const ProviderSaveConfigResponse = z
  .object({
    ok: z.boolean(),
    snapshot_filename: z.union([z.string(), z.null()]).optional(),
  })
  .passthrough();
const ProviderConfigVersion = z
  .object({
    filename: z.string(),
    created_at: z.string(),
    size_bytes: z.number().int(),
  })
  .passthrough();
const ProviderConfigVersionsResponse = z
  .object({ versions: z.array(ProviderConfigVersion) })
  .passthrough();
const ProviderConfigVersionDetail = z
  .object({ filename: z.string(), yaml: z.string() })
  .passthrough();
const ReportResponse = z
  .object({
    path: z.string(),
    markdown: z.string(),
    generated_at: z.string(),
    regenerated: z.boolean().optional().default(false),
  })
  .passthrough();
const RunSummary = z
  .object({
    run_id: z.string(),
    run_dir: z.string(),
    created_at: z.string(),
    created_ts: z.number(),
    status: z.string(),
    status_icon: z.union([z.string(), z.null()]).optional(),
    status_color: z.union([z.string(), z.null()]).optional(),
    attempts: z.number().int(),
    config_name: z.string(),
    mlflow_run_id: z.union([z.string(), z.null()]).optional(),
    started_at: z.union([z.string(), z.null()]).optional(),
    completed_at: z.union([z.string(), z.null()]).optional(),
    duration_seconds: z.union([z.number(), z.null()]).optional(),
    error: z.union([z.string(), z.null()]).optional(),
    group: z.string().optional().default("(root)"),
    pod_status: z.union([z.string(), z.null()]).optional(),
  })
  .passthrough();
const RunsListResponse = z
  .object({
    runs_dir: z.string(),
    groups: z.record(z.array(RunSummary)).optional(),
  })
  .passthrough();
const CreateRunRequest = z
  .object({
    run_id: z.union([z.string(), z.null()]),
    subgroup: z.union([z.string(), z.null()]),
  })
  .partial()
  .passthrough();
const LineageRefSchema = z
  .object({
    attempt_id: z.string(),
    stage_name: z.string(),
    outputs: z.object({}).partial().passthrough().optional(),
  })
  .passthrough();
const RunDetail = z
  .object({
    schema_version: z.number().int(),
    logical_run_id: z.string(),
    run_directory: z.string(),
    config_path: z.string(),
    config_abspath: z.union([z.string(), z.null()]).optional(),
    active_attempt_id: z.union([z.string(), z.null()]).optional(),
    pipeline_status: z.string(),
    training_critical_config_hash: z.string().optional().default(""),
    late_stage_config_hash: z.string().optional().default(""),
    model_dataset_config_hash: z.string().optional().default(""),
    root_mlflow_run_id: z.union([z.string(), z.null()]).optional(),
    mlflow_runtime_tracking_uri: z.union([z.string(), z.null()]).optional(),
    mlflow_ca_bundle_path: z.union([z.string(), z.null()]).optional(),
    attempts: z.array(z.object({}).partial().passthrough()).optional(),
    current_output_lineage: z.record(LineageRefSchema).optional(),
    status: z.string(),
    status_icon: z.union([z.string(), z.null()]).optional(),
    status_color: z.union([z.string(), z.null()]).optional(),
    running_attempt_no: z.union([z.number(), z.null()]).optional(),
    next_attempt_no: z.number().int().optional().default(1),
    is_locked: z.boolean().optional().default(false),
    lock_pid: z.union([z.number(), z.null()]).optional(),
    pod_status: z.union([z.string(), z.null()]).optional(),
  })
  .passthrough();
const DeleteIssueSchema = z
  .object({ run_dir: z.string(), phase: z.string(), message: z.string() })
  .passthrough();
const DeleteResultSchema = z
  .object({
    target: z.string(),
    run_dirs: z.array(z.string()).optional(),
    deleted_mlflow_run_ids: z.array(z.string()).optional(),
    local_deleted: z.boolean(),
    issues: z.array(DeleteIssueSchema).optional(),
    is_success: z.boolean(),
  })
  .passthrough();

export const schemas = {
  HealthStatus,
  ConfigValidateRequest,
  ConfigCheck,
  ConfigValidationResult,
  ValidationError,
  HTTPValidationError,
  ConfigTemplate,
  DefaultConfigResponse,
  PresetScopeOut,
  PresetRequirementsOut,
  ConfigPreset,
  ConfigPresetsResponse,
  PresetPreviewRequest,
  PresetDiffEntry,
  PresetRequirementCheck,
  PresetPlaceholderHint,
  PresetPreviewResponse,
  StageRun,
  AttemptDetail,
  StagesResponse,
  PreviewResponse,
  PathCheckSplit,
  PathCheckResponse,
  ValidateRequest,
  FormatCheckPayload,
  ErrorGroupPayload,
  PluginRunPayload,
  ValidateResponse,
  IntegrationTypeInfo,
  IntegrationTypesResponse,
  IntegrationSummary,
  CreateIntegrationRequest,
  IntegrationDetail,
  IntegrationConfigResponse,
  IntegrationSaveConfigRequest,
  IntegrationSaveConfigResponse,
  IntegrationConfigVersion,
  IntegrationConfigVersionsResponse,
  IntegrationConfigVersionDetail,
  IntegrationTokenRequest,
  ConnectionTestResult,
  LogChunk,
  LogFileInfo,
  config_path,
  RestartPoint,
  RestartPointsResponse,
  LaunchRequestSchema,
  LaunchResponse,
  InterruptResponse,
  ReportDefaultsResponse,
  PreflightRequest,
  MissingEnvSchema,
  InstanceErrorSchema,
  PreflightResponse,
  RequiredEnvSpec,
  LibRequirement,
  PluginManifest,
  PluginLoadError,
  PluginListResponse,
  ProjectSummary,
  CreateProjectRequest,
  ProjectDetail,
  UpdateProjectDescriptionRequest,
  StalePluginEntry,
  ConfigResponse,
  SaveConfigRequest,
  SaveConfigResponse,
  ProjectEnvResponse,
  ProjectEnvRequest,
  ConfigVersion,
  ConfigVersionsResponse,
  ConfigVersionDetail,
  ToggleFavoriteRequest,
  ToggleFavoriteResponse,
  limit,
  ProjectRunEntry,
  ProjectRunsResponse,
  ProviderTypeInfo,
  ProviderTypesResponse,
  ProviderSummary,
  CreateProviderRequest,
  ProviderDetail,
  ProviderConfigResponse,
  ProviderSaveConfigRequest,
  ProviderSaveConfigResponse,
  ProviderConfigVersion,
  ProviderConfigVersionsResponse,
  ProviderConfigVersionDetail,
  ReportResponse,
  RunSummary,
  RunsListResponse,
  CreateRunRequest,
  LineageRefSchema,
  RunDetail,
  DeleteIssueSchema,
  DeleteResultSchema,
};

const endpoints = makeApi([
  {
    method: "get",
    path: "/api/v1/config/default",
    alias: "config-default",
    requestFormat: "json",
    response: DefaultConfigResponse,
  },
  {
    method: "get",
    path: "/api/v1/config/presets",
    alias: "config-presets",
    description: `Return curated starter configs from &#x60;&#x60;community/presets/&#x60;&#x60;.

Each preset lives in its own folder with a &#x60;&#x60;manifest.toml&#x60;&#x60; (id,
display name, description, size tier, optional v2 scope/requirements/
placeholders) and the actual config YAML referenced via
&#x60;&#x60;[preset.entry_point]&#x60;&#x60;.`,
    requestFormat: "json",
    response: ConfigPresetsResponse,
  },
  {
    method: "post",
    path: "/api/v1/config/presets/:preset_id/preview",
    alias: "config-preview_preset",
    description: `Dry-run: apply &#x60;&#x60;preset_id&#x60;&#x60; to &#x60;&#x60;current_config&#x60;&#x60; and return the
resulting config plus a structured diff, requirements check, and
placeholder hints.

The endpoint never writes anything — it&#x27;s what the frontend calls to
populate the Apply-preset modal (three sections: what changes / what&#x27;s
preserved / what the user still needs to fill).`,
    requestFormat: "json",
    parameters: [
      {
        name: "body",
        type: "Body",
        schema: z
          .object({ current_config: z.object({}).partial().passthrough() })
          .partial()
          .passthrough(),
      },
      {
        name: "preset_id",
        type: "Path",
        schema: z.string(),
      },
    ],
    response: PresetPreviewResponse,
    errors: [
      {
        status: 422,
        description: `Validation Error`,
        schema: HTTPValidationError,
      },
    ],
  },
  {
    method: "get",
    path: "/api/v1/config/schema",
    alias: "config-schema",
    description: `Return the full PipelineConfig JSON schema for the UI builder.`,
    requestFormat: "json",
    response: z.object({}).partial().passthrough(),
  },
  {
    method: "post",
    path: "/api/v1/config/validate",
    alias: "config-validate",
    requestFormat: "json",
    parameters: [
      {
        name: "body",
        type: "Body",
        schema: z.object({ config_path: z.string() }).passthrough(),
      },
    ],
    response: ConfigValidationResult,
    errors: [
      {
        status: 422,
        description: `Validation Error`,
        schema: HTTPValidationError,
      },
    ],
  },
  {
    method: "get",
    path: "/api/v1/health",
    alias: "health-health",
    requestFormat: "json",
    response: HealthStatus,
  },
  {
    method: "get",
    path: "/api/v1/integrations",
    alias: "integrations-list_integrations",
    requestFormat: "json",
    response: z.array(IntegrationSummary),
  },
  {
    method: "post",
    path: "/api/v1/integrations",
    alias: "integrations-create_integration",
    requestFormat: "json",
    parameters: [
      {
        name: "body",
        type: "Body",
        schema: CreateIntegrationRequest,
      },
    ],
    response: IntegrationSummary,
    errors: [
      {
        status: 422,
        description: `Validation Error`,
        schema: HTTPValidationError,
      },
    ],
  },
  {
    method: "get",
    path: "/api/v1/integrations/:integration_id",
    alias: "integrations-get_integration",
    requestFormat: "json",
    parameters: [
      {
        name: "integration_id",
        type: "Path",
        schema: z.string(),
      },
    ],
    response: IntegrationDetail,
    errors: [
      {
        status: 422,
        description: `Validation Error`,
        schema: HTTPValidationError,
      },
    ],
  },
  {
    method: "delete",
    path: "/api/v1/integrations/:integration_id",
    alias: "integrations-delete_integration",
    requestFormat: "json",
    parameters: [
      {
        name: "integration_id",
        type: "Path",
        schema: z.string(),
      },
    ],
    response: z.void(),
    errors: [
      {
        status: 422,
        description: `Validation Error`,
        schema: HTTPValidationError,
      },
    ],
  },
  {
    method: "get",
    path: "/api/v1/integrations/:integration_id/config",
    alias: "integrations-get_config",
    requestFormat: "json",
    parameters: [
      {
        name: "integration_id",
        type: "Path",
        schema: z.string(),
      },
    ],
    response: IntegrationConfigResponse,
    errors: [
      {
        status: 422,
        description: `Validation Error`,
        schema: HTTPValidationError,
      },
    ],
  },
  {
    method: "put",
    path: "/api/v1/integrations/:integration_id/config",
    alias: "integrations-save_config",
    requestFormat: "json",
    parameters: [
      {
        name: "body",
        type: "Body",
        schema: z.object({ yaml: z.string() }).passthrough(),
      },
      {
        name: "integration_id",
        type: "Path",
        schema: z.string(),
      },
    ],
    response: IntegrationSaveConfigResponse,
    errors: [
      {
        status: 422,
        description: `Validation Error`,
        schema: HTTPValidationError,
      },
    ],
  },
  {
    method: "post",
    path: "/api/v1/integrations/:integration_id/config/validate",
    alias: "integrations-validate_config",
    requestFormat: "json",
    parameters: [
      {
        name: "body",
        type: "Body",
        schema: z.object({ yaml: z.string() }).passthrough(),
      },
      {
        name: "integration_id",
        type: "Path",
        schema: z.string(),
      },
    ],
    response: ConfigValidationResult,
    errors: [
      {
        status: 422,
        description: `Validation Error`,
        schema: HTTPValidationError,
      },
    ],
  },
  {
    method: "get",
    path: "/api/v1/integrations/:integration_id/config/versions",
    alias: "integrations-list_versions",
    requestFormat: "json",
    parameters: [
      {
        name: "integration_id",
        type: "Path",
        schema: z.string(),
      },
    ],
    response: IntegrationConfigVersionsResponse,
    errors: [
      {
        status: 422,
        description: `Validation Error`,
        schema: HTTPValidationError,
      },
    ],
  },
  {
    method: "get",
    path: "/api/v1/integrations/:integration_id/config/versions/:filename",
    alias: "integrations-read_version",
    requestFormat: "json",
    parameters: [
      {
        name: "integration_id",
        type: "Path",
        schema: z.string(),
      },
      {
        name: "filename",
        type: "Path",
        schema: z.string(),
      },
    ],
    response: IntegrationConfigVersionDetail,
    errors: [
      {
        status: 422,
        description: `Validation Error`,
        schema: HTTPValidationError,
      },
    ],
  },
  {
    method: "post",
    path: "/api/v1/integrations/:integration_id/config/versions/:filename/restore",
    alias: "integrations-restore_version",
    requestFormat: "json",
    parameters: [
      {
        name: "integration_id",
        type: "Path",
        schema: z.string(),
      },
      {
        name: "filename",
        type: "Path",
        schema: z.string(),
      },
    ],
    response: IntegrationSaveConfigResponse,
    errors: [
      {
        status: 422,
        description: `Validation Error`,
        schema: HTTPValidationError,
      },
    ],
  },
  {
    method: "post",
    path: "/api/v1/integrations/:integration_id/test-connection",
    alias: "integrations-test_connection",
    requestFormat: "json",
    parameters: [
      {
        name: "integration_id",
        type: "Path",
        schema: z.string(),
      },
    ],
    response: ConnectionTestResult,
    errors: [
      {
        status: 422,
        description: `Validation Error`,
        schema: HTTPValidationError,
      },
    ],
  },
  {
    method: "put",
    path: "/api/v1/integrations/:integration_id/token",
    alias: "integrations-set_token",
    description: `Write-only — body is never echoed back. Responses contain no token.`,
    requestFormat: "json",
    parameters: [
      {
        name: "body",
        type: "Body",
        schema: z.object({ token: z.string().min(1) }).passthrough(),
      },
      {
        name: "integration_id",
        type: "Path",
        schema: z.string(),
      },
    ],
    response: z.void(),
    errors: [
      {
        status: 422,
        description: `Validation Error`,
        schema: HTTPValidationError,
      },
    ],
  },
  {
    method: "delete",
    path: "/api/v1/integrations/:integration_id/token",
    alias: "integrations-delete_token",
    requestFormat: "json",
    parameters: [
      {
        name: "integration_id",
        type: "Path",
        schema: z.string(),
      },
    ],
    response: z.void(),
    errors: [
      {
        status: 422,
        description: `Validation Error`,
        schema: HTTPValidationError,
      },
    ],
  },
  {
    method: "get",
    path: "/api/v1/integrations/types",
    alias: "integrations-list_types",
    requestFormat: "json",
    response: IntegrationTypesResponse,
  },
  {
    method: "get",
    path: "/api/v1/plugins/:kind",
    alias: "plugins-list_plugins",
    requestFormat: "json",
    parameters: [
      {
        name: "kind",
        type: "Path",
        schema: z.enum(["reward", "validation", "evaluation", "reports"]),
      },
    ],
    response: PluginListResponse,
    errors: [
      {
        status: 422,
        description: `Validation Error`,
        schema: HTTPValidationError,
      },
    ],
  },
  {
    method: "post",
    path: "/api/v1/plugins/preflight",
    alias: "plugins-preflight",
    description: `Check that every plugin in &#x60;&#x60;request.config&#x60;&#x60; has its non-optional
&#x60;&#x60;[[required_env]]&#x60;&#x60; keys set in process env / project env.

The Launch modal calls this before enabling the launch button so a
user without the right credentials sees a &quot;set up before launch&quot;
chip with a deep link to the right Settings tab — instead of a
pipeline that runs for minutes and crashes mid-stage on the missing
key.

Returns &#x60;&#x60;ok&#x3D;true&#x60;&#x60; plus an empty &#x60;&#x60;missing&#x60;&#x60; list when the launch
is safe; &#x60;&#x60;ok&#x3D;false&#x60;&#x60; with structured rows otherwise. A malformed
config payload surfaces as a 422 with the full per-field error
list (Pydantic does the work).`,
    requestFormat: "json",
    parameters: [
      {
        name: "body",
        type: "Body",
        schema: PreflightRequest,
      },
    ],
    response: PreflightResponse,
    errors: [
      {
        status: 422,
        description: `Validation Error`,
        schema: HTTPValidationError,
      },
    ],
  },
  {
    method: "get",
    path: "/api/v1/plugins/reports/defaults",
    alias: "plugins-get_report_defaults",
    requestFormat: "json",
    response: ReportDefaultsResponse,
  },
  {
    method: "get",
    path: "/api/v1/projects",
    alias: "projects-list_projects",
    requestFormat: "json",
    response: z.array(ProjectSummary),
  },
  {
    method: "post",
    path: "/api/v1/projects",
    alias: "projects-create_project",
    requestFormat: "json",
    parameters: [
      {
        name: "body",
        type: "Body",
        schema: CreateProjectRequest,
      },
    ],
    response: ProjectSummary,
    errors: [
      {
        status: 422,
        description: `Validation Error`,
        schema: HTTPValidationError,
      },
    ],
  },
  {
    method: "get",
    path: "/api/v1/projects/:project_id",
    alias: "projects-get_project",
    requestFormat: "json",
    parameters: [
      {
        name: "project_id",
        type: "Path",
        schema: z.string(),
      },
    ],
    response: ProjectDetail,
    errors: [
      {
        status: 422,
        description: `Validation Error`,
        schema: HTTPValidationError,
      },
    ],
  },
  {
    method: "delete",
    path: "/api/v1/projects/:project_id",
    alias: "projects-delete_project",
    description: `Unregister a project. By default also removes the on-disk
workspace — pass &#x60;&#x60;?delete_files&#x3D;false&#x60;&#x60; to keep the directory.`,
    requestFormat: "json",
    parameters: [
      {
        name: "project_id",
        type: "Path",
        schema: z.string(),
      },
      {
        name: "delete_files",
        type: "Query",
        schema: z.boolean().optional().default(true),
      },
    ],
    response: z.void(),
    errors: [
      {
        status: 422,
        description: `Validation Error`,
        schema: HTTPValidationError,
      },
    ],
  },
  {
    method: "get",
    path: "/api/v1/projects/:project_id/config",
    alias: "projects-get_config",
    requestFormat: "json",
    parameters: [
      {
        name: "project_id",
        type: "Path",
        schema: z.string(),
      },
    ],
    response: ConfigResponse,
    errors: [
      {
        status: 422,
        description: `Validation Error`,
        schema: HTTPValidationError,
      },
    ],
  },
  {
    method: "put",
    path: "/api/v1/projects/:project_id/config",
    alias: "projects-save_config",
    requestFormat: "json",
    parameters: [
      {
        name: "body",
        type: "Body",
        schema: z.object({ yaml: z.string() }).passthrough(),
      },
      {
        name: "project_id",
        type: "Path",
        schema: z.string(),
      },
    ],
    response: SaveConfigResponse,
    errors: [
      {
        status: 422,
        description: `Validation Error`,
        schema: HTTPValidationError,
      },
    ],
  },
  {
    method: "post",
    path: "/api/v1/projects/:project_id/config/validate",
    alias: "projects-validate_config",
    requestFormat: "json",
    parameters: [
      {
        name: "body",
        type: "Body",
        schema: z.object({ yaml: z.string() }).passthrough(),
      },
      {
        name: "project_id",
        type: "Path",
        schema: z.string(),
      },
    ],
    response: ConfigValidationResult,
    errors: [
      {
        status: 422,
        description: `Validation Error`,
        schema: HTTPValidationError,
      },
    ],
  },
  {
    method: "get",
    path: "/api/v1/projects/:project_id/config/versions",
    alias: "projects-list_versions",
    requestFormat: "json",
    parameters: [
      {
        name: "project_id",
        type: "Path",
        schema: z.string(),
      },
    ],
    response: ConfigVersionsResponse,
    errors: [
      {
        status: 422,
        description: `Validation Error`,
        schema: HTTPValidationError,
      },
    ],
  },
  {
    method: "get",
    path: "/api/v1/projects/:project_id/config/versions/:filename",
    alias: "projects-read_version",
    requestFormat: "json",
    parameters: [
      {
        name: "project_id",
        type: "Path",
        schema: z.string(),
      },
      {
        name: "filename",
        type: "Path",
        schema: z.string(),
      },
    ],
    response: ConfigVersionDetail,
    errors: [
      {
        status: 422,
        description: `Validation Error`,
        schema: HTTPValidationError,
      },
    ],
  },
  {
    method: "put",
    path: "/api/v1/projects/:project_id/config/versions/:filename/favorite",
    alias: "projects-toggle_favorite",
    requestFormat: "json",
    parameters: [
      {
        name: "body",
        type: "Body",
        schema: z.object({ favorite: z.boolean() }).passthrough(),
      },
      {
        name: "project_id",
        type: "Path",
        schema: z.string(),
      },
      {
        name: "filename",
        type: "Path",
        schema: z.string(),
      },
    ],
    response: ToggleFavoriteResponse,
    errors: [
      {
        status: 422,
        description: `Validation Error`,
        schema: HTTPValidationError,
      },
    ],
  },
  {
    method: "post",
    path: "/api/v1/projects/:project_id/config/versions/:filename/restore",
    alias: "projects-restore_version",
    requestFormat: "json",
    parameters: [
      {
        name: "project_id",
        type: "Path",
        schema: z.string(),
      },
      {
        name: "filename",
        type: "Path",
        schema: z.string(),
      },
    ],
    response: SaveConfigResponse,
    errors: [
      {
        status: 422,
        description: `Validation Error`,
        schema: HTTPValidationError,
      },
    ],
  },
  {
    method: "get",
    path: "/api/v1/projects/:project_id/datasets/:dataset_key/path-check",
    alias: "datasets-check_dataset_paths",
    requestFormat: "json",
    parameters: [
      {
        name: "project_id",
        type: "Path",
        schema: z.string(),
      },
      {
        name: "dataset_key",
        type: "Path",
        schema: z.string(),
      },
    ],
    response: PathCheckResponse,
    errors: [
      {
        status: 422,
        description: `Validation Error`,
        schema: HTTPValidationError,
      },
    ],
  },
  {
    method: "get",
    path: "/api/v1/projects/:project_id/datasets/:dataset_key/preview",
    alias: "datasets-preview_dataset",
    requestFormat: "json",
    parameters: [
      {
        name: "project_id",
        type: "Path",
        schema: z.string(),
      },
      {
        name: "dataset_key",
        type: "Path",
        schema: z.string(),
      },
      {
        name: "split",
        type: "Query",
        schema: z.enum(["train", "eval"]).optional().default("train"),
      },
      {
        name: "offset",
        type: "Query",
        schema: z.number().int().gte(0).optional().default(0),
      },
      {
        name: "limit",
        type: "Query",
        schema: z.number().int().gte(1).lte(200).optional().default(50),
      },
    ],
    response: PreviewResponse,
    errors: [
      {
        status: 422,
        description: `Validation Error`,
        schema: HTTPValidationError,
      },
    ],
  },
  {
    method: "post",
    path: "/api/v1/projects/:project_id/datasets/:dataset_key/validate",
    alias: "datasets-validate_dataset",
    requestFormat: "json",
    parameters: [
      {
        name: "body",
        type: "Body",
        schema: ValidateRequest,
      },
      {
        name: "project_id",
        type: "Path",
        schema: z.string(),
      },
      {
        name: "dataset_key",
        type: "Path",
        schema: z.string(),
      },
    ],
    response: ValidateResponse,
    errors: [
      {
        status: 422,
        description: `Validation Error`,
        schema: HTTPValidationError,
      },
    ],
  },
  {
    method: "put",
    path: "/api/v1/projects/:project_id/description",
    alias: "projects-update_description",
    requestFormat: "json",
    parameters: [
      {
        name: "body",
        type: "Body",
        schema: z
          .object({ description: z.string().max(2000).default("") })
          .partial()
          .passthrough(),
      },
      {
        name: "project_id",
        type: "Path",
        schema: z.string(),
      },
    ],
    response: ProjectDetail,
    errors: [
      {
        status: 422,
        description: `Validation Error`,
        schema: HTTPValidationError,
      },
    ],
  },
  {
    method: "get",
    path: "/api/v1/projects/:project_id/env",
    alias: "projects-get_project_env",
    requestFormat: "json",
    parameters: [
      {
        name: "project_id",
        type: "Path",
        schema: z.string(),
      },
    ],
    response: ProjectEnvResponse,
    errors: [
      {
        status: 422,
        description: `Validation Error`,
        schema: HTTPValidationError,
      },
    ],
  },
  {
    method: "put",
    path: "/api/v1/projects/:project_id/env",
    alias: "projects-save_project_env",
    requestFormat: "json",
    parameters: [
      {
        name: "body",
        type: "Body",
        schema: ProjectEnvRequest,
      },
      {
        name: "project_id",
        type: "Path",
        schema: z.string(),
      },
    ],
    response: ProjectEnvResponse,
    errors: [
      {
        status: 422,
        description: `Validation Error`,
        schema: HTTPValidationError,
      },
    ],
  },
  {
    method: "get",
    path: "/api/v1/projects/:project_id/runs",
    alias: "projects-list_runs",
    description: `List runs launched from this project, newest-first.

Walks &#x60;&#x60;&lt;project&gt;/runs/&#x60;&#x60; directly — every sub-directory containing
&#x60;&#x60;pipeline_state.json&#x60;&#x60; is a run. &#x60;&#x60;status&#x3D;running&#x60;&#x60; / &#x60;&#x60;?limit&#x3D;20&#x60;&#x60;
filter the result. Returns an empty list when no runs have been
launched yet — that&#x27;s the expected steady-state for a fresh
project, not an error.`,
    requestFormat: "json",
    parameters: [
      {
        name: "project_id",
        type: "Path",
        schema: z.string(),
      },
      {
        name: "status",
        type: "Query",
        schema: config_path,
      },
      {
        name: "limit",
        type: "Query",
        schema: limit,
      },
    ],
    response: ProjectRunsResponse,
    errors: [
      {
        status: 422,
        description: `Validation Error`,
        schema: HTTPValidationError,
      },
    ],
  },
  {
    method: "get",
    path: "/api/v1/providers",
    alias: "providers-list_providers",
    requestFormat: "json",
    response: z.array(ProviderSummary),
  },
  {
    method: "post",
    path: "/api/v1/providers",
    alias: "providers-create_provider",
    requestFormat: "json",
    parameters: [
      {
        name: "body",
        type: "Body",
        schema: CreateProviderRequest,
      },
    ],
    response: ProviderSummary,
    errors: [
      {
        status: 422,
        description: `Validation Error`,
        schema: HTTPValidationError,
      },
    ],
  },
  {
    method: "get",
    path: "/api/v1/providers/:provider_id",
    alias: "providers-get_provider",
    requestFormat: "json",
    parameters: [
      {
        name: "provider_id",
        type: "Path",
        schema: z.string(),
      },
    ],
    response: ProviderDetail,
    errors: [
      {
        status: 422,
        description: `Validation Error`,
        schema: HTTPValidationError,
      },
    ],
  },
  {
    method: "delete",
    path: "/api/v1/providers/:provider_id",
    alias: "providers-delete_provider",
    requestFormat: "json",
    parameters: [
      {
        name: "provider_id",
        type: "Path",
        schema: z.string(),
      },
    ],
    response: z.void(),
    errors: [
      {
        status: 422,
        description: `Validation Error`,
        schema: HTTPValidationError,
      },
    ],
  },
  {
    method: "get",
    path: "/api/v1/providers/:provider_id/config",
    alias: "providers-get_config",
    requestFormat: "json",
    parameters: [
      {
        name: "provider_id",
        type: "Path",
        schema: z.string(),
      },
    ],
    response: ProviderConfigResponse,
    errors: [
      {
        status: 422,
        description: `Validation Error`,
        schema: HTTPValidationError,
      },
    ],
  },
  {
    method: "put",
    path: "/api/v1/providers/:provider_id/config",
    alias: "providers-save_config",
    requestFormat: "json",
    parameters: [
      {
        name: "body",
        type: "Body",
        schema: z.object({ yaml: z.string() }).passthrough(),
      },
      {
        name: "provider_id",
        type: "Path",
        schema: z.string(),
      },
    ],
    response: ProviderSaveConfigResponse,
    errors: [
      {
        status: 422,
        description: `Validation Error`,
        schema: HTTPValidationError,
      },
    ],
  },
  {
    method: "post",
    path: "/api/v1/providers/:provider_id/config/validate",
    alias: "providers-validate_config",
    requestFormat: "json",
    parameters: [
      {
        name: "body",
        type: "Body",
        schema: z.object({ yaml: z.string() }).passthrough(),
      },
      {
        name: "provider_id",
        type: "Path",
        schema: z.string(),
      },
    ],
    response: ConfigValidationResult,
    errors: [
      {
        status: 422,
        description: `Validation Error`,
        schema: HTTPValidationError,
      },
    ],
  },
  {
    method: "get",
    path: "/api/v1/providers/:provider_id/config/versions",
    alias: "providers-list_versions",
    requestFormat: "json",
    parameters: [
      {
        name: "provider_id",
        type: "Path",
        schema: z.string(),
      },
    ],
    response: ProviderConfigVersionsResponse,
    errors: [
      {
        status: 422,
        description: `Validation Error`,
        schema: HTTPValidationError,
      },
    ],
  },
  {
    method: "get",
    path: "/api/v1/providers/:provider_id/config/versions/:filename",
    alias: "providers-read_version",
    requestFormat: "json",
    parameters: [
      {
        name: "provider_id",
        type: "Path",
        schema: z.string(),
      },
      {
        name: "filename",
        type: "Path",
        schema: z.string(),
      },
    ],
    response: ProviderConfigVersionDetail,
    errors: [
      {
        status: 422,
        description: `Validation Error`,
        schema: HTTPValidationError,
      },
    ],
  },
  {
    method: "post",
    path: "/api/v1/providers/:provider_id/config/versions/:filename/restore",
    alias: "providers-restore_version",
    requestFormat: "json",
    parameters: [
      {
        name: "provider_id",
        type: "Path",
        schema: z.string(),
      },
      {
        name: "filename",
        type: "Path",
        schema: z.string(),
      },
    ],
    response: ProviderSaveConfigResponse,
    errors: [
      {
        status: 422,
        description: `Validation Error`,
        schema: HTTPValidationError,
      },
    ],
  },
  {
    method: "post",
    path: "/api/v1/providers/:provider_id/test-connection",
    alias: "providers-test_connection",
    requestFormat: "json",
    parameters: [
      {
        name: "provider_id",
        type: "Path",
        schema: z.string(),
      },
    ],
    response: ConnectionTestResult,
    errors: [
      {
        status: 422,
        description: `Validation Error`,
        schema: HTTPValidationError,
      },
    ],
  },
  {
    method: "put",
    path: "/api/v1/providers/:provider_id/token",
    alias: "providers-set_token",
    description: `Write-only token upload (e.g. RUNPOD_API_KEY). Never echoed back.`,
    requestFormat: "json",
    parameters: [
      {
        name: "body",
        type: "Body",
        schema: z.object({ token: z.string().min(1) }).passthrough(),
      },
      {
        name: "provider_id",
        type: "Path",
        schema: z.string(),
      },
    ],
    response: z.void(),
    errors: [
      {
        status: 422,
        description: `Validation Error`,
        schema: HTTPValidationError,
      },
    ],
  },
  {
    method: "delete",
    path: "/api/v1/providers/:provider_id/token",
    alias: "providers-delete_token",
    requestFormat: "json",
    parameters: [
      {
        name: "provider_id",
        type: "Path",
        schema: z.string(),
      },
    ],
    response: z.void(),
    errors: [
      {
        status: 422,
        description: `Validation Error`,
        schema: HTTPValidationError,
      },
    ],
  },
  {
    method: "get",
    path: "/api/v1/providers/types",
    alias: "providers-list_types",
    requestFormat: "json",
    response: ProviderTypesResponse,
  },
  {
    method: "get",
    path: "/api/v1/runs",
    alias: "runs-list_runs",
    requestFormat: "json",
    response: RunsListResponse,
  },
  {
    method: "post",
    path: "/api/v1/runs",
    alias: "runs-create_run",
    requestFormat: "json",
    parameters: [
      {
        name: "body",
        type: "Body",
        schema: CreateRunRequest,
      },
    ],
    response: RunSummary,
    errors: [
      {
        status: 422,
        description: `Validation Error`,
        schema: HTTPValidationError,
      },
    ],
  },
  {
    method: "get",
    path: "/api/v1/runs/:run_id",
    alias: "runs-get_run",
    requestFormat: "json",
    parameters: [
      {
        name: "run_id",
        type: "Path",
        schema: z.string(),
      },
    ],
    response: RunDetail,
    errors: [
      {
        status: 422,
        description: `Validation Error`,
        schema: HTTPValidationError,
      },
    ],
  },
  {
    method: "delete",
    path: "/api/v1/runs/:run_id",
    alias: "runs-delete_run",
    requestFormat: "json",
    parameters: [
      {
        name: "run_id",
        type: "Path",
        schema: z.string(),
      },
      {
        name: "mode",
        type: "Query",
        schema: z.string().optional().default("local_and_mlflow"),
      },
    ],
    response: DeleteResultSchema,
    errors: [
      {
        status: 422,
        description: `Validation Error`,
        schema: HTTPValidationError,
      },
    ],
  },
  {
    method: "get",
    path: "/api/v1/runs/:run_id/attempts/:attempt_no",
    alias: "attempts-get_attempt",
    requestFormat: "json",
    parameters: [
      {
        name: "attempt_no",
        type: "Path",
        schema: z.number().int(),
      },
      {
        name: "run_id",
        type: "Path",
        schema: z.string(),
      },
    ],
    response: AttemptDetail,
    errors: [
      {
        status: 422,
        description: `Validation Error`,
        schema: HTTPValidationError,
      },
    ],
  },
  {
    method: "get",
    path: "/api/v1/runs/:run_id/attempts/:attempt_no/logs",
    alias: "logs-get_log_chunk",
    requestFormat: "json",
    parameters: [
      {
        name: "attempt_no",
        type: "Path",
        schema: z.number().int(),
      },
      {
        name: "run_id",
        type: "Path",
        schema: z.string(),
      },
      {
        name: "file",
        type: "Query",
        schema: z.string().optional().default("pipeline.log"),
      },
      {
        name: "offset",
        type: "Query",
        schema: z.number().int().gte(0).optional().default(0),
      },
      {
        name: "max_bytes",
        type: "Query",
        schema: z.number().int().gte(0).optional().default(0),
      },
    ],
    response: LogChunk,
    errors: [
      {
        status: 422,
        description: `Validation Error`,
        schema: HTTPValidationError,
      },
    ],
  },
  {
    method: "get",
    path: "/api/v1/runs/:run_id/attempts/:attempt_no/logs/files",
    alias: "logs-list_files",
    requestFormat: "json",
    parameters: [
      {
        name: "attempt_no",
        type: "Path",
        schema: z.number().int(),
      },
      {
        name: "run_id",
        type: "Path",
        schema: z.string(),
      },
    ],
    response: z.array(LogFileInfo),
    errors: [
      {
        status: 422,
        description: `Validation Error`,
        schema: HTTPValidationError,
      },
    ],
  },
  {
    method: "get",
    path: "/api/v1/runs/:run_id/attempts/:attempt_no/stages",
    alias: "attempts-get_stages",
    requestFormat: "json",
    parameters: [
      {
        name: "attempt_no",
        type: "Path",
        schema: z.number().int(),
      },
      {
        name: "run_id",
        type: "Path",
        schema: z.string(),
      },
    ],
    response: StagesResponse,
    errors: [
      {
        status: 422,
        description: `Validation Error`,
        schema: HTTPValidationError,
      },
    ],
  },
  {
    method: "get",
    path: "/api/v1/runs/:run_id/default-launch-mode",
    alias: "launch-default_launch_mode",
    requestFormat: "json",
    parameters: [
      {
        name: "run_id",
        type: "Path",
        schema: z.string(),
      },
    ],
    response: z.record(z.string()),
    errors: [
      {
        status: 422,
        description: `Validation Error`,
        schema: HTTPValidationError,
      },
    ],
  },
  {
    method: "post",
    path: "/api/v1/runs/:run_id/interrupt",
    alias: "launch-interrupt",
    requestFormat: "json",
    parameters: [
      {
        name: "run_id",
        type: "Path",
        schema: z.string(),
      },
    ],
    response: InterruptResponse,
    errors: [
      {
        status: 422,
        description: `Validation Error`,
        schema: HTTPValidationError,
      },
    ],
  },
  {
    method: "get",
    path: "/api/v1/runs/:run_id/job/events",
    alias: "job-get_events",
    description: `Return up to &#x60;&#x60;limit&#x60;&#x60; events with &#x60;&#x60;offset &gt;&#x3D; since&#x60;&#x60;.

Bounded — the UI polls every couple of seconds and only needs
the slice since its last cursor; capping at 2000 prevents a
pathological &#x60;&#x60;since&#x3D;0&#x60;&#x60; request from saturating the WS replay
when the buffer is full.`,
    requestFormat: "json",
    parameters: [
      {
        name: "run_id",
        type: "Path",
        schema: z.string(),
      },
      {
        name: "attempt",
        type: "Query",
        schema: limit,
      },
      {
        name: "since",
        type: "Query",
        schema: z.number().int().gte(0).optional().default(0),
      },
      {
        name: "limit",
        type: "Query",
        schema: z.number().int().gte(1).lte(2000).optional().default(200),
      },
    ],
    response: z.object({}).partial().passthrough(),
    errors: [
      {
        status: 422,
        description: `Validation Error`,
        schema: HTTPValidationError,
      },
    ],
  },
  {
    method: "get",
    path: "/api/v1/runs/:run_id/job/logs",
    alias: "job-get_logs",
    description: `Return up to &#x60;&#x60;limit&#x60;&#x60; &#x60;&#x60;trainer_log&#x60;&#x60; events with &#x60;&#x60;offset &gt;&#x3D; since&#x60;&#x60;.

The trainer subprocess pipes its stdout / stderr through the
runner&#x27;s :class:&#x60;Supervisor&#x60;, which emits each line as a
&#x60;&#x60;trainer_log&#x60;&#x60; event with &#x60;&#x60;payload&#x3D;{&quot;kind&quot;: &quot;stdout&quot;|&quot;stderr&quot;,
&quot;line&quot;: &quot;...&quot;}&#x60;&#x60;. This endpoint surfaces those events as the
file-tail fallback for cases when the structured event callback
(&#x60;&#x60;RunnerEventCallback&#x60;&#x60;) self-disabled — the supervisor pump
has no opt-out, so every line of trainer output is observable.

Same partial-failure shape as &#x60;&#x60;/events&#x60;&#x60;: a transient WebSocket
failure mid-stream returns whatever was captured plus a
structured &#x60;&#x60;error&#x60;&#x60; so the polling UI keeps the cursor and
retries on the next tick.`,
    requestFormat: "json",
    parameters: [
      {
        name: "run_id",
        type: "Path",
        schema: z.string(),
      },
      {
        name: "attempt",
        type: "Query",
        schema: limit,
      },
      {
        name: "since",
        type: "Query",
        schema: z.number().int().gte(0).optional().default(0),
      },
      {
        name: "limit",
        type: "Query",
        schema: z.number().int().gte(1).lte(2000).optional().default(200),
      },
      {
        name: "stream",
        type: "Query",
        schema: z.array(z.string()).optional().default(["stdout", "stderr"]),
      },
    ],
    response: z.object({}).partial().passthrough(),
    errors: [
      {
        status: 422,
        description: `Validation Error`,
        schema: HTTPValidationError,
      },
    ],
  },
  {
    method: "get",
    path: "/api/v1/runs/:run_id/job/status",
    alias: "job-get_status",
    description: `Return submission metadata + current FSM snapshot.`,
    requestFormat: "json",
    parameters: [
      {
        name: "run_id",
        type: "Path",
        schema: z.string(),
      },
      {
        name: "attempt",
        type: "Query",
        schema: limit,
      },
    ],
    response: z.object({}).partial().passthrough(),
    errors: [
      {
        status: 422,
        description: `Validation Error`,
        schema: HTTPValidationError,
      },
    ],
  },
  {
    method: "post",
    path: "/api/v1/runs/:run_id/job/stop",
    alias: "job-stop_job",
    description: `Forward a graceful-stop request to the runner.`,
    requestFormat: "json",
    parameters: [
      {
        name: "run_id",
        type: "Path",
        schema: z.string(),
      },
      {
        name: "attempt",
        type: "Query",
        schema: limit,
      },
      {
        name: "grace",
        type: "Query",
        schema: limit,
      },
    ],
    response: z.object({}).partial().passthrough(),
    errors: [
      {
        status: 422,
        description: `Validation Error`,
        schema: HTTPValidationError,
      },
    ],
  },
  {
    method: "post",
    path: "/api/v1/runs/:run_id/launch",
    alias: "launch-launch",
    requestFormat: "json",
    parameters: [
      {
        name: "body",
        type: "Body",
        schema: LaunchRequestSchema,
      },
      {
        name: "run_id",
        type: "Path",
        schema: z.string(),
      },
    ],
    response: LaunchResponse,
    errors: [
      {
        status: 422,
        description: `Validation Error`,
        schema: HTTPValidationError,
      },
    ],
  },
  {
    method: "get",
    path: "/api/v1/runs/:run_id/report",
    alias: "reports-get_report",
    requestFormat: "json",
    parameters: [
      {
        name: "run_id",
        type: "Path",
        schema: z.string(),
      },
      {
        name: "regenerate",
        type: "Query",
        schema: z.boolean().optional().default(false),
      },
    ],
    response: ReportResponse,
    errors: [
      {
        status: 422,
        description: `Validation Error`,
        schema: HTTPValidationError,
      },
    ],
  },
  {
    method: "get",
    path: "/api/v1/runs/:run_id/restart-points",
    alias: "launch-restart_points",
    requestFormat: "json",
    parameters: [
      {
        name: "run_id",
        type: "Path",
        schema: z.string(),
      },
      {
        name: "config_path",
        type: "Query",
        schema: config_path,
      },
    ],
    response: RestartPointsResponse,
    errors: [
      {
        status: 422,
        description: `Validation Error`,
        schema: HTTPValidationError,
      },
    ],
  },
  {
    method: "post",
    path: "/api/v1/runs/:run_id/resume-pod",
    alias: "launch-resume_pod",
    description: `Phase 11.C-2 — wake a sleeping pod for the given run.

Web UI&#x27;s &quot;Resume&quot; button calls this BEFORE re-launching the
pipeline. Idempotent — pod already RUNNING returns success
without any GraphQL round-trip on the wake side.

Response shape:

* &#x60;&#x60;availability&#x60;&#x60; — final :class:&#x60;PodAvailability&#x60; enum value.
* &#x60;&#x60;ok&#x60;&#x60; — true iff the pod is in &#x60;&#x60;RUNNING&#x60;&#x60; state after this
  call (either was already running, or wake succeeded).
* &#x60;&#x60;message&#x60;&#x60; — human-readable detail for UI display.

Status codes:

* &#x60;&#x60;200&#x60;&#x60; — service ran cleanly. Inspect &#x60;&#x60;ok&#x60;&#x60; field for the
  verdict; failure modes (capacity exhausted, GONE pod) come
  back with &#x60;&#x60;200 ok&#x3D;false&#x60;&#x60; so the UI can render a meaningful
  error rather than a generic 5xx.`,
    requestFormat: "json",
    parameters: [
      {
        name: "run_id",
        type: "Path",
        schema: z.string(),
      },
    ],
    response: z.record(z.union([z.string(), z.boolean()])),
    errors: [
      {
        status: 422,
        description: `Validation Error`,
        schema: HTTPValidationError,
      },
    ],
  },
]);

export const api = new Zodios(endpoints);

export function createApiClient(baseUrl: string, options?: ZodiosOptions) {
  return new Zodios(baseUrl, endpoints, options);
}
