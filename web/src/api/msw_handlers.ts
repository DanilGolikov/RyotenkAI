/* eslint-disable */
// AUTO-GENERATED — DO NOT EDIT BY HAND.
// Regenerate with: `make regen-msw` (or `cd web && npm run gen:msw`).
// Source: web/src/api/openapi.json (produced by scripts/sync_openapi.py).
//
// Each handler returns a deterministic, contract-shaped 200 response
// synthesised from the response schema (example > default > zero value).
// Override per-test by passing an additional handler to MSW's
// `server.use(...)` — the latest registration wins.

import { http, HttpResponse } from 'msw'

export const handlers = [
  // health-health
  http.get("/api/v1/health", () => HttpResponse.json({"status":"ok","runs_dir":"","runs_dir_readable":false})),
  // config-validate
  http.post("/api/v1/config/validate", () => HttpResponse.json({"ok":false,"config_path":""})),
  // config-default
  http.get("/api/v1/config/default", () => HttpResponse.json({"runs_dir":""})),
  // config-schema
  http.get("/api/v1/config/schema", () => HttpResponse.json({})),
  // config-presets
  http.get("/api/v1/config/presets", () => HttpResponse.json({"presets":[]})),
  // config-preview_preset
  http.post("/api/v1/config/presets/:preset_id/preview", () => HttpResponse.json({"resulting_config":{},"diff":[],"requirements":[],"placeholders":[]})),
  // attempts-get_attempt
  http.get("/api/v1/runs/:run_id/attempts/:attempt_no", () => HttpResponse.json({"attempt_id":"","attempt_no":0,"runtime_name":"","requested_action":"","effective_action":"","status":"","started_at":""})),
  // attempts-get_stages
  http.get("/api/v1/runs/:run_id/attempts/:attempt_no/stages", () => HttpResponse.json({"stages":[]})),
  // datasets-preview_dataset
  http.get("/api/v1/projects/:project_id/datasets/:dataset_key/preview", () => HttpResponse.json({"rows":[],"has_more":false})),
  // datasets-check_dataset_paths
  http.get("/api/v1/projects/:project_id/datasets/:dataset_key/path-check", () => HttpResponse.json({"source_type":"local","train":{"exists":false}})),
  // datasets-validate_dataset
  http.post("/api/v1/projects/:project_id/datasets/:dataset_key/validate", () => HttpResponse.json({"duration_ms":0,"format_check":[],"plugin_results":[]})),
  // integrations-list_types
  http.get("/api/v1/integrations/types", () => HttpResponse.json({"types":[]})),
  // integrations-list_integrations
  http.get("/api/v1/integrations", () => HttpResponse.json([])),
  // integrations-create_integration
  http.post("/api/v1/integrations", () => HttpResponse.json({"id":"","name":"","type":"","path":"","created_at":""})),
  // integrations-get_integration
  http.get("/api/v1/integrations/:integration_id", () => HttpResponse.json({"id":"","name":"","type":"","path":"","created_at":"","updated_at":""})),
  // integrations-delete_integration
  http.delete("/api/v1/integrations/:integration_id", () => HttpResponse.json(null)),
  // integrations-get_config
  http.get("/api/v1/integrations/:integration_id/config", () => HttpResponse.json({"yaml":""})),
  // integrations-save_config
  http.put("/api/v1/integrations/:integration_id/config", () => HttpResponse.json({"ok":false})),
  // integrations-validate_config
  http.post("/api/v1/integrations/:integration_id/config/validate", () => HttpResponse.json({"ok":false,"config_path":""})),
  // integrations-list_versions
  http.get("/api/v1/integrations/:integration_id/config/versions", () => HttpResponse.json({"versions":[]})),
  // integrations-read_version
  http.get("/api/v1/integrations/:integration_id/config/versions/:filename", () => HttpResponse.json({"filename":"","yaml":""})),
  // integrations-restore_version
  http.post("/api/v1/integrations/:integration_id/config/versions/:filename/restore", () => HttpResponse.json({"ok":false})),
  // integrations-set_token
  http.put("/api/v1/integrations/:integration_id/token", () => HttpResponse.json(null)),
  // integrations-delete_token
  http.delete("/api/v1/integrations/:integration_id/token", () => HttpResponse.json(null)),
  // integrations-test_connection
  http.post("/api/v1/integrations/:integration_id/test-connection", () => HttpResponse.json({"ok":false})),
  // logs-get_log_chunk
  http.get("/api/v1/runs/:run_id/attempts/:attempt_no/logs", () => HttpResponse.json({"file":"","offset":0,"next_offset":0,"eof":false,"content":""})),
  // logs-list_files
  http.get("/api/v1/runs/:run_id/attempts/:attempt_no/logs/files", () => HttpResponse.json([])),
  // launch-restart_points
  http.get("/api/v1/runs/:run_id/restart-points", () => HttpResponse.json({"config_path":""})),
  // launch-default_launch_mode
  http.get("/api/v1/runs/:run_id/default-launch-mode", () => HttpResponse.json({})),
  // launch-launch
  http.post("/api/v1/runs/:run_id/launch", () => HttpResponse.json({"pid":0,"launched_at":"","command":[],"launcher_log":"","run_dir":""})),
  // launch-interrupt
  http.post("/api/v1/runs/:run_id/interrupt", () => HttpResponse.json({"interrupted":false})),
  // launch-resume_pod
  http.post("/api/v1/runs/:run_id/resume-pod", () => HttpResponse.json({})),
  // plugins-get_report_defaults
  http.get("/api/v1/plugins/reports/defaults", () => HttpResponse.json({"sections":[]})),
  // plugins-preflight
  http.post("/api/v1/plugins/preflight", () => HttpResponse.json({"ok":false})),
  // plugins-list_plugins
  http.get("/api/v1/plugins/:kind", () => HttpResponse.json({"kind":"reward","plugins":[]})),
  // projects-list_projects
  http.get("/api/v1/projects", () => HttpResponse.json([])),
  // projects-create_project
  http.post("/api/v1/projects", () => HttpResponse.json({"id":"","name":"","path":"","created_at":""})),
  // projects-get_project
  http.get("/api/v1/projects/:project_id", () => HttpResponse.json({"id":"","name":"","path":"","description":"","created_at":"","updated_at":""})),
  // projects-delete_project
  http.delete("/api/v1/projects/:project_id", () => HttpResponse.json(null)),
  // projects-update_description
  http.put("/api/v1/projects/:project_id/description", () => HttpResponse.json({"id":"","name":"","path":"","description":"","created_at":"","updated_at":""})),
  // projects-get_config
  http.get("/api/v1/projects/:project_id/config", () => HttpResponse.json({"yaml":""})),
  // projects-save_config
  http.put("/api/v1/projects/:project_id/config", () => HttpResponse.json({"ok":false})),
  // projects-get_project_env
  http.get("/api/v1/projects/:project_id/env", () => HttpResponse.json({"env":{}})),
  // projects-save_project_env
  http.put("/api/v1/projects/:project_id/env", () => HttpResponse.json({"env":{}})),
  // projects-validate_config
  http.post("/api/v1/projects/:project_id/config/validate", () => HttpResponse.json({"ok":false,"config_path":""})),
  // projects-list_versions
  http.get("/api/v1/projects/:project_id/config/versions", () => HttpResponse.json({"versions":[]})),
  // projects-read_version
  http.get("/api/v1/projects/:project_id/config/versions/:filename", () => HttpResponse.json({"filename":"","yaml":""})),
  // projects-restore_version
  http.post("/api/v1/projects/:project_id/config/versions/:filename/restore", () => HttpResponse.json({"ok":false})),
  // projects-toggle_favorite
  http.put("/api/v1/projects/:project_id/config/versions/:filename/favorite", () => HttpResponse.json({"favorite_versions":[]})),
  // projects-list_runs
  http.get("/api/v1/projects/:project_id/runs", () => HttpResponse.json({"runs":[]})),
  // providers-list_types
  http.get("/api/v1/providers/types", () => HttpResponse.json({"types":[]})),
  // providers-list_providers
  http.get("/api/v1/providers", () => HttpResponse.json([])),
  // providers-create_provider
  http.post("/api/v1/providers", () => HttpResponse.json({"id":"","name":"","type":"","path":"","created_at":""})),
  // providers-get_provider
  http.get("/api/v1/providers/:provider_id", () => HttpResponse.json({"id":"","name":"","type":"","path":"","created_at":"","updated_at":""})),
  // providers-delete_provider
  http.delete("/api/v1/providers/:provider_id", () => HttpResponse.json(null)),
  // providers-get_config
  http.get("/api/v1/providers/:provider_id/config", () => HttpResponse.json({"yaml":""})),
  // providers-save_config
  http.put("/api/v1/providers/:provider_id/config", () => HttpResponse.json({"ok":false})),
  // providers-validate_config
  http.post("/api/v1/providers/:provider_id/config/validate", () => HttpResponse.json({"ok":false,"config_path":""})),
  // providers-list_versions
  http.get("/api/v1/providers/:provider_id/config/versions", () => HttpResponse.json({"versions":[]})),
  // providers-read_version
  http.get("/api/v1/providers/:provider_id/config/versions/:filename", () => HttpResponse.json({"filename":"","yaml":""})),
  // providers-restore_version
  http.post("/api/v1/providers/:provider_id/config/versions/:filename/restore", () => HttpResponse.json({"ok":false})),
  // providers-set_token
  http.put("/api/v1/providers/:provider_id/token", () => HttpResponse.json(null)),
  // providers-delete_token
  http.delete("/api/v1/providers/:provider_id/token", () => HttpResponse.json(null)),
  // providers-test_connection
  http.post("/api/v1/providers/:provider_id/test-connection", () => HttpResponse.json({"ok":false})),
  // reports-get_report
  http.get("/api/v1/runs/:run_id/report", () => HttpResponse.json({"path":"","markdown":"","generated_at":""})),
  // job-get_status
  http.get("/api/v1/runs/:run_id/job/status", () => HttpResponse.json({})),
  // job-get_events
  http.get("/api/v1/runs/:run_id/job/events", () => HttpResponse.json({})),
  // job-get_logs
  http.get("/api/v1/runs/:run_id/job/logs", () => HttpResponse.json({})),
  // job-stop_job
  http.post("/api/v1/runs/:run_id/job/stop", () => HttpResponse.json({})),
  // runs-list_runs
  http.get("/api/v1/runs", () => HttpResponse.json({"runs_dir":""})),
  // runs-create_run
  http.post("/api/v1/runs", () => HttpResponse.json({"run_id":"","run_dir":"","created_at":"","created_ts":0,"status":"","attempts":0,"config_name":""})),
  // runs-get_run
  http.get("/api/v1/runs/:run_id", () => HttpResponse.json({"schema_version":0,"logical_run_id":"","run_directory":"","config_path":"","pipeline_status":"","status":""})),
  // runs-delete_run
  http.delete("/api/v1/runs/:run_id", () => HttpResponse.json({"target":"","local_deleted":false,"is_success":false})),
]
