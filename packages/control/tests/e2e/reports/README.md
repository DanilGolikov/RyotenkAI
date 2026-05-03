# E2E tests for the reporting system

## Overview

End-to-end tests cover the full report pipeline:
`ExperimentData` → `ReportBuilder` → `ReportComposer` → `MarkdownBlockRenderer` → `.md` file on disk.

## Layout

```
src/tests/e2e/reports/
├── __init__.py
├── conftest.py          # ExperimentData fixtures
├── test_report_e2e.py   # 13 e2e tests
└── README.md            # This file
```

## Test categories

### 1. Positive
- `test_positive_green_report` — successful run (GREEN)
- `test_report_structure_completeness` — main sections present

### 2. Negative
- `test_negative_red_report` — failed run with errors (RED)

### 3. Boundary
- `test_boundary_3_warnings_yellow` — exactly 3 WARN → YELLOW
- `test_boundary_5_warnings_red` — exactly 5 WARN → RED
- `test_4_warnings_yellow` — 4 WARN → YELLOW (>= 3 but < 5)

### 4. Edge / stress
- `test_crazy_empty_data` — almost empty payload
- `test_crazy_missing_fields` — missing required-like fields
- `test_crazy_mixed_severities` — 2 WARN + 10 INFO → GREEN (INFO ignored)

### 5. Health logic
- `test_run_failed_overrides_warnings` — FAILED run wins → RED

### 6. Fail-open
- `test_fail_open_plugin_error_block_visible_and_report_saved` — plugin crash produces error block, report still saved

### 7. Output
- `test_all_reports_saved` — files land under the output dir
- `test_report_encoding` — UTF-8 round-trip

## Running tests

### All report e2e tests
```bash
pytest src/tests/e2e/reports/test_report_e2e.py -v
```

### Verbose (shows prints)
```bash
pytest src/tests/e2e/reports/test_report_e2e.py -v -s
```

### Single test
```bash
pytest src/tests/e2e/reports/test_report_e2e.py::TestReportGenerationE2E::test_positive_green_report -v
```

### Unit + e2e (reports)
```bash
pytest src/tests/unit/reports/ src/tests/e2e/reports/ -v
```

## Generated reports

Reports are written under the repo’s artifact folder, e.g.:

```
<repo_root>/runs/tests_report/
```

This path is gitignored; open the `.md` files locally after a run.

Examples:
- `positive_green_report.md` — GREEN scenario
- `negative_red_report.md` — RED scenario
- `boundary_3_warnings_yellow.md` — YELLOW (3 WARN)
- `boundary_5_warnings_red.md` — RED (5 WARN)
- `crazy_empty_data.md` — empty-data scenario

## Fixtures (`conftest.py`)

- `reports_output_dir` — output directory for `.md` files
- `base_timestamp` — shared clock anchor
- `positive_experiment_data` — healthy experiment (GREEN)
- `negative_experiment_data` — failed experiment (RED)
- `boundary_3_warnings_data` — 3 WARN (YELLOW)
- `boundary_5_warnings_data` — 5 WARN (RED)
- `crazy_empty_data` — minimal payload
- `crazy_missing_fields_data` — missing fields
- `crazy_mixed_severities_data` — mixed INFO/WARN

## Coverage

- `ExperimentHealth` GREEN / YELLOW / RED
- WARN thresholds (>=3 → YELLOW, >=5 → RED)
- Priority: FAILED over warnings
- INFO events ignored for health
- Files saved to disk
- UTF-8 encoding
- Empty / invalid payloads handled

## Status

All 13 e2e tests are expected to pass.
