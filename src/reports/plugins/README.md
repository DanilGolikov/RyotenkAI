# Report Plugins

Plugin-based experiment report generation. Each plugin renders one block of the final Markdown report.

## Architecture

```
src/reports/plugins/
├── interfaces.py                # IReportBlockPlugin (Protocol), ReportBlock, ReportPluginContext
├── registry.py                  # ReportPluginRegistry (@register decorator)
├── discovery.py                 # Auto-discovery of plugins from builtins/
├── composer.py                  # ReportComposer — orchestrates render + fail-open
├── markdown_block_renderer.py   # Converts ReportBlock nodes to Markdown text
└── builtins/                    # Built-in plugins
    ├── header.py                # Run header (title, status, duration)
    ├── summary.py               # Phase summary table
    ├── issues.py                # Warnings & errors
    ├── stage_timeline.py        # Pipeline stage timeline
    ├── phase_details.py         # Per-phase training details
    ├── metrics_analysis.py      # Loss trends, metric analysis
    ├── dataset_validation.py    # Dataset validation results
    ├── evaluation_block.py      # Model evaluation results
    ├── model_configuration.py   # Model configuration details
    ├── training_configuration.py # Training configuration dump
    ├── config_dump.py           # Full config dump
    ├── memory_management.py     # GPU memory analysis
    ├── event_timeline.py        # Event timeline (optional)
    └── footer.py                # Report footer
```

## How It Works

```
ExperimentReportGenerator
  1. ensure_report_plugins_discovered()   — import all builtins/ modules
  2. build_report_plugins()               — instantiate + sort by order
  3. ReportComposer.compose(context)      — call render() on each plugin
  4. MarkdownBlockRenderer.render(blocks) — convert IR nodes to Markdown
```

If a plugin crashes during `render()`, the composer substitutes an error block (fail-open). The rest of the report is unaffected.

## Built-in Plugins

| Plugin | `plugin_id` | `order` | Description |
|--------|-------------|---------|-------------|
| `HeaderBlockPlugin` | `header` | 10 | Run title, status badge, duration |
| `SummaryBlockPlugin` | `summary` | 20 | Phase overview table (strategy, loss trends) |
| `IssuesBlockPlugin` | `issues` | 30 | Warnings and errors from all stages |
| `DatasetValidationBlockPlugin` | `dataset_validation` | 40 | Dataset validation plugin results |
| `EvaluationBlockPlugin` | `evaluation_block` | 45 | Model evaluation results |
| `ModelConfigurationBlockPlugin` | `model_configuration` | 50 | Model and adapter configuration |
| `MemoryManagementBlockPlugin` | `memory_management` | 60 | GPU memory allocation analysis |
| `TrainingConfigurationBlockPlugin` | `training_configuration` | 70 | Training hyperparameters |
| `PhaseDetailsBlockPlugin` | `phase_details` | 80 | Per-phase training details |
| `MetricsAnalysisBlockPlugin` | `metrics_analysis` | 90 | Loss trend analysis, key metrics |
| `StageTimelineBlockPlugin` | `stage_timeline` | 100 | Pipeline stage execution timeline |
| `EventTimelineBlockPlugin` | `event_timeline` | 100 | Event timeline |
| `ConfigDumpBlockPlugin` | `config_dump` | 110 | Full pipeline config snapshot |
| `FooterBlockPlugin` | `footer` | 120 | Report footer with timestamps |

## Creating a Custom Plugin

### 1. Create the plugin file

```python
# src/reports/plugins/builtins/my_block.py

from src.reports.document.nodes import DocBlock, Heading, Paragraph, inlines, txt
from src.reports.plugins.interfaces import ReportBlock, ReportPluginContext
from src.reports.plugins.registry import ReportPluginRegistry


@ReportPluginRegistry.register
class MyBlockPlugin:
    plugin_id = "my_block"
    title = "My Custom Block"
    order = 75  # between training_configuration (70) and phase_details (80)

    def render(self, ctx: ReportPluginContext) -> ReportBlock:
        nodes: list[DocBlock] = [
            Heading(2, inlines(txt(self.title))),
            Paragraph(inlines(txt("Custom content here."))),
        ]

        return ReportBlock(
            block_id=self.plugin_id,
            title=self.title,
            order=self.order,
            nodes=nodes,
        )
```

### 2. Registration

The `@ReportPluginRegistry.register` decorator handles registration automatically. The discovery system imports all modules from `builtins/` — just place the file there.

### 3. Important rules

- **`plugin_id`** must be unique across all plugins
- **`order`** must be unique (determines position in the final report)
- **`block_id`** in the returned `ReportBlock` must match `plugin_id`
- The `render()` method receives a `ReportPluginContext` with access to:
  - `ctx.run_id` — MLflow run ID
  - `ctx.data` — parsed experiment data
  - `ctx.report` — assembled `ExperimentReport` model
  - `ctx.data_provider` — interface to query experiment data
  - `ctx.logger` — logger instance

## Plugin API Reference

### IReportBlockPlugin (Protocol)

| Attribute / Method | Type | Description |
|---|---|---|
| `plugin_id` | `str` | Unique plugin identifier |
| `title` | `str` | Block title (used in headings and error blocks) |
| `order` | `int` | Render order (lower = appears first) |
| `render(ctx)` | `-> ReportBlock` | Build the block's IR nodes |

### ReportBlock

| Field | Type | Description |
|---|---|---|
| `block_id` | `str` | Must match `plugin_id` |
| `title` | `str` | Block title |
| `order` | `int` | Render order |
| `nodes` | `list[DocBlock]` | IR document nodes (Heading, Paragraph, Table, etc.) |
| `meta` | `dict` | Optional metadata |

### Document IR Nodes

Available node types from `src.reports.document.nodes`:

- `Heading(level, inlines)` — section headings (1-6)
- `Paragraph(inlines)` — text paragraphs
- `Table(headers, rows)` — data tables
- `BulletList(items)` — unordered lists
- `CodeBlock(language, code)` — code snippets
- `BlockQuote(inlines)` — quotes
- `HorizontalRule()` — visual separator

Inline helpers: `txt()`, `strong()`, `emph()`, `code()`, `inlines()`.

## Order Guidelines

| Range | Block type |
|-------|------------|
| 10–20 | Header and summary |
| 30–50 | Issues, validation, evaluation, model config |
| 60–80 | Memory, training config, phase details |
| 90–110 | Metrics, timelines, config dump |
| 120+ | Footer |

## Related Files

| File | Role |
|------|------|
| `src/reports/core/builder.py` | `ExperimentReportGenerator` — orchestrates the full report |
| `src/reports/models/report.py` | `ExperimentReport` data model |
| `src/reports/domain/` | Data entities and interfaces |
| `src/reports/document/nodes.py` | IR node types for report content |
