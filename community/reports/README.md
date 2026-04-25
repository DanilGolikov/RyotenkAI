# Report plugins

**What they do.** Build one block (section) of the experiment report written at the end of every run. The composer [`src/reports/plugins/composer.py`](../../src/reports/plugins/composer.py) calls `render()` on each plugin in `order`, collects the returned IR nodes (headings, paragraphs, tables), and [`src/reports/plugins/markdown_block_renderer.py`](../../src/reports/plugins/markdown_block_renderer.py) turns them into Markdown.

Fail-open: if a plugin raises during `render()`, the composer substitutes an error block — the rest of the report is unaffected.

## Where the engine lives

- Interfaces: [`src/reports/plugins/interfaces.py`](../../src/reports/plugins/interfaces.py) → `IReportBlockPlugin` (Protocol), `ReportBlock`, `ReportPluginContext`, `PluginExecutionRecord`
- Registry: [`src/reports/plugins/registry.py`](../../src/reports/plugins/registry.py) → attribute mapping happens at load time (see note below)
- Composer: [`src/reports/plugins/composer.py`](../../src/reports/plugins/composer.py)
- IR document nodes: [`src/reports/document/nodes.py`](../../src/reports/document/nodes.py)
- Inline helpers: `txt`, `strong`, `emph`, `code`, `inlines`

## Minimal layout

```
community/reports/<plugin_id>/
├── manifest.toml
└── plugin.py             # or plugin/ package
```

## `manifest.toml` format

```toml
schema_version = 4

[plugin]
id = "my_block"                # → self.plugin_id at runtime
kind = "reports"               # required literal
name = "My Block"              # → self.title at runtime
version = "1.0.0"
category = "structural"        # one of: structural / configuration / validation /
                               #         evaluation / diagnostics / metrics / training
stability = "stable"
description = "Renders the … block of the experiment report."

[plugin.entry_point]
module = "plugin"
class  = "MyBlockPlugin"
```

Report plugins do not use `params_schema` / `thresholds_schema` / `secrets` — they read everything from the `ReportPluginContext` they receive.

## Section order — comes from config, not the manifest

Position in the final Markdown report is decided by the pipeline config:

```yaml
reports:
  sections:
    - header
    - summary
    - my_block        # your plugin, wherever you want it
    - footer
```

If you omit `reports.sections`, the default built-in order is used (see `src/reports/plugins/defaults.py`). Plugins whose id isn't listed simply aren't rendered — that's how you opt-out of a section.

No more globally-unique `order` numbers in manifests: different users can place the same plugin at different positions just by editing their config, and two independent plugins can't collide.

## Manifest is the single source of truth

Three things you used to write on the class are now derived automatically:

| Surface | Source |
|---|---|
| `plugin_id` | `[plugin].id` in `manifest.toml` |
| `title` | `[plugin].name` in `manifest.toml` |
| `order` | position in `reports.sections` (assigned as `i * 10`) |

Don't declare any of them on the subclass — `ReportPlugin` exposes
`plugin_id` / `title` as manifest-backed properties and `order` as
per-instance state assigned by `build_report_plugins`.

## Class contract

```python
# community/reports/my_block/plugin.py
from __future__ import annotations

from src.reports.document.nodes import DocBlock, Heading, Paragraph, Table, inlines, txt
from src.reports.plugins.interfaces import ReportBlock, ReportPlugin, ReportPluginContext


class MyBlockPlugin(ReportPlugin):
    # No plugin_id / title / order ClassVars — they come from the manifest
    # and the user's reports.sections list. Just implement render().

    def render(self, ctx: ReportPluginContext) -> ReportBlock:
        nodes: list[DocBlock] = [
            Heading(2, inlines(txt(self.title))),  # → manifest [plugin].name
            Paragraph(inlines(txt(f"Run: {ctx.run_id}"))),
            Table(
                headers=["metric", "value"],
                rows=[["duration_s", str(ctx.data.summary.duration_s)]],
            ),
        ]
        return ReportBlock(
            block_id=self.plugin_id,   # → manifest [plugin].id
            title=self.title,          # → manifest [plugin].name
            order=self.order,          # → reports.sections position × 10
            nodes=nodes,
        )
```

### Rules

- `block_id` in the returned `ReportBlock` **must** equal `self.plugin_id`.
- `plugin_id` is auto-unique because the catalogue rejects duplicate ids at load time.
- Do **not** call `mlflow` / `open()` / network inside `render()` — the composer catches exceptions, but your block turns into an error placeholder. Query via `ctx.data_provider` and `ctx.data` instead.

## The `ReportPluginContext`

```python
@dataclass(frozen=True, slots=True)
class ReportPluginContext:
    run_id: str
    data_provider: IExperimentDataProvider   # query helpers (metrics, artifacts, …)
    data: ExperimentData                      # pre-fetched run data
    report: ExperimentReport                  # assembled report so far (read-only-ish)
    logger: Logger
    clock: Clock = datetime.now               # inject for tests
```

Most plugins only need `ctx.data` (typed entity with config, metrics, stages, phases) and `ctx.report` (lets a late-order plugin read what earlier plugins produced).

## Document IR

Available node types from [`src/reports/document/nodes.py`](../../src/reports/document/nodes.py):

- `Heading(level, inlines)` — 1–6
- `Paragraph(inlines)`
- `Table(headers, rows)`
- `BulletList(items)`
- `CodeBlock(language, code)`
- `BlockQuote(inlines)`
- `HorizontalRule()`

Inline helpers: `txt("…")`, `strong("…")`, `emph("…")`, `code("…")`, `inlines(*parts)`.

Build the IR tree; don't format Markdown by hand. The renderer handles escaping and layout.

## Order guidelines

| Range | Block type |
|---|---|
| 10–20 | Header, summary |
| 30–50 | Issues, dataset validation, evaluation, model config |
| 60–80 | Memory, training config, phase details |
| 90–110 | Metrics, timelines, config dump |
| 120+ | Footer |

`order` collisions are a loader-time error, pick a free slot.

## Enabling the plugin

Report plugins run automatically for every completed run — there is no per-plugin toggle in pipeline config today. Drop a new one into `community/reports/<id>/`, restart the API, and it joins the next generated report.

## Sharing

Folder or zip, same as other kinds.

## Examples in this directory

| Folder | Block | Order |
|---|---|---|
| `header/` | Run title + status + duration | 10 |
| `summary/` | Phase overview table | 20 |
| `issues/` | Warnings and errors | 30 |
| `dataset_validation/` | Validation plugin results | 40 |
| `evaluation_block/` | Evaluation plugin results | 45 |
| `model_configuration/` | Model / adapter config | 50 |
| `memory_management/` | GPU memory analysis | 60 |
| `training_configuration/` | Training hyperparameters | 70 |
| `phase_details/` | Per-phase training details | 80 |
| `metrics_analysis/` | Loss trend analysis | 90 |
| `stage_timeline/` | Pipeline stage timeline | 100 |
| `config_dump/` | Full pipeline config snapshot | 110 |
| `footer/` | Footer with timestamps | 120 |
