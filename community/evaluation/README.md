# Evaluation plugins

**What they do.** Score a trained model's answers on a held-out eval dataset. The runner [`src/evaluation/runner.py`](../../src/evaluation/runner.py) collects model outputs once (via `IModelInference`), then hands the complete list of `EvalSample`s to each enabled plugin in the order declared in the pipeline config YAML. Each plugin returns an `EvalResult` with a pass/fail verdict, numeric metrics, and recommendations.

Plugins **do not call inference themselves** — answers are already in `EvalSample.model_answer` when `evaluate()` is called.

## Where the engine lives

- Base class: [`src/evaluation/plugins/base.py`](../../src/evaluation/plugins/base.py) → `EvaluatorPlugin`, `EvalSample`, `EvalResult`
- Registry: [`src/evaluation/plugins/registry.py`](../../src/evaluation/plugins/registry.py)
- Secrets resolver (`EVAL_*` namespace): [`src/evaluation/plugins/secrets.py`](../../src/evaluation/plugins/secrets.py)
- Per-sample report helpers: [`src/evaluation/plugins/utils.py`](../../src/evaluation/plugins/utils.py) → `aggregate_scores`, `save_plugin_report`, `PluginReportRow`

## Minimal layout

```
community/evaluation/<plugin_id>/
├── manifest.toml
└── plugin.py             # or plugin/ package — both supported by the loader
```

## `manifest.toml` format

```toml
[plugin]
id = "my_judge"
kind = "evaluation"
name = "My Judge"
version = "1.0.0"
category = "semantic"          # semantic | syntax_check | llm_judge | custom
stability = "stable"
description = "Scores answers against the reference …"

[plugin.entry_point]
module = "plugin"
class  = "MyJudgePlugin"

[thresholds_schema.min_mean_score]
type = "number"
min = 0.0
max = 1.0
default = 0.7

[suggested_thresholds]
min_mean_score = 0.7

# Single source of truth for env vars this plugin needs. Loader-derived
# rule: secret=true AND optional=false → the registry auto-injects the
# resolved value as ``self._secrets["EVAL_MY_API_KEY"]``.
[[required_env]]
name = "EVAL_MY_API_KEY"
description = "API key for the upstream judge service"
optional = false
secret = true
managed_by = ""
```

`params_schema` / `thresholds_schema` drive form generation; `suggested_*` pre-fill values when a user adds the plugin to a project.

## Class contract

```python
# community/evaluation/my_judge/plugin.py
from __future__ import annotations

from src.evaluation.plugins.base import EvalResult, EvalSample, EvaluatorPlugin


class MyJudgePlugin(EvaluatorPlugin):
    requires_expected_answer = True   # skip this plugin if dataset has no expected_answer
    _secrets: dict[str, str]          # injected by EvaluationRunner

    @classmethod
    def get_description(cls) -> str:
        return "One-line description shown in the UI catalog"

    def evaluate(self, samples: list[EvalSample]) -> EvalResult:
        api_key = self._secrets["EVAL_MY_API_KEY"]
        min_mean = float(self._threshold("min_mean_score", 0.7))

        scores: list[float] = []
        failed: list[int] = []
        for idx, s in enumerate(samples):
            score = self._score(s, api_key)
            scores.append(score)
            if score < min_mean:
                failed.append(idx)

        mean = sum(scores) / len(scores) if scores else 0.0
        return EvalResult(
            plugin_name=self.name,
            passed=mean >= min_mean,
            metrics={"mean_score": round(mean, 4), "sample_count": float(len(samples))},
            errors=[] if mean >= min_mean else [f"mean={mean:.2%} < {min_mean:.2%}"],
            recommendations=self.get_recommendations(EvalResult(plugin_name=self.name, passed=mean >= min_mean)),
            sample_count=len(samples),
            failed_samples=failed,
        )

    def get_recommendations(self, result: EvalResult) -> list[str]:
        return [] if result.passed else ["Inspect failed samples", "Tune training data diversity"]

    def _score(self, sample: EvalSample, api_key: str) -> float:
        ...
```

Prefer the shared helpers in `src/evaluation/plugins/utils.py`:

- `aggregate_scores(scores=..., raw_scores=..., failed_indices=..., thresholds=..., threshold_key="min_mean_score")` builds an `EvalResult` with mean / p50 / distribution metrics for you.
- `save_plugin_report(self.name, rows, result)` writes `runs/<run>/evaluation/<plugin>_report.md` (gated by `self._save_report`).

## `EvalSample.metadata` — custom dataset fields

The runner reserves a small set of keys (`question`, `expected_answer`, `answer`, `context`, `messages`). **Everything else in each JSONL row lands in `sample.metadata`** untouched. Plugins read it without any schema changes:

```jsonl
{"question": "Users by country?", "expected_answer": "QUERY …", "schema_context": "TYPE User {…}", "difficulty": "easy"}
```

```python
schema = sample.metadata.get("schema_context", "")
```

`helixql_generated_syntax_backend` uses this exact mechanism to pick up the DB schema.

## Execution order

Plugins run in the order they appear in
``evaluation.evaluators.plugins`` in the user's config YAML — there is
no global priority field. Order cheap structural checks (syntax,
format) before deterministic semantic comparators, then aggregated
scorers, then external-API judges (Cerebras, OpenAI, …) — this keeps
costs predictable when an upstream check fails fast.

## Referencing the plugin from pipeline config

```yaml
evaluation:
  enabled: true
  dataset:
    path: data/eval/my_eval.jsonl
  evaluators:
    plugins:
      - id: judge_main
        plugin: my_judge        # matches [plugin] id
        enabled: true
        save_report: true       # writes per-sample Markdown next to the run
        params:
          model: llama-3.3-70b
          max_samples: 50
        thresholds:
          min_mean_score: 0.7
```

## Secrets (`EVAL_*`)

Declared in the manifest as `[[required_env]]` entries with `secret=true, optional=false`:

```toml
[[required_env]]
name = "EVAL_CEREBRAS_API_KEY"
description = "Cerebras Inference Cloud API key"
optional = false
secret = true
managed_by = ""
```

The loader derives `cls._required_secrets` from every such entry, and the registry resolves them from `secrets.env` (`EVAL_CEREBRAS_API_KEY=…`) into `self._secrets: dict[str, str]` before `evaluate()`. Missing secrets → the runner logs, skips this plugin, and continues with the others.

Plugins only see `EVAL_*` keys — system (`HF_TOKEN`, `RUNPOD_API_KEY`) and validation (`DTST_*`) namespaces are hard-isolated.

## Multi-file reference

Canonical example: [`cerebras_judge/plugin/`](cerebras_judge/plugin) — splits `interface.py` (Protocol + dataclass), `provider.py` (HTTP/SSL/retries) and `main.py` (the `EvaluatorPlugin` subclass). `__init__.py` re-exports the entry class.

## Sharing

Drop the folder into `community/evaluation/` or ship a zip (`community/evaluation/my_judge.zip`) with `manifest.toml` at its root. The loader extracts to `community/.cache/<sha256>/` on first load.

## Examples in this directory

| Folder | What it scores |
|---|---|
| `cerebras_judge/` | LLM-as-judge via Cerebras API (1–5 → [0,1]) |
| `helixql_semantic_match/` | Deterministic HelixQL similarity vs expected |
| `helixql_generated_syntax_backend/` | HelixQL compiles against schema (CLI) |
| `helixql_syntax/` | HelixQL regex-level syntax check |
