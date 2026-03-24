# Evaluation Plugins

Plugin-based model evaluation system. Each plugin receives already-collected model outputs and returns structured results with pass/fail verdict, metrics, and recommendations.

## How It Works

```
EvaluationRunner
  1. Load JSONL dataset → list of raw rows
  2. Collect model answers via IModelInference → list[EvalSample]
  3. For each enabled plugin (sorted by priority):
       result = plugin.evaluate(samples)
  4. Aggregate results → RunSummary (overall_passed / metrics / recommendations)
```

Plugins do NOT call inference themselves — `EvaluationRunner` collects all model answers first, then passes the complete `EvalSample` list to each plugin.

## Core Types

- **`EvalSample`** — question, model_answer, expected_answer, metadata dict
- **`EvalResult`** — pass/fail, metrics dict, errors list, recommendations list, failed_samples
- **`EvaluatorPlugin`** — abstract base class every plugin must implement

## Built-in Plugins

| Plugin | Type | Priority | Description |
|--------|------|----------|-------------|
| `helixql_generated_syntax_backend` | syntax_check | 8 | Backend syntax validation via real Helix CLI (check/compile) |
| `helixql_syntax` | syntax_check | 10 | Heuristic HelixQL syntax validation (regex-based) |
| `helixql_semantic_match` | semantic | 20 | Deterministic semantic similarity between generated and expected answers |
| `cerebras_judge` | llm_judge | 60 | LLM-as-a-judge evaluation via Cerebras API (1-5 scoring) |

## Dataset Metadata

Evaluation datasets are JSONL files. Each row has reserved fields and any number of custom metadata fields.

### Reserved fields

| Field | Required | Description |
|-------|----------|-------------|
| `question` | yes | Input prompt sent to the model |
| `expected_answer` | no | Ground-truth answer for comparison |
| `answer` | no | Alias for `expected_answer` |
| `context` | no | Additional context (reserved) |
| `messages` | no | Chat format alternative to `question`/`expected_answer` |

### Custom metadata

**All fields beyond the reserved ones are automatically collected into `EvalSample.metadata`** and passed to every plugin unchanged. Plugins access them via `sample.metadata.get("key")`.

This lets you attach arbitrary task-specific context to eval samples without changing the runner or plugin interface.

**Example dataset with metadata:**

```jsonl
{"question": "Generate a query for users", "expected_answer": "QUERY A() => N<User> RETURN N", "schema_context": "TYPE User { name: String }", "difficulty": "easy", "category": "basic_crud"}
{"question": "Find orders by date", "expected_answer": "QUERY B(d: Date) => N<Order> WHERE N.date == d RETURN N", "schema_context": "TYPE Order { date: Date, total: Float }", "difficulty": "medium", "category": "filtering"}
```

Here `schema_context`, `difficulty`, and `category` are all available in `sample.metadata`:

```python
def evaluate(self, samples: list[EvalSample]) -> EvalResult:
    for sample in samples:
        schema = sample.metadata.get("schema_context", "")
        difficulty = sample.metadata.get("difficulty", "unknown")
        # use them for scoring, filtering, reporting, etc.
```

**Real-world example:** the `helixql_generated_syntax_backend` plugin reads `sample.metadata.get("schema_context")` to obtain the database schema needed for backend validation. Without this metadata field, the plugin falls back to extracting the schema from the question text itself.

### Chat format

Instead of flat `question`/`expected_answer`, you can use OpenAI-style messages:

```jsonl
{"messages": [{"role": "user", "content": "Generate a query"}, {"role": "assistant", "content": "QUERY ..."}]}
```

The first `user` message becomes `question`, the first `assistant` message becomes `expected_answer`.

## Plugin Secrets (Environment Variables)

Plugins that need API keys or other secrets use the `@requires_secrets` decorator and the `EVAL_*` namespace.

### How it works

1. Secrets are stored in `secrets.env` under the `EVAL_` prefix
2. Plugins declare which secrets they need via `@requires_secrets("EVAL_...")`
3. `EvaluationRunner` resolves and injects secrets before calling `evaluate()`
4. Plugins access them via `self._secrets["EVAL_..."]`

### Namespace isolation

Plugins can **only** access secrets with the `EVAL_` prefix. System secrets (`HF_TOKEN`, `RUNPOD_API_KEY`, etc.) and dataset validation secrets (`DTST_*`) are completely isolated and inaccessible to evaluation plugins. This is enforced by `SecretsResolver` at resolve time.

Both evaluation (`EVAL_*`) and dataset validation (`DTST_*`) plugin systems share the same `@requires_secrets` decorator and `PluginSecretsResolver` base from `src/utils/plugin_secrets.py`.

### Example

**1. Add the secret to `secrets.env`:**

```env
EVAL_CEREBRAS_API_KEY=csk-xxxxxxxxxxxxxxxx
EVAL_MY_CUSTOM_API_KEY=sk-xxxxxxxxxxxxxxxx
```

**2. Declare the requirement in the plugin:**

```python
from src.evaluation.plugins.secrets import requires_secrets

@EvaluatorPluginRegistry.register
@requires_secrets("EVAL_MY_CUSTOM_API_KEY")
class MyExternalPlugin(EvaluatorPlugin):
    name = "my_external_plugin"
    priority = 50

    _secrets: dict[str, str]  # injected by EvaluationRunner

    def evaluate(self, samples: list[EvalSample]) -> EvalResult:
        api_key = self._secrets["EVAL_MY_CUSTOM_API_KEY"]
        # use api_key to call external API
        ...
```

If the required secret is missing from `secrets.env`, the runner logs the error and skips the plugin — the remaining plugins still execute normally.

## Creating a Custom Plugin

### 1. Create the plugin file

```python
# src/evaluation/plugins/my_category/my_plugin.py

from src.evaluation.plugins.base import EvalResult, EvalSample, EvaluatorPlugin
from src.evaluation.plugins.registry import EvaluatorPluginRegistry


@EvaluatorPluginRegistry.register
class MyPlugin(EvaluatorPlugin):
    name = "my_plugin"
    priority = 50  # lower = runs earlier (0-100)
    requires_expected_answer = True  # set to True if plugin needs expected_answer

    @classmethod
    def get_description(cls) -> str:
        return "Checks something specific about model outputs"

    def evaluate(self, samples: list[EvalSample]) -> EvalResult:
        passed_count = 0
        failed_indices = []

        for idx, sample in enumerate(samples):
            if self._check(sample):
                passed_count += 1
            else:
                failed_indices.append(idx)

        total = len(samples)
        pass_rate = passed_count / total if total else 0.0
        min_rate = float(self._threshold("min_pass_rate", 0.8))

        return EvalResult(
            plugin_name=self.name,
            passed=pass_rate >= min_rate,
            metrics={"pass_rate": round(pass_rate, 4), "total": total, "passed": passed_count},
            errors=[] if pass_rate >= min_rate else [f"pass_rate={pass_rate:.2%} < {min_rate:.2%}"],
            recommendations=self.get_recommendations(EvalResult(plugin_name=self.name, passed=pass_rate >= min_rate)),
            sample_count=total,
            failed_samples=failed_indices,
        )

    def get_recommendations(self, result: EvalResult) -> list[str]:
        if result.passed:
            return []
        return ["Review failed samples and adjust training data"]

    def _check(self, sample: EvalSample) -> bool:
        return sample.model_answer.strip() != ""
```

### 2. Add `__init__.py`

```python
# src/evaluation/plugins/my_category/__init__.py
from .my_plugin import MyPlugin

__all__ = ["MyPlugin"]
```

The discovery system (`discovery.py`) auto-imports all modules from `plugins/` subdirectories. No manual registration needed beyond the `@EvaluatorPluginRegistry.register` decorator.

### 3. Configure in pipeline config

```yaml
evaluation:
  enabled: true
  dataset:
    path: data/eval/my_eval_dataset.jsonl
  evaluators:
    plugins:
      - id: my_check
        plugin: my_plugin
        enabled: true
        save_report: true  # optional: write per-sample report to evaluation/my_plugin_report.md
        params:
          custom_param: value
        thresholds:
          min_pass_rate: 0.8
```

## Plugin API Reference

### EvaluatorPlugin (base class)

| Attribute / Method | Description |
|---|---|
| `name: str` | Unique plugin identifier (class variable) |
| `priority: int` | Execution order — lower runs first (class variable) |
| `requires_expected_answer: bool` | If True, plugin is skipped when dataset has no expected_answer |
| `params: dict` | Plugin parameters from config |
| `thresholds: dict` | Plugin thresholds from config |
| `evaluate(samples) -> EvalResult` | Main evaluation logic (abstract) |
| `get_description() -> str` | Human-readable description (abstract) |
| `get_recommendations(result) -> list[str]` | Actionable suggestions (abstract) |
| `_param(key, default)` | Read from `self.params` |
| `_threshold(key, default)` | Read from `self.thresholds` |
| `_validate_contract()` | Optional config validation at init time |

### Utilities

- **`aggregate_scores()`** — builds `EvalResult` from per-sample normalized scores (avoids boilerplate for score-based plugins)
- **`save_plugin_report()`** — writes a per-sample Markdown report to `runs/{run}/evaluation/{name}_report.md`
- **`PluginReportRow`** — data row for the Markdown report

## Model Client

The evaluation system uses `IModelInference` interface to collect model answers:

- **`OpenAICompatibleInferenceClient`** — OpenAI-compatible API (works with vLLM endpoints)
- **`MockInferenceClient`** — for testing (returns predetermined answers)

Custom clients: implement `IModelInference` from `src/evaluation/model_client/interfaces.py`.
