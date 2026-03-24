# Dataset Validation Plugin System

A flexible plugin system for validating datasets before training.

## Architecture

```
src/data/validation/
├── __init__.py              # Export main components
├── base.py                  # ValidationPlugin (ABC), ValidationResult
├── registry.py              # ValidationPluginRegistry (singleton, @register decorator)
├── discovery.py             # Auto-discovery of plugins from plugins/ directory
├── models.py                # Pydantic models (PluginConfig, AggregatedValidationResult)
├── secrets.py               # @requires_secrets decorator + DTST_* SecretsResolver
└── plugins/
    ├── __init__.py          # Exports ensure_validation_plugins_discovered()
    ├── base/                # Universal plugins
    │   ├── min_samples.py
    │   ├── avg_length.py
    │   ├── empty_ratio.py
    │   └── diversity.py
    ├── sft/                 # SFT-specific
    │   ├── deduplication.py
    │   └── helixql_gold_syntax_backend.py
    ├── dpo/                 # DPO-specific
    │   ├── preference_format.py
    │   ├── identical_pairs.py
    │   └── helixql_preference_semantics.py
    └── sapo/                # SAPO/GRPO-specific
        └── helixql_sapo_prompt_contract.py
```

Orchestrator: `src/pipeline/stages/dataset_validator.py` → `DatasetValidator`.
Shared secrets base: `src/utils/plugin_secrets.py` → `PluginSecretsResolver`, `requires_secrets`.

## Existing plugins

| Plugin | Category | `params` | `thresholds` | Priority |
|--------|-----------|----------|--------------|----------|
| `min_samples` | base | `sample_size` | `threshold` (100) | 10 |
| `avg_length` | base | `sample_size` | `min` (50), `max` (8192) | 20 |
| `empty_ratio` | base | `sample_size`, `min_chars` | `max_ratio` (0.05) | 30 |
| `diversity_score` | base | `sample_size` | `min_score` (0.3) | 40 |
| `deduplication` | sft | `sample_size` | `max_duplicate_ratio` (0.1) | 50 |
| `preference_format` | dpo | `sample_size`, `required_fields` | `min_valid_ratio` (0.95) | 60 |
| `identical_pairs` | dpo | `sample_size` | `max_identical_ratio` (0.01) | 70 |
| `helixql_gold_syntax_backend` | sft | `sample_size` | — | 40 |
| `helixql_preference_semantics` | dpo | `sample_size` | — | 60 |
| `helixql_sapo_prompt_contract` | sapo | — | — | 50 |

## Plugin Secrets (Environment Variables)

Validation plugins that need API keys or other credentials use the `@requires_secrets` decorator and the `DTST_*` namespace.

### How it works

1. Secrets are stored in `secrets.env` under the `DTST_` prefix
2. Plugins declare which secrets they need via `@requires_secrets("DTST_...")`
3. `DatasetValidator` resolves and injects secrets before calling `validate()`
4. Plugins access them via `self._secrets["DTST_..."]`

### Namespace isolation

Validation plugins can **only** access secrets with the `DTST_` prefix. System secrets (`HF_TOKEN`, `RUNPOD_API_KEY`, etc.) and evaluation secrets (`EVAL_*`) are completely isolated and inaccessible to validation plugins. This is enforced by `SecretsResolver` at resolve time.

### Example

**1. Add the secret to `secrets.env`:**

```env
DTST_SCHEMA_VALIDATOR_TOKEN=tok-xxxxxxxxxxxxxxxx
```

**2. Declare the requirement in the plugin:**

```python
from src.data.validation.secrets import requires_secrets

@ValidationPluginRegistry.register
@requires_secrets("DTST_SCHEMA_VALIDATOR_TOKEN")
class MyExternalValidator(ValidationPlugin):
    name = "my_external_validator"
    priority = 70

    _secrets: dict[str, str]  # injected by DatasetValidator

    def validate(self, dataset):
        token = self._secrets["DTST_SCHEMA_VALIDATOR_TOKEN"]
        # use token to call external validation API
        ...
```

If the required secret is missing from `secrets.env`, `DatasetValidator` raises `RuntimeError` with a clear message during plugin loading — before any validation starts.

## Adding a new plugin

### Step 1: Choose a category

- `plugins/base/` — applies to all dataset types
- `plugins/sft/` — specific to SFT datasets
- `plugins/dpo/` — specific to DPO datasets
- `plugins/<new>/` — new category (e.g. `plugins/cot/`)

### Step 2: Implement the plugin

```python
"""Description of your plugin."""
from __future__ import annotations
import time
from typing import TYPE_CHECKING, Any, ClassVar

from src.data.validation.base import ValidationPlugin, ValidationResult
from src.data.validation.registry import ValidationPluginRegistry
from src.utils.logger import logger

if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset


@ValidationPluginRegistry.register
class YourPluginValidator(ValidationPlugin):
    # Required attributes
    name: ClassVar[str] = "your_plugin_name"
    priority: ClassVar[int] = 50          # 10–90, lower = earlier
    expensive: ClassVar[bool] = False     # True if > 1 sec
    description: ClassVar[str] = "What the plugin checks"

    # Optional
    required_fields: ClassVar[list[str]] = []
    supports_streaming: ClassVar[bool] = True

    def validate(
        self,
        dataset: Dataset | IterableDataset,
    ) -> ValidationResult:
        start_time = time.time()

        # 1. Parameters from config
        threshold = self._threshold("threshold", 100)

        # 2. Sampling (for large datasets)
        sample_size = self._param("sample_size", 1000)
        samples = self._get_sample(dataset, sample_size)

        # 3. Validation logic
        value = self._calculate(samples)
        passed = value >= threshold

        # 4. Result: IMPORTANT — keep params/thresholds and metrics separate
        return ValidationResult(
            plugin_name=self.name,
            passed=passed,
            params={"sample_size": float(sample_size)},
            thresholds={"threshold": float(threshold)},
            metrics={                         # measured results
                "measured_value": float(value),
                "samples_checked": float(len(samples)),
            },
            warnings=[],
            errors=[f"Failed: {value} < {threshold}"] if not passed else [],
            execution_time_ms=(time.time() - start_time) * 1000,
        )

    def get_recommendations(self, result: ValidationResult) -> list[str]:
        if result.passed:
            return []
        return [
            f"Validation '{self.name}' failed",
            "1. First action",
            "2. Second action",
        ]

    def _calculate(self, samples) -> float:
        return 42.0  # Your logic
```

### Step 3: Place the file in the plugin namespace

```python
# Add a new module under
# src/data/validation/plugins/<category>/your_plugin.py
# discovery will pick it up automatically.
```

### Step 4: Use in config

```yaml
datasets:
  my_dataset:
    source_type: local
    train_path: "data/train.jsonl"
    validations:
      mode: fast
      critical_failures: 1
      plugins:
        - id: your_plugin_main
          plugin: your_plugin_name
          params:
            sample_size: 1000
          thresholds:
            threshold: 100
```

## Critical rule: params/thresholds vs metrics

`ValidationResult` has three semantic blocks:

```python
# CORRECT
return ValidationResult(
    params={"sample_size": float(500)},
    thresholds={
        "threshold": float(100),   # criteria — what is SET in config
        "min_score": float(0.5),
    },
    metrics={
        "sample_count": float(500),  # results — what was MEASURED
        "avg_score": float(0.75),
    },
    ...
)

# WRONG — mixing config and metrics in one field
```

Why it matters: reports show `config` and `metrics` in different columns.

## Priority guidelines

| Range | Plugin type |
|----------|-------------|
| 10–20 | Fast basic checks (min_samples, has_fields) |
| 30–40 | Statistical checks (avg_length, diversity) |
| 50–60 | Specific checks (deduplication, format checks) |
| 70–80 | Expensive checks (external API, model inference) |
| 90+ | Final checks |

## Base class helper methods

```python
# Sampling — works with both Dataset and IterableDataset
samples = self._get_sample(dataset, sample_size)

# Check whether dataset is large
if self._is_large_dataset(dataset, sample_size):
    samples = self._get_sample(dataset, sample_size)
```

## Streaming

If the plugin supports streaming, use `_get_sample()` — it unifies both types.  
If the plugin needs the full dataset, set `supports_streaming = False`.

## Testing

```python
# tests/unit/data/validation/plugins/base/test_your_plugin.py

class TestYourPluginValidator:
    def test_passes(self):
        dataset = Dataset.from_dict({"text": ["sample"] * 200})
        plugin = YourPluginValidator(thresholds={"threshold": 100})
        result = plugin.validate(dataset)

        assert result.passed is True
        assert "threshold" in result.thresholds
        assert "measured_value" in result.metrics
        assert "threshold" not in result.metrics   # do not mix!

    def test_fails(self):
        dataset = Dataset.from_dict({"text": ["sample"] * 10})
        plugin = YourPluginValidator(thresholds={"threshold": 100})
        result = plugin.validate(dataset)

        assert result.passed is False
        assert len(result.errors) > 0

    def test_recommendations(self):
        dataset = Dataset.from_dict({"text": ["sample"] * 10})
        plugin = YourPluginValidator(thresholds={"threshold": 100})
        result = plugin.validate(dataset)
        recs = plugin.get_recommendations(result)

        assert len(recs) > 0
```

```bash
pytest src/tests/unit/data/validation/ -v
```

## Debugging

```python
from src.data.validation.registry import ValidationPluginRegistry

# List all registered plugins
plugins = ValidationPluginRegistry.list_plugins()

# Get a specific plugin
plugin = ValidationPluginRegistry.get_plugin("min_samples", thresholds={"threshold": 100})
```

## Related files

| File | Role |
|------|------|
| `base.py` | `ValidationPlugin` ABC, `ValidationResult` |
| `registry.py` | `ValidationPluginRegistry` — `@register` decorator |
| `discovery.py` | Auto-discovery of plugins |
| `models.py` | `PluginConfig`, `AggregatedValidationResult` |
| `secrets.py` | `@requires_secrets` decorator, `DTST_*` SecretsResolver |
| `plugins/__init__.py` | `ensure_validation_plugins_discovered()` |
| `src/utils/plugin_secrets.py` | Shared base: `PluginSecretsResolver`, `requires_secrets` |
| `src/pipeline/stages/dataset_validator.py` | Validation orchestrator |
| `src/config/CONFIG_REFERENCE.md` | Configuration documentation |
