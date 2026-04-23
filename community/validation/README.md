# Validation plugins

**What they do.** Check the quality of a training/eval dataset *before* it ever reaches the trainer (empty samples, duplicates, format mismatches, domain-specific semantic checks). The orchestrator [`src/pipeline/stages/dataset_validator.py`](../../src/pipeline/stages/dataset_validator.py) runs the plugins listed in `datasets.<id>.validations.plugins[]` and fails the pipeline early on contract violations.

## Where the engine lives

Your plugin **subclasses** `ValidationPlugin` from [`src/data/validation/base.py`](../../src/data/validation/base.py). The registry, loader and secrets resolver stay in `src/` — you don't touch them.

- Base class: [`src/data/validation/base.py`](../../src/data/validation/base.py) → `ValidationPlugin`, `ValidationResult`, `ValidationErrorGroup`
- Registry: [`src/data/validation/registry.py`](../../src/data/validation/registry.py) (populated by the community catalog)
- Secrets resolver (`DTST_*` namespace): [`src/data/validation/secrets.py`](../../src/data/validation/secrets.py)
- Loader semantics: [`src/community/loader.py`](../../src/community/loader.py)

## Minimal layout

Each plugin is its own folder. Two legal shapes:

```
community/validation/<plugin_id>/
├── manifest.toml
└── plugin.py             # single-file plugin
```

or (for multi-file plugins — recommended once you split into helpers):

```
community/validation/<plugin_id>/
├── manifest.toml
└── plugin/               # package; relative imports work inside
    ├── __init__.py       # exports your plugin class
    ├── main.py
    └── scoring.py
```

You can also ship a plugin as a single `*.zip` archive at `community/validation/<file>.zip` — the loader unpacks it to `community/.cache/<sha256>/` on first access.

## `manifest.toml` format

```toml
[plugin]
id = "my_validator"            # unique within validation/, used in pipeline YAML
kind = "validation"            # required literal
name = "My Validator"          # human-readable (defaults to id)
version = "1.0.0"
category = "basic"             # free-form grouping hint for the UI
stability = "stable"           # stable | beta | experimental
description = "Checks that …"

[plugin.entry_point]
module = "plugin"              # "plugin.py" file OR "plugin/" package
class  = "MyValidator"         # subclass of ValidationPlugin

# --- UI / editor hints (all optional) --------------------------------------
[params_schema.sample_size]
type = "integer"
min = 1
default = 10000

[thresholds_schema.threshold]
type = "integer"
min = 1
default = 100

[suggested_params]
sample_size = 10000

[suggested_thresholds]
threshold = 100

# --- secrets (optional) ----------------------------------------------------
[secrets]
required = ["DTST_EXAMPLE_TOKEN"]
```

`suggested_*` values pre-fill the config form when a user first adds the plugin; `*_schema` is surfaced via `GET /plugins/validation` for the web UI.

## Class contract

```python
# community/validation/my_validator/plugin.py
from __future__ import annotations
import time
from typing import TYPE_CHECKING

from src.data.validation.base import ValidationPlugin, ValidationResult

if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset


class MyValidator(ValidationPlugin):
    # Optional knobs — default values live on the base class
    expensive = False
    supports_streaming = True

    @classmethod
    def get_description(cls) -> str:
        return "One-line human-readable description"

    def validate(self, dataset: Dataset | IterableDataset) -> ValidationResult:
        start = time.time()
        threshold = self._threshold("threshold", 100)
        samples = self._get_samples_for_validation(dataset)

        measured = len(samples)               # your logic here
        passed = measured >= threshold

        return ValidationResult(
            plugin_name=self.name,
            passed=passed,
            params=dict(self.params),
            thresholds={"threshold": float(threshold)},
            metrics={"sample_count": float(measured)},
            warnings=[],
            errors=[] if passed else [f"{measured} < {threshold}"],
            execution_time_ms=(time.time() - start) * 1000,
        )

    def get_recommendations(self, result: ValidationResult) -> list[str]:
        return [] if result.passed else ["Add more samples", "Use augmentation"]
```

You do **not** register the class manually. The community catalog loader reads `manifest.toml`, imports the class via `entry_point`, and attaches `name`/`priority`/`version`/`_required_secrets` from the manifest onto it.

## Keep three semantic blocks separate in `ValidationResult`

This is the single most important convention — reports render each block in a different column.

| Field | Holds |
|---|---|
| `params` | Knobs the user set in config (`sample_size`, `min_chars`, …) |
| `thresholds` | Pass/fail criteria from config (`threshold`, `max_ratio`, …) |
| `metrics` | What the plugin actually **measured** (`sample_count`, `avg_length`, …) |

Do not put thresholds into metrics or vice versa.

## Priority ranges

| Range | Plugin type |
|---|---|
| 10–20 | Cheap format/size checks (`min_samples`, `preference_format`) |
| 25–40 | Statistical checks (`avg_length`, `diversity_score`, `empty_ratio`) |
| 50–60 | Domain-specific checks (`deduplication`, HelixQL DPO/SAPO) |
| 70–80 | Expensive / external API checks |

Lower runs earlier. If your plugin is `expensive = True` the orchestrator may skip it in `mode: fast`.

## Referencing the plugin from pipeline config

```yaml
datasets:
  my_dataset:
    source_type: local
    source_local:
      local_paths:
        train: ./datasets/train.jsonl
    validations:
      mode: fast                 # fast | thorough
      critical_failures: 1
      plugins:
        - id: min_samples_main   # your reference id inside this dataset
          plugin: min_samples    # must match [plugin] id in manifest.toml
          apply_to: [train, eval]
          params:
            sample_size: 5000
          thresholds:
            threshold: 500
```

## Secrets (`DTST_*`)

If the plugin calls an external service, declare its secrets in the manifest:

```toml
[secrets]
required = ["DTST_SCHEMA_VALIDATOR_TOKEN"]
```

Add the value to `secrets.env` (`DTST_SCHEMA_VALIDATOR_TOKEN=…`). The `DatasetValidator` reads `cls._required_secrets` and injects them as `self._secrets: dict[str, str]` before calling `validate()`:

```python
class MyValidator(ValidationPlugin):
    _secrets: dict[str, str]  # filled by the runner

    def validate(self, dataset):
        token = self._secrets["DTST_SCHEMA_VALIDATOR_TOKEN"]
        ...
```

**Namespace isolation.** Plugins can only see keys starting with `DTST_`. System secrets (`HF_TOKEN`, `RUNPOD_API_KEY`) and evaluation secrets (`EVAL_*`) are inaccessible — the resolver raises if you try.

## Multi-file reference

Canonical example: [`community/evaluation/cerebras_judge/plugin/`](../evaluation/cerebras_judge/plugin) — shows a package layout with `interface.py`, `provider.py`, `main.py` and an `__init__.py` that re-exports the entry class.

## Sharing

- Folder form: drop `community/validation/<plugin_id>/` into the repo.
- Archive form: drop `community/validation/<name>.zip` — on the next API call the loader unpacks it into `community/.cache/<sha256>/` and imports from there. The zip must contain `manifest.toml` at its root (or one level deep).

## Examples in this directory

| Folder | What it checks |
|---|---|
| `min_samples/` | Minimum number of examples |
| `avg_length/` | Average text length in range |
| `diversity/` | Lexical diversity score |
| `empty_ratio/` | Fraction of empty/near-empty rows |
| `preference_format/` | DPO chosen/rejected schema |
| `identical_pairs/` | DPO chosen ≠ rejected |
| `deduplication/` | SFT duplicate rate |
| `helixql_gold_syntax_backend/` | HelixQL compiles against schema (SFT) |
| `helixql_preference_semantics/` | HelixQL chosen-better-than-rejected (DPO) |
| `helixql_sapo_prompt_contract/` | Prompt-only GRPO/SAPO contract |
