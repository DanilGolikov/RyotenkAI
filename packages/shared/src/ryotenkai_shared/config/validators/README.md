## Validators: LLM-friendly guide

This directory contains **centralized validators** for the config schema (`src/config/*`).
Here “validators” means checks for:
- **“if X, then Y”**
- **“combination A + B is not allowed”**
- **“when a feature is on, a block/field is required”**

Goals:
- **Fail-fast**: catch errors when loading YAML/config, before heavy stages run.
- **Centralization**: rule logic in one place; wiring rules in one place.
- **Predictability**: minimal “magic”, explicit calls and clear errors.

---

### What counts as a “validator” in this project

There are several validation levels:
- **Field-level** (`@field_validator`): one field (type/range/enum).
- **Model-level** (`@model_validator(mode="after")`): relationships inside one model.
- **Cross-config** (between blocks): e.g. `training.strategies[*].dataset` ↔ `datasets.<name>`.

The `validators/` directory is for **cross-field / cross-block** rules.
It is not for I/O: no filesystem, env, network, or secret checks.

---

### Where validators should live

Convention:
- `src/config/validators/<domain>.py` maps to a top-level config “domain”.
  Examples: `training`, `datasets`, `inference`, `providers`, `pipeline`.
- `src/config/validators/cross.py` is for rules that truly span multiple domains.

Important:
- **Rule code** lives in `validators/`.
- **Rule invocation** lives on the Pydantic config model (one place) so it is obvious what is checked.

---

### Required wiring pattern

Rule for each Pydantic config:
- The model must have **one** `@model_validator(mode="after")`.
- Method name is consistent project-wide: `_run_model_validators`.
- Inside — **explicit** calls to validator functions from `validators/`.

Example (simplified):

```python
from pydantic import model_validator

class TrainingOnlyConfig(...):
    @model_validator(mode="after")
    def _run_model_validators(self) -> "TrainingOnlyConfig":
        # local import: reduce circular imports
        from ..validators.training import validate_training_adalora_requires_block

        validate_training_adalora_requires_block(self)
        return self
```

---

### Structure of `validators/<domain>.py`

Domain files contain **plain validator functions** with clear names.
Names follow: **what they validate**.

Examples (illustrative):

```python
def validate_dataset_source_blocks(cfg: DatasetConfig) -> None: ...
def validate_inference_enabled_is_supported(cfg: InferenceConfig) -> None: ...
def validate_ssh(cfg: SSHConfig) -> None: ...
```

Why this helps:
- New rule = **add a function** (or extend an existing one).
- Invocations live **where needed** (on the specific config model).

---

### How to add a new rule (step by step)

1) **Pick a domain**:
- touches only `training` → `validators/training.py`
- touches `training` + `datasets` → `validators/cross.py` or `PipelineConfig` (see below)

2) **Add the check** in the right `validate_*` function.
Rule: a short comment (up to 3 lines) above each check describing what it enforces.

3) **Add tests**:
- one test for “valid”
- one for “invalid” (expect `ValidationError` or `ValueError` with a clear message)

4) **Error messages** must be actionable:
- cite YAML keys (`training.type`, `training.adalora`, …)
- suggest how to fix (“requires 'training.adalora:' section”)

---

### How to write rules (constraints and style)

Rules should be:
- **Pure**: read fields only, no side effects.
- **Cheap**: O(1) or O(n) over lists/dicts, no heavy work.
- **Deterministic**: same input → same result.

Do not:
- touch disk (check file/dir existence)
- read env/secrets
- make network calls

Do:
- check logical dependencies between flags/fields
- check “block required when option is on”
- check incompatible combinations

---

### Avoiding circular imports (critical)

Rule:
- `validators/*.py` must **not** import config models at runtime.
- For types, use `TYPE_CHECKING` only.

Template:

```python
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..training.schema import TrainingOnlyConfig
```

Import the validator function in the model with a **local** import inside `_run_model_validators`.

---

### Runtime validators: `validators/runtime.py`

A separate class of validations that need **runtime objects** (loaded secrets, plugin registries, etc.) and therefore **cannot** be Pydantic `model_validator`s.

Key differences from `cross.py`:

| | `cross.py` | `runtime.py` |
|---|---|---|
| Called from | `PipelineConfig._run_model_validators` (Pydantic) | `PipelineOrchestrator.__init__()` |
| Access to secrets | not allowed | allowed |
| When it runs | YAML parse time | Orchestrator init |
| Pattern | `validate_*(cfg)` | `validate_*(cfg, runtime_obj)` |

Rules for `runtime.py`:
- Functions take **both** arguments: config and runtime object.
- **No I/O**: no disk or network. Loaded `Secrets` is OK.
- Named `validate_<what>`.
- On error, raise `ValueError` with an actionable message (YAML field name + how to fix).

---

### Where cross-config validation already lives (mental model)

Some rules fit best at the “root” where all blocks are visible.
Example: `training.strategies[*].dataset` exists in `datasets.<name>`.

This is validated in `PipelineConfig` via:
- `PipelineConfig._run_model_validators()` → `validators/pipeline.py::validate_pipeline_config_references`
- `validators/cross.py::validate_pipeline_strategy_dataset_references`

That is correct because `TrainingOnlyConfig` alone does not know the `datasets` registry.

---

### Testing: practical minimum

Recommendations:
- Test at `PipelineConfig`/domain config level where the rule applies.
- For errors expect `pydantic_core.ValidationError` (if building the model) or `ValueError`
  (if calling the validator directly).
- Prefer **minimal valid** configs in tests (no extra noise).

Useful pattern:
- helper `_mk_cfg(...)` inside the test
- assert on `match="..."` with part of the error message

---

### Checklist (for LLMs and humans)

- [ ] Rule belongs to a domain? Chose `validators/<domain>.py`
- [ ] Rule is cross-domain? Chose `validators/cross.py` or `PipelineConfig`
- [ ] Added the rule in `validate_*` and a short comment above the check
- [ ] Error message cites YAML keys and how to fix
- [ ] No runtime imports of models (only `TYPE_CHECKING`)
- [ ] Tests for valid and invalid cases
