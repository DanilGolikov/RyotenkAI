# Reward plugins

**What they do.** Plug a **reward function** into GRPO/SAPO-style reinforcement-learning phases. The factory [`src/training/reward_plugins/factory.py`](../../src/training/reward_plugins/factory.py) is called from the training orchestrator, reads the plugin name from `phase_config.params["reward_plugin"]`, instantiates it, runs `setup()`, and collects the kwargs it wants to inject into the TRL `*Config` and `Trainer` constructors.

Unlike validation/evaluation plugins, reward plugins have **no thresholds** — they only return callables/weights consumed by TRL.

## Where the engine lives

- Base class: [`src/training/reward_plugins/base.py`](../../src/training/reward_plugins/base.py) → `RewardPlugin`
- Registry: [`src/training/reward_plugins/registry.py`](../../src/training/reward_plugins/registry.py)
- Factory (lifecycle + error handling): [`src/training/reward_plugins/factory.py`](../../src/training/reward_plugins/factory.py) → `build_reward_plugin_result`

## Minimal layout

```
community/reward/<plugin_id>/
├── manifest.toml
└── plugin.py             # or plugin/ package
```

## `manifest.toml` format

```toml
[plugin]
id = "my_reward"
kind = "reward"
name = "My Reward"
version = "1.0.0"
category = "semantic"
stability = "stable"
description = "Scores completions by …"
# Required for kind="reward": list the strategy_type values this plugin
# is compatible with. The UI uses this to filter the palette when
# attaching a reward plugin to a training strategy.
supported_strategies = ["grpo", "sapo"]

[plugin.entry_point]
module = "plugin"
class  = "MyRewardPlugin"

[params_schema.backend]
type = "enum"
options = ["compile", "semantic_only"]
default = "compile"

[params_schema.timeout_seconds]
type = "integer"
min = 1
max = 120
default = 10

[suggested_params]
backend = "compile"
timeout_seconds = 10
```

Thresholds are **not used** for reward plugins; omit `thresholds_schema` and `suggested_thresholds`.

## Class contract — lifecycle

```
__init__(params)          # cheap; just stash config
  ↓
setup()                   # install binaries, warm caches (idempotent)
  ↓
build_config_kwargs(...)  # returns {"reward_weights": [...], ...}
build_trainer_kwargs(...) # returns {"reward_funcs": [callable, ...], ...}
  ↓
[TRL training loop]
  ↓
teardown()                # cleanup, always in finally
```

```python
# community/reward/my_reward/plugin.py
from __future__ import annotations
from typing import Any

from src.training.reward_plugins.base import RewardPlugin


class MyRewardPlugin(RewardPlugin):
    def __init__(self, params: dict[str, Any]) -> None:
        super().__init__(params)               # calls _validate_params()

    def _validate_params(self) -> None:
        backend = self.params.get("backend", "compile")
        if backend not in {"compile", "semantic_only"}:
            raise ValueError(f"unknown backend={backend!r}")

    def setup(self) -> None:
        # install external CLI, download artifacts, etc. Must be idempotent.
        ...

    def build_config_kwargs(self, *, train_dataset, phase_config, pipeline_config) -> dict[str, Any]:
        return {"reward_weights": [1.0, 0.5]}   # passed to TRL *Config

    def build_trainer_kwargs(self, *, train_dataset, phase_config, pipeline_config) -> dict[str, Any]:
        def lexical_reward(completions, **kw) -> list[float]:
            return [float(len(c)) / 100 for c in completions]

        def domain_reward(completions, **kw) -> list[float]:
            ...

        return {"reward_funcs": [lexical_reward, domain_reward]}   # passed to TRL Trainer

    def teardown(self) -> None:
        ...
```

### Required vs optional methods

| Method | Required? |
|---|---|
| `build_trainer_kwargs` | **yes** (abstract) |
| `build_config_kwargs` | no (default `{}`) |
| `setup` / `teardown` | no (default no-op) |
| `_validate_params` | no (default no-op) |

### Reward function signature (inside `build_trainer_kwargs`)

TRL passes `completions: list[str]` and **keyword-only extras derived from dataset columns** (`prompts`, `reference_answer`, `schema_context`, …). The function returns `list[float]` of the same length. The real-world reference implementation in [`community/reward/helixql_compiler_semantic/`](helixql_compiler_semantic/) uses `_coerce_column(kwargs, "prompts", N)` to deal with TRL occasionally passing a single string instead of a list.

## Referencing the plugin from pipeline config

```yaml
training:
  type: grpo           # or sapo
  strategies:
    - strategy_type: grpo
      dataset: rl_dataset
      params:
        reward_plugin: my_reward         # matches [plugin] id in manifest
        reward_params:
          backend: compile
          timeout_seconds: 10
```

`reward_plugin` is **required** for GRPO/SAPO phases — the core trainer no longer ships with a built-in reward function.

## Dataset field requirements

Reward functions receive dataset columns as kwargs. If your plugin expects specific fields (e.g. `prompt`, `reference_answer`, `schema_context`), validate their presence at the top of `build_trainer_kwargs`:

```python
features = getattr(train_dataset, "features", {}) or {}
missing = {"prompt", "reference_answer"} - set(features)
if missing:
    raise ValueError(f"Dataset is missing required fields: {sorted(missing)}")
```

Fail early, not from inside a reward callback.

## Secrets

If the plugin calls an external service, declare its secrets in the manifest:

```toml
[secrets]
required = ["EVAL_MY_JUDGE_KEY"]
```

Reward plugins share the `EVAL_*` resolver with evaluation plugins (there is no dedicated `REWARD_*` namespace yet — see tech-debt note in the migration plan). The injection point is the `build_reward_plugin_result` factory; access in code via `self._secrets["EVAL_…"]`.

## Multi-file reference

[`community/evaluation/cerebras_judge/plugin/`](../evaluation/cerebras_judge/plugin) — same packaging principles apply. For complex reward plugins, split binary-install logic (`install_cli.py`), scoring (`scoring.py`) and the `RewardPlugin` subclass (`main.py`).

## Sharing

Folder or zip, same as other kinds. Place at `community/reward/<plugin_id>/` or `community/reward/<name>.zip`.

## Examples in this directory

| Folder | What it rewards |
|---|---|
| `helixql_compiler_semantic/` | HelixQL: `+1` if the generated query compiles against the schema, plus a semantic-similarity term vs the reference answer |
