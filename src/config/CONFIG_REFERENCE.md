# Pipeline Configuration Reference

Complete reference for all RyotenkAI Pipeline configuration parameters.

**Config file:** `src/config/pipeline_config.yaml`

**Version:** 7.0 (Modular Config + Path Hardcoding + Centralized Validation)

---

## ⚠️ BREAKING CHANGES in v7.0

**Config v7.0** adds config modularization, hardcoded output paths, and centralized validation.

**Config v6.0** introduced explicit required fields for critical parameters:

| Change | Description |
|-----------|----------|
| **ModelConfig** | `torch_dtype`, `trust_remote_code` → **REQUIRED** |
| **LoraConfig** | 8 fields → **REQUIRED** (5 base + 3 advanced) |
| **AdaLoraConfig** | 6 fields → **REQUIRED** (2 core + 4 LoRA) |
| **Hyperparameters** | Split: Global (5 core REQUIRED) + Phase (all optional) |
| **Datasets** | `training_paths` **REMOVED** (auto-generated) |
| **Training** | `output_dir` **REMOVED** (output paths are now hardcoded inside the run workspace) |
| **Strategies** | `checkpoint_output` **REMOVED** (phase folder is computed automatically) |
| **Inference** | `merge_image`, `serve_image` → **REQUIRED (single_node only)** |

**Migration guide:** `src/docs/feature/release/config_modularization_and_simplification_v7.md`

---

## Table of contents

1. [Model](#1-model---model-configuration)
2. [Training](#2-training---training-parameters)
3. [Datasets](#3-datasets---dataset-registry)
4. [Providers](#4-providers---gpu-providers)
5. [Inference](#5-inference---inference-endpoint-deployment)
6. [Evaluation](#6-evaluation---model-quality-evaluation)
7. [Experiment Tracking](#7-experiment-tracking---mlflow--huggingface-hub)
8. [Integration Test](#8-integration-test---integration-testing)
9. [Quick Reference](#quick-reference)

---

## 1. Model - Model configuration

```yaml
model:
  name: Qwen/Qwen2.5-7B-Instruct
  # ✅ v6.0 REQUIRED
  torch_dtype: bfloat16          # REQUIRED: bfloat16 (A100/H100), float16 (V100/T4)
  trust_remote_code: true        # REQUIRED: true for Qwen/Phi, false for Llama
  
  # Optional: Technical defaults
  device_map: auto               # auto = optimal GPU placement
  flash_attention: false         # Flash Attention 2 (requires flash-attn)
```

### Parameters:

| Parameter | Type | v6.0 Status | Description |
|----------|-----|-------------|----------|
| `name` | string | **REQUIRED** | Full model name on HuggingFace Hub |
| `torch_dtype` | string | **✅ REQUIRED (v6.0)** | Precision: `bfloat16`, `float16`, `float32`, `auto` |
| `trust_remote_code` | bool | **✅ REQUIRED (v6.0)** | Trust model code. True for Qwen/Phi, false for Llama/Mistral |
| `tokenizer_name` | string | Optional | Tokenizer name. If `null`, `name` is used |
| `device_map` | string | Optional (`"auto"`) | GPU placement: `auto`, `cuda:0`, `balanced` |
| `flash_attention` | bool | Optional (`false`) | Flash Attention 2 (~2x speedup, requires GPU support) |

> **v6.0 CHANGE:** `torch_dtype` and `trust_remote_code` are now **required** for explicit configuration. Defaults were removed to avoid silent misconfiguration.

---

## 2. Training - Training parameters

```yaml
training:
  provider: single_node           # Provider selection
  type: qlora                     # qlora, lora, adalora

  # ✅ v6.0 REQUIRED: LoRA config (8 fields REQUIRED)
  lora:
    # REQUIRED: Base LoRA (5 fields)
    r: 16
    lora_alpha: 32
    lora_dropout: 0.05
    bias: none
    target_modules: all-linear
    
    # REQUIRED: Advanced (3 fields)
    use_dora: false
    use_rslora: false
    init_lora_weights: gaussian
    
    # Optional: QLoRA quantization (defaults: nf4, bfloat16, true)

  # ✅ v6.0 REQUIRED: Global hyperparams (5 core fields REQUIRED)
  hyperparams:
    # REQUIRED: Core (5 fields)
    per_device_train_batch_size: 4
    gradient_accumulation_steps: 4
    learning_rate: 2.0e-4
    warmup_ratio: 0.05
    epochs: 3
    
    # Optional: Advanced (defaults applied)

  strategies:
    - strategy_type: sft
      dataset: default
      hyperparams:              # PhaseHyperparametersConfig (all optional)
        epochs: 2               # Override global
        learning_rate: 1.0e-4   # Override global
  
  # ✅ v6.x Hardcoded output paths (no training.output_dir / checkpoint_output)
  # - output root: {run_workspace}/output
  # - per phase:   {run_workspace}/output/phase_<idx>_<strategy_type>
```

### 2.1 Basic parameters

| Parameter | Type | Default | Description |
|----------|-----|--------------|----------|
| `provider` | string | - | **Required.** Provider name from the `providers` section |
| `type` | string | `"qlora"` | **Training type:** `qlora`, `lora`, `adalora` |

> **v6.x CHANGE:** `training.output_dir` was removed. The output folder is now created automatically inside the run workspace:
> - `{run_workspace}/output/phase_<idx>_<strategy_type>/checkpoint-*`

### 2.2 Hyperparameters (v6.0: Global + Phase Split)

> **v6.0 BREAKING CHANGE:** Hyperparameters are split into **GlobalHyperparametersConfig** (5 core REQUIRED) and **PhaseHyperparametersConfig** (all optional).

**Merge Priority:** `Phase > Global` (system defaults are **not** applied for core fields)

#### Global Hyperparameters (`training.hyperparams`)

**5 Core Fields REQUIRED:**

| Parameter | Type | v6.0 Status | Description |
|----------|-----|-------------|----------|
| `per_device_train_batch_size` | int | **✅ REQUIRED** | Per-device batch size on GPU |
| `gradient_accumulation_steps` | int | **✅ REQUIRED** | Gradient accumulation steps |
| `learning_rate` | float | **✅ REQUIRED** | Learning rate (>0, <1) |
| `warmup_ratio` | float | **✅ REQUIRED** | Warmup ratio (0.0-1.0) |
| `epochs` | int | **✅ REQUIRED** | Number of epochs (1-50) |

**Advanced Fields (Optional with defaults):**

| Parameter | Type | Default | Description |
|----------|-----|---------|----------|
| `weight_decay` | float | `0.01` | Weight decay |
| `lr_scheduler_type` | string | `"cosine"` | LR scheduler type |
| `optim` | string | `None` | Optimizer (auto-detect if unset) |
| `bf16` | bool | `true` | Use bfloat16 |
| `fp16` | bool | `false` | Use float16 |
| `gradient_checkpointing` | bool | `true` | Gradient Checkpointing |
| `logging_steps` | int | `10` | Log every N steps |
| `save_steps` | int | `500` | Save a checkpoint every N steps |

#### Phase Hyperparameters (`strategies[].hyperparams`)

**All Fields Optional** (for selective overrides):

```yaml
strategies:
  - strategy_type: sft
    hyperparams:             # PhaseHyperparametersConfig
      epochs: 2              # Override global epochs
      learning_rate: 1e-4    # Override global LR
      # Rest inherited from Global
```

#### Strategy-Specific Parameters

**SFT / CPT / CoT:**

| Parameter | Type | Default | Description |
|----------|-----|---------|----------|
| `max_length` | int | `2048` | Maximum sequence length |
| `packing` | bool | `false` | Dataset packing for speed |

**DPO / ORPO:**

| Parameter | Type | Default | Description |
|----------|-----|---------|----------|
| `beta` | float | `0.1` | Beta (KL penalty) |
| `max_length` | int | `2048` | Maximum length |

**SAPO (GRPO):**

| Parameter | Type | v6.0 Status | Description |
|----------|-----|-------------|----------|
| `max_prompt_length` | int | **REQUIRED** | Max prompt length |
| `max_completion_length` | int | **REQUIRED** | Max generation length |
| `num_generations` | int | `4` | Generations per prompt |
| `sapo_temperature_pos` | float | `1.0` | SAPO Positive Temperature |
| `sapo_temperature_neg` | float | `1.0` | SAPO Negative Temperature |

> **v6.0 NOTE:** The SAPO validator enforces `max_prompt_length` and `max_completion_length` when `strategy_type=sapo`.

### 2.3 LoRA Configuration (v6.0: 8 Fields REQUIRED)

> **v6.0 BREAKING CHANGE:** Base and advanced LoRA fields are now **required**. QLoRA quantization fields remain optional with best-practice defaults.

```yaml
training:
  type: qlora                     # qlora, lora, or adalora
  lora:
    # ✅ v6.0 REQUIRED: Base LoRA (5 fields)
    r: 16                         # REQUIRED: Rank (1-256)
    lora_alpha: 32                # REQUIRED: Scaling (typically 2*r)
    lora_dropout: 0.05            # REQUIRED: Dropout (0.0-0.5)
    bias: none                    # REQUIRED: none, all, lora_only
    target_modules: all-linear    # REQUIRED: "all-linear" or list

    # ✅ v6.0 REQUIRED: Advanced (3 fields)
    use_dora: false               # REQUIRED: DoRA
    use_rslora: false             # REQUIRED: rsLoRA
    init_lora_weights: gaussian   # REQUIRED: gaussian, eva, pissa, olora, loftq

    # Optional: QLoRA quantization (defaults: nf4, bfloat16, true)
    # bnb_4bit_quant_type: nf4
    # bnb_4bit_compute_dtype: bfloat16
    # bnb_4bit_use_double_quant: true
```

#### LoRA Parameters:

| Parameter | Type | v6.0 Status | Constraints | Description |
|----------|-----|-------------|-------------|----------|
| `r` | int | **✅ REQUIRED** | 1-256 | Rank — LoRA matrix dimensionality |
| `lora_alpha` | int | **✅ REQUIRED** | >= 1 | Scaling factor (typically 2*r) |
| `lora_dropout` | float | **✅ REQUIRED** | 0.0-0.5 | Dropout for regularization |
| `bias` | string | **✅ REQUIRED** | `none`, `all`, `lora_only` | Bias mode |
| `target_modules` | string/list | **✅ REQUIRED** | - | Which layers to adapt |
| `use_dora` | bool | **✅ REQUIRED** | - | DoRA: Weight-Decomposed LoRA |
| `use_rslora` | bool | **✅ REQUIRED** | - | rsLoRA: Rank-Stabilized LoRA |
| `init_lora_weights` | string | **✅ REQUIRED** | See below | Weight initialization |

**init_lora_weights Options:** `gaussian`, `eva`, `pissa`, `olora`, `loftq`, `true`, `false`

#### QLoRA Parameters (Optional):

| Parameter | Type | Default | Description |
|----------|-----|---------|----------|
| `bnb_4bit_quant_type` | string | `"nf4"` | Quantization type: `nf4` (recommended), `fp4` |
| `bnb_4bit_compute_dtype` | string | `"bfloat16"` | Compute dtype: `bfloat16`, `float16` |
| `bnb_4bit_use_double_quant` | bool | `true` | Nested quantization to save memory |

> **v6.0 RATIONALE:** QLoRA fields have well-established best-practice defaults and rarely need changes.

### 2.4 AdaLoRA Configuration (v6.0: 6 Fields REQUIRED)

> **v6.0 BREAKING CHANGE:** Core AdaLoRA and common LoRA fields are now **required**. Scheduling fields remain optional.

```yaml
training:
  type: adalora
  adalora:
    # ✅ v6.0 REQUIRED: AdaLoRA core (2 fields)
    init_r: 12                    # REQUIRED: Initial rank
    target_r: 8                   # REQUIRED: Target rank after pruning
    
    # ✅ v6.0 REQUIRED: Common LoRA (4 fields)
    lora_alpha: 32                # REQUIRED
    lora_dropout: 0.05            # REQUIRED
    bias: none                    # REQUIRED
    target_modules: all-linear    # REQUIRED
    
    # Optional: Scheduling (defaults: 200, 1000, 10, 0.85, 0.85)
    # tinit: 200
    # tfinal: 1000
    # delta_t: 10
    # beta1: 0.85
    # beta2: 0.85
```

#### AdaLoRA Parameters:

| Parameter | Type | v6.0 Status | Default | Description |
|----------|-----|-------------|---------|----------|
| `init_r` | int | **✅ REQUIRED** | - | Initial rank for all adapters |
| `target_r` | int | **✅ REQUIRED** | - | Target average rank after pruning |
| `lora_alpha` | int | **✅ REQUIRED** | - | Scaling factor |
| `lora_dropout` | float | **✅ REQUIRED** | - | Dropout |
| `bias` | string | **✅ REQUIRED** | - | Bias type |
| `target_modules` | string/list | **✅ REQUIRED** | - | Target modules |
| `tinit` | int | Optional | `200` | Steps until pruning starts |
| `tfinal` | int | Optional | `1000` | Step when pruning ends |
| `delta_t` | int | Optional | `10` | Interval between pruning steps |
| `beta1` | float | Optional | `0.85` | EMA coefficient for importance |
| `beta2` | float | Optional | `0.85` | EMA coefficient for second moment |

### 2.5 Strategies

```yaml
strategies:
  - strategy_type: cpt          # Phase 1: Pre-training
    dataset: pretrain_data
    hyperparams:
      epochs: 1
      learning_rate: 1e-5

  - strategy_type: sft          # Phase 2: Fine-tuning
    dataset: instructions
    hyperparams:
      epochs: 3
      learning_rate: 2e-4

  - strategy_type: dpo          # Phase 3: Alignment
    dataset: preferences
    hyperparams:
      epochs: 1
      learning_rate: 5e-6
      beta: 0.1
```

| Strategy | Purpose | Typical LR | Data format |
|-----------|------------|-------------|---------------|
| `cpt` | Continued Pre-Training | 1e-5 | plain text |
| `sft` | Supervised Fine-Tuning | 2e-4 | instruction/response |
| `cot` | Chain-of-Thought | 2e-5 | instruction/reasoning/answer |
| `dpo` | Direct Preference Optimization | 5e-6 | chosen/rejected pairs |
| `orpo` | Odds Ratio Preference Optimization | 1e-5 | chosen/rejected pairs |
| `sapo` | Soft Adaptive Policy Optimization | 1e-6 | prompts (RL) |

---

## 3. Datasets - Dataset registry

> **v6.0 BREAKING CHANGE:** `training_paths` were removed from the config. Paths are generated automatically as `data/{strategy_type}/{basename(local_paths.train)}`.

### 3.1 Local Datasets (v6.0)

```yaml
datasets:
  default:
    source_type: local
    source_local:
      local_paths:
        train: data/datasets/train.jsonl  # ✅ v6.0 REQUIRED (no default)
        eval: data/datasets/eval.jsonl    # Optional
      
      # ✅ v6.0 REMOVED: training_paths (auto-generated)
      # Pattern: data/{strategy_type}/{basename(local_paths.train)}

    validations:
      mode: fast
      critical_failures: 1
      plugins:
        - id: min_samples_main
          plugin: min_samples
          thresholds: {threshold: 100}
```

### Path Auto-Generation (v6.0)

**Pattern:** `data/{strategy_type}/{basename(local_path)}`

**Examples:**
```
Strategy: sft
local_paths.train = "data/datasets/train.jsonl"
  → auto-generated: "data/sft/train.jsonl"

Strategy: dpo
local_paths.train = "/abs/path/preferences.jsonl"
  → auto-generated: "data/dpo/preferences.jsonl"

Multi-phase (CPT → SFT):
  - CPT: data/cpt/corpus.jsonl
  - SFT: data/sft/instructions.jsonl
```

**Benefits:**
- ✅ Removes redundancy and path-related bugs
- ✅ Automatic dataset isolation per strategy
- ✅ Simplifies configuration

### Dataset Parameters:

| Parameter | Type | v6.0 Status | Description |
|----------|-----|-------------|----------|
| `source_type` | string | Required | `local` or `huggingface` |
| `source_local.local_paths.train` | string | **✅ REQUIRED (v6.0)** | Local path to train (no default) |
| `source_local.local_paths.eval` | string | Optional (`null`) | Local path to eval |
| `source_hf.train_id` | string | Required (for HF) | HuggingFace dataset id |
| `max_samples` | int | Optional | Example limit for tests |

> **v6.0 NOTE:** `source_local.training_paths` was fully removed. The deployment manager generates paths at runtime.

### 3.2 Dataset Validation

Dataset validation plugins share one shape:
- `id` — identity of the plugin instance
- `plugin` — registered validation plugin name
- `params` — runtime/execution settings
- `thresholds` — pass/fail criteria

```yaml
validations:
  mode: fast                 # fast (≤10k) or full (entire dataset)
  critical_failures: 1       # Stop threshold
  plugins:
    - id: min_samples_train
      plugin: min_samples
      apply_to: [train]
      thresholds: {threshold: 100}
    - id: avg_length_main
      plugin: avg_length
      thresholds: {min: 50, max: 8192}
    - id: diversity_main
      plugin: diversity_score
      thresholds: {min_score: 0.3}
    - id: gold_syntax_main
      plugin: helixql_gold_syntax_backend
      params:
        timeout_seconds: 30
      thresholds:
        min_pass_rate: 0.98
```

---

## 4. Providers - GPU providers

`providers:` is a registry of **named** GPU providers. The active provider is selected via `training.provider`
(string = key under `providers:`).

General pattern (built-in providers):
- `connect`: transport/SSH
- `cleanup`: resource cleanup policy
- `training`: training-specific settings (timeouts, docker runtime, etc.)

### 4.1 `single_node` (local GPU host over SSH)

Example (recommended alias mode):

```yaml
providers:
  single_node:
    connect:
      ssh:
        alias: pc

    cleanup:
      cleanup_workspace: false
      keep_on_error: true
      on_interrupt: true

    training:
      workspace_path: /home/user/workspace
      docker_image: "ryotenkai/ryotenkai-training-runtime:latest"
      training_start_timeout: 120
      gpu_type: "RTX 4060"
      mock_mode: false
```

> SSH fields (host/user/key_path/alias/timeouts) are documented in `src/config/providers/ssh.py` (`SSHConfig`).

#### Parameters: `providers.single_node.cleanup`

| Parameter | Type | Default | Description |
|----------|-----|--------------|----------|
| `cleanup_workspace` | bool | `false` | Delete the run directory after the pipeline finishes |
| `keep_on_error` | bool | `true` | Keep the run directory on error (for debugging) |
| `on_interrupt` | bool | `true` | Apply cleanup policy on Ctrl+C (SIGINT) |

#### Parameters: `providers.single_node.training`

| Parameter | Type | Default | Description |
|----------|-----|--------------|----------|
| `workspace_path` | string | - | **REQUIRED.** Base directory on the remote host; `runs/<run_name>/...` is created inside |
| `training_start_timeout` | int | `120` | Max seconds to wait for training to start |
| `gpu_type` | string \| null | `null` | GPU type for logs/reports (`null` = auto-detect) |
| `mock_mode` | bool | `false` | Test mode without real execution |
| `execution_mode` | string | `"docker"` | How training runs on the host. Only `docker` is supported |
| `docker_image` | string | - | **REQUIRED.** Training runtime Docker image |
| `docker_shm_size` | string | `"16g"` | Container `--shm-size` (increase for heavy dataloaders/large batches) |
| `docker_container_name_prefix` | string | `"ryotenkai_training"` | Container name prefix (must start with `ryotenkai_training` so TrainingMonitor can find it) |

### 4.2 `runpod` (RunPod cloud)

Example:

```yaml
providers:
  runpod:
    connect:
      ssh:
        key_path: "~/.ssh/id_ed25519_runpod"

    cleanup:
      auto_delete_pod: true
      keep_pod_on_error: false
      on_interrupt: true
      terminate_after_retrieval: false

    training:
      gpu_type: "NVIDIA A40"            # REQUIRED
      cloud_type: "COMMUNITY"           # COMMUNITY is cheaper than Secure Cloud
      image_name: "ryotenkai/ryotenkai-training-runtime:latest"
      container_disk_gb: 50
      volume_disk_gb: 20
      ports: "8888/http,22/tcp"
      template_id: null
```

**Important:**
- Workspace inside the pod is **hardcoded** to `/workspace` (not configurable).
- RunPod API base URL is **not configurable** (runtime constant `RUNPOD_API_BASE_URL`).
- RyotenkAI uses **`runpodctl-first`** for pod control and batch file transfer where possible, with fallback to direct RunPod API/REST.
- Runtime operations **inside** the pod still use **direct SSH over exposed TCP** (training start, health checks, tunnels, merge/vLLM).
- Legacy keys `providers.runpod.type` and `providers.runpod.api_base_url` were removed; if present, the validator requires removing them.

#### Parameters: `providers.runpod.connect.ssh`

| Parameter | Type | Default | Description |
|----------|-----|--------------|----------|
| `key_path` | string | - | **REQUIRED.** Path to private SSH key used for direct SSH over exposed TCP into the pod runtime |

#### Parameters: `providers.runpod.cleanup`

| Parameter | Type | Default | Description |
|----------|-----|--------------|----------|
| `auto_delete_pod` | bool | `true` | Automatically delete the pod after the pipeline finishes |
| `keep_pod_on_error` | bool | `false` | Keep the pod running on error (for debugging) |
| `on_interrupt` | bool | `true` | Delete the pod on Ctrl+C (SIGINT) when `true` |
| `terminate_after_retrieval` | bool | `false` | Delete the training pod right after ModelRetriever (before InferenceDeployer / ModelEvaluator). Saves GPU budget when inference or eval stages are enabled |

#### Parameters: `providers.runpod.training`

| Parameter | Type | Default | Description |
|----------|-----|--------------|----------|
| `gpu_type` | string | - | **REQUIRED.** GPU type (RunPod string, e.g. `"NVIDIA A40"`, `"NVIDIA RTX A4000"`) |
| `cloud_type` | string | `"ALL"` | Cloud type: `ALL`, `SECURE`, `COMMUNITY`. `COMMUNITY` is cheaper; `SECURE` is more reliable |
| `image_name` | string | - | **REQUIRED.** Docker image for training runtime |
| `container_disk_gb` | int | `100` | Container disk size (GB) |
| `volume_disk_gb` | int | `20` | Persistent volume size (GB, mounted at `/workspace`) |
| `ports` | string | `"8888/http,22/tcp"` | Exposed ports (`port/protocol`) |
| `template_id` | string \| null | `null` | Optional template ID (prebuilt image/startup) |

#### Parameters: `providers.runpod.inference` (for `inference.provider=runpod`)

Example config with Network Volume (stable storage, tied to a datacenter):

```yaml
providers:
  runpod:
    inference:
      volume:                            # Optional: omit or null — pod is created in any DC
        id: null                         # If set — resolved directly by id
        name: "ryotenkai-hf-cache"           # Name for lookup/auto-create
        data_center_id: "EU-RO-1"        # Required if id is not set
        size_gb: 50
      pod:
        name_prefix: "ryotenkai-vllm-pod"
        image_name: "ryotenkai/inference-vllm:latest"
        gpu_type_ids:
          - "NVIDIA RTX 2000 Ada Generation"
          - "NVIDIA RTX 4000 Ada Generation"
        gpu_count: 1
        container_disk_gb: 50
        volume_disk_gb: 50               # Pod Volume — used only without a network volume
        ports: ["22/tcp"]
      serve:
        port: 8000
```

**Without Network Volume** (`volume: null`): the pod is created in any datacenter where the requested GPU is available → much higher availability. Data in `/workspace` is lost when the pod is deleted (re-persisted on stop/start).

| Parameter | Type | Default | Description |
|----------|-----|--------------|----------|
| `volume` | object \| null | `null` | Network Volume config. If `null` — pod is created without DC affinity |
| `volume.id` | string \| null | `null` | Existing volume ID (preferred for stability) |
| `volume.name` | string | - | **REQUIRED (if volume is set).** Volume name for lookup/auto-create |
| `volume.data_center_id` | string \| null | `null` | RunPod DC id (e.g. `EU-RO-1`). Required if `volume.id` is not set |
| `volume.size_gb` | int | `50` | Volume size (GB, on auto-create) |
| `pod.name_prefix` | string | `"ryotenkai-vllm-pod"` | Pod name prefix |
| `pod.image_name` | string | - | **REQUIRED.** Docker image inference runtime |
| `pod.gpu_type_ids` | list[string] | 7 types | Preferred GPU list (first available) |
| `pod.gpu_count` | int | `1` | Number of GPUs |
| `pod.container_disk_gb` | int | `50` | Container disk size (GB) |
| `pod.volume_disk_gb` | int | `50` | Pod Volume size (GB, mounted at `/workspace`). Used only without a network volume. Ignored by RunPod when a network volume is present |
| `pod.ports` | list[string] | `["22/tcp"]` | Exposed ports |
| `serve.port` | int | `8000` | vLLM port inside the pod (SSH tunnel target) |

---

## 5. Inference - Inference endpoint deployment

> **v6.0 CHANGE:** `merge_image` and `serve_image` were added for explicit version tracking; **required only for** `inference.provider=single_node` (two-container Docker strategy).

```yaml
providers:
  # Inference-only provider: RunPod Serverless (vLLM)
  runpod_serverless:
    template:
      # Created/reused; pipeline appends a hash suffix from the base model
      name: ryotenkai-vllm-template
      image_name: runpod/worker-v1-vllm:v2.2.0stable-cuda12.1.0
      container_disk_gb: 50
      volume_in_gb: 0
    endpoint:
      name_prefix: ryotenkai-vllm
      # Priority GPU list: RunPod falls back to the next type when stock is low.
      # 16GB tier (cheapest) + 24GB tier (better availability).
      gpu_type_ids:
        - "NVIDIA RTX 2000 Ada Generation"
        - "NVIDIA RTX 4000 Ada Generation"
        - "NVIDIA RTX A4000"
        - "NVIDIA RTX A4500"
        - "NVIDIA L4"
        - "NVIDIA RTX A5000"
        - "NVIDIA GeForce RTX 3090"
      gpu_count: 1
      flashboot: true
      idle_timeout_seconds: 5
      # Settings for generated `chat_inference.py` (reduces max-workers-busy warnings)
      resume_workers_min: 1
      resume_workers_max: 2
      resume_idle_timeout_seconds: 60
    lora:
      lora_name: ryotenkai_adapter
      runtime_lora_updating: false

inference:
  enabled: true
  provider: runpod_serverless   # single_node | runpod_serverless | runpod
  engine: vllm
  
  engines:
    vllm:
      # Docker images are used only when inference.provider=single_node (two-container Docker strategy)
      # merge_image: ryotenkai/inference-vllm:v1.0
      # serve_image: ryotenkai/inference-vllm:v1.0
      
      # Optional: Technical params (defaults applied)
      # tensor_parallel_size: 1
      # max_model_len: 4096
      # gpu_memory_utilization: 0.90
  
  common:
    model_source: auto
    lora:
      merge_before_deploy: false
```

### Inference Providers (`inference.provider`)

- `single_node`: SSH + Docker(vLLM) on one machine. Endpoint access is usually via SSH tunnel.
- `runpod_serverless`: RunPod Serverless vLLM endpoint. The pipeline **prepares the endpoint and stops it (workers=0)**, while `chat_inference.py` **sets workers=1**, waits for readiness, and starts chat. LoRA (if any) is picked up at worker start via `LORA_MODULES`.
- `runpod`: RunPod Pods. The pipeline uses **runpodctl-first** for safe pod control where possible, but still relies on **direct SSH** for runtime work inside the pod. It prepares the Pod and parks it (stopped); interactive scripts (`chat_inference.py` / `stop_inference.py`) control start/stop and the SSH tunnel. Network Volume is optional: without it the pod is created in any available datacenter with the requested GPU (higher availability), but `/workspace` storage is lost when the pod is deleted.

### Inference Parameters:

| Parameter | Type | v6.0 Status | Description |
|----------|-----|-------------|----------|
| `engines.vllm.merge_image` | string | Required (single_node) | Docker image for merge job (two-container strategy) |
| `engines.vllm.serve_image` | string | Required (single_node) | Docker image for vLLM server (two-container strategy) |
| `tensor_parallel_size` | int | Optional (`1`) | Tensor parallelism |
| `max_model_len` | int | Optional (`4096`) | Max sequence length |
| `gpu_memory_utilization` | float | Optional (`0.90`) | GPU memory (0.0-1.0) |

> **v6.0 RATIONALE:** Explicit Docker image versions are critical for reproducibility and troubleshooting (especially single_node/two-container).

---

## 6. Evaluation - Model quality evaluation

> **New in v7.x:** Plugin-based evaluation of the trained model. Runs after inference deploy:
> `InferenceDeployer → ModelEvaluator`.
> Fail-fast: `evaluation.enabled=true` **requires** `inference.enabled=true` (validated when the config is loaded).

```yaml
evaluation:
  enabled: false                        # Enable/disable eval stage

  dataset:
    path: data/eval/helixql_eval.jsonl  # Path to eval dataset (JSONL)

  save_answers_md: true                 # Save answers.md under runs/{run}/evaluation/answers.md

  evaluators:
    - id: syntax_main                  # Unique instance id
      plugin: helixql_syntax           # Registered plugin name
      enabled: true
      save_report: false               # Save helixql_syntax_report.md under runs/{run}/evaluation/
      params: {}
      thresholds:
        min_valid_ratio: 0.80          # 80% of answers must be valid HelixQL

    - id: judge_main
      plugin: cerebras_judge           # Cerebras LLM judge (1-5 scoring)
      enabled: false                   # Requires EVAL_CEREBRAS_API_KEY in secrets.env
      save_report: false               # Save cerebras_judge_report.md with per-sample reasoning
      params:
        model: "gpt-oss-120b"          # Available: gpt-oss-120b, llama3.1-8b
        max_samples: 50                # Max samples (cost cap)
        temperature: 0.0               # Deterministic scores
        max_tokens: 512                # Enough for JSON + CoT reasoning
        max_retries: 3                 # Retries on rate limit / 5xx
      thresholds:
        min_mean_score: 0.6            # Normalized threshold (= score 3.4 on 1–5 scale)
```

The evaluation stage uses a **plugin-based** architecture (similar to dataset validators):

1. **EvaluationRunner** loads the eval dataset (JSONL) and collects model answers via `IModelInference` (HTTP client to the inference endpoint).
2. After collecting answers (if `save_answers_md: true`) — `runs/{run}/evaluation/answers.md` is saved with task / model answer / expected answer triples.
3. Collected `EvalSample` (question, model_answer, expected_answer) are passed to **plugins** in priority order.
4. Each plugin returns `EvalResult` (passed/failed, metrics, recommendations).
5. Aggregated `RunSummary` is logged to MLflow (metrics + `eval_results.json` artifact).

### Lifecycle when evaluation.enabled=true

| Step | Stage | Action |
|-----|-------|----------|
| 1 | InferenceDeployer | Calls `provider.activate_for_eval()` → live endpoint URL |
| 2 | ModelEvaluator | Creates inference client via `ModelClientFactory.create(engine, url, model)` |
| 3 | ModelEvaluator | Runs `EvaluationRunner` → collect answers → plugins → `RunSummary` |
| 4 | InferenceDeployer.cleanup() | Calls `provider.deactivate_after_eval()` (for runpod_pods: deletes Pod, keeps Volume) |

### 6.1 Basic parameters

| Parameter | Type | Default | Description |
|----------|-----|--------------|----------|
| `enabled` | bool | `false` | Enable eval stage. **Requires** `inference.enabled=true` |
| `dataset.path` | string | - | Path to JSONL eval dataset (absolute or relative) |
| `save_answers_md` | bool | `true` | Save `answers.md` with model answers under `runs/{run}/evaluation/` |

### 6.2 Eval dataset format (JSONL)

Two formats are supported:

**Flat format:**
```json
{"question": "...", "expected_answer": "...", "context": "..."}
```

**Chat/messages format** (compatible with training datasets):
```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

> `expected_answer` is optional. Plugins with `requires_expected_answer=True` are skipped if no sample contains expected_answer.

### 6.3 Evaluators — groups and plugins

**Structure:** `evaluators.{group_name}.{plugin_name}`

Each plugin:

| Parameter | Type | Default | Description |
|----------|-----|--------------|----------|
| `enabled` | bool | `true` | Enable/disable plugin |
| `save_report` | bool | `false` | Save `{plugin_name}_report.md` under `runs/{run}/evaluation/`. Plugins fill the report (e.g. `cerebras_judge` — per-sample reasoning, `helixql_syntax` — invalidity reason) |
| `params` | dict | `{}` | Plugin parameters (implementation-specific) |
| `thresholds` | dict | `{}` | Thresholds for pass/fail |

#### Available plugins

| Plugin | Group | Description | Thresholds |
|--------|--------|----------|------------|
| `helixql_syntax` | `syntax_check` | Checks HelixQL syntax in model answers | `min_valid_ratio` (0.0-1.0, default: 0.8) |
| `cerebras_judge` | `llm_judge` | LLM-as-judge via Cerebras API (1–5 scale, normalized to [0,1]) | `min_mean_score` (0.0-1.0, default: 0.6) |

**`cerebras_judge` params:**

| Parameter | Type | Default | Description |
|----------|-----|--------------|----------|
| `model` | string | `"gpt-oss-120b"` | Cerebras judge model. Available: `gpt-oss-120b`, `llama3.1-8b`, `qwen-3-235b-a22b-instruct-2507` |
| `max_samples` | int | `50` | Max samples for scoring (cost cap) |
| `temperature` | float | `0.0` | Generation temperature (0.0 recommended for stability) |
| `max_tokens` | int | `512` | Max tokens in judge response |
| `max_retries` | int | `3` | Retries on rate limit / 5xx errors |

**Scoring:**
- The judge returns integer 1–5 in JSON with `reasoning` (CoT explanation)
- Normalization: `normalized = (score - 1) / 4` → [0, 1]
- Metrics in `EvalResult`: `mean_score`, `p50_score`, `score_distribution` (distribution over 1..5), `sample_count`

**Requires:** `EVAL_CEREBRAS_API_KEY=<key>` in `secrets.env`

### 6.4 Inference Client Factory

The client for collecting model answers is selected by `inference.engine`:

| Engine | Client | Description |
|--------|--------|----------|
| `vllm` | `OpenAICompatibleInferenceClient` | OpenAI-compatible API (vLLM, etc.) |

> Add a new engine = add an entry in `_ENGINE_BUILDERS` (OCP, no ModelEvaluator changes).

### 6.5 Provider support

| Provider | `activate_for_eval()` | `deactivate_after_eval()` |
|----------|----------------------|--------------------------|
| `single_node` | Ok(existing_url) — endpoint already running | No-op — user-managed |
| `runpod` (Pods) | Starts pod → SSH tunnel → LoRA merge → vLLM → returns URL | Stops vLLM, closes tunnel, deletes Pod (Volume kept). Fallback: if `activate_for_eval` was not called but pod exists — deletes it too |
| `runpod_serverless` | Err — not supported (cold start incompatible with eval loop) | No-op |

---

## 7. Experiment Tracking - MLflow + HuggingFace Hub

All integrations live in one place: `experiment_tracking:`.

Rules:
- if the block is **omitted** → integration is off
- if the block is **present** → inner fields are **validated strictly** (even when `enabled: false`)

```yaml
experiment_tracking:
  mlflow:
    enabled: true
    tracking_uri: "http://localhost:5002"
    experiment_name: ryotenkai
    log_artifacts: true
    log_model: false
    # run_description_file: null

    system_metrics_sampling_interval: 5
    system_metrics_samples_before_logging: 1
    system_metrics_callback_enabled: false
    system_metrics_callback_interval: 10

  huggingface:
    enabled: true
    repo_id: "<your-org>/ryotenkai-model"
    private: true
```

### 7.1 MLflow (`experiment_tracking.mlflow`)

| Parameter | Type | Default | Description |
|----------|-----|--------------|----------|
| `enabled` | bool | - | Enable MLflow |
| `tracking_uri` | string | - | URI MLflow tracking server |
| `experiment_name` | string | - | Experiment name in MLflow |
| `log_artifacts` | bool | - | Log artifacts (files) |
| `log_model` | bool | - | Log full model artifact (can be very large) |
| `run_description_file` | string \| null | `null` | Path to `.md` run description file |
| `system_metrics_sampling_interval` | int | `5` | System metrics sampling interval (sec) |
| `system_metrics_samples_before_logging` | int | `1` | Samples to collect before logging |
| `system_metrics_callback_enabled` | bool | `false` | Enable manual SystemMetricsCallback (may hang on some cloud images) |
| `system_metrics_callback_interval` | int | `10` | Log system metrics every N training steps (if callback enabled) |

### 7.2 HuggingFace Hub (`experiment_tracking.huggingface`)

Env:
- `HF_TOKEN` — **required** (write access)

| Parameter | Type | Default | Description |
|----------|-----|--------------|----------|
| `enabled` | bool | - | Enable upload to HF Hub |
| `repo_id` | string | - | Full repo id: `username/model-name` |
| `private` | bool | - | Make the repository private |

---

## 8. Integration Test - Integration testing

Optional smoke test after training: run the chat app and issue a few requests.

If the `integration_test:` block is omitted — the stage is skipped.

```yaml
integration_test:
  chat_launch_command: "python src/chat/app.py --model-path {model_path}"
  test_queries:
    - "What is HelixQL?"
    - "Show me an example of vector search in HelixQL"
  timeout_seconds: 300
```

| Parameter | Type | Default | Description |
|----------|-----|--------------|----------|
| `chat_launch_command` | string \| null | `null` | Launch command template (`{model_path}` substituted) |
| `test_queries` | list[string] | `[]` | Queries for the smoke test |
| `timeout_seconds` | int | `300` | Stage timeout (sec) |

---

## Quick Reference

### v7.0 Minimal Valid Config

```yaml
model:
  name: Qwen/Qwen2.5-7B-Instruct
  torch_dtype: bfloat16  # ✅ REQUIRED
  trust_remote_code: true  # ✅ REQUIRED

training:
  provider: single_node
  type: qlora
  
  lora:
    # ✅ REQUIRED: 8 fields
    r: 16
    lora_alpha: 32
    lora_dropout: 0.05
    bias: none
    target_modules: all-linear
    use_dora: false
    use_rslora: false
    init_lora_weights: gaussian
  
  hyperparams:
    # ✅ REQUIRED: 5 core fields
    per_device_train_batch_size: 4
    gradient_accumulation_steps: 4
    learning_rate: 2.0e-4
    warmup_ratio: 0.05
    epochs: 3
  
  strategies:
    - strategy_type: sft

datasets:
  default:
    source_type: local
    source_local:
      local_paths:
        train: data/train.jsonl  # ✅ REQUIRED

providers:
  single_node:
    connect:
      ssh:
        alias: pc
    cleanup:
      cleanup_workspace: false
      keep_on_error: true
      on_interrupt: true
    training:
      workspace_path: /home/user/workspace
      docker_image: "ryotenkai/ryotenkai-training-runtime:latest"

inference:
  enabled: false
  engines:
    vllm:
      # merge_image/serve_image required only for inference.provider=single_node when inference.enabled=true
      # merge_image: ryotenkai/inference-vllm:v1.0
      # serve_image: ryotenkai/inference-vllm:v1.0
```

---

**Document version:** 7.2
**Updated:** March 9, 2026

### Changes in v7.2:

- **`providers.runpod.training.gpu_type`** → now **REQUIRED** (default `NVIDIA A40` removed — must be set explicitly)
- **`providers.runpod.training.cloud_type`** → `COMMUNITY` recommended for cost (~20–30% cheaper than `SECURE`)
- **Evaluation `save_plugin_reports`** → global flag removed. Use per-plugin `save_report: bool`
- **`helixql_syntax`** → supports `save_report` (report includes invalidity reason per sample)
- **`cerebras_judge` default model** → changed from `llama-3.3-70b` (non-existent) to `gpt-oss-120b`
- **Available Cerebras models** (March 2026): `gpt-oss-120b`, `llama3.1-8b`, `qwen-3-235b-a22b-instruct-2507`, `zai-glm-4.7`
- **`runpod` Pods `activate_for_eval`/`deactivate_after_eval`** → fully implemented (SSH tunnel, LoRA merge, vLLM)
- **`deactivate_after_eval` fallback** → on Ctrl+C before `activate_for_eval`, pod is deleted via API
- **`validate_eval_plugin_secrets`** → fail-fast at startup if enabled plugin is missing secret

### Changes in v7.1:

- **Evaluation stage (NEW):**
  - New top-level `evaluation:` block — plugin-based evaluation of the trained model
  - Plugin: `helixql_syntax` — HelixQL syntax validity
  - Plugin: `cerebras_judge` — LLM-as-judge via Cerebras API (1–5 scale)
  - `ModelClientFactory` — extensible inference client factory per engine
  - `IntegrationTest` stage **removed** (replaced by `evaluation` stage)
  - Fail-fast: `evaluation.enabled=true` requires `inference.enabled=true`
  - Fail-fast: an enabled plugin with a missing EVAL_* secret stops the pipeline at startup

### Changes in v7.0 (BREAKING CHANGES):

- **Config modularization:** split into modules under `src/config/`
- **Unified secret loading:** file-first policy, `secrets.env` over `os.environ`

### Changes in v6.0 (BREAKING CHANGES):

- `torch_dtype`, `trust_remote_code` → **REQUIRED**
- LoRA: 8 fields → **REQUIRED**
- AdaLoRA: 6 fields → **REQUIRED**
- Hyperparameters: Global (5 core REQUIRED) + Phase (all optional)
- `training_paths` **REMOVED** (auto-generated)
- `merge_image`, `serve_image` → **REQUIRED** (`inference.provider=single_node` only)
