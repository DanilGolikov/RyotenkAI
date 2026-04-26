<p align="center">
  <img src="docs/logo_final.png" alt="RyotenkAI" width="400">
</p>
<h1 align="center">RyotenkAI</h1>

<p align="center">
  Declarative LLM fine-tuning.<br>
  Provide a YAML config and datasets, and RyotenkAI orchestrates validation, GPU provisioning, training, inference deployment, evaluation, and reporting.
</p>

<p align="center">
  🇬🇧 English |
  <a href="i18n/README.ru.md">🇷🇺 Русский</a> |
  <a href="i18n/README.ja.md">🇯🇵 日本語</a> |
  <a href="i18n/README.zh-CN.md">🇨🇳 简体中文</a> |
  <a href="i18n/README.ko.md">🇰🇷 한국어</a> |
  <a href="i18n/README.es.md">🇪🇸 Español</a> |
  <a href="i18n/README.he.md">🇮🇱 עברית</a>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> ·
  <a href="#how-it-works">How It Works</a> ·
  <a href="#training-strategies">Strategies</a> ·
  <a href="#gpu-providers">Providers</a> ·
  <a href="#plugin-systems">Plugins</a> ·
  <a href="#configuration">Configuration</a>
</p>

<p align="center">
  <a href="https://discord.gg/QqDM2DbY">
    <img src="https://img.shields.io/badge/Discord-Join%20Community-5865F2?logo=discord&logoColor=white" alt="Join our Discord">
  </a>
  <br>
  <img src="https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white" alt="Python 3.12+">
  <img src="https://img.shields.io/badge/PyTorch-2.5-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/%F0%9F%A4%97_Transformers-4.x-FFD21E" alt="Transformers">
  <img src="https://img.shields.io/badge/TRL-multi--strategy-blueviolet" alt="TRL">
  <img src="https://img.shields.io/badge/PEFT-LoRA%20%7C%20QLoRA-8A2BE2" alt="PEFT">
  <br>
  <img src="https://img.shields.io/badge/vLLM-inference-00ADD8?logo=v&logoColor=white" alt="vLLM">
  <img src="https://img.shields.io/badge/MLflow-tracking-0194E2?logo=mlflow&logoColor=white" alt="MLflow">
  <img src="https://img.shields.io/badge/Docker-containerized-2496ED?logo=docker&logoColor=white" alt="Docker">
  <img src="https://img.shields.io/badge/RunPod-cloud_GPU-673AB7" alt="RunPod">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License: MIT">
</p>

---

## What is RyotenkAI?

RyotenkAI is a declarative control plane for LLM fine-tuning. You describe the workflow in YAML, provide datasets, and RyotenkAI executes the full lifecycle: dataset validation, GPU provisioning, multi-phase training, model retrieval, inference deployment, evaluation, and experiment reporting in MLflow.

| Manual fine-tuning workflow | RyotenkAI |
|---|---|
| Manually check dataset quality | Plugin-based validation: format, duplicates, length, diversity |
| SSH into GPU, run scripts | One command: provisions GPU, deploys training, monitors |
| Wait and hope it works | Real-time monitoring: GPU metrics, loss curves, OOM detection |
| Download weights manually | Auto-retrieves adapters, merges LoRA, publishes to HF Hub |
| Spin up inference server | Deploys vLLM endpoint with health checks |
| Test outputs by hand | Plugin-based evaluation: syntax, semantic match, LLM-as-judge |
| Write notes in a doc | MLflow experiment tracking + generated Markdown report |

---

## How It Works

### Pipeline Flow

```
YAML Config
    │
    ▼
┌─────────────────┐
│ Dataset Validator│  Validate data quality before training
│ (plugin system)  │  min_samples, diversity, format, custom
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  GPU Deployer    │  Provision compute (SSH or RunPod API)
│                  │  Deploy training container + code
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│Training Monitor  │  Track process, parse logs, detect OOM
│                  │  GPU metrics, loss curves, health checks
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Model Retriever  │  Download adapters / merged weights
│                  │  Optionally publish to HuggingFace Hub
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│Inference Deployer│  Start vLLM server in Docker
│                  │  Health checks, OpenAI-compatible API
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Model Evaluator  │  Run evaluation plugins on live endpoint
│ (plugin system)  │  syntax, semantic match, LLM judge, custom
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│Report Generator  │  Collect all data from MLflow
│ (plugin system)  │  Render Markdown experiment report
└─────────────────┘
```

### Training Execution

```
Pipeline (control plane)              GPU Provider (single_node / RunPod)
         │                                         │
    SSH / API ──────────────────────────► Docker container
         │                                   │
    rsync code ─────────────────────────►  /workspace/
         │                                   │
    start training ─────────────────────►  accelerate launch train.py
         │                                   │
    monitor ◄───────────── logs, markers, GPU metrics
         │                                   │
    retrieve artifacts ◄────── adapters, checkpoints, merged weights
```

### Training Strategy Chain (inside GPU container)

Multi-phase training with automatic state management, OOM recovery, and checkpointing:

```
run_training(config.yaml)
  │
  ├── MemoryManager.auto_configure()     Detect GPU tier, set VRAM thresholds
  │     └── GPUPreset: margin, critical%, max_retries
  │
  ├── load_model_and_tokenizer()         Load base model (no PEFT yet)
  │     └── MemoryManager: snapshot before/after, CUDA cache cleanup
  │
  ├── DataBuffer.init_pipeline()         Initialize state tracking
  │     └── pipeline_state.json          Phase statuses, checkpoint paths
  │     └── phase_0_sft/                 Per-phase output directories
  │     └── phase_1_dpo/
  │
  └── ChainRunner.run(strategies)        Execute phase chain
        │
        │   For each phase (e.g. CPT → SFT → DPO):
        │
        ▼
  ┌─────────────────────────────────────────────────────────┐
  │  PhaseExecutor.execute(phase_idx, phase, model, buffer) │
  │                                                         │
  │  1. buffer.mark_phase_started(idx)                      │
  │     └── Atomic state save to pipeline_state.json        │
  │                                                         │
  │  2. StrategyFactory.create(phase.strategy_type)         │
  │     ├── SFTStrategy     (messages → instruction tuning) │
  │     ├── DPOStrategy     (chosen/rejected → alignment)   │
  │     ├── ORPOStrategy    (preference → odds ratio)       │
  │     ├── GRPOStrategy    (reward-guided RL)              │
  │     ├── SAPOStrategy    (self-aligned preference)       │
  │     └── CPTStrategy     (raw text → domain adaptation)  │
  │                                                         │
  │  3. dataset_loader.load_for_phase(phase)                │
  │     └── strategy.validate_dataset + prepare_dataset     │
  │                                                         │
  │  4. TrainerFactory.create_from_phase(...)               │
  │     ├── strategy.get_trainer_class() → TRL Trainer      │
  │     ├── Merge hyperparams: global ∪ phase overrides     │
  │     ├── Create PEFT config (LoRA / QLoRA / AdaLoRA)     │
  │     ├── Attach callbacks (MLflow, GPU metrics)          │
  │     └── Wrapped in MemoryManager.with_memory_protection │
  │                                                         │
  │  5. trainer.train()                                     │
  │     └── MemoryManager.with_memory_protection            │
  │           ├── Monitor VRAM usage                        │
  │           ├── On OOM → aggressive_cleanup + retry       │
  │           └── max_retries from GPU tier preset          │
  │                                                         │
  │  6. Save checkpoint-final                               │
  │     ├── buffer.mark_phase_completed(metrics)            │
  │     └── buffer.cleanup_old_checkpoints(keep_last=2)     │
  │                                                         │
  └─────────────────────┬───────────────────────────────────┘
                        │
                        ▼  model passed to next phase (in-memory)
                        │
               ┌────────┴────────┐
               │  Next phase?    │
               │  idx < total    │──── No ──► return trained model
               └────────┬────────┘
                        │ Yes
                        ▼
                 (repeat PhaseExecutor)
```

### DataBuffer — state management between phases

```
DataBuffer
  │
  ├── Pipeline State (pipeline_state.json)
  │     {
  │       "status": "running",
  │       "phases": [
  │         { "strategy": "sft", "status": "completed", "checkpoint": "phase_0_sft/checkpoint-final" },
  │         { "strategy": "dpo", "status": "running",   "checkpoint": null }
  │       ]
  │     }
  │
  ├── Phase Directories
  │     output/
  │     ├── phase_0_sft/
  │     │   ├── checkpoint-500/     (intermediate, auto-cleaned)
  │     │   ├── checkpoint-1000/    (intermediate, auto-cleaned)
  │     │   └── checkpoint-final/   (kept — input for next phase)
  │     └── phase_1_dpo/
  │         └── checkpoint-final/
  │
  ├── Resume Logic
  │     On crash/restart:
  │       1. load_state() → find first non-completed phase
  │       2. get_model_path_for_phase(idx) → previous checkpoint-final
  │       3. Load PEFT adapters onto base model
  │       4. get_resume_checkpoint(idx) → mid-phase checkpoint (if any)
  │       5. Continue training from where it stopped
  │
  └── Cleanup
        cleanup_old_checkpoints(keep_last=2)
        Removes intermediate checkpoint-N/ dirs, preserves checkpoint-final
```

### MemoryManager — GPU OOM protection

```
MemoryManager.auto_configure()
  │
  ├── Detect GPU: name, VRAM, compute capability
  │     ├── RTX 4060  (8GB)  → consumer_low   tier
  │     ├── RTX 4090  (24GB) → consumer_high  tier
  │     ├── A100      (80GB) → datacenter     tier
  │     └── Unknown          → safe fallback
  │
  ├── GPUPreset per tier:
  │     margin_mb:    reserved VRAM headroom (512–4096 MB)
  │     critical_pct: trigger OOM recovery (85–95%)
  │     warning_pct:  log warning (70–85%)
  │     max_retries:  auto-retry attempts (1–3)
  │
  └── with_memory_protection(operation):
        ┌─────────────────────────────┐
        │  Attempt 1                  │
        │  ├── Check VRAM headroom    │
        │  ├── Run operation          │
        │  └── Success → return       │
        │                             │
        │  OOM detected?              │
        │  ├── aggressive_cleanup()   │
        │  │   ├── gc.collect()       │
        │  │   ├── torch.cuda.empty_cache()
        │  │   └── Clear gradients    │
        │  ├── Log OOM event (MLflow) │
        │  └── Retry (up to max)      │
        │                             │
        │  All retries exhausted?     │
        │  └── OOMRecoverableError    │
        └─────────────────────────────┘
```

### Evaluation Flow

```
EvaluationRunner
  1. Load JSONL eval dataset → list of (question, expected_answer, metadata)
  2. Collect model answers via vLLM endpoint → list[EvalSample]
  3. For each enabled plugin (sorted by priority):
       result = plugin.evaluate(samples)
  4. Aggregate results → RunSummary (passed/failed, metrics, recommendations)
```

### Report Generation Flow

```
ryotenkai report <run_dir>
  │
  ▼
MLflow ──► pull runs, metrics, artifacts, configs
  │
  ▼
Build report model (phases, issues, timeline)
  │
  ▼
Run plugins (each renders one section of the report)
  │
  ▼
Render Markdown → experiment_report.md
  │
  └── logged back to MLflow as artifact
```

---

## Quick Start

### One-command setup

```bash
git clone https://github.com/DanilGolikov/ryotenkai.git
cd ryotenkai
bash setup.sh
source .venv/bin/activate
```

### Configure

1. Edit `secrets.env` with your API keys (RunPod, HuggingFace)
2. Copy and customize the example config:

```bash
cp src/config/pipeline_config.yaml my_config.yaml
# Edit my_config.yaml with your model, dataset, and provider settings
```

### Run

```bash
# Validate your config
ryotenkai config validate --config my_config.yaml

# Start full pipeline
ryotenkai run start --config my_config.yaml

# Stream logs in another terminal
ryotenkai runs logs runs/<run_id> --follow
```

> **CLI surface follows the kubectl-style `noun verb` convention.**
> Old plain commands (`ryotenkai train`, `runs-list`, `inspect-run`,
> `config-validate`, `validate-dataset`, ...) and the legacy `./run.sh`
> wrapper were removed in v1.1. See the migration table below.

---

## Configuration

RyotenkAI uses a single YAML config file (schema v7). Key sections:

```yaml
model:
  name: "Qwen/Qwen2.5-0.5B-Instruct"

training:
  type: qlora                    # qlora | lora | adalora | full
  provider: single_node          # single_node | runpod
  strategies:
    - strategy_type: sft
      hyperparams: { epochs: 3 }
    - strategy_type: dpo
      hyperparams: { epochs: 1 }

datasets:
  default:
    source_hf:
      train_id: "your-org/dataset"

providers:
  single_node:
    connect:
      ssh: { alias: pc }
    training:
      workspace_path: /home/user/workspace
      docker_image: "ryotenkai/ryotenkai-training-runtime:latest"

mlflow:
  tracking_uri: "http://localhost:5002"
  experiment_name: ryotenkai
```

Full config reference: [`src/config/CONFIG_REFERENCE.md`](src/config/CONFIG_REFERENCE.md)

---

## Training Strategies

Multi-phase training with strategy chaining. Strategies define **what** to train; adapters (LoRA, QLoRA, AdaLoRA, Full FT) define **how**.

| Strategy | Signal | Use Case |
|----------|--------|----------|
| **CPT** (Continued Pre-Training) | raw text | Inject domain knowledge |
| **SFT** (Supervised Fine-Tuning) | instruction → response pairs | Teach the model a task format |
| **CoT** (Chain-of-Thought) | reasoning traces | Improve step-by-step reasoning |
| **DPO** (Direct Preference Optimization) | chosen vs rejected pairs | Align with human preferences |
| **ORPO** (Odds Ratio Preference Optimization) | chosen vs rejected pairs | Alignment without separate reward model |
| **GRPO** (Group Relative Policy Optimization) | reward-guided RL | Reinforcement learning from rewards |
| **SAPO** (Self-Aligned Preference Optimization) | chosen vs rejected + self-alignment | Improved preference learning |

Strategies can be chained: `CPT → SFT → DPO` runs sequentially, each phase building on the previous checkpoint. The chain is fully configurable in YAML.

---

## GPU Providers

Providers handle GPU provisioning for training and inference. Both training and inference have separate provider interfaces.

| Provider | Type | Training | Inference | How it connects |
|----------|------|----------|-----------|-----------------|
| **single_node** | Local | SSH to your GPU server | vLLM via Docker over SSH | `~/.ssh/config` alias or explicit host/port/key |
| **RunPod** | Cloud | Pod via GraphQL API | Volume + Pod provisioning | API key in `secrets.env` |

### single_node

Direct SSH access to a machine with a GPU. The pipeline deploys a Docker container with the training runtime, syncs code, runs training, and retrieves artifacts — all over SSH. Inference deploys a vLLM container on the same host.

Features: auto GPU detection (`nvidia-smi`), health checks, workspace cleanup.

### RunPod

Cloud GPU via RunPod API. The pipeline creates a pod with the requested GPU type, waits for SSH readiness, runs training, and optionally deletes the pod on completion. Inference provisions a persistent volume and a separate pod.

Features: spot instances, multiple GPU types, auto-cleanup (`cleanup.auto_delete_pod`).

---

## Plugin Systems

RyotenkAI has three plugin systems, all following the same pattern: `@register` decorator, auto-discovery, and namespace-isolated secrets via `secrets.env`.

### Dataset Validation

Validates datasets before training starts. Plugins check format, quality, diversity, and domain-specific constraints. Runs as the first pipeline stage — training won't start if validation fails.

Secrets namespace: `DTST_*` — Docs: [`src/data/validation/README.md`](src/data/validation/README.md)

### Evaluation

Evaluates model quality after training against a live vLLM endpoint. Plugins run deterministic checks (syntax, semantic match) and LLM-as-judge scoring. Results feed into the experiment report.

Secrets namespace: `EVAL_*` — Docs: [`src/evaluation/plugins/README.md`](src/evaluation/plugins/README.md)

### Report Generation

Generates experiment reports from MLflow data. Each plugin renders one section of a Markdown document (header, summary, metrics, issues, etc.). The final report is logged back to MLflow as an artifact.

Docs: [`src/reports/plugins/README.md`](src/reports/plugins/README.md)

All plugin systems support custom plugins — implement the base class, decorate with `@register`, and the pipeline discovers them automatically.

---

## MLflow Integration

Start the MLflow stack:

```bash
make docker-mlflow-up
```

Access the UI at `http://localhost:5002`. All pipeline runs are tracked with metrics, artifacts, and config snapshots.

---

## Docker Images

| Image | Purpose |
|-------|---------|
| `ryotenkai/ryotenkai-training-runtime` | CUDA + PyTorch + dependencies for training |
| `ryotenkai/inference-vllm` | vLLM inference runtime (serve + merge deps + SSH) |

Build locally or push to Docker Hub. See [`docker/training/README.md`](docker/training/README.md) and [`docker/inference/README.md`](docker/inference/README.md).

---

## CLI Reference

The CLI follows kubectl-style `noun verb`. Every read command supports
`-o text|json|yaml`; write commands support `--dry-run`.

### Workflow

| Command | Description |
|---------|-------------|
| `ryotenkai run start --config <path>` | Start a fresh training run |
| `ryotenkai run resume <run_dir>` | Resume from the first failed/pending stage |
| `ryotenkai run restart <run_dir> --from-stage <name|N>` | Restart from a specific stage |
| `ryotenkai run interrupt <run_dir>` | Send SIGINT to a detached run |
| `ryotenkai run restart-points <run_dir>` | List stages this run can restart from |

### Inspect

| Command | Description |
|---------|-------------|
| `ryotenkai runs ls [dir]` | List all runs with summary |
| `ryotenkai runs inspect <run_dir>` | Inspect a run (attempts, stages, lineage) |
| `ryotenkai runs status <run_dir>` | One snapshot or live polling |
| `ryotenkai runs diff <run_dir>` | Compare config hashes between attempts |
| `ryotenkai runs logs <run_dir> [--follow]` | Read or tail pipeline.log |
| `ryotenkai runs report <run_dir>` | Generate the MLflow experiment report |
| `ryotenkai runs rm <run_dir>` | Delete a run (local + MLflow by default) |

### Config / dataset

| Command | Description |
|---------|-------------|
| `ryotenkai config validate --config <path>` | Static config pre-flight checks |
| `ryotenkai config show --config <path>` | Print parsed config (`-o yaml` recommended) |
| `ryotenkai config explain --config <path>` | Short summary of model + dataset + training |
| `ryotenkai config schema` | Dump JSON Schema for the pipeline config |
| `ryotenkai dataset validate --config <path>` | Run dataset Stage 0 (requires explicit plugins) |

### Project

| Command | Description |
|---------|-------------|
| `ryotenkai project ls` | List registered workspace projects |
| `ryotenkai project show <id>` | Project metadata + config |
| `ryotenkai project use <id>` | Make `<id>` the active project for follow-up commands |
| `ryotenkai project current` | Print the active project (or "no project selected") |
| `ryotenkai project create <name>` | Create + register a new project |
| `ryotenkai project rm <id>` | Unregister a project |
| `ryotenkai project env <id>` | Print persisted env vars |
| `ryotenkai project run [<id>]` | Launch the project's current config |

### Plugin / preset toolkit

| Command | Description |
|---------|-------------|
| `ryotenkai plugin ls --kind <kind>` | List installed plugins by kind |
| `ryotenkai plugin show <kind> <id>` | Full manifest for a plugin |
| `ryotenkai plugin scaffold <kind> <id>` | Bootstrap a new plugin folder + skeleton |
| `ryotenkai plugin sync <path> [--bump]` | Refresh manifest from class introspection |
| `ryotenkai plugin sync-envs <path>` | Re-render `[[required_env]]` from `REQUIRED_ENV` |
| `ryotenkai plugin pack <path>` | Zip a plugin/preset folder for distribution |
| `ryotenkai plugin validate <path>` | Validate manifest.toml (no Python import) |
| `ryotenkai plugin install <path \| --git URL --ref SHA>` | Install a plugin from folder, zip, or git |
| `ryotenkai plugin preflight --config <path>` | Pre-launch missing-env + instance-shape gate |
| `ryotenkai plugin stale --config <path>` | List references to plugins absent from the catalog |
| `ryotenkai preset ls / show / apply / diff` | Discover and apply community presets |

### Misc

| Command | Description |
|---------|-------------|
| `ryotenkai smoke <dir>` | Batch-run every config in a directory in parallel |
| `ryotenkai server start` | Run the FastAPI web backend (foreground) |
| `ryotenkai version` | Show version info |

### Migration from v1.0 plain commands

| Old (removed) | New |
|---------------|-----|
| `./run.sh config.yaml` | `ryotenkai run start --config config.yaml` |
| `./run.sh runs/<id> --resume` | `ryotenkai run resume runs/<id>` |
| `./run.sh runs/<id> --inspect` | `ryotenkai runs inspect runs/<id>` |
| `./run.sh /path --smoke` | `ryotenkai smoke /path` |
| `ryotenkai train` | `ryotenkai run start` |
| `ryotenkai train-local` | `ryotenkai run start --local` (planned for v1.2) |
| `ryotenkai validate-dataset` | `ryotenkai dataset validate` |
| `ryotenkai config-validate` | `ryotenkai config validate` |
| `ryotenkai inspect-run`, `runs-list`, `logs`, `run-status`, `run-diff`, `report` | `ryotenkai runs <verb>` |
| `ryotenkai list-restart-points` | `ryotenkai run restart-points` |
| `ryotenkai community {scaffold,sync,sync-envs,pack}` | `ryotenkai plugin <verb>` |
| `ryotenkai serve` | `ryotenkai server start` |

---

## Web UI

Browser-based control plane for the pipeline. The FastAPI backend and React frontend are sibling clients to the same file-based state store used by the CLI — they don't wrap the CLI via subprocess.

```bash
# Backend
ryotenkai serve --runs-dir runs --port 8000      # OpenAPI at :8000/docs

# Frontend (dev)
cd web && npm install && npm run dev             # Vite on :5173, proxies /api to :8000

# Frontend (prod — served by FastAPI)
cd web && npm run build
cd .. && ryotenkai serve                         # mounts web/dist at /
```

See [docs/web-ui.md](docs/web-ui.md) for the full HTTP/WebSocket contract and architecture notes.

---

## Development

### Setup

```bash
bash setup.sh
source .venv/bin/activate
```

### Tests

```bash
make test          # all tests
make test-unit     # unit tests only
make test-fast     # skip slow tests
make test-cov      # with coverage
```

### Linting

```bash
make lint          # check
make format        # auto-format
make fix-all       # auto-fix
```

### Pre-commit

Pre-commit hooks run automatically. To run manually:

```bash
make pre-commit
```

---

## Project Structure

```
ryotenkai/
├── src/
│   ├── config/          # Configuration schemas (Pydantic v2)
│   ├── pipeline/        # Orchestration and stage implementations
│   ├── training/        # Training strategies and orchestration
│   ├── providers/       # GPU providers (single_node, RunPod)
│   ├── evaluation/      # Model evaluation plugins
│   ├── data/            # Dataset handling and validation plugins
│   ├── reports/         # Report generation plugins
│   ├── utils/           # Shared utilities
│   └── tests/           # Test suite
├── docker/
│   ├── training/        # Training runtime Docker image
│   ├── inference/       # Inference Docker images
│   └── mlflow/          # MLflow stack (docker-compose)
├── scripts/             # Utility scripts
├── docs/                # Documentation and diagrams
├── setup.sh             # One-command setup
├── Makefile             # Development commands
└── pyproject.toml       # Package metadata and tool config
```

## Community

Join the Discord server for support, roadmap discussion, configs, and fine-tuning workflow talk:

[discord.gg/QqDM2DbY](https://discord.gg/QqDM2DbY)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[MIT](LICENSE) © Golikov Daniil
