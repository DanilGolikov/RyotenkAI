<p align="center">
  <img src="../logo_final.png" alt="RyotenkAI" width="400">
</p>
<h1 align="center">RyotenkAI</h1>

<p align="center">
  声明式 LLM fine-tuning。<br>
  只需提供 YAML 配置和数据集，RyotenkAI 就会编排验证、GPU 准备、训练、推理部署、评估和报告生成的完整流程。
</p>

<p align="center">
  <a href="../../README.md">🇬🇧 English</a> |
  <a href="README.ru.md">🇷🇺 Русский</a> |
  <a href="README.ja.md">🇯🇵 日本語</a> |
  🇨🇳 简体中文 |
  <a href="README.ko.md">🇰🇷 한국어</a> |
  <a href="README.es.md">🇪🇸 Español</a> |
  <a href="README.he.md">🇮🇱 עברית</a>
</p>

<p align="center">
  <a href="#快速开始">快速开始</a> ·
  <a href="#工作原理">工作原理</a> ·
  <a href="#训练策略">训练策略</a> ·
  <a href="#gpu-提供方">GPU 提供方</a> ·
  <a href="#插件系统">插件系统</a> ·
  <a href="#配置">配置</a>
</p>

<p align="center">
  <a href="https://discord.gg/QqDM2DbY">
    <img src="https://img.shields.io/badge/Discord-加入社区-5865F2?logo=discord&logoColor=white" alt="加入 Discord">
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

## RyotenkAI 是什么

RyotenkAI 是一个面向 LLM fine-tuning 的声明式 control plane。你用 YAML 描述工作流并提供数据集后，RyotenkAI 会执行完整生命周期：数据集验证、GPU 资源准备、多阶段训练、模型回收、推理部署、评估，以及在 MLflow 中生成实验报告。

| 手工 fine-tuning 工作流 | RyotenkAI |
|---|---|
| 手动检查数据集质量 | 基于插件的验证：format、duplicates、length、diversity |
| SSH 到 GPU 上运行脚本 | 一条命令完成 GPU 准备、训练部署和监控 |
| 等待并祈祷一切正常 | 实时监控 GPU 指标、loss curve 和 OOM 检测 |
| 手动下载权重 | 自动获取 adapters，merge LoRA，并可发布到 HF Hub |
| 单独搭建 inference server | 部署带 health checks 的 vLLM endpoint |
| 手工检查模型输出 | 基于插件的评估：syntax、semantic match、LLM-as-judge |
| 在文档里手写实验记录 | MLflow 实验跟踪 + 自动生成 Markdown 报告 |

---

## 工作原理

### Pipeline 流程

```text
YAML Config
    │
    ▼
┌─────────────────┐
│ Dataset Validator│  在训练前验证数据质量
│ (plugin system)  │  min_samples, diversity, format, custom
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  GPU Deployer    │  准备计算资源 (SSH 或 RunPod API)
│                  │  部署 training container 和代码
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│Training Monitor  │  跟踪进程、解析日志、检测 OOM
│                  │  GPU metrics, loss curves, health checks
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Model Retriever  │  下载 adapters / merged weights
│                  │  可选发布到 HuggingFace Hub
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│Inference Deployer│  在 Docker 中启动 vLLM server
│                  │  Health checks, OpenAI-compatible API
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Model Evaluator  │  在 live endpoint 上运行评估插件
│ (plugin system)  │  syntax, semantic match, LLM judge, custom
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│Report Generator  │  从 MLflow 收集全部数据
│ (plugin system)  │  生成 Markdown 实验报告
└─────────────────┘
```

### 训练执行流程

```text
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

### 训练策略链 (GPU container 内部)

支持自动状态管理、OOM 恢复和 checkpoint 管理的多阶段训练：

```text
run_training(config.yaml)
  │
  ├── MemoryManager.auto_configure()     检测 GPU tier 并设置 VRAM 阈值
  │     └── GPUPreset: margin, critical%, max_retries
  │
  ├── load_model_and_tokenizer()         加载基础模型 (此时还没有 PEFT)
  │     └── MemoryManager: 前后 snapshot 与 CUDA cache cleanup
  │
  ├── DataBuffer.init_pipeline()         初始化状态追踪
  │     └── pipeline_state.json          各阶段状态与 checkpoint 路径
  │     └── phase_0_sft/                 各阶段输出目录
  │     └── phase_1_dpo/
  │
  └── ChainRunner.run(strategies)        执行 phase chain
        │
        │   对于每个 phase (例如 CPT → SFT → DPO):
        │
        ▼
  ┌─────────────────────────────────────────────────────────┐
  │  PhaseExecutor.execute(phase_idx, phase, model, buffer) │
  │                                                         │
  │  1. buffer.mark_phase_started(idx)                      │
  │     └── 原子化保存状态到 pipeline_state.json            │
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
  │     ├── 合并 hyperparams: global ∪ phase overrides      │
  │     ├── 创建 PEFT config (LoRA / QLoRA / AdaLoRA)       │
  │     ├── 挂载 callbacks (MLflow, GPU metrics)            │
  │     └── 包裹在 MemoryManager.with_memory_protection 中   │
  │                                                         │
  │  5. trainer.train()                                     │
  │     └── MemoryManager.with_memory_protection            │
  │           ├── 监控 VRAM 使用量                          │
  │           ├── OOM 时执行 aggressive_cleanup + retry     │
  │           └── max_retries 取自 GPU tier preset          │
  │                                                         │
  │  6. 保存 checkpoint-final                               │
  │     ├── buffer.mark_phase_completed(metrics)            │
  │     └── buffer.cleanup_old_checkpoints(keep_last=2)     │
  │                                                         │
  └─────────────────────┬───────────────────────────────────┘
                        │
                        ▼  模型以内存形式传给下一阶段
                        │
               ┌────────┴────────┐
               │  还有下一阶段?   │
               │  idx < total    │──── No ──► 返回训练后的模型
               └────────┬────────┘
                        │ Yes
                        ▼
                 (重复 PhaseExecutor)
```

### DataBuffer - 阶段间状态管理

```text
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
  │     │   ├── checkpoint-500/     (中间 checkpoint，自动清理)
  │     │   ├── checkpoint-1000/    (中间 checkpoint，自动清理)
  │     │   └── checkpoint-final/   (保留，作为下一阶段输入)
  │     └── phase_1_dpo/
  │         └── checkpoint-final/
  │
  ├── Resume Logic
  │     当崩溃或重启时:
  │       1. load_state() → 找到第一个未完成的 phase
  │       2. get_model_path_for_phase(idx) → 上一个 checkpoint-final
  │       3. 将 PEFT adapters 加载到基础模型上
  │       4. get_resume_checkpoint(idx) → 当前 phase 的中间 checkpoint (如果有)
  │       5. 从中断位置继续训练
  │
  └── Cleanup
         cleanup_old_checkpoints(keep_last=2)
         删除中间 checkpoint-N/ 目录，保留 checkpoint-final
```

### MemoryManager - GPU OOM 保护

```text
MemoryManager.auto_configure()
  │
  ├── 检测 GPU: 名称、VRAM、compute capability
  │     ├── RTX 4060  (8GB)  → consumer_low   tier
  │     ├── RTX 4090  (24GB) → consumer_high  tier
  │     ├── A100      (80GB) → datacenter     tier
  │     └── Unknown          → safe fallback
  │
  ├── 每个 tier 对应的 GPUPreset:
  │     margin_mb:    预留 VRAM 余量 (512-4096 MB)
  │     critical_pct: 触发 OOM recovery 的阈值 (85-95%)
  │     warning_pct:  输出 warning 的阈值 (70-85%)
  │     max_retries:  自动重试次数 (1-3)
  │
  └── with_memory_protection(operation):
         ┌─────────────────────────────┐
         │  Attempt 1                  │
         │  ├── 检查 VRAM 余量         │
         │  ├── 执行 operation         │
         │  └── Success → return       │
         │                             │
         │  OOM detected?              │
         │  ├── aggressive_cleanup()   │
         │  │   ├── gc.collect()       │
         │  │   ├── torch.cuda.empty_cache()
         │  │   └── 清理 gradients     │
         │  ├── 将 OOM event 记录到 MLflow
         │  └── 重试 (直到 max)        │
         │                             │
         │  所有重试都失败?            │
         │  └── OOMRecoverableError    │
         └─────────────────────────────┘
```

### 评估流程

```text
EvaluationRunner
  1. 加载 JSONL eval dataset → (question, expected_answer, metadata) 列表
  2. 通过 vLLM endpoint 收集模型回答 → list[EvalSample]
  3. 按 priority 顺序运行每个启用的 plugin:
       result = plugin.evaluate(samples)
  4. 聚合结果 → RunSummary (passed/failed, metrics, recommendations)
```

### 报告生成流程

```text
ryotenkai runs report <run_dir>
  │
  ▼
MLflow ──► 拉取 runs、metrics、artifacts、configs
  │
  ▼
构建报告模型 (phases, issues, timeline)
  │
  ▼
运行 plugins (每个插件渲染一个报告章节)
  │
  ▼
渲染 Markdown → experiment_report.md
  │
  └── 再作为 artifact 记录回 MLflow
```

---

## 快速开始

### 一条命令完成安装

```bash
git clone https://github.com/DanilGolikov/ryotenkai.git
cd ryotenkai
bash setup.sh
source .venv/bin/activate
```

### 配置

1. 在 `secrets.env` 中填入你的 API keys (RunPod, HuggingFace)
2. 复制示例配置并按需修改

```bash
cp src/config/pipeline_config.yaml my_config.yaml
# 在 my_config.yaml 中编辑模型、数据集和 provider 设置
```

### 运行

```bash
# 验证配置
ryotenkai config validate --config my_config.yaml

# 启动完整 pipeline
ryotenkai run start --config my_config.yaml

# 或在本地运行训练 (用于开发)
ryotenkai run start --local --config my_config.yaml
```

### 交互式 TUI

```bash
ryotenkai tui
```

TUI 提供可导航的仪表盘，用于浏览 runs、查看各 stage 状态，以及监控 live pipelines。

---

## 配置

RyotenkAI 使用单个 YAML 配置文件 (schema v7)。关键部分如下：

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

完整配置参考：[`../src/config/CONFIG_REFERENCE.md`](../../src/config/CONFIG_REFERENCE.md)

---

## 训练策略

RyotenkAI 支持通过 strategy chaining 进行多阶段训练。strategies 定义 **训练什么**，adapters (LoRA、QLoRA、AdaLoRA、Full FT) 定义 **如何训练**。

| 策略 | 信号 | 使用场景 |
|------|------|----------|
| **CPT** (Continued Pre-Training) | raw text | 注入领域知识 |
| **SFT** (Supervised Fine-Tuning) | instruction → response pairs | 教会模型任务格式 |
| **CoT** (Chain-of-Thought) | reasoning traces | 提升逐步推理能力 |
| **DPO** (Direct Preference Optimization) | chosen vs rejected pairs | 对齐人类偏好 |
| **ORPO** (Odds Ratio Preference Optimization) | chosen vs rejected pairs | 无需单独 reward model 的 alignment |
| **GRPO** (Group Relative Policy Optimization) | reward-guided RL | 基于 reward 的强化学习 |
| **SAPO** (Self-Aligned Preference Optimization) | chosen vs rejected + self-alignment | 改进偏好学习 |

策略可以串联。例如 `CPT → SFT → DPO` 会按顺序执行，每个 phase 都建立在前一个 checkpoint 之上。整条链都可以通过 YAML 完整配置。

---

## GPU 提供方

提供方负责训练与推理所需的 GPU 资源准备。训练和推理分别使用独立的 provider interfaces。

| Provider | 类型 | Training | Inference | 连接方式 |
|----------|------|----------|-----------|----------|
| **single_node** | 本地 | SSH 到你的 GPU 服务器 | 通过 SSH 在 Docker 中运行 vLLM | `~/.ssh/config` alias 或显式 host/port/key |
| **RunPod** | 云端 | 通过 GraphQL API 创建 Pod | Provision volume + pod | API key 放在 `secrets.env` 中 |

### single_node

通过 SSH 直接访问带 GPU 的机器。Pipeline 会部署带有 training runtime 的 Docker container、同步代码、运行训练并回收 artifacts，整个过程都通过 SSH 完成。Inference 则会在同一台机器上部署 vLLM container。

特性：自动 GPU 检测 (`nvidia-smi`)、health checks、workspace cleanup。

### RunPod

通过 RunPod API 使用云 GPU。Pipeline 会按所需 GPU 类型创建 pod，等待 SSH 就绪后启动训练，并可在完成后自动删除 pod。Inference 会额外准备持久化 volume 和单独的 pod。

特性：spot instances、多种 GPU 类型、自动清理 (`cleanup.auto_delete_pod`)。

---

## 插件系统

RyotenkAI 有三套 plugin system，全部遵循同样的模式：`@register` 装饰器、自动发现，以及通过 `secrets.env` 实现命名空间隔离的 secrets 管理。

### 数据集验证

在训练开始前验证数据集。插件会检查 format、quality、diversity 以及领域特定约束。这是 pipeline 的第一阶段，如果验证失败，训练不会启动。

Secrets namespace: `DTST_*` - Docs: [`../src/data/validation/README.md`](../src/data/validation/README.md)

### 评估

训练结束后，针对 live vLLM endpoint 评估模型质量。插件会运行确定性检查 (syntax、semantic match) 和 LLM-as-judge 评分，结果会写入实验报告。

Secrets namespace: `EVAL_*` - Docs: [`../src/evaluation/plugins/README.md`](../src/evaluation/plugins/README.md)

### 报告生成

基于 MLflow 数据生成实验报告。每个插件负责渲染 Markdown 文档中的一个部分 (header、summary、metrics、issues 等)。最终报告会作为 artifact 再次记录到 MLflow。

Docs: [`../src/reports/plugins/README.md`](../src/reports/plugins/README.md)

所有 plugin system 都支持自定义插件。只需实现 base class、添加 `@register`，pipeline 就会自动发现它们。

---

## MLflow 集成

启动 MLflow stack：

```bash
make docker-mlflow-up
```

UI 地址为 `http://localhost:5002`。所有 pipeline runs 都会连同 metrics、artifacts 和 config snapshots 一起被跟踪。

---

## Docker 镜像

| 镜像 | 用途 |
|------|------|
| `ryotenkai/ryotenkai-training-runtime` | 用于训练的 CUDA + PyTorch + 依赖环境 |
| `ryotenkai/inference-vllm` | vLLM inference runtime (serve + merge deps + SSH) |

可以本地构建，也可以推送到 Docker Hub。参见 [`../docker/training/README.md`](../../docker/training/README.md) 和 [`../docker/inference/README.md`](../../docker/inference/README.md)。

---

## CLI 参考

| 命令 | 说明 |
|------|------|
| `ryotenkai run start --config <path>` | 运行完整 training pipeline |
| `ryotenkai run start --local --config <path>` | 在本地运行训练 (不使用远程 GPU) |
| `ryotenkai dataset validate --config <path>` | 仅运行数据集验证 |
| `ryotenkai config validate --config <path>` | 执行静态 pre-flight checks |
| `ryotenkai info --config <path>` | 显示 pipeline 与模型配置 |
| `ryotenkai tui [run_dir]` | 启动交互式 TUI |
| `ryotenkai runs inspect <run_dir>` | 检查某个 run 目录 |
| `ryotenkai runs ls [dir]` | 列出所有 runs 及摘要 |
| `ryotenkai runs logs <run_dir>` | 显示某个 run 的 pipeline log |
| `ryotenkai runs status <run_dir>` | 实时监控正在运行的 pipeline |
| `ryotenkai runs diff <run_dir>` | 对比不同尝试之间的配置差异 |
| `ryotenkai runs report <run_dir>` | 生成 MLflow 实验报告 |
| `ryotenkai version` | 显示版本信息 |

---

## Terminal UI (TUI)

RyotenkAI 内置了一个终端界面，用于监控和查看 training runs：

```bash
ryotenkai tui             # 浏览所有 runs
ryotenkai tui <run_dir>   # 打开某个具体 run
```

**Runs list** - 查看所有 pipeline runs 的状态、耗时和配置名称：

<p align="center">
  <img src="../docs/screenshots/tui_runs_list.png" alt="TUI Runs List" width="800">
</p>

**Run detail** - 深入查看某个 run 的 stages、timing、outputs 和 validation results：

<p align="center">
  <img src="../docs/screenshots/tui_run_detail.png" alt="TUI Run Detail" width="800">
</p>

**Evaluation answers** - 并排查看模型输出与 expected answers：

<p align="center">
  <img src="../docs/screenshots/tui_eval_answers.png" alt="TUI Evaluation Answers" width="800">
</p>

TUI 包含 **Details**、**Logs**、**Inference**、**Eval** 和 **Report** 标签页，让你无需离开终端就能理解整个 training run。

---

## 开发

### Setup

```bash
bash setup.sh
source .venv/bin/activate
```

### 测试

```bash
make test          # 所有测试
make test-unit     # 仅 unit tests
make test-fast     # 跳过 slow tests
make test-cov      # 带 coverage
```

### Lint

```bash
make lint          # 检查
make format        # 自动格式化
make fix-all       # 自动修复
```

### Pre-commit

Pre-commit hooks 会自动运行。如需手动执行：

```bash
make pre-commit
```

---

## 项目结构

```text
ryotenkai/
├── src/
│   ├── config/          # Configuration schemas (Pydantic v2)
│   ├── pipeline/        # Orchestration and stage implementations
│   ├── training/        # 训练策略与 orchestration
│   ├── providers/       # GPU providers (single_node, RunPod)
│   ├── evaluation/      # 模型评估插件
│   ├── data/            # 数据集处理与验证插件
│   ├── reports/         # 报告生成插件
│   ├── tui/             # Terminal UI (Textual)
│   ├── utils/           # 通用工具
│   └── tests/           # 测试套件
├── docker/
│   ├── training/        # Training runtime Docker image
│   ├── inference/       # Inference Docker images
│   └── mlflow/          # MLflow stack (docker-compose)
├── scripts/             # Utility scripts
├── docs/                # 文档与图示
├── setup.sh             # 一条命令完成安装
├── Makefile             # 开发命令
└── pyproject.toml       # 包元数据与工具配置
```

## 社区

如果你想获得支持、讨论 roadmap、分享配置或交流 fine-tuning workflow，欢迎加入 Discord 服务器：

[discord.gg/QqDM2DbY](https://discord.gg/QqDM2DbY)

## 贡献

请参阅 [`../CONTRIBUTING.md`](../../CONTRIBUTING.md)。

## 许可证

[MIT](../../LICENSE) © Golikov Daniil
