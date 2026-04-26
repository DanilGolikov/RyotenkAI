<p align="center">
  <img src="../docs/logo_final.png" alt="RyotenkAI" width="400">
</p>
<h1 align="center">RyotenkAI</h1>

<p align="center">
  선언형 LLM fine-tuning.<br>
  YAML 설정과 데이터셋만 제공하면 RyotenkAI가 검증, GPU 준비, 학습, 추론 배포, 평가, 리포트 생성을 전체적으로 오케스트레이션합니다.
</p>

<p align="center">
  <a href="../README.md">🇬🇧 English</a> |
  <a href="README.ru.md">🇷🇺 Русский</a> |
  <a href="README.ja.md">🇯🇵 日本語</a> |
  <a href="README.zh-CN.md">🇨🇳 简体中文</a> |
  🇰🇷 한국어 |
  <a href="README.es.md">🇪🇸 Español</a> |
  <a href="README.he.md">🇮🇱 עברית</a>
</p>

<p align="center">
  <a href="#빠른-시작">빠른 시작</a> ·
  <a href="#작동-방식">작동 방식</a> ·
  <a href="#학습-전략">전략</a> ·
  <a href="#gpu-프로바이더">프로바이더</a> ·
  <a href="#플러그인-시스템">플러그인</a> ·
  <a href="#설정">설정</a>
</p>

<p align="center">
  <a href="https://discord.gg/QqDM2DbY">
    <img src="https://img.shields.io/badge/Discord-커뮤니티%20참여-5865F2?logo=discord&logoColor=white" alt="Discord 참여">
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

## RyotenkAI란?

RyotenkAI는 LLM fine-tuning을 위한 선언형 control plane입니다. 워크플로를 YAML로 기술하고 데이터셋을 지정하면, RyotenkAI가 데이터셋 검증, GPU 프로비저닝, 다단계 학습, 모델 회수, 추론 배포, 평가, 그리고 MLflow 실험 리포팅까지 전체 라이프사이클을 실행합니다.

| 수동 fine-tuning 워크플로 | RyotenkAI |
|---|---|
| 데이터셋 품질을 수동으로 확인 | 플러그인 기반 검증: format, duplicates, length, diversity |
| GPU 서버에 SSH 접속 후 스크립트 실행 | 한 번의 명령으로 GPU 준비, 학습 배포, 모니터링 수행 |
| 잘 되기를 바라며 기다림 | GPU 메트릭, loss curve, OOM 감지를 실시간으로 모니터링 |
| 가중치를 직접 다운로드 | adapters 자동 회수, LoRA merge, HF Hub 게시 가능 |
| inference server를 따로 띄움 | health checks가 포함된 vLLM endpoint 배포 |
| 출력을 손으로 점검 | syntax, semantic match, LLM-as-judge 기반 플러그인 평가 |
| 문서에 수기로 기록 | MLflow 실험 추적 + 생성된 Markdown 리포트 |

---

## 작동 방식

### 파이프라인 흐름

```text
YAML Config
    │
    ▼
┌─────────────────┐
│ Dataset Validator│  학습 전에 데이터 품질을 검증
│ (plugin system)  │  min_samples, diversity, format, custom
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  GPU Deployer    │  연산 자원 준비 (SSH 또는 RunPod API)
│                  │  training container + 코드 배포
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│Training Monitor  │  프로세스 추적, 로그 파싱, OOM 감지
│                  │  GPU metrics, loss curves, health checks
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Model Retriever  │  adapters / merged weights 다운로드
│                  │  필요 시 HuggingFace Hub에 게시
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│Inference Deployer│  Docker에서 vLLM server 시작
│                  │  Health checks, OpenAI-compatible API
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Model Evaluator  │  live endpoint에서 평가 플러그인 실행
│ (plugin system)  │  syntax, semantic match, LLM judge, custom
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│Report Generator  │  MLflow에서 모든 데이터 수집
│ (plugin system)  │  Markdown 실험 리포트 렌더링
└─────────────────┘
```

### 학습 실행 흐름

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

### 학습 전략 체인 (GPU container 내부)

자동 상태 관리, OOM 복구, checkpoint 관리를 포함한 다단계 학습:

```text
run_training(config.yaml)
  │
  ├── MemoryManager.auto_configure()     GPU tier를 감지하고 VRAM 임계값 설정
  │     └── GPUPreset: margin, critical%, max_retries
  │
  ├── load_model_and_tokenizer()         베이스 모델 로드 (아직 PEFT 없음)
  │     └── MemoryManager: 전후 snapshot, CUDA cache cleanup
  │
  ├── DataBuffer.init_pipeline()         상태 추적 초기화
  │     └── pipeline_state.json          phase 상태와 checkpoint 경로
  │     └── phase_0_sft/                 phase별 출력 디렉터리
  │     └── phase_1_dpo/
  │
  └── ChainRunner.run(strategies)        phase chain 실행
        │
        │   각 phase에 대해 (예: CPT → SFT → DPO):
        │
        ▼
  ┌─────────────────────────────────────────────────────────┐
  │  PhaseExecutor.execute(phase_idx, phase, model, buffer) │
  │                                                         │
  │  1. buffer.mark_phase_started(idx)                      │
  │     └── pipeline_state.json에 상태를 원자적으로 저장   │
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
  │     ├── hyperparams 병합: global ∪ phase overrides      │
  │     ├── PEFT config 생성 (LoRA / QLoRA / AdaLoRA)       │
  │     ├── callbacks 연결 (MLflow, GPU metrics)            │
  │     └── MemoryManager.with_memory_protection으로 보호    │
  │                                                         │
  │  5. trainer.train()                                     │
  │     └── MemoryManager.with_memory_protection            │
  │           ├── VRAM 사용량 모니터링                      │
  │           ├── OOM 시 aggressive_cleanup + retry         │
  │           └── max_retries는 GPU tier preset에서 결정    │
  │                                                         │
  │  6. checkpoint-final 저장                               │
  │     ├── buffer.mark_phase_completed(metrics)            │
  │     └── buffer.cleanup_old_checkpoints(keep_last=2)     │
  │                                                         │
  └─────────────────────┬───────────────────────────────────┘
                        │
                        ▼  모델이 메모리 상에서 다음 phase로 전달됨
                        │
               ┌────────┴────────┐
               │  다음 phase?     │
               │  idx < total    │──── No ──► 학습된 모델 반환
               └────────┬────────┘
                        │ Yes
                        ▼
                 (PhaseExecutor 반복)
```

### DataBuffer - phase 사이의 상태 관리

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
  │     │   ├── checkpoint-500/     (중간 checkpoint, 자동 정리)
  │     │   ├── checkpoint-1000/    (중간 checkpoint, 자동 정리)
  │     │   └── checkpoint-final/   (보존되며 다음 phase 입력으로 사용)
  │     └── phase_1_dpo/
  │         └── checkpoint-final/
  │
  ├── Resume Logic
  │     크래시/재시작 시:
  │       1. load_state() → 완료되지 않은 첫 번째 phase 탐색
  │       2. get_model_path_for_phase(idx) → 이전 checkpoint-final
  │       3. 베이스 모델에 PEFT adapters 로드
  │       4. get_resume_checkpoint(idx) → 중간 checkpoint (있다면)
  │       5. 중단된 지점부터 학습 재개
  │
  └── Cleanup
         cleanup_old_checkpoints(keep_last=2)
         중간 checkpoint-N/ 디렉터리를 제거하고 checkpoint-final은 유지
```

### MemoryManager - GPU OOM 보호

```text
MemoryManager.auto_configure()
  │
  ├── GPU 감지: 이름, VRAM, compute capability
  │     ├── RTX 4060  (8GB)  → consumer_low   tier
  │     ├── RTX 4090  (24GB) → consumer_high  tier
  │     ├── A100      (80GB) → datacenter     tier
  │     └── Unknown          → safe fallback
  │
  ├── tier별 GPUPreset:
  │     margin_mb:    VRAM 여유분 예약 (512-4096 MB)
  │     critical_pct: OOM recovery 발동 임계값 (85-95%)
  │     warning_pct:  warning 로그 임계값 (70-85%)
  │     max_retries:  자동 재시도 횟수 (1-3)
  │
  └── with_memory_protection(operation):
         ┌─────────────────────────────┐
         │  Attempt 1                  │
         │  ├── VRAM 여유 확인         │
         │  ├── operation 실행         │
         │  └── Success → return       │
         │                             │
         │  OOM detected?              │
         │  ├── aggressive_cleanup()   │
         │  │   ├── gc.collect()       │
         │  │   ├── torch.cuda.empty_cache()
         │  │   └── gradients 정리     │
         │  ├── OOM event를 MLflow에 기록
         │  └── 재시도 (max까지)       │
         │                             │
         │  모든 재시도 실패?          │
         │  └── OOMRecoverableError    │
         └─────────────────────────────┘
```

### 평가 흐름

```text
EvaluationRunner
  1. JSONL eval dataset 로드 → (question, expected_answer, metadata) 목록
  2. vLLM endpoint를 통해 모델 응답 수집 → list[EvalSample]
  3. 활성화된 각 plugin을 priority 순으로 실행:
       result = plugin.evaluate(samples)
  4. 결과 집계 → RunSummary (passed/failed, metrics, recommendations)
```

### 리포트 생성 흐름

```text
ryotenkai runs report <run_dir>
  │
  ▼
MLflow ──► runs, metrics, artifacts, configs 가져오기
  │
  ▼
리포트 모델 구성 (phases, issues, timeline)
  │
  ▼
plugins 실행 (각 plugin이 한 섹션을 렌더링)
  │
  ▼
Markdown 렌더링 → experiment_report.md
  │
  └── artifact로 다시 MLflow에 기록
```

---

## 빠른 시작

### 한 번에 설정하기

```bash
git clone https://github.com/DanilGolikov/ryotenkai.git
cd ryotenkai
bash setup.sh
source .venv/bin/activate
```

### 설정

1. `secrets.env`에 API keys (RunPod, HuggingFace)를 입력합니다
2. 예제 설정을 복사한 뒤 필요한 값으로 수정합니다

```bash
cp src/config/pipeline_config.yaml my_config.yaml
# 모델, 데이터셋, provider 설정을 my_config.yaml에서 수정
```

### 실행

```bash
# 설정 검증
ryotenkai config validate --config my_config.yaml

# 전체 pipeline 실행
ryotenkai run start --config my_config.yaml

# 또는 로컬에서 학습 실행 (개발용)
ryotenkai run start --local --config my_config.yaml
```

### 인터랙티브 TUI

```bash
ryotenkai tui
```

TUI는 runs를 탐색하고, stage 상태를 확인하며, live pipelines를 모니터링할 수 있는 대시보드를 제공합니다.

---

## 설정

RyotenkAI는 단일 YAML 설정 파일 (schema v7)을 사용합니다. 주요 섹션은 다음과 같습니다:

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

전체 설정 레퍼런스: [`../src/config/CONFIG_REFERENCE.md`](../src/config/CONFIG_REFERENCE.md)

---

## 학습 전략

RyotenkAI는 strategy chaining 기반의 다단계 학습을 지원합니다. strategies는 **무엇을** 학습할지 정의하고, adapters (LoRA, QLoRA, AdaLoRA, Full FT)는 **어떻게** 학습할지 정의합니다.

| 전략 | 신호 | 사용 사례 |
|------|------|----------|
| **CPT** (Continued Pre-Training) | raw text | 도메인 지식 주입 |
| **SFT** (Supervised Fine-Tuning) | instruction → response pairs | 작업 형식을 모델에 학습 |
| **CoT** (Chain-of-Thought) | reasoning traces | 단계별 추론 개선 |
| **DPO** (Direct Preference Optimization) | chosen vs rejected pairs | 사람의 선호에 맞춘 alignment |
| **ORPO** (Odds Ratio Preference Optimization) | chosen vs rejected pairs | 별도의 reward model 없이 alignment |
| **GRPO** (Group Relative Policy Optimization) | reward-guided RL | reward 기반 강화학습 |
| **SAPO** (Self-Aligned Preference Optimization) | chosen vs rejected + self-alignment | 개선된 preference learning |

전략은 연결해서 사용할 수 있습니다. `CPT → SFT → DPO`는 순차적으로 실행되며, 각 phase는 이전 checkpoint를 기반으로 진행됩니다. 전체 체인은 YAML로 완전히 설정 가능합니다.

---

## GPU 프로바이더

프로바이더는 학습과 추론을 위한 GPU 자원 준비를 담당합니다. training과 inference는 각각 별도의 provider interface를 가집니다.

| Provider | 유형 | Training | Inference | 연결 방식 |
|----------|------|----------|-----------|-----------|
| **single_node** | 로컬 | GPU 서버에 SSH | SSH를 통해 Docker 위에서 vLLM 실행 | `~/.ssh/config` alias 또는 명시적 host/port/key |
| **RunPod** | 클라우드 | GraphQL API를 통한 Pod | Volume + Pod 프로비저닝 | API key를 `secrets.env`에 설정 |

### single_node

GPU가 있는 머신에 직접 SSH로 접속합니다. 파이프라인은 training runtime이 포함된 Docker container를 배포하고, 코드를 동기화하며, 학습을 실행하고, artifacts를 회수합니다. 모든 과정이 SSH를 통해 이루어집니다. Inference는 같은 호스트에 vLLM container를 배포합니다.

특징: 자동 GPU 감지 (`nvidia-smi`), health checks, workspace cleanup.

### RunPod

RunPod API를 사용하는 클라우드 GPU 방식입니다. 요청한 GPU 타입으로 pod를 생성하고, SSH 준비가 끝나면 학습을 시작합니다. 완료 후 필요에 따라 pod를 자동 삭제할 수 있습니다. Inference는 persistent volume과 별도 pod를 프로비저닝합니다.

특징: spot instances, 여러 GPU 타입, 자동 정리 (`cleanup.auto_delete_pod`).

---

## 플러그인 시스템

RyotenkAI에는 세 가지 plugin system이 있으며, 모두 같은 패턴을 따릅니다: `@register` 데코레이터, 자동 탐색, 그리고 `secrets.env`를 통한 namespace 분리형 secret 관리입니다.

### 데이터셋 검증

학습 시작 전에 데이터셋을 검증합니다. 플러그인은 format, quality, diversity, 도메인 특화 제약을 검사합니다. 이는 파이프라인의 첫 stage이며, 검증에 실패하면 학습이 시작되지 않습니다.

Secrets namespace: `DTST_*` - Docs: [`../src/data/validation/README.md`](../src/data/validation/README.md)

### 평가

학습 후 live vLLM endpoint를 대상으로 모델 품질을 평가합니다. 플러그인은 deterministic checks (syntax, semantic match)와 LLM-as-judge scoring을 실행하며, 결과는 실험 리포트에 반영됩니다.

Secrets namespace: `EVAL_*` - Docs: [`../src/evaluation/plugins/README.md`](../src/evaluation/plugins/README.md)

### 리포트 생성

MLflow 데이터를 기반으로 실험 리포트를 생성합니다. 각 플러그인은 Markdown 문서의 한 섹션 (header, summary, metrics, issues 등)을 렌더링합니다. 최종 리포트는 artifact로 다시 MLflow에 기록됩니다.

Docs: [`../src/reports/plugins/README.md`](../src/reports/plugins/README.md)

모든 plugin system은 커스텀 플러그인도 지원합니다. base class를 구현하고 `@register`를 붙이면 파이프라인이 자동으로 탐지합니다.

---

## MLflow 연동

MLflow stack을 시작합니다:

```bash
make docker-mlflow-up
```

UI는 `http://localhost:5002`에서 접근할 수 있습니다. 모든 pipeline run은 metrics, artifacts, config snapshots와 함께 추적됩니다.

---

## Docker 이미지

| 이미지 | 용도 |
|--------|------|
| `ryotenkai/ryotenkai-training-runtime` | 학습용 CUDA + PyTorch + 의존성 환경 |
| `ryotenkai/inference-vllm` | vLLM inference runtime (serve + merge deps + SSH) |

로컬에서 빌드하거나 Docker Hub로 push할 수 있습니다. 자세한 내용은 [`../docker/training/README.md`](../docker/training/README.md) 와 [`../docker/inference/README.md`](../docker/inference/README.md) 를 참고하세요.

---

## CLI 레퍼런스

| 명령어 | 설명 |
|--------|------|
| `ryotenkai run start --config <path>` | 전체 training pipeline 실행 |
| `ryotenkai run start --local --config <path>` | 로컬에서 학습 실행 (원격 GPU 없음) |
| `ryotenkai dataset validate --config <path>` | 데이터셋 검증만 실행 |
| `ryotenkai config validate --config <path>` | 정적 pre-flight checks 실행 |
| `ryotenkai info --config <path>` | 파이프라인 및 모델 설정 표시 |
| `ryotenkai tui [run_dir]` | 인터랙티브 TUI 실행 |
| `ryotenkai runs inspect <run_dir>` | run 디렉터리 확인 |
| `ryotenkai runs ls [dir]` | 모든 runs를 요약과 함께 표시 |
| `ryotenkai runs logs <run_dir>` | 특정 run의 pipeline log 표시 |
| `ryotenkai runs status <run_dir>` | 실행 중인 pipeline 라이브 모니터링 |
| `ryotenkai runs diff <run_dir>` | 시도 간 config 차이 비교 |
| `ryotenkai runs report <run_dir>` | MLflow 실험 리포트 생성 |
| `ryotenkai version` | 버전 정보 표시 |

---

## Terminal UI (TUI)

RyotenkAI에는 training runs를 모니터링하고 살펴볼 수 있는 내장 터미널 인터페이스가 포함되어 있습니다:

```bash
ryotenkai tui             # 모든 runs 탐색
ryotenkai tui <run_dir>   # 특정 run 열기
```

**Runs list** - 모든 pipeline runs를 status, duration, config name과 함께 한눈에 확인:

<p align="center">
  <img src="../docs/screenshots/tui_runs_list.png" alt="TUI Runs List" width="800">
</p>

**Run detail** - 특정 run 안으로 들어가 stages, timing, outputs, validation results를 확인:

<p align="center">
  <img src="../docs/screenshots/tui_run_detail.png" alt="TUI Run Detail" width="800">
</p>

**Evaluation answers** - 모델 출력과 expected answers를 나란히 비교:

<p align="center">
  <img src="../docs/screenshots/tui_eval_answers.png" alt="TUI Evaluation Answers" width="800">
</p>

TUI에는 **Details**, **Logs**, **Inference**, **Eval**, **Report** 탭이 있어 터미널을 벗어나지 않고도 training run 전체를 이해할 수 있습니다.

---

## 개발

### Setup

```bash
bash setup.sh
source .venv/bin/activate
```

### 테스트

```bash
make test          # 전체 테스트
make test-unit     # unit tests만
make test-fast     # slow tests 건너뜀
make test-cov      # coverage 포함
```

### Lint

```bash
make lint          # 검사
make format        # 자동 포맷
make fix-all       # 자동 수정
```

### Pre-commit

Pre-commit hooks는 자동으로 실행됩니다. 수동 실행:

```bash
make pre-commit
```

---

## 프로젝트 구조

```text
ryotenkai/
├── src/
│   ├── config/          # Configuration schemas (Pydantic v2)
│   ├── pipeline/        # Orchestration and stage implementations
│   ├── training/        # 학습 전략과 orchestration
│   ├── providers/       # GPU providers (single_node, RunPod)
│   ├── evaluation/      # 모델 평가 플러그인
│   ├── data/            # 데이터셋 처리 및 검증 플러그인
│   ├── reports/         # 리포트 생성 플러그인
│   ├── tui/             # Terminal UI (Textual)
│   ├── utils/           # 공용 유틸리티
│   └── tests/           # 테스트 스위트
├── docker/
│   ├── training/        # Training runtime Docker image
│   ├── inference/       # Inference Docker images
│   └── mlflow/          # MLflow stack (docker-compose)
├── scripts/             # Utility scripts
├── docs/                # 문서와 다이어그램
├── setup.sh             # 한 번에 setup
├── Makefile             # 개발용 명령
└── pyproject.toml       # 패키지 메타데이터와 tool 설정
```

## 커뮤니티

지원, roadmap 논의, config 공유, fine-tuning workflow 대화를 원하면 Discord 서버에 참여하세요:

[discord.gg/QqDM2DbY](https://discord.gg/QqDM2DbY)

## 기여하기

[`../CONTRIBUTING.md`](../CONTRIBUTING.md) 를 참고하세요.

## 라이선스

[MIT](../LICENSE) © Golikov Daniil
